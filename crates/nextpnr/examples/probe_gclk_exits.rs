//! From a single GCLK peer (one tile), enumerate pips_downhill and show
//! the destination wire name + type. Lets us see exactly what kind of
//! wires the GCLK spine feeds into besides pure clock paths.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use rustc_hash::FxHashMap;
use std::path::Path;

fn describe(db: &ChipDb, w: WireId) -> String {
    let info = db.wire_info(w);
    let name_id: i32 = unsafe { nextpnr::read_packed!(*info, name) };
    let wname = db.constid_str(name_id).unwrap_or("<anon>").to_string();
    format!(
        "{}({}:{}) type={}",
        wname,
        w.tile(),
        w.index(),
        db.wire_type(w)
    )
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    // M2_GCLK_B1 at [197,115] from earlier log: tile 35962, widx 1025
    let src_tile: i32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(35962);
    let src_idx: i32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1025);

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let w = WireId::new(src_tile, src_idx);
    println!("Wire: {}", describe(&db, w));

    let info = db.wire_info(w);
    let pips = info.pips_downhill.get();
    println!("{} pips_downhill:", pips.len());
    let mut by_type: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for &pip_idx in pips {
        let pip = PipId::new(w.tile(), pip_idx);
        let dst = db.pip_dst_wire(pip);
        let dst_info = db.wire_info(dst);
        let name_id: i32 = unsafe { nextpnr::read_packed!(*dst_info, name) };
        let dname = db.constid_str(name_id).unwrap_or("<anon>").to_string();
        let dtype = db.wire_type(dst).to_string();
        println!("  -> {} (dst type={})", dname, dtype);
        by_type.entry(dtype).or_default().push(dname);
    }

    // Aggregate across ALL peers of this wire's node.
    println!("\n=== Aggregate across node peers ===");
    let mut agg: FxHashMap<(String, String), usize> = FxHashMap::default();
    let mut total = 0usize;
    db.node_wires_cb(w, |peer| {
        let info = db.wire_info(peer);
        let src_name_id: i32 = unsafe { nextpnr::read_packed!(*info, name) };
        let src_wname = db.constid_str(src_name_id).unwrap_or("<anon>").to_string();
        for &pip_idx in info.pips_downhill.get() {
            let pip = PipId::new(peer.tile(), pip_idx);
            let dst = db.pip_dst_wire(pip);
            let dinfo = db.wire_info(dst);
            let dnid: i32 = unsafe { nextpnr::read_packed!(*dinfo, name) };
            let dname = db.constid_str(dnid).unwrap_or("<anon>").to_string();
            let dtype = db.wire_type(dst).to_string();
            *agg.entry((format!("{}/{}", src_wname, dtype), dname))
                .or_insert(0) += 1;
            total += 1;
        }
    });
    println!("Total {} peer-pip exits", total);
    let mut items: Vec<_> = agg.iter().collect();
    items.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    for ((src_wname, dname), count) in items.iter().take(30) {
        println!("  [{:>6}×] {} -> {}", count, src_wname, dname);
    }
}

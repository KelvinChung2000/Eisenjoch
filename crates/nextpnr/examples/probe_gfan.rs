//! Inspect a GFAN wire: who drives it (pips_uphill) and what it drives
//! (pips_downhill). Lets us tell whether it's a dedicated clock-to-fabric
//! tap or a shared general-routing wire.

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
    let target_tile: i32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(35962);
    let wire_name: String = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "M1_GFAN0".to_string());

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");

    // Scan tile-type wire list for matching name.
    let tt = db.tile_type(target_tile);
    let wires = tt.wires.get();
    let mut target: Option<WireId> = None;
    for (idx, w) in wires.iter().enumerate() {
        let nid: i32 = unsafe { nextpnr::read_packed!(*w, name) };
        let nm = db.constid_str(nid).unwrap_or("<anon>");
        if nm == wire_name {
            target = Some(WireId::new(target_tile, idx as i32));
            break;
        }
    }
    let w = target.expect("wire not found");
    println!("Wire: {}", describe(&db, w));

    let info = db.wire_info(w);
    let ups = info.pips_uphill.get();
    let downs = info.pips_downhill.get();
    println!("  pips_uphill   = {}", ups.len());
    println!("  pips_downhill = {}", downs.len());

    println!("\n--- uphill (who drives this) ---");
    let mut up_types: FxHashMap<String, usize> = FxHashMap::default();
    for &pidx in ups {
        let pip = PipId::new(w.tile(), pidx);
        let src = db.pip_src_wire(pip);
        let info = db.wire_info(src);
        let nid: i32 = unsafe { nextpnr::read_packed!(*info, name) };
        let nm = db.constid_str(nid).unwrap_or("<anon>").to_string();
        let t = db.wire_type(src).to_string();
        *up_types.entry(t.clone()).or_insert(0) += 1;
        println!("  <- {} ({})", nm, t);
    }
    println!("  by type: {:?}", up_types);

    println!("\n--- downhill (what this drives) ---");
    let mut down_types: FxHashMap<String, usize> = FxHashMap::default();
    for &pidx in downs {
        let pip = PipId::new(w.tile(), pidx);
        let dst = db.pip_dst_wire(pip);
        let info = db.wire_info(dst);
        let nid: i32 = unsafe { nextpnr::read_packed!(*info, name) };
        let nm = db.constid_str(nid).unwrap_or("<anon>").to_string();
        let t = db.wire_type(dst).to_string();
        *down_types.entry(t.clone()).or_insert(0) += 1;
        println!("  -> {} ({})", nm, t);
    }
    println!("  by type: {:?}", down_types);
}

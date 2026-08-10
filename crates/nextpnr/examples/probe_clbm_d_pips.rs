//! Inspect pips downhill of M3_CLBLM_M_D@(260,113) and pips uphill of
//! M3_CLBLM_M_CX@(262,112). Report the wire names and their tile type.
use nextpnr::chipdb::{ChipDb, PipId, WireId};
use std::path::Path;

fn find_wire_by_name(db: &ChipDb, tile: i32, target: &str) -> Option<WireId> {
    let tt = db.tile_type(tile);
    for (wi, wire) in tt.wires.get().iter().enumerate() {
        let nid: i32 = unsafe { nextpnr::read_packed!(*wire, name) };
        if let Some(s) = db.constid_str(nid) {
            if s == target {
                return Some(WireId::new(tile, wi as i32));
            }
        }
    }
    None
}

fn wname(db: &ChipDb, w: WireId) -> String {
    let (x, y) = db.tile_xy(w.tile());
    format!("{}@({},{})", db.wire_name(w), x, y)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&path)).expect("load");

    let src_tile = db.tile_by_xy(260, 113);
    let src = find_wire_by_name(&db, src_tile, "M3_CLBLM_M_D").expect("src");
    let src_info = db.wire_info(src);
    println!(
        "SRC: M3_CLBLM_M_D@(260,113) tile_type={}",
        db.tile_type_name(src_tile)
    );
    println!(
        "  pips_downhill ({} total):",
        src_info.pips_downhill.get().len()
    );
    for &pip_idx in src_info.pips_downhill.get() {
        let pip = PipId::new(src.tile(), pip_idx);
        let dst = db.pip_dst_wire(pip);
        let dst_info = db.wire_info(dst);
        println!(
            "    -> {} (down_next={}, up_back={})",
            wname(&db, dst),
            dst_info.pips_downhill.get().len(),
            dst_info.pips_uphill.get().len(),
        );
    }
    println!("  node peers:");
    db.node_wires_cb(src, |nw| println!("    peer: {}", wname(&db, nw)));

    println!();
    let dst_tile = db.tile_by_xy(262, 112);
    let dst = find_wire_by_name(&db, dst_tile, "M3_CLBLM_M_CX").expect("dst");
    let dst_info = db.wire_info(dst);
    println!(
        "DST: M3_CLBLM_M_CX@(262,112) tile_type={}",
        db.tile_type_name(dst_tile)
    );
    println!(
        "  pips_uphill ({} total):",
        dst_info.pips_uphill.get().len()
    );
    for &pip_idx in dst_info.pips_uphill.get() {
        let pip = PipId::new(dst.tile(), pip_idx);
        let src_w = db.pip_src_wire(pip);
        let src_info = db.wire_info(src_w);
        println!(
            "    {} <- (up_back={}, down_next={})",
            wname(&db, src_w),
            src_info.pips_uphill.get().len(),
            src_info.pips_downhill.get().len(),
        );
    }
    println!("  node peers:");
    db.node_wires_cb(dst, |nw| println!("    peer: {}", wname(&db, nw)));
}

use nextpnr::chipdb::{ChipDb, WireId};
use std::path::Path;

fn find_wire_by_name(db: &ChipDb, tile: i32, target: &str) -> Option<WireId> {
    let tt = db.tile_type(tile);
    for (wi, wire) in tt.wires.get().iter().enumerate() {
        let nid: i32 = unsafe { nextpnr::read_packed!(*wire, name) };
        if let Some(s) = db.constid_str(nid) {
            if s == target { return Some(WireId::new(tile, wi as i32)); }
        }
    }
    None
}

fn main() {
    let chipdb_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load");

    for y in 98..=111 {
        let tile = db.tile_by_xy(309, y);
        let tname = db.tile_type_name(tile).to_string();
        let s_wire = find_wire_by_name(&db, tile, "CLK_RELAY_IN_0_S");
        let n_wire = find_wire_by_name(&db, tile, "CLK_RELAY_IN_0_N");
        let s_nid = s_wire.and_then(|w| db.node_id(w));
        let n_nid = n_wire.and_then(|w| db.node_id(w));
        println!("x=309 y={:3} tile={} type={:<10} S_node={:?} N_node={:?}", y, tile, tname, s_nid, n_nid);
    }
}

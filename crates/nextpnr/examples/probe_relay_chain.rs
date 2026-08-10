//! Sweep across a row at fixed y and report whether each tile exposes the
//! CLK_RELAY_IN_0 mesh (E/W/N/S wires + node_id). Identifies columns that
//! break the horizontal chain (BRAM/DSP/NULL tiles without the mesh).

use nextpnr::chipdb::{ChipDb, WireId};
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

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let y: i32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(98);

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let width = db.width();

    println!("Sweep y={} across x=0..{}", y, width);
    let directions = ["E", "W", "N", "S"];
    for x in 0..width {
        let tile = db.tile_by_xy(x, y);
        let tt_name = db.tile_type_name(tile).to_string();

        let mut present: Vec<&str> = Vec::new();
        let mut with_nid: Vec<&str> = Vec::new();
        let side = std::env::var("PROBE_SIDE").unwrap_or_else(|_| "OUT".to_string());
        for d in directions {
            let wname = format!("CLK_RELAY_{}_0_{}", side, d);
            if let Some(w) = find_wire_by_name(&db, tile, &wname) {
                present.push(d);
                if db.node_id(w).is_some() {
                    with_nid.push(d);
                }
            }
        }

        if present.is_empty() {
            println!("x={:3} tile={} type={:<12} NO_MESH", x, tile, tt_name);
        } else {
            let e_nid = find_wire_by_name(&db, tile, &format!("CLK_RELAY_{}_0_E", side))
                .and_then(|w| db.node_id(w));
            let w_nid = find_wire_by_name(&db, tile, &format!("CLK_RELAY_{}_0_W", side))
                .and_then(|w| db.node_id(w));
            println!(
                "x={:3} tile={} type={:<12} E_node={:?} W_node={:?}",
                x, tile, tt_name, e_nid, w_nid
            );
        }
    }
}

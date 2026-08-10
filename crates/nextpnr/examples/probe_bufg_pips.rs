//! Inspect PIPs around BUFG0_I@(310,111) and CLK_RELAY_IN_0_W@(310,111) to
//! see why A* reaches (309,111) but not (310,111).

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

fn wname(db: &ChipDb, w: WireId) -> String {
    let (x, y) = db.tile_xy(w.tile());
    format!("{}@({},{})", db.wire_name(w), x, y)
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load");

    let clk_bufg_tile = db.tile_by_xy(310, 111);
    println!(
        "CLK_BUFG tile at (310,111) = {} type={}",
        clk_bufg_tile,
        db.tile_type_name(clk_bufg_tile)
    );

    let iob_tile_309 = db.tile_by_xy(309, 111);
    println!(
        "IOB tile at (309,111) = {} type={}",
        iob_tile_309,
        db.tile_type_name(iob_tile_309)
    );

    // Find BUFG0_I.
    let bufg_i = find_wire_by_name(&db, clk_bufg_tile, "BUFG0_I").expect("BUFG0_I");
    let bi_info = db.wire_info(bufg_i);
    println!(
        "\nBUFG0_I @ (310,111) = {}:{}",
        bufg_i.tile(),
        bufg_i.index()
    );
    println!(
        "  pips_uphill: {} pips, pips_downhill: {} pips",
        bi_info.pips_uphill.get().len(),
        bi_info.pips_downhill.get().len()
    );
    println!("  uphill pips (wires that drive BUFG0_I):");
    for &pip_idx in bi_info.pips_uphill.get() {
        let pip = nextpnr::chipdb::PipId::new(bufg_i.tile(), pip_idx);
        let src = db.pip_src_wire(pip);
        println!("    {} -> BUFG0_I", wname(&db, src));
    }

    // Find CLK_RELAY_IN_0_W @ (310,111) — should exist and be in same node as
    // CLK_RELAY_IN_0_E @ (309,111).
    let relay_w_310 = find_wire_by_name(&db, clk_bufg_tile, "CLK_RELAY_IN_0_W");
    match relay_w_310 {
        Some(w) => {
            let info = db.wire_info(w);
            println!(
                "\nCLK_RELAY_IN_0_W @ (310,111) = {}:{}",
                w.tile(),
                w.index()
            );
            println!("  node_id: {:?}", db.node_id(w));
            println!(
                "  pips_uphill: {} pips, pips_downhill: {} pips",
                info.pips_uphill.get().len(),
                info.pips_downhill.get().len()
            );
            println!("  downhill pips (wires CLK_RELAY_IN_0_W drives):");
            for &pip_idx in info.pips_downhill.get() {
                let pip = nextpnr::chipdb::PipId::new(w.tile(), pip_idx);
                let dst = db.pip_dst_wire(pip);
                println!("    CLK_RELAY_IN_0_W -> {}", wname(&db, dst));
            }
        }
        None => {
            println!("\nCLK_RELAY_IN_0_W @ (310,111): NOT FOUND in tile type");
        }
    }

    // And CLK_RELAY_IN_0_E @ (309,111) — what's there?
    let relay_e_309 = find_wire_by_name(&db, iob_tile_309, "CLK_RELAY_IN_0_E");
    match relay_e_309 {
        Some(w) => {
            let info = db.wire_info(w);
            println!(
                "\nCLK_RELAY_IN_0_E @ (309,111) = {}:{}",
                w.tile(),
                w.index()
            );
            println!("  node_id: {:?}", db.node_id(w));
            println!(
                "  pips_downhill: {}, uphill: {}",
                info.pips_downhill.get().len(),
                info.pips_uphill.get().len()
            );
            println!("  downhill pips from CLK_RELAY_IN_0_E @ (309,111):");
            for &pip_idx in info.pips_downhill.get() {
                let pip = nextpnr::chipdb::PipId::new(w.tile(), pip_idx);
                let dst = db.pip_dst_wire(pip);
                println!("    -> {}", wname(&db, dst));
            }
            println!("  node peers (wires electrically equivalent to this wire):");
            db.node_wires_cb(w, |peer| {
                println!("    peer: {}", wname(&db, peer));
            });
        }
        None => {
            println!("\nCLK_RELAY_IN_0_E @ (309,111): NOT FOUND");
        }
    }
}

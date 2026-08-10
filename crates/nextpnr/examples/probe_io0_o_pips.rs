//! Dump all downhill PIPs from IO0_O @ (0,98).
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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&path)).expect("load");

    let tile = db.tile_by_xy(0, 98);
    println!("tile (0,98) = {} type={}", tile, db.tile_type_name(tile));
    let io_o = find_wire_by_name(&db, tile, "IO0_O").expect("IO0_O");
    println!(
        "IO0_O @ (0,98) = {}:{}, pips_downhill = {}",
        io_o.tile(),
        io_o.index(),
        db.wire_info(io_o).pips_downhill.get().len()
    );
    println!("IO0_O node_id: {:?}", db.node_id(io_o));
    println!("IO0_O peers:");
    db.node_wires_cb(io_o, |peer| {
        let (x, y) = db.tile_xy(peer.tile());
        println!("  peer: {}@({},{})", db.wire_name(peer), x, y);
    });

    println!("\nIO0_O downhill PIPs (dst wire, dst tile):");
    for &pip_idx in db.wire_info(io_o).pips_downhill.get() {
        let pip = PipId::new(io_o.tile(), pip_idx);
        let dst = db.pip_dst_wire(pip);
        let (x, y) = db.tile_xy(dst.tile());
        let dst_name = db.wire_name(dst);
        let dst_nid = db.node_id(dst);
        let n_peers = {
            let mut n = 0usize;
            db.node_wires_cb(dst, |_| n += 1);
            n
        };
        let n_down = db.wire_info(dst).pips_downhill.get().len();
        println!(
            "  -> {}@({},{}) (tile type={}, downhill={}, node_id={:?}, peers={})",
            dst_name,
            x,
            y,
            db.tile_type_name(dst.tile()),
            n_down,
            dst_nid,
            n_peers
        );
    }
}

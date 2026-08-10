//! List all peers of CLK_RELAY_IN_0_E @ (0,98) and (1,98), and verify
//! internal PIPs at (1,98).
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

fn dump(db: &ChipDb, name: &str, x: i32, y: i32) {
    let tile = db.tile_by_xy(x, y);
    let w = match find_wire_by_name(db, tile, name) {
        Some(w) => w,
        None => {
            println!("{}@({},{}): NOT FOUND", name, x, y);
            return;
        }
    };
    println!("\n== {}@({},{}) = {}:{} tile_type={} ==", name, x, y, w.tile(), w.index(), db.tile_type_name(tile));
    let info = db.wire_info(w);
    println!("  node_id: {:?}", db.node_id(w));
    println!("  pips_downhill: {}", info.pips_downhill.get().len());
    for &pip_idx in info.pips_downhill.get() {
        let pip = PipId::new(w.tile(), pip_idx);
        let dst = db.pip_dst_wire(pip);
        println!("    down -> {}", wname(db, dst));
    }
    println!("  pips_uphill: {}", info.pips_uphill.get().len());
    for &pip_idx in info.pips_uphill.get() {
        let pip = PipId::new(w.tile(), pip_idx);
        let src = db.pip_src_wire(pip);
        println!("    {} -> up", wname(db, src));
    }
    println!("  node_wires_cb peers:");
    let mut count = 0;
    db.node_wires_cb(w, |peer| {
        count += 1;
        println!("    peer[{}]: {}", count, wname(db, peer));
    });
    println!("  total peers: {}", count);
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&path)).expect("load");

    dump(&db, "IO_BRIDGE_OUT0", 0, 98);
    dump(&db, "IO_BRIDGE_OUT0", 1, 98);
}

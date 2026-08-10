//! Probe M3_CLBLM_LOGIC_OUTS15 downhill reachability.
use nextpnr::chipdb::{ChipDb, PipId, WireId};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;
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

    let tile = db.tile_by_xy(260, 113);
    let w = find_wire_by_name(&db, tile, "M3_CLBLM_LOGIC_OUTS15").expect("wire");
    println!("M3_CLBLM_LOGIC_OUTS15@(260,113):");

    let mut ns: Vec<WireId> = vec![w];
    db.node_wires_cb(w, |nw| ns.push(nw));
    println!("  node size: {}", ns.len());
    for n in &ns {
        println!("    {}", wname(&db, *n));
    }

    // BFS 3 hops downhill.
    let mut visited: FxHashSet<WireId> = FxHashSet::default();
    let mut q: VecDeque<(WireId, usize)> = VecDeque::new();
    for &n in &ns {
        visited.insert(n);
        q.push_back((n, 0));
    }
    while let Some((cur, depth)) = q.pop_front() {
        if depth >= 4 {
            continue;
        }
        let info = db.wire_info(cur);
        for &pip_idx in info.pips_downhill.get() {
            let pip = PipId::new(cur.tile(), pip_idx);
            let nx = db.pip_dst_wire(pip);
            if !visited.insert(nx) {
                continue;
            }
            let peers_empty = {
                let mut any = false;
                db.node_wires_cb(nx, |_| any = true);
                !any
            };
            println!(
                "    depth={} {} -> {} (peers_empty={})",
                depth,
                wname(&db, cur),
                wname(&db, nx),
                peers_empty
            );
            q.push_back((nx, depth + 1));
            if visited.len() > 200 {
                return;
            }
        }
    }
    println!("reachable within 4 hops: {}", visited.len());
}

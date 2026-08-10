//! Forward BFS from a source wire (given by tile:index) through pips_downhill,
//! counting wire-type distribution at each cost level. If the clock network
//! is topologically independent, we should only see GCLK/CLK types descending
//! from BUFG_O. Any large ROUTING/SPAN counts = chipdb leak from clock → fabric.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use rustc_hash::{FxHashMap, FxHashSet};
use std::path::Path;

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    // BUFG1_O from earlier probe = tile 34831, wire 3.
    let src_tile: i32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(34831);
    let src_idx: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_depth: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(6);

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let src = WireId::new(src_tile, src_idx);
    println!("Source wire: {}:{} type={}", src.tile(), src.index(), db.wire_type(src));

    let mut visited: FxHashSet<WireId> = FxHashSet::default();
    let mut frontier: Vec<WireId> = vec![src];
    visited.insert(src);
    db.node_wires_cb(src, |p| { if visited.insert(p) { frontier.push(p); } });
    println!("Cost 0: {} wires (src + node peers)", frontier.len());

    for depth in 1..=max_depth {
        let mut next: Vec<WireId> = Vec::new();
        let mut type_count: FxHashMap<String, usize> = FxHashMap::default();
        for &w in &frontier {
            let info = db.wire_info(w);
            for &pip_idx in info.pips_downhill.get() {
                let pip = PipId::new(w.tile(), pip_idx);
                let dst = db.pip_dst_wire(pip);
                if !visited.insert(dst) { continue; }
                *type_count.entry(db.wire_type(dst).to_string()).or_insert(0) += 1;
                next.push(dst);
                db.node_wires_cb(dst, |p| {
                    if visited.insert(p) {
                        *type_count.entry(db.wire_type(p).to_string()).or_insert(0) += 1;
                        next.push(p);
                    }
                });
            }
        }
        println!("Cost {}: {} new wires; visited total {}", depth, next.len(), visited.len());
        let mut items: Vec<_> = type_count.iter().collect();
        items.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
        for (t, c) in items.iter().take(8) {
            println!("    {:<18} {}", t, c);
        }
        if next.is_empty() { break; }
        frontier = next;
    }
    println!("\nTotal reachable: {}", visited.len());
}

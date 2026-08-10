//! Replay A* on IO0_O -> BUFG0_I and print the first 100 pops with heap
//! state, to find where the chain-walk stalls.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

const H_WEIGHT: DelayT = 1000;

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

fn heuristic(db: &ChipDb, wire: WireId, dst: WireId) -> DelayT {
    let (wx, wy) = db.tile_xy(wire.tile());
    let (dx, dy) = db.tile_xy(dst.tile());
    let m = (wx - dx).abs() + (wy - dy).abs();
    (m as DelayT).saturating_mul(H_WEIGHT)
}

#[derive(Clone, Copy)]
struct QE {
    wire: WireId,
    cost: DelayT,
    estimate: DelayT,
}
impl PartialEq for QE { fn eq(&self, o: &Self) -> bool { self.estimate == o.estimate } }
impl Eq for QE {}
impl PartialOrd for QE { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for QE {
    fn cmp(&self, o: &Self) -> Ordering {
        o.estimate.cmp(&self.estimate).then_with(|| o.cost.cmp(&self.cost))
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&path)).expect("load");
    let ctx = Context::new(db);
    let db = ctx.chipdb();

    let src = find_wire_by_name(db, db.tile_by_xy(0, 98), "IO0_O").unwrap();
    let dst = find_wire_by_name(db, db.tile_by_xy(310, 111), "BUFG0_I").unwrap();

    let mut heap: BinaryHeap<QE> = BinaryHeap::new();
    let mut visited: FxHashMap<WireId, DelayT> = FxHashMap::default();
    let mut nodes_expanded: FxHashSet<u64> = FxHashSet::default();

    heap.push(QE {
        wire: src,
        cost: 0,
        estimate: heuristic(db, src, dst),
    });
    visited.insert(src, 0);

    let mut step = 0usize;
    let max_steps = 2000;
    while let Some(e) = heap.pop() {
        step += 1;
        if step > max_steps {
            println!("Hit max_steps={}, stopping", max_steps);
            break;
        }
        let prev_score = visited.get(&e.wire).copied().unwrap_or(DelayT::MAX);
        if e.cost > prev_score {
            if step <= 100 {
                println!("step {}: STALE pop {} cost={} prev_score={}",
                    step, wname(db, e.wire), e.cost, prev_score);
            }
            continue;
        }

        if e.wire == dst {
            println!("step {}: REACHED dst {} at cost={}", step, wname(db, e.wire), e.cost);
            return;
        }

        let nid = db.node_id(e.wire);
        let new_node = match nid {
            Some(id) => nodes_expanded.insert(id),
            None => true,
        };

        if step <= 50 || (step <= 200 && step % 10 == 0) {
            println!("step {}: pop {} cost={} f={} nid={:?} new_node={} heap_len={}",
                step, wname(db, e.wire), e.cost, e.estimate, nid, new_node, heap.len());
        }

        if !new_node {
            continue;
        }

        let expand = |member: WireId, visited: &mut FxHashMap<WireId, DelayT>, heap: &mut BinaryHeap<QE>| -> usize {
            let mut pushed = 0;
            if member != e.wire {
                match visited.entry(member) {
                    std::collections::hash_map::Entry::Occupied(mut en) => {
                        if *en.get() > e.cost {
                            en.insert(e.cost);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(en) => {
                        en.insert(e.cost);
                    }
                }
            }
            let info = db.wire_info(member);
            for &pip_idx in info.pips_downhill.get() {
                let pip = PipId::new(member.tile(), pip_idx);
                let next = db.pip_dst_wire(pip);
                let new_cost = e.cost + 1;
                let score = new_cost;
                let prev = visited.get(&next).copied().unwrap_or(DelayT::MAX);
                if prev <= score { continue; }
                visited.insert(next, score);
                let h = heuristic(db, next, dst);
                heap.push(QE { wire: next, cost: new_cost, estimate: score + h });
                pushed += 1;
            }
            pushed
        };

        let mut p = expand(e.wire, &mut visited, &mut heap);

        if let Some(_nid_val) = nid {
            db.node_wires_cb(e.wire, |peer| {
                p += expand(peer, &mut visited, &mut heap);
            });
        }

        if step <= 50 || (step <= 200 && step % 10 == 0) {
            println!("  -> expanded, pushed={}", p);
        }
    }
    println!("Heap drained at step {}", step);
}

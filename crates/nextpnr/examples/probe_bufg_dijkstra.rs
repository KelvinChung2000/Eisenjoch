//! Bare Dijkstra from BUFG0_O@(310,111) without any placement — just chipdb
//! graph reachability. Tests what's reachable from the BUFG source, and at
//! what pop-count cost.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

struct DijkstraModel;

impl PathCostModel for DijkstraModel {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, _ctx: &Context, _wire: WireId, _dst: WireId) -> DelayT {
        0
    }
}

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
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    let db = ctx.chipdb();

    // Source: BUFG0_O at (310, 111)
    let src_tile = db.tile_by_xy(310, 111);
    let src = find_wire_by_name(db, src_tile, "BUFG0_O").expect("BUFG0_O");
    println!("src: BUFG0_O @ tile({},{}) = {}:{}", 310, 111, src.tile(), src.index());

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    db.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });
    println!("src node size: {}", src_set.len());

    // Try to reach some nearby wires at different distances.
    let targets = [
        (300, 111, "CLK_RELAY_OUT_0_E"),
        (280, 111, "CLK_RELAY_OUT_0_E"),
        (260, 111, "CLK_RELAY_OUT_0_E"),
        (200, 111, "CLK_RELAY_OUT_0_E"),
        (100, 111, "CLK_RELAY_OUT_0_E"),
    ];

    for (tx, ty, tname) in targets {
        let ttile = db.tile_by_xy(tx, ty);
        let dst = match find_wire_by_name(db, ttile, tname) {
            Some(w) => w,
            None => {
                println!("TGT ({},{}) {}: NOT FOUND", tx, ty, tname);
                continue;
            }
        };
        for &limit in &[1_000_000usize, 10_000_000, 50_000_000] {
            let opts = AStarOptions {
                visit_limit: Some(limit),
                exhaustive: false,
                retain_trace: false,
                stop_on_first_touch: false,
            };
            let t = std::time::Instant::now();
            let r = astar_search(&ctx, &DijkstraModel, &src_set, dst, &opts);
            let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
            match r.path {
                Some(path) => {
                    println!(
                        "TGT ({},{}) {}: REACHABLE in {} pips, limit={} ({:.1} ms)",
                        tx, ty, tname, path.len(), limit, dt_ms
                    );
                    break;
                }
                None => {
                    println!(
                        "TGT ({},{}) {}: UNREACHABLE at limit={} ({:.1} ms)",
                        tx, ty, tname, limit, dt_ms
                    );
                }
            }
        }
    }
}

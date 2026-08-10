//! Dijkstra reachability probe: can IO0_O@(0,98) reach BUFG0_I@(310,111)?
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
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    let db = ctx.chipdb();

    // Source: IO0_O at (0, 98)
    let src_tile = db.tile_by_xy(0, 98);
    let src = find_wire_by_name(db, src_tile, "IO0_O").expect("IO0_O");
    println!(
        "src: IO0_O @ tile({},{}) = {}:{}",
        0,
        98,
        src.tile(),
        src.index()
    );

    // Destination: BUFG0_I at (310, 111)
    let dst_tile = db.tile_by_xy(310, 111);
    let dst = find_wire_by_name(db, dst_tile, "BUFG0_I").expect("BUFG0_I");
    println!(
        "dst: BUFG0_I @ tile({},{}) = {}:{}",
        310,
        111,
        dst.tile(),
        dst.index()
    );

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    db.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });
    println!("src node size: {}", src_set.len());

    let mut dst_set: FxHashSet<WireId> = FxHashSet::default();
    dst_set.insert(dst);
    db.node_wires_cb(dst, |nw| {
        dst_set.insert(nw);
    });
    println!("dst node size: {}", dst_set.len());

    for &limit in &[1_000_000usize, 10_000_000, 50_000_000, 100_000_000] {
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
                    "REACHABLE in {} pips, limit={} ({:.1} ms)",
                    path.len(),
                    limit,
                    dt_ms
                );
                // Print first 10 and last 10 pips with wire names.
                for (i, &pip) in path.iter().enumerate() {
                    if i < 10 || i >= path.len().saturating_sub(10) {
                        let p = ctx.pip(pip);
                        let sw = p.src_wire();
                        let dw = p.dst_wire();
                        let sname = db.wire_name(sw.id()).to_string();
                        let dname = db.wire_name(dw.id()).to_string();
                        let (sx, sy) = db.tile_xy(sw.id().tile());
                        let (dx, dy) = db.tile_xy(dw.id().tile());
                        println!(
                            "  pip[{:3}]: {}@({},{}) -> {}@({},{})",
                            i, sname, sx, sy, dname, dx, dy
                        );
                    } else if i == 10 {
                        println!("  ...");
                    }
                }
                return;
            }
            None => {
                println!("UNREACHABLE at limit={} ({:.1} ms)", limit, dt_ms);
            }
        }
    }
}

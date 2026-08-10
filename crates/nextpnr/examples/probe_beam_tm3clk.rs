//! Replay the router's beam search / A* cleanup on tm3_clk_v2 to see why
//! it fails. Runs three configs on the same IO0_O@(0,98) -> BUFG0_I@(310,111)
//! route:
//!   A. BEAM cost model w/ weighted manhattan heuristic (BEAM_H_WEIGHT=1000),
//!      visit_limit=100_000 — mirrors beam_search_route in raster.rs.
//!   B. Same cost model, visit_limit=500_000 — mirrors per_sink_limit in cleanup.
//!   C. Same cost model, visit_limit=5_000_000 — diagnostic: how big does the
//!      budget need to be?
//!   D. Admissible manhattan heuristic (h weight = 1), visit_limit=500_000
//!      — to test whether inflated heuristic is the problem.
//!   E. Pure Dijkstra (h = 0), same visit_limit as D.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{
    astar_search, default_pip_cost, AStarExit, AStarOptions, PathCostModel,
};
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

const BEAM_H_WEIGHT: DelayT = 1000;

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

struct BeamModel<'a> {
    chipdb: &'a ChipDb,
    h_weight: DelayT,
}
impl<'a> PathCostModel for BeamModel<'a> {
    fn pip_cost(&self, _ctx: &Context, _pip: PipId) -> DelayT {
        1
    }
    fn heuristic(&self, _ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        let (wx, wy) = self.chipdb.tile_xy(wire.tile());
        let (dx, dy) = self.chipdb.tile_xy(dst.tile());
        let m = (wx - dx).abs() + (wy - dy).abs();
        (m as DelayT).saturating_mul(self.h_weight)
    }
}

struct DefaultPipCostModel<'a> {
    chipdb: &'a ChipDb,
    h_weight: DelayT,
}
impl<'a> PathCostModel for DefaultPipCostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, _ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        let (wx, wy) = self.chipdb.tile_xy(wire.tile());
        let (dx, dy) = self.chipdb.tile_xy(dst.tile());
        let m = (wx - dx).abs() + (wy - dy).abs();
        (m as DelayT).saturating_mul(self.h_weight)
    }
}

fn run<M: PathCostModel>(
    label: &str,
    ctx: &Context,
    model: &M,
    src_set: &FxHashSet<WireId>,
    dst: WireId,
    visit_limit: usize,
    stop_on_first_touch: bool,
) {
    let opts = AStarOptions {
        visit_limit: Some(visit_limit),
        exhaustive: false,
        retain_trace: true,
        stop_on_first_touch,
    };
    let t = std::time::Instant::now();
    let r = astar_search(ctx, model, src_set, dst, &opts);
    let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
    let exit = r.trace.exit;
    let visits = r.trace.visit_count;
    let visited_len = r.trace.visited.len();
    match r.path {
        Some(path) => {
            println!(
                "  [{}] OK path={} pips visits={} visited_wires={} exit={:?} ({:.1} ms)",
                label,
                path.len(),
                visits,
                visited_len,
                exit,
                dt_ms
            );
        }
        None => {
            println!(
                "  [{}] FAIL visits={} visited_wires={} exit={:?} ({:.1} ms)",
                label,
                visits,
                visited_len,
                exit,
                dt_ms
            );
            // Also report: what was the best f-value the search saw for dst?
            if let Some(&(cost, pen, _, _)) = r.trace.visited.get(&dst) {
                println!(
                    "        dst was visited at cost={} penalty={} (score={})",
                    cost,
                    pen,
                    cost + pen
                );
            }
            // Max manhattan reached.
            let chipdb = ctx.chipdb();
            let (dx, dy) = chipdb.tile_xy(dst.tile());
            let (src_x, src_y) = {
                let w = *src_set.iter().next().unwrap();
                chipdb.tile_xy(w.tile())
            };
            let src_manh = (src_x - dx).abs() + (src_y - dy).abs();
            let mut min_remaining = i32::MAX;
            let mut at_min: Option<(i32, i32)> = None;
            for &w in r.trace.visited.keys() {
                let (wx, wy) = chipdb.tile_xy(w.tile());
                let m = (wx - dx).abs() + (wy - dy).abs();
                if m < min_remaining {
                    min_remaining = m;
                    at_min = Some((wx, wy));
                }
            }
            println!(
                "        src manhattan to dst={}; search reached min manhattan={} at {:?}",
                src_manh, min_remaining, at_min
            );
            // Dump all visited wires in tiles (309,111) and (310,111).
            let t309 = chipdb.tile_by_xy(309, 111);
            let t310 = chipdb.tile_by_xy(310, 111);
            println!("        visited wires at (309,111):");
            for (&w, &(cost, pen, pip, from)) in r.trace.visited.iter() {
                if w.tile() == t309 {
                    let name = chipdb.wire_name(w).to_string();
                    let from_name = chipdb.wire_name(from).to_string();
                    let (fx, fy) = chipdb.tile_xy(from.tile());
                    println!(
                        "          {} (idx={}) cost={} pen={} pip={:?} from={}@({},{})",
                        name, w.index(), cost, pen, pip.is_some(), from_name, fx, fy
                    );
                }
            }
            println!("        visited wires at (310,111):");
            for (&w, &(cost, pen, pip, from)) in r.trace.visited.iter() {
                if w.tile() == t310 {
                    let name = chipdb.wire_name(w).to_string();
                    let from_name = chipdb.wire_name(from).to_string();
                    let (fx, fy) = chipdb.tile_xy(from.tile());
                    println!(
                        "          {} (idx={}) cost={} pen={} pip={:?} from={}@({},{})",
                        name, w.index(), cost, pen, pip.is_some(), from_name, fx, fy
                    );
                }
            }
            // Count CLK_RELAY_IN_0_E visited at each x along y=98 and at x=309 for all y.
            println!("        CLK_RELAY_IN_0_E @ y=98 visited by x:");
            for x in 0..=310 {
                let tile = chipdb.tile_by_xy(x, 98);
                let mut found_e = None;
                let tt = chipdb.tile_type(tile);
                for (wi, wire) in tt.wires.get().iter().enumerate() {
                    let nid: i32 = unsafe { nextpnr::read_packed!(*wire, name) };
                    if let Some(s) = chipdb.constid_str(nid) {
                        if s == "CLK_RELAY_IN_0_E" {
                            let w = WireId::new(tile, wi as i32);
                            if let Some(&(cost, _, _, _)) = r.trace.visited.get(&w) {
                                found_e = Some(cost);
                                break;
                            }
                        }
                    }
                }
                if let Some(c) = found_e {
                    if x <= 5 || x >= 305 {
                        println!("          x={} cost={}", x, c);
                    }
                }
            }
            // Also check x=309 for each y.
            let mut max_x_with_relay = 0;
            for x in 0..=310 {
                let tile = chipdb.tile_by_xy(x, 98);
                let tt = chipdb.tile_type(tile);
                for (wi, wire) in tt.wires.get().iter().enumerate() {
                    let nid: i32 = unsafe { nextpnr::read_packed!(*wire, name) };
                    if let Some(s) = chipdb.constid_str(nid) {
                        if s == "CLK_RELAY_IN_0_E" {
                            let w = WireId::new(tile, wi as i32);
                            if r.trace.visited.contains_key(&w) {
                                max_x_with_relay = x;
                                break;
                            }
                        }
                    }
                }
            }
            println!("        max x at y=98 with CLK_RELAY_IN_0_E visited: {}", max_x_with_relay);
            // Also check CLK_RELAY_IN_0_W, _N, _S
            for wname in &["CLK_RELAY_IN_0_W", "CLK_RELAY_IN_0_N", "CLK_RELAY_IN_0_S"] {
                let mut max_x = -1;
                for x in 0..=310 {
                    let tile = chipdb.tile_by_xy(x, 98);
                    let tt = chipdb.tile_type(tile);
                    for (wi, wire) in tt.wires.get().iter().enumerate() {
                        let nid: i32 = unsafe { nextpnr::read_packed!(*wire, name) };
                        if let Some(s) = chipdb.constid_str(nid) {
                            if s == *wname {
                                let w = WireId::new(tile, wi as i32);
                                if r.trace.visited.contains_key(&w) {
                                    max_x = x;
                                    break;
                                }
                            }
                        }
                    }
                }
                println!("        max x at y=98 with {} visited: {}", wname, max_x);
            }
        }
    }
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
        });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    let db = ctx.chipdb();

    let src_tile = db.tile_by_xy(0, 98);
    let src = find_wire_by_name(db, src_tile, "IO0_O").expect("IO0_O");
    let dst_tile = db.tile_by_xy(310, 111);
    let dst = find_wire_by_name(db, dst_tile, "BUFG0_I").expect("BUFG0_I");

    println!(
        "src: IO0_O @ ({},{}) = {}:{}; dst: BUFG0_I @ ({},{}) = {}:{}",
        0, 98, src.tile(), src.index(), 310, 111, dst.tile(), dst.index()
    );

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    db.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });
    println!("src node size: {}", src_set.len());

    let src_info = db.wire_info(src);
    let dst_info = db.wire_info(dst);
    println!(
        "src.pips_downhill={}, dst.pips_uphill={}",
        src_info.pips_downhill.get().len(),
        dst_info.pips_uphill.get().len()
    );

    println!();
    println!("=== beam config: pip_cost=1, h = manhattan*1000 (BEAM_H_WEIGHT) ===");
    let beam_model = BeamModel {
        chipdb: db,
        h_weight: BEAM_H_WEIGHT,
    };
    run("A beam 100k stop-on-touch", &ctx, &beam_model, &src_set, dst, 100_000, true);
    run("B cleanup 500k stop-on-touch", &ctx, &beam_model, &src_set, dst, 500_000, true);
    run("C big 5M stop-on-touch", &ctx, &beam_model, &src_set, dst, 5_000_000, true);
    run("A-nostop 100k", &ctx, &beam_model, &src_set, dst, 100_000, false);
    run("B-nostop 500k", &ctx, &beam_model, &src_set, dst, 500_000, false);
    run("C-nostop 5M", &ctx, &beam_model, &src_set, dst, 5_000_000, false);

    println!();
    println!("=== admissible: pip_cost=default, h = manhattan*1 ===");
    let adm = DefaultPipCostModel { chipdb: db, h_weight: 1 };
    run("D1 500k", &ctx, &adm, &src_set, dst, 500_000, false);
    run("D2 5M", &ctx, &adm, &src_set, dst, 5_000_000, false);

    println!();
    println!("=== dijkstra: pip_cost=default, h = 0 ===");
    let dij = DefaultPipCostModel { chipdb: db, h_weight: 0 };
    run("E 5M", &ctx, &dij, &src_set, dst, 5_000_000, false);
}

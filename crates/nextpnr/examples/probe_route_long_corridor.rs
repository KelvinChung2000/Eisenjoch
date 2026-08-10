//! Diagnose why the raster router fails on long IO-pad → fabric nets.
//!
//! Focus: `tm3_vidin_vs` on sv3 at opt_trans Steiner λ=0.05. Driver is at an
//! IO pad on the left edge; the eight sinks pile up in a narrow cluster on
//! the east side of the device once the Steiner term pulls them together.
//!
//! Questions this probe answers for the worst-distance sink:
//!   1. Is the sink reachable at all? (unbounded astar, 1M visit budget)
//!   2. Does the ±k corridor bbox prevent reachability? (k ∈ {2, 5, 11})
//!   3. Is it a visit-budget problem? (unbounded bbox, 100k budget, mimicking
//!      beam_search_route's default)
//!   4. Where does the search tail off? (visited-wire count + exit reason)
//!
//! The hypothesis space is:
//!   A. Chipdb connectivity gap         → all runs fail with HeapDrained
//!   B. Corridor too narrow             → unbounded succeeds, ±2 fails
//!   C. Visit cap too small for length  → all HeapDrained or VisitLimit,
//!                                         unbounded with 1M succeeds
//!   D. Rip-up negotiation needed       → bindings not present in probe (new
//!                                         ctx), so if probe succeeds but
//!                                         real router fails, it's D

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::metrics::BoundingBox;
use nextpnr::netlist::NetId;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::router::astar::{
    astar_search, default_pip_cost, AStarExit, AStarOptions, PathCostModel,
};
use nextpnr::router::common::{collect_sink_wires, resolve_source_wire};
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn find_net(ctx: &Context, name: &str) -> Option<NetId> {
    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if ctx.name_of(net.name_id()) == name {
            return Some(net_idx);
        }
    }
    None
}

/// Cost model: manhattan heuristic, default pip cost, optional bbox corridor.
struct ProbeModel<'a> {
    bboxes: &'a [BoundingBox],
    dst_x: i32,
    dst_y: i32,
}

impl<'a> PathCostModel for ProbeModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, _dst: WireId) -> DelayT {
        let (wx, wy) = ctx.chipdb().tile_xy(wire.tile());
        (wx - self.dst_x).abs() + (wy - self.dst_y).abs()
    }
    fn bboxes(&self) -> &[BoundingBox] {
        self.bboxes
    }
}

/// Build one bbox covering the rectangle from src_tile to dst_tile, padded by
/// `margin` on every side, clamped to the chip.
fn line_bbox(chipdb: &ChipDb, sx: i32, sy: i32, dx: i32, dy: i32, margin: i32) -> BoundingBox {
    let w = chipdb.width();
    let h = chipdb.height();
    let x0 = (sx.min(dx) - margin).max(0);
    let x1 = (sx.max(dx) + margin).min(w - 1);
    let y0 = (sy.min(dy) - margin).max(0);
    let y1 = (sy.max(dy) + margin).min(h - 1);
    BoundingBox { x0, y0, x1, y1 }
}

fn run_one(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    dst_x: i32,
    dst_y: i32,
    bboxes: &[BoundingBox],
    visit_limit: usize,
    label: &str,
) {
    let model = ProbeModel {
        bboxes,
        dst_x,
        dst_y,
    };
    let opts = AStarOptions {
        visit_limit: Some(visit_limit),
        exhaustive: false,
        retain_trace: true,
        stop_on_first_touch: false,
    };
    let res = astar_search(ctx, &model, src_wires, dst_wire, &opts);
    let exit = match res.trace.exit {
        AStarExit::Reached => "REACHED",
        AStarExit::HeapDrained => "HEAP_DRAINED",
        AStarExit::VisitLimit => "VISIT_LIMIT",
    };
    let path_len = res.path.as_ref().map(|p| p.len()).unwrap_or(0);
    let visited = res.trace.visited.len();
    let score = if res.trace.best_score == DelayT::MAX {
        -1
    } else {
        res.trace.best_score
    };
    let bbox_str = if bboxes.is_empty() {
        "unbounded".to_string()
    } else {
        let b = &bboxes[0];
        format!("[{}..{}, {}..{}] ({}x{})", b.x0, b.x1, b.y0, b.y1, b.x1 - b.x0 + 1, b.y1 - b.y0 + 1)
    };
    println!(
        "  {:<22}  bbox={:<30}  budget={:>7}  exit={:<12}  visits={:>7}  path_pips={:>4}  score={}",
        label, bbox_str, visit_limit, exit, res.trace.visit_count, path_len, score
    );
    println!(
        "                          visited_wires_in_map={} (distinct wires touched)",
        visited
    );
}

fn diagnose_net(ctx: &Context, net_name: &str) {
    let Some(net_idx) = find_net(ctx, net_name) else {
        println!("net '{}' not found", net_name);
        return;
    };
    let src_wire = match resolve_source_wire(ctx, net_idx) {
        Ok(Some(w)) => w,
        Ok(None) => {
            println!("net '{}' has no source (no driver placed)", net_name);
            return;
        }
        Err(e) => {
            println!("net '{}' resolve_source_wire error: {}", net_name, e);
            return;
        }
    };
    let sink_wires = collect_sink_wires(ctx, net_idx);
    if sink_wires.is_empty() {
        println!("net '{}' has no sinks", net_name);
        return;
    }

    let chipdb = ctx.chipdb();
    let (src_x, src_y) = chipdb.tile_xy(src_wire.tile());

    println!("\n========== net '{}' ==========", net_name);
    println!(
        "  driver wire = {:?}  tile=({},{})  pips_downhill={}  pips_uphill={}",
        src_wire,
        src_x,
        src_y,
        chipdb.wire_info(src_wire).pips_downhill.get().len(),
        chipdb.wire_info(src_wire).pips_uphill.get().len(),
    );

    // Unique downhill-reached tiles from driver (first-hop fan-out).
    let mut first_hop_tiles: FxHashSet<(i32, i32)> = FxHashSet::default();
    for &pip_i in chipdb.wire_info(src_wire).pips_downhill.get() {
        let pip = PipId::new(src_wire.tile(), pip_i);
        let dw = chipdb.pip_dst_wire(pip);
        first_hop_tiles.insert(chipdb.tile_xy(dw.tile()));
    }
    println!(
        "  driver first-hop tiles: {} unique  samples: {:?}",
        first_hop_tiles.len(),
        first_hop_tiles.iter().take(8).collect::<Vec<_>>()
    );

    // Build src set with node equivs (same as router does).
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src_wire);
    chipdb.node_wires_cb(src_wire, |nw| {
        src_set.insert(nw);
    });

    // Find worst (furthest) sink by manhattan.
    let mut worst: Option<(WireId, i32, i32, i32)> = None;
    println!("  sinks ({}):", sink_wires.len());
    for &sw in &sink_wires {
        let (sx, sy) = chipdb.tile_xy(sw.tile());
        let d = (sx - src_x).abs() + (sy - src_y).abs();
        let sw_up = chipdb.wire_info(sw).pips_uphill.get().len();
        let sw_down = chipdb.wire_info(sw).pips_downhill.get().len();
        println!(
            "    sink {:?} tile=({},{})  manhattan={}  up={}  down={}",
            sw, sx, sy, d, sw_up, sw_down
        );
        if worst.map_or(true, |(_, _, _, wd)| d > wd) {
            worst = Some((sw, sx, sy, d));
        }
    }
    let (worst_sink, wx, wy, wd) = worst.unwrap();
    println!(
        "  worst sink: {:?} tile=({},{}) manhattan={}",
        worst_sink, wx, wy, wd
    );

    // Confirm first-hop fan-out: how many downhill destinations fall inside a
    // ±2 bbox around the source tile and inside a ±2 bbox around the raster
    // line? Tight answer here tells us whether the IO pad can even step off
    // the edge inside a narrow corridor.
    let line_bb_2 = line_bbox(chipdb, src_x, src_y, wx, wy, 2);
    let line_bb_5 = line_bbox(chipdb, src_x, src_y, wx, wy, 5);
    let line_bb_11 = line_bbox(chipdb, src_x, src_y, wx, wy, 11);
    let in_bb = |bb: &BoundingBox, x: i32, y: i32| {
        x >= bb.x0 && x <= bb.x1 && y >= bb.y0 && y <= bb.y1
    };
    let fh_in2 = first_hop_tiles.iter().filter(|(x, y)| in_bb(&line_bb_2, *x, *y)).count();
    let fh_in5 = first_hop_tiles.iter().filter(|(x, y)| in_bb(&line_bb_5, *x, *y)).count();
    let fh_in11 = first_hop_tiles.iter().filter(|(x, y)| in_bb(&line_bb_11, *x, *y)).count();
    println!(
        "  driver first-hop tiles inside line bbox: margin=2 → {}/{}   margin=5 → {}/{}   margin=11 → {}/{}",
        fh_in2,
        first_hop_tiles.len(),
        fh_in5,
        first_hop_tiles.len(),
        fh_in11,
        first_hop_tiles.len()
    );

    // Run A* variants.
    println!("\n  --- A* runs to worst sink ---");

    // (1) Unbounded bbox, huge budget: ground-truth reachability.
    run_one(
        ctx,
        &src_set,
        worst_sink,
        wx,
        wy,
        &[],
        1_000_000,
        "unbounded, 1M",
    );

    // (2) Unbounded bbox, 100k budget: is it budget-bounded?
    run_one(
        ctx,
        &src_set,
        worst_sink,
        wx,
        wy,
        &[],
        100_000,
        "unbounded, 100k",
    );

    // (3-5) Corridor widths.
    for (m, budget) in [(2i32, 100_000usize), (5, 200_000), (11, 400_000)] {
        let bb = line_bbox(chipdb, src_x, src_y, wx, wy, m);
        run_one(
            ctx,
            &src_set,
            worst_sink,
            wx,
            wy,
            std::slice::from_ref(&bb),
            budget,
            &format!("line±{}", m),
        );
    }

    // (6) Reachability: exhaustive Dijkstra (h=0) from driver, 5M budget.
    //     Does the sink wire or its tile ever appear in `visited`?
    println!("  --- reachability check (exhaustive Dijkstra, h=0, 5M budget) ---");
    struct DijkstraModel;
    impl PathCostModel for DijkstraModel {
        fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
            default_pip_cost(ctx, pip)
        }
        fn heuristic(&self, _ctx: &Context, _wire: WireId, _dst: WireId) -> DelayT {
            0
        }
    }
    let opts = AStarOptions {
        visit_limit: Some(5_000_000),
        exhaustive: true,
        retain_trace: true,
        stop_on_first_touch: false,
    };
    let res = astar_search(ctx, &DijkstraModel, &src_set, worst_sink, &opts);
    let exit = match res.trace.exit {
        AStarExit::Reached => "REACHED",
        AStarExit::HeapDrained => "HEAP_DRAINED",
        AStarExit::VisitLimit => "VISIT_LIMIT",
    };
    let sink_visited = res.trace.visited.contains_key(&worst_sink);
    let sink_tile = worst_sink.tile();
    let wires_in_sink_tile: usize = res
        .trace
        .visited
        .keys()
        .filter(|w| w.tile() == sink_tile)
        .count();
    let mut tiles_reached: FxHashSet<i32> = FxHashSet::default();
    for w in res.trace.visited.keys() {
        tiles_reached.insert(w.tile());
    }
    println!(
        "  exit={}  visits={}  visited_wires={}  distinct_tiles_reached={}",
        exit,
        res.trace.visit_count,
        res.trace.visited.len(),
        tiles_reached.len(),
    );
    println!(
        "  sink_wire_visited={}  wires_in_sink_tile={}  sink_tile={}",
        sink_visited, wires_in_sink_tile, sink_tile,
    );

    // Per-sink reachability: was the sink (or its tile) touched by the
    // exhaustive Dijkstra? Answers the "patchy-connectivity" hypothesis —
    // some sinks may be in reachable tiles, others not.
    println!("  --- per-sink reachability (same Dijkstra run) ---");
    let mut reached_sinks = 0usize;
    let mut reached_tiles = 0usize;
    for &sw in &sink_wires {
        let (sx, sy) = chipdb.tile_xy(sw.tile());
        let d = (sx - src_x).abs() + (sy - src_y).abs();
        let wire_in = res.trace.visited.contains_key(&sw);
        let tile_hits = res
            .trace
            .visited
            .keys()
            .filter(|w| w.tile() == sw.tile())
            .count();
        if wire_in {
            reached_sinks += 1;
        }
        if tile_hits > 0 {
            reached_tiles += 1;
        }
        println!(
            "    sink tile=({:>3},{:>3}) dist={:<4} wire_reached={} wires_in_tile={}",
            sx, sy, d, wire_in, tile_hits
        );
    }
    println!(
        "  summary: sinks_reached_wire={}/{}, sinks_reached_tile={}/{}",
        reached_sinks,
        sink_wires.len(),
        reached_tiles,
        sink_wires.len()
    );
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });
    let lambda: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);

    println!("========== Placing sv3 with opt_trans Steiner λ={} ==========", lambda);
    let mut ctx = load_fresh(&chipdb_path, &design_path);
    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = 50;
    cfg.num_threads = 8;
    cfg.steiner_weight = lambda;
    place_opt_trans(&mut ctx, &cfg).expect("opt_trans place");

    for name in ["tm3_vidin_vs", "tm3_vidin_cref", "tm3_vidin_vpo[0]"] {
        diagnose_net(&ctx, name);
    }
}

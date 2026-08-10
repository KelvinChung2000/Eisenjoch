//! Probe: replicate what A* cleanup does for $signal$257.
//! Uses the Lookahead heuristic + visit_limit=500_000 just like raster.rs
//! A* cleanup does. If this REACHES, the router is losing the path
//! somewhere in apply_route_plan or the cleanup flow. If this FAILS, the
//! lookahead heuristic is misleading A* past its visit budget.
use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarExit, AStarOptions, PathCostModel};
use nextpnr::router::lookahead::Lookahead;
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

struct LookaheadModel<'a> {
    la: &'a Lookahead,
}

impl<'a> PathCostModel for LookaheadModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        self.la.estimate_delay(ctx.chipdb(), wire, dst)
    }
}

struct NoHeuristic;
impl PathCostModel for NoHeuristic {
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

fn try_with_model<M: PathCostModel>(
    label: &str,
    ctx: &Context,
    model: &M,
    src_set: &FxHashSet<WireId>,
    dst: WireId,
    visit_limit: usize,
) {
    let opts = AStarOptions {
        visit_limit: Some(visit_limit),
        exhaustive: false,
        retain_trace: true,
        stop_on_first_touch: false,
    };
    let t = std::time::Instant::now();
    let r = astar_search(ctx, model, src_set, dst, &opts);
    let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
    let exit = match r.trace.exit {
        AStarExit::Reached => "Reached",
        AStarExit::VisitLimit => "VisitLimit",
        AStarExit::HeapDrained => "HeapDrained",
    };
    println!(
        "  [{}] exit={} visits={}/{} best_score={} time={:.1}ms path_len={}",
        label,
        exit,
        r.trace.visit_count,
        r.trace.max_visits,
        r.trace.best_score,
        dt_ms,
        r.path.as_ref().map(|p| p.len()).unwrap_or(0),
    );
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);

    println!("Building lookahead...");
    let t = std::time::Instant::now();
    let lookahead = Lookahead::build(&ctx);
    println!("  built in {:.1}s", t.elapsed().as_secs_f64());

    let chipdb = ctx.chipdb();

    // Case 1: $signal$257 (easiest — manhattan=3, Dijkstra found 15 pips)
    println!("\n=== $signal$257: M3_CLBLM_M_D@(260,113) -> M3_CLBLM_M_CX@(262,112) ===");
    let src = find_wire_by_name(chipdb, chipdb.tile_by_xy(260, 113), "M3_CLBLM_M_D").expect("src");
    let dst = find_wire_by_name(chipdb, chipdb.tile_by_xy(262, 112), "M3_CLBLM_M_CX").expect("dst");
    let h = lookahead.estimate_delay(chipdb, src, dst);
    println!("  heuristic(src, dst) = {}", h);

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    chipdb.node_wires_cb(src, |nw| { src_set.insert(nw); });
    println!("  src_set.len={}", src_set.len());

    for &limit in &[500_000usize, 5_000_000, 50_000_000] {
        try_with_model(&format!("lookahead limit={}", limit), &ctx, &LookaheadModel { la: &lookahead }, &src_set, dst, limit);
    }
    try_with_model("no_heuristic limit=50M", &ctx, &NoHeuristic, &src_set, dst, 50_000_000);
}

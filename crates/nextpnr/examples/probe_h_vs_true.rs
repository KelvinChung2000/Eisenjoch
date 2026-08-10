//! Quantify how loose the admissible lookahead is on failing IO/clock
//! routes. For each (src, dst) pair, report:
//!   h_lookahead = lookahead.estimate_delay(src, dst)   (admissible LB)
//!   true_min   = un-budgeted Dijkstra h=0, ignore_bindings  (actual LB)
//!   ratio      = true_min / h_lookahead   (>= 1; bigger = looser)
//!
//! If h is much smaller than true_min, A* with an admissible h explores
//! every wire in the gap, hits the visit budget, and fails.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::router::lookahead::Lookahead;
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;
use std::time::Instant;

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

fn measure(
    ctx: &Context,
    la: &Lookahead,
    label: &str,
    src_name: &str,
    sx: i32,
    sy: i32,
    dst_name: &str,
    dx: i32,
    dy: i32,
) {
    let db = ctx.chipdb();
    let st = db.tile_by_xy(sx, sy);
    let dt = db.tile_by_xy(dx, dy);
    let Some(src) = find_wire_by_name(db, st, src_name) else {
        println!("  !! {}: src {} not found at ({},{})", label, src_name, sx, sy);
        return;
    };
    let Some(dst) = find_wire_by_name(db, dt, dst_name) else {
        println!("  !! {}: dst {} not found at ({},{})", label, dst_name, dx, dy);
        return;
    };

    let h = la.estimate_delay(db, src, dst, nextpnr::router::lookahead::UNKNOWN_CLASS);
    let manhattan = (dx - sx).abs() + (dy - sy).abs();

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    db.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });

    // Un-budgeted Dijkstra (h=0). Visit cap is generous; if we can't
    // reach within 50M pops, treat as "reachable cost > 50M" — that
    // already proves h is utterly loose.
    let opts = AStarOptions {
        visit_limit: Some(50_000_000),
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    let t0 = Instant::now();
    let r = astar_search(ctx, &DijkstraModel, &src_set, dst, &opts);
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    match r.path {
        Some(path) => {
            // Reconstruct path cost.
            let mut g = 0i64;
            for &pip in &path {
                g += default_pip_cost(ctx, pip) as i64;
            }
            let ratio = if h > 0 { g as f64 / h as f64 } else { f64::INFINITY };
            println!(
                "{:<26}  manhattan={:>4}  h_lookahead={:>6}  true_cost={:>6}  ratio={:>5.2}  pips={:>4}  dijkstra_ms={:.0}",
                label, manhattan, h, g, ratio, path.len(), dt_ms
            );
        }
        None => {
            println!(
                "{:<26}  manhattan={:>4}  h_lookahead={:>6}  true_cost=UNREACHABLE@50M  pips=0  dijkstra_ms={:.0}",
                label, manhattan, h, dt_ms
            );
        }
    }
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    println!("Building lookahead...");
    let t0 = Instant::now();
    let la = Lookahead::build(&ctx);
    println!("  built in {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!();

    println!(
        "{:<26}  {:>9}  {:>11}  {:>10}  {:>5}  {:>4}",
        "case", "manhattan", "h_lookahead", "true_cost", "ratio", "pips"
    );

    // IO-pad-driven failing nets observed in the per-edge sv3 log. These
    // are the routes that exit=VisitLimit at 500k pops.
    measure(&ctx, &la, "tm3_vidin_cref ->130,118", "IO0_O", 0, 183, "M3_CLBLM_M_D1", 130, 118);
    measure(&ctx, &la, "tm3_vidin_cref ->148,102", "IO0_O", 0, 183, "M3_CLBLM_M_D1", 148, 102);
    measure(&ctx, &la, "tm3_vidin_cref ->199,90",  "IO0_O", 0, 183, "M3_CLBLM_M_D1", 199, 90);
    measure(&ctx, &la, "tm3_vidin_vs   ->104,105", "IO1_O", 0, 152, "M3_CLBLM_M_D1", 104, 105);
    measure(&ctx, &la, "tm3_vidin_vs   ->148,113", "IO1_O", 0, 152, "M3_CLBLM_M_D1", 148, 113);
    measure(&ctx, &la, "tm3_vidin_href ->189,117", "IO0_O", 309, 119, "M3_CLBLM_M_D1", 189, 117);
    measure(&ctx, &la, "tm3_vidin_href ->131,106", "IO0_O", 309, 119, "M3_CLBLM_M_D1", 131, 106);

    // Reference: a successful nearby CLB->CLB hop. Should have ratio ~ 1.
    println!();
    println!("Reference (successful short routes):");
    measure(&ctx, &la, "M3 short hop 1",        "M3_CLBLM_M_D", 100, 100, "M3_CLBLM_M_CX", 102, 99);
    measure(&ctx, &la, "M3 short hop 2",        "M3_CLBLM_M_A", 145, 108, "M3_CLBLM_M_DX", 144, 108);
}

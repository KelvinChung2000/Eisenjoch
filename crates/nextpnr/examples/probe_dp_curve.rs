//! Dump the DP cost curve for one specific class/side, to check
//! admissibility against a probe Dijkstra.

use nextpnr::chipdb::{ChipDb, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::router::lookahead::Lookahead;
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

struct Dij;
impl PathCostModel for Dij {
    fn pip_cost(&self, ctx: &Context, p: nextpnr::chipdb::PipId) -> DelayT {
        default_pip_cost(ctx, p)
    }
    fn heuristic(&self, _: &Context, _: WireId, _: WireId) -> DelayT {
        0
    }
}

fn find_wire(db: &ChipDb, tile: i32, target: &str) -> Option<WireId> {
    let tt = db.tile_type(tile);
    for (wi, w) in tt.wires.get().iter().enumerate() {
        let nid: i32 = unsafe { nextpnr::read_packed!(*w, name) };
        if let Some(s) = db.constid_str(nid) {
            if s == target {
                return Some(WireId::new(tile, wi as i32));
            }
        }
    }
    None
}

fn main() {
    let p = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let db = ChipDb::load(Path::new(p)).expect("load");
    let ctx = Context::new(db);
    let la = Lookahead::build(&ctx);
    let cdb = ctx.chipdb();

    // tm3_vidin_href: IO0_O@(309,119) → M3_CLBLM_M_D1@(189,117), dx=-120 (W), dy=-2 (N).
    let src_tile = 119 * cdb.width() + 309;
    let src = find_wire(cdb, src_tile, "IO0_O").expect("IO0_O");
    let (sx, sy) = cdb.tile_xy(src.tile());
    println!(
        "src @ ({sx},{sy}) tt={} name=IO0_O",
        cdb.tile_type_name(src.tile())
    );

    println!("h(src, dst) for varying dst at offset (dx, dy) from src:");
    for &(dx, dy) in &[
        (-1, 0),
        (-12, 0),
        (-24, 0),
        (-60, 0),
        (-120, 0),
        (0, -2),
        (-120, -2),
    ] {
        let tx = (sx + dx).clamp(0, cdb.width() - 1);
        let ty = (sy + dy).clamp(0, cdb.height() - 1);
        let dst_tile = ty * cdb.width() + tx;
        let dst = WireId::new(dst_tile, 0);
        let h = la.estimate_delay(cdb, src, dst, nextpnr::router::lookahead::UNKNOWN_CLASS);
        println!("  (dx={dx:>4}, dy={dy:>3}): h_lookahead={h:>6}");
    }

    // For the most-egregious case, run Dijkstra to get the true cost
    // assuming ANY destination wire on the dst tile is acceptable.
    println!("\nDijkstra (h=0) min cost from IO0_O@(309,119) to any wire on tile (189,117):");
    let dst_tile = 117 * cdb.width() + 189;
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    cdb.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });
    // Pick one dst wire (needs to be a real wire in dst_tile).
    let dst = find_wire(cdb, dst_tile, "M3_CLBLM_M_D1").expect("dst");
    let opts = AStarOptions {
        visit_limit: Some(20_000_000),
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    let r = astar_search(&ctx, &Dij, &src_set, dst, &opts);
    if let Some(path) = r.path {
        let g: i64 = path
            .iter()
            .map(|&pip| default_pip_cost(&ctx, pip) as i64)
            .sum();
        println!("  true_cost = {g}, pips = {}", path.len());
    } else {
        println!("  Dijkstra exhausted budget without reaching dst");
    }
}

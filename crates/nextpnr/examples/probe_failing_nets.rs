//! Pure-chipdb reachability probe for the 3 sv3 failing nets.
//! If these sinks are unreachable from their sources, it's a chipdb problem.
//! If they ARE reachable, the router is the bottleneck.
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

fn try_route(ctx: &Context, src_name: &str, sx: i32, sy: i32, dst_name: &str, dx: i32, dy: i32) {
    let db = ctx.chipdb();
    let src_tile = db.tile_by_xy(sx, sy);
    let Some(src) = find_wire_by_name(db, src_tile, src_name) else {
        println!("  !! src {}@({},{}) NOT FOUND (tile_type={})",
            src_name, sx, sy, db.tile_type_name(src_tile));
        return;
    };
    let dst_tile = db.tile_by_xy(dx, dy);
    let Some(dst) = find_wire_by_name(db, dst_tile, dst_name) else {
        println!("  !! dst {}@({},{}) NOT FOUND (tile_type={})",
            dst_name, dx, dy, db.tile_type_name(dst_tile));
        return;
    };

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    db.node_wires_cb(src, |nw| { src_set.insert(nw); });

    let mut dst_set: FxHashSet<WireId> = FxHashSet::default();
    dst_set.insert(dst);
    db.node_wires_cb(dst, |nw| { dst_set.insert(nw); });
    println!("  src_node={} dst_node={} manhattan={}",
        src_set.len(), dst_set.len(), (dx - sx).abs() + (dy - sy).abs());

    for &limit in &[10_000_000usize, 50_000_000, 200_000_000] {
        let opts = AStarOptions {
            visit_limit: Some(limit),
            exhaustive: false,
            retain_trace: false,
            stop_on_first_touch: false,
        };
        let t = std::time::Instant::now();
        let r = astar_search(ctx, &DijkstraModel, &src_set, dst, &opts);
        let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
        match r.path {
            Some(path) => {
                println!("  REACHABLE in {} pips at limit={} ({:.1} ms)", path.len(), limit, dt_ms);
                return;
            }
            None => {
                println!("  UNREACHABLE at limit={} ({:.1} ms)", limit, dt_ms);
            }
        }
    }
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);

    println!("\n== $signal$257: M3_CLBLM_M_D@(260,113) -> M3_CLBLM_M_CX@(262,112) ==");
    try_route(&ctx, "M3_CLBLM_M_D", 260, 113, "M3_CLBLM_M_CX", 262, 112);

    println!("\n== tm3_vidin_cref: IO0_O@(0,183) -> M3_CLBLM_M_D1@(2,95) ==");
    try_route(&ctx, "IO0_O", 0, 183, "M3_CLBLM_M_D1", 2, 95);

    println!("\n== tm3_vidin_vs: IO1_O@(0,152) -> M3_CLBLM_M_D1@(82,129) ==");
    try_route(&ctx, "IO1_O", 0, 152, "M3_CLBLM_M_D1", 82, 129);

    println!("\n== $signal$92: M3_CLBLM_M_A@(145,108) -> M3_CLBLM_M_DX@(144,108) ==");
    try_route(&ctx, "M3_CLBLM_M_A", 145, 108, "M3_CLBLM_M_DX", 144, 108);
}

//! Run the actual beam-search routine that sv3 uses, for the
//! tm3_vidin_cref's first failing sink. Dump astar visits, the
//! frontier expansion behaviour, and whether any path is found.
//!
//! Goal: find out why beam_search_route returns None even though the
//! IOB output now has direct LOGIC_OUTS_L0 → CLB tileconn.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::read_packed;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

const BEAM_H_WEIGHT: DelayT = 1000;

struct BeamModel<'a> {
    chipdb: &'a ChipDb,
}

impl<'a> PathCostModel for BeamModel<'a> {
    fn pip_cost(&self, _: &Context, _: PipId) -> DelayT { 1 }
    fn wire_penalty(&self, _: &Context, _: WireId) -> DelayT { 0 }
    fn heuristic(&self, _: &Context, w: WireId, dst: WireId) -> DelayT {
        let (wx, wy) = self.chipdb.tile_xy(w.tile());
        let (dx, dy) = self.chipdb.tile_xy(dst.tile());
        let m = (wx - dx).abs() + (wy - dy).abs();
        (m as DelayT).saturating_mul(BEAM_H_WEIGHT)
    }
}

fn find_wire(db: &ChipDb, tile: i32, name: &str) -> Option<WireId> {
    let tt = db.tile_type(tile);
    for (wi, w) in tt.wires.get().iter().enumerate() {
        let nid: i32 = unsafe { read_packed!(*w, name) };
        if let Some(s) = db.constid_str(nid) {
            if s == name {
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
    let cdb = ctx.chipdb();

    // Source: IO0_O at (0, 183)
    let src_tile = 183 * cdb.width();
    let src = find_wire(cdb, src_tile, "IO0_O").expect("IO0_O");
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    cdb.node_wires_cb(src, |nw| { src_set.insert(nw); });
    println!("src tile @ (0,183), IO0_O, src_set size = {}", src_set.len());

    // Try a CLOSE sink: (2, 95). Pick its first M1_LOGIC_OUTS_L0 wire as dst.
    // (the actual tm3_vidin_cref sink wire would be a SLICE input — we'll
    //  approximate by picking a representative fabric wire in the target tile.)
    // First test: trivial 1-hop neighbour, M1_LOGIC_OUTS_L0 at (1,183).
    {
        let neigh_tile = 183 * cdb.width() + 1;
        let dst_neigh = find_wire(cdb, neigh_tile, "M1_LOGIC_OUTS_L0").expect("neigh dst");
        let model_neigh = BeamModel { chipdb: cdb };
        let opts_neigh = AStarOptions {
            visit_limit: Some(10_000),
            exhaustive: false,
            retain_trace: true,
            stop_on_first_touch: true,
        };
        let r = astar_search(&ctx, &model_neigh, &src_set, dst_neigh, &opts_neigh);
        println!("\n[neighbour test] src=IO0_O@(0,183) -> dst=M1_LOGIC_OUTS_L0@(1,183): path={:?} visits={}", r.path.as_ref().map(|p| p.len()), r.trace.visit_count);
    }

    let dst_tile = 95 * cdb.width() + 2;
    println!("dst tile @ (2,95), tt = {}", cdb.tile_type_name(dst_tile));
    // Pick the wire with the most pips_uphill (representative).
    let dst_tt = cdb.tile_type(dst_tile);
    let mut best_dst: Option<(WireId, usize)> = None;
    for (wi, _) in dst_tt.wires.get().iter().enumerate() {
        let w = WireId::new(dst_tile, wi as i32);
        let info = cdb.wire_info(w);
        let up = info.pips_uphill.get().len();
        if up > best_dst.map(|x| x.1).unwrap_or(0) {
            best_dst = Some((w, up));
        }
    }
    let (dst, dst_up) = best_dst.expect("dst");
    let info = cdb.wire_info(dst);
    let nid: i32 = unsafe { read_packed!(*info, name) };
    println!("  picked dst wire: {} pips_uphill={dst_up}", cdb.constid_str(nid).unwrap_or("?"));

    // Run beam-style A*.
    let model = BeamModel { chipdb: cdb };
    let opts = AStarOptions {
        visit_limit: Some(100_000),
        exhaustive: false,
        retain_trace: true,
        stop_on_first_touch: true,
    };
    // Pure Dijkstra check: is dst reachable from src AT ALL?
    {
        struct Dij;
        impl PathCostModel for Dij {
            fn pip_cost(&self, _: &Context, _: PipId) -> DelayT { 1 }
            fn heuristic(&self, _: &Context, _: WireId, _: WireId) -> DelayT { 0 }
        }
        let opts_d = AStarOptions {
            visit_limit: Some(2_000_000),
            exhaustive: false,
            retain_trace: false,
            stop_on_first_touch: true,
        };
        let rd = astar_search(&ctx, &Dij, &src_set, dst, &opts_d);
        println!(
            "[dijkstra check] src=IO0_O@(0,183) -> dst=M1_IMUX_L0@(2,95): path = {:?} pips, visits = {}",
            rd.path.as_ref().map(|p| p.len()),
            rd.trace.visit_count,
        );
    }

    let r = astar_search(&ctx, &model, &src_set, dst, &opts);
    println!("\nastar result: path = {:?} pips, visits = {}", r.path.as_ref().map(|p| p.len()), r.trace.visit_count);
    if let Some(path) = &r.path {
        let total_cost: i64 = path.iter().map(|&pip| default_pip_cost(&ctx, pip) as i64).sum();
        println!("  total cost = {total_cost}");
        for (i, &pip) in path.iter().take(8).enumerate() {
            let s = cdb.pip_src_wire(pip);
            let d = cdb.pip_dst_wire(pip);
            let sn = {
                let info = cdb.wire_info(s);
                let nid: i32 = unsafe { read_packed!(*info, name) };
                cdb.constid_str(nid).unwrap_or("?")
            };
            let dn = {
                let info = cdb.wire_info(d);
                let nid: i32 = unsafe { read_packed!(*info, name) };
                cdb.constid_str(nid).unwrap_or("?")
            };
            let (sx, sy) = cdb.tile_xy(s.tile());
            let (dx, dy) = cdb.tile_xy(d.tile());
            println!("  [{i:>2}] ({sx},{sy}){sn} -> ({dx},{dy}){dn}");
        }
        if path.len() > 8 { println!("  ... +{} more", path.len() - 8); }
    } else {
        // Diagnose: what's in the trace.visited frontier? How far did we get?
        let v = &r.trace.visited;
        let (sx, sy) = cdb.tile_xy(src.tile());
        let mut max_dx: i32 = 0;
        let mut max_dy: i32 = 0;
        let mut tile_set: FxHashSet<i32> = FxHashSet::default();
        let mut visited_dst_tile: usize = 0;
        let dst_t = dst.tile();
        for (&w, _) in v.iter() {
            let (wx, wy) = cdb.tile_xy(w.tile());
            let dx = (wx - sx).abs();
            let dy = (wy - sy).abs();
            if dx > max_dx { max_dx = dx; }
            if dy > max_dy { max_dy = dy; }
            tile_set.insert(w.tile());
            if w.tile() == dst_t { visited_dst_tile += 1; }
        }
        println!("  visited entries: {} wires across {} tiles, max_dx={max_dx}, max_dy={max_dy}", v.len(), tile_set.len());
        println!("  wires visited inside dst tile (2,95) = {visited_dst_tile}");
        let dst_present = v.contains_key(&dst);
        println!("  M1_IMUX_L0 itself in visited = {dst_present}");

        // Check what wires drive M1_IMUX_L0 — are any of them visited?
        let dst_info = cdb.wire_info(dst);
        let pips_up = dst_info.pips_uphill.get();
        let mut up_in_visited = 0;
        for &pip in pips_up {
            let pid = PipId::new(dst.tile(), pip);
            let upstream = cdb.pip_src_wire(pid);
            if v.contains_key(&upstream) { up_in_visited += 1; }
        }
        println!("  pips uphill of dst = {}, with src wire in visited = {up_in_visited}", pips_up.len());
    }
}

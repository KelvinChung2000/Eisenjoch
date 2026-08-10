//! Router3: Gupta-Sproull raster router with beam-search PIP walking.
//!
//! Uses the Gupta-Sproull anti-aliased line drawing algorithm to establish a
//! tile corridor, then walks the actual PIP graph with a greedy beam search
//! scored by timing (progress toward sink / PIP delay). Long wires naturally
//! win because they cover many tiles for one PIP traversal.
//!
//! Iterates with congestion feedback. Intended for fast wirelength estimation.

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::AStarExit;
use super::astar::{
    astar_search, default_pip_cost, reconstruct_path_to, AStarOptions, PathCostModel,
};
use super::common::{
    apply_route_plan, collect_routable_nets, collect_sink_wires, find_local_const_pip,
    is_global_clock_wire, resolve_source_wire, source_wire_const_value, unroute_net,
    validate_all_routed, RoutePlan, SinkRoute,
};
use super::maze::{astar_route, astar_route_explore, astar_route_multihop, astar_route_with_trace};
use super::RouterError;

/// True iff every valid sink of `net` is bound to the net (directly or via a
/// node-equivalent peer). Used to identify nets the primary pass left
/// partially unrouted so the cleanup A* can finish them.
fn net_fully_routed(ctx: &Context, net_id: NetId) -> bool {
    let net = ctx.net(net_id);
    let chipdb = ctx.chipdb();
    for user in net.users().iter() {
        if !user.is_valid() {
            continue;
        }
        let Some(bel) = ctx.cell(user.cell).bel() else {
            continue;
        };
        let Some(sink_wv) = bel.pin_wire(user.port) else {
            continue;
        };
        let sink_wire = sink_wv.id();
        if net.wires().contains_key(&sink_wire) {
            continue;
        }
        let mut peer_in_tree = false;
        chipdb.node_wires_cb(sink_wire, |peer| {
            if !peer_in_tree && net.wires().contains_key(&peer) {
                peer_in_tree = true;
            }
        });
        if !peer_in_tree {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Scoped rip-up negotiation
// ---------------------------------------------------------------------------
//
// When A* cleanup fails for a net whose routing tree is still empty, the
// failure mode is usually a blocked corridor: no legal path exists because
// every candidate path passes through wires claimed by other nets, and the
// binding-aware cost model correctly refuses to plan through them. Generic
// PathFinder-style negotiation does not help here — `wire_cong` is zero
// (nobody is double-using a wire), so the history-cost mechanism has no
// signal to amplify.
//
// Scoped rip-up resolves a stuck net by:
//   1. Computing `min_cost` = total pip-cost of the theoretical shortest
//      corridor (ignoring bindings). Used only as a diagnostic baseline.
//   2. Trying a binding-aware A* with extended visit budget — finds a free
//      detour through unbound wires if one exists.
//   3. If any detour exists, applying it. We accept slow detours: at this
//      stage of cleanup a routed-but-slow net beats an unrouted one, and
//      timing-driven rejection only makes sense when we have headroom to
//      chase a cleaner solution.
//   4. Only if NO detour exists at all, ripping up ALL foreign nets occupying
//      the min-cost corridor and re-routing.
//
// On any cascade failure during re-route the entire change rolls back so the
// graph state is preserved exactly as we found it.

/// Total rip-up attempts per `run()` invocation. Prevents pathological cases
/// (e.g. a mutual-blocker loop) from consuming unbounded time.
const MAX_RIPUP_ATTEMPTS_PER_RUN: u32 = 8;

/// Visit budget for the exploratory (bindings-ignored) Dijkstra used to
/// identify blockers. With a bbox-restricted search the heap stays bounded
/// by the corridor area (not chip area), so even pure h=0 Dijkstra
/// terminates in a few million pops on the worst sv3 cases.
const EXPLORE_VISIT_LIMIT: usize = 10_000_000;

/// Lower bound on the explore A*'s bounding-box margin. The actual margin
/// is `max(this, manhattan_span * 3)` so a small net still has a wide
/// corridor for plausible detours and a long net gets headroom
/// proportional to its span. Distinct from the per-net routing bbox
/// margin (small, ~8) because exploration may detour through
/// neighbouring tiles to find ANY path.
const EXPLORE_BBOX_MARGIN_MIN: i32 = 60;
const EXPLORE_BBOX_SPAN_MULT: i32 = 3;

/// Visit budget for the binding-aware detour A* and post-rip-up reroute
/// of the net we're trying to recover. Weighted-A* heuristic prunes much
/// harder than Dijkstra, so this can be smaller than the explore budget
/// while still covering the worst observed cases. Keep above ~3M because
/// some sv3 SLICEM-D paths need a wide visit shell before the heuristic
/// can pull tight.
const SCOPED_RIPUP_RETRY_LIMIT: usize = 5_000_000;

/// Visit budget for each per-sink re-route after rip-up. The blockers that
/// were already routed must each find a new home through tiles where our
/// net now occupies the path they previously used, so this needs to be on
/// par with the post-rip-up reroute budget for our own net.
const RIPUP_REROUTE_LIMIT: usize = 5_000_000;

/// Snapshot of a net's routed state, sufficient to restore it if rip-up+retry
/// fails. Captures the wire/pip pairs the net owned via `apply_route_plan`;
/// node-equivalent wires are re-derived from the chipdb at restore time.
struct NetSnapshot {
    net: NetId,
    wires: Vec<(WireId, Option<PipId>)>,
}

/// Capture a net's current routing so it can be restored by `restore_net`.
fn snapshot_net(ctx: &Context, net: NetId) -> NetSnapshot {
    let wires: Vec<(WireId, Option<PipId>)> = ctx
        .net(net)
        .wires()
        .iter()
        .map(|(&w, pm)| (w, pm.pip))
        .collect();
    NetSnapshot { net, wires }
}

/// Fully unroute a net: unbind every wire (including node-equivalents) and
/// every PIP it owns, and clear its wire map. Differs from
/// `super::common::unroute_net` in that it also walks `node_wires_cb` to
/// release node-equivalent bindings that `try_bind_wire_node` installed,
/// ensuring the corridor is truly free for the net we're trying to place.
fn ripup_net_with_equivs(ctx: &mut Context, net: NetId) {
    let entries: Vec<(WireId, Option<PipId>)> = ctx
        .net(net)
        .wires()
        .iter()
        .map(|(&w, pm)| (w, pm.pip))
        .collect();
    for (wire, pip) in &entries {
        let mut equivs: Vec<WireId> = Vec::new();
        ctx.chipdb().node_wires_cb(*wire, |nw| equivs.push(nw));
        ctx.unbind_wire(*wire);
        for nw in equivs {
            ctx.unbind_wire(nw);
        }
        if let Some(p) = pip {
            ctx.unbind_pip(*p);
        }
    }
    ctx.design.net_edit(net).clear_wires();
}

/// Re-apply a snapshot after a failed rip-up attempt. Returns false if any
/// wire re-bind fails (which would indicate external state corruption — the
/// caller should treat this as a hard error).
fn restore_net(ctx: &mut Context, snap: &NetSnapshot) -> bool {
    for (wire, pip) in &snap.wires {
        if ctx
            .try_bind_wire_node(*wire, snap.net, crate::common::PlaceStrength::Strong)
            .is_err()
        {
            return false;
        }
        if let Some(p) = pip {
            ctx.bind_pip(*p, snap.net, crate::common::PlaceStrength::Strong);
        }
        ctx.design
            .net_edit(snap.net)
            .add_wire(*wire, *pip, crate::common::PlaceStrength::Strong);
    }
    true
}

/// Re-route an already-unbound net from scratch using the binding-aware
/// MazeCostModel. Returns true on success. On any failure (unroutable sink,
/// apply conflict), the caller is responsible for rolling back: this
/// function may partially apply before failing, so callers unroute the net
/// before restoring snapshots.
fn reroute_from_scratch(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    lookahead: &super::lookahead::Lookahead,
    visit_limit: usize,
    bbox: Option<&crate::metrics::BoundingBox>,
) -> bool {
    let source_wire = match resolve_source_wire(ctx, net) {
        Ok(Some(w)) => w,
        _ => return false,
    };
    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return true; // no users to route — treat as trivially routed
    }

    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    ctx.chipdb().node_wires_cb(source_wire, |nw| {
        tree_wires.insert(nw);
    });

    let mut sink_routes = Vec::new();
    for &sink_wire in &sink_wires {
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }
        match astar_route(
            ctx,
            net,
            &tree_wires,
            sink_wire,
            wire_penalty,
            bbox,
            50,
            Some(lookahead),
            Some(visit_limit),
        ) {
            Some(pips) => {
                for &pip in &pips {
                    let dw = ctx.chipdb().pip_dst_wire(pip);
                    tree_wires.insert(dw);
                    ctx.chipdb().node_wires_cb(dw, |nw| {
                        tree_wires.insert(nw);
                    });
                }
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                let net_name = ctx.name_of(ctx.net(net).name_id()).to_owned();
                let (sxt, syt) = ctx.chipdb().tile_xy(sink_wire.tile());
                let (srx, sry) = ctx.chipdb().tile_xy(source_wire.tile());
                let bbox_str = bbox
                    .map(|b| format!("[{},{}]x[{},{}]", b.x0, b.x1, b.y0, b.y1))
                    .unwrap_or_else(|| "none".to_string());
                eprintln!(
                    "    reroute_from_scratch FAIL: net='{}' src=({},{}) sink=({},{}) bbox={}",
                    net_name, srx, sry, sxt, syt, bbox_str,
                );
                return false;
            }
        }
    }

    let plan = RoutePlan {
        net,
        source_wire,
        sink_routes,
    };
    apply_route_plan(ctx, &plan).is_ok()
}

/// Attempt to route `net` by temporarily ripping up foreign nets that occupy
/// its theoretical shortest corridor. Returns true if the net is now routed.
///
/// Precondition: `net`'s routing tree is empty (normal A* cleanup has already
/// failed). On any failure, the full graph state is rolled back and this
/// function returns false.
fn scoped_ripup_route(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    lookahead: &super::lookahead::Lookahead,
    budget: &mut u32,
    neg_state: &mut super::common::NegotiationState,
) -> bool {
    if *budget == 0 {
        return false;
    }

    let source_wire = match resolve_source_wire(ctx, net) {
        Ok(Some(w)) => w,
        _ => return false,
    };
    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return false;
    }

    // Build source wire set (source + node equivalents).
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(source_wire);
    ctx.chipdb().node_wires_cb(source_wire, |nw| {
        src_set.insert(nw);
    });

    // Step 1: theoretical-min path (ignore bindings). Establishes the timing
    // baseline (`min_cost`) and the full set of foreign nets occupying it.
    // Bounded to the net's BEL bounding box plus a margin scaled by the
    // net's manhattan span — without this, pure Dijkstra walks the entire
    // chip and the heap blows up before the sink can be reached. The margin
    // must comfortably exceed any plausible detour, so it scales with the
    // direct path length: a 1-tile net needs a tight corridor; a 100-tile
    // net may need to detour 100 tiles in either dimension.
    let first_sink = sink_wires[0];
    let bel_bbox_for_span = crate::metrics::compute_bbox(ctx, net, 0);
    let span = (bel_bbox_for_span.x1 - bel_bbox_for_span.x0)
        .max(bel_bbox_for_span.y1 - bel_bbox_for_span.y0)
        .max(0);
    let explore_margin = EXPLORE_BBOX_MARGIN_MIN.max(span * EXPLORE_BBOX_SPAN_MULT);
    let explore_bbox = crate::metrics::compute_bbox(ctx, net, explore_margin);
    let explore_path = match astar_route_explore(
        ctx,
        net,
        &src_set,
        first_sink,
        wire_penalty,
        Some(&explore_bbox),
        Some(lookahead),
        Some(EXPLORE_VISIT_LIMIT),
    ) {
        Some(p) => p,
        None => return false,
    };
    let min_cost: i64 = explore_path
        .iter()
        .map(|&p| lookahead.pip_cost(ctx.chipdb(), p) as i64)
        .sum();

    let mut blockers: FxHashSet<NetId> = FxHashSet::default();
    for &pip in &explore_path {
        let dst = ctx.chipdb().pip_dst_wire(pip);
        // Binding is at node granularity (try_bind_wire_node), so a wire
        // whose dst is unbound can still be unroutable if any node-equivalent
        // wire belongs to another net. Walk the node to catch all blockers.
        if let Some((owner, _)) = ctx.wire_binding(dst) {
            if owner != net {
                blockers.insert(owner);
            }
        }
        ctx.chipdb().node_wires_cb(dst, |nw| {
            if let Some((owner, _)) = ctx.wire_binding(nw) {
                if owner != net {
                    blockers.insert(owner);
                }
            }
        });
    }
    if blockers.is_empty() {
        // Corridor already clear — normal A* should have succeeded but hit
        // some other limit; scoped rip-up can't help.
        return false;
    }

    // Step 2: try a binding-aware A* on the same first_sink with extended
    // visit budget. If it returns a path, every wire is unbound (binding
    // hard-block ensures that), so this is a true zero-blocker detour.
    let detour_path = astar_route(
        ctx,
        net,
        &src_set,
        first_sink,
        wire_penalty,
        None,
        0,
        Some(lookahead),
        Some(SCOPED_RIPUP_RETRY_LIMIT),
    );

    // Step 3: if any binding-aware detour exists, accept it regardless of cost.
    // A slow routed net beats an unrouted one — timing-driven rejection only
    // makes sense when we have headroom to chase a cleaner solution, and at
    // this point in cleanup we don't.
    if let Some(detour) = &detour_path {
        let detour_cost: i64 = detour
            .iter()
            .map(|&p| lookahead.pip_cost(ctx.chipdb(), p) as i64)
            .sum();
        // No rip-up here, so no budget consumed — just an alternate reroute.
        let ok = reroute_from_scratch(
            ctx,
            net,
            wire_penalty,
            lookahead,
            SCOPED_RIPUP_RETRY_LIMIT,
            Some(&explore_bbox),
        );
        eprintln!(
            "  scoped_ripup: detour cost {} (min {}) accepted (applied={})",
            detour_cost, min_cost, ok,
        );
        if ok {
            return true;
        }
        // reroute_from_scratch (which routes ALL sinks, not just first_sink)
        // couldn't apply it; fall through to rip-up.
    } else {
        eprintln!(
            "  scoped_ripup: no detour at all (min_cost={} blockers={}), ripping up",
            min_cost,
            blockers.len(),
        );
    }

    // Snapshot every blocker before unrouting. Order matters only for
    // restoration symmetry; a Vec<NetSnapshot> preserves insertion order.
    let blocker_list: Vec<NetId> = blockers.iter().copied().collect();
    let snapshots: Vec<NetSnapshot> = blocker_list.iter().map(|&b| snapshot_net(ctx, b)).collect();

    for &b in &blocker_list {
        ripup_net_with_equivs(ctx, b);
        neg_state.bump_rip_up(b);
    }

    // Retry our net with the corridor now free. Use the same
    // binding-aware route path as the normal cleanup loop.
    if !reroute_from_scratch(
        ctx,
        net,
        wire_penalty,
        lookahead,
        SCOPED_RIPUP_RETRY_LIMIT,
        Some(&explore_bbox),
    ) {
        // Our net still can't route — undo and restore blockers.
        // Use equiv-aware ripup because apply_route_plan binds node
        // equivalents via try_bind_wire_node; plain unroute_net would
        // leave them stale and cause the subsequent restore_net to fail.
        ripup_net_with_equivs(ctx, net);
        for snap in &snapshots {
            if !restore_net(ctx, snap) {
                eprintln!(
                    "  scoped_ripup: CRITICAL restore failure for blocker net (idx={:?})",
                    snap.net
                );
            }
        }
        return false;
    }

    // Re-route each blocker. On any failure, fully roll back.
    for (i, &b) in blocker_list.iter().enumerate() {
        // Blocker reroute uses the blocker's own bbox, not ours — they may
        // need to find a path through entirely different tiles now that our
        // net occupies their previous corridor. Margin scales with span the
        // same way explore does, so a long blocker still has room to detour.
        let blocker_span_bbox = crate::metrics::compute_bbox(ctx, b, 0);
        let blocker_span = (blocker_span_bbox.x1 - blocker_span_bbox.x0)
            .max(blocker_span_bbox.y1 - blocker_span_bbox.y0)
            .max(0);
        let blocker_margin = EXPLORE_BBOX_MARGIN_MIN.max(blocker_span * EXPLORE_BBOX_SPAN_MULT);
        let blocker_bbox = crate::metrics::compute_bbox(ctx, b, blocker_margin);
        if !reroute_from_scratch(
            ctx,
            b,
            wire_penalty,
            lookahead,
            RIPUP_REROUTE_LIMIT,
            Some(&blocker_bbox),
        ) {
            // Undo blocker partial re-route (reroute_from_scratch is
            // atomic-on-failure so no bindings should exist, but defend
            // against node-equiv leaks anyway), undo our net, then undo
            // any blockers we already successfully re-routed earlier in
            // the loop, then restore every original snapshot.
            ripup_net_with_equivs(ctx, b);
            ripup_net_with_equivs(ctx, net);
            for snap in &snapshots[..i] {
                ripup_net_with_equivs(ctx, snap.net);
            }
            for snap in &snapshots {
                if !restore_net(ctx, snap) {
                    eprintln!(
                        "  scoped_ripup: CRITICAL restore failure for blocker net (idx={:?})",
                        snap.net
                    );
                }
            }
            return false;
        }
    }

    *budget -= 1;
    eprintln!(
        "  scoped_ripup: routed net via {} blocker rip-ups (budget={} left)",
        blocker_list.len(),
        *budget
    );
    true
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RasterRouterCfg {
    pub max_iterations: usize,
    pub initial_cong_weight: f32,
    pub cong_growth: f32,
    pub filter_radius: f32,
    pub beam_width: usize,
    pub max_beam_steps: usize,
    /// Margin (in tiles) around each net's bounding box for A* pruning.
    /// 0 disables bbox pruning (search the whole chip).
    pub bbox_margin: i32,
    /// If true, use pure greedy line-walk instead of beam search.
    pub use_greedy: bool,
    pub verbose: bool,
}

impl Default for RasterRouterCfg {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            initial_cong_weight: 0.05,
            cong_growth: 1.1,
            filter_radius: 1.0,
            beam_width: 128,
            max_beam_steps: 1000,
            bbox_margin: 8,
            use_greedy: false,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CongestionMap
// ---------------------------------------------------------------------------

struct CongestionMap {
    usage: Vec<f32>,
    history: Vec<f32>,
    #[allow(dead_code)]
    width: i32,
    #[allow(dead_code)]
    height: i32,
}

impl CongestionMap {
    fn new(width: i32, height: i32) -> Self {
        let n = (width * height) as usize;
        Self {
            usage: vec![0.0; n],
            history: vec![0.0; n],
            width,
            height,
        }
    }

    fn reset_usage(&mut self) {
        self.usage.fill(0.0);
    }

    fn add_usage(&mut self, tile: i32, amount: f32) {
        self.usage[tile as usize] += amount;
    }

    fn update_history(&mut self) {
        for i in 0..self.usage.len() {
            let excess = self.usage[i] - 1.0;
            if excess > 0.0 {
                // Blend: keep 50% of old history + add new excess.
                self.history[i] = self.history[i] * 0.5 + excess;
            } else {
                // Decay history on non-congested tiles.
                self.history[i] *= 0.5;
            }
        }
    }

    fn congestion_at(&self, tile: i32) -> f32 {
        // History only - usage is reset each pass.
        self.history[tile as usize]
    }

    fn max_usage(&self) -> f32 {
        self.usage.iter().cloned().fold(0.0f32, f32::max)
    }

    fn num_congested_tiles(&self) -> usize {
        self.usage.iter().filter(|&&u| u > 1.0).count()
    }
}

// ---------------------------------------------------------------------------
// Gupta-Sproull rasterizer
// ---------------------------------------------------------------------------

/// Build a set of tiles that form the routing corridor via Gupta-Sproull
/// anti-aliased line drawing. Returns a HashSet for O(1) corridor membership
/// testing, plus the ordered path for direction guidance.
fn raster_corridor(
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    cong: &CongestionMap,
    cong_weight: f32,
    filter_radius: f32,
    chip_width: i32,
    chip_height: i32,
) -> (Vec<(i32, i32)>, FxHashSet<i32>) {
    let mut path = Vec::new();
    let mut corridor = FxHashSet::default();

    if x0 == x1 && y0 == y1 {
        path.push((x0, y0));
        corridor.insert(y0 * chip_width + x0);
        return (path, corridor);
    }

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let steep = dy > dx;

    let (mut px, mut py, end_x, _end_y, adx, ady) = if steep {
        (y0, x0, y1, x1, dy, dx)
    } else {
        (x0, y0, x1, y1, dx, dy)
    };

    let step_x: i32 = if end_x > px { 1 } else { -1 };
    let step_y: i32 = if _end_y > py { 1 } else { -1 };

    let mut d = 2 * ady - adx;
    let mut two_v_dx: i32;
    let inv_denom = 1.0 / (2.0 * ((adx * adx + ady * ady) as f32).sqrt());
    let two_dx_inv_denom = 2.0 * adx as f32 * inv_denom;

    let to_tile = |major: i32, minor: i32| -> (i32, i32) {
        if steep {
            (minor, major)
        } else {
            (major, minor)
        }
    };

    let tile_idx = |x: i32, y: i32| -> i32 { y * chip_width + x };

    let eval = |major: i32, minor: i32, dist: f32| -> f32 {
        let (cx, cy) = to_tile(major, minor);
        if cx < 0 || cx >= chip_width || cy < 0 || cy >= chip_height {
            return f32::NEG_INFINITY;
        }
        let coverage = (1.0 - dist / filter_radius).max(0.0);
        coverage - cong_weight * cong.congestion_at(tile_idx(cx, cy))
    };

    // Add all three candidate pixels to corridor (wider corridor for beam search).
    let mut add_pixel = |major: i32, minor: i32| {
        let (cx, cy) = to_tile(major, minor);
        if cx >= 0 && cx < chip_width && cy >= 0 && cy < chip_height {
            corridor.insert(tile_idx(cx, cy));
        }
    };

    // First pixel.
    let (fx, fy) = to_tile(px, py);
    path.push((fx, fy));
    add_pixel(px, py);
    add_pixel(px, py + 1);
    add_pixel(px, py - 1);

    for _ in 0..adx {
        px += step_x;
        if d >= 0 {
            py += step_y;
            d -= 2 * adx;
            two_v_dx = d + adx;
        } else {
            two_v_dx = d + adx;
        }
        d += 2 * ady;

        let dist_main = (two_v_dx as f32 * inv_denom).abs();
        let dist_plus = two_dx_inv_denom - dist_main;
        let dist_minus = two_dx_inv_denom + dist_main;

        let p_main = eval(px, py, dist_main);
        let p_plus = eval(px, py + step_y, dist_plus);
        let p_minus = eval(px, py - step_y, dist_minus);

        let chosen_minor = if p_main >= p_plus && p_main >= p_minus {
            py
        } else if p_plus > p_minus {
            py + step_y
        } else {
            py - step_y
        };

        let (cx, cy) = to_tile(px, chosen_minor);
        if let Some(&last) = path.last() {
            if last != (cx, cy) {
                path.push((cx, cy));
            }
        } else {
            path.push((cx, cy));
        }

        // Add all three candidates to the corridor for beam width.
        add_pixel(px, py);
        add_pixel(px, py + step_y);
        add_pixel(px, py - step_y);

        py = chosen_minor;
    }

    // Ensure endpoint.
    let (ex, ey) = to_tile(end_x, _end_y);
    if path.last() != Some(&(ex, ey)) {
        path.push((ex, ey));
    }
    corridor.insert(tile_idx(ex, ey));

    (path, corridor)
}

// ---------------------------------------------------------------------------
// Per-sink A* (bbox-pruned, lookahead heuristic)
// ---------------------------------------------------------------------------

/// Weighted A* cost model: uniform pip cost (g = number of pips) and
/// manhattan heuristic. h is admissible (≤ true pip count); scaling by
/// `H_WEIGHT > 1` trades optimality for search speed (returned cost ≤
/// `H_WEIGHT × optimal`). `H_WEIGHT = 2` is a typical balance.
const H_WEIGHT: DelayT = 2;

struct RasterCostModel<'a> {
    net: NetId,
    bbox: Option<&'a BoundingBox>,
    cong: &'a CongestionMap,
    cong_weight: f32,
}

impl<'a> PathCostModel for RasterCostModel<'a> {
    fn pip_cost(&self, _ctx: &Context, _pip: PipId) -> DelayT {
        1
    }

    fn wire_penalty(&self, ctx: &Context, wire: WireId) -> DelayT {
        if let Some((owner, _)) = ctx.wire_binding(wire) {
            if owner != self.net {
                return DelayT::MAX / 4;
            }
        }
        (self.cong_weight * self.cong.congestion_at(wire.tile())) as DelayT
    }

    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        let chipdb = ctx.chipdb();
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        let (dx, dy) = chipdb.tile_xy(dst.tile());
        let m = ((wx - dx).abs() + (wy - dy).abs()) as DelayT;
        m.saturating_mul(H_WEIGHT)
    }

    fn bboxes(&self) -> &[BoundingBox] {
        match self.bbox {
            Some(bb) => std::slice::from_ref(bb),
            None => &[],
        }
    }
}

fn route_sink_astar(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    bbox: Option<&BoundingBox>,
    cong: &CongestionMap,
    cong_weight: f32,
) -> Option<Vec<PipId>> {
    let model = RasterCostModel {
        net,
        bbox,
        cong,
        cong_weight,
    };
    let opts = AStarOptions {
        visit_limit: None,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: true,
    };
    astar_search(ctx, &model, src_wires, dst_wire, &opts).path
}

// ---------------------------------------------------------------------------
// Segment decomposition + bounded local search
// ---------------------------------------------------------------------------

/// Decompose a rasterized tile path into segments at direction changes.
/// Each segment is a run of consecutive same-direction tiles.
/// Returns vec of (start_idx, end_idx) into the path array.
fn decompose_segments(path: &[(i32, i32)]) -> Vec<(usize, usize)> {
    if path.len() < 2 {
        return vec![(0, path.len().saturating_sub(1))];
    }

    let mut segments = Vec::new();
    let mut seg_start = 0;
    let mut prev_dx = path[1].0 - path[0].0;
    let mut prev_dy = path[1].1 - path[0].1;

    for i in 2..path.len() {
        let dx = path[i].0 - path[i - 1].0;
        let dy = path[i].1 - path[i - 1].1;
        if (dx.signum(), dy.signum()) != (prev_dx.signum(), prev_dy.signum()) {
            segments.push((seg_start, i - 1));
            seg_start = i - 1; // overlap: end of prev = start of next
            prev_dx = dx;
            prev_dy = dy;
        }
    }
    segments.push((seg_start, path.len() - 1));
    segments
}

/// Cost model for `segment_astar`. Corridor pruning is expressed as a union
/// of small per-path-point bboxes (see `bboxes`); congestion enters the cost
/// as a per-wire penalty proportional to the arrival tile's usage.
struct SegmentCostModel<'a> {
    bboxes: &'a [BoundingBox],
    cong: &'a CongestionMap,
    cong_weight: f32,
    target_x: i32,
    target_y: i32,
}

impl<'a> PathCostModel for SegmentCostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn wire_penalty(&self, _ctx: &Context, wire: WireId) -> DelayT {
        (self.cong_weight * self.cong.congestion_at(wire.tile())) as DelayT
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, _dst: WireId) -> DelayT {
        let (wx, wy) = ctx.chipdb().tile_xy(wire.tile());
        (wx - self.target_x).abs() + (wy - self.target_y).abs()
    }
    fn bboxes(&self) -> &[BoundingBox] {
        self.bboxes
    }
}

/// Build a union of small bboxes covering a corridor: one
/// `(2*margin+1) × (2*margin+1)` box around each tile on the path, clamped
/// to the grid. Used by `segment_astar` to express corridor-shaped tile
/// pruning via `PathCostModel::bboxes`.
fn corridor_bboxes(
    chipdb: &crate::chipdb::ChipDb,
    path: &[(i32, i32)],
    start: usize,
    end: usize,
    margin: i32,
) -> Vec<BoundingBox> {
    let w = chipdb.width();
    let h = chipdb.height();
    let mut out: Vec<BoundingBox> = Vec::with_capacity(end - start + 1);
    for i in start..=end {
        let (x, y) = path[i];
        let x0 = (x - margin).max(0);
        let y0 = (y - margin).max(0);
        let x1 = (x + margin).min(w - 1);
        let y1 = (y + margin).min(h - 1);
        out.push(BoundingBox { x0, y0, x1, y1 });
    }
    out
}

/// Pick a representative wire in `target_tile`: the wire with the most
/// `pips_uphill`, so it's reachable from many directions. Used as the
/// destination anchor when the actual goal is "any wire in `target_tile`".
fn representative_wire_in_tile(chipdb: &crate::chipdb::ChipDb, target_tile: i32) -> Option<WireId> {
    let tt = chipdb.tile_type(target_tile);
    let n_wires = tt.wires.get().len();
    let mut best: Option<(WireId, usize)> = None;
    for wi in 0..n_wires {
        let w = WireId::new(target_tile, wi as i32);
        let up = chipdb.wire_info(w).pips_uphill.get().len();
        match best {
            Some((_, bu)) if up <= bu => {}
            _ => best = Some((w, up)),
        }
    }
    best.map(|(w, _)| w)
}

/// Confined A* search within a segment's tile region.
///
/// Thin wrapper over [`astar_search`] that expresses the corridor as a union
/// of small bboxes (`PathCostModel::bboxes`). The destination is the final
/// sink when this is the last segment; otherwise a representative wire in
/// `target_tile` (the one with the most `pips_uphill`), which the search
/// will reach after any wire in `target_tile` enters the frontier via
/// intra-tile switch-matrix PIPs.
///
/// Returns `(pips, end_wire)` on success. `end_wire` is the `dst` wire
/// (reached by the path); callers seed the next segment from it plus its
/// node peers.
fn segment_astar(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    target_tile: i32,
    final_dst: Option<WireId>,
    bboxes: &[BoundingBox],
    cong: &CongestionMap,
    cong_weight: f32,
    max_expansions: usize,
) -> Option<(Vec<PipId>, WireId)> {
    let chipdb = ctx.chipdb();
    let (target_x, target_y) = chipdb.tile_xy(target_tile);

    // Anchor for the heuristic. For the final segment this is the real
    // sink; for intermediate segments we pick any well-connected wire in
    // `target_tile` so the search has a pole to aim at.
    let anchor = match final_dst {
        Some(w) => w,
        None => representative_wire_in_tile(chipdb, target_tile)?,
    };

    if src_wires.contains(&anchor) {
        return Some((Vec::new(), anchor));
    }

    let model = SegmentCostModel {
        bboxes,
        cong,
        cong_weight,
        target_x,
        target_y,
    };
    let opts = AStarOptions {
        visit_limit: Some(max_expansions),
        // Intermediate segments need the search to keep expanding even
        // after the anchor wire pops so we can harvest any other wire in
        // target_tile reached along the way. Final segments stop on the
        // real sink (exhaustive=false) for speed.
        exhaustive: final_dst.is_none(),
        retain_trace: final_dst.is_none(),
        stop_on_first_touch: false,
    };
    let result = astar_search(ctx, &model, src_wires, anchor, &opts);

    match final_dst {
        Some(dst) => result.path.map(|pips| (pips, dst)),
        None => {
            // Scan the visited map for the lowest-cost wire that landed
            // in `target_tile`; reconstruct a path to it. This
            // generalises the old "accept any wire in target_tile" hit
            // condition without adding multi-goal support to the kernel.
            let mut best: Option<(WireId, DelayT)> = None;
            for (&w, &(cost, pen, _, _)) in result.trace.visited.iter() {
                if w.tile() != target_tile {
                    continue;
                }
                let score = cost + pen;
                if best.map_or(true, |(_, b)| score < b) {
                    best = Some((w, score));
                }
            }
            let (end_wire, _) = best?;
            let pips = reconstruct_path_to(chipdb, &result.trace.visited, end_wire)?;
            Some((pips, end_wire))
        }
    }
}

/// Unrestricted A* for same-tile (or short-distance) routing: no bbox
/// pruning, manhattan heuristic, 20k-pop budget.
///
/// Wrapper over [`astar_search`]; kept separate from the full router
/// `astar_route` because this path never uses the lookahead and never sees
/// wire-penalty state.
fn local_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
) -> Option<Vec<PipId>> {
    struct LocalModel;
    impl PathCostModel for LocalModel {
        fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
            default_pip_cost(ctx, pip)
        }
        fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
            let chipdb = ctx.chipdb();
            let (wx, wy) = chipdb.tile_xy(wire.tile());
            let (dx, dy) = chipdb.tile_xy(dst.tile());
            (wx - dx).abs() + (wy - dy).abs()
        }
    }
    let opts = AStarOptions {
        visit_limit: Some(20_000),
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &LocalModel, src_wires, dst_wire, &opts).path
}

/// Route using segment decomposition: break the rasterized line into
/// direction-change segments, do confined A* within each segment's tiles.
fn segment_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    path: &[(i32, i32)],
    cong: &CongestionMap,
    cong_weight: f32,
) -> Option<Vec<PipId>> {
    let chipdb = ctx.chipdb();

    let mut dst_nodes: FxHashSet<WireId> = FxHashSet::default();
    dst_nodes.insert(dst_wire);
    chipdb.node_wires_cb(dst_wire, |nw| {
        dst_nodes.insert(nw);
    });

    for src in src_wires {
        if dst_nodes.contains(src) {
            return Some(Vec::new());
        }
    }

    // Same-tile: use local route (no tile confinement, generous budget).
    if path.len() < 2 {
        return local_route(ctx, src_wires, dst_wire);
    }

    let segments = decompose_segments(path);
    let mut all_pips: Vec<PipId> = Vec::new();
    let mut current_wires: FxHashSet<WireId> = src_wires.clone();

    for (seg_idx, &(start, end)) in segments.iter().enumerate() {
        let is_last = seg_idx == segments.len() - 1;
        let seg_len = end - start + 1;

        let margin: i32 = 1;

        // Corridor pruning: one small bbox per path tile, union = corridor.
        let bboxes = corridor_bboxes(chipdb, path, start, end, margin);

        let (tx, ty) = path[end];
        let target_tile = chipdb.tile_by_xy(tx, ty);
        // For the final segment the real sink wire is the goal. For
        // intermediate segments we only need to *reach* target_tile; the
        // wrapper picks a representative wire with the most pips_uphill.
        let final_dst = if is_last { Some(dst_wire) } else { None };

        // Confined A* within this segment's tiles.
        let budget = (seg_len * 3000).max(5000) as usize;
        match segment_astar(
            ctx,
            &current_wires,
            target_tile,
            final_dst,
            &bboxes,
            cong,
            cong_weight,
            budget,
        ) {
            Some((pips, end_wire)) => {
                all_pips.extend_from_slice(&pips);

                // Update current wires for next segment.
                current_wires.clear();
                current_wires.insert(end_wire);
                chipdb.node_wires_cb(end_wire, |nw| {
                    current_wires.insert(nw);
                });
                // Also add PIP destination wires.
                for &pip in &pips {
                    let dw = chipdb.pip_dst_wire(pip);
                    current_wires.insert(dw);
                    chipdb.node_wires_cb(dw, |nw| {
                        current_wires.insert(nw);
                    });
                }
            }
            None => {
                // Diagnostic: which segment failed?
                let (sx, sy) = path[start];
                let (ex, ey) = path[end];
                let seg_len = end - start + 1;
                eprintln!(
                    "  seg_fail: seg {}/{} ({},{})→({},{}) len={} tiles={} is_last={}",
                    seg_idx,
                    segments.len(),
                    sx,
                    sy,
                    ex,
                    ey,
                    seg_len,
                    bboxes.len(),
                    is_last,
                );
                return None;
            }
        }
    }

    Some(all_pips)
}

// ---------------------------------------------------------------------------
// Net routing
// ---------------------------------------------------------------------------

/// Route a net using pure greedy line-walk (no beam search).
fn route_net_greedy(
    ctx: &Context,
    net: NetId,
    cong: &CongestionMap,
    cong_weight: f32,
    cfg: &RasterRouterCfg,
) -> Result<RoutePlan, RouterError> {
    let source_wire = match resolve_source_wire(ctx, net)? {
        Some(w) => w,
        None => {
            return Ok(RoutePlan {
                net,
                source_wire: WireId::INVALID,
                sink_routes: Vec::new(),
            });
        }
    };

    let chipdb = ctx.chipdb();
    let chip_w = chipdb.width();
    let chip_h = chipdb.height();

    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());

    let mut tree_tiles: Vec<(i32, i32)> = vec![(src_x, src_y)];
    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    chipdb.node_wires_cb(source_wire, |nw| {
        tree_wires.insert(nw);
    });

    let mut remaining_sinks: Vec<(WireId, i32, i32)> = sink_wires
        .iter()
        .map(|&w| {
            let (sx, sy) = chipdb.tile_xy(w.tile());
            (w, sx, sy)
        })
        .collect();

    let mut sink_routes = Vec::new();

    while !remaining_sinks.is_empty() {
        let mut best_sink_idx = 0;
        let mut best_tree_tile = tree_tiles[0];
        let mut best_dist = i32::MAX;
        for (si, &(_, sx, sy)) in remaining_sinks.iter().enumerate() {
            for &(tx, ty) in &tree_tiles {
                let d = (sx - tx).abs() + (sy - ty).abs();
                if d < best_dist {
                    best_dist = d;
                    best_sink_idx = si;
                    best_tree_tile = (tx, ty);
                }
            }
        }

        let (sink_wire, sink_x, sink_y) = remaining_sinks.remove(best_sink_idx);

        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }

        // For same-tile routing, use unconstrained local_route.
        // For everything else, use segment-based confined A*.
        let same_tile =
            best_dist == 0 || (sink_x == best_tree_tile.0 && sink_y == best_tree_tile.1);
        let use_local = same_tile;

        let route_result = if use_local {
            local_route(ctx, &tree_wires, sink_wire)
        } else {
            // Rasterize the line path.
            let (path, _corridor) = raster_corridor(
                best_tree_tile.0,
                best_tree_tile.1,
                sink_x,
                sink_y,
                cong,
                cong_weight,
                cfg.filter_radius,
                chip_w,
                chip_h,
            );
            segment_route(ctx, &tree_wires, sink_wire, &path, cong, cong_weight)
        };

        match route_result {
            Some(pips) => {
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    let (tx, ty) = chipdb.tile_xy(dst.tile());
                    tree_tiles.push((tx, ty));
                    tree_wires.insert(dst);
                    chipdb.node_wires_cb(dst, |nw| {
                        tree_wires.insert(nw);
                    });
                }
                tree_wires.insert(sink_wire);
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                let net_name = ctx.net(net).name_id();
                let (sx, sy) = chipdb.tile_xy(source_wire.tile());
                let est_delay = ctx.estimate_delay_for_net(net);
                let constraint = ctx.net(net).clock_constraint();
                let n_users = ctx.net(net).num_users();
                eprintln!(
                    "  FAIL net={} src=({},{}) sink=({},{}) dist={} delay={} clk={} users={} local={}",
                    ctx.name_of(net_name), sx, sy, sink_x, sink_y,
                    best_dist, est_delay, constraint, n_users, use_local,
                );
                return Err(RouterError::NoPath(ctx.name_of(net_name).to_owned()));
            }
        }
    }

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

/// Beam-search raster routing. This path deliberately takes no lookahead: it
/// is a bbox-pruned raster expansion, not an A* with a cost-to-go heuristic.
/// (The lookahead built in `route_raster` is for the A* cleanup path, which
/// does consume it.) The parameter used to be threaded in here and ignored.
fn route_net_raster(
    ctx: &Context,
    net: NetId,
    cong: &CongestionMap,
    cong_weight: f32,
    cfg: &RasterRouterCfg,
) -> Result<RoutePlan, RouterError> {
    let source_wire = match resolve_source_wire(ctx, net)? {
        Some(w) => w,
        None => {
            return Ok(RoutePlan {
                net,
                source_wire: WireId::INVALID,
                sink_routes: Vec::new(),
            });
        }
    };

    let chipdb = ctx.chipdb();

    // Skip constant nets.
    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let bbox = if cfg.bbox_margin > 0 {
        Some(crate::metrics::compute_bbox(ctx, net, cfg.bbox_margin))
    } else {
        None
    };

    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    chipdb.node_wires_cb(source_wire, |nw| {
        tree_wires.insert(nw);
    });

    // Nearest-unrouted-sink Steiner heuristic: route the sink with smallest
    // manhattan to source first; subsequent sinks attach via tree_wires.
    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());
    let mut remaining_sinks: Vec<(WireId, i32, i32)> = sink_wires
        .iter()
        .map(|&w| {
            let (sx, sy) = chipdb.tile_xy(w.tile());
            (w, sx, sy)
        })
        .collect();
    remaining_sinks.sort_by_key(|&(_, sx, sy)| (sx - src_x).abs() + (sy - src_y).abs());

    // Per-net time budget: 2s base + 10ms per sink. Prevents high-fanout nets
    // from monopolizing routing time across rip-up-reroute passes.
    let net_deadline = std::time::Instant::now()
        + std::time::Duration::from_millis(2000 + remaining_sinks.len() as u64 * 10);

    let mut sink_routes = Vec::new();

    for (sink_wire, sink_x, sink_y) in remaining_sinks {
        if std::time::Instant::now() > net_deadline {
            break;
        }
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }

        match route_sink_astar(
            ctx,
            net,
            &tree_wires,
            sink_wire,
            bbox.as_ref(),
            cong,
            cong_weight,
        ) {
            Some(pips) => {
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    tree_wires.insert(dst);
                    chipdb.node_wires_cb(dst, |nw| {
                        tree_wires.insert(nw);
                    });
                }
                tree_wires.insert(sink_wire);

                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                static FAIL_COUNT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 100 {
                    let net_name = ctx.net(net).name_id();
                    let (sxw, syw) = chipdb.tile_xy(source_wire.tile());
                    let dist = (sink_x - sxw).abs() + (sink_y - syw).abs();
                    eprintln!(
                        "  raster_fail #{}: net='{}' src=({},{}) sink=({},{}) dist={} tree={}",
                        n,
                        ctx.name_of(net_name),
                        sxw,
                        syw,
                        sink_x,
                        sink_y,
                        dist,
                        tree_wires.len(),
                    );
                }
                // Preserve partial progress: keep sinks routed so far, skip
                // this sink, and let the A* cleanup pass attempt it later.
                continue;
            }
        }
    }

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

// ---------------------------------------------------------------------------
// Router implementation
// ---------------------------------------------------------------------------

pub struct RasterRouter;

impl super::Router for RasterRouter {
    type Config = RasterRouterCfg;

    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), RouterError> {
        let nets = collect_routable_nets(ctx);
        self.route_nets(ctx, cfg, &nets)
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), RouterError> {
        let chip_w = ctx.chipdb().width();
        let chip_h = ctx.chipdb().height();

        eprintln!(
            "RasterRouter: {} nets on {}x{}, beam_width={}, max_steps={}",
            nets.len(),
            chip_w,
            chip_h,
            cfg.beam_width,
            cfg.max_beam_steps,
        );

        let mut cong = CongestionMap::new(chip_w, chip_h);
        let mut cong_weight = cfg.initial_cong_weight;
        let mut best_routed = 0usize;
        let mut best_wl = 0u64;

        // Pre-reserve BEL-pin wires. Every alive net's driver output wire and
        // every sink (user) input wire is bound to its owning net before any
        // routing begins. This prevents one net's route from claiming a
        // routing node that includes another net's driver output — a common
        // failure mode on dense designs where sibling bus bits ('vert[2]' vs
        // 'vert[3]') had adjacent BEL outputs sharing a node. Beam search
        // already skips PIPs whose dst is owned by a different net, so
        // reserved pins naturally steer routes around them.
        let mut pin_wires: Vec<(crate::chipdb::WireId, NetId)> = Vec::new();
        for &net in nets {
            let n = ctx.net(net);
            if !n.is_alive() || !n.has_driver() {
                continue;
            }
            if let Some(driver) = n.driver_cell_port() {
                if let Some(bel) = ctx.cell(driver.cell).bel() {
                    if let Some(w) = bel.pin_wire(driver.port) {
                        pin_wires.push((w.id(), net));
                    }
                }
            }
            for user in n.users() {
                if !user.is_valid() {
                    continue;
                }
                if let Some(ubel) = ctx.cell(user.cell).bel() {
                    if let Some(uw) = ubel.pin_wire(user.port) {
                        pin_wires.push((uw.id(), net));
                    }
                }
            }
        }
        let mut reservations_applied = 0usize;
        let mut reservation_conflicts = 0usize;
        // Temporarily disabled for diagnostic: pre-reservation binds whole
        // routing nodes and may block local routes in dense tiles.
        if std::env::var("NPNR_SKIP_PIN_RESERVE").ok().as_deref() != Some("1") {
            for (wire, net) in pin_wires {
                match ctx.try_bind_wire_node(wire, net, crate::common::PlaceStrength::Strong) {
                    Ok(()) => reservations_applied += 1,
                    Err(_) => reservation_conflicts += 1,
                }
            }
        }
        eprintln!(
            "RasterRouter: pre-reserved {} BEL-pin wires ({} conflicts)",
            reservations_applied, reservation_conflicts
        );

        // Wire-level negotiation state (PathFinder-style).
        let neg_cfg = super::common::NegotiationCfg::default();
        let mut neg_state = super::common::NegotiationState::new(neg_cfg);

        // Pre-build lookahead for A* cleanup (reused across iterations).
        let lookahead = super::lookahead::Lookahead::build(ctx);

        // Scoped rip-up budget, shared across all passes within this run.
        let mut ripup_budget: u32 = MAX_RIPUP_ATTEMPTS_PER_RUN;

        // Iteration 0: route ALL nets (beam search).
        // Iterations 1+: selective rip-up of congested nets only (PathFinder-style).
        for iter in 0..cfg.max_iterations {
            let nets_to_route: Vec<NetId>;

            if iter == 0 {
                // First pass: route everything.
                nets_to_route = nets.to_vec();
                cong.reset_usage();
            } else {
                // Subsequent passes: rip up congested nets AND retry any net
                // whose routing tree is still empty (iter-0 bind-conflict or
                // beam-search failure). Empty-tree nets contribute no wire
                // usage so find_congested_nets can't surface them.
                let congested = neg_state.find_congested_nets(&ctx.design);
                // Sticky-rip-up gate: any net we've attempted to re-route in a
                // prior pass becomes "sticky" — its current routing tree is
                // preserved and the cleanup A* at the end of this pass tries
                // to extend it to the missing sinks. Conflict partners must
                // detour around the sticky net. Without this, the rip-up loop
                // tears down a partial tree (e.g. tree=853 on an IO-to-fabric
                // net), the next pass routes from scratch and fails on both
                // sinks, and the partial progress is permanently lost.
                let mut sticky_skipped = 0usize;
                let mut bumped_in_pass = 0usize;
                let mut to_retry: Vec<NetId> = Vec::with_capacity(congested.len());
                for &net in &congested {
                    if neg_state.is_sticky(net) {
                        sticky_skipped += 1;
                        continue;
                    }
                    neg_state.remove_net_usage(&ctx.design, net);
                    if ctx.net(net).wires().len() > 0 {
                        unroute_net(ctx, net);
                    }
                    neg_state.bump_rip_up(net);
                    bumped_in_pass += 1;
                    to_retry.push(net);
                }
                // Collect retry candidates without mutating ctx. Nothing is
                // unrouted at this gate: partial trees are deliberately
                // preserved (see below) and empty ones have nothing to release,
                // so the former `to_unroute` list was always empty and its
                // `unroute_net` loop never ran.
                for idx in ctx.design.iter_net_indices() {
                    let n = ctx.net(idx);
                    if !n.is_alive() || !n.has_driver() || n.num_users() == 0 {
                        continue;
                    }
                    // `n` borrows ctx immutably and is not used past this point,
                    // so NLL already releases it before `net_fully_routed`.
                    // (An explicit `drop(n)` here was a no-op: `Net<'_>` is Copy.)
                    let empty = n.wires().is_empty();
                    if empty || !net_fully_routed(ctx, idx) {
                        if !empty {
                            // Never rip up a partial tree at the gate. A
                            // partial is concrete progress (e.g. tree=853 on a
                            // long IO-to-fabric route) that cleanup A* can
                            // extend; rebuilding from empty under unchanged
                            // congestion just repeats whatever failure
                            // produced the partial. Track it for diagnostics.
                            neg_state.ever_seen_partial.insert(idx);
                            sticky_skipped += 1;
                            continue;
                        }
                        // Empty trees always get retried — there is nothing to
                        // preserve, and forcing a retry here is the only way
                        // they get bound at all (route_net_raster on iter 0
                        // failed → empty plan → no source bind).
                        to_retry.push(idx);
                    }
                }
                eprintln!(
                    "  pass {} gate: bumped={} sticky_skipped={} (threshold={})",
                    iter, bumped_in_pass, sticky_skipped, neg_state.cfg.sticky_ripup_threshold,
                );
                to_retry.sort_unstable();
                to_retry.dedup();
                if to_retry.is_empty() && sticky_skipped == 0 {
                    eprintln!(
                        "RasterRouter: no wire congestion at pass {}, validating",
                        iter
                    );
                    return validate_all_routed(ctx);
                }
                cong.update_history();
                cong_weight *= cfg.cong_growth;
                // Rebuild tile congestion from remaining (non-ripped-up) routes.
                cong.reset_usage();
                for &net in nets {
                    for &wire in ctx.net(net).wires().keys() {
                        cong.add_usage(wire.tile(), 1.0);
                    }
                }
                nets_to_route = to_retry;
            }

            // Sort nets: high-fanout first.
            let mut sorted_nets: Vec<(NetId, f64)> = nets_to_route
                .iter()
                .map(|&net| {
                    let n_users = ctx.net(net).num_users() as f64;
                    let score = n_users * 100.0;
                    (net, score)
                })
                .collect();
            sorted_nets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if iter == 0 {
                let top5: Vec<(f64, usize)> = sorted_nets
                    .iter()
                    .take(5)
                    .map(|(net, s)| (*s, ctx.net(*net).num_users()))
                    .collect();
                eprintln!("  priority top-5 (score, fanout): {:?}", top5);
            }

            let ordered_nets: Vec<NetId> = sorted_nets.iter().map(|x| x.0).collect();

            // Split: high-fanout sequential, rest parallel.
            let hf_cutoff = 20usize;
            let mut hf_nets: Vec<NetId> = Vec::new();
            let mut lf_nets: Vec<NetId> = Vec::new();
            for &net in &ordered_nets {
                if ctx.net(net).num_users() >= hf_cutoff {
                    hf_nets.push(net);
                } else {
                    lf_nets.push(net);
                }
            }

            let mut hf_routed = 0usize;
            for &net in &hf_nets {
                let result = if cfg.use_greedy {
                    route_net_greedy(ctx, net, &cong, cong_weight, cfg)
                } else {
                    route_net_raster(ctx, net, &cong, cong_weight, cfg)
                };
                if let Ok(plan) = result {
                    if plan.source_wire.is_valid() && !plan.sink_routes.is_empty() {
                        match apply_route_plan(ctx, &plan) {
                            Ok(()) => {
                                for sr in &plan.sink_routes {
                                    for &pip in &sr.pips {
                                        let tile = ctx.chipdb().pip_dst_wire(pip).tile();
                                        cong.add_usage(tile, 1.0);
                                    }
                                }
                                hf_routed += 1;
                            }
                            Err(_conflict) => {}
                        }
                    }
                }
            }

            if iter == 0 && !hf_nets.is_empty() {
                eprintln!(
                    "  high-fanout: {}/{} routed sequentially",
                    hf_routed,
                    hf_nets.len()
                );
            }

            // Serialize low-fanout planning so each plan sees prior applies.
            // Parallel planning with par_iter() was racing: workers planned
            // independently through the same node, and the bind-conflict check
            // in apply_route_plan rejected >60% of plans. Sequential planning
            // lets each net observe current bindings via cong/ctx state.
            let mut routed = hf_routed;
            let mut failed = hf_nets.len() - hf_routed;

            for &net in &lf_nets {
                let result = if cfg.use_greedy {
                    route_net_greedy(ctx, net, &cong, cong_weight, cfg)
                } else {
                    route_net_raster(ctx, net, &cong, cong_weight, cfg)
                };
                match result {
                    Ok(plan) if plan.source_wire.is_valid() && !plan.sink_routes.is_empty() => {
                        match apply_route_plan(ctx, &plan) {
                            Ok(()) => {
                                for sr in &plan.sink_routes {
                                    for &pip in &sr.pips {
                                        let tile = ctx.chipdb().pip_dst_wire(pip).tile();
                                        cong.add_usage(tile, 1.0);
                                    }
                                }
                                routed += 1;
                            }
                            Err(_conflict) => {
                                failed += 1;
                            }
                        }
                    }
                    _ => {
                        failed += 1;
                    }
                }
            }

            // Update wire-level negotiation state.
            neg_state.update_usage(&ctx.design);
            neg_state.update_history();
            neg_state.present_cost *= neg_state.cfg.present_cost_growth;
            neg_state.present_cost = neg_state.present_cost.min(8.0);

            let congested = cong.num_congested_tiles();
            let wire_congested = neg_state.count_congested_wires();
            let wl = routed as u64; // use routed count as proxy for quality

            eprintln!(
                "RasterRouter pass {}: routed={}, failed={}, tile_cong={}, wire_cong={}, max_usage={:.0}, wl={}, cong_w={:.2}",
                iter, routed, failed, congested, wire_congested, cong.max_usage(), wl, cong_weight,
            );

            if routed > best_routed || (routed == best_routed && wl < best_wl) {
                best_routed = routed;
                best_wl = wl;
            }

            if congested == 0 && wire_congested == 0 && failed == 0 {
                eprintln!("RasterRouter: converged at pass {}", iter);
                return Ok(());
            }

            // Cleanup pass: route every net that isn't fully routed —
            // empty-tree AND partial-tree alike. Iterate over ALL alive nets,
            // not just `ordered_nets`, so sticky nets (which bypass the rip-up
            // gate at iter>=1 and stay out of the routing loop) still get a
            // chance to have their missing sinks completed from their preserved
            // partial tree.
            let mut failed_nets: Vec<NetId> = ctx
                .design
                .iter_net_indices()
                .filter(|&net| {
                    let n = ctx.net(net);
                    n.is_alive()
                        && n.has_driver()
                        && n.num_users() > 0
                        && !net_fully_routed(ctx, net)
                })
                .collect();
            if !failed_nets.is_empty() {
                // Route const nets first: their per-sink lookup is O(1) and
                // deterministic, so we never lose a const net to the wall-clock
                // budget getting burned by a slow non-const A*.
                failed_nets.sort_by_key(|&n| {
                    let cv = resolve_source_wire(ctx, n)
                        .ok()
                        .flatten()
                        .map(|w| source_wire_const_value(ctx, w))
                        .unwrap_or(0);
                    if cv != 0 {
                        0
                    } else if resolve_source_wire(ctx, n)
                        .ok()
                        .flatten()
                        .map(|w| is_global_clock_wire(ctx, w))
                        .unwrap_or(false)
                    {
                        1
                    } else {
                        2
                    }
                });

                if !failed_nets.is_empty() {
                    // Build wire penalty from negotiation state for A* cost.
                    let wire_penalty: FxHashMap<WireId, crate::timing::DelayT> = neg_state
                        .wire_history
                        .iter()
                        .map(|(&w, &h)| {
                            (
                                w,
                                (h * neg_state.cfg.history_cost_multiplier)
                                    as crate::timing::DelayT,
                            )
                        })
                        .collect();

                    // Hoist clock-backbone enumeration out of the per-net
                    // loop. Iterating every wire in the chipdb for each failed
                    // net costs ~2.5M wires × ~16 nets per cleanup pass —
                    // slow enough to eat the entire 30 s wall budget before
                    // the first A* runs. Compute once per iteration.
                    let mut clock_backbone_wires: Vec<WireId> = Vec::new();
                    let needs_clock_backbone = failed_nets.iter().any(|&n| {
                        resolve_source_wire(ctx, n)
                            .ok()
                            .flatten()
                            .map(|w| is_global_clock_wire(ctx, w))
                            .unwrap_or(false)
                    });
                    if needs_clock_backbone {
                        for w in ctx.chipdb().wires() {
                            if is_global_clock_wire(ctx, w) {
                                clock_backbone_wires.push(w);
                            }
                        }
                    }

                    eprintln!(
                        "RasterRouter: A* cleanup for {} failed nets (iter {})",
                        failed_nets.len(),
                        iter,
                    );

                    let mut cleanup_routed = 0usize;

                    for &net in &failed_nets {
                        let source_wire = match resolve_source_wire(ctx, net) {
                            Ok(Some(w)) => w,
                            _ => continue,
                        };

                        let mut sink_wires = collect_sink_wires(ctx, net);

                        // Sort sinks nearest-first so the routing tree grows
                        // toward the destination, making subsequent A* calls cheaper.
                        let (sx, sy) = ctx.chipdb().tile_xy(source_wire.tile());
                        sink_wires.sort_by_key(|&sw| {
                            let (wx, wy) = ctx.chipdb().tile_xy(sw.tile());
                            (wx - sx).abs() + (wy - sy).abs()
                        });

                        let src_const = source_wire_const_value(ctx, source_wire);

                        let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
                        tree_wires.insert(source_wire);
                        ctx.chipdb().node_wires_cb(source_wire, |nw| {
                            tree_wires.insert(nw);
                        });
                        // Seed the partial-routed tree so cleanup A* can reach
                        // unrouted sinks via the net's existing pip chain.
                        for &w in ctx.net(net).wires().keys() {
                            tree_wires.insert(w);
                            ctx.chipdb().node_wires_cb(w, |nw| {
                                tree_wires.insert(nw);
                            });
                        }

                        // Global-clock nets are driven by a single BUFG output
                        // whose `pips_downhill` is near-empty. Seeding A* with
                        // the whole clock backbone lets the search start from
                        // anywhere nearby — without this the heap drains after
                        // ≤4 expansions and no sink is reachable.
                        if is_global_clock_wire(ctx, source_wire) {
                            for &w in &clock_backbone_wires {
                                tree_wires.insert(w);
                            }
                        }

                        let mut sink_routes = Vec::new();
                        // For constant nets, remember each sink's switch-matrix
                        // anchor (the `pip.src` whose `const_value` matches).
                        // These are registered as leaves after apply_route_plan
                        // so the validator's chain walk can terminate at them.
                        let mut const_anchors: FxHashSet<WireId> = FxHashSet::default();

                        for &sink_wire in &sink_wires {
                            // Constant-net fast path: each tile's switch matrix
                            // ties GND/VCC to a local wire reachable from every
                            // const-consuming sink via one PIP. Use that path
                            // directly — it's deterministic, O(|pips_uphill|),
                            // and avoids the multi-source A* heap blow-up that
                            // a chip-wide const pool would cause.
                            if src_const != 0 {
                                if let Some(pip) = find_local_const_pip(ctx, sink_wire, src_const) {
                                    let anchor = ctx.chipdb().pip_src_wire(pip);
                                    const_anchors.insert(anchor);
                                    // Let subsequent sinks see this anchor as
                                    // part of the routing tree so A* fallback
                                    // (if triggered) skips re-routing to it.
                                    tree_wires.insert(anchor);
                                    tree_wires.insert(sink_wire);
                                    ctx.chipdb().node_wires_cb(sink_wire, |nw| {
                                        tree_wires.insert(nw);
                                    });
                                    sink_routes.push(SinkRoute {
                                        sink_wire,
                                        pips: vec![pip],
                                    });
                                    continue;
                                }
                            }

                            if tree_wires.contains(&sink_wire) {
                                sink_routes.push(SinkRoute {
                                    sink_wire,
                                    pips: vec![],
                                });
                                continue;
                            }

                            // Use multi-pip-hop expansion for long-distance
                            // sinks. Single-hop A* on IOB→interior routes
                            // wastes its visit budget wandering in the IOB
                            // column because the lookahead has no sample for
                            // IO wire types and every short-range hop looks
                            // equally promising. Two-hop expansion lets the
                            // heuristic see past the IOB wall.
                            let (sink_tx_early, sink_ty_early) =
                                ctx.chipdb().tile_xy(sink_wire.tile());
                            let manhattan = (sink_tx_early - sx).abs() + (sink_ty_early - sy).abs();

                            // Spatial bound: a bbox enclosing all of this net's
                            // BEL locations plus a margin scaled by the net's
                            // span. Inside a finite bbox, A* either finds a
                            // path or drains its state space — there is no
                            // legitimate reason to also impose an artificial
                            // visit cap. Leaving `visit_limit=None` lets the
                            // astar core compute a default from the bbox.
                            let cleanup_margin =
                                EXPLORE_BBOX_MARGIN_MIN.max(manhattan * EXPLORE_BBOX_SPAN_MULT);
                            let cleanup_bbox =
                                crate::metrics::compute_bbox(ctx, net, cleanup_margin);
                            let per_sink_limit: Option<usize> = None;

                            let route_fn = if manhattan > 50 {
                                astar_route_multihop
                            } else {
                                astar_route
                            };

                            let trace_cleanup = std::env::var("RASTER_CLEANUP_TRACE").is_ok();
                            let pips_opt = if trace_cleanup {
                                let (pips, trace) = astar_route_with_trace(
                                    ctx,
                                    net,
                                    &tree_wires,
                                    sink_wire,
                                    &wire_penalty,
                                    Some(&cleanup_bbox),
                                    50,
                                    Some(&lookahead),
                                    per_sink_limit,
                                    manhattan > 50,
                                );
                                if pips.is_none() {
                                    let net_name = ctx.name_of(ctx.net(net).name_id()).to_owned();
                                    let (sxt, syt) = ctx.chipdb().tile_xy(sink_wire.tile());
                                    let exit_label = match trace.exit {
                                        AStarExit::Reached => "Reached",
                                        AStarExit::VisitLimit => "VisitLimit",
                                        AStarExit::HeapDrained => "HeapDrained",
                                    };
                                    eprintln!(
                                        "  cleanup_fail: net='{}' sink=({},{}) manhattan={} exit={} visits={} budget={} visited_wires={}",
                                        net_name,
                                        sxt,
                                        syt,
                                        manhattan,
                                        exit_label,
                                        trace.visit_count,
                                        trace.max_visits,
                                        trace.visited.len(),
                                    );
                                }
                                pips
                            } else {
                                route_fn(
                                    ctx,
                                    net,
                                    &tree_wires,
                                    sink_wire,
                                    &wire_penalty,
                                    Some(&cleanup_bbox),
                                    50,
                                    Some(&lookahead),
                                    per_sink_limit,
                                )
                            };

                            match pips_opt {
                                Some(pips) => {
                                    for &pip in &pips {
                                        let dw = ctx.chipdb().pip_dst_wire(pip);
                                        tree_wires.insert(dw);
                                        ctx.chipdb().node_wires_cb(dw, |nw| {
                                            tree_wires.insert(nw);
                                        });
                                    }
                                    sink_routes.push(SinkRoute { sink_wire, pips });
                                }
                                None => {
                                    // Partial routing: bump history for wires in the
                                    // rasterized corridor tiles so that subsequent
                                    // iterations route away from this congested region.
                                    let (sink_tx, sink_ty) = ctx.chipdb().tile_xy(sink_wire.tile());
                                    let (_path, corridor_tiles) = raster_corridor(
                                        sx,
                                        sy,
                                        sink_tx,
                                        sink_ty,
                                        &cong,
                                        cong_weight,
                                        cfg.filter_radius,
                                        chip_w,
                                        chip_h,
                                    );
                                    for &tile_idx in &corridor_tiles {
                                        let tile = tile_idx as i32;
                                        let num_wires =
                                            ctx.chipdb().tile_type(tile).wires.get().len();
                                        for wi in 0..num_wires {
                                            let w = WireId::new(tile, wi as i32);
                                            *neg_state.wire_history.entry(w).or_insert(0.0) += 0.5;
                                        }
                                    }
                                    continue;
                                }
                            }
                        }

                        // Scoped rip-up: if normal A* cleanup routed nothing for
                        // this net, its corridor is occupied by other nets and
                        // the binding-aware cost model correctly refused to plan
                        // through them. Try ripping up the minimal blocker set
                        // (exploration A* identifies exactly which foreign nets
                        // need to move), then re-route both this net and each
                        // blocker. Full rollback on any cascade so the baseline
                        // is preserved.
                        if sink_routes.is_empty() && ripup_budget > 0 {
                            if scoped_ripup_route(
                                ctx,
                                net,
                                &wire_penalty,
                                &lookahead,
                                &mut ripup_budget,
                                &mut neg_state,
                            ) {
                                cleanup_routed += 1;
                                continue;
                            }
                        }

                        // Apply the route if at least some sinks succeeded.
                        if !sink_routes.is_empty() {
                            // For A*-routed const sinks the first PIP's `src`
                            // is a switch-matrix const wire — gather those too
                            // so the validator accepts mixed local + A* const
                            // routes.
                            if src_const != 0 {
                                for sink in &sink_routes {
                                    if let Some(&first_pip) = sink.pips.first() {
                                        let anchor = ctx.chipdb().pip_src_wire(first_pip);
                                        if ctx.chipdb().wire_info(anchor).const_value == src_const {
                                            const_anchors.insert(anchor);
                                        }
                                    }
                                }
                            }

                            let plan = RoutePlan {
                                net,
                                source_wire,
                                sink_routes,
                            };
                            let apply_res = apply_route_plan(ctx, &plan);
                            if apply_res.is_ok() {
                                // Register const anchors as leaves so the
                                // validator's chain walk terminates there via
                                // the const_value match rule. Binding is
                                // idempotent per net; anchor wires are tile-
                                // local so cross-net conflicts only arise if
                                // two different nets carry the same const
                                // value (normally only one GND and one VCC
                                // net exist).
                                let mut anchor_conflict = false;
                                for &anchor in &const_anchors {
                                    if ctx
                                        .try_bind_wire_node(
                                            anchor,
                                            net,
                                            crate::common::PlaceStrength::Strong,
                                        )
                                        .is_err()
                                    {
                                        anchor_conflict = true;
                                        break;
                                    }
                                    ctx.design.net_edit(net).add_wire(
                                        anchor,
                                        None,
                                        crate::common::PlaceStrength::Strong,
                                    );
                                }
                                if anchor_conflict {
                                    unroute_net(ctx, net);
                                } else if net_fully_routed(ctx, net) {
                                    cleanup_routed += 1;
                                }
                            }
                        }
                    }

                    // Recompute remaining failures from ground truth: anything
                    // still partial after this cleanup pass should drive the
                    // next PathFinder iteration. Subtracting `cleanup_routed`
                    // from the prior `failed` undercounts when the first pass
                    // routed nets only partially (the per-sink loop applies a
                    // plan with missing sinks, leaving the net partial but
                    // counted as "routed").
                    // Count from the full design, not `ordered_nets`. Sticky
                    // nets (skipped at the gate) are not in `ordered_nets` but
                    // are still genuine failures if they aren't fully routed,
                    // and undercounting trips the `failed == 0` early-exit
                    // before they get more cleanup attempts.
                    failed = ctx
                        .design
                        .iter_net_indices()
                        .filter(|&n| {
                            let net = ctx.net(n);
                            net.is_alive()
                                && net.has_driver()
                                && net.num_users() > 0
                                && !net_fully_routed(ctx, n)
                        })
                        .count();
                    eprintln!(
                        "  A* cleanup: +{} fully-routed, {} still partial/failed",
                        cleanup_routed, failed,
                    );
                }
            }

            if failed == 0 {
                eprintln!("RasterRouter: all nets routed at pass {}, validating", iter);
                return validate_all_routed(ctx);
            }
        }

        eprintln!(
            "RasterRouter: {} passes, best_routed={}, congested={}",
            cfg.max_iterations,
            best_routed,
            cong.num_congested_tiles(),
        );
        Err(RouterError::Congestion(
            cfg.max_iterations,
            cong.num_congested_tiles(),
        ))
    }
}

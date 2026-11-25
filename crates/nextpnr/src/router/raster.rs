//! Router3: Gupta-Sproull raster router with beam-search PIP walking.
//!
//! Uses the Gupta-Sproull anti-aliased line drawing algorithm to establish a
//! tile corridor, then walks the actual PIP graph with a greedy beam search
//! scored by timing (progress toward sink / PIP delay). Long wires naturally
//! win because they cover many tiles for one PIP traversal.
//!
//! Iterates with congestion feedback. Intended for fast wirelength estimation.

use crate::chipdb::{PipId, WireId};
use crate::common::DelayT;
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::netlist::NetId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::AStarExit;
use super::astar::{
    astar_search, default_pip_cost, reconstruct_path_to, AStarOptions, PathCostModel,
};
use super::common::{
    apply_route_plan, collect_routable_nets, collect_sink_wires, find_local_const_pip,
    is_global_net, resolve_source_wire, source_wire_const_value, unroute_net, validate_all_routed,
    RoutePlan, SinkRoute,
};
use super::maze::{
    astar_route, astar_route_explore, astar_route_explore_exit, astar_route_multihop,
    astar_route_with_trace,
};
use super::RouterError;

/// True iff every valid sink of `net` is in the net's routing tree. Used to
/// identify nets the primary pass left partially unrouted so the cleanup A*
/// can finish them.
fn net_fully_routed(ctx: &Context, net_id: NetId) -> bool {
    let net = ctx.net(net_id);
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
        if !net.wires().contains_key(&sink_wv.id()) {
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

/// In-loop rip-up budgets. The cleanup-stage constants above assume ≤8
/// invocations per run; the in-loop sweep visits every failed net every
/// pass (tens of thousands on FPGA01), so each attempt must be orders of
/// magnitude cheaper and give up early rather than search exhaustively.
const INLOOP_EXPLORE_LIMIT: usize = 200_000;
/// Escalated explore budget when the tight-bbox explore *drains* (a proof
/// nothing exists at `bbox_margin`, not a budget miss): the retry widens to
/// the cleanup-proven `max(60, span*3)` margin, whose bbox holds millions of
/// wires, so the tight budget would misreport it as capped.
const INLOOP_WIDE_EXPLORE_LIMIT: usize = 1_000_000;
/// Sink attempts per net per sweep. Sinks recover independently now, so this
/// bounds the worst-case cost of one very broken net, not the recovery.
const INLOOP_SINKS_PER_NET: usize = 4;
/// Wall budget per sweep, sized to one retry pass. v1 spent 2059s to recover
/// 16 nets, 3.4x a pass for a quarter of its yield; the sweep must never
/// cost more than the pass it augments. Unvisited nets carry to next sweep.
const INLOOP_SWEEP_WALL_SECS: u64 = 600;
/// A corridor needing more movers than this is contested, not blocked;
/// surgical rip-up would cascade, so the attempt is skipped.
const INLOOP_MAX_BLOCKERS: usize = 8;
/// Never rip a victim whose reroute means replaying this many sinks; the
/// rollback cost of a late failure grows with victim fanout.
const INLOOP_MAX_FANOUT: usize = 64;
/// Per-victim rip count per sweep. Two nets trading the same corridor
/// every pass is the oscillation that sank gate-level partial rip-up
/// (18511 partials rebuilt into 5229); the ledger caps the trade.
const INLOOP_RIPS_PER_VICTIM: u32 = 2;

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
                    tree_wires.insert(ctx.chipdb().pip_dst_wire(pip));
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

    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(source_wire);

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
        // Binding is per node, so the node's one binding names its owner.
        let dst = ctx.chipdb().pip_dst_wire(pip);
        if let Some((owner, _)) = ctx.wire_binding(dst) {
            if owner != net {
                blockers.insert(owner);
            }
        }
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
        unroute_net(ctx, b);
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
        unroute_net(ctx, net);
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
            unroute_net(ctx, b);
            unroute_net(ctx, net);
            for snap in &snapshots[..i] {
                unroute_net(ctx, snap.net);
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

/// Per-sweep skip and outcome counters, printed once per pass so a sweep
/// that recovers nothing still says which guard rejected its attempts. The
/// explore exits split budget misses (`capped_*`) from infeasibility proofs
/// (`dead_*`, cached for the rest of the run), and the two reroute failure
/// modes are separated because they point at different fixes: `extend_fail`
/// at the retry budget or cost model, `victim_fail` at the victim bboxes.
#[derive(Default)]
struct InloopStats {
    attempted: usize,
    nets_recovered: usize,
    sinks_recovered: usize,
    ripups: usize,
    free_extends: usize,
    skip_dead_cache: usize,
    skip_fanout: usize,
    skip_too_many: usize,
    skip_ledger: usize,
    capped_tight: usize,
    capped_wide: usize,
    wide_found: usize,
    dead_short: usize,
    dead_mid: usize,
    dead_long: usize,
    extend_fail: usize,
    victim_fail: usize,
}

/// One in-loop recovery pass over a net's missing sinks.
///
/// Sinks recover independently: each attempt explores unbound for a corridor,
/// names the blockers on it, rips only the blockers, *extends* the net's
/// existing tree to the sink (never rebuilding sinks that already routed —
/// v1's all-or-nothing reroute rolled back a freed corridor whenever any
/// other sink still failed, 1647 of 1663 attempts), then reroutes each victim.
/// Any failure rolls the net and the victims back to their exact prior state,
/// so a partial tree survives every failed attempt.
///
/// The explore runs at the routing margin first; if it *drains* — an
/// infeasibility proof for that bbox, not a budget miss — it escalates once
/// to the cleanup-proven `max(60, span*3)` margin. Draining there too is
/// permanent for the run (the unbound graph and placement are fixed), so the
/// `(net, sink)` lands in `dead_ends` and is never explored again.
fn inloop_ripup_net(
    ctx: &mut Context,
    net: NetId,
    cfg: &RasterRouterCfg,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    lookahead: &super::lookahead::Lookahead,
    neg_state: &mut super::common::NegotiationState,
    victim_ledger: &mut FxHashMap<NetId, u32>,
    dead_ends: &mut FxHashSet<(NetId, WireId)>,
    reroute_limit: usize,
    stats: &mut InloopStats,
) -> bool {
    if ctx.net(net).num_users() > INLOOP_MAX_FANOUT {
        stats.skip_fanout += 1;
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

    let mut tree_wires: FxHashSet<WireId> = ctx.net(net).wires().keys().copied().collect();
    tree_wires.insert(source_wire);

    let missing: Vec<WireId> = sink_wires
        .iter()
        .copied()
        .filter(|s| !tree_wires.contains(s))
        .collect();

    let tight_bbox = crate::metrics::compute_bbox(ctx, net, cfg.bbox_margin.max(1));
    let bel_bbox = crate::metrics::compute_bbox(ctx, net, 0);
    let span = (bel_bbox.x1 - bel_bbox.x0)
        .max(bel_bbox.y1 - bel_bbox.y0)
        .max(0);
    let wide_margin = EXPLORE_BBOX_MARGIN_MIN.max(span * EXPLORE_BBOX_SPAN_MULT);
    let wide_bbox = crate::metrics::compute_bbox(ctx, net, wide_margin);

    for &sink in missing.iter().take(INLOOP_SINKS_PER_NET) {
        if dead_ends.contains(&(net, sink)) {
            stats.skip_dead_cache += 1;
            continue;
        }
        stats.attempted += 1;

        // Tight explore first: cheap, and its corridor is one the extension
        // can actually use. Escalate only on a drain, which proves the tight
        // bbox empty rather than the budget short.
        let (mut explore_path, exit) = astar_route_explore_exit(
            ctx,
            net,
            &tree_wires,
            sink,
            wire_penalty,
            Some(&tight_bbox),
            Some(lookahead),
            Some(INLOOP_EXPLORE_LIMIT),
        );
        let mut attempt_bbox = &tight_bbox;
        if explore_path.is_none() {
            match exit {
                AStarExit::VisitLimit => {
                    stats.capped_tight += 1;
                    continue;
                }
                AStarExit::HeapDrained => {
                    let (wide_path, wide_exit) = astar_route_explore_exit(
                        ctx,
                        net,
                        &tree_wires,
                        sink,
                        wire_penalty,
                        Some(&wide_bbox),
                        Some(lookahead),
                        Some(INLOOP_WIDE_EXPLORE_LIMIT),
                    );
                    match (wide_path, wide_exit) {
                        (Some(p), _) => {
                            stats.wide_found += 1;
                            explore_path = Some(p);
                            attempt_bbox = &wide_bbox;
                        }
                        (None, AStarExit::HeapDrained) => {
                            dead_ends.insert((net, sink));
                            let (sx, sy) = ctx.chipdb().tile_xy(source_wire.tile());
                            let (tx, ty) = ctx.chipdb().tile_xy(sink.tile());
                            let dist = (tx - sx).abs() + (ty - sy).abs();
                            if dist <= 20 {
                                stats.dead_short += 1;
                            } else if dist <= 60 {
                                stats.dead_mid += 1;
                            } else {
                                stats.dead_long += 1;
                            }
                            continue;
                        }
                        (None, _) => {
                            stats.capped_wide += 1;
                            continue;
                        }
                    }
                }
                AStarExit::Reached => continue,
            }
        }
        let explore_path = explore_path.unwrap();

        let mut blockers: FxHashSet<NetId> = FxHashSet::default();
        for &pip in &explore_path {
            let dst = ctx.chipdb().pip_dst_wire(pip);
            if let Some((owner, _)) = ctx.wire_binding(dst) {
                if owner != net {
                    blockers.insert(owner);
                }
            }
        }

        let mut blocker_list: Vec<NetId> = blockers.into_iter().collect();
        blocker_list.sort_unstable();
        if blocker_list.len() > INLOOP_MAX_BLOCKERS {
            stats.skip_too_many += 1;
            continue;
        }
        if blocker_list
            .iter()
            .any(|&b| ctx.net(b).num_users() > INLOOP_MAX_FANOUT)
        {
            stats.skip_fanout += 1;
            continue;
        }
        if blocker_list
            .iter()
            .any(|&b| victim_ledger.get(&b).copied().unwrap_or(0) >= INLOOP_RIPS_PER_VICTIM)
        {
            stats.skip_ledger += 1;
            continue;
        }

        // Own snapshot taken before the extension so a victim-reroute failure
        // can restore the pre-extension tree exactly.
        let own_snapshot = snapshot_net(ctx, net);
        let snapshots: Vec<NetSnapshot> =
            blocker_list.iter().map(|&b| snapshot_net(ctx, b)).collect();
        for &b in &blocker_list {
            unroute_net(ctx, b);
            *victim_ledger.entry(b).or_insert(0) += 1;
            neg_state.bump_rip_up(b);
        }

        // Extension binds nothing until apply_route_plan, and that call is
        // atomic, so the only state to restore on failure is the victims'.
        let extension = astar_route(
            ctx,
            net,
            &tree_wires,
            sink,
            wire_penalty,
            Some(attempt_bbox),
            50,
            Some(lookahead),
            Some(reroute_limit),
        )
        .and_then(|pips| {
            let plan = RoutePlan {
                net,
                source_wire,
                sink_routes: vec![SinkRoute {
                    sink_wire: sink,
                    pips,
                }],
            };
            apply_route_plan(ctx, &plan).ok().map(|()| plan)
        });
        let plan = match extension {
            Some(p) => p,
            None => {
                for snap in &snapshots {
                    if !restore_net(ctx, snap) {
                        eprintln!(
                            "  inloop_ripup: CRITICAL restore failure for victim {:?}",
                            snap.net
                        );
                    }
                }
                stats.extend_fail += 1;
                continue;
            }
        };

        let mut victims_ok = true;
        for (i, &b) in blocker_list.iter().enumerate() {
            let vb_span_bbox = crate::metrics::compute_bbox(ctx, b, 0);
            let vb_span = (vb_span_bbox.x1 - vb_span_bbox.x0)
                .max(vb_span_bbox.y1 - vb_span_bbox.y0)
                .max(0);
            let vb_margin = EXPLORE_BBOX_MARGIN_MIN.max(vb_span * EXPLORE_BBOX_SPAN_MULT);
            let victim_bbox = crate::metrics::compute_bbox(ctx, b, vb_margin);
            if !reroute_from_scratch(
                ctx,
                b,
                wire_penalty,
                lookahead,
                reroute_limit,
                Some(&victim_bbox),
            ) {
                // Roll back: the failed victim, our extension, the victims
                // already rerouted, then restore every snapshot.
                unroute_net(ctx, b);
                unroute_net(ctx, net);
                for snap in &snapshots[..i] {
                    unroute_net(ctx, snap.net);
                }
                if !restore_net(ctx, &own_snapshot) {
                    eprintln!(
                        "  inloop_ripup: CRITICAL restore failure for own net {:?}",
                        net
                    );
                }
                for snap in &snapshots {
                    if !restore_net(ctx, snap) {
                        eprintln!(
                            "  inloop_ripup: CRITICAL restore failure for victim {:?}",
                            snap.net
                        );
                    }
                }
                stats.victim_fail += 1;
                victims_ok = false;
                break;
            }
        }
        if !victims_ok {
            continue;
        }

        // Sink landed: fold its corridor into the tree for the next sink.
        for sr in &plan.sink_routes {
            for &pip in &sr.pips {
                tree_wires.insert(ctx.chipdb().pip_dst_wire(pip));
            }
            tree_wires.insert(sr.sink_wire);
        }
        stats.sinks_recovered += 1;
        if blocker_list.is_empty() {
            stats.free_extends += 1;
        }
        stats.ripups += blocker_list.len();
    }

    let fully = net_fully_routed(ctx, net);
    if fully {
        stats.nets_recovered += 1;
    }
    fully
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
    /// Per-sink A* pop budget. `None` defers to the kernel default, which on
    /// FPGA01 lets a doomed sink spend ~214k pops, 44x a successful one and
    /// over half the router's total pops. A flat cap trades those for sinks
    /// that would have been found late in the search.
    pub sink_visit_limit: Option<usize>,
    /// Same budget for the per-iteration cleanup A*, which the pass-0 cap does
    /// not reach. `None` keeps the derived default.
    pub cleanup_visit_limit: Option<usize>,
    /// Same budget for the exclusive finisher's per-sink A*. The flat cap that
    /// makes negotiation tractable starves the finisher: every one of its 20477
    /// lost sinks exits on `VisitLimit`, and 95% of them sit 16 or more tiles
    /// from the nearest wire already in their own tree. `None` restores the
    /// derived default, which scales as the square of that same tree distance.
    pub finish_visit_limit: Option<usize>,
    /// Base of the finisher's per-net wall-clock budget, before the 10ms per
    /// sink it adds. `None` removes the deadline, leaving `finish_visit_limit`
    /// as the only bound; a larger pop budget pushes per-sink cost up by the
    /// same factor, so a fixed wall would abandon sinks the budget was raised
    /// to reach and measure the deadline instead.
    pub finish_net_deadline: Option<std::time::Duration>,
    /// Drop the per-iteration cleanup stage and let the pass loop's tile
    /// congestion history carry the design on its own. Also unfreezes partial
    /// trees at the rip-up gate: they are preserved only so cleanup A* can
    /// extend them, so keeping that half without the other floors the unrouted
    /// count at however many partials pass 0 left.
    pub skip_cleanup: bool,
    /// Run the scoped rip-up sweep on every failed net at the end of each
    /// pass, and restrict the cleanup stage to its const and clock fast
    /// paths. The chip-scale cleanup A* recovers 1 ordinary net in 98 at
    /// 8.1s each; the sweep spends the budget on the mechanism behind 13 of
    /// cleanup's 14 recoveries instead.
    pub inloop_ripup: bool,
    /// PathFinder-style negotiation: wires may be over-subscribed within a
    /// pass, priced by present over-use and accumulated history, and plans
    /// live in router-owned state until no wire is shared, when they commit
    /// through `apply_route_plan`. Exclusive binding cannot converge on the
    /// FPGA01 residual — displacement has to chain (commit c8c34a7) — and
    /// over-use pricing is the mechanism that chains it.
    pub overuse: bool,
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
            sink_visit_limit: None,
            cleanup_visit_limit: None,
            finish_visit_limit: None,
            finish_net_deadline: Some(std::time::Duration::from_millis(2000)),
            skip_cleanup: false,
            inloop_ripup: false,
            overuse: false,
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
/// manhattan heuristic. The lookahead build measures ≈2.789 pips per tile of
/// manhattan distance, so `H_WEIGHT = 2` was still admissible and searched
/// near-optimally. 5 is ~1.79× weighted, the knee of the 2026-08-26 FPGA01
/// sweep (measurements/route_ab/fpga01_hweight{3,4,5,6}.log): pass-0 unrouted
/// 41560 → 17010 and wall 1005s → 469s, with 6 regressing on both failure
/// classes.
const H_WEIGHT: DelayT = 5;

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

/// Per-sink A* cost, accumulated across a pass. The progress line reports
/// ~320ms per net processed on FPGA01 without saying where it goes; these
/// counters split it by search outcome so a cheap `Reached` is distinguishable
/// from one that grinds to `max_visits`.
mod astar_probe {
    use super::AStarExit;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    pub struct Bucket {
        pub calls: AtomicUsize,
        pub pops: AtomicUsize,
        pub at_limit: AtomicUsize,
    }

    impl Bucket {
        const fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                pops: AtomicUsize::new(0),
                at_limit: AtomicUsize::new(0),
            }
        }
        pub fn take(&self) -> (usize, usize, usize) {
            (
                self.calls.load(Relaxed),
                self.pops.load(Relaxed),
                self.at_limit.load(Relaxed),
            )
        }
    }

    pub static REACHED: Bucket = Bucket::new();
    pub static DRAINED: Bucket = Bucket::new();
    pub static LIMITED: Bucket = Bucket::new();

    /// `at_limit` counts searches that used at least 90% of the budget in
    /// effect, which catches a `Reached` that only just beat the cap.
    pub fn record(exit: AStarExit, visit_count: usize, max_visits: usize) {
        let b = match exit {
            AStarExit::Reached => &REACHED,
            AStarExit::HeapDrained => &DRAINED,
            AStarExit::VisitLimit => &LIMITED,
        };
        b.calls.fetch_add(1, Relaxed);
        b.pops.fetch_add(visit_count, Relaxed);
        if max_visits > 0 && visit_count * 10 >= max_visits * 9 {
            b.at_limit.fetch_add(1, Relaxed);
        }
    }

    /// One line summarising mean pops per sink by exit reason.
    pub fn summary() -> String {
        let mut out = String::new();
        for (name, b) in [
            ("reached", &REACHED),
            ("drained", &DRAINED),
            ("limited", &LIMITED),
        ] {
            let (calls, pops, at_limit) = b.take();
            let mean = if calls > 0 {
                pops as f64 / calls as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                " {name}={calls} pops/sink={mean:.0} at_limit={at_limit}"
            ));
        }
        out
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
    sink_visit_limit: Option<usize>,
) -> (Option<Vec<PipId>>, AStarExit, usize) {
    let model = RasterCostModel {
        net,
        bbox,
        cong,
        cong_weight,
    };
    let opts = AStarOptions {
        visit_limit: sink_visit_limit,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: true,
    };
    let res = astar_search(ctx, &model, src_wires, dst_wire, &opts);
    astar_probe::record(res.trace.exit, res.trace.visit_count, res.trace.max_visits);
    (res.path, res.trace.exit, res.trace.min_approach)
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

/// Pick a representative wire in `target_tile`: the tile wire with the most
/// `pips_uphill`, so it's reachable from many directions. Used as the
/// destination anchor when the actual goal is "any wire in `target_tile`".
/// Returned as its node's canonical wire, which is what searches reach.
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
    best.map(|(w, _)| chipdb.canonical_wire(w))
}

/// True iff `wire`'s node has a tile wire in `tile`.
fn node_touches_tile(chipdb: &crate::chipdb::ChipDb, wire: WireId, tile: i32) -> bool {
    let mut hit = false;
    chipdb.node_members_cb(wire, |m| hit |= m.tile() == tile);
    hit
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
            // Scan the visited map for the lowest-cost node that reaches
            // into `target_tile`; reconstruct a path to it. This
            // generalises the old "accept any wire in target_tile" hit
            // condition without adding multi-goal support to the kernel.
            let mut best: Option<(WireId, DelayT)> = None;
            for (&w, &(cost, pen, _, _)) in result.trace.visited.iter() {
                if !node_touches_tile(chipdb, w, target_tile) {
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

    if src_wires.contains(&dst_wire) {
        return Some(Vec::new());
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
                // Also add PIP destination wires.
                for &pip in &pips {
                    current_wires.insert(chipdb.pip_dst_wire(pip));
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
                    // Where the pip sits, not where its node is rooted.
                    let (tx, ty) = chipdb.tile_xy(pip.tile());
                    tree_tiles.push((tx, ty));
                    tree_wires.insert(dst);
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
/// Population census of why a finisher net ends part-built. The `raster_fail`
/// print is capped at the first 100 ever and covers only the A* failure arm,
/// so it cannot see the per-net deadline break: that path leaves the tail of
/// `remaining_sinks` unattempted and prints nothing. Counted over every net
/// and printed once per pass beside `failmode`, so the two halves of the
/// unrouted count can be attributed rather than inferred.
mod partial_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static NETS: AtomicU64 = AtomicU64::new(0);
    pub static NETS_PART: AtomicU64 = AtomicU64::new(0);
    pub static NETS_DEADLINE: AtomicU64 = AtomicU64::new(0);
    pub static LOST_ONE: AtomicU64 = AtomicU64::new(0);
    pub static LOST_FEW: AtomicU64 = AtomicU64::new(0);
    pub static LOST_MANY: AtomicU64 = AtomicU64::new(0);
    pub static LOST_ALL: AtomicU64 = AtomicU64::new(0);
    pub static SINKS_ASTAR: AtomicU64 = AtomicU64::new(0);
    pub static SINKS_DEADLINE: AtomicU64 = AtomicU64::new(0);
    static D_0_15: AtomicU64 = AtomicU64::new(0);
    static D_16_63: AtomicU64 = AtomicU64::new(0);
    static D_64_255: AtomicU64 = AtomicU64::new(0);
    static D_256UP: AtomicU64 = AtomicU64::new(0);
    static T_0: AtomicU64 = AtomicU64::new(0);
    static T_1_31: AtomicU64 = AtomicU64::new(0);
    static T_32_255: AtomicU64 = AtomicU64::new(0);
    static T_256UP: AtomicU64 = AtomicU64::new(0);
    static X_LIMIT: AtomicU64 = AtomicU64::new(0);
    static X_DRAIN: AtomicU64 = AtomicU64::new(0);
    static X_REACHED: AtomicU64 = AtomicU64::new(0);
    static TD_0_3: AtomicU64 = AtomicU64::new(0);
    static TD_4_15: AtomicU64 = AtomicU64::new(0);
    static TD_16_63: AtomicU64 = AtomicU64::new(0);
    static TD_64UP: AtomicU64 = AtomicU64::new(0);
    static AP_0_3: AtomicU64 = AtomicU64::new(0);
    static AP_4_15: AtomicU64 = AtomicU64::new(0);
    static AP_16_63: AtomicU64 = AtomicU64::new(0);
    static AP_64UP: AtomicU64 = AtomicU64::new(0);

    /// One A*-failed sink, bucketed by remaining Manhattan distance and by the
    /// size of the tree already built for the net.
    pub fn note_astar_fail(
        dist: i32,
        tree: usize,
        tree_dist: i32,
        exit: super::AStarExit,
        approach: usize,
    ) {
        SINKS_ASTAR.fetch_add(1, Relaxed);
        match exit {
            super::AStarExit::VisitLimit => &X_LIMIT,
            super::AStarExit::HeapDrained => &X_DRAIN,
            super::AStarExit::Reached => &X_REACHED,
        }
        .fetch_add(1, Relaxed);
        match approach {
            0..=3 => &AP_0_3,
            4..=15 => &AP_4_15,
            16..=63 => &AP_16_63,
            _ => &AP_64UP,
        }
        .fetch_add(1, Relaxed);
        match tree_dist {
            i32::MIN..=3 => &TD_0_3,
            4..=15 => &TD_4_15,
            16..=63 => &TD_16_63,
            _ => &TD_64UP,
        }
        .fetch_add(1, Relaxed);
        match dist {
            i32::MIN..=15 => &D_0_15,
            16..=63 => &D_16_63,
            64..=255 => &D_64_255,
            _ => &D_256UP,
        }
        .fetch_add(1, Relaxed);
        match tree {
            0 => &T_0,
            1..=31 => &T_1_31,
            32..=255 => &T_32_255,
            _ => &T_256UP,
        }
        .fetch_add(1, Relaxed);
    }

    /// One net's outcome. `lost` counts sinks with no route, whatever the
    /// cause; `hit_deadline` records that the sink loop broke early, so the
    /// unattempted tail is separable from sinks A* genuinely could not reach.
    pub fn note_net(total: usize, lost: usize, hit_deadline: bool) {
        NETS.fetch_add(1, Relaxed);
        if hit_deadline {
            NETS_DEADLINE.fetch_add(1, Relaxed);
        }
        if lost == 0 {
            return;
        }
        NETS_PART.fetch_add(1, Relaxed);
        if lost == total {
            LOST_ALL.fetch_add(1, Relaxed);
        } else if lost == 1 {
            LOST_ONE.fetch_add(1, Relaxed);
        } else if lost <= 3 {
            LOST_FEW.fetch_add(1, Relaxed);
        } else {
            LOST_MANY.fetch_add(1, Relaxed);
        }
    }

    pub fn summary() -> String {
        format!(
            "nets={} part={} [lost 1={} 2-3={} 4+={} all={}] \
sinks[astar={} deadline={}] nets_hit_deadline={} \
dist[0-15={} 16-63={} 64-255={} 256+={}] tree[0={} 1-31={} 32-255={} 256+={}] \
exit[limit={} drain={} reached={}] treedist[0-3={} 4-15={} 16-63={} 64+={}] \
approach[0-3={} 4-15={} 16-63={} 64+={}]",
            NETS.load(Relaxed),
            NETS_PART.load(Relaxed),
            LOST_ONE.load(Relaxed),
            LOST_FEW.load(Relaxed),
            LOST_MANY.load(Relaxed),
            LOST_ALL.load(Relaxed),
            SINKS_ASTAR.load(Relaxed),
            SINKS_DEADLINE.load(Relaxed),
            NETS_DEADLINE.load(Relaxed),
            D_0_15.load(Relaxed),
            D_16_63.load(Relaxed),
            D_64_255.load(Relaxed),
            D_256UP.load(Relaxed),
            T_0.load(Relaxed),
            T_1_31.load(Relaxed),
            T_32_255.load(Relaxed),
            T_256UP.load(Relaxed),
            X_LIMIT.load(Relaxed),
            X_DRAIN.load(Relaxed),
            X_REACHED.load(Relaxed),
            TD_0_3.load(Relaxed),
            TD_4_15.load(Relaxed),
            TD_16_63.load(Relaxed),
            TD_64UP.load(Relaxed),
            AP_0_3.load(Relaxed),
            AP_4_15.load(Relaxed),
            AP_16_63.load(Relaxed),
            AP_64UP.load(Relaxed),
        )
    }
}

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

    // Per-net time budget: a base plus 10ms per sink, stopping high-fanout nets
    // from monopolizing routing time across rip-up-reroute passes. Wall-clock,
    // so it makes a net's outcome depend on machine load; `None` drops it and
    // leaves the per-sink pop budget as the only bound.
    let net_deadline = cfg.finish_net_deadline.map(|base| {
        std::time::Instant::now()
            + base
            + std::time::Duration::from_millis(remaining_sinks.len() as u64 * 10)
    });

    let mut sink_routes = Vec::new();
    let n_sinks = remaining_sinks.len();
    let mut hit_deadline = false;

    for (si, (sink_wire, sink_x, sink_y)) in remaining_sinks.into_iter().enumerate() {
        if net_deadline.is_some_and(|d| std::time::Instant::now() > d) {
            hit_deadline = true;
            partial_census::SINKS_DEADLINE
                .fetch_add((n_sinks - si) as u64, std::sync::atomic::Ordering::Relaxed);
            break;
        }
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }

        let (sink_path, sink_exit, sink_approach) = route_sink_astar(
            ctx,
            net,
            &tree_wires,
            sink_wire,
            bbox.as_ref(),
            cong,
            cong_weight,
            cfg.finish_visit_limit,
        );
        match sink_path {
            Some(pips) => {
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    tree_wires.insert(dst);
                }
                tree_wires.insert(sink_wire);

                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                static FAIL_COUNT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (sxw, syw) = chipdb.tile_xy(source_wire.tile());
                let dist = (sink_x - sxw).abs() + (sink_y - syw).abs();
                // The search starts from `tree_wires`, so source-to-sink
                // distance describes only the first sink of a net. A sink 200
                // tiles from the source can sit 5 tiles from a branch, and
                // long-search and locally-blocked want opposite fixes.
                let tree_dist = tree_wires
                    .iter()
                    .map(|&w| {
                        let (tx, ty) = chipdb.tile_xy(w.tile());
                        (sink_x - tx).abs() + (sink_y - ty).abs()
                    })
                    .min()
                    .unwrap_or(dist);
                partial_census::note_astar_fail(
                    dist,
                    tree_wires.len(),
                    tree_dist,
                    sink_exit,
                    sink_approach,
                );
                if n < 100 {
                    let net_name = ctx.net(net).name_id();
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

    partial_census::note_net(n_sinks, n_sinks - sink_routes.len(), hit_deadline);

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

// ---------------------------------------------------------------------------
// Over-use negotiation (PathFinder)
// ---------------------------------------------------------------------------

/// Fixed-point scale for every term of the over-use cost model. `DelayT` is
/// an integer and `wire_penalty` used to cast an unscaled `present × occ +
/// history` straight to it, so every price below 1.0 truncated to zero: at
/// present 0.5-0.86 plain pair sharing was free for the first four passes,
/// and history accrued in 0.25 steps the cast discarded. Pip cost, wire
/// penalty and heuristic all carry this factor, so the ratios between them
/// are unchanged and only the resolution improves.
const OVERUSE_COST_SCALE: DelayT = 16;

/// Compact indices for the routing nodes negotiation has seen, so occupancy
/// and history live in flat vectors that grow with the nodes plans actually
/// use. A table over every tile wire on the chip costs memory in proportion
/// to the device whatever the design, and every multi-tile node in it was
/// counted once per tile it crosses.
struct NodeTable {
    ids: FxHashMap<WireId, u32>,
    wires: Vec<WireId>,
}

impl NodeTable {
    fn new() -> Self {
        Self {
            ids: FxHashMap::default(),
            wires: Vec::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.wires.len()
    }

    /// Index of a node some plan has used, or `None` if none has: it then
    /// has no occupancy and no history.
    #[inline]
    fn get(&self, w: WireId) -> Option<usize> {
        self.ids.get(&w).map(|&i| i as usize)
    }

    fn intern(&mut self, w: WireId) -> u32 {
        *self.ids.entry(w).or_insert_with(|| {
            let i = u32::try_from(self.wires.len()).expect("more than u32::MAX routed nodes");
            self.wires.push(w);
            i
        })
    }

    fn tile_of(&self, i: usize) -> i32 {
        self.wires[i].tile()
    }
}

/// Node indices a plan occupies: source, every pip destination and every
/// sink, deduplicated, so one net contributes exactly one unit of demand per
/// routing node it crosses. Interns new nodes and grows `occ` and `hist` to
/// cover them.
fn plan_wire_indices(
    ctx: &Context,
    nt: &mut NodeTable,
    occ: &mut Vec<u16>,
    hist: &mut Vec<f32>,
    plan: &RoutePlan,
) -> Vec<u32> {
    let chipdb = ctx.chipdb();
    let mut v: Vec<u32> = Vec::with_capacity(1 + plan.sink_routes.len());
    v.push(nt.intern(plan.source_wire));
    for sr in &plan.sink_routes {
        v.push(nt.intern(sr.sink_wire));
        for &pip in &sr.pips {
            v.push(nt.intern(chipdb.pip_dst_wire(pip)));
        }
    }
    occ.resize(nt.len(), 0);
    hist.resize(nt.len(), 0.0);
    v.sort_unstable();
    v.dedup();
    v
}

/// exclusive model's `DelayT::MAX/4` wall. BEL-pin reservations stay walls —
/// a foreign net planning through a sink pin's node can never be priced out
/// of the way, only displaced, which negotiation cannot express.
struct OveruseCostModel<'a> {
    net: NetId,
    bbox: Option<&'a BoundingBox>,
    nt: &'a NodeTable,
    occ: &'a [u16],
    hist: &'a [f32],
    reserved: &'a FxHashMap<WireId, NetId>,
    present: f32,
    /// Price-aware heuristic multiplier, ≥ 1.0. Prices inflate g while a
    /// manhattan h estimates base cost only, so pruning decays as present
    /// grows — pass walls went 92s → 1950s on that mechanism alone. The
    /// caller scales this with the price level to keep search width flat.
    h_scale: f32,
}

impl<'a> PathCostModel for OveruseCostModel<'a> {
    fn pip_cost(&self, _ctx: &Context, _pip: PipId) -> DelayT {
        OVERUSE_COST_SCALE
    }

    fn wire_penalty(&self, _ctx: &Context, wire: WireId) -> DelayT {
        if let Some(&owner) = self.reserved.get(&wire) {
            if owner != self.net {
                return DelayT::MAX / 4;
            }
        }
        let Some(i) = self.nt.get(wire) else {
            return 0;
        };
        let price = self.present * self.occ[i] as f32 + self.hist[i];
        (price * OVERUSE_COST_SCALE as f32).round() as DelayT
    }

    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        let chipdb = ctx.chipdb();
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        let (dx, dy) = chipdb.tile_xy(dst.tile());
        let m = ((wx - dx).abs() + (wy - dy).abs()) as f32;
        (m * H_WEIGHT as f32 * self.h_scale * OVERUSE_COST_SCALE as f32) as DelayT
    }

    fn bboxes(&self) -> &[BoundingBox] {
        match self.bbox {
            Some(bb) => std::slice::from_ref(bb),
            None => &[],
        }
    }
}

fn route_sink_overuse(
    ctx: &Context,
    model: &OveruseCostModel,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    sink_visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    let opts = AStarOptions {
        visit_limit: sink_visit_limit,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: true,
    };
    let res = astar_search(ctx, model, src_wires, dst_wire, &opts);
    astar_probe::record(res.trace.exit, res.trace.visit_count, res.trace.max_visits);
    res.path
}

/// One net's planning result. `deadline_missing` counts the sinks skipped by
/// the wall-clock `net_deadline` rather than exhausted by the A*: that branch
/// makes a pass's outcome depend on machine load, so it has to be observable
/// before any A/B on this loop can be called deterministic.
struct PlanOutcome {
    plan: RoutePlan,
    missing: Vec<WireId>,
    deadline_missing: usize,
}

/// Plan one net under congestion pricing. Same tree-growth shape as
/// `route_net_raster`; returns the plan and the sinks that stayed missing
/// (visit cap or per-net deadline), which keep the net dirty next pass.
///
/// A sink in `missed` failed on an earlier pass: prices push its cheapest
/// path past what the manhattan heuristic predicts, so the default cap and
/// margin misreport it as unroutable. It retries with the cap doubled per
/// consecutive miss (≤8×) inside the cleanup-proven `max(60, 3·span)` bbox,
/// while every other sink keeps the cheap limits.
fn plan_net_overuse(
    ctx: &Context,
    net: NetId,
    source_wire: WireId,
    nt: &NodeTable,
    occ: &[u16],
    hist: &[f32],
    reserved: &FxHashMap<WireId, NetId>,
    present: f32,
    missed: &FxHashMap<(NetId, WireId), u8>,
    escalate: bool,
    h_scale: f32,
    cfg: &RasterRouterCfg,
) -> PlanOutcome {
    let chipdb = ctx.chipdb();
    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return PlanOutcome {
            plan: RoutePlan {
                net,
                source_wire,
                sink_routes: Vec::new(),
            },
            missing: Vec::new(),
            deadline_missing: 0,
        };
    }

    let bbox = if cfg.bbox_margin > 0 {
        Some(crate::metrics::compute_bbox(ctx, net, cfg.bbox_margin))
    } else {
        None
    };
    // Built lazily: most nets have no escalated sink, and the wide bbox is
    // only correct to reuse because it depends on the net, not the sink.
    let mut wide_bbox: Option<BoundingBox> = None;

    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);

    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());
    let mut remaining_sinks: Vec<(WireId, i32, i32)> = sink_wires
        .iter()
        .map(|&w| {
            let (sx, sy) = chipdb.tile_xy(w.tile());
            (w, sx, sy)
        })
        .collect();
    remaining_sinks.sort_by_key(|&(_, sx, sy)| (sx - src_x).abs() + (sy - src_y).abs());

    let n_sinks = remaining_sinks.len();
    let net_deadline =
        std::time::Instant::now() + std::time::Duration::from_millis(2000 + n_sinks as u64 * 10);

    let mut sink_routes = Vec::new();
    let mut missing: Vec<WireId> = Vec::new();
    let mut deadline_missing = 0usize;
    for (sink_wire, _, _) in remaining_sinks {
        if std::time::Instant::now() > net_deadline {
            missing.push(sink_wire);
            deadline_missing += 1;
            continue;
        }
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }
        let level = if escalate {
            missed.get(&(net, sink_wire)).copied().unwrap_or(0)
        } else {
            0
        };
        let (sink_bbox, sink_limit) = if level > 0 {
            if wide_bbox.is_none() {
                let bel_bbox = crate::metrics::compute_bbox(ctx, net, 0);
                let span = (bel_bbox.x1 - bel_bbox.x0)
                    .max(bel_bbox.y1 - bel_bbox.y0)
                    .max(0);
                let margin = EXPLORE_BBOX_MARGIN_MIN.max(span * EXPLORE_BBOX_SPAN_MULT);
                wide_bbox = Some(crate::metrics::compute_bbox(ctx, net, margin));
            }
            (
                wide_bbox.as_ref(),
                cfg.sink_visit_limit.map(|v| v << level.min(3)),
            )
        } else {
            (bbox.as_ref(), cfg.sink_visit_limit)
        };
        let model = OveruseCostModel {
            net,
            bbox: sink_bbox,
            nt,
            occ,
            hist,
            reserved,
            present,
            h_scale,
        };
        match route_sink_overuse(ctx, &model, &tree_wires, sink_wire, sink_limit) {
            Some(pips) => {
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    tree_wires.insert(dst);
                }
                tree_wires.insert(sink_wire);
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => missing.push(sink_wire),
        }
    }

    PlanOutcome {
        plan: RoutePlan {
            net,
            source_wire,
            sink_routes,
        },
        missing,
        deadline_missing,
    }
}

/// PathFinder pass loop. Plans live in router-owned state, wires may be
/// shared, and each pass replans exactly the nets touching shared wires
/// under prices that grow until sharing stops paying. Convergence commits
/// every plan through the atomic `apply_route_plan`; const nets, clock nets
/// and anything negotiation could not finish then go through the exclusive
/// path, whose cleanup stage owns their special-case routing.
fn route_nets_overuse(
    ctx: &mut Context,
    cfg: &RasterRouterCfg,
    nets: &[NetId],
) -> Result<(), RouterError> {
    let env_f32 = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .map(|v| {
                v.parse()
                    .unwrap_or_else(|_| panic!("{name} must be a float"))
            })
            .unwrap_or(default)
    };
    let present0 = env_f32("NPNR_ROUTE_PRESENT", 0.5);
    let growth = env_f32("NPNR_ROUTE_PRESENT_GROWTH", 1.4);
    let hist_gain = env_f32("NPNR_ROUTE_HIST_GAIN", 1.0);
    // Escalated retries only pay once contention is low enough for them to
    // succeed: ungated, 40k escalated sinks retried expensively every
    // mid-game pass and mostly failed (commit 7d4ee4b), 2.5x the wall for
    // 20% of the missing.
    let esc_gate = env_f32("NPNR_ROUTE_ESC_GATE", 1_500_000.0) as usize;
    // Heuristic scale per unit of present price. Prices inflate g while the
    // manhattan h estimates base cost only, so search width grows with the
    // price level; scaling h back restores the pruning at the cost of extra
    // inadmissibility on top of H_WEIGHT's.
    let h_price = env_f32("NPNR_ROUTE_H_PRICE", 0.5);
    // Wall budget for the negotiation loop. On expiry the loop exits and
    // commits best-effort, so every run yields an end-to-end unrouted count
    // comparable with the exclusive baseline instead of dying mid-pass.
    let neg_wall = std::time::Duration::from_secs_f32(env_f32("NPNR_ROUTE_NEG_WALL", 1800.0));
    // Parallel planning against occupancy frozen at pass start. Plans never
    // bind ctx in this mode, so the historical objection (60% bind-conflict
    // rejections) is structurally void; the cost is in-pass staleness —
    // nets stop seeing each other's same-pass choices — which negotiation
    // absorbs as slightly higher over-use per pass.
    let par_plan = std::env::var("NPNR_ROUTE_PAR").ok().as_deref() == Some("1");
    // Present is a within-pass separator, not the long-run one: uncapped it
    // compounded to 78 by pass 15 and froze the system (every wire
    // astronomical, relative differences gone, over-use RISING). History
    // carries the durable separation, so present saturates here.
    let present_max = env_f32("NPNR_ROUTE_PRESENT_MAX", 8.0);
    // Fraction of dirty nets replanned per pass. Frozen-cost planning makes
    // colliding nets move in lockstep — both sides of a shared wire detour
    // identically and re-collide; replanning only a (net, pass)-hashed
    // subset breaks the duel asymmetrically.
    let par_frac = env_f32("NPNR_ROUTE_PAR_FRAC", 0.5);
    // Without decay, chronic wires accumulate history without bound: real
    // path costs outrun any fixed heuristic scale (pass cost compounded 25s
    // -> 115s by pass 40) and stale congestion stays repellent after the
    // contention has moved. Decayed each pass before the new excess lands.
    let hist_decay = env_f32("NPNR_ROUTE_HIST_DECAY", 0.9);
    let h_max = env_f32("NPNR_ROUTE_H_MAX", 6.0);
    // Chunk size for parallel planning. Whole-pass frozen costs drain
    // over-use at ~0.3%/pass because colliding nets never see each other
    // move; the sequential regime's 20-50%/pass drain is intra-pass
    // visibility. Chunks bound the staleness to chunk_size nets: usage
    // applies between chunks, so later chunks price earlier chunks' moves.
    let par_chunk = env_f32("NPNR_ROUTE_PAR_CHUNK", 4096.0) as usize;
    // Asymmetric conflict resolution. Both owners of a contested node used to
    // replan every pass, so both fled and both re-collided elsewhere: over-use
    // oscillated without progress, and 67% of it was plain pairs (occ=2)
    // smeared over 83% of the tiles. Here nets claim their wires in a fixed
    // order and only the net that finds a wire already claimed yields, so each
    // contested node keeps an incumbent and sheds exactly one occupant.
    let single_mover = std::env::var("NPNR_ROUTE_SINGLE_MOVER").ok().as_deref() == Some("1");

    let mut nt = NodeTable::new();
    let mut occ: Vec<u16> = Vec::new();
    let mut hist: Vec<f32> = Vec::new();
    eprintln!(
        "RasterRouter[overuse]: {} nets, present={} growth={} hist_gain={}",
        nets.len(),
        present0,
        growth,
        hist_gain,
    );

    // Pin reservations mirror the exclusive pre-reserve but stay in router
    // state: nothing binds into ctx until commit.
    let mut reserved: FxHashMap<WireId, NetId> = FxHashMap::default();
    let mut reservation_conflicts = 0usize;
    for &net in nets {
        let n = ctx.net(net);
        if !n.is_alive() || !n.has_driver() {
            continue;
        }
        let mut pins: Vec<WireId> = Vec::new();
        if let Some(driver) = n.driver_cell_port() {
            if let Some(bel) = ctx.cell(driver.cell).bel() {
                if let Some(w) = bel.pin_wire(driver.port) {
                    pins.push(w.id());
                }
            }
        }
        for user in n.users() {
            if !user.is_valid() {
                continue;
            }
            if let Some(ubel) = ctx.cell(user.cell).bel() {
                if let Some(uw) = ubel.pin_wire(user.port) {
                    pins.push(uw.id());
                }
            }
        }
        for w in pins {
            match reserved.entry(w) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(net);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if *e.get() != net {
                        reservation_conflicts += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "RasterRouter[overuse]: {} reserved pin wires ({} conflicts)",
        reserved.len(),
        reservation_conflicts,
    );

    // Const and global nets never route through the priced A* — the const
    // fast path and the whole-chip global search live in the exclusive
    // cleanup — so they go straight to the finisher.
    let mut ordinary: Vec<NetId> = Vec::new();
    for &net in nets {
        if let Ok(Some(w)) = resolve_source_wire(ctx, net) {
            if source_wire_const_value(ctx, w) == 0 && !is_global_net(ctx, net) {
                ordinary.push(net);
            }
        }
    }
    // High fanout first, then net id: identical ordering intent to the
    // exclusive pass, and fully deterministic.
    ordinary.sort_by_key(|&n| (usize::MAX - ctx.net(n).num_users(), n));

    struct PlanEntry {
        plan: RoutePlan,
        idxs: Vec<u32>,
        missing: usize,
    }
    let mut plans: FxHashMap<NetId, PlanEntry> = FxHashMap::default();
    // (net, sink) -> consecutive misses; drives the escalated retry and is
    // cleared the pass the sink routes.
    let mut missed: FxHashMap<(NetId, WireId), u8> = FxHashMap::default();
    let mut prev_overused = usize::MAX;
    let mut last_pass_secs = 0f64;
    let mut mean_pen = 0f32;
    let mut present = present0;
    let t_all = std::time::Instant::now();
    // Fixed-point diagnostics. The pass line reports the residual's size and
    // shape; these report who produces it — whether the same nets yield every
    // pass, what they collide with, and whether replanning moves them at all.
    let mut prev_movers: FxHashSet<NetId> = FxHashSet::default();
    let mut yield_streak: FxHashMap<NetId, u32> = FxHashMap::default();

    for iter in 0..cfg.max_iterations {
        // Claims survive the pass so the diagnostic can separate mover-vs-
        // incumbent collisions from mover-vs-mover ones; empty outside
        // single-mover mode, where no net holds a claim.
        let mut claimed: Vec<u64> = Vec::new();
        let mut mover_contested = 0usize;
        let mut mover_missing = 0usize;
        let mut mover_noplan = 0usize;
        let mut plan_unchanged = 0usize;
        let mut deadline_missing_total = 0usize;
        let dirty: Vec<NetId> = if iter == 0 {
            ordinary.clone()
        } else if single_mover {
            // One sequential claim sweep in net order; a bitset over the
            // flat wire space keeps it O(total plan wires).
            claimed = vec![0u64; (nt.len() + 63) / 64];
            let mut movers: Vec<NetId> = Vec::new();
            for &n in &ordinary {
                let Some(e) = plans.get(&n) else {
                    mover_noplan += 1;
                    movers.push(n);
                    continue;
                };
                let contested = e.idxs.iter().any(|&i| {
                    let i = i as usize;
                    claimed[i >> 6] & (1u64 << (i & 63)) != 0
                });
                if contested {
                    mover_contested += 1;
                    movers.push(n);
                } else if e.missing > 0 {
                    mover_missing += 1;
                    movers.push(n);
                } else {
                    for &i in &e.idxs {
                        let i = i as usize;
                        claimed[i >> 6] |= 1u64 << (i & 63);
                    }
                }
            }
            movers
        } else {
            ordinary
                .iter()
                .copied()
                .filter(|n| match plans.get(n) {
                    Some(e) => e.missing > 0 || e.idxs.iter().any(|&i| occ[i as usize] > 1),
                    None => true,
                })
                .collect()
        };
        // Convergence is judged on the full dirty set; the fraction filter
        // below only spreads its replanning over passes.
        if dirty.is_empty() {
            break;
        }
        let to_route: Vec<NetId> = if par_plan && par_frac < 1.0 && iter > 0 {
            dirty
                .into_iter()
                .filter(|n| {
                    use std::hash::{Hash, Hasher};
                    let mut h = rustc_hash::FxHasher::default();
                    (*n, iter).hash(&mut h);
                    (h.finish() % 1000) < (par_frac * 1000.0) as u64
                })
                .collect()
        } else {
            dirty
        };
        if to_route.is_empty() {
            // The hash left this pass's subset empty; spin the pass cheaply.
            present = (present * growth).min(present_max);
            continue;
        }

        if t_all.elapsed() >= neg_wall {
            eprintln!(
                "RasterRouter[overuse]: negotiation wall budget spent at pass {}, committing",
                iter
            );
            break;
        }
        // Escalation reinflates over-use by design (readmitted demand), so
        // committing straight after an escalated pass maximises conflicts
        // (33430 on the 8881045 run). With under ~2.5 pass-times of wall
        // left, escalation yields to cheap deflation passes so commit lands
        // on a settled state.
        let remaining = neg_wall.as_secs_f64() - t_all.elapsed().as_secs_f64();
        // Sequential-only: in the parallel regime the 2.5x window flip-flops
        // as escalated and cheap passes alternate, bouncing recoveries in and
        // out, and the finisher absorbs commit conflicts anyway.
        let cooldown = !par_plan && last_pass_secs > 0.0 && remaining < 2.5 * last_pass_secs;
        if cooldown && prev_overused < esc_gate {
            eprintln!(
                "RasterRouter[overuse]: {:.0}s wall left, escalation cooling down for commit",
                remaining
            );
        }
        let escalate = prev_overused < esc_gate && !cooldown;
        // The ceiling is the negotiation-blindness bound: at h_scale ~14 the
        // par4 run froze — searches beeline, detours are invisible to the
        // heuristic, and over-use sat at 2.78M for fifteen flat passes with
        // every price signal screaming. Width costs pops; blindness costs
        // convergence.
        let h_scale = (1.0 + h_price * (present + mean_pen)).min(h_max);
        let t_pass = std::time::Instant::now();
        if par_plan {
            // Chunked parallel pass: within a chunk workers price against
            // frozen state, between chunks usage applies, so a chunk is the
            // staleness bound. Application stays in input order per chunk,
            // keeping the pass deterministic.
            use rayon::prelude::*;
            for chunk_nets in to_route.chunks(par_chunk.max(1)) {
                for &net in chunk_nets {
                    if let Some(e) = plans.get(&net) {
                        for &wi in &e.idxs {
                            occ[wi as usize] = occ[wi as usize].saturating_sub(1);
                        }
                    }
                }
                let ctx_ref: &Context = ctx;
                let occ_ref: &[u16] = &occ;
                let missed_ref = &missed;
                let nt_ref: &NodeTable = &nt;
                let hist_ref: &[f32] = &hist;
                let results: Vec<(NetId, PlanOutcome)> = chunk_nets
                    .par_iter()
                    .filter_map(|&net| {
                        let source_wire = resolve_source_wire(ctx_ref, net).ok().flatten()?;
                        let outcome = plan_net_overuse(
                            ctx_ref,
                            net,
                            source_wire,
                            nt_ref,
                            occ_ref,
                            hist_ref,
                            &reserved,
                            present,
                            missed_ref,
                            escalate,
                            h_scale,
                            cfg,
                        );
                        Some((net, outcome))
                    })
                    .collect();
                for (net, outcome) in results {
                    // Interning is sequential and in input order, so node
                    // indices stay deterministic across runs.
                    let idxs = plan_wire_indices(ctx, &mut nt, &mut occ, &mut hist, &outcome.plan);
                    let PlanOutcome {
                        plan,
                        missing: missing_list,
                        deadline_missing,
                    } = outcome;
                    deadline_missing_total += deadline_missing;
                    if plans.get(&net).is_some_and(|e| e.idxs == idxs) {
                        plan_unchanged += 1;
                    }
                    for &wi in &idxs {
                        occ[wi as usize] = occ[wi as usize].saturating_add(1);
                    }
                    for sr in &plan.sink_routes {
                        if let std::collections::hash_map::Entry::Occupied(mut e) =
                            missed.entry((net, sr.sink_wire))
                        {
                            let l = *e.get();
                            if l <= 1 {
                                e.remove();
                            } else {
                                e.insert(l - 1);
                            }
                        }
                    }
                    for &s in &missing_list {
                        let e = missed.entry((net, s)).or_insert(0);
                        *e = (*e + 1).min(8);
                    }
                    let missing = missing_list.len();
                    plans.insert(
                        net,
                        PlanEntry {
                            plan,
                            idxs,
                            missing,
                        },
                    );
                }
            }
        } else {
            let mut last_progress = t_pass;
            for (i, &net) in to_route.iter().enumerate() {
                if let Some(e) = plans.get(&net) {
                    for &wi in &e.idxs {
                        occ[wi as usize] = occ[wi as usize].saturating_sub(1);
                    }
                }
                let source_wire = match resolve_source_wire(ctx, net) {
                    Ok(Some(w)) => w,
                    _ => continue,
                };
                let PlanOutcome {
                    plan,
                    missing: missing_list,
                    deadline_missing,
                } = plan_net_overuse(
                    ctx,
                    net,
                    source_wire,
                    &nt,
                    &occ,
                    &hist,
                    &reserved,
                    present,
                    &missed,
                    escalate,
                    h_scale,
                    cfg,
                );
                deadline_missing_total += deadline_missing;
                let idxs = plan_wire_indices(ctx, &mut nt, &mut occ, &mut hist, &plan);
                if plans.get(&net).is_some_and(|e| e.idxs == idxs) {
                    plan_unchanged += 1;
                }
                for &wi in &idxs {
                    occ[wi as usize] = occ[wi as usize].saturating_add(1);
                }
                // Decay, not clear: a sink that recovered at an escalated budget
                // reverts to the cheap cap next replan if its entry vanishes, and
                // re-strands under the now-higher prices — the pass-4→5 sawtooth
                // (26415→45325 missing) was exactly the recovered population.
                for sr in &plan.sink_routes {
                    if let std::collections::hash_map::Entry::Occupied(mut e) =
                        missed.entry((net, sr.sink_wire))
                    {
                        let l = *e.get();
                        if l <= 1 {
                            e.remove();
                        } else {
                            e.insert(l - 1);
                        }
                    }
                }
                for &s in &missing_list {
                    let e = missed.entry((net, s)).or_insert(0);
                    *e = (*e + 1).min(8);
                }
                let missing = missing_list.len();
                plans.insert(
                    net,
                    PlanEntry {
                        plan,
                        idxs,
                        missing,
                    },
                );
                if cfg.verbose
                    && ((i + 1) % 5000 == 0
                        || last_progress.elapsed() >= std::time::Duration::from_secs(60))
                {
                    last_progress = std::time::Instant::now();
                    let secs = t_pass.elapsed().as_secs_f64();
                    eprintln!(
                        "  neg pass {} progress: {}/{} nets, {:.1}s, {:.2} nets/s  astar:{}",
                        iter,
                        i + 1,
                        to_route.len(),
                        secs,
                        (i + 1) as f64 / secs,
                        astar_probe::summary(),
                    );
                }
            }
        }

        let mut overused = 0usize;
        let mut max_occ = 0u16;
        let mut pen_sum = 0f64;
        let mut pen_cnt = 0usize;
        // Structure of the residual, not just its size: occ=2 smeared over
        // the chip is a pricing failure, occ>=4 concentrated in a few tiles
        // is a capacity wall, and the two want opposite fixes.
        let mut occ2 = 0usize;
        let mut occ3 = 0usize;
        let mut occ4p = 0usize;
        let mut tile_over: FxHashMap<i32, u32> = FxHashMap::default();
        for i in 0..nt.len() {
            if hist[i] > 0.0 {
                hist[i] *= hist_decay;
                if hist[i] < 0.01 {
                    hist[i] = 0.0;
                }
            }
            let o = occ[i];
            if o > 0 {
                // The mean penalty a route actually pays per used wire; the
                // next pass's heuristic scales by it, so search width tracks
                // the real price level (present AND history) instead of a
                // stale proxy frozen at the present cap.
                pen_sum += (present * o as f32 + hist[i]) as f64;
                pen_cnt += 1;
            }
            if o > 1 {
                overused += 1;
                match o {
                    2 => occ2 += 1,
                    3 => occ3 += 1,
                    _ => occ4p += 1,
                }
                *tile_over.entry(nt.tile_of(i)).or_insert(0) += 1;
                hist[i] += (o - 1) as f32 * hist_gain;
                if o > max_occ {
                    max_occ = o;
                }
            }
        }
        // Concentration: what share of the over-use sits in the worst 1% of
        // tiles that carry any. Near 1% means smeared, near 50% means a wall.
        let mut tile_counts: Vec<u32> = tile_over.values().copied().collect();
        tile_counts.sort_unstable_by(|a, b| b.cmp(a));
        let top1 = (tile_counts.len() / 100).max(1);
        let top_share = if overused > 0 {
            tile_counts[..top1.min(tile_counts.len())]
                .iter()
                .map(|&c| c as usize)
                .sum::<usize>() as f64
                / overused as f64
        } else {
            0.0
        };
        let tiles_touched = tile_counts.len();
        mean_pen = if pen_cnt > 0 {
            (pen_sum / pen_cnt as f64) as f32
        } else {
            0.0
        };
        let missing_total: usize = plans.values().map(|e| e.missing).sum();
        eprintln!(
            "RasterRouter[overuse] pass {}: replanned={} overused_wires={} max_occ={} missing_sinks={} escalated={} esc_on={} present={:.2} mean_pen={:.2} occ[2={} 3={} 4+={}] tiles={} top1%={:.1}% {:.1}s (total {:.1}s)",
            iter,
            to_route.len(),
            overused,
            max_occ,
            missing_total,
            missed.len(),
            escalate,
            present,
            mean_pen,
            occ2,
            occ3,
            occ4p,
            tiles_touched,
            top_share * 100.0,
            t_pass.elapsed().as_secs_f64(),
            t_all.elapsed().as_secs_f64(),
        );
        // A mover blocked by a claimed wire is blocked by an incumbent that
        // will not move this pass; a mover on an unclaimed shared wire is
        // blocked by another mover. The two want opposite fixes, so they are
        // counted apart.
        let mut mover_on_claimed = 0usize;
        let mut mover_on_shared = 0usize;
        for &n in &to_route {
            let Some(e) = plans.get(&n) else { continue };
            for &i in &e.idxs {
                let i = i as usize;
                // Plans made this pass can name nodes interned after the
                // claim sweep sized the bitset; those are unclaimed.
                if claimed
                    .get(i >> 6)
                    .is_some_and(|word| word & (1u64 << (i & 63)) != 0)
                {
                    mover_on_claimed += 1;
                } else if occ[i] > 1 {
                    mover_on_shared += 1;
                }
            }
        }
        let overlap_prev = to_route.iter().filter(|n| prev_movers.contains(n)).count();
        prev_movers = to_route.iter().copied().collect();
        yield_streak.retain(|n, _| prev_movers.contains(n));
        for &n in &to_route {
            *yield_streak.entry(n).or_insert(0) += 1;
        }
        let mut streak = [0usize; 4];
        for &v in yield_streak.values() {
            let b = match v {
                0..=1 => 0,
                2..=4 => 1,
                5..=14 => 2,
                _ => 3,
            };
            streak[b] += 1;
        }
        // Fanout of the nets sitting on the occ>=4 nodes that now dominate the
        // residual: if a few high-fanout trees own them, fanout-first
        // incumbency is the wrong claim order for exactly those nodes.
        let mut occ4_fanout = [0usize; 5];
        for e in plans.values() {
            let b = match ctx.net(e.plan.net).num_users() {
                0..=1 => 0,
                2..=4 => 1,
                5..=16 => 2,
                17..=64 => 3,
                _ => 4,
            };
            for &i in &e.idxs {
                if occ[i as usize] >= 4 {
                    occ4_fanout[b] += 1;
                }
            }
        }
        let pct = |a: usize, b: usize| {
            if b > 0 {
                100.0 * a as f64 / b as f64
            } else {
                0.0
            }
        };
        eprintln!(
            "RasterRouter[overuse] diag pass {}: movers[contested={} missing={} noplan={}] overlap_prev={:.1}% unchanged={:.1}% mover_wires[claimed={} shared={}] streak[1={} 2-4={} 5-14={} 15+={}] occ4_fanout[1={} 2-4={} 5-16={} 17-64={} 65+={}] deadline_miss={}",
            iter,
            mover_contested,
            mover_missing,
            mover_noplan,
            pct(overlap_prev, to_route.len()),
            pct(plan_unchanged, to_route.len()),
            mover_on_claimed,
            mover_on_shared,
            streak[0],
            streak[1],
            streak[2],
            streak[3],
            occ4_fanout[0],
            occ4_fanout[1],
            occ4_fanout[2],
            occ4_fanout[3],
            occ4_fanout[4],
            deadline_missing_total,
        );
        prev_overused = overused;
        last_pass_secs = t_pass.elapsed().as_secs_f64();
        present = (present * growth).min(present_max);

        if overused == 0 && missing_total == 0 {
            eprintln!(
                "RasterRouter[overuse]: negotiation converged at pass {}",
                iter
            );
            break;
        }
    }

    // Commit. Plans are consistent when negotiation converged. When the wall
    // budget cut it short, nets whose plans touch no shared wire cannot
    // conflict with each other and commit first; the still-contended rest
    // commit greedily and the losers fall through to the finisher unbound.
    let mut commit_order: Vec<NetId> = ordinary.clone();
    commit_order.sort_by_key(|n| {
        let dirty = plans
            .get(n)
            .map(|e| e.idxs.iter().any(|&i| occ[i as usize] > 1))
            .unwrap_or(true);
        (dirty, *n)
    });
    let mut committed = 0usize;
    let mut commit_conflicts = 0usize;
    let mut commit_partial = 0usize;
    for &net in &commit_order {
        let Some(e) = plans.get(&net) else { continue };
        if !e.plan.source_wire.is_valid() || e.plan.sink_routes.is_empty() {
            continue;
        }
        match apply_route_plan(ctx, &e.plan) {
            Ok(()) => {
                committed += 1;
                if e.missing > 0 {
                    commit_partial += 1;
                }
            }
            Err(_) => commit_conflicts += 1,
        }
    }
    eprintln!(
        "RasterRouter[overuse]: committed={} ({} partial) conflicts={} {:.1}s total",
        committed,
        commit_partial,
        commit_conflicts,
        t_all.elapsed().as_secs_f64(),
    );

    // Finisher: const, clock and every net negotiation left unfinished, on
    // the exclusive path whose cleanup owns their routing.
    let finish: Vec<NetId> = nets
        .iter()
        .copied()
        .filter(|&n| {
            let net = ctx.net(n);
            net.is_alive() && net.has_driver() && net.num_users() > 0 && !net_fully_routed(ctx, n)
        })
        .collect();
    if finish.is_empty() {
        return validate_all_routed(ctx);
    }
    eprintln!(
        "RasterRouter[overuse]: {} nets to the exclusive finisher",
        finish.len()
    );
    let mut fcfg = cfg.clone();
    fcfg.overuse = false;
    route_nets_exclusive(ctx, &fcfg, &finish)
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
        if cfg.overuse {
            route_nets_overuse(ctx, cfg, nets)
        } else {
            route_nets_exclusive(ctx, cfg, nets)
        }
    }
}

/// The exclusive-binding pass loop: every wire has at most one owner, plans
/// bind as they apply, and recovery is the cleanup stage. Also the finisher
/// for the over-use path, which hands it only const nets, clock nets and
/// whatever negotiation left unrouted.
fn route_nets_exclusive(
    ctx: &mut Context,
    cfg: &RasterRouterCfg,
    nets: &[NetId],
) -> Result<(), RouterError> {
    {
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

        // (net, sink) pairs whose wide-bbox unbound explore drained: the
        // unbound graph and the placement are fixed, so these stay
        // unreachable for the whole run and are never explored again.
        let mut inloop_dead: FxHashSet<(NetId, WireId)> = FxHashSet::default();

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
                let mut partials_to_rip: Vec<NetId> = Vec::new();
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
                        if !empty && cfg.skip_cleanup {
                            // No cleanup stage to extend this partial, so the
                            // pass loop is its only remaining chance. Ripped up
                            // after this loop: `iter_net_indices` holds ctx
                            // immutably.
                            partials_to_rip.push(idx);
                            continue;
                        }
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
                // Usage is read off `net.wires()`, so release it before the
                // unroute, matching the congested loop above.
                for &idx in &partials_to_rip {
                    neg_state.remove_net_usage(&ctx.design, idx);
                    unroute_net(ctx, idx);
                    to_retry.push(idx);
                }
                if cfg.skip_cleanup {
                    eprintln!(
                        "  pass {} gate: {} partials ripped up",
                        iter,
                        partials_to_rip.len()
                    );
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
            let mut apply_conflicts = 0usize;
            let mut empty_plans = 0usize;
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
                            Err(_conflict) => {
                                apply_conflicts += 1;
                            }
                        }
                    } else {
                        empty_plans += 1;
                    }
                } else {
                    empty_plans += 1;
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

            // A pass over 105k low-fanout nets is sequential and can outlast an
            // hour, and the raster_fail print is capped at the first 100 ever,
            // so without this the pass is silent from net 100 to the end.
            // Cadence is wall time rather than net count alone: the pass rate is
            // only bounded above (105k nets unfinished in 5400s), and a count-only
            // trigger stays silent for the whole run if the rate is under 1 net/s.
            let t_lf = std::time::Instant::now();
            let mut last_progress = t_lf;
            for (i, &net) in lf_nets.iter().enumerate() {
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
                                apply_conflicts += 1;
                            }
                        }
                    }
                    _ => {
                        failed += 1;
                        empty_plans += 1;
                    }
                }
                if cfg.verbose
                    && ((i + 1) % 5000 == 0
                        || last_progress.elapsed() >= std::time::Duration::from_secs(60))
                {
                    last_progress = std::time::Instant::now();
                    let secs = t_lf.elapsed().as_secs_f64();
                    eprintln!(
                        "  pass {} progress: {}/{} lf nets, routed={}, failed={}, {:.1}s, {:.2} nets/s  astar:{}",
                        iter,
                        i + 1,
                        lf_nets.len(),
                        routed,
                        failed,
                        secs,
                        (i + 1) as f64 / secs,
                        astar_probe::summary(),
                    );
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
            eprintln!("  pass {} astar: {}", iter, astar_probe::summary());
            eprintln!(
                "  pass {} failmode: apply_conflicts={} empty_plans={}  astar:{}",
                iter,
                apply_conflicts,
                empty_plans,
                astar_probe::summary(),
            );
            eprintln!("  pass {} partials: {}", iter, partial_census::summary());

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
            // In-loop scoped rip-up: recover failed nets while the pass state
            // is live instead of leaving them to the cleanup stage, whose
            // chip-scale A* recovers 1 net in 98 at 8.1s each while 13 of its
            // 14 recoveries come from the scoped rip-up it gates behind that
            // A*. Each attempt either fully routes the net or rolls back to
            // the exact prior state, so partial trees survive a failed try.
            if cfg.inloop_ripup && !failed_nets.is_empty() {
                let wire_penalty: FxHashMap<WireId, crate::common::DelayT> = neg_state
                    .wire_history
                    .iter()
                    .map(|(&w, &h)| {
                        (
                            w,
                            (h * neg_state.cfg.history_cost_multiplier) as crate::common::DelayT,
                        )
                    })
                    .collect();
                // Twice the per-sink cap: the corridor was just emptied for
                // this net, so a success should land well under the pass
                // budget and anything beyond it is the doomed-search shape
                // the cap exists to stop.
                let reroute_limit = cfg.sink_visit_limit.map(|v| v * 2).unwrap_or(100_000);
                let mut victim_ledger: FxHashMap<NetId, u32> = FxHashMap::default();
                let mut stats = InloopStats::default();
                let t_sweep = std::time::Instant::now();
                let mut last_sweep_report = t_sweep;
                // Cheapest explores first: failures skew short-span, and the
                // wall budget below should be spent where attempts are
                // plentiful, not on a few wide-bbox stragglers.
                let mut sweep_targets: Vec<(NetId, i32)> = failed_nets
                    .iter()
                    .map(|&n| {
                        let bb = crate::metrics::compute_bbox(ctx, n, 0);
                        (n, (bb.x1 - bb.x0).max(bb.y1 - bb.y0).max(0))
                    })
                    .collect();
                sweep_targets.sort_by_key(|&(n, span)| (span, n));
                let mut sweep_visited = 0usize;
                for (i, &(net, _)) in sweep_targets.iter().enumerate() {
                    if t_sweep.elapsed().as_secs() >= INLOOP_SWEEP_WALL_SECS {
                        break;
                    }
                    sweep_visited = i + 1;
                    if cfg.verbose
                        && last_sweep_report.elapsed() >= std::time::Duration::from_secs(60)
                    {
                        last_sweep_report = std::time::Instant::now();
                        eprintln!(
                            "  inloop_ripup progress: {}/{} nets, nets_rec={}, sinks_rec={}, ripups={}, {:.1}s",
                            i,
                            sweep_targets.len(),
                            stats.nets_recovered,
                            stats.sinks_recovered,
                            stats.ripups,
                            t_sweep.elapsed().as_secs_f64(),
                        );
                    }
                    inloop_ripup_net(
                        ctx,
                        net,
                        cfg,
                        &wire_penalty,
                        &lookahead,
                        &mut neg_state,
                        &mut victim_ledger,
                        &mut inloop_dead,
                        reroute_limit,
                        &mut stats,
                    );
                }
                eprintln!(
                    "  inloop_ripup pass {}: visited={}/{} attempted={} nets_rec={} sinks_rec={} free_ext={} ripups={} {:.1}s skip[dead_cache={} fanout={} too_many={} ledger={}] explore[capped_tight={} wide_found={} capped_wide={} dead_short={} dead_mid={} dead_long={}] fail[extend={} victim={}]",
                    iter,
                    sweep_visited,
                    sweep_targets.len(),
                    stats.attempted,
                    stats.nets_recovered,
                    stats.sinks_recovered,
                    stats.free_extends,
                    stats.ripups,
                    t_sweep.elapsed().as_secs_f64(),
                    stats.skip_dead_cache,
                    stats.skip_fanout,
                    stats.skip_too_many,
                    stats.skip_ledger,
                    stats.capped_tight,
                    stats.wide_found,
                    stats.capped_wide,
                    stats.dead_short,
                    stats.dead_mid,
                    stats.dead_long,
                    stats.extend_fail,
                    stats.victim_fail,
                );
                failed_nets = ctx
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
                failed = failed_nets.len();
                // Ordinary nets had their in-loop chance; only the cleanup
                // stage's deterministic const fast path and whole-chip global
                // search still earn their cost.
                if !cfg.skip_cleanup {
                    failed_nets.retain(|&n| {
                        resolve_source_wire(ctx, n)
                            .ok()
                            .flatten()
                            .map(|w| source_wire_const_value(ctx, w) != 0 || is_global_net(ctx, n))
                            .unwrap_or(false)
                    });
                }
            }

            if cfg.skip_cleanup {
                // Ground truth, not the pass counter: `routed` counts a net
                // whose plan applied with sinks missing, so `failed` alone
                // undercounts by the number of partials.
                let empty_trees = failed_nets
                    .iter()
                    .filter(|&&n| ctx.net(n).wires().is_empty())
                    .count();
                failed = failed_nets.len();
                eprintln!(
                    "  skip_cleanup: {} unrouted ({} empty, {} partial)",
                    failed,
                    empty_trees,
                    failed - empty_trees,
                );
            } else if !failed_nets.is_empty() {
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
                    } else if is_global_net(ctx, n) {
                        1
                    } else {
                        2
                    }
                });

                if !failed_nets.is_empty() {
                    // Build wire penalty from negotiation state for A* cost.
                    let wire_penalty: FxHashMap<WireId, crate::common::DelayT> = neg_state
                        .wire_history
                        .iter()
                        .map(|(&w, &h)| {
                            (
                                w,
                                (h * neg_state.cfg.history_cost_multiplier)
                                    as crate::common::DelayT,
                            )
                        })
                        .collect();

                    eprintln!(
                        "RasterRouter: A* cleanup for {} failed nets (iter {})",
                        failed_nets.len(),
                        iter,
                    );

                    let mut cleanup_routed = 0usize;
                    // The loop below ran 73 minutes on FPGA01's 41560 failed
                    // nets without printing anything, so its rate was unknown.
                    let mut cleanup_ripups = 0usize;
                    let t_cleanup = std::time::Instant::now();
                    let mut last_cleanup_report = t_cleanup;

                    for (cleanup_i, &net) in failed_nets.iter().enumerate() {
                        if cfg.verbose
                            && last_cleanup_report.elapsed() >= std::time::Duration::from_secs(60)
                        {
                            last_cleanup_report = std::time::Instant::now();
                            let secs = t_cleanup.elapsed().as_secs_f64();
                            eprintln!(
                                "  cleanup progress: {}/{} nets, routed={}, ripups={}, {:.1}s, {:.2} nets/s",
                                cleanup_i,
                                failed_nets.len(),
                                cleanup_routed,
                                cleanup_ripups,
                                secs,
                                cleanup_i as f64 / secs,
                            );
                        }
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
                        // Seed the partial-routed tree so cleanup A* can reach
                        // unrouted sinks via the net's existing pip chain.
                        for &w in ctx.net(net).wires().keys() {
                            tree_wires.insert(w);
                        }

                        // A global net leaves its buffer on dedicated routing
                        // that can run far outside the sinks' box, so its
                        // searches get the whole chip (below). It starts from
                        // its own driver, so every route traces back to it.
                        let global = is_global_net(ctx, net);

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
                            // span. The bbox is finite but not small: FPGA01
                            // cleanup FAIL lines report bbox=[0,310]x[0,222],
                            // the whole chip, so the derived budget reaches the
                            // full 693k pops per search. `cleanup_visit_limit`
                            // overrides it; `None` keeps the derived default.
                            let cleanup_margin =
                                EXPLORE_BBOX_MARGIN_MIN.max(manhattan * EXPLORE_BBOX_SPAN_MULT);
                            // A global net's dedicated routing can leave the
                            // sinks' box far behind (a buffer at the chip's
                            // centre feeding one corner), so it searches the
                            // whole chip.
                            let cleanup_bbox = if global {
                                BoundingBox {
                                    x0: 0,
                                    y0: 0,
                                    x1: ctx.chipdb().width() - 1,
                                    y1: ctx.chipdb().height() - 1,
                                }
                            } else {
                                crate::metrics::compute_bbox(ctx, net, cleanup_margin)
                            };
                            let per_sink_limit: Option<usize> = cfg.cleanup_visit_limit;

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
                                    // Once per node: history is keyed by node, and a
                                    // node crossing several corridor tiles is still
                                    // one conductor.
                                    let mut corridor_nodes: FxHashSet<WireId> =
                                        FxHashSet::default();
                                    for &tile_idx in &corridor_tiles {
                                        let tile = tile_idx as i32;
                                        let num_wires =
                                            ctx.chipdb().tile_type(tile).wires.get().len();
                                        for wi in 0..num_wires {
                                            let w = ctx
                                                .chipdb()
                                                .canonical_wire(WireId::new(tile, wi as i32));
                                            if corridor_nodes.insert(w) {
                                                *neg_state.wire_history.entry(w).or_insert(0.0) +=
                                                    0.5;
                                            }
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
                                cleanup_ripups += 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::common::PlaceStrength;

    /// The synthetic chipdb gives each tile one pip, W0 -> W1, and no
    /// inter-tile nodes, so tile 0's W1 is reachable from its W0 by exactly
    /// one hop and by no other route.
    const W0: WireId = WireId::new(0, 0);
    const W1: WireId = WireId::new(0, 1);

    fn ctx_with_two_nets() -> (Context, NetId, NetId) {
        let mut ctx = Context::new(make_test_chipdb());
        let owner_name = ctx.id("owner");
        let intruder_name = ctx.id("intruder");
        let owner = ctx.design.add_net(owner_name);
        let intruder = ctx.design.add_net(intruder_name);
        (ctx, owner, intruder)
    }

    fn route_w0_to_w1(ctx: &Context, net: NetId) -> Option<Vec<PipId>> {
        let cong = CongestionMap::new(ctx.chipdb().width(), ctx.chipdb().height());
        let mut src = FxHashSet::default();
        src.insert(W0);
        route_sink_astar(ctx, net, &src, W1, None, &cong, 0.0, Some(10_000)).0
    }

    #[test]
    fn routes_to_a_free_sink_wire() {
        let (ctx, _owner, intruder) = ctx_with_two_nets();
        let pips = route_w0_to_w1(&ctx, intruder).expect("W0 -> W1 is one pip");
        assert_eq!(pips.len(), 1);
    }

    /// The search prices a foreign-bound wire at `DelayT::MAX / 4` rather
    /// than excluding it, so it can return a plan `apply_route_plan` must
    /// reject. Excluding it via `PathCostModel::is_blocked` was built and
    /// measured: FPGA01 pass 0 kept identical search sizes (177852 sinks
    /// reached at 3217 pops each, 83703 capped) and identical outcomes
    /// (82069 routed, 41560 unrouted) while wall time rose 1005s -> 1175s,
    /// because every candidate edge then pays a `wire_binding` lookup on top
    /// of the one `wire_penalty` already does. The rejected plans it would
    /// have saved do not exist: `apply_conflicts` is 0 on both FPGA01 and
    /// stereovision3. This test pins the behaviour so the exclusion is not
    /// reintroduced as an obvious cleanup.
    #[test]
    fn prices_a_foreign_bound_wire_rather_than_excluding_it() {
        let (mut ctx, owner, intruder) = ctx_with_two_nets();
        ctx.try_bind_wire_node(W1, owner, PlaceStrength::Strong)
            .expect("W1 is free before the test binds it");
        assert!(route_w0_to_w1(&ctx, intruder).is_some());
    }

    /// The net's own wires stay usable: a partial tree is extended through
    /// the wires it already holds.
    #[test]
    fn plans_through_its_own_bound_wire() {
        let (mut ctx, _owner, intruder) = ctx_with_two_nets();
        ctx.try_bind_wire_node(W1, intruder, PlaceStrength::Strong)
            .expect("W1 is free before the test binds it");
        assert!(route_w0_to_w1(&ctx, intruder).is_some());
    }
}

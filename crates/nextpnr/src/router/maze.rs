//! Router1: A* rip-up and reroute router.
//!
//! Implements an A* routing algorithm with estimate-based pruning (matching
//! the C++ nextpnr router1 approach). Routes each net independently, then
//! detects congestion and rips up congested nets for rerouting with increased
//! penalties.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    find_congested_wires, is_global_clock_pip, resolve_source_wire, source_wire_const_value,
    unroute_net, RoutePlan, SinkRoute,
};
use super::RouterError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the Router1 algorithm.
#[derive(Debug, Clone)]
pub struct Router1Cfg {
    /// Maximum number of rip-up-and-reroute iterations.
    pub max_iterations: usize,
    /// Penalty added to a wire each time it is involved in congestion.
    pub rip_up_penalty: DelayT,
    /// Weight multiplier for congestion cost.
    pub congestion_weight: f64,
    /// Margin (in tiles) added around each net's bounding box for A* pruning.
    /// Set to 0 to disable bounding-box pruning.
    pub bb_margin: i32,
    /// Precision slack for estimate-based pruning. Candidates within this
    /// margin of the best estimate are kept. Higher = more exploration.
    pub estimate_precision: DelayT,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for Router1Cfg {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            rip_up_penalty: 10,
            congestion_weight: 1.0,
            bb_margin: 3,
            estimate_precision: 50,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// A* priority queue entry
// ---------------------------------------------------------------------------

/// An entry in the A* search priority queue.
#[derive(Clone)]
pub struct QueueEntry {
    pub wire: WireId,
    /// g(n): accumulated cost from the source.
    pub cost: DelayT,
    /// Penalty component of the cost (congestion penalties).
    pub penalty: DelayT,
    /// f(n) = g(n) + h(n): total estimated cost.
    pub estimate: DelayT,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by estimate, break ties by lower cost.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
    }
}

// ---------------------------------------------------------------------------
// Router state
// ---------------------------------------------------------------------------

/// Internal mutable state for the Router1 algorithm.
pub struct Router1State {
    pub wire_penalty: FxHashMap<WireId, DelayT>,
    pub wire_usage: FxHashMap<WireId, u32>,
    pub wire_nets: FxHashMap<WireId, FxHashSet<NetId>>,
}

impl Router1State {
    pub fn new() -> Self {
        Self {
            wire_penalty: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
            wire_nets: FxHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub struct Router1;

impl super::Router for Router1 {
    type Config = Router1Cfg;

    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::RouterError> {
        let nets = collect_routable_nets(ctx);
        self.route_nets(ctx, cfg, &nets)
    }

    fn route_net(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        net: crate::netlist::NetId,
    ) -> Result<(), super::RouterError> {
        let wire_penalty = FxHashMap::default();
        let plan = compute_route_r1(
            ctx,
            net,
            &wire_penalty,
            cfg.bb_margin,
            cfg.estimate_precision,
            None,
        )?;
        if plan.source_wire.is_valid() {
            let _ = super::common::apply_route_plan(ctx, &plan);
        }
        Ok(())
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), super::RouterError> {
        use rayon::prelude::*;

        let mut state = Router1State::new();

        // Build lookahead table for A* heuristic.
        let lookahead = super::lookahead::Lookahead::build(ctx.chipdb(), ctx.speed_grade_idx(), 40);
        let lookahead = std::sync::Arc::new(lookahead);

        // Phase 1: Parallel initial route computation.
        let plans: Vec<Result<RoutePlan, RouterError>> = nets
            .par_iter()
            .map(|&net| {
                compute_route_r1(
                    &*ctx,
                    net,
                    &state.wire_penalty,
                    cfg.bb_margin,
                    cfg.estimate_precision,
                    Some(&lookahead),
                )
            })
            .collect();

        // Phase 2: Serial apply + build wire->net reverse index.
        for plan in plans {
            let plan = plan?;
            if plan.source_wire.is_valid() {
                let _ = apply_route_plan(ctx, &plan);
            }
            add_wire_usage(ctx, &mut state, plan.net);
        }

        // Phase 3: Rip-up-and-reroute loop.
        let net_set: FxHashSet<NetId> = nets.iter().copied().collect();
        for iter in 0..cfg.max_iterations {
            let congested = find_congested_nets_fast(&state, &net_set);
            if congested.is_empty() {
                eprintln!("Router1: converged at iteration {}", iter);
                return Ok(());
            }

            if cfg.verbose || iter % 50 == 0 {
                let congested_wires = state.wire_usage.values().filter(|&&c| c > 1).count();
                eprintln!(
                    "Router1 iter {}: {} congested nets, {} congested wires",
                    iter,
                    congested.len(),
                    congested_wires,
                );
            }

            // Increase penalties for congested wires.
            let congested_wires: Vec<WireId> = state
                .wire_usage
                .iter()
                .filter_map(|(&w, &c)| (c > 1).then_some(w))
                .collect();
            for wire in &congested_wires {
                *state.wire_penalty.entry(*wire).or_insert(0) += cfg.rip_up_penalty;
            }

            // Rip up congested nets.
            for &net in &congested {
                remove_wire_usage(ctx, &mut state, net);
                unroute_net(ctx, net);
            }

            // Parallel reroute.
            let plans: Vec<Result<RoutePlan, RouterError>> = congested
                .par_iter()
                .map(|&net| {
                    compute_route_r1(
                        &*ctx,
                        net,
                        &state.wire_penalty,
                        cfg.bb_margin,
                        cfg.estimate_precision,
                        Some(&lookahead),
                    )
                })
                .collect();

            for plan in plans {
                let plan = plan?;
                if plan.source_wire.is_valid() {
                    let _ = apply_route_plan(ctx, &plan);
                }
                add_wire_usage(ctx, &mut state, plan.net);
            }
        }

        let remaining = find_congested_nets_fast(&state, &net_set);
        if remaining.is_empty() {
            Ok(())
        } else {
            Err(RouterError::Congestion(cfg.max_iterations, remaining.len()))
        }
    }
}

// ---------------------------------------------------------------------------
// Single-net routing
// ---------------------------------------------------------------------------

pub fn route_net(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    let plan = compute_route_r1(ctx, net, wire_penalty, 0, 50, None)?;
    if plan.source_wire.is_valid() {
        let _ = apply_route_plan(ctx, &plan);
    }
    Ok(())
}

pub fn compute_route_r1(
    ctx: &Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bb_margin: i32,
    estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
) -> Result<RoutePlan, RouterError> {
    let source_wire = match resolve_source_wire(ctx, net)? {
        Some(w) => w,
        None => {
            return Ok(RoutePlan {
                net,
                source_wire: WireId::INVALID,
                sink_routes: vec![],
            });
        }
    };

    let bbox = if bb_margin > 0 {
        Some(crate::metrics::compute_bbox(ctx, net, bb_margin))
    } else {
        None
    };

    let mut sink_wires = collect_sink_wires(ctx, net);
    // Sort sinks nearest-first so the routing tree grows toward the destination,
    // making subsequent A* calls cheaper.
    let chipdb = ctx.chipdb();
    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());
    sink_wires.sort_by_key(|&sw| {
        let (wx, wy) = chipdb.tile_xy(sw.tile());
        (wx - src_x).abs() + (wy - src_y).abs()
    });

    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    tree_wires.extend(ctx.net(net).wire_ids());

    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        let const_wires = collect_constant_source_wires(ctx, const_val);
        tree_wires.extend(const_wires);
    }

    let mut sink_routes = Vec::new();
    for sink_wire in sink_wires {
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }
        match astar_route(
            ctx,
            &tree_wires,
            sink_wire,
            wire_penalty,
            bbox.as_ref(),
            estimate_precision,
            lookahead,
            None,
        ) {
            Some(pips) => {
                for &pip in &pips {
                    tree_wires.insert(ctx.chipdb().pip_dst_wire(pip));
                }
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                let net_name = ctx.net(net).name_id();
                let chipdb = ctx.chipdb();
                let (sx, sy) = chipdb.tile_xy(source_wire.tile());
                let (dx, dy) = chipdb.tile_xy(sink_wire.tile());
                println!(
                    "NO_PATH net={} src=({},{}) dst=({},{})",
                    ctx.name_of(net_name),
                    sx,
                    sy,
                    dx,
                    dy,
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

// ---------------------------------------------------------------------------
// A* search with estimate-based pruning
// ---------------------------------------------------------------------------

/// A* search from multiple source wires to a single destination wire.
///
/// Key speed optimizations (matching C++ nextpnr router1):
///
/// 1. **Estimate-based pruning**: once any path is found, candidates whose
///    `estimate / 2` exceeds `best_score + precision` are skipped. This is
///    far tighter than bounding-box pruning.
///
/// 2. **Score-based culling**: before inserting a PIP destination into the
///    queue, reject it if `new_score > best_score + precision`.
///
/// 3. **Adaptive visit limit**: after finding the first path, cap further
///    exploration to `2 * visit_count` to prevent unbounded search on
///    hard instances.
/// Why the A* loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AStarExit {
    /// Destination was reached at least once; path is available.
    Reached,
    /// Priority queue emptied without reaching destination.
    HeapDrained,
    /// `visit_count` exceeded the (possibly adaptive) `max_visits` budget.
    VisitLimit,
}

/// Diagnostic snapshot of an A* search. Populated by
/// [`astar_route_with_trace`] regardless of success or failure.
pub struct AStarTrace {
    /// Total nodes popped from the priority queue.
    pub visit_count: usize,
    /// Visit budget in effect when the loop exited.
    pub max_visits: usize,
    /// Best `cost + penalty` found at `dst_wire`, or `DelayT::MAX` if never reached.
    pub best_score: DelayT,
    /// Reason the loop exited.
    pub exit: AStarExit,
    /// Full visited map: wire -> (cost, penalty, parent pip, came-from wire).
    ///
    /// Useful for post-mortem analysis: tile bbox of exploration, closest
    /// wire to dst, whether dst was enqueued but visit-budget hit, etc.
    pub visited: FxHashMap<WireId, (DelayT, DelayT, Option<PipId>, WireId)>,
}

pub fn astar_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    astar_route_with_trace(
        ctx,
        src_wires,
        dst_wire,
        wire_penalty,
        bbox,
        estimate_precision,
        lookahead,
        visit_limit,
    )
    .0
}

/// Like [`astar_route`] but always returns an [`AStarTrace`] alongside the
/// (optional) path. Used by router diagnostics to answer questions like
/// "how far did A* get before giving up?".
pub fn astar_route_with_trace(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> (Option<Vec<PipId>>, AStarTrace) {
    if src_wires.contains(&dst_wire) {
        let mut visited = FxHashMap::default();
        visited.insert(dst_wire, (0, 0, None, dst_wire));
        return (
            Some(Vec::new()),
            AStarTrace {
                visit_count: 0,
                max_visits: 0,
                best_score: 0,
                exit: AStarExit::Reached,
                visited,
            },
        );
    }

    let chipdb = ctx.chipdb();

    // Heuristic function.
    let h = |src: WireId, dst: WireId| -> DelayT {
        match lookahead {
            Some(la) => la.estimate_delay(chipdb, src, dst),
            None => ctx.estimate_delay(src, dst),
        }
    };

    let init_cap = src_wires.len().saturating_mul(8).max(16);
    let mut heap = BinaryHeap::with_capacity(init_cap);
    // visited: wire -> (best_cost, penalty, Option<pip>, came_from_wire)
    let mut visited: FxHashMap<WireId, (DelayT, DelayT, Option<PipId>, WireId)> =
        FxHashMap::with_capacity_and_hasher(init_cap, Default::default());

    // Best total score (delay + penalty) found so far. Starts at MAX.
    let mut best_score: DelayT = DelayT::MAX;
    // Adaptive visit limit: caps total node visits to prevent OOM on dense graphs.
    // Budget scales with grid area — larger chips get more budget.
    let grid_area = (chipdb.width() as usize) * (chipdb.height() as usize);
    let mut max_visits: usize =
        visit_limit.unwrap_or_else(|| grid_area.saturating_mul(10).max(100_000));
    let mut visit_count: usize = 0;

    // Seed with all source wires.
    for &src in src_wires {
        let hval = h(src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            penalty: 0,
            estimate: hval,
        });
        visited.insert(src, (0, 0, None, src));
    }

    let mut hit_visit_limit = false;
    while let Some(entry) = heap.pop() {
        visit_count += 1;
        if visit_count > max_visits {
            hit_visit_limit = true;
            break;
        }

        // Estimate-based pruning: skip if this candidate is clearly worse
        // than the best path found. The /2 allows exploring alternatives
        // that might be slightly longer but avoid congestion.
        if best_score < DelayT::MAX {
            if entry.estimate / 2 - estimate_precision > best_score {
                continue;
            }
        }

        // Skip if we already found a cheaper path to this wire.
        if let Some(&(prev_cost, prev_pen, _, _)) = visited.get(&entry.wire) {
            if entry.cost + entry.penalty > prev_cost + prev_pen {
                continue;
            }
        }

        // Check destination hit.
        if entry.wire == dst_wire {
            let score = entry.cost + entry.penalty;
            if score < best_score {
                best_score = score;
                // Set adaptive visit limit: allow 2x current visits more.
                if max_visits == usize::MAX {
                    max_visits = visit_count * 2 + 100;
                }
            }
            // Don't break - keep searching for potentially better paths
            // (with lower penalty). The visit limit will stop us.
            continue;
        }

        // Node expansion: spread to other tiles via routing nodes.
        {
            let mut expand_node = |nw: WireId| {
                let new_cost = entry.cost;
                let new_pen = entry.penalty;
                if let Some(&(pc, pp, _, _)) = visited.get(&nw) {
                    if new_cost + new_pen >= pc + pp {
                        return;
                    }
                }
                visited.insert(nw, (new_cost, new_pen, None, entry.wire));
                let hval = h(nw, dst_wire);
                let est = new_cost + new_pen + hval;

                // Score-based culling before insertion.
                if best_score < DelayT::MAX && est / 2 - estimate_precision > best_score {
                    return;
                }

                heap.push(QueueEntry {
                    wire: nw,
                    cost: new_cost,
                    penalty: new_pen,
                    estimate: est,
                });
            };
            if let Some(bb) = bbox {
                chipdb.node_wires_in_bbox_cb(entry.wire, bb, &mut expand_node);
            } else {
                chipdb.node_wires_cb(entry.wire, &mut expand_node);
            }
        }

        // PIP expansion.
        let wire_info = chipdb.wire_info(entry.wire);
        let downhill = wire_info.pips_downhill.get();

        for &pip_idx in downhill {
            let pip = PipId::new(entry.wire.tile(), pip_idx);
            let next_wire = chipdb.pip_dst_wire(pip);

            // Bounding box pruning (coarse spatial filter).
            if let Some(bb) = bbox {
                let (wx, wy) = chipdb.tile_xy(next_wire.tile());
                if !bb.contains(wx, wy) {
                    continue;
                }
            }

            let is_global_clock = is_global_clock_pip(ctx, pip);
            let pip_delay = if is_global_clock {
                0
            } else {
                ctx.pip(pip).delay().max_delay()
            };
            let penalty = wire_penalty.get(&next_wire).copied().unwrap_or(0);
            let step_cost = if is_global_clock { 0 } else { 1 };
            let new_cost = entry.cost + pip_delay + step_cost;
            let new_pen = entry.penalty + penalty;

            // Score-based culling: reject if clearly worse than best.
            let new_score = new_cost + new_pen;
            if best_score < DelayT::MAX && new_score - estimate_precision > best_score {
                continue;
            }

            // Skip if visited with better score.
            if let Some(&(pc, pp, _, _)) = visited.get(&next_wire) {
                if new_cost + new_pen >= pc + pp {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, new_pen, Some(pip), entry.wire));

            let hval = h(next_wire, dst_wire);
            let est = new_score + hval;

            // Pre-insertion culling.
            if best_score < DelayT::MAX && est / 2 - estimate_precision > best_score {
                continue;
            }

            heap.push(QueueEntry {
                wire: next_wire,
                cost: new_cost,
                penalty: new_pen,
                estimate: est,
            });
        }
    }

    let exit = if best_score < DelayT::MAX {
        AStarExit::Reached
    } else if hit_visit_limit {
        AStarExit::VisitLimit
    } else {
        AStarExit::HeapDrained
    };

    // Extract best path if destination was reached.
    let path = if best_score < DelayT::MAX {
        let mut pips = Vec::new();
        let mut current = dst_wire;
        loop {
            let Some(&(_, _, pip, from)) = visited.get(&current) else {
                break;
            };
            match pip {
                Some(p) => {
                    pips.push(p);
                    current = chipdb.pip_src_wire(p);
                }
                None => {
                    if from == current {
                        break;
                    }
                    current = from;
                }
            }
        }
        pips.reverse();
        Some(pips)
    } else {
        None
    };

    let trace = AStarTrace {
        visit_count,
        max_visits,
        best_score,
        exit,
        visited,
    };
    (path, trace)
}

// ---------------------------------------------------------------------------
// Congestion detection
// ---------------------------------------------------------------------------

fn add_wire_usage(ctx: &Context, state: &mut Router1State, net_idx: NetId) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }
    for &wire in net.wires().keys() {
        *state.wire_usage.entry(wire).or_default() += 1;
        state.wire_nets.entry(wire).or_default().insert(net_idx);
    }
}

fn remove_wire_usage(ctx: &Context, state: &mut Router1State, net_idx: NetId) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }
    for &wire in net.wires().keys() {
        if let Some(count) = state.wire_usage.get_mut(&wire) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.wire_usage.remove(&wire);
                state.wire_nets.remove(&wire);
            } else if let Some(nets) = state.wire_nets.get_mut(&wire) {
                nets.remove(&net_idx);
            }
        }
    }
}

fn find_congested_nets_fast(state: &Router1State, net_set: &FxHashSet<NetId>) -> Vec<NetId> {
    let mut nets = FxHashSet::default();
    for (&wire, &usage) in &state.wire_usage {
        if usage > 1 {
            if let Some(wire_nets) = state.wire_nets.get(&wire) {
                for &net in wire_nets {
                    if net_set.contains(&net) {
                        nets.insert(net);
                    }
                }
            }
        }
    }
    nets.into_iter().collect()
}

pub fn find_congested_nets(ctx: &Context) -> Vec<NetId> {
    let congested_wires = find_congested_wires(ctx);
    if congested_wires.is_empty() {
        return Vec::new();
    }

    let congested: FxHashSet<WireId> = congested_wires.iter().copied().collect();
    let mut nets = FxHashSet::default();

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        if net.wires().keys().any(|wire| congested.contains(wire)) {
            nets.insert(net_idx);
        }
    }

    nets.into_iter().collect()
}

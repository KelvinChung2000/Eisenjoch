//! Router1: A* rip-up and reroute router.
//!
//! This module implements an iterative A* routing algorithm that routes each net
//! independently, then detects congestion (wires used by multiple nets) and
//! rips up congested nets for rerouting with increased penalties. The process
//! repeats until all congestion is resolved or the iteration limit is reached.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    find_congested_wires, resolve_source_wire, source_wire_const_value, unroute_net, RoutePlan,
    SinkRoute,
};
use super::RouterError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the Router1 algorithm.
pub struct Router1Cfg {
    /// Maximum number of rip-up-and-reroute iterations.
    pub max_iterations: usize,
    /// Penalty added to a wire each time it is involved in congestion.
    pub rip_up_penalty: DelayT,
    /// Weight multiplier for congestion cost.
    pub congestion_weight: f64,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for Router1Cfg {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            rip_up_penalty: 10,
            congestion_weight: 1.0,
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
    /// The wire this entry represents.
    pub wire: WireId,
    /// g(n): accumulated cost from the source to this wire.
    pub cost: DelayT,
    /// f(n) = g(n) + h(n): total estimated cost through this wire.
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
        // Reverse ordering so BinaryHeap (max-heap) behaves as a min-heap
        // by estimate. Break ties by preferring lower g-cost.
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
    /// Per-wire penalty that increases when a wire is involved in congestion.
    pub wire_penalty: FxHashMap<WireId, DelayT>,
    /// Per-wire usage count, updated incrementally as nets are ripped up/rerouted.
    pub wire_usage: FxHashMap<WireId, u32>,
}

impl Router1State {
    pub fn new() -> Self {
        Self {
            wire_penalty: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Router1: A* rip-up and reroute.
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
        _cfg: &Self::Config,
        net: crate::netlist::NetId,
    ) -> Result<(), super::RouterError> {
        let wire_penalty = FxHashMap::default();
        route_net_impl(ctx, net, &wire_penalty)
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), super::RouterError> {
        use rayon::prelude::*;

        let mut state = Router1State::new();

        // Phase 1: Parallel initial route computation.
        // Reborrow ctx as &Context for shared read access across threads.
        let plans: Vec<Result<RoutePlan, RouterError>> = nets
            .par_iter()
            .map(|&net| compute_route_r1(&*ctx, net, &state.wire_penalty))
            .collect();

        // Phase 2: Serial apply.
        for plan in plans {
            let plan = plan?;
            if plan.source_wire.is_valid() {
                apply_route_plan(ctx, &plan);
            }
            add_wire_usage(ctx, &mut state.wire_usage, plan.net);
        }

        // Phase 3: Rip-up-and-reroute loop.
        let net_set: FxHashSet<NetId> = nets.iter().copied().collect();
        for _iter in 0..cfg.max_iterations {
            let congested_wires: Vec<WireId> = state
                .wire_usage
                .iter()
                .filter_map(|(&w, &c)| (c > 1).then_some(w))
                .collect();
            let congested: Vec<NetId> = find_nets_touching_wires(ctx, &congested_wires)
                .into_iter()
                .filter(|n| net_set.contains(n))
                .collect();
            if congested.is_empty() {
                return Ok(());
            }

            // Increase penalties for congested wires.
            for wire in &congested_wires {
                *state.wire_penalty.entry(*wire).or_insert(0) += cfg.rip_up_penalty;
            }

            // Rip up congested nets.
            for &net in &congested {
                remove_wire_usage(ctx, &mut state.wire_usage, net);
                unroute_net(ctx, net);
            }

            // Parallel reroute.
            let plans: Vec<Result<RoutePlan, RouterError>> = congested
                .par_iter()
                .map(|&net| compute_route_r1(&*ctx, net, &state.wire_penalty))
                .collect();

            for plan in plans {
                let plan = plan?;
                if plan.source_wire.is_valid() {
                    apply_route_plan(ctx, &plan);
                }
                add_wire_usage(ctx, &mut state.wire_usage, plan.net);
            }
        }

        // Check remaining congestion.
        let remaining_congested: Vec<WireId> = state
            .wire_usage
            .iter()
            .filter_map(|(&w, &c)| (c > 1).then_some(w))
            .collect();
        if remaining_congested.is_empty() {
            Ok(())
        } else {
            let congested_nets = find_nets_touching_wires(ctx, &remaining_congested)
                .into_iter()
                .filter(|n| net_set.contains(n))
                .count();
            Err(RouterError::Congestion(cfg.max_iterations, congested_nets))
        }
    }
}

// ---------------------------------------------------------------------------
// Single-net routing
// ---------------------------------------------------------------------------

/// Route a single net from its driver to all of its sinks using A* search.
///
/// For each user (sink) of the net, we find the sink cell's BEL pin wire and
/// run A* from the current routing tree to that sink wire. The resulting path
/// of PIPs is then bound in the context.
#[cfg(feature = "test-utils")]
pub fn route_net(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    route_net_impl(ctx, net, wire_penalty)
}

/// Pure computation function: compute a route plan for a single net without
/// mutating the Context. The returned `RoutePlan` can later be applied via
/// `apply_route_plan`.
pub fn compute_route_r1(
    ctx: &Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
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

    let sink_wires = collect_sink_wires(ctx, net);

    // Track routing tree locally (no Context mutation).
    let mut tree_wires: Vec<WireId> = vec![source_wire];
    // Include already-routed wires from the net.
    tree_wires.extend(ctx.net(net).wire_ids());

    // For constant nets (GND/VCC), add all matching constant wires across the
    // chip as additional source points. Each tile has local constant wires
    // connected to the switch matrix, so routing can start from any of them.
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
        match astar_route(ctx, &tree_wires, sink_wire, wire_penalty) {
            Some(pips) => {
                // Add destination wires of each PIP to tree.
                for &pip in &pips {
                    tree_wires.push(ctx.pip(pip).dst_wire().id());
                }
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                let net_name = ctx.net(net).name_id();
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

fn route_net_impl(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    let plan = compute_route_r1(ctx, net, wire_penalty)?;
    if plan.source_wire.is_valid() {
        apply_route_plan(ctx, &plan);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A* search
// ---------------------------------------------------------------------------

/// Run A* search from multiple source wires to a single destination wire.
///
/// Searches from all `src_wires` simultaneously, which allows the algorithm
/// to find the shortest path from any wire already in the routing tree.
/// Uses `estimate_delay` as the heuristic function.
///
/// Returns a sequence of PIPs forming the path in forward order (source to
/// destination), or `None` if no path exists.
pub fn astar_route(
    ctx: &Context,
    src_wires: &[WireId],
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Option<Vec<PipId>> {
    let src_set: FxHashSet<WireId> = src_wires.iter().copied().collect();

    // Trivial case: destination is already in the source set.
    if src_set.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let init_capacity = src_wires.len().saturating_mul(8).max(16);
    let mut heap = BinaryHeap::with_capacity(init_capacity);
    // visited: wire -> (best cost, Option<pip>, came_from wire)
    // For source wires: pip=None, came_from=self
    // For pip edges: pip=Some(pip), came_from=pip.src_wire
    // For node jumps: pip=None, came_from=source wire in the node
    let mut visited: FxHashMap<WireId, (DelayT, Option<PipId>, WireId)> =
        FxHashMap::with_capacity_and_hasher(init_capacity, Default::default());

    // Seed the search with all source wires.
    for &src in src_wires {
        let h = ctx.estimate_delay(src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            estimate: h,
        });
        visited.insert(src, (0, None, src));
    }

    while let Some(entry) = heap.pop() {
        // Skip if we already found a cheaper path to this wire.
        if let Some(&(prev_cost, _, _)) = visited.get(&entry.wire) {
            if entry.cost > prev_cost {
                continue;
            }
        }

        // Check if we reached the destination.
        if entry.wire == dst_wire {
            // Trace back the path through visited.
            let mut pips = Vec::new();
            let mut current = dst_wire;
            loop {
                let Some(&(_, pip, from)) = visited.get(&current) else {
                    break;
                };
                match pip {
                    Some(p) => {
                        pips.push(p);
                        current = ctx.pip(p).src_wire().id();
                    }
                    None => {
                        // Node jump or source wire.
                        if from == current {
                            break; // Reached a source wire.
                        }
                        current = from; // Follow node jump back.
                    }
                }
            }
            pips.reverse();
            return Some(pips);
        }

        // Expand: iterate all downhill pips from this wire.
        let wire_info = ctx.chipdb().wire_info(entry.wire);
        let downhill_indices = wire_info.pips_downhill.get();

        for &pip_index in downhill_indices {
            let pip = PipId::new(entry.wire.tile(), pip_index);
            let next_wire = ctx.pip(pip).dst_wire().id();

            let pip_delay = ctx.pip(pip).delay().max_delay();
            let penalty = wire_penalty.get(&next_wire).copied().unwrap_or(0);
            let new_cost = entry.cost + pip_delay + penalty + 1;

            if let Some(&(prev_cost, _, _)) = visited.get(&next_wire) {
                if new_cost >= prev_cost {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, Some(pip), entry.wire));

            let h = ctx.estimate_delay(next_wire, dst_wire);
            heap.push(QueueEntry {
                wire: next_wire,
                cost: new_cost,
                estimate: new_cost + h,
            });
        }

        // Node expansion: if this wire is part of a multi-tile node,
        // add all other wires in the same node as reachable at zero extra cost.
        for nw in ctx.chipdb().node_wires(entry.wire) {
            if nw == entry.wire {
                continue;
            }
            if let Some(&(prev_cost, _, _)) = visited.get(&nw) {
                if entry.cost >= prev_cost {
                    continue;
                }
            }
            visited.insert(nw, (entry.cost, None, entry.wire));
            let h = ctx.estimate_delay(nw, dst_wire);
            heap.push(QueueEntry {
                wire: nw,
                cost: entry.cost,
                estimate: entry.cost + h,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Congestion detection
// ---------------------------------------------------------------------------

fn add_wire_usage(ctx: &Context, wire_usage: &mut FxHashMap<WireId, u32>, net_idx: NetId) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }
    for &wire in net.wires().keys() {
        *wire_usage.entry(wire).or_default() += 1;
    }
}

fn remove_wire_usage(ctx: &Context, wire_usage: &mut FxHashMap<WireId, u32>, net_idx: NetId) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }
    for &wire in net.wires().keys() {
        if let Some(count) = wire_usage.get_mut(&wire) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                wire_usage.remove(&wire);
            }
        }
    }
}

fn find_nets_touching_wires(ctx: &Context, congested_wires: &[WireId]) -> Vec<NetId> {
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

/// Find all nets that use at least one congested wire.
///
/// Returns a deduplicated list of net indices.
pub fn find_congested_nets(ctx: &Context) -> Vec<NetId> {
    let congested_wires = find_congested_wires(ctx);
    find_nets_touching_wires(ctx, &congested_wires)
}


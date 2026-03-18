//! Router2: Negotiation-based PathFinder router.
//!
//! This module implements a negotiation-based routing algorithm inspired by the
//! PathFinder approach. Unlike Router1's simple rip-up and reroute with fixed
//! penalties, Router2 uses a negotiation scheme where wires shared by multiple
//! nets receive increasing present-congestion costs plus a historical cost that
//! accumulates over iterations. This encourages nets to find alternative paths
//! rather than fighting over the same congested wires.
//!
//! The algorithm also uses bounding-box pruning during A* search: for each net,
//! a bounding box is computed from the locations of all connected cells, expanded
//! by a configurable margin. During search, wires outside this bounding box are
//! skipped, reducing the search space.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::{BoundingBox, compute_bbox};
use crate::netlist::NetId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    resolve_source_wire, source_wire_const_value, unroute_net, RoutePlan, SinkRoute,
};
use super::RouterError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the Router2 (negotiation-based) algorithm.
#[derive(Clone)]
pub struct Router2Cfg {
    /// Maximum number of negotiation iterations.
    pub max_iterations: usize,
    /// Base cost added to every wire traversal.
    pub base_cost: f64,
    /// Multiplier applied to present-congestion penalty.
    pub present_cost_multiplier: f64,
    /// Multiplier applied to historical congestion penalty.
    pub history_cost_multiplier: f64,
    /// Initial value of the present-congestion cost factor.
    pub initial_present_cost: f64,
    /// Growth factor applied to the present-congestion cost each iteration.
    pub present_cost_growth: f64,
    /// Margin (in tiles) added around the bounding box of each net.
    pub bb_margin: i32,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for Router2Cfg {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            history_cost_multiplier: 1.0,
            initial_present_cost: 1.0,
            present_cost_growth: 1.5,
            bb_margin: 3,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// A* priority queue entry (f64-based costs)
// ---------------------------------------------------------------------------

/// An entry in the Router2 A* search priority queue.
///
/// Uses f64 costs (unlike Router1's integer DelayT costs) to accommodate the
/// floating-point negotiation cost model.
#[derive(Clone)]
pub struct R2QueueEntry {
    /// The wire this entry represents.
    pub wire: WireId,
    /// g(n): accumulated cost from the source to this wire.
    pub cost: f64,
    /// f(n) = g(n) + h(n): total estimated cost through this wire.
    pub estimate: f64,
}

impl Eq for R2QueueEntry {}

impl PartialEq for R2QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}

impl PartialOrd for R2QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for R2QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap (max-heap) behaves as a min-heap
        // by estimate. Break ties by preferring lower g-cost.
        other
            .estimate
            .total_cmp(&self.estimate)
            .then_with(|| other.cost.total_cmp(&self.cost))
    }
}

// ---------------------------------------------------------------------------
// Router2 state
// ---------------------------------------------------------------------------

/// Internal mutable state for the Router2 negotiation algorithm.
pub struct Router2State {
    /// Configuration reference.
    pub cfg: Router2Cfg,
    /// Current present-congestion cost factor (grows each iteration).
    pub present_cost: f64,
    /// Per-wire historical congestion cost, accumulated over iterations.
    pub wire_history: FxHashMap<WireId, f64>,
    /// Per-wire current usage count (how many nets use each wire).
    pub wire_usage: FxHashMap<WireId, u32>,
    /// Per-wire owner: last net that claimed the wire. When exactly one net
    /// uses a wire, this identifies the owner (no present-cost penalty for
    /// the owner).
    pub wire_owner: FxHashMap<WireId, NetId>,
}

impl Router2State {
    /// Create a new Router2 state from the given configuration.
    pub fn new(cfg: &Router2Cfg) -> Self {
        let present_cost = cfg.initial_present_cost;
        Self {
            cfg: cfg.clone(),
            present_cost,
            wire_history: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
            wire_owner: FxHashMap::default(),
        }
    }

    /// Compute the negotiation-based cost of using a wire for a given net.
    ///
    /// The cost has three components:
    /// 1. Base cost (constant per wire).
    /// 2. Present-congestion penalty: proportional to the number of other nets
    ///    currently using the wire, scaled by the present cost factor.
    /// 3. Historical penalty: accumulated from prior iterations where the wire
    ///    was congested.
    pub fn wire_cost(&self, wire: WireId, net_idx: NetId) -> f64 {
        let usage = self.wire_usage.get(&wire).copied().unwrap_or(0);
        let is_own = self.wire_owner.get(&wire) == Some(&net_idx);
        let present_penalty = if is_own { 0.0 } else { usage as f64 };
        let history = self.wire_history.get(&wire).copied().unwrap_or(0.0);
        self.cfg.base_cost
            + present_penalty * self.present_cost * self.cfg.present_cost_multiplier
            + history * self.cfg.history_cost_multiplier
    }

    /// Update the historical congestion costs.
    ///
    /// For every wire that is currently used by more than one net, the excess
    /// usage (usage - 1) is added to the wire's history cost.
    pub fn update_history(&mut self) {
        for (&wire, &usage) in &self.wire_usage {
            if usage > 1 {
                *self.wire_history.entry(wire).or_default() += (usage - 1) as f64;
            }
        }
    }

    /// Recompute wire usage and ownership from the current design state.
    #[cfg(feature = "test-utils")]
    pub fn update_usage(&mut self, design: &crate::netlist::Design) {
        self.wire_usage.clear();
        self.wire_owner.clear();
        for net_idx in design.iter_net_indices() {
            let net = design.net(net_idx);
            if !net.alive {
                continue;
            }
            for &wire in net.wires.keys() {
                *self.wire_usage.entry(wire).or_default() += 1;
                self.wire_owner.insert(wire, net_idx);
            }
        }
    }

    /// Increment usage/owner state from one net's currently routed wires.
    pub fn add_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetId) {
        let net = design.net(net_idx);
        if !net.alive {
            return;
        }

        for &wire in net.wires.keys() {
            *self.wire_usage.entry(wire).or_default() += 1;
            self.wire_owner.insert(wire, net_idx);
        }
    }

    /// Decrement usage/owner state for one net's currently routed wires.
    pub fn remove_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetId) {
        let net = design.net(net_idx);
        if !net.alive {
            return;
        }

        for &wire in net.wires.keys() {
            if let Some(count) = self.wire_usage.get_mut(&wire) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.wire_usage.remove(&wire);
                    self.wire_owner.remove(&wire);
                }
            }
        }
    }

    /// Find all nets that touch at least one congested wire (usage > 1).
    pub fn find_congested_nets(&self, design: &crate::netlist::Design) -> Vec<NetId> {
        let congested_wires: FxHashSet<WireId> = self
            .wire_usage
            .iter()
            .filter(|(_, &u)| u > 1)
            .map(|(&w, _)| w)
            .collect();

        if congested_wires.is_empty() {
            return Vec::new();
        }

        let mut nets = FxHashSet::default();
        for net_idx in design.iter_net_indices() {
            let net = design.net(net_idx);
            if !net.alive {
                continue;
            }
            if net.wires.keys().any(|w| congested_wires.contains(w)) {
                nets.insert(net_idx);
            }
        }
        nets.into_iter().collect()
    }

    /// Count the number of wires with usage > 1 (congested wires).
    pub fn count_congested_wires(&self) -> usize {
        self.wire_usage.values().filter(|&&u| u > 1).count()
    }
}

// ---------------------------------------------------------------------------
// A* search with negotiation costs and bounding box pruning
// ---------------------------------------------------------------------------

/// Run A* search with negotiation costs from multiple source wires to a single
/// destination wire.
///
/// This is similar to Router1's `astar_route`, but:
/// - Uses floating-point costs from the negotiation cost model.
/// - Prunes wires that fall outside the bounding box (with margin).
pub fn astar_route_r2(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    net_idx: NetId,
    state: &Router2State,
    bbox: &BoundingBox,
) -> Option<Vec<PipId>> {
    // Trivial case: destination is already in the source set.
    if src_wires.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let chipdb = ctx.chipdb();
    let init_capacity = src_wires.len().saturating_mul(8).max(16);
    let mut heap = BinaryHeap::with_capacity(init_capacity);
    // visited: wire -> (best cost, Option<pip>, came_from wire)
    let mut visited: FxHashMap<WireId, (f64, Option<PipId>, WireId)> =
        FxHashMap::with_capacity_and_hasher(init_capacity, Default::default());

    for &src in src_wires {
        let h = ctx.estimate_delay(src, dst_wire) as f64;
        heap.push(R2QueueEntry {
            wire: src,
            cost: 0.0,
            estimate: h,
        });
        visited.insert(src, (0.0, None, src));
    }

    while let Some(entry) = heap.pop() {
        if let Some(&(prev_cost, _, _)) = visited.get(&entry.wire) {
            if entry.cost > prev_cost {
                continue;
            }
        }

        if entry.wire == dst_wire {
            let mut pips = Vec::new();
            let mut current = dst_wire;
            loop {
                let Some(&(_, pip, from)) = visited.get(&current) else {
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
            return Some(pips);
        }

        let wire_info = chipdb.wire_info(entry.wire);
        let downhill_indices = wire_info.pips_downhill.get();

        for &pip_index in downhill_indices {
            let pip = PipId::new(entry.wire.tile(), pip_index);
            let next_wire = chipdb.pip_dst_wire(pip);

            let (wx, wy) = chipdb.tile_xy(next_wire.tile());
            if !bbox.contains(wx, wy) {
                continue;
            }

            let pip_delay = ctx.pip(pip).delay().max_delay() as f64;
            let negotiation_cost = state.wire_cost(next_wire, net_idx);
            let new_cost = entry.cost + pip_delay + negotiation_cost;

            if let Some(&(prev_cost, _, _)) = visited.get(&next_wire) {
                if new_cost >= prev_cost {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, Some(pip), entry.wire));

            let h = ctx.estimate_delay(next_wire, dst_wire) as f64;
            heap.push(R2QueueEntry {
                wire: next_wire,
                cost: new_cost,
                estimate: new_cost + h,
            });
        }

        // Node expansion for inter-tile routing nodes (allocation-free).
        chipdb.node_wires_cb(entry.wire, |nw| {
            if let Some(&(prev_cost, _, _)) = visited.get(&nw) {
                if entry.cost >= prev_cost {
                    return;
                }
            }
            visited.insert(nw, (entry.cost, None, entry.wire));
            let h = ctx.estimate_delay(nw, dst_wire) as f64;
            heap.push(R2QueueEntry {
                wire: nw,
                cost: entry.cost,
                estimate: entry.cost + h,
            });
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Single-net routing (Router2 variant)
// ---------------------------------------------------------------------------

/// Pure computation: plan a route for a single net without mutating Context.
///
/// Returns a `RoutePlan` that can later be applied via `apply_route_plan`.
/// The function resolves the source wire, computes a bounding box, collects
/// sink wires, and runs A* search for each sink using the negotiation cost
/// model in `state`.
pub fn compute_route_r2(
    ctx: &Context,
    net: NetId,
    state: &Router2State,
    bb_margin: i32,
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

    let bbox = compute_bbox(ctx, net, bb_margin);
    let sink_wires = collect_sink_wires(ctx, net);

    // Track routing tree locally using HashSet for O(1) contains checks.
    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    tree_wires.extend(ctx.net(net).wire_ids());

    // For constant nets, add all matching constant wires as additional sources.
    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        tree_wires.extend(collect_constant_source_wires(ctx, const_val));
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
        match astar_route_r2(ctx, &tree_wires, sink_wire, net, state, &bbox) {
            Some(pips) => {
                // Extend the local tree with newly reached wires.
                for &pip in &pips {
                    tree_wires.insert(ctx.chipdb().pip_dst_wire(pip));
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

/// Route a single net using Router2's negotiation-based A* search.
///
/// Computes a route plan via `compute_route_r2` and applies it to the context.
fn route_net_r2(
    ctx: &mut Context,
    net_idx: NetId,
    state: &Router2State,
) -> Result<(), RouterError> {
    let plan = compute_route_r2(ctx, net_idx, state, state.cfg.bb_margin)?;
    if plan.source_wire.is_valid() {
        apply_route_plan(ctx, &plan);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Router2: Negotiation-based PathFinder router.
pub struct Router2;

impl super::Router for Router2 {
    type Config = Router2Cfg;

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
        let state = Router2State::new(cfg);
        route_net_r2(ctx, net, &state)
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[crate::netlist::NetId],
    ) -> Result<(), super::RouterError> {
        use rayon::prelude::*;

        let mut state = Router2State::new(cfg);

        // Phase 1: Parallel initial route computation
        let plans: Vec<Result<RoutePlan, RouterError>> = nets
            .par_iter()
            .map(|&net| compute_route_r2(ctx, net, &state, cfg.bb_margin))
            .collect();

        // Serial apply
        for plan in plans {
            let plan = plan?;
            if plan.source_wire.is_valid() {
                apply_route_plan(ctx, &plan);
            }
            state.add_net_usage(&ctx.design, plan.net);
        }

        // Phase 2: Negotiation loop with parallel reroute phases
        let net_set: FxHashSet<crate::netlist::NetId> = nets.iter().copied().collect();
        for _iter in 0..state.cfg.max_iterations {
            let congested: Vec<_> = state
                .find_congested_nets(&ctx.design)
                .into_iter()
                .filter(|n| net_set.contains(n))
                .collect();
            if congested.is_empty() {
                return Ok(());
            }

            for &net_idx in &congested {
                state.remove_net_usage(&ctx.design, net_idx);
                unroute_net(ctx, net_idx);
            }
            state.update_history();

            // Parallel reroute
            let plans: Vec<Result<RoutePlan, RouterError>> = congested
                .par_iter()
                .map(|&net| compute_route_r2(ctx, net, &state, cfg.bb_margin))
                .collect();

            for plan in plans {
                let plan = plan?;
                if plan.source_wire.is_valid() {
                    apply_route_plan(ctx, &plan);
                }
                state.add_net_usage(&ctx.design, plan.net);
            }
            state.present_cost *= state.cfg.present_cost_growth;
        }

        let remaining = state.count_congested_wires();
        if remaining == 0 {
            Ok(())
        } else {
            Err(RouterError::Congestion(state.cfg.max_iterations, remaining))
        }
    }
}


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
use crate::metrics::{compute_bbox, BoundingBox};
use crate::netlist::NetId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    resolve_source_wire, source_wire_const_value, unroute_net, NegotiationCfg, NegotiationState,
    RoutePlan, SinkRoute,
};
use super::RouterError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the Router2 (negotiation-based) algorithm.
#[derive(Clone)]
pub struct Router2Cfg {
    /// Shared negotiation cost-model parameters.
    pub negotiation: NegotiationCfg,
    /// Maximum number of negotiation iterations.
    pub max_iterations: usize,
    /// Margin (in tiles) added around the bounding box of each net.
    pub bb_margin: i32,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for Router2Cfg {
    fn default() -> Self {
        Self {
            negotiation: NegotiationCfg::default(),
            max_iterations: 50,
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
// Router2 state — thin alias over the shared NegotiationState
// ---------------------------------------------------------------------------

/// Internal mutable state for the Router2 negotiation algorithm.
///
/// This is a type alias for the shared `NegotiationState` from `common`.
pub type Router2State = NegotiationState;

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
    bb_margin: i32,
) -> Result<(), RouterError> {
    let plan = compute_route_r2(ctx, net_idx, state, bb_margin)?;
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
        let state = NegotiationState::new(cfg.negotiation.clone());
        route_net_r2(ctx, net, &state, cfg.bb_margin)
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[crate::netlist::NetId],
    ) -> Result<(), super::RouterError> {
        use rayon::prelude::*;

        let mut state = NegotiationState::new(cfg.negotiation.clone());

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
        for _iter in 0..cfg.max_iterations {
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
            Err(RouterError::Congestion(cfg.max_iterations, remaining))
        }
    }
}

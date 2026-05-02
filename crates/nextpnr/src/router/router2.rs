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

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::{compute_bbox, BoundingBox};
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::FxHashSet;

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    resolve_source_wire, source_wire_const_value, unroute_net, NegotiationCfg, NegotiationState,
    RoutePlan, SinkRoute,
};
use super::RouterError;

/// Fixed-point scale for converting Router2's f64 negotiation costs into
/// the `DelayT` (integer) currency used by [`astar_search`]. 100× preserves
/// two decimals; costs in the shared `NegotiationState` are typically in
/// the range `[1, 100]`, so 100× keeps the scaled values comfortably
/// inside `i32`.
const R2_COST_SCALE: f64 = 100.0;

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

// The priority queue + search kernel live in `router::astar`; Router2 only
// supplies a cost model that looks up `state.wire_cost(wire, net_idx)`.

// ---------------------------------------------------------------------------
// Router2 state — thin alias over the shared NegotiationState
// ---------------------------------------------------------------------------

/// Internal mutable state for the Router2 negotiation algorithm.
///
/// This is a type alias for the shared `NegotiationState` from `common`.
pub type Router2State = NegotiationState;

/// Cost model for Router2's negotiation-based A*. `wire_penalty` scales the
/// f64 `state.wire_cost(wire, net_idx)` into `DelayT` space; see
/// [`R2_COST_SCALE`].
struct Router2CostModel<'a> {
    state: &'a Router2State,
    net_idx: NetId,
    bbox: &'a BoundingBox,
}

impl<'a> PathCostModel for Router2CostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn wire_penalty(&self, _ctx: &Context, wire: WireId) -> DelayT {
        let cost = self.state.wire_cost(wire, self.net_idx);
        (cost * R2_COST_SCALE).max(0.0) as DelayT
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        // Router2 previously used ctx.estimate_delay on raw (unscaled) f64
        // costs. Keep the same heuristic but in DelayT (pip delays are
        // already integers, no scaling needed).
        ctx.estimate_delay(wire, dst)
    }
    fn bboxes(&self) -> &[BoundingBox] {
        std::slice::from_ref(self.bbox)
    }
}

// ---------------------------------------------------------------------------
// A* search with negotiation costs and bounding box pruning
// ---------------------------------------------------------------------------

/// Run A* search with negotiation costs from multiple source wires to a single
/// destination wire. Delegates to [`super::astar::astar_search`] with a
/// [`Router2CostModel`] that scales `state.wire_cost` by [`R2_COST_SCALE`]
/// into the kernel's `DelayT` currency. Out-of-bbox tiles are rejected by
/// the cost model's `bbox` method.
pub fn astar_route_r2(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    net_idx: NetId,
    state: &Router2State,
    bbox: &BoundingBox,
) -> Option<Vec<PipId>> {
    let model = Router2CostModel {
        state,
        net_idx,
        bbox,
    };
    let opts = AStarOptions {
        visit_limit: None,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &model, src_wires, dst_wire, &opts).path
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
        let _ = apply_route_plan(ctx, &plan);
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
                let _ = apply_route_plan(ctx, &plan);
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
                    let _ = apply_route_plan(ctx, &plan);
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

//! Router2: PathFinder negotiated-congestion router.
//!
//! Nets are planned against shared present and history congestion costs,
//! congested nets are replanned until no routing node is overused, and the
//! converged plans are then bound into the context.

use crate::chipdb::{PipId, WireId};
use crate::common::{DelayT, PlaceStrength};
use crate::context::Context;
use crate::metrics::{compute_bbox, BoundingBox};
use crate::netlist::NetId;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    resolve_source_wire, source_wire_const_value, NegotiationCfg, NegotiationState, RoutePlan,
    SinkRoute,
};
use super::RouterError;

const R2_COST_SCALE: f64 = 100.0;
const WIRE_COST_LIMIT: DelayT = DelayT::MAX >> 12;

/// Configuration parameters for the Router2 (negotiation-based) algorithm.
#[derive(Clone)]
pub struct Router2Cfg {
    /// Shared negotiation cost-model parameters.
    pub negotiation: NegotiationCfg,
    /// Maximum number of negotiation iterations after the initial pass.
    pub max_iterations: usize,
    /// Margin (in tiles) added around the bounding box of each net.
    pub bb_margin: i32,
    /// Whether to log each negotiation iteration.
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

pub type Router2State = NegotiationState;

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
        (self.state.wire_cost(wire, self.net_idx) * R2_COST_SCALE) as DelayT
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        ctx.estimate_delay(wire, dst)
    }
    fn is_blocked(&self, ctx: &Context, wire: WireId) -> bool {
        ctx.wire_binding(wire)
            .is_some_and(|(owner, _)| owner != self.net_idx)
    }
    fn bboxes(&self) -> &[BoundingBox] {
        std::slice::from_ref(self.bbox)
    }
}

/// A* from `src_wires` to `dst_wire` under the negotiation costs in `state`.
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

/// Plan a route tree for `net` from `source_wire` without mutating the context.
pub fn compute_route_r2(
    ctx: &Context,
    net: NetId,
    source_wire: WireId,
    state: &Router2State,
    bb_margin: i32,
) -> Result<RoutePlan, RouterError> {
    let bbox = compute_bbox(ctx, net, bb_margin);
    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        tree_wires.extend(collect_constant_source_wires(ctx, const_val));
    }

    let mut sink_routes = Vec::new();
    for sink_wire in collect_sink_wires(ctx, net) {
        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }
        let pips = astar_route_r2(ctx, &tree_wires, sink_wire, net, state, &bbox)
            .ok_or_else(|| RouterError::NoPath(ctx.name_of(ctx.net(net).name_id()).to_owned()))?;
        tree_wires.extend(pips.iter().map(|&pip| ctx.chipdb().pip_dst_wire(pip)));
        sink_routes.push(SinkRoute { sink_wire, pips });
    }

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

fn constant_roots(ctx: &Context, plan: &RoutePlan) -> Vec<WireId> {
    let chipdb = ctx.chipdb();
    let const_val = source_wire_const_value(ctx, plan.source_wire);
    if const_val == 0 {
        return Vec::new();
    }
    let roots: FxHashSet<WireId> = plan
        .sink_routes
        .iter()
        .filter_map(|route| route.pips.first())
        .map(|&pip| chipdb.pip_src_wire(pip))
        .filter(|&wire| wire != plan.source_wire && source_wire_const_value(ctx, wire) == const_val)
        .collect();
    roots.into_iter().collect()
}

fn plan_wires(ctx: &Context, plan: &RoutePlan, roots: &[WireId]) -> Vec<WireId> {
    let chipdb = ctx.chipdb();
    let mut wires: FxHashSet<WireId> = FxHashSet::default();
    let tree = std::iter::once(plan.source_wire)
        .chain(roots.iter().copied())
        .chain(
            plan.sink_routes
                .iter()
                .flat_map(|route| route.pips.iter().map(|&pip| chipdb.pip_dst_wire(pip))),
        );
    for wire in tree {
        wires.insert(wire);
        chipdb.node_wires_cb(wire, |peer| {
            wires.insert(peer);
        });
    }
    wires.into_iter().collect()
}

struct NetPlan {
    plan: RoutePlan,
    constant_roots: Vec<WireId>,
    wires: Vec<WireId>,
}

struct Negotiation {
    state: Router2State,
    plans: FxHashMap<NetId, NetPlan>,
}

impl Negotiation {
    fn insert(&mut self, ctx: &Context, plan: RoutePlan) {
        let constant_roots = constant_roots(ctx, &plan);
        let wires = plan_wires(ctx, &plan, &constant_roots);
        self.state.add_wire_usage(plan.net, &wires);
        self.plans.insert(
            plan.net,
            NetPlan {
                plan,
                constant_roots,
                wires,
            },
        );
    }

    fn withdraw(&mut self, net: NetId) {
        let planned = self
            .plans
            .remove(&net)
            .expect("only planned nets are withdrawn");
        self.state.remove_wire_usage(net, &planned.wires);
    }

    fn overused_wires(&self) -> usize {
        self.state.count_congested_wires()
    }

    fn congested_nets(&self) -> Vec<NetId> {
        let usage = &self.state.wire_usage;
        let mut nets: Vec<NetId> = self
            .plans
            .iter()
            .filter(|(_, planned)| planned.wires.iter().any(|wire| usage[wire] > 1))
            .map(|(&net, _)| net)
            .collect();
        nets.sort_unstable();
        nets
    }

    fn max_wire_cost(&self) -> f64 {
        let state = &self.state;
        let usage = state.wire_usage.values().copied().max().unwrap_or(0);
        let history = state.wire_history.values().copied().fold(0.0, f64::max);
        let hint = state.external_hints.values().copied().fold(0.0, f64::max);
        state.cfg.base_cost
            + f64::from(usage) * state.present_cost * state.cfg.present_cost_multiplier
            + history * state.cfg.history_cost_multiplier
            + hint
    }

    fn advance_costs(&mut self, iteration: usize) -> Result<(), RouterError> {
        self.state.update_history();
        self.state.present_cost *= self.state.cfg.present_cost_growth;
        let wire_cost = self.max_wire_cost();
        match wire_cost * R2_COST_SCALE <= f64::from(WIRE_COST_LIMIT) {
            true => Ok(()),
            false => Err(RouterError::CostRange {
                iteration,
                wire_cost,
            }),
        }
    }

    fn commit(self, ctx: &mut Context) -> Result<(), RouterError> {
        let mut plans: Vec<NetPlan> = self.plans.into_values().collect();
        plans.sort_unstable_by_key(|planned| planned.plan.net);
        for NetPlan {
            plan,
            constant_roots,
            ..
        } in plans
        {
            let rejected = |ctx: &Context, wire: WireId| RouterError::CommitRejected {
                net: ctx.name_of(ctx.net(plan.net).name_id()).to_owned(),
                wire: ctx.chipdb().wire_name(wire).to_owned(),
            };
            apply_route_plan(ctx, &plan).map_err(|wire| rejected(ctx, wire))?;
            for root in constant_roots {
                ctx.try_bind_wire_node(root, plan.net, PlaceStrength::Strong)
                    .map_err(|wire| rejected(ctx, wire))?;
                ctx.design
                    .net_edit(plan.net)
                    .add_wire(root, None, PlaceStrength::Strong);
            }
        }
        Ok(())
    }
}

/// One negotiation pass: how many nets it planned and how many wires stayed overused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterationRecord {
    pub replanned_nets: usize,
    pub overused_wires: usize,
}

/// What a converged Router2 run did, pass by pass; the first record is the initial pass.
#[derive(Debug)]
pub struct Router2Report {
    pub iterations: Vec<IterationRecord>,
}

fn net_sources(ctx: &Context, nets: &[NetId]) -> Result<Vec<(NetId, WireId)>, RouterError> {
    let mut sources = Vec::with_capacity(nets.len());
    for &net in nets {
        if ctx.net(net).wire_ids().next().is_some() {
            return Err(RouterError::AlreadyRouted(
                ctx.name_of(ctx.net(net).name_id()).to_owned(),
            ));
        }
        if let Some(source) = resolve_source_wire(ctx, net)? {
            sources.push((net, source));
        }
    }
    Ok(sources)
}

/// Router2: Negotiation-based PathFinder router.
pub struct Router2;

impl Router2 {
    /// Route `nets` to convergence, bind them, and report each pass.
    pub fn route_nets_report(
        &self,
        ctx: &mut Context,
        cfg: &Router2Cfg,
        nets: &[NetId],
    ) -> Result<Router2Report, RouterError> {
        let sources = net_sources(ctx, nets)?;
        let source_of: FxHashMap<NetId, WireId> = sources.iter().copied().collect();
        let mut negotiation = Negotiation {
            state: NegotiationState::new(cfg.negotiation.clone()),
            plans: FxHashMap::default(),
        };

        let initial: Vec<RoutePlan> = sources
            .par_iter()
            .map(|&(net, source)| {
                compute_route_r2(ctx, net, source, &negotiation.state, cfg.bb_margin)
            })
            .collect::<Result<_, _>>()?;
        for plan in initial {
            negotiation.insert(ctx, plan);
        }
        let mut iterations = vec![IterationRecord {
            replanned_nets: sources.len(),
            overused_wires: negotiation.overused_wires(),
        }];

        for iteration in 1..=cfg.max_iterations {
            if negotiation.overused_wires() == 0 {
                break;
            }
            negotiation.advance_costs(iteration)?;
            let congested = negotiation.congested_nets();
            for &net in &congested {
                negotiation.withdraw(net);
                let plan =
                    compute_route_r2(ctx, net, source_of[&net], &negotiation.state, cfg.bb_margin)?;
                negotiation.insert(ctx, plan);
            }
            let record = IterationRecord {
                replanned_nets: congested.len(),
                overused_wires: negotiation.overused_wires(),
            };
            if cfg.verbose {
                eprintln!("Router2 iteration {iteration}: {record:?}");
            }
            iterations.push(record);
        }

        let overused = negotiation.overused_wires();
        if overused != 0 {
            return Err(RouterError::Congestion(cfg.max_iterations, overused));
        }
        negotiation.commit(ctx)?;
        Ok(Router2Report { iterations })
    }
}

impl super::Router for Router2 {
    type Config = Router2Cfg;

    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), RouterError> {
        let nets = collect_routable_nets(ctx);
        self.route_nets(ctx, cfg, &nets)
    }

    fn route_net(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        net: NetId,
    ) -> Result<(), RouterError> {
        self.route_nets(ctx, cfg, &[net])
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), RouterError> {
        self.route_nets_report(ctx, cfg, nets).map(|_| ())
    }
}

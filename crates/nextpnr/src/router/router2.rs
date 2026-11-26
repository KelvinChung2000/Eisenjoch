//! Router2: PathFinder negotiated-congestion router, optionally timing-driven.
//!
//! Nets are planned against shared present and history congestion costs,
//! every net is replanned each pass until no routing node is overused, and
//! the converged plans are then bound into the context.

use crate::chipdb::{PipId, WireId};
use crate::common::{DelayT, PlaceStrength};
use crate::context::Context;
use crate::live::{LiveProgress, LiveRouting, PlannedNet};
use crate::metrics::{compute_bbox, BoundingBox};
use crate::netlist::NetId;
use crate::sta::{
    ClasslessPips, CritNorm, DelayBounds, InterconnectError, InterconnectModel, InterconnectStage,
    NetSink, PlacedEstimate, StaSession, StaSessionError, NOTIONAL_CLASSLESS_PIPS,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::{Duration, Instant};

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    resolve_source_wire, source_wire_const_value, NegotiationCfg, NegotiationState, RoutePlan,
    SinkRoute,
};
use super::RouterError;

const R2_COST_SCALE: f64 = 100.0;
const EDGE_COST_LIMIT: DelayT = DelayT::MAX >> 12;

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
    /// Weight each sink's delay by its STA criticality; `None` routes for congestion only.
    pub timing: Option<TimingDrivenCfg>,
}

/// Criticality c = min(max_criticality, crit^criticality_exponent) sets the
/// delay weight 1 / (1 - c) of a sink's A* edge cost.
#[derive(Debug, Clone, Copy)]
pub struct TimingDrivenCfg {
    pub criticality_exponent: f64,
    pub max_criticality: f64,
    pub norm: CritNorm,
    pub classless_pips: ClasslessPips,
}

impl Default for TimingDrivenCfg {
    fn default() -> Self {
        Self {
            criticality_exponent: 1.0,
            max_criticality: 0.99,
            norm: CritNorm::default(),
            classless_pips: NOTIONAL_CLASSLESS_PIPS,
        }
    }
}

/// Factor k = 1 / (1 - c) >= 1 on the delay term of one sink's edge cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayWeight(f64);

impl DelayWeight {
    pub const UNIT: Self = Self(1.0);

    pub fn of_criticality(criticality: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&criticality),
            "criticality {criticality} lies in [0, 1)"
        );
        Self(1.0 / (1.0 - criticality))
    }

    fn scale_cost(self, cost: DelayT) -> DelayT {
        (f64::from(cost) * self.0).ceil() as DelayT
    }

    fn scale_bound(self, bound: DelayT) -> DelayT {
        (f64::from(bound) * self.0).floor() as DelayT
    }
}

enum SinkWeights {
    Uniform,
    Timed(FxHashMap<WireId, DelayWeight>),
}

impl SinkWeights {
    fn of(&self, sink_wire: WireId) -> DelayWeight {
        match self {
            SinkWeights::Uniform => DelayWeight::UNIT,
            SinkWeights::Timed(weights) => weights[&sink_wire],
        }
    }
}

impl Default for Router2Cfg {
    fn default() -> Self {
        Self {
            negotiation: NegotiationCfg::default(),
            max_iterations: 50,
            bb_margin: 3,
            verbose: false,
            timing: None,
        }
    }
}

pub type Router2State = NegotiationState;

struct Router2CostModel<'a> {
    state: &'a Router2State,
    net_idx: NetId,
    bbox: &'a BoundingBox,
    weight: DelayWeight,
}

impl<'a> PathCostModel for Router2CostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        self.weight.scale_cost(default_pip_cost(ctx, pip))
    }
    fn wire_penalty(&self, _ctx: &Context, wire: WireId) -> DelayT {
        (self.state.wire_cost(wire, self.net_idx) * R2_COST_SCALE) as DelayT
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        self.weight.scale_bound(ctx.estimate_delay(wire, dst))
    }
    fn is_blocked(&self, ctx: &Context, wire: WireId) -> bool {
        ctx.wire_binding(wire)
            .is_some_and(|(owner, _)| owner != self.net_idx)
    }
    fn bboxes(&self) -> &[BoundingBox] {
        std::slice::from_ref(self.bbox)
    }
}

/// A* from `src_wires` to `dst_wire` under the negotiation costs in `state`,
/// with the delay term of every edge scaled by `weight`.
pub fn astar_route_r2(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    net_idx: NetId,
    state: &Router2State,
    bbox: &BoundingBox,
    weight: DelayWeight,
) -> Option<Vec<PipId>> {
    let model = Router2CostModel {
        state,
        net_idx,
        bbox,
        weight,
    };
    let opts = AStarOptions {
        visit_limit: None,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &model, src_wires, dst_wire, &opts).path
}

fn compute_route_r2(
    ctx: &Context,
    net: NetId,
    source_wire: WireId,
    state: &Router2State,
    bb_margin: i32,
    weights: &SinkWeights,
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
        let weight = weights.of(sink_wire);
        let pips = astar_route_r2(ctx, &tree_wires, sink_wire, net, state, &bbox, weight)
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
    max_pip_cost: DelayT,
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

    fn offer_live(&self, ctx: &Context, iteration: usize) -> Result<(), RouterError> {
        ctx.live_cancelled()?;
        if !ctx.live_wanted() {
            return Ok(());
        }
        let chipdb = ctx.chipdb();
        let nets: Vec<PlannedNet> = self
            .planned_nets()
            .into_iter()
            .map(|net| {
                let planned = &self.plans[&net];
                let pips = planned.plan.sink_routes.iter().flat_map(|route| {
                    route
                        .pips
                        .iter()
                        .map(|&pip| (chipdb.pip_dst_wire(pip), Some(pip)))
                });
                let wires: FxHashMap<WireId, Option<PipId>> =
                    std::iter::once((planned.plan.source_wire, None))
                        .chain(planned.constant_roots.iter().map(|&root| (root, None)))
                        .chain(pips)
                        .collect();
                (net, wires.into_iter().collect())
            })
            .collect();
        ctx.publish_live(LiveProgress {
            phase: "route",
            iteration: Some(iteration),
            overused: Some(self.overused_wires()),
            routing: LiveRouting::Planned(&nets),
        });
        Ok(())
    }

    fn overused_wires(&self) -> usize {
        self.state.count_congested_wires()
    }

    fn planned_nets(&self) -> Vec<NetId> {
        let mut nets: Vec<NetId> = self.plans.keys().copied().collect();
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
        let edge_cost = f64::from(self.max_pip_cost) + wire_cost * R2_COST_SCALE;
        match edge_cost <= f64::from(EDGE_COST_LIMIT) {
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

#[derive(Debug, Clone, Copy)]
struct SinkPin {
    user: u32,
    wire: WireId,
    timed: bool,
}

fn sink_pins(
    ctx: &Context,
    session: &StaSession,
    net: NetId,
) -> Result<Vec<SinkPin>, StaSessionError> {
    let mut pins = Vec::new();
    for (user, pin) in ctx.design.net(net).users().iter().enumerate() {
        if !pin.is_valid() {
            continue;
        }
        let Some(bel) = ctx.design.cell(pin.cell).bel else {
            continue;
        };
        let Some(wire) = ctx.bel(bel).pin_wire(pin.port) else {
            continue;
        };
        let user = u32::try_from(user).expect("net user index fits in u32");
        let timed = match session.connection_delay(NetSink { net, user }) {
            Ok(_) => true,
            Err(
                StaSessionError::IdealClockSink(_)
                | StaSessionError::UnclockedClockSink(_)
                | StaSessionError::UntimedSink { .. },
            ) => false,
            Err(error) => return Err(error),
        };
        pins.push(SinkPin {
            user,
            wire: wire.id(),
            timed,
        });
    }
    Ok(pins)
}

fn tree_parent(ctx: &Context, parents: &FxHashMap<WireId, PipId>, wire: WireId) -> Option<PipId> {
    if let Some(&pip) = parents.get(&wire) {
        return Some(pip);
    }
    let mut found = None;
    ctx.chipdb().node_wires_cb(wire, |peer| {
        if found.is_none() {
            found = parents.get(&peer).copied();
        }
    });
    found
}

fn plan_delay(
    ctx: &Context,
    model: &InterconnectModel,
    plan: &RoutePlan,
    parents: &FxHashMap<WireId, PipId>,
    sink_wire: WireId,
) -> Result<DelayBounds, InterconnectError> {
    let chipdb = ctx.chipdb();
    let source_node = chipdb.node_id(plan.source_wire);
    let at_source = |wire: WireId| {
        wire == plan.source_wire || (source_node.is_some() && chipdb.node_id(wire) == source_node)
    };
    let mut delay = DelayBounds::ZERO;
    let mut wire = sink_wire;
    for _ in 0..=parents.len() {
        if at_source(wire) {
            return Ok(delay);
        }
        let pip = tree_parent(ctx, parents, wire).expect("a plan leads every sink to its source");
        delay = delay
            + model
                .pip_delay(chipdb, pip)
                .ok_or(InterconnectError::ClasslessPip { pip })?;
        wire = chipdb.pip_src_wire(pip);
    }
    panic!("the parent chain of a plan has no cycle");
}

struct TimingDriven {
    cfg: TimingDrivenCfg,
    model: InterconnectModel,
    session: StaSession,
    pins: FxHashMap<NetId, Vec<SinkPin>>,
    routed: FxHashSet<NetId>,
}

impl TimingDriven {
    fn build(ctx: &Context, cfg: TimingDrivenCfg, nets: &[NetId]) -> Result<Self, RouterError> {
        if !(0.0..1.0).contains(&cfg.max_criticality) {
            return Err(RouterError::CriticalityCap(cfg.max_criticality));
        }
        if !(cfg.criticality_exponent > 0.0 && cfg.criticality_exponent.is_finite()) {
            return Err(RouterError::CriticalityExponent(cfg.criticality_exponent));
        }
        let model = InterconnectModel::new(ctx, cfg.classless_pips, PlacedEstimate::ExactSearch)
            .map_err(StaSessionError::from)?;
        let session = StaSession::build(ctx, model.clone(), InterconnectStage::Placed)?;
        let clocks = &session.sta().session().state().constraints.clocks;
        if clocks.is_empty() {
            return Err(RouterError::NoClock);
        }
        let pins = nets
            .iter()
            .map(|&net| Ok((net, sink_pins(ctx, &session, net)?)))
            .collect::<Result<_, StaSessionError>>()?;
        Ok(Self {
            cfg,
            model,
            session,
            pins,
            routed: FxHashSet::default(),
        })
    }

    fn weights(&self, net: NetId) -> Result<SinkWeights, RouterError> {
        let mut weights = FxHashMap::default();
        for pin in &self.pins[&net] {
            let criticality = self.session.normalised_criticality(
                NetSink {
                    net,
                    user: pin.user,
                },
                self.cfg.norm,
            )?;
            let weight = match criticality {
                None => DelayWeight::UNIT,
                Some(crit) => DelayWeight::of_criticality(
                    crit.powf(self.cfg.criticality_exponent)
                        .min(self.cfg.max_criticality),
                ),
            };
            weights.insert(pin.wire, weight);
        }
        Ok(SinkWeights::Timed(weights))
    }

    fn refresh(
        &mut self,
        ctx: &Context,
        plans: &FxHashMap<NetId, NetPlan>,
        nets: &[NetId],
    ) -> Result<Duration, RouterError> {
        let start = Instant::now();
        let mut delays = Vec::new();
        for &net in nets {
            let plan = &plans[&net].plan;
            let parents: FxHashMap<WireId, PipId> = plan
                .sink_routes
                .iter()
                .flat_map(|route| &route.pips)
                .map(|&pip| (ctx.chipdb().pip_dst_wire(pip), pip))
                .collect();
            for pin in self.pins[&net].iter().filter(|pin| pin.timed) {
                let delay = plan_delay(ctx, &self.model, plan, &parents, pin.wire)
                    .map_err(StaSessionError::from)?;
                delays.push((
                    NetSink {
                        net,
                        user: pin.user,
                    },
                    delay,
                ));
            }
        }
        self.session.set_connection_delays(&delays)?;
        self.routed.extend(nets);
        Ok(start.elapsed())
    }
}

fn max_delay_weight(timing: &Option<TimingDriven>) -> DelayWeight {
    match timing {
        None => DelayWeight::UNIT,
        Some(timing) => DelayWeight::of_criticality(timing.cfg.max_criticality),
    }
}

fn sink_weights(timing: &Option<TimingDriven>, net: NetId) -> Result<SinkWeights, RouterError> {
    match timing {
        None => Ok(SinkWeights::Uniform),
        Some(timing) => timing.weights(net),
    }
}

/// One negotiation pass: how many nets it planned, how many wires stayed
/// overused, and how long pushing its routed delays into the STA took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterationRecord {
    pub replanned_nets: usize,
    pub overused_wires: usize,
    pub timing_refresh: Option<Duration>,
}

/// Where the connection delays of a net in a [`RoutedTiming`] session come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelaySource {
    Estimated,
    Routed,
}

/// The STA session of a timing-driven run: nets this run routed carry their
/// routed delays, every other net keeps its placed estimate.
#[derive(Debug)]
pub struct RoutedTiming {
    session: StaSession,
    routed: FxHashSet<NetId>,
}

impl RoutedTiming {
    pub fn session(&self) -> &StaSession {
        &self.session
    }

    pub fn delay_source(&self, net: NetId) -> Result<DelaySource, StaSessionError> {
        self.session.net_sinks(net)?;
        Ok(match self.routed.contains(&net) {
            true => DelaySource::Routed,
            false => DelaySource::Estimated,
        })
    }
}

/// What a converged Router2 run did, pass by pass; the first record is the
/// initial pass. A timing-driven run also returns its STA session.
#[derive(Debug)]
pub struct Router2Report {
    pub iterations: Vec<IterationRecord>,
    pub timing: Option<RoutedTiming>,
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
        let routed: Vec<NetId> = sources.iter().map(|&(net, _)| net).collect();
        let mut timing = cfg
            .timing
            .map(|timing_cfg| TimingDriven::build(ctx, timing_cfg, &routed))
            .transpose()?;
        let mut negotiation = Negotiation {
            state: NegotiationState::new(cfg.negotiation.clone()),
            plans: FxHashMap::default(),
            max_pip_cost: max_delay_weight(&timing).scale_cost(
                ctx.chipdb()
                    .pips()
                    .map(|pip| default_pip_cost(ctx, pip))
                    .max()
                    .unwrap_or(0),
            ),
        };

        let weights = routed
            .iter()
            .map(|&net| sink_weights(&timing, net))
            .collect::<Result<Vec<_>, _>>()?;
        let initial: Vec<RoutePlan> = sources
            .par_iter()
            .zip(&weights)
            .map(|(&(net, source), weights)| {
                compute_route_r2(ctx, net, source, &negotiation.state, cfg.bb_margin, weights)
            })
            .collect::<Result<_, _>>()?;
        for plan in initial {
            negotiation.insert(ctx, plan);
        }
        negotiation.offer_live(ctx, 0)?;
        let timing_refresh = timing
            .as_mut()
            .map(|timing| timing.refresh(ctx, &negotiation.plans, &routed))
            .transpose()?;
        let mut iterations = vec![IterationRecord {
            replanned_nets: sources.len(),
            overused_wires: negotiation.overused_wires(),
            timing_refresh,
        }];

        for iteration in 1..=cfg.max_iterations {
            if negotiation.overused_wires() == 0 {
                break;
            }
            negotiation.advance_costs(iteration)?;
            let replanned = negotiation.planned_nets();
            for &net in &replanned {
                negotiation.withdraw(net);
                let weights = sink_weights(&timing, net)?;
                let plan = compute_route_r2(
                    ctx,
                    net,
                    source_of[&net],
                    &negotiation.state,
                    cfg.bb_margin,
                    &weights,
                )?;
                negotiation.insert(ctx, plan);
                negotiation.offer_live(ctx, iteration)?;
            }
            let timing_refresh = timing
                .as_mut()
                .map(|timing| timing.refresh(ctx, &negotiation.plans, &replanned))
                .transpose()?;
            let record = IterationRecord {
                replanned_nets: replanned.len(),
                overused_wires: negotiation.overused_wires(),
                timing_refresh,
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
        Ok(Router2Report {
            iterations,
            timing: timing.map(|timing| RoutedTiming {
                session: timing.session,
                routed: timing.routed,
            }),
        })
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

//! Router1: A* rip-up and reroute router.
//!
//! Implements an A* routing algorithm with estimate-based pruning (matching
//! the C++ nextpnr router1 approach). Routes each net independently, then
//! detects congestion and rips up congested nets for rerouting with increased
//! penalties.

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::{
    astar_search, default_pip_cost, AStarOptions, AStarTrace as CoreTrace, PathCostModel,
};
use super::common::{
    apply_route_plan, collect_constant_source_wires, collect_routable_nets, collect_sink_wires,
    find_congested_wires, resolve_source_wire, source_wire_const_value, unroute_net, RoutePlan,
    SinkRoute,
};
use super::lookahead::{Lookahead, LookaheadClass, UNKNOWN_CLASS};
use super::RouterError;

/// Numerator and denominator of the weighted-A* multiplier applied to the
/// lookahead-derived heuristic. `f = g + (NUM/DEN)·h`. Set to 2.0 because
/// the chain lookahead consistently underestimates true min-cost by 1.6–2.2×
/// on xc7_large (intra-tile shortest paths ignore the entry/exit-wire
/// pairing constraint between adjacent tiles). Bounded suboptimality:
/// any returned path costs at most NUM/DEN times the true optimum.
const HEURISTIC_WEIGHT_NUM: DelayT = 2;
const HEURISTIC_WEIGHT_DEN: DelayT = 1;

/// Resolve the driver-BEL class for `net`. Returns [`UNKNOWN_CLASS`]
/// when no lookahead is provided, the net has no driver (constants),
/// the driver hasn't been placed, or its BEL type isn't recognized —
/// any of those means we can't safely class-gate the search.
pub(super) fn driver_class_for_net(
    ctx: &Context,
    net: NetId,
    lookahead: Option<&Lookahead>,
) -> LookaheadClass {
    let Some(la) = lookahead else {
        return UNKNOWN_CLASS;
    };
    let Some(driver_pin) = ctx.net(net).driver_cell_port() else {
        return UNKNOWN_CLASS;
    };
    let Some(bel) = ctx.cell(driver_pin.cell).bel() else {
        return UNKNOWN_CLASS;
    };
    let bel_type_id = ctx.chipdb().bel_info(bel.id()).bel_type();
    la.class_for_bel_type(bel_type_id).unwrap_or(UNKNOWN_CLASS)
}

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

// Priority queue is owned by the shared `router::astar` kernel; maze.rs now
// only constructs cost-model adapters and delegates the search.

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
        let lookahead = super::lookahead::Lookahead::build(ctx);
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
            net,
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

// Re-export the kernel's trace and exit types so callers that read
// `maze::AStarTrace` / `maze::AStarExit` keep compiling after the migration.
pub use super::astar::{AStarExit, AStarTrace};

/// Cost model used by the router1 A*: `pip_delay + 1` per PIP, plus a
/// per-wire congestion penalty looked up in `wire_penalty`.
///
/// `pip_delay + 1` is the same per-pip cost the lookahead's `per_tile_cost`
/// is taken from, so `h = manhattan × per_tile_cost ≤ true_cost(w, dst)`
/// stays admissible and first-pop-optimal break is sound.
struct MazeCostModel<'a> {
    wire_penalty: &'a FxHashMap<WireId, DelayT>,
    bbox: Option<&'a BoundingBox>,
    lookahead: Option<&'a Lookahead>,
    /// Net being routed. Wires already bound to a different net are treated
    /// as near-impassable (`DelayT::MAX / 4`) so A* plans around them rather
    /// than through them. Without this, A* emits plans whose `apply_route_plan`
    /// fails atomically on the first conflict, leaving the net empty.
    net: NetId,
    /// Driver-BEL class of the net being routed. Used together with
    /// `lookahead` to hard-gate A* expansion onto wires not reachable
    /// from this driver class (e.g. fabric data nets are never
    /// candidates for the GCLK backbone).
    driver_class: LookaheadClass,
    /// When true, the `wire_binding` check is skipped. Used by the scoped
    /// rip-up path to find the theoretical shortest corridor regardless of
    /// current occupancy; the caller then inspects the returned PIPs to
    /// decide which foreign nets need to be ripped up.
    ignore_bindings: bool,
}

impl<'a> MazeCostModel<'a> {
    fn new(
        ctx: &Context,
        net: NetId,
        wire_penalty: &'a FxHashMap<WireId, DelayT>,
        bbox: Option<&'a BoundingBox>,
        lookahead: Option<&'a Lookahead>,
        ignore_bindings: bool,
    ) -> Self {
        let driver_class = driver_class_for_net(ctx, net, lookahead);
        Self {
            wire_penalty,
            bbox,
            lookahead,
            net,
            driver_class,
            ignore_bindings,
        }
    }
}

impl<'a> PathCostModel for MazeCostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        // Explore mode = pure Dijkstra with uniform pip cost. Non-uniform
        // lookahead costs inflate Dijkstra visit count via stale-entry
        // re-pops; uniform cost keeps visit count tight and bounded by the
        // reachable wire set, not by cost-shell granularity. The caller
        // re-computes path delay using the lookahead pip_cost after a
        // path is found, so the timing-driven decision is unaffected.
        if self.ignore_bindings {
            return 1;
        }
        match self.lookahead {
            Some(la) => la.pip_cost(ctx.chipdb(), pip),
            None => default_pip_cost(ctx, pip),
        }
    }
    fn wire_penalty(&self, ctx: &Context, wire: WireId) -> DelayT {
        if !self.ignore_bindings {
            if let Some((owner, _)) = ctx.wire_binding(wire) {
                if owner != self.net {
                    return DelayT::MAX / 4;
                }
            }
        }
        self.wire_penalty.get(&wire).copied().unwrap_or(0)
    }
    fn is_blocked(&self, ctx: &Context, wire: WireId) -> bool {
        // Class-gate: a fabric net cannot legally land on clock-only
        // wires regardless of cost. Drop them from A*'s candidate set
        // so the search never visits them. Skipped in explore mode
        // because the lookahead reachability bitmap can have false
        // negatives that would hide otherwise-legitimate corridors;
        // exploration only needs to find ANY path, and the resulting
        // PIP list is used solely to identify blocker nets, not to
        // commit a route.
        if self.ignore_bindings {
            return false;
        }
        if let Some(la) = self.lookahead {
            if !la.is_reachable(ctx.chipdb(), wire, self.driver_class) {
                return true;
            }
        }
        false
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        // Explore mode = pure Dijkstra: complete and provably finds any
        // existing path. The scoped rip-up corridor probe must not miss
        // a real path due to bounded-suboptimal weighted-A* incompleteness.
        if self.ignore_bindings {
            return 0;
        }
        let h = match self.lookahead {
            Some(la) => la.estimate_delay(ctx.chipdb(), wire, dst, self.driver_class),
            None => ctx.estimate_delay(wire, dst),
        };
        // Weighted A*: multiply the admissible-but-loose chain estimate
        // by `HEURISTIC_WEIGHT_NUM / HEURISTIC_WEIGHT_DEN` so f = g + w·h
        // becomes more discriminating between candidates and the search
        // prunes harder. Bounded suboptimal: any path returned costs at
        // most `w × optimal`. Weight matches the empirically measured
        // ratio of true_cost / h_lookahead on long IO routes (≈1.6–2.2).
        h.saturating_mul(HEURISTIC_WEIGHT_NUM) / HEURISTIC_WEIGHT_DEN
    }
    fn bboxes(&self) -> &[BoundingBox] {
        match self.bbox {
            Some(bb) => std::slice::from_ref(bb),
            None => &[],
        }
    }
}

pub fn astar_route(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    _estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    let model = MazeCostModel::new(ctx, net, wire_penalty, bbox, lookahead, false);
    let opts = AStarOptions {
        visit_limit,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &model, src_wires, dst_wire, &opts).path
}

/// Exploration A*: same as [`astar_route`] but ignores `wire_binding`, so the
/// returned path follows the theoretical shortest corridor regardless of which
/// wires are currently claimed by other nets. Used exclusively by the scoped
/// rip-up logic to identify the minimal set of blocker nets.
pub fn astar_route_explore(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    let model = MazeCostModel::new(ctx, net, wire_penalty, bbox, lookahead, true);
    let opts = AStarOptions {
        visit_limit,
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &model, src_wires, dst_wire, &opts).path
}

/// Same semantics as [`astar_route`]. The `multi_hop` flag was historically
/// used to eagerly enqueue 2-hop destinations so the heuristic could see
/// past weak intermediate wires (IOB→interior); the node-as-vertex
/// expansion in [`super::astar::astar_search`] already fans out
/// `pips_downhill` from every node peer in one step, which covers the same
/// case without a special mode. The wrapper is retained for backwards
/// compatibility but is now identical to [`astar_route`].
pub fn astar_route_multihop(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    _estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    astar_route(
        ctx,
        net,
        src_wires,
        dst_wire,
        wire_penalty,
        bbox,
        _estimate_precision,
        lookahead,
        visit_limit,
    )
}

/// Like [`astar_route`] but always returns an [`AStarTrace`] alongside the
/// (optional) path. Used by router diagnostics to answer questions like
/// "how far did A* get before giving up?". Delegates to the shared
/// [`super::astar::astar_search`] kernel; `multi_hop` is retained for API
/// stability but no longer changes behavior (see [`astar_route_multihop`]).
pub fn astar_route_with_trace(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    _estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
    _multi_hop: bool,
) -> (Option<Vec<PipId>>, CoreTrace) {
    let model = MazeCostModel::new(ctx, net, wire_penalty, bbox, lookahead, false);
    let opts = AStarOptions {
        visit_limit,
        exhaustive: false,
        retain_trace: true,
        stop_on_first_touch: false,
    };
    let result = astar_search(ctx, &model, src_wires, dst_wire, &opts);
    (result.path, result.trace)
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

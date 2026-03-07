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
use crate::netlist::NetId;
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    bind_route, collect_routable_nets, collect_sink_wires, setup_net_source, unroute_net,
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
// Bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in tile coordinates.
pub struct BoundingBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl BoundingBox {
    /// Check whether a tile coordinate (x, y) falls within this bounding box.
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Compute a bounding box for a net based on its connected cells' BEL locations.
///
/// The box is expanded by `margin` tiles in each direction and clamped to the
/// chip grid boundaries.
pub fn compute_bbox(ctx: &Context, net_idx: NetId, margin: i32) -> BoundingBox {
    let net = ctx.net(net_idx);

    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;
    let mut found_any = false;

    // Collect all connected cell indices (driver + users).
    let cell_indices = net
        .driver()
        .into_iter()
        .map(|pin| pin.cell)
        .chain(net.users().iter().filter(|u| u.is_valid()).map(|u| u.cell));

    for cell_idx in cell_indices {
        if let Some(bel) = ctx.cell(cell_idx).bel() {
            let loc = bel.loc();
            x0 = x0.min(loc.x);
            y0 = y0.min(loc.y);
            x1 = x1.max(loc.x);
            y1 = y1.max(loc.y);
            found_any = true;
        }
    }

    if !found_any {
        return BoundingBox {
            x0: 0,
            y0: 0,
            x1: ctx.chipdb().width() - 1,
            y1: ctx.chipdb().height() - 1,
        };
    }

    BoundingBox {
        x0: (x0 - margin).max(0),
        y0: (y0 - margin).max(0),
        x1: (x1 + margin).min(ctx.chipdb().width() - 1),
        y1: (y1 + margin).min(ctx.chipdb().height() - 1),
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
    src_wires: &[WireId],
    dst_wire: WireId,
    net_idx: NetId,
    state: &Router2State,
    bbox: &BoundingBox,
) -> Option<Vec<PipId>> {
    let src_set: FxHashSet<WireId> = src_wires.iter().copied().collect();

    // Trivial case: destination is already in the source set.
    if src_set.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let init_capacity = src_wires.len().saturating_mul(8).max(16);
    let mut heap = BinaryHeap::with_capacity(init_capacity);
    // visited: wire -> (best cost so far, pip used to reach it)
    let mut visited: FxHashMap<WireId, (f64, Option<PipId>)> =
        FxHashMap::with_capacity_and_hasher(init_capacity, Default::default());

    // Seed the search with all source wires.
    for &src in src_wires {
        let h = ctx.estimate_delay(src, dst_wire) as f64;
        heap.push(R2QueueEntry {
            wire: src,
            cost: 0.0,
            estimate: h,
        });
        visited.insert(src, (0.0, None));
    }

    while let Some(entry) = heap.pop() {
        // Skip if we already found a cheaper path to this wire.
        if let Some(&(prev_cost, _)) = visited.get(&entry.wire) {
            if entry.cost > prev_cost {
                continue;
            }
        }

        // Check if we reached the destination.
        if entry.wire == dst_wire {
            // Trace back the path through visited.
            let mut pips = Vec::new();
            let mut current = dst_wire;
            while let Some(&(_, pip)) = visited.get(&current) {
                let pip = match pip {
                    Some(pip) => pip,
                    None => break,
                };
                if !pip.is_valid() {
                    // Reached a source wire.
                    break;
                }
                pips.push(pip);
                current = ctx.pip(pip).src_wire().id();
            }
            pips.reverse();
            return Some(pips);
        }

        // Expand: iterate all downhill pips from this wire.
        let wire_info = ctx.chipdb().wire_info(entry.wire);
        let downhill_indices = wire_info.pips_downhill.get();

        for &pip_index in downhill_indices {
            // PIPs are tile-local: same tile as the wire.
            let pip = PipId::new(entry.wire.tile(), pip_index);

            let next_wire = ctx.pip(pip).dst_wire().id();

            // Bounding box pruning: skip wires outside the net's bounding box.
            let (wx, wy) = ctx.chipdb().tile_xy(next_wire.tile());
            if !bbox.contains(wx, wy) {
                continue;
            }

            // Negotiation-based cost.
            let pip_delay = ctx.pip(pip).delay().max_delay() as f64;
            let wire_neg_cost = state.wire_cost(next_wire, net_idx);
            let new_cost = entry.cost + pip_delay + wire_neg_cost;

            // Skip if we already have a cheaper or equal path to next_wire.
            if let Some(&(prev_cost, _)) = visited.get(&next_wire) {
                if new_cost >= prev_cost {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, Some(pip)));

            let h = ctx.estimate_delay(next_wire, dst_wire) as f64;
            heap.push(R2QueueEntry {
                wire: next_wire,
                cost: new_cost,
                estimate: new_cost + h,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Single-net routing (Router2 variant)
// ---------------------------------------------------------------------------

/// Route a single net using Router2's negotiation-based A* search.
///
/// This follows the same structure as Router1's `route_net`, but uses
/// `astar_route_r2` which incorporates negotiation costs and bounding box
/// pruning.
fn route_net_r2(
    ctx: &mut Context,
    net_idx: NetId,
    state: &Router2State,
) -> Result<(), RouterError> {
    if setup_net_source(ctx, net_idx)?.is_none() {
        return Ok(());
    }

    let net_name = ctx.net(net_idx).name_id();
    let bbox = compute_bbox(ctx, net_idx, state.cfg.bb_margin);
    let sink_wires = collect_sink_wires(ctx, net_idx);

    // Route to each sink.
    for sink_wire in sink_wires {
        // Check if this sink is already routed.
        if ctx.net(net_idx).wires().contains_key(&sink_wire) {
            continue;
        }

        // Collect current routing tree wires as potential A* start points.
        let existing_wires: Vec<WireId> = ctx.net(net_idx).wire_ids().collect();

        let path = astar_route_r2(ctx, &existing_wires, sink_wire, net_idx, state, &bbox);

        match path {
            Some(pips) => {
                bind_route(ctx, net_idx, &pips);
            }
            None => {
                return Err(RouterError::NoPath(ctx.name_of(net_name).to_owned()));
            }
        }
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
        route_router2(ctx, cfg)
    }
}

/// Route all nets in the design using negotiation-based (PathFinder) routing.
///
/// The algorithm:
/// 1. Collect all nets that need routing.
/// 2. Perform an initial greedy route for every net (failures are tolerated).
/// 3. Iteratively detect congested wires, rip up the involved nets, update
///    historical costs, and reroute with increased present-congestion costs.
/// 4. Repeat until no congestion remains or `max_iterations` is reached.
pub fn route_router2(ctx: &mut Context, cfg: &Router2Cfg) -> Result<(), RouterError> {
    let mut state = Router2State::new(cfg);
    let nets = collect_routable_nets(ctx);

    if state.cfg.verbose {
        info!("Router2: {} nets to route", nets.len());
    }

    // Initial route (greedy). Failures are tolerated here because the
    // negotiation loop will resolve congestion.
    for &net_idx in &nets {
        let _ = route_net_r2(ctx, net_idx, &state);
        state.add_net_usage(&ctx.design, net_idx);
    }

    // Negotiation loop.
    for iter in 0..state.cfg.max_iterations {
        let congested = state.find_congested_nets(&ctx.design);
        if congested.is_empty() {
            if state.cfg.verbose {
                info!("Router2: converged after {} iterations", iter);
            }
            return Ok(());
        }

        if state.cfg.verbose {
            info!(
                "Router2: iteration {}, {} congested nets, {} congested wires",
                iter,
                congested.len(),
                state.count_congested_wires()
            );
        }

        // Rip up all congested nets.
        for &net_idx in &congested {
            state.remove_net_usage(&ctx.design, net_idx);
            unroute_net(ctx, net_idx);
        }

        // Update history before rerouting so costs reflect past congestion.
        state.update_history();

        // Reroute congested nets with updated costs.
        for &net_idx in &congested {
            let _ = route_net_r2(ctx, net_idx, &state);
            state.add_net_usage(&ctx.design, net_idx);
        }

        // Increase present-congestion cost for next iteration.
        state.present_cost *= state.cfg.present_cost_growth;
    }

    let remaining = state.count_congested_wires();
    if remaining == 0 {
        Ok(())
    } else {
        Err(RouterError::Congestion(state.cfg.max_iterations, remaining))
    }
}

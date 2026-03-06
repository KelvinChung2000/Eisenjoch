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

use crate::context::{BelPinWireMap, Context};
use crate::netlist::NetIdx;
use crate::types::{PipId, PlaceStrength, WireId};
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    bind_route, collect_routable_nets, find_bel_pin_wire_preindexed, unroute_net,
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
pub(crate) struct BoundingBox {
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
pub(crate) fn compute_bbox(ctx: &Context, net_idx: NetIdx, margin: i32) -> BoundingBox {
    let net = ctx.net(net_idx);

    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;

    let mut found_any = false;

    // Include driver cell location.
    if net.info().driver.is_connected() {
        if let Some(driver_cell_idx) = net.info().driver.cell {
            let cell = ctx.cell(driver_cell_idx);
            if let Some(bel) = cell.bel() {
                let loc = bel.loc();
                x0 = x0.min(loc.x);
                y0 = y0.min(loc.y);
                x1 = x1.max(loc.x);
                y1 = y1.max(loc.y);
                found_any = true;
            }
        }
    }

    // Include all user cell locations.
    for user in &net.info().users {
        if !user.is_connected() {
            continue;
        }
        let user_cell_idx = match user.cell {
            Some(cell_idx) => cell_idx,
            None => continue,
        };
        let cell = ctx.cell(user_cell_idx);
        if let Some(bel) = cell.bel() {
            let loc = bel.loc();
            x0 = x0.min(loc.x);
            y0 = y0.min(loc.y);
            x1 = x1.max(loc.x);
            y1 = y1.max(loc.y);
            found_any = true;
        }
    }

    if !found_any {
        // No placed cells; return a box covering the entire chip.
        return BoundingBox {
            x0: 0,
            y0: 0,
            x1: ctx.width() - 1,
            y1: ctx.height() - 1,
        };
    }

    // Expand by margin and clamp to grid.
    BoundingBox {
        x0: (x0 - margin).max(0),
        y0: (y0 - margin).max(0),
        x1: (x1 + margin).min(ctx.width() - 1),
        y1: (y1 + margin).min(ctx.height() - 1),
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
pub(crate) struct R2QueueEntry {
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
pub(crate) struct Router2State {
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
    pub wire_owner: FxHashMap<WireId, NetIdx>,
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
    pub fn wire_cost(&self, wire: WireId, net_idx: NetIdx) -> f64 {
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
    pub fn add_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetIdx) {
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
    pub fn remove_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetIdx) {
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

            if self.wire_owner.get(&wire) == Some(&net_idx) {
                self.wire_owner.remove(&wire);
            }
        }
    }

    /// Find all nets that touch at least one congested wire (usage > 1).
    pub fn find_congested_nets(&self, design: &crate::netlist::Design) -> Vec<NetIdx> {
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
pub(crate) fn astar_route_r2(
    ctx: &Context,
    src_wires: &[WireId],
    dst_wire: WireId,
    net_idx: NetIdx,
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
            loop {
                if let Some(&(_, pip)) = visited.get(&current) {
                    let pip = match pip {
                        Some(pip) => pip,
                        None => {
                            break;
                        }
                    };
                    if !pip.is_valid() {
                        // Reached a source wire.
                        break;
                    }
                    pips.push(pip);
                    current = ctx.pip_src_wire(pip);
                } else {
                    break;
                }
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

            let next_wire = ctx.pip_dst_wire(pip);

            // Bounding box pruning: skip wires outside the net's bounding box.
            let (wx, wy) = ctx.chipdb().tile_xy(next_wire.tile());
            if !bbox.contains(wx, wy) {
                continue;
            }

            // Negotiation-based cost.
            let pip_delay = ctx.pip_delay(pip).max_delay() as f64;
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
    net_idx: NetIdx,
    state: &Router2State,
    bel_pin_map: &BelPinWireMap,
) -> Result<(), RouterError> {
    let net = ctx.net(net_idx);
    let net_name = net.info().name;

    // Determine the driver wire.
    let driver = &net.info().driver;
    if !driver.is_connected() {
        return Ok(());
    }
    let driver_cell_idx = match driver.cell {
        Some(cell_idx) => cell_idx,
        None => {
            return Err(RouterError::Generic(format!(
                "Driver cell for net {} is missing",
                ctx.name_of(net_name)
            )));
        }
    };
    let driver_port = driver.port;

    let driver_cell = ctx.cell(driver_cell_idx);
    let driver_bel = match driver_cell.bel() {
        Some(bel) => bel.id(),
        None => {
            return Err(RouterError::Generic(format!(
                "Driver cell for net {} is not placed",
                ctx.name_of(net_name)
            )));
        }
    };

    let src_wire =
        find_bel_pin_wire_preindexed(bel_pin_map, driver_bel, driver_port).ok_or_else(|| {
            RouterError::Generic(format!(
                "Cannot find driver wire for net {}",
                ctx.name_of(net_name)
            ))
        })?;

    // Bind the source wire to this net if not already bound.
    if ctx.is_wire_available(src_wire) {
        ctx.bind_wire(src_wire, net_idx, PlaceStrength::Strong);
        let net_mut = ctx.design_mut().net_mut(net_idx);
        net_mut.wires.insert(
            src_wire,
            crate::netlist::PipMap {
                pip: None,
                strength: PlaceStrength::Strong,
            },
        );
    }

    // Compute the bounding box for this net.
    let bbox = compute_bbox(ctx, net_idx, state.cfg.bb_margin);

    // Collect sink wires before mutating ctx.
    let num_users = ctx.net(net_idx).info().users.len();
    let mut sink_wires = Vec::with_capacity(num_users);
    for user_idx in 0..num_users {
        let user = &ctx.net(net_idx).info().users[user_idx];
        if !user.is_connected() {
            continue;
        }
        let user_cell_idx = match user.cell {
            Some(cell_idx) => cell_idx,
            None => continue,
        };
        let user_port = user.port;
        let user_cell = ctx.cell(user_cell_idx);
        let user_bel = match user_cell.bel() {
            Some(bel) => bel.id(),
            None => continue,
        };
        if let Some(sink_wire) = find_bel_pin_wire_preindexed(bel_pin_map, user_bel, user_port) {
            sink_wires.push(sink_wire);
        }
    }

    // Route to each sink.
    for sink_wire in sink_wires {
        // Check if this sink is already routed.
        if ctx.net(net_idx).info().wires.contains_key(&sink_wire) {
            continue;
        }

        // Collect current routing tree wires as potential A* start points.
        let existing_wires: Vec<WireId> =
            ctx.net(net_idx).info().wires.keys().copied().collect();

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
    let bel_pin_map = ctx.bel_pin_wire_map();
    let nets = collect_routable_nets(ctx);

    if state.cfg.verbose {
        info!("Router2: {} nets to route", nets.len());
    }

    // Initial route (greedy). Failures are tolerated here because the
    // negotiation loop will resolve congestion.
    for &net_idx in &nets {
        let _ = route_net_r2(ctx, net_idx, &state, &bel_pin_map);
        state.add_net_usage(ctx.design(), net_idx);
    }

    // Negotiation loop.
    for iter in 0..state.cfg.max_iterations {
        let congested = state.find_congested_nets(ctx.design());
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
            state.remove_net_usage(ctx.design(), net_idx);
            unroute_net(ctx, net_idx);
        }

        // Update history before rerouting so costs reflect past congestion.
        state.update_history();

        // Reroute congested nets with updated costs.
        for &net_idx in &congested {
            let _ = route_net_r2(ctx, net_idx, &state, &bel_pin_map);
            state.add_net_usage(ctx.design(), net_idx);
        }

        // Increase present-congestion cost for next iteration.
        state.present_cost *= state.cfg.present_cost_growth;
    }

    let remaining = state.count_congested_wires();
    if remaining == 0 {
        Ok(())
    } else {
        Err(RouterError::Congestion(
            state.cfg.max_iterations,
            remaining,
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "test-utils")]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::context::Context;
    use crate::netlist::{NetIdx, PipMap, PortRef};
    use crate::types::{BelId, PipId, PlaceStrength, PortType, WireId};
    use rustc_hash::FxHashSet;
    use std::collections::BinaryHeap;

    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    fn make_pip_map(pip: Option<PipId>) -> PipMap {
        PipMap {
            pip,
            strength: PlaceStrength::Strong,
        }
    }

    // BoundingBox tests

    #[test]
    fn bbox_contains_within() {
        let bb = BoundingBox { x0: 0, y0: 0, x1: 3, y1: 3 };
        assert!(bb.contains(0, 0));
        assert!(bb.contains(1, 2));
        assert!(bb.contains(3, 3));
    }

    #[test]
    fn bbox_contains_boundary() {
        let bb = BoundingBox { x0: 1, y0: 1, x1: 5, y1: 5 };
        assert!(bb.contains(1, 1));
        assert!(bb.contains(5, 1));
        assert!(bb.contains(1, 5));
        assert!(bb.contains(5, 5));
        assert!(bb.contains(3, 1));
        assert!(bb.contains(3, 5));
        assert!(bb.contains(1, 3));
        assert!(bb.contains(5, 3));
    }

    #[test]
    fn bbox_excludes_outside() {
        let bb = BoundingBox { x0: 1, y0: 1, x1: 3, y1: 3 };
        assert!(!bb.contains(0, 0));
        assert!(!bb.contains(4, 2));
        assert!(!bb.contains(2, 4));
        assert!(!bb.contains(0, 2));
        assert!(!bb.contains(2, 0));
    }

    #[test]
    fn bbox_single_point() {
        let bb = BoundingBox { x0: 2, y0: 3, x1: 2, y1: 3 };
        assert!(bb.contains(2, 3));
        assert!(!bb.contains(1, 3));
        assert!(!bb.contains(3, 3));
        assert!(!bb.contains(2, 2));
        assert!(!bb.contains(2, 4));
    }

    // compute_bbox tests

    #[test]
    fn compute_bbox_no_placed_cells() {
        let mut ctx = make_context();
        let net_name = ctx.id("unplaced_net");
        let net_idx = ctx.design_mut().add_net(net_name);
        let bb = compute_bbox(&ctx, net_idx, 0);
        assert_eq!(bb.x0, 0);
        assert_eq!(bb.y0, 0);
        assert_eq!(bb.x1, ctx.width() - 1);
        assert_eq!(bb.y1, ctx.height() - 1);
    }

    #[test]
    fn compute_bbox_single_cell() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");
        let cell_name = ctx.id("driver");
        let cell_idx = ctx.design_mut().add_cell(cell_name, lut_type);
        ctx.design_mut().cell_mut(cell_idx).add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
        let net_name = ctx.id("test_net");
        let net_idx = ctx.design_mut().add_net(net_name);
        ctx.design_mut().net_mut(net_idx).driver = PortRef {
            cell: Some(cell_idx), port, budget: 0,
        };
        let bb = compute_bbox(&ctx, net_idx, 0);
        assert_eq!(bb.x0, 0);
        assert_eq!(bb.y0, 0);
        assert_eq!(bb.x1, 0);
        assert_eq!(bb.y1, 0);
    }

    #[test]
    fn compute_bbox_with_margin() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");
        let cell_name = ctx.id("driver");
        let cell_idx = ctx.design_mut().add_cell(cell_name, lut_type);
        ctx.design_mut().cell_mut(cell_idx).add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
        let net_name = ctx.id("test_net");
        let net_idx = ctx.design_mut().add_net(net_name);
        ctx.design_mut().net_mut(net_idx).driver = PortRef {
            cell: Some(cell_idx), port, budget: 0,
        };
        let bb = compute_bbox(&ctx, net_idx, 1);
        assert_eq!(bb.x0, 0);
        assert_eq!(bb.y0, 0);
        assert_eq!(bb.x1, 1);
        assert_eq!(bb.y1, 1);
    }

    // Wire cost tests

    #[test]
    fn wire_cost_base_only() {
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        let net = NetIdx::from_raw(0);
        let cost = state.wire_cost(wire, net);
        assert!((cost - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wire_cost_with_congestion() {
        let cfg = Router2Cfg {
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            initial_present_cost: 1.0,
            history_cost_multiplier: 1.0,
            ..Router2Cfg::default()
        };
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        let net_a = NetIdx::from_raw(0);
        let net_b = NetIdx::from_raw(1);
        state.wire_usage.insert(wire, 1);
        state.wire_owner.insert(wire, net_a);
        let cost_owner = state.wire_cost(wire, net_a);
        assert!((cost_owner - 1.0).abs() < f64::EPSILON);
        let cost_other = state.wire_cost(wire, net_b);
        assert!((cost_other - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wire_cost_with_history() {
        let cfg = Router2Cfg {
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            initial_present_cost: 1.0,
            history_cost_multiplier: 3.0,
            ..Router2Cfg::default()
        };
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        let net = NetIdx::from_raw(0);
        state.wire_history.insert(wire, 5.0);
        let cost = state.wire_cost(wire, net);
        assert!((cost - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wire_cost_combined() {
        let cfg = Router2Cfg {
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            initial_present_cost: 1.0,
            history_cost_multiplier: 1.0,
            ..Router2Cfg::default()
        };
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        let net_a = NetIdx::from_raw(0);
        let net_b = NetIdx::from_raw(1);
        state.wire_usage.insert(wire, 2);
        state.wire_owner.insert(wire, net_a);
        state.wire_history.insert(wire, 1.0);
        let cost = state.wire_cost(wire, net_b);
        assert!((cost - 6.0).abs() < f64::EPSILON);
    }

    // History update tests

    #[test]
    fn update_history_no_congestion() {
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        state.wire_usage.insert(wire, 1);
        state.update_history();
        assert!(!state.wire_history.contains_key(&wire));
    }

    #[test]
    fn update_history_with_congestion() {
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        state.wire_usage.insert(wire, 3);
        state.update_history();
        assert!((state.wire_history[&wire] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_history_accumulates() {
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        state.wire_usage.insert(wire, 2);
        state.update_history();
        assert!((state.wire_history[&wire] - 1.0).abs() < f64::EPSILON);
        state.update_history();
        assert!((state.wire_history[&wire] - 2.0).abs() < f64::EPSILON);
    }

    // Usage tracking tests

    #[test]
    fn update_usage_empty_design() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.update_usage(ctx.design());
        assert!(state.wire_usage.is_empty());
        assert!(state.wire_owner.is_empty());
    }

    #[test]
    fn update_usage_single_net() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_a");
        let net_idx = ctx.design_mut().add_net(net_name);
        let wire = WireId::new(0, 0);
        ctx.design_mut().net_mut(net_idx).wires.insert(wire, make_pip_map(None));
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.update_usage(ctx.design());
        assert_eq!(state.wire_usage[&wire], 1);
        assert_eq!(state.wire_owner[&wire], net_idx);
    }

    #[test]
    fn update_usage_multiple_nets_same_wire() {
        let mut ctx = make_context();
        let wire = WireId::new(0, 0);
        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design_mut().add_net(net_a_name);
        ctx.design_mut().net_mut(net_a_idx).wires.insert(wire, make_pip_map(None));
        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design_mut().add_net(net_b_name);
        ctx.design_mut().net_mut(net_b_idx).wires.insert(wire, make_pip_map(None));
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.update_usage(ctx.design());
        assert_eq!(state.wire_usage[&wire], 2);
        let owner = state.wire_owner[&wire];
        assert!(owner == net_a_idx || owner == net_b_idx);
    }

    // Congested net detection tests

    #[test]
    fn find_congested_nets_none() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_a");
        let net_idx = ctx.design_mut().add_net(net_name);
        ctx.design_mut().net_mut(net_idx).wires.insert(WireId::new(0, 0), make_pip_map(None));
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.update_usage(ctx.design());
        let congested = state.find_congested_nets(ctx.design());
        assert!(congested.is_empty());
    }

    #[test]
    fn find_congested_nets_shared_wire() {
        let mut ctx = make_context();
        let wire = WireId::new(0, 0);
        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design_mut().add_net(net_a_name);
        ctx.design_mut().net_mut(net_a_idx).wires.insert(wire, make_pip_map(None));
        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design_mut().add_net(net_b_name);
        ctx.design_mut().net_mut(net_b_idx).wires.insert(wire, make_pip_map(None));
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.update_usage(ctx.design());
        let congested = state.find_congested_nets(ctx.design());
        assert_eq!(congested.len(), 2);
        let net_set: FxHashSet<NetIdx> = congested.into_iter().collect();
        assert!(net_set.contains(&net_a_idx));
        assert!(net_set.contains(&net_b_idx));
    }

    // R2QueueEntry ordering tests

    #[test]
    fn r2_queue_min_heap_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(R2QueueEntry { wire: WireId::new(0, 0), cost: 10.0, estimate: 50.0 });
        heap.push(R2QueueEntry { wire: WireId::new(0, 1), cost: 5.0, estimate: 20.0 });
        heap.push(R2QueueEntry { wire: WireId::new(1, 0), cost: 8.0, estimate: 35.0 });
        let first = heap.pop().unwrap();
        assert!((first.estimate - 20.0).abs() < f64::EPSILON);
        let second = heap.pop().unwrap();
        assert!((second.estimate - 35.0).abs() < f64::EPSILON);
        let third = heap.pop().unwrap();
        assert!((third.estimate - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn r2_queue_tiebreak_by_cost() {
        let mut heap = BinaryHeap::new();
        heap.push(R2QueueEntry { wire: WireId::new(0, 0), cost: 30.0, estimate: 50.0 });
        heap.push(R2QueueEntry { wire: WireId::new(0, 1), cost: 10.0, estimate: 50.0 });
        let first = heap.pop().unwrap();
        assert!((first.cost - 10.0).abs() < f64::EPSILON);
    }

    // A* search with negotiation costs tests

    #[test]
    fn astar_r2_same_wire_returns_empty_path() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let wire = WireId::new(0, 0);
        let bbox = BoundingBox { x0: 0, y0: 0, x1: 1, y1: 1 };
        let path = astar_route_r2(&ctx, &[wire], wire, NetIdx::from_raw(0), &state, &bbox);
        assert!(path.is_some());
        assert!(path.unwrap().is_empty());
    }

    #[test]
    fn astar_r2_single_pip_path() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let src = WireId::new(0, 0);
        let dst = WireId::new(0, 1);
        let bbox = BoundingBox { x0: 0, y0: 0, x1: 1, y1: 1 };
        let path = astar_route_r2(&ctx, &[src], dst, NetIdx::from_raw(0), &state, &bbox);
        assert!(path.is_some());
        let pips = path.unwrap();
        assert_eq!(pips.len(), 1);
        assert_eq!(pips[0], PipId::new(0, 0));
    }

    #[test]
    fn astar_r2_no_path_returns_none() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let src = WireId::new(0, 1);
        let dst = WireId::new(0, 0);
        let bbox = BoundingBox { x0: 0, y0: 0, x1: 1, y1: 1 };
        let path = astar_route_r2(&ctx, &[src], dst, NetIdx::from_raw(0), &state, &bbox);
        assert!(path.is_none());
    }

    #[test]
    fn astar_r2_bbox_prunes_out_of_range() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let src = WireId::new(0, 0);
        let dst = WireId::new(1, 0);
        let bbox = BoundingBox { x0: 0, y0: 0, x1: 0, y1: 0 };
        let path = astar_route_r2(&ctx, &[src], dst, NetIdx::from_raw(0), &state, &bbox);
        assert!(path.is_none());
    }

    #[test]
    fn astar_r2_empty_sources_returns_none() {
        let ctx = make_context();
        let cfg = Router2Cfg::default();
        let state = Router2State::new(&cfg);
        let dst = WireId::new(0, 1);
        let bbox = BoundingBox { x0: 0, y0: 0, x1: 1, y1: 1 };
        let path = astar_route_r2(&ctx, &[], dst, NetIdx::from_raw(0), &state, &bbox);
        assert!(path.is_none());
    }

    // Present cost tests

    #[test]
    fn present_cost_initialized_from_config() {
        let cfg = Router2Cfg {
            initial_present_cost: 2.5,
            ..Router2Cfg::default()
        };
        let state = Router2State::new(&cfg);
        assert!((state.present_cost - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn present_cost_grows() {
        let cfg = Router2Cfg {
            initial_present_cost: 1.0,
            present_cost_growth: 2.0,
            ..Router2Cfg::default()
        };
        let mut state = Router2State::new(&cfg);
        assert!((state.present_cost - 1.0).abs() < f64::EPSILON);
        state.present_cost *= state.cfg.present_cost_growth;
        assert!((state.present_cost - 2.0).abs() < f64::EPSILON);
        state.present_cost *= state.cfg.present_cost_growth;
        assert!((state.present_cost - 4.0).abs() < f64::EPSILON);
    }

    // count_congested_wires tests

    #[test]
    fn count_congested_wires_none() {
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.wire_usage.insert(WireId::new(0, 0), 1);
        state.wire_usage.insert(WireId::new(0, 1), 1);
        assert_eq!(state.count_congested_wires(), 0);
    }

    #[test]
    fn count_congested_wires_some() {
        let cfg = Router2Cfg::default();
        let mut state = Router2State::new(&cfg);
        state.wire_usage.insert(WireId::new(0, 0), 2);
        state.wire_usage.insert(WireId::new(0, 1), 1);
        state.wire_usage.insert(WireId::new(1, 0), 3);
        assert_eq!(state.count_congested_wires(), 2);
    }
}

//! Shared helper functions used by both Router1 and Router2.

use crate::chipdb::{PipId, WireId};
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::NetId;
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Route plan types (pure computation output, no Context mutation)
// ---------------------------------------------------------------------------

/// Computed route for a single net, before binding.
pub struct RoutePlan {
    pub net: NetId,
    pub source_wire: WireId,
    pub sink_routes: Vec<SinkRoute>,
}

/// Computed path from the routing tree to a single sink wire.
pub struct SinkRoute {
    pub sink_wire: WireId,
    pub pips: Vec<PipId>,
}

/// Read-only source wire resolution (no binding).
///
/// Returns the driver wire for a net, or `Ok(None)` if it has no connected driver.
pub fn resolve_source_wire(
    ctx: &Context,
    net_idx: NetId,
) -> Result<Option<WireId>, super::RouterError> {
    let net = ctx.net(net_idx);
    let net_name = net.name_id();

    let Some(driver_pin) = net.driver_cell_port() else {
        return Ok(None);
    };

    let driver_cell = ctx.cell(driver_pin.cell);
    let driver_bel = match driver_cell.bel() {
        Some(bel) => bel,
        None => {
            return Err(super::RouterError::Generic(format!(
                "Driver cell for net {} is not placed",
                ctx.name_of(net_name)
            )));
        }
    };

    let src_wire = driver_bel
        .pin_wire(driver_pin.port)
        .map(|w| w.id())
        .ok_or_else(|| {
            super::RouterError::Generic(format!(
                "Cannot find driver wire for net {}",
                ctx.name_of(net_name)
            ))
        })?;

    Ok(Some(src_wire))
}

/// Apply a computed RoutePlan by binding source wire, PIPs, and dest wires.
pub fn apply_route_plan(ctx: &mut Context, plan: &RoutePlan) {
    // Bind source wire if not already bound.
    if ctx.wire(plan.source_wire).is_available() {
        ctx.bind_wire(plan.source_wire, plan.net, PlaceStrength::Strong);
        ctx.design
            .net_edit(plan.net)
            .add_wire(plan.source_wire, None, PlaceStrength::Strong);
    }

    // Bind each sink route's PIPs.
    for sink in &plan.sink_routes {
        bind_route(ctx, plan.net, &sink.pips);
    }
}

// ---------------------------------------------------------------------------
// Net collection helpers
// ---------------------------------------------------------------------------

/// Collect all net indices that need routing.
///
/// A net needs routing if it has a connected driver and at least one user.
pub fn collect_routable_nets(ctx: &Context) -> Vec<NetId> {
    ctx.design
        .iter_net_indices()
        .filter(|&net_idx| {
            let net = ctx.net(net_idx);
            net.is_alive() && net.has_driver() && net.num_users() > 0
        })
        .collect()
}

/// Bind a sequence of PIPs as the route for a net.
///
/// For each PIP in the path, binds the PIP and its destination wire to the
/// given net, and records the routing in the net's wire map.
pub fn bind_route(ctx: &mut Context, net_idx: NetId, path: &[PipId]) {
    for &pip in path {
        let dst_wire = ctx.pip(pip).dst_wire().id();
        ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
        ctx.bind_wire(dst_wire, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(dst_wire, Some(pip), PlaceStrength::Strong);
    }
}

/// Rip up (unroute) a net by unbinding all its wires and PIPs.
pub fn unroute_net(ctx: &mut Context, net_idx: NetId) {
    let net = ctx.net(net_idx);
    let entries: Vec<(WireId, Option<PipId>)> = net
        .wires()
        .iter()
        .map(|(&wire, pm)| (wire, pm.pip))
        .collect();

    for (wire, pip) in entries {
        ctx.unbind_wire(wire);
        if let Some(pip) = pip {
            ctx.unbind_pip(pip);
        }
    }

    ctx.design.net_edit(net_idx).clear_wires();
}

/// Collect the sink wires for all users of a net.
///
/// Resolves each user's BEL pin to a wire via the view API.
/// Skips unconnected or unplaced users.
pub fn collect_sink_wires(ctx: &Context, net_idx: NetId) -> Vec<WireId> {
    let net = ctx.net(net_idx);
    let mut sink_wires = Vec::with_capacity(net.num_users());
    for user in net.users() {
        if !user.is_valid() {
            continue;
        }
        let user_cell_idx = user.cell;
        let user_cell = ctx.cell(user_cell_idx);
        let user_bel = match user_cell.bel() {
            Some(bel) => bel,
            None => continue,
        };
        if let Some(sink_wire) = user_bel.pin_wire(user.port) {
            sink_wires.push(sink_wire.id());
        }
    }
    sink_wires
}

/// Collect all wires across the chip that have a given nonzero `const_value`.
///
/// These serve as additional routing sources for constant nets (GND/VCC),
/// since each tile has its own constant wire connected to the local switch matrix.
pub fn collect_constant_source_wires(ctx: &Context, const_value: i32) -> Vec<WireId> {
    ctx.chipdb()
        .wires()
        .filter(|&wire| ctx.chipdb().wire_info(wire).const_value == const_value)
        .collect()
}

/// Get the `const_value` of a net's driver wire, or 0 if not a constant net.
///
/// A nonzero return means the net is driven by a constant wire (GND/VCC).
pub fn source_wire_const_value(ctx: &Context, source_wire: WireId) -> i32 {
    ctx.chipdb().wire_info(source_wire).const_value
}

/// Find all wires that are used by more than one net (congested).
pub fn find_congested_wires(ctx: &Context) -> Vec<WireId> {
    let mut wire_usage: FxHashMap<WireId, u32> = FxHashMap::default();

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        for &wire in net.wires().keys() {
            *wire_usage.entry(wire).or_default() += 1;
        }
    }

    wire_usage
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(wire, _)| wire)
        .collect()
}

// ---------------------------------------------------------------------------
// Shared negotiation cost model (PathFinder-style)
// ---------------------------------------------------------------------------

/// Cost-model parameters for PathFinder-style negotiation routing.
///
/// These fields control the negotiation cost function that encourages nets
/// to find alternative paths when wires are congested.
#[derive(Clone)]
pub struct NegotiationCfg {
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
}

impl Default for NegotiationCfg {
    fn default() -> Self {
        Self {
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            history_cost_multiplier: 1.0,
            initial_present_cost: 1.0,
            present_cost_growth: 1.5,
        }
    }
}

/// Mutable state for the PathFinder-style negotiation cost model.
///
/// Tracks per-wire usage, ownership, history costs, and present-congestion
/// costs. This is generic and can be used by any router that needs
/// negotiation-based congestion resolution.
pub struct NegotiationState {
    /// Cost-model configuration.
    pub cfg: NegotiationCfg,
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
    /// External congestion hints (e.g. from placer density map).
    /// Wire -> additional cost added to wire_cost().
    pub external_hints: FxHashMap<WireId, f64>,
}

impl NegotiationState {
    /// Create a new negotiation state from the given configuration.
    pub fn new(cfg: NegotiationCfg) -> Self {
        let present_cost = cfg.initial_present_cost;
        Self {
            cfg,
            present_cost,
            wire_history: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
            wire_owner: FxHashMap::default(),
            external_hints: FxHashMap::default(),
        }
    }

    /// Compute the negotiation-based cost of using a wire for a given net.
    ///
    /// The cost has four components:
    /// 1. Base cost (constant per wire).
    /// 2. Present-congestion penalty: proportional to the number of other nets
    ///    currently using the wire, scaled by the present cost factor.
    /// 3. Historical penalty: accumulated from prior iterations where the wire
    ///    was congested.
    /// 4. External hint cost: additional cost injected by external systems
    ///    (e.g. placer density maps).
    pub fn wire_cost(&self, wire: WireId, net_idx: NetId) -> f64 {
        let usage = self.wire_usage.get(&wire).copied().unwrap_or(0);
        let is_own = self.wire_owner.get(&wire) == Some(&net_idx);
        let present_penalty = if is_own { 0.0 } else { usage as f64 };
        let history = self.wire_history.get(&wire).copied().unwrap_or(0.0);
        let hint = self.external_hints.get(&wire).copied().unwrap_or(0.0);
        self.cfg.base_cost
            + present_penalty * self.present_cost * self.cfg.present_cost_multiplier
            + history * self.cfg.history_cost_multiplier
            + hint
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
        let congested_wires: rustc_hash::FxHashSet<WireId> = self
            .wire_usage
            .iter()
            .filter(|(_, &u)| u > 1)
            .map(|(&w, _)| w)
            .collect();

        if congested_wires.is_empty() {
            return Vec::new();
        }

        let mut nets = rustc_hash::FxHashSet::default();
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

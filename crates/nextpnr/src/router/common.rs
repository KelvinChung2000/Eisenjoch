//! Shared helper functions used by both Router1 and Router2.

use crate::context::Context;
use crate::chipdb::{PipId, WireId};
use crate::common::PlaceStrength;
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
pub(crate) fn collect_sink_wires(ctx: &Context, net_idx: NetId) -> Vec<WireId> {
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

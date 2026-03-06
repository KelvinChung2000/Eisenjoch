//! Shared helper functions used by both Router1 and Router2.

use crate::context::{BelPinWireMap, Context};
use crate::netlist::{NetIdx, PipMap};
use crate::types::{BelId, IdString, PipId, PlaceStrength, WireId};
use rustc_hash::FxHashMap;

/// Build a lookup map from (BelId, pin name) to the corresponding wire.
pub(crate) fn build_bel_pin_wire_map(ctx: &Context) -> BelPinWireMap {
    ctx.bel_pin_wire_map()
}

/// Look up a BEL pin wire using a pre-built map.
#[inline]
pub(crate) fn find_bel_pin_wire_preindexed(
    bel_pin_map: &BelPinWireMap,
    bel: BelId,
    port_name: IdString,
) -> Option<WireId> {
    bel_pin_map.get(&(bel, port_name)).copied()
}

/// Collect all net indices that need routing.
///
/// A net needs routing if it has a connected driver and at least one user.
pub(crate) fn collect_routable_nets(ctx: &Context) -> Vec<NetIdx> {
    let mut result = Vec::new();
    for net_idx in ctx.design().iter_net_indices() {
        let net = ctx.net(net_idx);
        let info = net.info();
        if info.alive && info.has_driver() && info.num_users() > 0 {
            result.push(net_idx);
        }
    }
    result
}

/// Bind a sequence of PIPs as the route for a net.
///
/// For each PIP in the path, binds the PIP and its destination wire to the
/// given net, and records the routing in the net's wire map.
pub(crate) fn bind_route(ctx: &mut Context, net_idx: NetIdx, path: &[PipId]) {
    for &pip in path {
        let dst_wire = ctx.pip_dst_wire(pip);
        ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
        ctx.bind_wire(dst_wire, net_idx, PlaceStrength::Strong);
        let net = ctx.design_mut().net_mut(net_idx);
        net.wires.insert(
            dst_wire,
            PipMap {
                pip: Some(pip),
                strength: PlaceStrength::Strong,
            },
        );
    }
}

/// Rip up (unroute) a net by unbinding all its wires and PIPs.
pub(crate) fn unroute_net(ctx: &mut Context, net_idx: NetIdx) {
    let net = ctx.net(net_idx);
    let entries: Vec<(WireId, Option<PipId>)> = net
        .info()
        .wires
        .iter()
        .map(|(&wire, pm)| (wire, pm.pip))
        .collect();

    for (wire, pip) in entries {
        ctx.unbind_wire(wire);
        if let Some(pip) = pip {
            ctx.unbind_pip(pip);
        }
    }

    ctx.design_mut().net_mut(net_idx).wires.clear();
}

/// Find all wires that are used by more than one net (congested).
pub(crate) fn find_congested_wires(ctx: &Context) -> Vec<WireId> {
    let mut wire_usage: FxHashMap<WireId, u32> = FxHashMap::default();

    for net_idx in ctx.design().iter_net_indices() {
        let net = ctx.net(net_idx);
        let info = net.info();
        if !info.alive {
            continue;
        }
        for &wire in info.wires.keys() {
            *wire_usage.entry(wire).or_default() += 1;
        }
    }

    wire_usage
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(wire, _)| wire)
        .collect()
}

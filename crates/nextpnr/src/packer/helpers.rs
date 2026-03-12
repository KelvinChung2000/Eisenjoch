//! Utility functions for netlist manipulation during packing.

use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellId, NetId, PortType};

/// Disconnect a port from its net.
///
/// Removes the port from the net's driver or users list and clears the port's
/// net reference. If the port is not connected or does not exist, this is a
/// no-op.
#[cfg(feature = "test-utils")]
pub fn disconnect_port(ctx: &mut Context, cell: CellId, port: IdString) {
    let cell_info = ctx.design.cell(cell);
    let Some(net_idx) = cell_info.port_net(port) else {
        return;
    };
    let user_idx = cell_info.port_user_idx(port);

    if let Some(user_idx) = user_idx {
        ctx.design
            .net_edit(net_idx)
            .disconnect_user(user_idx as usize);
    } else {
        ctx.design.net_edit(net_idx).clear_driver();
    }

    ctx.design.cell_edit(cell).set_port_net(port, None, None);
}

/// Connect a port to a net.
///
/// If the port is an output or inout, it becomes the net's driver.
/// If the port is an input, it is added to the net's users list.
pub fn connect_port(ctx: &mut Context, cell: CellId, port: IdString, net: NetId) {
    let port_type = ctx
        .design
        .cell(cell)
        .port_type(port)
        .unwrap_or(PortType::In);

    if matches!(port_type, PortType::Out | PortType::InOut) {
        ctx.design.net_edit(net).set_driver(cell, port);
        ctx.design
            .cell_edit(cell)
            .set_port_net(port, Some(net), None);
    } else {
        let idx = ctx.design.net_edit(net).add_user(cell, port);
        ctx.design
            .cell_edit(cell)
            .set_port_net(port, Some(net), Some(idx));
    }
}

/// Get the net connected to a cell port, if any.
#[cfg(feature = "test-utils")]
pub fn get_net_for_port(ctx: &Context, cell: CellId, port: IdString) -> Option<NetId> {
    ctx.design.cell(cell).port_net(port)
}

/// Check if a net has exactly one connected user.
#[cfg(feature = "test-utils")]
pub fn is_single_fanout(ctx: &Context, net: NetId) -> bool {
    let net_info = ctx.design.net(net);
    net_info.users().iter().filter(|u| u.is_connected()).count() == 1
}

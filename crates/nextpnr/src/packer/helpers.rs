//! Utility functions for netlist manipulation during packing.

use crate::context::Context;
use crate::netlist::{CellIdx, NetIdx, PortRef};
use crate::types::{IdString, PortType};

/// Disconnect a port from its net.
///
/// Removes the port from the net's driver or users list and clears the port's
/// net reference. If the port is not connected or does not exist, this is a
/// no-op.
#[allow(dead_code)]
pub(crate) fn disconnect_port(ctx: &mut Context, cell: CellIdx, port: IdString) {
    let cell_info = ctx.design().cell(cell);
    let (net_idx, user_idx) = match cell_info.port(port) {
        Some(port_info) if port_info.net.is_some() => (port_info.net, port_info.user_idx),
        _ => return,
    };
    let net_idx = match net_idx {
        Some(net_idx) => net_idx,
        None => return,
    };

    if let Some(user_idx) = user_idx {
        // This port was a user (sink) of the net.
        ctx.net_edit(net_idx).disconnect_user(user_idx as usize);
    } else {
        // This port was the driver of the net.
        ctx.net_edit(net_idx).clear_driver();
    }

    // Clear the port's net reference.
    ctx.cell_edit(cell).set_port_net(port, None, None);
}

/// Connect a port to a net.
///
/// If the port is an output or inout, it becomes the net's driver.
/// If the port is an input, it is added to the net's users list.
pub(crate) fn connect_port(ctx: &mut Context, cell: CellIdx, port: IdString, net: NetIdx) {
    let port_type = ctx
        .design()
        .cell(cell)
        .port(port)
        .map(|p| p.port_type)
        .unwrap_or(PortType::In);

    if port_type == PortType::Out || port_type == PortType::InOut {
        // Set as driver.
        ctx.net_edit(net).set_driver_raw(PortRef::connected(cell, port, 0));
        ctx.cell_edit(cell).set_port_net(port, Some(net), None);
    } else {
        // Add as user.
        let idx = ctx.net_edit(net).add_user(cell, port);
        ctx.cell_edit(cell).set_port_net(port, Some(net), Some(idx));
    }
}

/// Rename a cell port, preserving its connection state.
///
/// If the old port does not exist, this is a no-op.
#[allow(dead_code)]
pub(crate) fn rename_port(ctx: &mut Context, cell: CellIdx, old_name: IdString, new_name: IdString) {
    ctx.cell_edit(cell).rename_port(old_name, new_name);
}

/// Get the net connected to a cell port, if any.
#[allow(dead_code)]
pub(crate) fn get_net_for_port(ctx: &Context, cell: CellIdx, port: IdString) -> Option<NetIdx> {
    ctx.design().cell(cell).port(port).and_then(|p| p.net)
}

/// Check if a net has exactly one connected user.
#[allow(dead_code)]
pub(crate) fn is_single_fanout(ctx: &Context, net: NetIdx) -> bool {
    let net_info = ctx.design().net(net);
    net_info.users.iter().filter(|u| u.is_connected()).count() == 1
}

/// Remove a cell from the design by name (marks it dead).
#[allow(dead_code)]
pub(crate) fn remove_cell(ctx: &mut Context, cell_name: IdString) {
    ctx.remove_cell(cell_name);
}

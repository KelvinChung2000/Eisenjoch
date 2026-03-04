//! Utility functions for netlist manipulation during packing.

use npnr_netlist::{CellIdx, Design, NetIdx, PortRef};
use npnr_types::{IdString, PortType};

/// Disconnect a port from its net.
///
/// Removes the port from the net's driver or users list and clears the port's
/// net reference. If the port is not connected or does not exist, this is a
/// no-op.
pub fn disconnect_port(design: &mut Design, cell: CellIdx, port: IdString) {
    let cell_info = design.cell(cell);
    let (net_idx, user_idx) = match cell_info.port(port) {
        Some(port_info) if port_info.net.is_some() => (port_info.net, port_info.user_idx),
        _ => return,
    };

    if user_idx >= 0 {
        // This port was a user (sink) of the net.
        let net = design.net_mut(net_idx);
        if (user_idx as usize) < net.users.len() {
            net.users[user_idx as usize] = PortRef::unconnected();
        }
    } else {
        // This port was the driver of the net.
        let net = design.net_mut(net_idx);
        net.driver = PortRef::unconnected();
    }

    // Clear the port's net reference.
    if let Some(port_info) = design.cell_mut(cell).port_mut(port) {
        port_info.net = NetIdx::NONE;
        port_info.user_idx = -1;
    }
}

/// Connect a port to a net.
///
/// If the port is an output or inout, it becomes the net's driver.
/// If the port is an input, it is added to the net's users list.
pub fn connect_port(design: &mut Design, cell: CellIdx, port: IdString, net: NetIdx) {
    let port_type = design
        .cell(cell)
        .port(port)
        .map(|p| p.port_type)
        .unwrap_or(PortType::In);

    if port_type == PortType::Out || port_type == PortType::InOut {
        // Set as driver.
        let net_info = design.net_mut(net);
        net_info.driver = PortRef {
            cell,
            port,
            budget: 0,
        };
        if let Some(p) = design.cell_mut(cell).port_mut(port) {
            p.net = net;
            p.user_idx = -1;
        }
    } else {
        // Add as user.
        let net_info = design.net_mut(net);
        let idx = net_info.users.len() as i32;
        net_info.users.push(PortRef {
            cell,
            port,
            budget: 0,
        });
        if let Some(p) = design.cell_mut(cell).port_mut(port) {
            p.net = net;
            p.user_idx = idx;
        }
    }
}

/// Rename a cell port, preserving its connection state.
///
/// If the old port does not exist, this is a no-op.
pub fn rename_port(design: &mut Design, cell: CellIdx, old_name: IdString, new_name: IdString) {
    let cell_info = design.cell_mut(cell);
    if let Some(mut port_info) = cell_info.ports.remove(&old_name) {
        port_info.name = new_name;
        cell_info.ports.insert(new_name, port_info);
    }
}

/// Get the net connected to a cell port, if any.
pub fn get_net_for_port(design: &Design, cell: CellIdx, port: IdString) -> Option<NetIdx> {
    design
        .cell(cell)
        .port(port)
        .and_then(|p| if p.net.is_some() { Some(p.net) } else { None })
}

/// Check if a net has exactly one connected user.
pub fn is_single_fanout(design: &Design, net: NetIdx) -> bool {
    let net_info = design.net(net);
    net_info.users.iter().filter(|u| u.is_connected()).count() == 1
}

/// Remove a cell from the design by name (marks it dead).
pub fn remove_cell(design: &mut Design, cell_name: IdString) {
    design.remove_cell(cell_name);
}

//! Packing passes that transform the netlist into architecture-specific cells.

use super::helpers::connect_port;
use super::PackerError;
use crate::context::Context;
use crate::netlist::{CellId, Cluster};
use crate::types::PortType;

/// Ensure GND/VCC constant-driver cells and nets exist.
///
/// Creates `$PACKER_GND` and `$PACKER_VCC` cells with output port "Y", and
/// `$PACKER_GND_NET` and `$PACKER_VCC_NET` nets, connecting the drivers.
/// Idempotent: safe to call multiple times.
pub fn pack_constants(ctx: &mut Context) -> Result<(), PackerError> {
    let gnd_name = ctx.id("$PACKER_GND");
    let vcc_name = ctx.id("$PACKER_VCC");
    let gnd_net_name = ctx.id("$PACKER_GND_NET");
    let vcc_net_name = ctx.id("$PACKER_VCC_NET");
    let y_port = ctx.id("Y");

    // Ensure nets exist.
    let gnd_net_idx = ctx
        .design
        .net_by_name(gnd_net_name)
        .unwrap_or_else(|| ctx.design.add_net(gnd_net_name));
    let vcc_net_idx = ctx
        .design
        .net_by_name(vcc_net_name)
        .unwrap_or_else(|| ctx.design.add_net(vcc_net_name));

    // Ensure GND driver cell exists and is connected.
    if ctx.design.cell_by_name(gnd_name).is_none() {
        let gnd_type = ctx.id("GND");
        let idx = ctx.design.add_cell(gnd_name, gnd_type);
        ctx.design.cell_edit(idx).add_port(y_port, PortType::Out);
        connect_port(ctx, idx, y_port, gnd_net_idx);
    }

    // Ensure VCC driver cell exists and is connected.
    if ctx.design.cell_by_name(vcc_name).is_none() {
        let vcc_type = ctx.id("VCC");
        let idx = ctx.design.add_cell(vcc_name, vcc_type);
        ctx.design.cell_edit(idx).add_port(y_port, PortType::Out);
        connect_port(ctx, idx, y_port, vcc_net_idx);
    }

    Ok(())
}

/// Remap IO pseudo-cells to the architecture-specific IOB type.
///
/// Cells of type `$nextpnr_IBUF`, `$nextpnr_OBUF`, or `$nextpnr_IOBUF` are
/// changed to type `IOB`.
pub fn pack_io(ctx: &mut Context) -> Result<(), PackerError> {
    let ibuf_type = ctx.id("$nextpnr_IBUF");
    let obuf_type = ctx.id("$nextpnr_OBUF");
    let iobuf_type = ctx.id("$nextpnr_IOBUF");
    let iob_type = ctx.id("IOB");

    let cells_to_remap: Vec<CellId> = ctx
        .design
        .iter_cell_indices()
        .filter(|&idx| {
            let cell = ctx.design.cell(idx);
            cell.alive
                && (cell.cell_type == ibuf_type
                    || cell.cell_type == obuf_type
                    || cell.cell_type == iobuf_type)
        })
        .collect();

    for idx in cells_to_remap {
        ctx.design.cell_edit(idx).set_type(iob_type);
    }

    Ok(())
}

/// Merge LUT4 cells with directly-connected DFF cells into clusters.
///
/// A LUT4 whose output port "O" drives exactly one DFF's input port "D"
/// (single-fanout net) will be merged: the LUT becomes the cluster root and
/// the FF becomes a cluster member in `Design::clusters`.
pub fn pack_lut_ff(ctx: &mut Context) -> Result<(), PackerError> {
    let lut4_type = ctx.id("LUT4");
    let dff_type = ctx.id("DFF");
    let o_port = ctx.id("O");
    let d_port = ctx.id("D");
    let q_port = ctx.id("Q");

    // Collect LUT -> FF merge pairs.
    let mut merges: Vec<(CellId, CellId)> = Vec::new();

    for (cell_idx, cell) in ctx.design.iter_cells() {
        if !cell.alive || cell.cell_type != lut4_type {
            continue;
        }

        // Check if "O" port drives exactly one FF "D" port.
        let net_idx = match ctx.cell(cell_idx).port_net(o_port) {
            Some(net_idx) => net_idx,
            None => continue,
        };
        let net = ctx.design.net(net_idx);
        if net.users.iter().filter(|u| u.is_connected()).count() != 1 {
            continue;
        }
        let user = &net.users[0];
        if !user.is_connected() || user.port != d_port {
            continue;
        }
        let ff_cell_idx = match user.cell {
            Some(cell_idx) => cell_idx,
            None => continue,
        };
        let ff_cell = ctx.design.cell(ff_cell_idx);
        if ff_cell.alive && ff_cell.cell_type == dff_type {
            merges.push((cell_idx, ff_cell_idx));
        }
    }

    // Apply merges.
    for (lut_idx, ff_idx) in merges {
        let lut = ctx.design.cell(lut_idx);
        if lut.cluster.is_some() && lut.cluster != Some(lut_idx) {
            continue; // Already part of another cluster.
        }

        // LUT is cluster root.
        ctx.design.cell_edit(lut_idx).set_cluster(Some(lut_idx));

        // FF belongs to LUT's cluster.
        ctx.design.cell_edit(ff_idx).set_cluster(Some(lut_idx));

        // Record explicit cluster membership.
        let cluster = ctx
            .design
            .clusters
            .entry(lut_idx)
            .or_insert_with(|| Cluster::new(lut_idx));
        cluster.add_member(ff_idx);

        // Copy FF Q port output to LUT as QF port, if the FF has a Q port.
        let ff_q_net = ctx.cell(ff_idx).port_net(q_port);
        if let Some(net_idx) = ff_q_net {
            let qf_port = ctx.id("QF");
            ctx.design
                .cell_edit(lut_idx)
                .add_port(qf_port, PortType::Out);
            ctx.design
                .cell_edit(lut_idx)
                .set_port_net(qf_port, Some(net_idx), None);
        }
    }

    Ok(())
}

/// Pack carry chains by linking CARRY4 cells via CO/CI ports.
///
/// Identifies chain heads (CARRY4 cells whose CI is not driven by another
/// CARRY4) and walks the chain forward through CO -> CI connections,
/// linking cells via explicit cluster membership.
pub fn pack_carry(ctx: &mut Context) -> Result<(), PackerError> {
    let carry_type = ctx.id("CARRY4");
    let co_port = ctx.id("CO");
    let ci_port = ctx.id("CI");

    // Find carry chain heads: CARRY4 cells whose CI is not driven by another CARRY4.
    let mut chain_heads: Vec<CellId> = Vec::new();

    for (cell_idx, cell) in ctx.design.iter_cells() {
        if !cell.alive || cell.cell_type != carry_type {
            continue;
        }

        let driven_by_carry = ctx
            .cell(cell_idx)
            .port_net(ci_port)
            .and_then(|net_idx| ctx.net(net_idx).driver_cell_port().map(|pin| pin.cell))
            .is_some_and(|driver| ctx.design.cell(driver).cell_type == carry_type);

        if !driven_by_carry {
            chain_heads.push(cell_idx);
        }
    }

    // Build chains from heads.
    for head in chain_heads {
        let mut current = head;
        ctx.design.cell_edit(current).set_cluster(Some(head));
        let cluster = ctx
            .design
            .clusters
            .entry(head)
            .or_insert_with(|| Cluster::new(head));
        cluster.add_member(head);

        loop {
            let co_net = ctx.cell(current).port_net(co_port);

            let next = co_net.and_then(|net_idx| {
                let net = ctx.design.net(net_idx);
                net.users
                    .iter()
                    .find(|u| {
                        if !u.is_connected() || u.port != ci_port {
                            return false;
                        }
                        let user_cell = match u.cell {
                            Some(cell_idx) => cell_idx,
                            None => return false,
                        };
                        ctx.design.cell(user_cell).alive
                            && ctx.design.cell(user_cell).cell_type == carry_type
                    })
                    .and_then(|u| u.cell)
            });

            match next {
                Some(next_idx) => {
                    ctx.design.cell_edit(next_idx).set_cluster(Some(head));
                    if let Some(cluster) = &mut ctx.design.clusters.get_mut(&head) {
                        cluster.add_member(next_idx);
                    }
                    current = next_idx;
                }
                None => break,
            }
        }
    }

    Ok(())
}

/// Pass-through for remaining cells.
///
/// Currently a no-op since remaining cells are already valid and need no
/// transformation.
pub fn pack_remaining(_ctx: &mut Context) -> Result<(), PackerError> {
    Ok(())
}

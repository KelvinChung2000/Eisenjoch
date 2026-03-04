//! Packing passes that transform the netlist into architecture-specific cells.

use crate::helpers::connect_port;
use crate::PackerError;
use npnr_netlist::{CellIdx, Design, NetIdx};
use npnr_types::{IdStringPool, PortType};

/// Ensure GND/VCC constant-driver cells and nets exist.
///
/// Creates `$PACKER_GND` and `$PACKER_VCC` cells with output port "Y", and
/// `$PACKER_GND_NET` and `$PACKER_VCC_NET` nets, connecting the drivers.
/// Idempotent: safe to call multiple times.
pub fn pack_constants(design: &mut Design, id_pool: &IdStringPool) -> Result<(), PackerError> {
    let gnd_name = id_pool.intern("$PACKER_GND");
    let vcc_name = id_pool.intern("$PACKER_VCC");
    let gnd_net_name = id_pool.intern("$PACKER_GND_NET");
    let vcc_net_name = id_pool.intern("$PACKER_VCC_NET");
    let y_port = id_pool.intern("Y");

    // Ensure nets exist.
    let gnd_net_idx = if let Some(idx) = design.net_by_name(gnd_net_name) {
        idx
    } else {
        design.add_net(gnd_net_name)
    };
    let vcc_net_idx = if let Some(idx) = design.net_by_name(vcc_net_name) {
        idx
    } else {
        design.add_net(vcc_net_name)
    };

    // Ensure GND driver cell exists and is connected.
    if design.cell_by_name(gnd_name).is_none() {
        let gnd_type = id_pool.intern("GND");
        let idx = design.add_cell(gnd_name, gnd_type);
        design.cell_mut(idx).add_port(y_port, PortType::Out);
        connect_port(design, idx, y_port, gnd_net_idx);
    }

    // Ensure VCC driver cell exists and is connected.
    if design.cell_by_name(vcc_name).is_none() {
        let vcc_type = id_pool.intern("VCC");
        let idx = design.add_cell(vcc_name, vcc_type);
        design.cell_mut(idx).add_port(y_port, PortType::Out);
        connect_port(design, idx, y_port, vcc_net_idx);
    }

    Ok(())
}

/// Remap IO pseudo-cells to the architecture-specific IOB type.
///
/// Cells of type `$nextpnr_IBUF`, `$nextpnr_OBUF`, or `$nextpnr_IOBUF` are
/// changed to type `IOB`.
pub fn pack_io(design: &mut Design, id_pool: &IdStringPool) -> Result<(), PackerError> {
    let ibuf_type = id_pool.intern("$nextpnr_IBUF");
    let obuf_type = id_pool.intern("$nextpnr_OBUF");
    let iobuf_type = id_pool.intern("$nextpnr_IOBUF");
    let iob_type = id_pool.intern("IOB");

    let cells_to_remap: Vec<CellIdx> = design
        .cells
        .values()
        .copied()
        .filter(|&idx| {
            let cell = design.cell(idx);
            cell.alive
                && (cell.cell_type == ibuf_type
                    || cell.cell_type == obuf_type
                    || cell.cell_type == iobuf_type)
        })
        .collect();

    for idx in cells_to_remap {
        design.cell_mut(idx).cell_type = iob_type;
    }

    Ok(())
}

/// Merge LUT4 cells with directly-connected DFF cells into clusters.
///
/// A LUT4 whose output port "O" drives exactly one DFF's input port "D"
/// (single-fanout net) will be merged: the LUT becomes the cluster root and
/// the FF becomes a cluster member linked via `cluster_next`.
pub fn pack_lut_ff(design: &mut Design, id_pool: &IdStringPool) -> Result<(), PackerError> {
    let lut4_type = id_pool.intern("LUT4");
    let dff_type = id_pool.intern("DFF");
    let o_port = id_pool.intern("O");
    let d_port = id_pool.intern("D");
    let q_port = id_pool.intern("Q");

    // Collect LUT -> FF merge pairs.
    let mut merges: Vec<(CellIdx, CellIdx)> = Vec::new();

    for (&_name, &cell_idx) in &design.cells {
        let cell = design.cell(cell_idx);
        if !cell.alive || cell.cell_type != lut4_type {
            continue;
        }

        // Check if "O" port drives exactly one FF "D" port.
        let port_info = match cell.port(o_port) {
            Some(p) if p.net.is_some() => p,
            _ => continue,
        };

        let net = design.net(port_info.net);
        if net.users.len() != 1 {
            continue;
        }
        let user = &net.users[0];
        if !user.is_connected() || user.port != d_port {
            continue;
        }
        let ff_cell = design.cell(user.cell);
        if ff_cell.alive && ff_cell.cell_type == dff_type {
            merges.push((cell_idx, user.cell));
        }
    }

    // Apply merges.
    for (lut_idx, ff_idx) in merges {
        let lut = design.cell(lut_idx);
        if lut.cluster.is_some() && lut.cluster != lut_idx {
            continue; // Already part of another cluster.
        }

        // LUT is cluster root.
        design.cell_mut(lut_idx).cluster = lut_idx;

        // FF belongs to LUT's cluster.
        design.cell_mut(ff_idx).cluster = lut_idx;

        // Link FF into the cluster list.
        let old_next = design.cell(lut_idx).cluster_next;
        design.cell_mut(lut_idx).cluster_next = ff_idx;
        design.cell_mut(ff_idx).cluster_next = old_next;

        // Copy FF Q port output to LUT as QF port, if the FF has a Q port.
        let ff_q_net = design.cell(ff_idx).port(q_port).and_then(|p| {
            if p.net.is_some() {
                Some(p.net)
            } else {
                None
            }
        });
        if let Some(net_idx) = ff_q_net {
            let qf_port = id_pool.intern("QF");
            let lut = design.cell_mut(lut_idx);
            lut.add_port(qf_port, PortType::Out);
            if let Some(p) = lut.port_mut(qf_port) {
                p.net = net_idx;
            }
        }
    }

    Ok(())
}

/// Pack carry chains by linking CARRY4 cells via CO/CI ports.
///
/// Identifies chain heads (CARRY4 cells whose CI is not driven by another
/// CARRY4) and walks the chain forward through CO -> CI connections,
/// linking cells via the cluster mechanism.
pub fn pack_carry(design: &mut Design, id_pool: &IdStringPool) -> Result<(), PackerError> {
    let carry_type = id_pool.intern("CARRY4");
    let co_port = id_pool.intern("CO");
    let ci_port = id_pool.intern("CI");

    // Find carry chain heads: CARRY4 cells whose CI is not driven by another CARRY4.
    let mut chain_heads: Vec<CellIdx> = Vec::new();

    for (&_name, &cell_idx) in &design.cells {
        let cell = design.cell(cell_idx);
        if !cell.alive || cell.cell_type != carry_type {
            continue;
        }

        let is_head = match cell.port(ci_port) {
            None => true,
            Some(ci_info) if ci_info.net == NetIdx::NONE => true,
            Some(ci_info) => {
                let net = design.net(ci_info.net);
                if !net.driver.is_connected() {
                    true
                } else {
                    let driver_cell = design.cell(net.driver.cell);
                    driver_cell.cell_type != carry_type
                }
            }
        };

        if is_head {
            chain_heads.push(cell_idx);
        }
    }

    // Build chains from heads.
    for head in chain_heads {
        let mut current = head;
        design.cell_mut(current).cluster = head;

        loop {
            let co_net = design
                .cell(current)
                .port(co_port)
                .and_then(|p| if p.net.is_some() { Some(p.net) } else { None });

            let next = co_net.and_then(|net_idx| {
                let net = design.net(net_idx);
                net.users
                    .iter()
                    .find(|u| {
                        u.is_connected()
                            && u.port == ci_port
                            && design.cell(u.cell).alive
                            && design.cell(u.cell).cell_type == carry_type
                    })
                    .map(|u| u.cell)
            });

            match next {
                Some(next_idx) => {
                    design.cell_mut(current).cluster_next = next_idx;
                    design.cell_mut(next_idx).cluster = head;
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
pub fn pack_remaining(_design: &mut Design, _id_pool: &IdStringPool) -> Result<(), PackerError> {
    Ok(())
}

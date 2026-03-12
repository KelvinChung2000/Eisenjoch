//! Architecture-generic packing passes.
//!
//! These passes handle constant drivers and IO buffer remapping, which are
//! universal across all architectures. Architecture-specific packing (clustering
//! cells based on shared wires, carry chains, etc.) is handled by the
//! database-driven rule engine in the parent module.

use super::helpers::connect_port;
use super::PackerError;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellId, PortType};

/// Ensure GND/VCC constant-driver cells and nets exist.
///
/// Creates `$PACKER_GND` and `$PACKER_VCC` cells with output port "Y", and
/// `$PACKER_GND_NET` and `$PACKER_VCC_NET` nets, connecting the drivers.
/// Idempotent: safe to call multiple times.
pub fn pack_constants(ctx: &mut Context) -> Result<(), PackerError> {
    let y_port = ctx.id("Y");

    // Detect architecture-specific constant driver types from the chipdb.
    // Himbaechel architectures typically use GND_DRV/VCC_DRV; fall back to GND/VCC.
    let gnd_drv = ctx.id("GND_DRV");
    let vcc_drv = ctx.id("VCC_DRV");
    let gnd_type = if ctx.has_bel_type(gnd_drv) { gnd_drv } else { ctx.id("GND") };
    let vcc_type = if ctx.has_bel_type(vcc_drv) { vcc_drv } else { ctx.id("VCC") };

    // Output pin name: GND_DRV uses "GND", VCC_DRV uses "VCC", generic uses "Y".
    let gnd_port = if gnd_type == gnd_drv { ctx.id("GND") } else { y_port };
    let vcc_port = if vcc_type == vcc_drv { ctx.id("VCC") } else { y_port };

    let gnd_idx = ensure_const_driver(
        ctx, "$PACKER_GND", "$PACKER_GND_NET", gnd_type, gnd_port, y_port,
    );
    let vcc_idx = ensure_const_driver(
        ctx, "$PACKER_VCC", "$PACKER_VCC_NET", vcc_type, vcc_port, y_port,
    );

    // Bind constant driver cells to BELs so the router can resolve their output wires.
    bind_to_first_available_bel(ctx, gnd_idx, gnd_type);
    bind_to_first_available_bel(ctx, vcc_idx, vcc_type);

    Ok(())
}

/// Create or update a constant driver cell and its net.
///
/// If the cell already exists, updates its type and renames its output port if
/// needed. Otherwise creates the cell, adds the output port, and connects it to
/// the net.
fn ensure_const_driver(
    ctx: &mut Context,
    cell_name: &str,
    net_name: &str,
    cell_type: IdString,
    out_port: IdString,
    y_port: IdString,
) -> CellId {
    let cell_name_id = ctx.id(cell_name);
    let net_name_id = ctx.id(net_name);

    let net_idx = ctx
        .design
        .net_by_name(net_name_id)
        .unwrap_or_else(|| ctx.design.add_net(net_name_id));

    if let Some(idx) = ctx.design.cell_by_name(cell_name_id) {
        ctx.design.cell_edit(idx).set_type(cell_type);
        if out_port != y_port {
            ctx.design.cell_edit(idx).rename_port(y_port, out_port);
            ctx.design.net_edit(net_idx).set_driver(idx, out_port);
        }
        idx
    } else {
        let idx = ctx.design.add_cell(cell_name_id, cell_type);
        ctx.design.cell_edit(idx).add_port(out_port, PortType::Out);
        connect_port(ctx, idx, out_port, net_idx);
        idx
    }
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

    let cells_to_remap: Vec<_> = ctx
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

/// Pass-through for remaining cells.
///
/// Currently a no-op since remaining cells are already valid and need no
/// transformation.
pub fn pack_remaining(_ctx: &mut Context) -> Result<(), PackerError> {
    Ok(())
}

/// Bind a cell to the first available BEL of the given type.
/// If no BEL is available (e.g. minimal/synthetic chipdb), silently skips.
fn bind_to_first_available_bel(
    ctx: &mut Context,
    cell_idx: CellId,
    bel_type: IdString,
) {
    if ctx.design.cell(cell_idx).bel.is_some() {
        return;
    }
    let bel = ctx
        .bels_for_bucket(bel_type)
        .find(|b| b.is_available())
        .map(|b| b.id());
    if let Some(bel) = bel {
        ctx.bind_bel(bel, cell_idx, PlaceStrength::Locked);
    }
}

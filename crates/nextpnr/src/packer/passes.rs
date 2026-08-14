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
use crate::netlist::{Cluster, CellId, PortType};

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
    let gnd_type = if ctx.has_bel_type(gnd_drv) {
        gnd_drv
    } else {
        ctx.id("GND")
    };
    let vcc_type = if ctx.has_bel_type(vcc_drv) {
        vcc_drv
    } else {
        ctx.id("VCC")
    };

    // Output pin name: GND_DRV uses "GND", VCC_DRV uses "VCC", generic uses "Y".
    let gnd_port = if gnd_type == gnd_drv {
        ctx.id("GND")
    } else {
        y_port
    };
    let vcc_port = if vcc_type == vcc_drv {
        ctx.id("VCC")
    } else {
        y_port
    };

    let gnd_idx = ensure_const_driver(
        ctx,
        "$PACKER_GND",
        "$PACKER_GND_NET",
        gnd_type,
        gnd_port,
        y_port,
    );
    let vcc_idx = ensure_const_driver(
        ctx,
        "$PACKER_VCC",
        "$PACKER_VCC_NET",
        vcc_type,
        vcc_port,
        y_port,
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

/// Delete nextpnr's IO pseudo-cells, for flows where synthesis already inserted
/// real IO buffers.
///
/// Port of `HimbaechelHelpers::remove_nextpnr_iobs` (upstream `4d235150`).
///
/// [`pack_io`] *remaps* `$nextpnr_IBUF` onto an `IOB` bel, which is correct when
/// the pseudo-cell is itself the IO cell. When synthesis ran `iopadmap` the pad
/// net already reaches a real buffer, so the pseudo-cell has to be removed
/// instead: remapping it would double-book the IO site and, on a fabric with as
/// many IO bels as pads, fail placement outright.
///
/// Upstream also errors when a pseudo-cell connects to anything other than the
/// architecture's declared top-level port types. That check needs an arch
/// cell-type list we do not have here, so it is deliberately omitted; the caller
/// is expected to have synthesised with matching IO buffer insertion.
///
/// Note that `disconnect_user` blanks the slot to `CellPin::INVALID` rather
/// than removing it, so the net keeps its user-slot indices (other cells cache
/// theirs). Anything walking `net.users()` afterwards must skip invalid pins.
///
/// Returns the number of pseudo-cells removed.
pub fn remove_nextpnr_iobs(ctx: &mut Context) -> Result<usize, PackerError> {
    let pseudo_types = [
        ctx.id("$nextpnr_IBUF"),
        ctx.id("$nextpnr_OBUF"),
        ctx.id("$nextpnr_IOBUF"),
    ];
    let pseudo_ports = [ctx.id("I"), ctx.id("O"), ctx.id("IO")];

    let victims: Vec<CellId> = ctx
        .design
        .iter_alive_cells()
        .filter(|(_, cell)| pseudo_types.contains(&cell.cell_type))
        .map(|(idx, _)| idx)
        .collect();

    for idx in &victims {
        for &port in &pseudo_ports {
            let cell = ctx.design.cell(*idx);
            let Some(net) = cell.port_net(port) else {
                continue;
            };
            let user_idx = cell.port_user_idx(port);
            let drives = ctx
                .design
                .net(net)
                .driver()
                .is_some_and(|d| d.cell == *idx && d.port == port);

            if drives {
                ctx.design.net_edit(net).clear_driver();
            } else if let Some(u) = user_idx {
                ctx.design.net_edit(net).disconnect_user(u as usize);
            }
            ctx.design.cell_edit(*idx).set_port_net(port, None, None);
        }
        ctx.design.cell_edit(*idx).mark_dead();
    }

    Ok(victims.len())
}

/// A (cell type, port) pair. Upstream's `CellTypePort`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellTypePort {
    pub cell_type: IdString,
    pub port: IdString,
}

impl CellTypePort {
    pub fn new(cell_type: IdString, port: IdString) -> Self {
        Self { cell_type, port }
    }
}

/// Constrain driver-sink pairs into one cluster, so they place as a unit.
///
/// Port of `HimbaechelHelpers::constrain_cell_pairs` (upstream `4d235150`).
///
/// This is how nextpnr makes site legality *structural* rather than a filter.
/// On the example uarch a slice's LUT and FF may only share a site if the FF is
/// driven by that LUT; constraining each such pair to `delta_z = 1` means every
/// shared slice is legal by construction, and the placer never has to discover
/// the rule by trial and error.
///
/// Each driver takes at most one sink, and cells already in a cluster are left
/// alone -- both matching upstream, which breaks out of the port loop after the
/// first match. With `allow_fanout = false` a driver whose net has more than one
/// live user is skipped entirely, since the pair could not be honoured for all
/// of them.
///
/// Returns the number of pairs constrained.
pub fn constrain_cell_pairs(
    ctx: &mut Context,
    src_ports: &[CellTypePort],
    sink_ports: &[CellTypePort],
    delta_z: i32,
    allow_fanout: bool,
) -> usize {
    let candidates: Vec<CellId> = ctx
        .design
        .iter_alive_cells()
        .filter(|(_, cell)| cell.cluster.is_none())
        .map(|(idx, _)| idx)
        .collect();

    let mut constrained = 0usize;
    for root in candidates {
        // A cell constrained as some earlier root's child is no longer free.
        if ctx.design.cell(root).cluster.is_some() {
            continue;
        }
        let root_type = ctx.design.cell(root).cell_type;

        let Some((net, _)) = src_ports
            .iter()
            .filter(|s| s.cell_type == root_type)
            .find_map(|s| {
                let cell = ctx.design.cell(root);
                if cell.port_type(s.port) != Some(PortType::Out) {
                    return None;
                }
                cell.port_net(s.port).map(|net| (net, s.port))
            })
        else {
            continue;
        };

        // `users` keeps blanked slots after disconnect_user, so count live ones.
        let live: Vec<_> = ctx
            .design
            .net(net)
            .users()
            .iter()
            .copied()
            .filter(|u| u.cell.is_some())
            .collect();
        if !allow_fanout && live.len() > 1 {
            continue;
        }

        let Some(sink) = live.into_iter().find(|u| {
            let sink_cell = ctx.design.cell(u.cell);
            sink_cell.cluster.is_none()
                && sink_ports
                    .iter()
                    .any(|p| p.cell_type == sink_cell.cell_type && p.port == u.port)
        }) else {
            continue;
        };

        ctx.design.cell_edit(root).set_cluster(Some(root));
        ctx.design.cell_edit(root).set_constraints(0, 0, 0, false);
        ctx.design.cell_edit(sink.cell).set_cluster(Some(root));
        ctx.design
            .cell_edit(sink.cell)
            .set_constraints(0, 0, delta_z, false);

        let cluster = ctx
            .design
            .clusters
            .entry(root)
            .or_insert_with(|| Cluster::new(root));
        cluster.members.push(sink.cell);
        cluster.constr_children.push(sink.cell);
        constrained += 1;
    }

    constrained
}

/// Bind explicit BUFG cells to BUFG BELs.
///
/// Yosys keeps BUFG as a real cell when clkbufmap is enabled, so there is no
/// net rewriting here; the packer only reserves physical BUFG sites.
pub fn pack_bufg(ctx: &mut Context) -> Result<(), PackerError> {
    let bufg_type = ctx.id("BUFG");
    if !ctx.has_bel_type(bufg_type) {
        return Ok(());
    }

    let cells_to_bind: Vec<_> = ctx
        .design
        .iter_cell_indices()
        .filter(|&idx| {
            let cell = ctx.design.cell(idx);
            cell.alive && cell.cell_type == bufg_type
        })
        .collect();

    for idx in cells_to_bind {
        bind_to_first_available_bel(ctx, idx, bufg_type);
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
fn bind_to_first_available_bel(ctx: &mut Context, cell_idx: CellId, bel_type: IdString) {
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

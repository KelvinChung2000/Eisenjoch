//! Checkpoint restore logic and design fingerprinting.

use log::info;
use rustc_hash::FxHashSet;

use super::{CellSig, Checkpoint, CheckpointError, DesignDiff, DesignFingerprint, NetSig};
use crate::chipdb::{BelId, PipId, WireId};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellId, NetId};
use crate::router::common::bind_route;

/// Report of what was restored from a checkpoint.
pub struct RestoreReport {
    pub cells_restored: usize,
    pub cells_skipped: usize,
    pub nets_restored: usize,
    pub nets_skipped: usize,
    pub cells_to_place: Vec<CellId>,
    pub nets_to_route: Vec<NetId>,
    pub diff: DesignDiff,
}

/// Restore placements and routes from a checkpoint into the context.
///
/// Matched cells are placed as `Fixed` so the placer treats them as immovable.
/// After calling this, run `placer.place()` and `router.route()` normally -
/// they will skip the Fixed cells and already-routed nets.
pub fn restore(ctx: &mut Context, checkpoint: &Checkpoint) -> Result<RestoreReport, CheckpointError> {
    // Compute current design fingerprint.
    let current_fp = compute_fingerprint(ctx);
    let diff = DesignDiff::compute(&checkpoint.fingerprint, &current_fp);

    // Build skip sets: cells/nets that were removed or changed.
    let skip_cells: FxHashSet<&str> = diff
        .removed_cells
        .iter()
        .chain(diff.changed_cells.iter())
        .map(String::as_str)
        .collect();
    let skip_nets: FxHashSet<&str> = diff
        .removed_nets
        .iter()
        .chain(diff.changed_nets.iter())
        .map(String::as_str)
        .collect();

    let mut cells_restored = 0usize;
    let mut cells_skipped = 0usize;
    let mut restored_cell_ids: FxHashSet<IdString> = FxHashSet::default();

    // Restore placements.
    for cp in &checkpoint.placements {
        if skip_cells.contains(cp.cell_name.as_str()) {
            cells_skipped += 1;
            continue;
        }

        let cell_name_id = ctx.id_pool.intern(&cp.cell_name);
        let cell_idx = match ctx.design.cell_by_name(cell_name_id) {
            Some(idx) => idx,
            None => {
                cells_skipped += 1;
                continue;
            }
        };

        let bel = BelId::new(cp.bel_tile, cp.bel_index);
        if !bel.is_valid() || !ctx.bel(bel).is_available() {
            cells_skipped += 1;
            continue;
        }

        if ctx.bind_bel(bel, cell_idx, PlaceStrength::Fixed) {
            cells_restored += 1;
            restored_cell_ids.insert(cell_name_id);
        } else {
            cells_skipped += 1;
        }
    }

    let mut nets_restored = 0usize;
    let mut nets_skipped = 0usize;
    let mut nets_to_route_set: FxHashSet<NetId> = FxHashSet::default();

    // Restore routes.
    for nr in &checkpoint.routes {
        if skip_nets.contains(nr.net_name.as_str()) {
            nets_skipped += 1;
            continue;
        }

        let net_name_id = ctx.id_pool.intern(&nr.net_name);
        let net_idx = match ctx.design.net_by_name(net_name_id) {
            Some(idx) => idx,
            None => {
                nets_skipped += 1;
                continue;
            }
        };

        // Check all cells on this net are restored.
        let net = ctx.net(net_idx);
        let cell_is_restored =
            |cell_idx: CellId| restored_cell_ids.contains(&ctx.cell(cell_idx).name_id());

        let driver_restored = net
            .driver()
            .map_or(true, |pin| cell_is_restored(pin.cell));
        let users_restored = net
            .users()
            .iter()
            .filter(|u| u.is_valid())
            .all(|u| cell_is_restored(u.cell));

        if !driver_restored || !users_restored {
            nets_skipped += 1;
            nets_to_route_set.insert(net_idx);
            continue;
        }

        // Bind the source wire.
        let src_wire = WireId::new(nr.source_wire_tile, nr.source_wire_index);
        if src_wire.is_valid() && ctx.wire(src_wire).is_available() {
            ctx.bind_wire(src_wire, net_idx, PlaceStrength::Strong);
            ctx.design
                .net_edit(net_idx)
                .add_wire(src_wire, None, PlaceStrength::Strong);
        }

        // Bind the PIP path.
        let pips: Vec<PipId> = nr.pips.iter().map(|&(t, i)| PipId::new(t, i)).collect();
        bind_route(ctx, net_idx, &pips);
        nets_restored += 1;
    }

    // Collect cells that still need placement (alive, no BEL).
    let cells_to_place: Vec<CellId> = ctx
        .design
        .iter_alive_cells()
        .filter(|(_, cell)| cell.bel.is_none())
        .map(|(ci, _)| ci)
        .collect();

    // Collect nets that still need routing.
    let nets_to_route: Vec<NetId> = ctx
        .design
        .iter_alive_nets()
        .filter_map(|(ni, net)| {
            if !net.has_driver() || net.num_users() == 0 {
                return None;
            }
            if net.wires.is_empty() || nets_to_route_set.contains(&ni) {
                Some(ni)
            } else {
                None
            }
        })
        .collect();

    info!(
        "Checkpoint restore: {} cells restored, {} skipped, {} nets restored, {} skipped",
        cells_restored, cells_skipped, nets_restored, nets_skipped
    );
    info!(
        "Checkpoint restore: {} cells to place, {} nets to route",
        cells_to_place.len(),
        nets_to_route.len()
    );

    Ok(RestoreReport {
        cells_restored,
        cells_skipped,
        nets_restored,
        nets_skipped,
        cells_to_place,
        nets_to_route,
        diff,
    })
}

/// Compute a design fingerprint from the current context.
pub fn compute_fingerprint(ctx: &Context) -> DesignFingerprint {
    let mut cell_signatures: Vec<CellSig> = ctx
        .design
        .iter_alive_cells()
        .map(|(_, cell)| CellSig {
            name: ctx.name_of(cell.name).to_owned(),
            cell_type: ctx.name_of(cell.cell_type).to_owned(),
            port_count: cell.num_ports(),
        })
        .collect();
    cell_signatures.sort_by(|a, b| a.name.cmp(&b.name));

    let mut net_signatures: Vec<NetSig> = ctx
        .design
        .iter_alive_nets()
        .map(|(_, net)| {
            let (driver_cell, driver_port) = match net.driver() {
                Some(pin) if pin.is_valid() => {
                    let cell = ctx.design.cell(pin.cell);
                    (
                        ctx.name_of(cell.name).to_owned(),
                        ctx.name_of(pin.port).to_owned(),
                    )
                }
                _ => (String::new(), String::new()),
            };
            NetSig {
                name: ctx.name_of(net.name).to_owned(),
                driver_cell,
                driver_port,
                user_count: net.num_users(),
            }
        })
        .collect();
    net_signatures.sort_by(|a, b| a.name.cmp(&b.name));

    DesignFingerprint {
        cell_signatures,
        net_signatures,
    }
}

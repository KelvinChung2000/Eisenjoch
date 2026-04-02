//! Swap move logic for SA placer.

use crate::chipdb::BelId;
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::metrics::net_hpwl;
use crate::netlist::{CellId, NetId};

use super::congestion::CongestionCache;

/// Collect all nets that are touched by a given cell (both as driver and as user).
fn nets_for_cell(ctx: &Context, cell_idx: CellId) -> Vec<NetId> {
    let cell = ctx.cell(cell_idx);
    let mut nets = Vec::new();
    for pin in cell.ports() {
        if let Some(net_idx) = pin.view(ctx).net_id() {
            nets.push(net_idx);
        }
    }
    nets.sort_unstable();
    nets.dedup();
    nets
}

/// Compute HPWL for a set of nets (used for incremental delta computation).
///
/// Uses parallel iteration for large net lists (> 16 nets) to amortize rayon overhead.
pub(super) fn hpwl_for_nets(ctx: &Context, net_indices: &[NetId]) -> f64 {
    if net_indices.len() > 16 {
        use rayon::prelude::*;
        net_indices.par_iter().map(|&idx| net_hpwl(ctx, idx)).sum()
    } else {
        net_indices.iter().map(|&idx| net_hpwl(ctx, idx)).sum()
    }
}

/// Result of a proposed swap move.
pub struct SwapResult {
    /// The delta in HPWL cost (negative = improvement).
    pub delta_cost: f64,
    /// The delta in congestion cost (negative = improvement).
    pub delta_congestion: f64,
    /// Whether the move was actually performed on the context.
    pub performed: bool,
    /// The nets affected by this swap (needed for congestion revert).
    pub affected_nets: Vec<NetId>,
}

impl SwapResult {
    /// A no-op result: no move was performed, no cost change.
    pub(super) fn noop() -> Self {
        Self {
            delta_cost: 0.0,
            delta_congestion: 0.0,
            performed: false,
            affected_nets: Vec::new(),
        }
    }
}

/// Attempt a swap move: move `cell` to `target_bel`.
///
/// If `target_bel` is occupied by another cell, the two cells are swapped.
/// The function computes the delta cost by measuring HPWL of affected nets
/// before and after the move. If a congestion cache is provided, it also
/// computes the delta congestion by removing and re-adding demand for
/// affected nets.
///
/// The move is always performed (bind/unbind) so the caller can decide whether
/// to accept or revert. If the caller rejects, it must call `revert_swap`.
pub fn try_swap(
    ctx: &mut Context,
    cell_idx: CellId,
    target_bel: BelId,
    mut congestion: Option<&mut CongestionCache>,
) -> SwapResult {
    let cell = ctx.cell(cell_idx);
    let old_bel = cell.bel().map(|b| b.id());

    // If we are already at the target, no-op.
    if old_bel == Some(target_bel) {
        return SwapResult::noop();
    }

    let old_bel = match old_bel {
        Some(bel) => bel,
        None => return SwapResult::noop(),
    };

    // Determine if there is a cell at the target bel.
    let other_cell_idx = ctx.bel(target_bel).bound_cell().map(|c| c.id());

    // Check that the other cell (if any) is moveable.
    if let Some(oci) = other_cell_idx {
        let other_cell = ctx.cell(oci);
        if other_cell.bel_strength().is_locked() {
            return SwapResult::noop();
        }
    }

    // Collect affected nets before the move.
    let mut affected_nets = nets_for_cell(ctx, cell_idx);
    if let Some(oci) = other_cell_idx {
        let mut other_nets = nets_for_cell(ctx, oci);
        affected_nets.append(&mut other_nets);
        affected_nets.sort_unstable();
        affected_nets.dedup();
    }

    // Compute HPWL cost before.
    let cost_before = hpwl_for_nets(ctx, &affected_nets);

    // Remove congestion demand for affected nets (at old positions).
    let congestion_before = if let Some(ref mut cache) = congestion {
        for &net in &affected_nets {
            cache.add_net_demand(ctx, net, -1.0);
        }
        cache.total_congestion_cost()
    } else {
        0.0
    };

    // Unbind both cells.
    ctx.unbind_bel(old_bel);
    if other_cell_idx.is_some() {
        ctx.unbind_bel(target_bel);
    }

    // Bind cell to target_bel.
    ctx.bind_bel(target_bel, cell_idx, PlaceStrength::Placer);

    // Bind other cell to old_bel (if swap).
    if let Some(oci) = other_cell_idx {
        ctx.bind_bel(old_bel, oci, PlaceStrength::Placer);
    }

    // Compute HPWL cost after.
    let cost_after = hpwl_for_nets(ctx, &affected_nets);

    // Add congestion demand for affected nets (at new positions).
    let delta_congestion = if let Some(cache) = congestion {
        for &net in &affected_nets {
            cache.add_net_demand(ctx, net, 1.0);
        }
        let congestion_after = cache.total_congestion_cost();
        congestion_after - congestion_before
    } else {
        0.0
    };

    SwapResult {
        delta_cost: cost_after - cost_before,
        delta_congestion,
        performed: true,
        affected_nets,
    }
}

/// Revert a swap move (undo the last try_swap that was performed).
pub fn revert_swap(
    ctx: &mut Context,
    cell_idx: CellId,
    old_bel: BelId,
    other_cell_idx: Option<CellId>,
    current_bel: BelId,
) {
    // Unbind current positions.
    ctx.unbind_bel(current_bel);
    if let Some(oci) = other_cell_idx {
        ctx.unbind_bel(old_bel);
        // Restore other cell to its original position (current_bel was its old bel).
        ctx.bind_bel(current_bel, oci, PlaceStrength::Placer);
    }
    // Restore cell to old_bel.
    ctx.bind_bel(old_bel, cell_idx, PlaceStrength::Placer);
}

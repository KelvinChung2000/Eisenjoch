//! Boundary cell locking for optimal transport placer.

use crate::common::PlaceStrength;
use crate::context::Context;
use rustc_hash::FxHashSet;

/// Lock cells whose BELs only exist on boundary/IO tiles.
///
/// These cells (IOB, clock buffers, etc.) cannot benefit from the continuous
/// solve because their valid positions are sparse and fixed at the chip edge.
/// Locking them makes them fixed anchors in the Kirchhoff system, naturally
/// pulling connected logic toward the boundary via pressure gradients.
pub(super) fn lock_boundary_cells(ctx: &mut Context) {
    // Collect all cell types present in the design.
    let mut cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for (_id, cell) in ctx.design.iter_alive_cells() {
        cell_types.insert(cell.cell_type);
    }

    // For each cell type, check if its BELs are on CLB tiles (>= 4 BELs per tile).
    // Cell types with NO BELs on CLB tiles are boundary/IO types.
    let mut clb_cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for &ct in &cell_types {
        let on_clb = ctx.bels_for_bucket(ct).any(|b| {
            let loc = b.loc();
            let tile = ctx.chipdb().tile_by_xy(loc.x, loc.y);
            ctx.chipdb().tile_type(tile).bels.len() >= 4
        });
        if on_clb {
            clb_cell_types.insert(ct);
        }
    }

    let mut locked_count = 0usize;
    let cell_ids: Vec<_> = ctx
        .design
        .iter_alive_cells()
        .map(|(id, _)| id)
        .collect();

    for cell_id in cell_ids {
        let cell = ctx.design.cell(cell_id);
        if cell.bel_strength.is_locked() {
            continue;
        }
        if clb_cell_types.contains(&cell.cell_type) {
            continue;
        }
        if cell.bel.is_some() {
            ctx.design.cell_mut(cell_id).bel_strength = PlaceStrength::Locked;
            locked_count += 1;
        }
    }

    if locked_count > 0 {
        eprintln!("Locked {} boundary/IO cells as fixed anchors", locked_count);
    }
}

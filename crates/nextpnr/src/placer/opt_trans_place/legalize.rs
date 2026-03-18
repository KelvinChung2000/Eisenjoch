//! BEL legalization for the optimal transport placer.
//!
//! Uses minimum-displacement LAPJV bipartite matching for final legalization.
//! Pure distance² cost — the continuous solver already encodes pressure and
//! congestion in cell positions, so no additional penalties are needed.

use crate::context::Context;
use crate::placer::legalize::{legalize_bipartite, DistanceCost};
use crate::placer::PlacerError;

use super::state::OptTransState;

/// Bipartite (LAPJV) legalization: minimum-displacement cell-to-BEL assignment.
///
/// Uses pure distance² cost to preserve the continuous solver's positions as
/// closely as possible. The solver already accounts for wire distance, density,
/// and congestion — legalization just snaps to the nearest valid BELs.
pub fn legalize_opt_trans(
    ctx: &mut Context,
    state: &OptTransState,
    lap_max_cells: usize,
) -> Result<f64, PlacerError> {
    // Convert virtual grid positions to physical for BEL matching.
    let x0 = state.network.x0 as f64;
    let y0 = state.network.y0 as f64;
    let phys_x: Vec<f64> = state.cell_x.iter().map(|&x| x + x0).collect();
    let phys_y: Vec<f64> = state.cell_y.iter().map(|&y| y + y0).collect();
    legalize_bipartite(
        ctx,
        &state.idx_to_cell,
        &phys_x,
        &phys_y,
        &DistanceCost,
        lap_max_cells,
    )
}

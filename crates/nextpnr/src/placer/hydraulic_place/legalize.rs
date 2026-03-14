//! BEL legalization for the Hydraulic placer.
//!
//! Uses minimum-displacement LAPJV bipartite matching for final legalization.
//! Pure distance² cost — the continuous solver already encodes pressure and
//! congestion in cell positions, so no additional penalties are needed.

use crate::context::Context;
use crate::placer::legalize::{legalize_bipartite, DistanceCost};
use crate::placer::PlacerError;

use super::state::HydraulicState;

/// Bipartite (LAPJV) legalization: minimum-displacement cell-to-BEL assignment.
///
/// Uses pure distance² cost to preserve the continuous solver's positions as
/// closely as possible. The solver already accounts for wire distance, density,
/// and congestion — legalization just snaps to the nearest valid BELs.
pub fn legalize_hydraulic(
    ctx: &mut Context,
    state: &HydraulicState,
    lap_max_cells: usize,
) -> Result<f64, PlacerError> {
    legalize_bipartite(
        ctx,
        &state.idx_to_cell,
        &state.cell_x,
        &state.cell_y,
        &DistanceCost,
        lap_max_cells,
    )
}

//! Pressure-aware BEL legalization for the Hydraulic placer.
//!
//! Assigns cells to discrete BELs using a cost function that combines
//! distance from the continuous position with junction pressure (congestion).
//! The pressure has physical units -- no artificial weight multiplier.

use crate::common::PlaceStrength;
use crate::context::Context;
use crate::placer::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use super::state::HydraulicState;

/// Legalize all movable cells to discrete BEL positions.
///
/// Cost: cost(cell, bel) = distance^2 + |local_pressure(tile)|
///
/// Returns total squared displacement (quality metric).
pub fn legalize_hydraulic(
    ctx: &mut Context,
    state: &HydraulicState,
) -> Result<f64, PlacerError> {
    unbind_movable_cells(ctx, &state.idx_to_cell);

    // Sort cells: largest displacement from center first.
    let mut cell_order: Vec<usize> = (0..state.num_cells()).collect();
    let cx = state.network.width as f64 / 2.0;
    let cy = state.network.height as f64 / 2.0;
    cell_order.sort_by(|&a, &b| {
        let da = (state.cell_x[a] - cx).powi(2) + (state.cell_y[a] - cy).powi(2);
        let db = (state.cell_x[b] - cx).powi(2) + (state.cell_y[b] - cy).powi(2);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut total_displacement = 0.0;

    for &solver_idx in &cell_order {
        let cell_id = state.idx_to_cell[solver_idx];
        let cell_type = ctx.design.cell(cell_id).cell_type;
        let target_x = state.cell_x[solver_idx];
        let target_y = state.cell_y[solver_idx];

        let mut best_bel = None;
        let mut best_cost = f64::INFINITY;

        for bel_view in ctx.bels_for_bucket(cell_type) {
            if !bel_view.is_available() {
                continue;
            }

            let loc = bel_view.loc();
            let dx = loc.x as f64 - target_x;
            let dy = loc.y as f64 - target_y;
            let dist_sq = dx * dx + dy * dy;

            let jidx = state.network.junction_index(loc.x, loc.y);
            let pressure = state.network.junctions[jidx].pressure.abs();
            // Pure physical cost: distance + pressure (no artificial weight).
            let cost = dist_sq + pressure;

            if cost < best_cost {
                best_cost = cost;
                best_bel = Some(bel_view.id());
            }
        }

        let bel = best_bel.ok_or_else(|| {
            PlacerError::NoBelsAvailable(ctx.name_of(cell_type).to_owned())
        })?;

        if !ctx.bind_bel(bel, cell_id, PlaceStrength::Placer) {
            let cell_name = ctx.design.cell(cell_id).name;
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to bind cell {} during hydraulic legalization",
                ctx.name_of(cell_name)
            )));
        }

        let loc = ctx.bel(bel).loc();
        let dx = loc.x as f64 - target_x;
        let dy = loc.y as f64 - target_y;
        total_displacement += dx * dx + dy * dy;

        place_cluster_children(ctx, cell_id, bel)?;
    }

    Ok(total_displacement)
}

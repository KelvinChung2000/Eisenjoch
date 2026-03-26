//! Simple greedy nearest-BEL legalization for ElectroPlace.

use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use crate::legalize::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use super::Legalizer;

/// Simple greedy nearest-BEL legalization.
pub struct GreedyLegalizer;

impl Legalizer for GreedyLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
    ) -> Result<f64, PlacerError> {
        legalize_electro(ctx, idx_to_cell, cell_x, cell_y)
    }
}

pub fn legalize_electro(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> Result<f64, PlacerError> {
    unbind_movable_cells(ctx, idx_to_cell);

    let mut total_displacement = 0.0;

    for (i, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        let target_x = cell_x[i];
        let target_y = cell_y[i];

        let mut best_bel = None;
        let mut best_cost = f64::INFINITY;

        for bel_view in ctx.bels_for_bucket(cell_type) {
            if !bel_view.is_available() {
                continue;
            }
            let loc = bel_view.loc();
            let dx = loc.x as f64 - target_x;
            let dy = loc.y as f64 - target_y;
            let cost = dx * dx + dy * dy;

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
                "Failed to bind cell {} during ElectroPlace legalization",
                ctx.name_of(cell_name)
            )));
        }

        total_displacement += best_cost;
        place_cluster_children(ctx, cell_id, bel)?;
    }

    Ok(total_displacement)
}

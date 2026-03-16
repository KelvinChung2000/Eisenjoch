//! General bipartite legalization module using Jonker-Volgenant algorithm.
//!
//! Assigns cells to discrete BELs via minimum-cost bipartite matching.
//! Cost functions are pluggable via the LegalizeCost trait.

use crate::chipdb::BelId;
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use ndarray::Array2;
use rustc_hash::FxHashMap;

/// Cost function trait for legalization.
///
/// `cell_x` and `cell_y` are the continuous solver positions, indexed by solver index.
pub trait LegalizeCost {
    fn cost(&self, cell_x: f64, cell_y: f64, bel_x: i32, bel_y: i32) -> f64;
}

/// Distance-squared cost: snaps cells to the nearest valid BELs.
pub struct DistanceCost;

impl LegalizeCost for DistanceCost {
    fn cost(&self, cell_x: f64, cell_y: f64, bel_x: i32, bel_y: i32) -> f64 {
        let dx = bel_x as f64 - cell_x;
        let dy = bel_y as f64 - cell_y;
        dx * dx + dy * dy
    }
}

/// Legalize cells to discrete BELs using bipartite matching (Jonker-Volgenant).
///
/// Groups cells by type, builds cost matrices, solves assignment.
/// Returns total squared displacement.
pub fn legalize_bipartite(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    cost: &dyn LegalizeCost,
    lap_max_cells: usize,
) -> Result<f64, PlacerError> {
    unbind_movable_cells(ctx, idx_to_cell);

    // Group cells by type
    let mut groups: FxHashMap<crate::common::IdString, Vec<usize>> = FxHashMap::default();
    for (solver_idx, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        groups.entry(cell_type).or_default().push(solver_idx);
    }

    let mut total_displacement = 0.0;

    for (&cell_type, cell_indices) in &groups {
        let n_cells = cell_indices.len();

        if n_cells > lap_max_cells {
            return Err(PlacerError::PlacementFailed(format!(
                "Cell type {} has {} cells, exceeding lap_max_cells limit of {}",
                ctx.name_of(cell_type),
                n_cells,
                lap_max_cells,
            )));
        }

        // Collect available BELs
        let bels: Vec<(BelId, i32, i32)> = ctx
            .bels_for_bucket(cell_type)
            .filter(|b| b.is_available())
            .map(|b| {
                let loc = b.loc();
                (b.id(), loc.x, loc.y)
            })
            .collect();

        let n_bels = bels.len();

        if n_bels < n_cells {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (need {} BELs but only {} available)",
                ctx.name_of(cell_type),
                n_cells,
                n_bels,
            )));
        }

        // Build cost matrix [dim x dim] - lapjv requires square matrix
        let dim = n_cells.max(n_bels);
        let mut cost_matrix = Array2::<f64>::zeros((dim, dim));

        for (row, &solver_idx) in cell_indices.iter().enumerate() {
            let cx = cell_x[solver_idx];
            let cy = cell_y[solver_idx];
            for (col, &(_, bx, by)) in bels.iter().enumerate() {
                cost_matrix[[row, col]] = cost.cost(cx, cy, bx, by);
            }
        }

        // Solve using lapjv
        let result = lapjv::lapjv(&cost_matrix)
            .map_err(|e| PlacerError::PlacementFailed(format!("LAPJV solver failed: {:?}", e)))?;

        // Bind cells to assigned BELs
        for (row, &solver_idx) in cell_indices.iter().enumerate() {
            let col = result.0[row];
            if col >= n_bels {
                return Err(PlacerError::PlacementFailed(format!(
                    "LAPJV assigned cell to dummy BEL for type {}",
                    ctx.name_of(cell_type),
                )));
            }

            let (bel_id, bx, by) = bels[col];
            let cell_id = idx_to_cell[solver_idx];

            if !ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer) {
                let cell_name = ctx.design.cell(cell_id).name;
                return Err(PlacerError::PlacementFailed(format!(
                    "Failed to bind cell {} during bipartite legalization",
                    ctx.name_of(cell_name)
                )));
            }

            // Compute displacement
            let dx = bx as f64 - cell_x[solver_idx];
            let dy = by as f64 - cell_y[solver_idx];
            total_displacement += dx * dx + dy * dy;

            place_cluster_children(ctx, cell_id, bel_id)?;
        }
    }

    Ok(total_displacement)
}

use crate::context::Context;
use crate::metrics::{compute_net_demand, compute_tile_capacities};
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::common::{verify_cluster_placement, verify_shared_mux_legality};
use crate::placer::legalize::sorted::sorted_legalize;
use crate::placer::PlacerError;

use super::state::HeapState;

impl HeapState {
    /// Legalize the placement: assign each movable cell to the nearest
    /// available BEL of matching bucket type.
    ///
    /// Delegates to the shared `sorted_legalize` algorithm, then updates
    /// HeapState cell_x/cell_y from the bound BEL positions.
    pub(super) fn legalize(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
        // Delegate to shared sorted legalization (now shared-mux aware).
        let type_aware = TypeAwarePlacement::build(ctx, 0, 0);
        sorted_legalize(
            ctx,
            &self.movable_cells,
            &self.cell_x,
            &self.cell_y,
            &type_aware,
        )?;

        // Hard post-checks. Cluster placement runs first because the
        // shared-mux conflicts are usually a downstream symptom of a
        // cluster child rebound to an unrelated BEL by the legalizer's
        // any-available-BEL fallback.
        verify_cluster_placement(ctx)?;
        verify_shared_mux_legality(ctx)?;

        // HeAP-specific: update cell_x/cell_y from bound BEL positions.
        for (i, &cell_idx) in self.movable_cells.iter().enumerate() {
            if let Some(bel) = ctx.cell(cell_idx).bel() {
                let loc = bel.loc();
                self.cell_x[i] = loc.x as f64;
                self.cell_y[i] = loc.y as f64;
            }
        }

        Ok(())
    }

    /// Compute congestion-aware displacement targets for each movable cell.
    ///
    /// For each cell, estimates the local congestion gradient from the edge-demand
    /// grid and computes a target position shifted away from congested edges.
    /// Returns (target_x, target_y, force_weight) for each movable cell.
    pub(super) fn compute_congestion_targets(&self, ctx: &Context) -> Vec<(f64, f64, f64)> {
        let n = self.movable_cells.len();
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let wu = grid_w as usize;
        let hu = grid_h as usize;
        let congestion_weight = self.cfg.congestion_weight;

        // Build capacity grids (total_wires / 4 per direction).
        let (h_capacity, v_capacity) = compute_tile_capacities(ctx);

        // Build demand grids by iterating all alive nets and tracing Bresenham lines.
        let (h_demand, v_demand) = compute_net_demand(ctx);

        // Compute per-cell congestion displacement targets.
        let max_x = (grid_w - 1) as f64;
        let max_y = (grid_h - 1) as f64;

        (0..n)
            .map(|i| {
                let cx = self.cell_x[i];
                let cy = self.cell_y[i];
                let ix = cx.round() as i32;
                let iy = cy.round() as i32;

                // Get congestion at surrounding edges (0.0 if at boundary).
                let east_c = if ix >= 0 && (ix as usize) + 1 < wu && iy >= 0 && (iy as usize) < hu {
                    h_demand[iy as usize][ix as usize] / h_capacity[iy as usize][ix as usize]
                } else {
                    0.0
                };
                let west_c = if ix > 0 && (ix as usize) < wu && iy >= 0 && (iy as usize) < hu {
                    h_demand[iy as usize][(ix - 1) as usize]
                        / h_capacity[iy as usize][(ix - 1) as usize]
                } else {
                    0.0
                };
                let south_c = if iy >= 0 && (iy as usize) + 1 < hu && ix >= 0 && (ix as usize) < wu
                {
                    v_demand[iy as usize][ix as usize] / v_capacity[iy as usize][ix as usize]
                } else {
                    0.0
                };
                let north_c = if iy > 0 && (iy as usize) < hu && ix >= 0 && (ix as usize) < wu {
                    v_demand[(iy - 1) as usize][ix as usize]
                        / v_capacity[(iy - 1) as usize][ix as usize]
                } else {
                    0.0
                };

                // Displacement: push away from congested edges.
                let dx = west_c - east_c; // positive = push east (away from west congestion)
                let dy = north_c - south_c; // positive = push south (away from north congestion)
                let max_c = east_c.max(west_c).max(south_c).max(north_c);

                // Only apply force if congestion is above 1.0 (over-capacity).
                if max_c > 1.0 {
                    let target_x = (cx + dx).clamp(0.0, max_x);
                    let target_y = (cy + dy).clamp(0.0, max_y);
                    let force = self.alpha * congestion_weight * (max_c - 1.0);
                    (target_x, target_y, force)
                } else {
                    (cx, cy, 0.0)
                }
            })
            .collect()
    }
}

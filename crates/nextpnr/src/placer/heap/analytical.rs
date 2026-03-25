use crate::context::Context;
use crate::placer::solver::{Solver, SparseSystem};
use crate::placer::PlacerError;
use log::debug;
use rustc_hash::FxHashSet;

use super::state::HeapState;

impl HeapState {
    /// Build and solve the quadratic wirelength minimization system.
    ///
    /// For 2-pin nets: direct connection between the two cells.
    /// For multi-pin nets (>2 pins): star model with virtual center node.
    /// Anchor forces pull cells toward their current spread positions.
    pub(super) fn solve_analytical(&mut self, ctx: &Context) -> Result<(), PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(());
        }

        let mut sys_x = SparseSystem::new(n);
        let mut sys_y = SparseSystem::new(n);

        let weight = self.cfg.beta;

        // Process each net.
        for (_net_idx, net) in ctx.design.iter_alive_nets() {
            if !net.driver.is_connected() || net.users.is_empty() {
                continue;
            }

            // Collect all movable cell indices on this net, and fixed positions.
            let mut movable_on_net: Vec<usize> = Vec::new();
            let mut movable_seen: FxHashSet<usize> = FxHashSet::default();
            let mut fixed_positions: Vec<(f64, f64)> = Vec::new();

            // Driver cell.
            let drv_cell_idx = net.driver.cell;
            if let Some(&idx) = self.cell_to_idx.get(&drv_cell_idx) {
                if movable_seen.insert(idx) {
                    movable_on_net.push(idx);
                }
            } else {
                // Fixed cell: get its location.
                let cell = ctx.cell(drv_cell_idx);
                if let Some(bel) = cell.bel() {
                    let loc = bel.loc();
                    fixed_positions.push((loc.x as f64, loc.y as f64));
                }
            }

            // User cells.
            for user in &net.users {
                if !user.is_connected() {
                    continue;
                }
                let user_cell_idx = user.cell;
                if let Some(&idx) = self.cell_to_idx.get(&user_cell_idx) {
                    if movable_seen.insert(idx) {
                        movable_on_net.push(idx);
                    }
                } else {
                    let cell = ctx.cell(user_cell_idx);
                    if let Some(bel) = cell.bel() {
                        let loc = bel.loc();
                        fixed_positions.push((loc.x as f64, loc.y as f64));
                    }
                }
            }

            let total_pins = movable_on_net.len() + fixed_positions.len();
            if total_pins < 2 {
                continue;
            }

            if total_pins == 2 && movable_on_net.len() == 2 {
                // 2-pin net, both movable: direct connection.
                let i = movable_on_net[0];
                let j = movable_on_net[1];
                sys_x.add_connection(i, j, weight);
                sys_y.add_connection(i, j, weight);
            } else if total_pins == 2 && movable_on_net.len() == 1 && fixed_positions.len() == 1 {
                // 2-pin net, one movable and one fixed: anchor.
                let i = movable_on_net[0];
                let (fx, fy) = fixed_positions[0];
                sys_x.add_anchor(i, fx, weight);
                sys_y.add_anchor(i, fy, weight);
            } else {
                // Multi-pin net: star model.
                // Compute the centroid of all pins.
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                for &idx in &movable_on_net {
                    sum_x += self.cell_x[idx];
                    sum_y += self.cell_y[idx];
                }
                for &(fx, fy) in &fixed_positions {
                    sum_x += fx;
                    sum_y += fy;
                }
                let centroid_x = sum_x / total_pins as f64;
                let centroid_y = sum_y / total_pins as f64;

                // Connect each movable cell to the centroid with weight
                // proportional to 1/(num_pins - 1) to normalize.
                let star_weight = weight * (total_pins as f64) / ((total_pins - 1) as f64);
                for &idx in &movable_on_net {
                    sys_x.add_anchor(idx, centroid_x, star_weight);
                    sys_y.add_anchor(idx, centroid_y, star_weight);
                }
            }
        }

        // Add anchor forces toward current positions (spreading forces).
        for i in 0..n {
            sys_x.add_anchor(i, self.cell_x[i], self.alpha);
            sys_y.add_anchor(i, self.cell_y[i], self.alpha);
        }

        // Add congestion-aware forces.
        if let Some(ref targets) = self.congestion_targets {
            for i in 0..n {
                let (tx, ty, w) = targets[i];
                if w > 0.0 {
                    sys_x.add_anchor(i, tx, w);
                    sys_y.add_anchor(i, ty, w);
                }
            }
        }

        // Solve X and Y systems in parallel using rayon::join.
        // Split borrows so each closure gets its own &mut slice.
        let tol = self.cfg.solver_tolerance;
        let max_si = self.cfg.max_solver_iters;
        let cell_x = &mut self.cell_x;
        let cell_y = &mut self.cell_y;
        let (iters_x, iters_y) = rayon::join(
            || sys_x.solve(cell_x, tol, max_si),
            || sys_y.solve(cell_y, tol, max_si),
        );

        debug!(
            "HeAP: analytical solve: CG iters x={}, y={}",
            iters_x, iters_y
        );

        // Clamp positions to grid bounds, and region-constrained cells to their region bbox.
        let max_x = (self.grid_w - 1) as f64;
        let max_y = (self.grid_h - 1) as f64;
        for i in 0..n {
            self.cell_x[i] = self.cell_x[i].clamp(0.0, max_x);
            self.cell_y[i] = self.cell_y[i].clamp(0.0, max_y);

            if let Some(region_idx) = self.cell_region[i] {
                if let Some(bbox) = ctx.design.region(region_idx).bounding_box() {
                    self.cell_x[i] = self.cell_x[i].clamp(bbox.x0 as f64, bbox.x1 as f64);
                    self.cell_y[i] = self.cell_y[i].clamp(bbox.y0 as f64, bbox.y1 as f64);
                }
            }
        }

        Ok(())
    }
}

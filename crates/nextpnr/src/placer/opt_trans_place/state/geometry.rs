//! Geometric helpers: pin positions, bilinear interpolation, HPWL, routing demand, clamping.

use rayon::prelude::*;

use crate::context::Context;
use crate::netlist::CellId;

use super::OptTransState;

const PARALLEL_THRESHOLD: usize = 4096;

impl OptTransState {
    /// Continuous position of a pin (movable: from cell_x/y, fixed: from BEL).
    pub fn pin_pos(&self, ctx: &Context, cell_id: CellId) -> (f64, f64) {
        if let Some(&idx) = self.cell_to_idx.get(&cell_id) {
            (self.cell_x[idx], self.cell_y[idx])
        } else {
            // Fixed cell: convert physical BEL position to virtual grid coords
            let cell = ctx.design.cell(cell_id);
            if let Some(bel) = cell.bel {
                let loc = ctx.bel(bel).loc();
                ((loc.x - self.network.x0) as f64, (loc.y - self.network.y0) as f64)
            } else {
                (self.network.width as f64 / 2.0, self.network.height as f64 / 2.0)
            }
        }
    }

    /// Nearest tile for a pin (for timing BFS and legalization diagnostics).
    pub fn pin_tile(&self, ctx: &Context, cell_id: CellId) -> (i32, i32) {
        let (x, y) = self.pin_pos(ctx, cell_id);
        let tx = (x.round() as i32).clamp(0, self.network.width - 1);
        let ty = (y.round() as i32).clamp(0, self.network.height - 1);
        (tx, ty)
    }

    /// Clamp continuous position to grid and compute bilinear cell coordinates.
    /// Returns (x0, y0, fx, fy) where x0/y0 are the lower-left tile indices
    /// and fx/fy are the fractional offsets within the cell.
    pub(super) fn bilinear_cell(&self, x: f64, y: f64) -> (i32, i32, f64, f64) {
        let max_x = (self.network.width - 1) as f64;
        let max_y = (self.network.height - 1) as f64;
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);

        let x0 = (x.floor() as i32).clamp(0, self.network.width - 2);
        let y0 = (y.floor() as i32).clamp(0, self.network.height - 2);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        (x0, y0, fx, fy)
    }

    /// Bilinear weights: maps continuous (x, y) to 4 surrounding tiles with weights.
    /// Returns [(tile_x, tile_y, weight); 4].
    pub(super) fn bilinear_weights(&self, x: f64, y: f64) -> [(i32, i32, f64); 4] {
        let (x0, y0, fx, fy) = self.bilinear_cell(x, y);
        [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x0 + 1, y0, fx * (1.0 - fy)),
            (x0, y0 + 1, (1.0 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy),
        ]
    }

    /// Bilinear gradient of a scalar field at cell position.
    ///
    /// Given 4 corner values (f00, f10, f01, f11) and fractional offsets (fx, fy):
    ///   df/dx = (1-fy)(f10-f00) + fy(f11-f01)
    ///   df/dy = (1-fx)(f01-f00) + fx(f11-f10)
    #[inline]
    pub(super) fn bilinear_gradient(fx: f64, fy: f64, f00: f64, f10: f64, f01: f64, f11: f64) -> (f64, f64) {
        let gx = (1.0 - fy) * (f10 - f00) + fy * (f11 - f01);
        let gy = (1.0 - fx) * (f01 - f00) + fx * (f11 - f10);
        (gx, gy)
    }

    /// Compute per-cell gradients in parallel (or sequentially for small N),
    /// returning separate x and y component vectors.
    pub(super) fn parallel_gradient(&self, f: impl Fn(usize) -> (f64, f64) + Sync) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_cells();
        let pairs: Vec<(f64, f64)> = if n >= PARALLEL_THRESHOLD {
            (0..n).into_par_iter().map(&f).collect()
        } else {
            (0..n).map(f).collect()
        };
        pairs.into_iter().unzip()
    }

    /// Per-cell bilinear gradient of a 2D scalar field stored in row-major order.
    pub(crate) fn field_gradient(&self, field: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
        self.parallel_gradient(|i| {
            let (x0, y0, fx, fy) = self.bilinear_cell(self.cell_x[i], self.cell_y[i]);
            let row0 = y0 as usize * w;
            let row1 = (y0 + 1) as usize * w;
            let col0 = x0 as usize;
            let col1 = (x0 + 1) as usize;
            Self::bilinear_gradient(fx, fy, field[row0 + col0], field[row0 + col1], field[row1 + col0], field[row1 + col1])
        })
    }

    /// Continuous HPWL: sum of half-perimeter bounding boxes from continuous positions.
    /// No legalization needed -- uses cell_x/cell_y directly.
    pub fn continuous_hpwl(&self, ctx: &Context) -> f64 {
        let mut total = 0.0;
        for (_, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let (mut min_x, mut max_x) = (dx, dx);
            let (mut min_y, mut max_y) = (dy, dy);

            let mut has_valid_sink = false;
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                has_valid_sink = true;
                let (x, y) = self.pin_pos(ctx, user.cell);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            if !has_valid_sink {
                continue;
            }

            total += (max_x - min_x) + (max_y - min_y);
        }
        total
    }

    /// Build a routing congestion field: for each tile, estimate how many
    /// net bounding boxes pass through it.  Tiles with high routing demand
    /// should be treated as congested even if cell density is low.
    ///
    /// Returns a field of the same size as the density field.
    pub fn build_routing_demand_field(&self, ctx: &Context) -> Vec<f64> {
        let w = self.network.width as usize;
        let h = self.network.height as usize;
        let mut demand = vec![0.0_f64; w * h];

        for (_, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };
            if net.num_users() == 0 { continue; }

            // Compute bounding box of the net in virtual coords
            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let (mut min_x, mut max_x) = (dx, dx);
            let (mut min_y, mut max_y) = (dy, dy);

            for user in net.users() {
                if !user.is_valid() { continue; }
                let (ux, uy) = self.pin_pos(ctx, user.cell);
                min_x = min_x.min(ux);
                max_x = max_x.max(ux);
                min_y = min_y.min(uy);
                max_y = max_y.max(uy);
            }

            // Add demand to all tiles in the bounding box
            let x0 = (min_x.floor() as i32).max(0) as usize;
            let x1 = (max_x.ceil() as i32).min(w as i32 - 1) as usize;
            let y0 = (min_y.floor() as i32).max(0) as usize;
            let y1 = (max_y.ceil() as i32).min(h as i32 - 1) as usize;

            // Weight inversely with bbox area (larger nets spread demand thinner)
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)).max(1) as f64;
            let weight = 1.0 / area;

            for ty in y0..=y1 {
                let row = ty * w;
                for tx in x0..=x1 {
                    demand[row + tx] += weight;
                }
            }
        }

        demand
    }

    pub fn clamp_positions(&mut self) {
        let max_x = (self.network.width - 1) as f64;
        let max_y = (self.network.height - 1) as f64;
        crate::placer::common::clamp_positions(&mut self.cell_x, &mut self.cell_y, max_x, max_y);
    }

    /// Clamp positions to an expanding bounding box centered on the IO centroid.
    /// `progress` goes from 0.0 (initial tight box) to 1.0 (full grid).
    /// The box starts at `box_initial_half` (just enough BELs for all cells)
    /// and expands to the full grid.
    pub fn clamp_to_box(&mut self, progress: f64) {
        let (cx, cy) = self.box_center;
        let grid_x = (self.network.width - 1) as f64;
        let grid_y = (self.network.height - 1) as f64;

        // Interpolate half-extent from initial tight box to full grid extent.
        let half_x = self.box_initial_half + (grid_x - self.box_initial_half) * progress;
        let half_y = self.box_initial_half + (grid_y - self.box_initial_half) * progress;

        let (min_x, max_x) = ((cx - half_x).max(0.0), (cx + half_x).min(grid_x));
        let (min_y, max_y) = ((cy - half_y).max(0.0), (cy + half_y).min(grid_y));
        for i in 0..self.cell_x.len() {
            self.cell_x[i] = self.cell_x[i].clamp(min_x, max_x);
            self.cell_y[i] = self.cell_y[i].clamp(min_y, max_y);
        }
    }
}

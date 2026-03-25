//! Density field computation, energy, overlap metrics, and density gradients.

use crate::context::Context;

use super::OptTransState;

impl OptTransState {
    /// Build normalized density field using smooth bell-shaped cell basis functions.
    ///
    /// Each cell contributes a Gaussian bump to surrounding tiles:
    ///   contribution = exp(-dist^2/(2 sigma^2))
    /// where sigma = cell_radius (proportional to 1/sqrt(avg_capacity)).
    ///
    /// Unlike bilinear splatting (2x2 tile support), the bell function has a
    /// wider footprint (~3 sigma radius). This gives co-located cells different
    /// gradients because their exact sub-tile positions produce different
    /// contributions to surrounding tiles. Critical for breaking the symmetry
    /// of overlapping cells at centroid init.
    ///
    /// Density is normalized by per-tile BEL capacity: rho=1.0 means full.
    pub fn build_density_field(&self, _ctx: &Context) -> Vec<f64> {
        let w = self.network.width as usize;
        let h = self.network.height as usize;
        let n = self.num_cells();

        // Cell radius: each cell occupies ~1/avg_bels of a tile.
        // sigma = sqrt(1/avg_bels) gives appropriate spreading width.
        let cell_sigma = (1.0 / self.avg_bels).sqrt().max(0.3);
        let radius = (3.0 * cell_sigma).ceil() as i32;
        let inv_2sigma2 = 1.0 / (2.0 * cell_sigma * cell_sigma);

        let mut density = vec![0.0; w * h];
        let wi = w as i32;
        let hi = h as i32;

        for i in 0..n {
            let px = self.cell_x[i];
            let py = self.cell_y[i];
            let floor_x = px.floor() as i32;
            let floor_y = py.floor() as i32;

            let ty_lo = (floor_y - radius).max(0);
            let ty_hi = (floor_y + radius).min(hi - 1);
            let tx_lo = (floor_x - radius).max(0);
            let tx_hi = (floor_x + radius).min(wi - 1);

            // Splat bell function over tiles within radius.
            for ty in ty_lo..=ty_hi {
                let dy = ty as f64 + 0.5 - py;
                let dy2 = dy * dy;
                let row_offset = ty as usize * w;
                for tx in tx_lo..=tx_hi {
                    let dx = tx as f64 + 0.5 - px;
                    let dist2 = dx * dx + dy2;
                    let weight = (-dist2 * inv_2sigma2).exp();
                    density[row_offset + tx as usize] += weight;
                }
            }
        }

        // Normalize by per-tile BEL capacity (cached).
        for i in 0..w * h {
            density[i] /= self.tile_bel_cap[i];
        }
        density
    }

    /// Quadratic capacity deviation energy: E_density = sum (rho_i - 1)^2.
    ///
    /// Penalizes ALL deviation from capacity: overcrowded tiles push cells out,
    /// underfilled tiles pull cells in. Targets uniform capacity usage.
    pub fn density_energy(&self, ctx: &Context) -> f64 {
        let density = self.build_density_field(ctx);
        Self::density_energy_from_field(&density)
    }

    /// Quadratic capacity deviation from a pre-computed density field.
    pub fn density_energy_from_field(density: &[f64]) -> f64 {
        density.iter()
            .map(|&rho| { let dev = rho - 1.0; dev * dev })
            .sum()
    }

    /// Cell overlap metrics from the density field.
    ///
    /// Returns (overflow_ratio, max_density, overflow_count):
    /// - overflow_ratio: fraction of occupied tiles where rho > 1.0 (above capacity)
    /// - max_density: highest rho value across all tiles
    /// - overflow_count: number of tiles exceeding capacity
    pub fn overlap_metrics(&self, ctx: &Context) -> (f64, f64, usize) {
        let density = self.build_density_field(ctx);
        Self::overlap_metrics_from_field(&density)
    }

    /// Overlap metrics from a pre-computed density field.
    pub fn overlap_metrics_from_field(density: &[f64]) -> (f64, f64, usize) {
        let mut max_rho = 0.0_f64;
        let mut overflow_count = 0usize;
        let mut occupied_count = 0usize;
        for &rho in density {
            max_rho = max_rho.max(rho);
            if rho > 1e-6 {
                occupied_count += 1;
                if rho > 1.0 {
                    overflow_count += 1;
                }
            }
        }
        let overflow_ratio = if occupied_count > 0 {
            overflow_count as f64 / occupied_count as f64
        } else {
            0.0
        };
        (overflow_ratio, max_rho, overflow_count)
    }

    /// Compute density gradient using quadratic capacity deviation.
    ///
    /// Potential at each tile: P[tile] = lambda[tile] * (rho[tile] - 1)^2
    /// where lambda is the per-tile Augmented Lagrangian multiplier.
    ///
    /// Penalizes all deviation from capacity.
    pub fn compute_density_gradient(
        &self,
        ctx: &Context,
        sigma: f64,
        tile_multipliers: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let density = self.build_density_field(ctx);
        self.density_gradient_from_field(&density, sigma, tile_multipliers)
    }

    /// Density gradient from a pre-computed density field.
    pub fn density_gradient_from_field(
        &self,
        density: &[f64],
        sigma: f64,
        tile_multipliers: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let w = self.network.width as usize;
        let h = self.network.height as usize;
        let num_tiles = w * h;

        // Quadratic capacity deviation: P = lambda * (rho - 1)^2 for ALL tiles.
        let mut pressure = vec![0.0; num_tiles];
        for i in 0..num_tiles {
            let dev = density[i] - 1.0;
            pressure[i] = tile_multipliers[i] * dev * dev;
        }

        // Gaussian blur + gradient.
        let blurred = gaussian_blur_2d(&pressure, w, h, sigma);
        self.field_gradient(&blurred, w)
    }
}

/// Separable Gaussian blur on a 2D grid.
pub(super) fn gaussian_blur_2d(input: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma < 0.5 {
        return input.to_vec();
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = (0..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let norm: f64 = kernel[0] + 2.0 * kernel[1..].iter().sum::<f64>();

    // Horizontal pass.
    let mut temp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = input[y * w + x] * kernel[0];
            for k in 1..=radius {
                let left = if x >= k { x - k } else { 0 };
                let right = (x + k).min(w - 1);
                sum += input[y * w + left] * kernel[k];
                sum += input[y * w + right] * kernel[k];
            }
            temp[y * w + x] = sum / norm;
        }
    }

    // Vertical pass.
    let mut output = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = temp[y * w + x] * kernel[0];
            for k in 1..=radius {
                let up = if y >= k { y - k } else { 0 };
                let down = (y + k).min(h - 1);
                sum += temp[up * w + x] * kernel[k];
                sum += temp[down * w + x] * kernel[k];
            }
            output[y * w + x] = sum / norm;
        }
    }

    output
}

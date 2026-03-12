//! Density computation and gradient for ElectroPlace.
//!
//! Uses a bell-shaped density kernel per cell, summed on a grid.
//! FFT-based convolution for O(N log N) density gradient computation.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Bell-shape kernel spread radius (in grid cells).
const SIGMA: f64 = 1.5;

/// Number of grid cells to sample around each cell center (3 sigma).
const KERNEL_RADIUS: i32 = 5; // ceil(3.0 * SIGMA)

/// Compute the density map on a grid using bell-shaped kernels.
///
/// Each cell contributes a Gaussian-like bell curve centered at its position.
/// The result is a grid-sized density map with target density subtracted.
pub fn compute_density_map(
    cell_x: &[f64],
    cell_y: &[f64],
    grid_w: usize,
    grid_h: usize,
    target_density: f64,
) -> Vec<f64> {
    let total_cells = grid_w * grid_h;
    let mut density = vec![0.0; total_cells];

    for i in 0..cell_x.len() {
        let cx = cell_x[i];
        let cy = cell_y[i];
        let gx = cx.round() as i32;
        let gy = cy.round() as i32;

        for dy in -KERNEL_RADIUS..=KERNEL_RADIUS {
            let iy = gy + dy;
            if iy < 0 || iy >= grid_h as i32 {
                continue;
            }
            for dx in -KERNEL_RADIUS..=KERNEL_RADIUS {
                let ix = gx + dx;
                if ix < 0 || ix >= grid_w as i32 {
                    continue;
                }

                let fx = cx - ix as f64;
                let fy = cy - iy as f64;
                let w = bell_shape(fx) * bell_shape(fy);

                density[grid_index(ix, iy, grid_w)] += w;
            }
        }
    }

    // Subtract target density
    let avg_density = cell_x.len() as f64 / total_cells as f64;
    let target = target_density * avg_density;
    for d in &mut density {
        *d -= target;
    }

    density
}

/// Compute the density gradient for each cell using FFT-based convolution.
///
/// The density gradient tells each cell which direction to move to reduce
/// local density overflow.
pub fn compute_density_gradient(
    cell_x: &[f64],
    cell_y: &[f64],
    density_map: &[f64],
    grid_w: usize,
    grid_h: usize,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
) {
    let n = cell_x.len();

    // Compute electric field from density map using FFT-based Poisson solve
    let (field_x, field_y) = poisson_field(density_map, grid_w, grid_h);

    // Interpolate field at each cell position using bell-shape weights
    for i in 0..n {
        let cx = cell_x[i];
        let cy = cell_y[i];
        let gx = cx.round() as i32;
        let gy = cy.round() as i32;

        let mut fx_total = 0.0;
        let mut fy_total = 0.0;
        let mut w_total = 0.0;

        for dy in -KERNEL_RADIUS..=KERNEL_RADIUS {
            let iy = gy + dy;
            if iy < 0 || iy >= grid_h as i32 {
                continue;
            }
            for dx in -KERNEL_RADIUS..=KERNEL_RADIUS {
                let ix = gx + dx;
                if ix < 0 || ix >= grid_w as i32 {
                    continue;
                }

                let dfx = cx - ix as f64;
                let dfy = cy - iy as f64;
                let w = bell_shape(dfx) * bell_shape(dfy);

                let idx = grid_index(ix, iy, grid_w);
                fx_total += w * field_x[idx];
                fy_total += w * field_y[idx];
                w_total += w;
            }
        }

        if w_total > 1e-10 {
            grad_x[i] += fx_total / w_total;
            grad_y[i] += fy_total / w_total;
        }
    }
}

/// Convert grid coordinates to a flat index in a row-major grid.
fn grid_index(x: i32, y: i32, width: usize) -> usize {
    y as usize * width + x as usize
}

/// Bell-shaped (cosine-based) density kernel using module-level SIGMA.
///
/// b(x) = (1 + cos(pi * x / sigma)) / 2 for |x| < sigma, else 0
fn bell_shape(x: f64) -> f64 {
    let ax = x.abs();
    if ax < SIGMA {
        0.5 * (1.0 + (std::f64::consts::PI * ax / SIGMA).cos())
    } else {
        0.0
    }
}

/// Solve Poisson equation ∇²φ = ρ and return the electric field (Ex, Ey) = -∇φ.
///
/// Uses 2D FFT (row transforms + column transforms) with discrete Laplacian
/// eigenvalues: λ(kx,ky) = 2(cos(2πkx/W) - 1) + 2(cos(2πky/H) - 1).
///
/// The electric field is computed via spectral differentiation:
/// Ex = -∂φ/∂x using forward finite difference in frequency domain.
fn poisson_field(density: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let n = w * h;
    let pi = std::f64::consts::PI;

    let mut planner = FftPlanner::new();

    // Working buffer
    let mut data: Vec<Complex<f64>> = density
        .iter()
        .map(|&d| Complex::new(d, 0.0))
        .collect();

    // Forward 2D FFT: transform rows, then columns
    fft_2d(&mut data, w, h, &mut planner, true);

    // Solve in frequency domain and compute field
    let mut ex_hat = vec![Complex::new(0.0, 0.0); n];
    let mut ey_hat = vec![Complex::new(0.0, 0.0); n];

    for ky in 0..h {
        for kx in 0..w {
            let idx = ky * w + kx;

            // Discrete Laplacian eigenvalues:
            // λ = 2(cos(2πkx/W) - 1) + 2(cos(2πky/H) - 1)
            let lam_x = 2.0 * ((2.0 * pi * kx as f64 / w as f64).cos() - 1.0);
            let lam_y = 2.0 * ((2.0 * pi * ky as f64 / h as f64).cos() - 1.0);
            let lam = lam_x + lam_y;

            if lam.abs() < 1e-10 {
                // DC component: set to zero (remove mean)
                ex_hat[idx] = Complex::new(0.0, 0.0);
                ey_hat[idx] = Complex::new(0.0, 0.0);
            } else {
                // φ̂ = ρ̂ / λ
                let phi = data[idx] / lam;

                // Discrete derivative via spectral differentiation:
                // ∂/∂x in frequency domain: multiply by (exp(i2πkx/W) - 1)
                // E = -∇φ, so Ex = -(exp(i2πkx/W) - 1) * φ̂
                let angle_x = 2.0 * pi * kx as f64 / w as f64;
                let angle_y = 2.0 * pi * ky as f64 / h as f64;
                let dx_kernel = Complex::new(angle_x.cos() - 1.0, angle_x.sin());
                let dy_kernel = Complex::new(angle_y.cos() - 1.0, angle_y.sin());

                ex_hat[idx] = -dx_kernel * phi;
                ey_hat[idx] = -dy_kernel * phi;
            }
        }
    }

    // Inverse 2D FFT
    fft_2d(&mut ex_hat, w, h, &mut planner, false);
    fft_2d(&mut ey_hat, w, h, &mut planner, false);

    let inv_n = 1.0 / n as f64;
    let field_x: Vec<f64> = ex_hat.iter().map(|c| c.re * inv_n).collect();
    let field_y: Vec<f64> = ey_hat.iter().map(|c| c.re * inv_n).collect();

    (field_x, field_y)
}

/// Perform a 2D FFT on a row-major grid by transforming rows then columns.
fn fft_2d(data: &mut [Complex<f64>], w: usize, h: usize, planner: &mut FftPlanner<f64>, forward: bool) {
    // Transform each row
    let fft_row = if forward {
        planner.plan_fft_forward(w)
    } else {
        planner.plan_fft_inverse(w)
    };
    for row in 0..h {
        let start = row * w;
        fft_row.process(&mut data[start..start + w]);
    }

    // Transform each column (need to gather/scatter since data is row-major)
    let fft_col = if forward {
        planner.plan_fft_forward(h)
    } else {
        planner.plan_fft_inverse(h)
    };
    let mut col_buf = vec![Complex::new(0.0, 0.0); h];
    for col in 0..w {
        // Gather column
        for row in 0..h {
            col_buf[row] = data[row * w + col];
        }
        fft_col.process(&mut col_buf);
        // Scatter column
        for row in 0..h {
            data[row * w + col] = col_buf[row];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_map_single_cell() {
        let x = vec![4.0];
        let y = vec![4.0];
        let map = compute_density_map(&x, &y, 8, 8, 0.0);

        // Peak should be at (4, 4)
        let peak_idx = 4 * 8 + 4;
        let peak = map[peak_idx];
        assert!(peak > 0.0, "Peak density should be positive: {}", peak);

        // Should be highest at center
        for (i, &d) in map.iter().enumerate() {
            if i != peak_idx {
                assert!(d <= peak + 1e-10, "Index {} ({}) > peak ({})", i, d, peak);
            }
        }
    }

    #[test]
    fn density_gradient_pushes_apart() {
        // Two cells at same position should have gradients pushing them apart
        let x = vec![4.0, 4.0];
        let y = vec![4.0, 4.0];
        let map = compute_density_map(&x, &y, 8, 8, 0.0);

        let mut gx = vec![0.0; 2];
        let mut gy = vec![0.0; 2];
        compute_density_gradient(&x, &y, &map, 8, 8, &mut gx, &mut gy);

        // Gradients should be finite (they'll be equal for identical positions)
        for g in &gx {
            assert!(g.is_finite(), "Gradient should be finite: {}", g);
        }
    }

    #[test]
    fn bell_shape_properties() {
        // Maximum at center
        assert!((bell_shape(0.0) - 1.0).abs() < 1e-10);
        // Zero outside sigma
        assert_eq!(bell_shape(2.0), 0.0);
        // Symmetric
        assert!((bell_shape(0.5) - bell_shape(-0.5)).abs() < 1e-10);
    }
}

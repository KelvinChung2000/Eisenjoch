//! DCT-based density computation and Poisson solve for ElectroPlace.
//!
//! Aligned with placer_static.cc: uses DCT-II forward, spectral Poisson solve,
//! and mixed DST/DCT inverse transforms for field recovery.
//! Cell-to-bin mapping uses slither (fractional bin overlap based on cell area).

use rustdct::DctPlanner;
use std::f64::consts::PI;

/// Compute the grid size as the next power of two >= max(width, height).
pub fn grid_size(width: usize, height: usize) -> usize {
    let m = width.max(height);
    m.next_power_of_two().max(4)
}

/// Compute the bin range and bounding box for a unit-area cell centered at (cx, cy).
///
/// Returns (x_lo, x_hi, y_lo, y_hi, bx_lo, bx_hi, by_lo, by_hi).
#[inline]
fn cell_bin_overlap(
    cx: f64,
    cy: f64,
    grid_w: usize,
    grid_h: usize,
) -> (f64, f64, f64, f64, usize, usize, usize, usize) {
    let x_lo = cx - 0.5;
    let x_hi = cx + 0.5;
    let y_lo = cy - 0.5;
    let y_hi = cy + 0.5;

    let bx_lo = (x_lo.floor() as i32).max(0) as usize;
    let bx_hi = (x_hi.ceil() as usize).min(grid_w);
    let by_lo = (y_lo.floor() as i32).max(0) as usize;
    let by_hi = (y_hi.ceil() as usize).min(grid_h);

    (x_lo, x_hi, y_lo, y_hi, bx_lo, bx_hi, by_lo, by_hi)
}

/// Compute the fractional overlap between a cell bounding box and a bin.
#[inline]
fn bin_overlap_area(
    x_lo: f64, x_hi: f64, y_lo: f64, y_hi: f64,
    bx: usize, by: usize,
) -> f64 {
    let ox = (x_hi.min(bx as f64 + 1.0) - x_lo.max(bx as f64)).max(0.0);
    let oy = (y_hi.min(by as f64 + 1.0) - y_lo.max(by as f64)).max(0.0);
    ox * oy
}

/// Compute concrete (raw) density by mapping cells to bins via slither overlap.
///
/// Each cell occupies area=1 (one BEL). The density at each bin is the sum
/// of fractional overlaps with all cells.
pub fn compute_concrete_density(
    cell_x: &[f64],
    cell_y: &[f64],
    grid_w: usize,
    grid_h: usize,
) -> Vec<f64> {
    let mut density = vec![0.0; grid_w * grid_h];

    for i in 0..cell_x.len() {
        let (x_lo, x_hi, y_lo, y_hi, bx_lo, bx_hi, by_lo, by_hi) =
            cell_bin_overlap(cell_x[i], cell_y[i], grid_w, grid_h);

        for by in by_lo..by_hi {
            for bx in bx_lo..bx_hi {
                density[by * grid_w + bx] += bin_overlap_area(x_lo, x_hi, y_lo, y_hi, bx, by);
            }
        }
    }

    density
}

/// Compute overlap metric: sum(max(0, d-1)) / sum(d).
///
/// Returns 0 for non-overlapping placements, positive for overlapping.
pub fn compute_overlap(concrete_density: &[f64]) -> f64 {
    let total: f64 = concrete_density.iter().sum();
    if total < 1e-30 {
        return 0.0;
    }
    let overflow: f64 = concrete_density.iter().map(|&d| (d - 1.0).max(0.0)).sum();
    overflow / total
}

/// DCT-based density solve: compute electric field (Ex, Ey) from density.
///
/// Steps:
/// 1. Subtract target density to get charge density rho
/// 2. Forward DCT-II (rows then columns)
/// 3. Spectral scaling and Poisson solve
/// 4. Inverse transforms: IDST-rows/IDCT-cols for Ex, IDCT-rows/IDST-cols for Ey
/// 5. Return field grids (Ex, Ey)
pub fn compute_density_field(
    density: &[f64],
    grid_w: usize,
    grid_h: usize,
    target_density: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = grid_w * grid_h;

    // Compute charge density: rho = density - target
    let avg_density = density.iter().sum::<f64>() / n as f64;
    let target = target_density * avg_density.max(1e-30);
    let mut rho: Vec<f64> = density.iter().map(|&d| d - target).collect();

    // Forward DCT-II (rows then columns)
    let mut planner = DctPlanner::new();
    dct2_2d(&mut rho, grid_w, grid_h, &mut planner);

    // Spectral scaling: first row/col by 0.5, all by 4/(m*m)
    let scale = 4.0 / (grid_w as f64 * grid_h as f64);
    for ky in 0..grid_h {
        for kx in 0..grid_w {
            let idx = ky * grid_w + kx;
            let mut s = scale;
            if kx == 0 {
                s *= 0.5;
            }
            if ky == 0 {
                s *= 0.5;
            }
            rho[idx] *= s;
        }
    }

    // Poisson solve and field computation in spectral domain
    let mut phi = vec![0.0; n];
    let mut ex_spec = vec![0.0; n];
    let mut ey_spec = vec![0.0; n];

    for ky in 0..grid_h {
        let wy = PI * ky as f64 / grid_h as f64;
        let wy2 = 2.0 * (1.0 - wy.cos());
        for kx in 0..grid_w {
            let idx = ky * grid_w + kx;
            let wx = PI * kx as f64 / grid_w as f64;
            let wx2 = 2.0 * (1.0 - wx.cos());
            let denom = wx2 + wy2;

            if denom < 1e-30 {
                phi[idx] = 0.0;
                ex_spec[idx] = 0.0;
                ey_spec[idx] = 0.0;
            } else {
                phi[idx] = rho[idx] / denom;
                ex_spec[idx] = phi[idx] * wx;
                ey_spec[idx] = phi[idx] * wy;
            }
        }
    }

    // Inverse transforms for Ex: IDST-rows, IDCT-cols
    let mut ex = ex_spec;
    idst_rows(&mut ex, grid_w, grid_h, &mut planner);
    idct_cols(&mut ex, grid_w, grid_h, &mut planner);

    // Inverse transforms for Ey: IDCT-rows, IDST-cols
    let mut ey = ey_spec;
    idct_rows(&mut ey, grid_w, grid_h, &mut planner);
    idst_cols(&mut ey, grid_w, grid_h, &mut planner);

    (ex, ey)
}

/// Compute density gradient for each cell by interpolating the field at cell positions.
///
/// Uses slither weights (fractional bin overlap) for interpolation.
pub fn compute_density_gradient(
    cell_x: &[f64],
    cell_y: &[f64],
    field_x: &[f64],
    field_y: &[f64],
    grid_w: usize,
    grid_h: usize,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
) {
    for i in 0..cell_x.len() {
        let (x_lo, x_hi, y_lo, y_hi, bx_lo, bx_hi, by_lo, by_hi) =
            cell_bin_overlap(cell_x[i], cell_y[i], grid_w, grid_h);

        let mut fx_total = 0.0;
        let mut fy_total = 0.0;
        let mut w_total = 0.0;

        for by in by_lo..by_hi {
            for bx in bx_lo..bx_hi {
                let w = bin_overlap_area(x_lo, x_hi, y_lo, y_hi, bx, by);
                if w > 0.0 {
                    let idx = by * grid_w + bx;
                    fx_total += w * field_x[idx];
                    fy_total += w * field_y[idx];
                    w_total += w;
                }
            }
        }

        if w_total > 1e-10 {
            grad_x[i] += fx_total / w_total;
            grad_y[i] += fy_total / w_total;
        }
    }
}

// --- DCT/DST transform helpers ---

/// Forward DCT-II on a 2D grid (rows then columns).
fn dct2_2d(data: &mut [f64], w: usize, h: usize, planner: &mut DctPlanner<f64>) {
    let dct_row = planner.plan_dct2(w);
    for row in 0..h {
        let start = row * w;
        dct_row.process_dct2(&mut data[start..start + w]);
    }

    let dct_col = planner.plan_dct2(h);
    let mut col_buf = vec![0.0; h];
    for col in 0..w {
        for row in 0..h {
            col_buf[row] = data[row * w + col];
        }
        dct_col.process_dct2(&mut col_buf);
        for row in 0..h {
            data[row * w + col] = col_buf[row];
        }
    }
}

/// Inverse DCT (IDCT) on rows.
fn idct_rows(data: &mut [f64], w: usize, h: usize, planner: &mut DctPlanner<f64>) {
    let idct = planner.plan_dct3(w);
    for row in 0..h {
        let start = row * w;
        idct.process_dct3(&mut data[start..start + w]);
    }
}

/// Inverse DCT (IDCT) on columns.
fn idct_cols(data: &mut [f64], w: usize, h: usize, planner: &mut DctPlanner<f64>) {
    let idct = planner.plan_dct3(h);
    let mut col_buf = vec![0.0; h];
    for col in 0..w {
        for row in 0..h {
            col_buf[row] = data[row * w + col];
        }
        idct.process_dct3(&mut col_buf);
        for row in 0..h {
            data[row * w + col] = col_buf[row];
        }
    }
}

/// Inverse DST (IDST) on rows.
fn idst_rows(data: &mut [f64], w: usize, h: usize, planner: &mut DctPlanner<f64>) {
    let idst = planner.plan_dst3(w);
    for row in 0..h {
        let start = row * w;
        idst.process_dst3(&mut data[start..start + w]);
    }
}

/// Inverse DST (IDST) on columns.
fn idst_cols(data: &mut [f64], w: usize, h: usize, planner: &mut DctPlanner<f64>) {
    let idst = planner.plan_dst3(h);
    let mut col_buf = vec![0.0; h];
    for col in 0..w {
        for row in 0..h {
            col_buf[row] = data[row * w + col];
        }
        idst.process_dst3(&mut col_buf);
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
        let map = compute_concrete_density(&x, &y, 8, 8);

        // Cell at (4,4) with area 1x1 should produce density ~1.0 at (4,4)
        let peak_idx = 4 * 8 + 4;
        let peak = map[peak_idx];
        assert!(peak > 0.0, "Peak density should be positive: {}", peak);
    }

    #[test]
    fn overlap_zero_for_spread() {
        // 4 cells spread far apart on 8x8 grid
        let x = vec![1.0, 3.0, 5.0, 7.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let density = compute_concrete_density(&x, &y, 8, 8);
        let overlap = compute_overlap(&density);
        assert!(
            overlap < 0.01,
            "Spread cells should have near-zero overlap: {}",
            overlap
        );
    }

    #[test]
    fn overlap_positive_for_clustered() {
        // Many cells at same integer position - each cell centered on bin,
        // so all area goes to one bin. 8 cells at bin (4,4) = density 8.
        let x = vec![4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5];
        let y = vec![4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5];
        let density = compute_concrete_density(&x, &y, 8, 8);
        let overlap = compute_overlap(&density);
        assert!(
            overlap > 0.0,
            "Clustered cells should have positive overlap: {}",
            overlap
        );
    }

    #[test]
    fn density_field_finite_values() {
        let x = vec![2.0, 6.0, 4.0];
        let y = vec![2.0, 6.0, 4.0];
        let density = compute_concrete_density(&x, &y, 8, 8);
        let (fx, fy) = compute_density_field(&density, 8, 8, 1.0);

        for val in fx.iter().chain(fy.iter()) {
            assert!(val.is_finite(), "Field value should be finite: {}", val);
        }
    }

    #[test]
    fn density_gradient_finite() {
        let x = vec![2.0, 6.0, 4.0];
        let y = vec![2.0, 6.0, 4.0];
        let density = compute_concrete_density(&x, &y, 8, 8);
        let (fx, fy) = compute_density_field(&density, 8, 8, 1.0);

        let mut gx = vec![0.0; 3];
        let mut gy = vec![0.0; 3];
        compute_density_gradient(&x, &y, &fx, &fy, 8, 8, &mut gx, &mut gy);

        for g in gx.iter().chain(gy.iter()) {
            assert!(g.is_finite(), "Gradient should be finite: {}", g);
        }
    }
}

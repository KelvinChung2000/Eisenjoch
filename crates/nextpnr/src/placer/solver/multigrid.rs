//! Multigrid V-cycle solver for grid Laplacian systems.
//!
//! Provides O(N) solution of (A + εI)x = b where A is a 2D grid Laplacian
//! (5-point stencil) and ε is a small regularization to ensure non-singularity.
//! Uses geometric multigrid with:
//! - Full-weighting restriction (fine → coarse)
//! - Bilinear prolongation (coarse → fine)
//! - Weighted Jacobi smoother (ω = 2/3)
//! - Direct solve on coarsest level (≤ 4×4)

/// Multigrid V-cycle solver for 2D grid Laplacian systems.
pub struct MultigridSolver {
    /// Grid hierarchy from fine (index 0) to coarsest.
    levels: Vec<GridLevel>,
    /// Number of pre-smoothing iterations.
    pre_smooth: usize,
    /// Number of post-smoothing iterations.
    post_smooth: usize,
    /// Regularization parameter (added to diagonal).
    epsilon: f64,
}

struct GridLevel {
    width: usize,
    height: usize,
}

impl MultigridSolver {
    /// Create a new multigrid solver for a grid of the given dimensions.
    ///
    /// Builds a hierarchy of coarsening levels until the grid is ≤ 4×4.
    pub fn new(width: usize, height: usize) -> Self {
        let mut levels = vec![GridLevel { width, height }];
        let mut w = width;
        let mut h = height;

        while w > 4 || h > 4 {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            levels.push(GridLevel {
                width: w,
                height: h,
            });
        }

        Self {
            levels,
            pre_smooth: 3,
            post_smooth: 3,
            epsilon: 0.01, // Small regularization for non-singularity
        }
    }

    /// Set the regularization parameter.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Solve the regularized grid Laplacian system (A + εI)x = b using V-cycles.
    pub fn solve(&self, rhs: &[f64], x: &mut [f64], v_cycles: usize) {
        let n = self.levels[0].width * self.levels[0].height;
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(x.len(), n);

        for _ in 0..v_cycles {
            self.v_cycle(0, rhs, x);
        }
    }

    /// Perform one V-cycle starting at the given level.
    fn v_cycle(&self, level: usize, rhs: &[f64], x: &mut [f64]) {
        let w = self.levels[level].width;
        let h = self.levels[level].height;

        // Coarsest level: direct solve
        if level == self.levels.len() - 1 {
            direct_solve(x, rhs, w, h, self.epsilon);
            return;
        }

        // Pre-smooth
        smooth(x, rhs, w, h, self.epsilon, self.pre_smooth);

        // Compute residual: r = b - (A + εI)x
        let mut residual = vec![0.0; w * h];
        compute_residual(x, rhs, &mut residual, w, h, self.epsilon);

        // Restrict residual to coarser grid
        let cw = self.levels[level + 1].width;
        let ch = self.levels[level + 1].height;
        let coarse_rhs = restrict(&residual, w, h, cw, ch);

        // Solve on coarser grid (error equation)
        let mut coarse_correction = vec![0.0; cw * ch];
        self.v_cycle(level + 1, &coarse_rhs, &mut coarse_correction);

        // Prolongate correction and add to solution
        let fine_correction = prolongate(&coarse_correction, cw, ch, w, h);
        for i in 0..x.len() {
            x[i] += fine_correction[i];
        }

        // Post-smooth
        smooth(x, rhs, w, h, self.epsilon, self.post_smooth);
    }

    /// Residual norm ||b - (A+εI)x||₂ for convergence checking.
    pub fn residual_norm(&self, rhs: &[f64], x: &[f64]) -> f64 {
        let w = self.levels[0].width;
        let h = self.levels[0].height;
        let mut res = vec![0.0; w * h];
        compute_residual(x, rhs, &mut res, w, h, self.epsilon);
        res.iter().map(|r| r * r).sum::<f64>().sqrt()
    }
}

/// Apply the regularized 5-point Laplacian: ((A + εI)x)[i].
///
/// Stencil: center = num_neighbors + ε, off-diagonal = -1 per neighbor.
#[inline]
fn apply_operator(x: &[f64], w: usize, h: usize, epsilon: f64, ix: usize, iy: usize) -> f64 {
    let idx = iy * w + ix;
    let mut neighbors = 0.0;
    let mut count = 0;

    if ix > 0 {
        neighbors += x[idx - 1];
        count += 1;
    }
    if ix + 1 < w {
        neighbors += x[idx + 1];
        count += 1;
    }
    if iy > 0 {
        neighbors += x[idx - w];
        count += 1;
    }
    if iy + 1 < h {
        neighbors += x[idx + w];
        count += 1;
    }

    (count as f64 + epsilon) * x[idx] - neighbors
}

/// Diagonal element at grid point (ix, iy).
#[inline]
fn diagonal(w: usize, h: usize, epsilon: f64, ix: usize, iy: usize) -> f64 {
    let mut count = 0;
    if ix > 0 { count += 1; }
    if ix + 1 < w { count += 1; }
    if iy > 0 { count += 1; }
    if iy + 1 < h { count += 1; }
    count as f64 + epsilon
}

/// Compute residual: r = b - (A+εI)x.
fn compute_residual(x: &[f64], rhs: &[f64], residual: &mut [f64], w: usize, h: usize, epsilon: f64) {
    for iy in 0..h {
        for ix in 0..w {
            let idx = iy * w + ix;
            residual[idx] = rhs[idx] - apply_operator(x, w, h, epsilon, ix, iy);
        }
    }
}

/// Weighted Jacobi smoother (ω = 2/3).
fn smooth(x: &mut [f64], rhs: &[f64], w: usize, h: usize, epsilon: f64, iters: usize) {
    let omega = 2.0 / 3.0;
    let mut x_new = vec![0.0; w * h];

    for _ in 0..iters {
        for iy in 0..h {
            for ix in 0..w {
                let idx = iy * w + ix;
                let ax = apply_operator(x, w, h, epsilon, ix, iy);
                let diag = diagonal(w, h, epsilon, ix, iy);
                x_new[idx] = x[idx] + omega * (rhs[idx] - ax) / diag;
            }
        }
        x.copy_from_slice(&x_new);
    }
}

/// Full-weighting restriction (fine → coarse).
fn restrict(fine: &[f64], fw: usize, fh: usize, cw: usize, ch: usize) -> Vec<f64> {
    let mut coarse = vec![0.0; cw * ch];

    for cy in 0..ch {
        for cx in 0..cw {
            let fx = cx * 2;
            let fy = cy * 2;

            let mut sum = 0.0;
            let mut weight = 0.0;

            // Center (weight 4)
            if fx < fw && fy < fh {
                sum += 4.0 * fine[fy * fw + fx];
                weight += 4.0;
            }

            // Edge-adjacent (weight 2)
            if fx + 1 < fw && fy < fh {
                sum += 2.0 * fine[fy * fw + fx + 1];
                weight += 2.0;
            }
            if fx > 0 && fy < fh {
                sum += 2.0 * fine[fy * fw + fx - 1];
                weight += 2.0;
            }
            if fx < fw && fy + 1 < fh {
                sum += 2.0 * fine[(fy + 1) * fw + fx];
                weight += 2.0;
            }
            if fx < fw && fy > 0 {
                sum += 2.0 * fine[(fy - 1) * fw + fx];
                weight += 2.0;
            }

            // Corners (weight 1)
            if fx + 1 < fw && fy + 1 < fh {
                sum += fine[(fy + 1) * fw + fx + 1];
                weight += 1.0;
            }
            if fx > 0 && fy + 1 < fh {
                sum += fine[(fy + 1) * fw + fx - 1];
                weight += 1.0;
            }
            if fx + 1 < fw && fy > 0 {
                sum += fine[(fy - 1) * fw + fx + 1];
                weight += 1.0;
            }
            if fx > 0 && fy > 0 {
                sum += fine[(fy - 1) * fw + fx - 1];
                weight += 1.0;
            }

            coarse[cy * cw + cx] = sum / weight;
        }
    }

    coarse
}

/// Bilinear prolongation (coarse → fine).
fn prolongate(coarse: &[f64], cw: usize, ch: usize, fw: usize, fh: usize) -> Vec<f64> {
    let mut fine = vec![0.0; fw * fh];

    for fy in 0..fh {
        for fx in 0..fw {
            let cx_f = fx as f64 / 2.0;
            let cy_f = fy as f64 / 2.0;

            let cx0 = (cx_f.floor() as usize).min(cw.saturating_sub(1));
            let cy0 = (cy_f.floor() as usize).min(ch.saturating_sub(1));
            let cx1 = (cx0 + 1).min(cw - 1);
            let cy1 = (cy0 + 1).min(ch - 1);

            let sx = cx_f - cx0 as f64;
            let sy = cy_f - cy0 as f64;

            fine[fy * fw + fx] =
                (1.0 - sx) * (1.0 - sy) * coarse[cy0 * cw + cx0]
                + sx * (1.0 - sy) * coarse[cy0 * cw + cx1]
                + (1.0 - sx) * sy * coarse[cy1 * cw + cx0]
                + sx * sy * coarse[cy1 * cw + cx1];
        }
    }

    fine
}

/// Direct solve for small grids using many Jacobi iterations.
fn direct_solve(x: &mut [f64], rhs: &[f64], w: usize, h: usize, epsilon: f64) {
    smooth(x, rhs, w, h, epsilon, 100);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_8x8() {
        let w = 8;
        let h = 8;
        let n = w * h;
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let solver = MultigridSolver::new(w, h);
        solver.solve(&rhs, &mut x, 20);

        let res_norm = solver.residual_norm(&rhs, &x);
        assert!(
            res_norm < 1.0,
            "Residual norm {} should be < 1.0",
            res_norm
        );
    }

    #[test]
    fn convergence_improves_with_cycles() {
        let w = 16;
        let h = 16;
        let n = w * h;
        let rhs = vec![1.0; n];

        let solver = MultigridSolver::new(w, h);

        let mut x1 = vec![0.0; n];
        solver.solve(&rhs, &mut x1, 5);
        let res1 = solver.residual_norm(&rhs, &x1);

        let mut x2 = vec![0.0; n];
        solver.solve(&rhs, &mut x2, 20);
        let res2 = solver.residual_norm(&rhs, &x2);

        assert!(
            res2 <= res1 + 1e-10,
            "More V-cycles should not increase residual: {} vs {}",
            res2,
            res1
        );
    }

    #[test]
    fn small_grid_direct() {
        let w = 3;
        let h = 3;
        let n = w * h;
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let solver = MultigridSolver::new(w, h);
        solver.solve(&rhs, &mut x, 5);

        let res_norm = solver.residual_norm(&rhs, &x);
        assert!(res_norm < 1.0, "Residual {} on 3×3 grid", res_norm);
    }

    #[test]
    fn zero_rhs_gives_zero_solution() {
        let w = 8;
        let h = 8;
        let n = w * h;
        let rhs = vec![0.0; n];
        let mut x = vec![0.0; n];

        let solver = MultigridSolver::new(w, h);
        solver.solve(&rhs, &mut x, 5);

        for val in &x {
            assert!(val.abs() < 1e-10, "Expected zero solution, got {}", val);
        }
    }

    #[test]
    fn restrict_then_prolongate_preserves_constant() {
        let fw = 8;
        let fh = 8;
        let cw = 4;
        let ch = 4;
        let fine = vec![5.0; fw * fh];

        let coarse = restrict(&fine, fw, fh, cw, ch);
        let restored = prolongate(&coarse, cw, ch, fw, fh);

        for (i, val) in restored.iter().enumerate() {
            assert!(
                (val - 5.0).abs() < 1e-10,
                "Index {}: expected 5.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn residual_decreases() {
        let w = 8;
        let h = 8;
        let n = w * h;
        let rhs = vec![1.0; n];

        let solver = MultigridSolver::new(w, h);

        let mut x = vec![0.0; n];
        let res0 = solver.residual_norm(&rhs, &x);

        solver.solve(&rhs, &mut x, 10);
        let res1 = solver.residual_norm(&rhs, &x);

        assert!(res1 < res0, "Residual should decrease: {} -> {}", res0, res1);
    }
}

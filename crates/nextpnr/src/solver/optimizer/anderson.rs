//! Anderson acceleration for fixed-point iteration.
//!
//! Type-I Anderson mixing with depth m: stores last m iterates and residuals,
//! solves a least-squares problem each step to find optimal mixing coefficients.

/// Anderson accelerator for fixed-point iteration x = g(x).
///
/// Given current iterate x_k and fixed-point map g(x_k), computes an
/// accelerated next iterate by mixing previous iterates and residuals.
pub struct AndersonAccelerator {
    /// Mixing depth (number of previous iterates to use).
    m: usize,
    /// Problem dimension.
    n: usize,
    /// Ring buffer of previous residuals r_k = g(x_k) - x_k.
    residuals: Vec<Vec<f64>>,
    /// Ring buffer of previous iterates x_k.
    iterates: Vec<Vec<f64>>,
    /// Number of stored history entries.
    count: usize,
    /// Most recent residual norm.
    last_residual_norm: f64,
}

impl AndersonAccelerator {
    /// Create a new Anderson accelerator with mixing depth `m` and problem
    /// dimension `n`.
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            residuals: Vec::with_capacity(m + 1),
            iterates: Vec::with_capacity(m + 1),
            count: 0,
            last_residual_norm: f64::INFINITY,
        }
    }

    /// Perform one Anderson-accelerated step.
    ///
    /// `x` is the current iterate x_k.
    /// `residual` is g(x_k) - x_k (the fixed-point residual).
    /// Returns the accelerated next iterate.
    pub fn step(&mut self, x: &[f64], residual: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(residual.len(), self.n);

        // Compute and store residual norm.
        self.last_residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

        // Store current iterate and residual in ring buffer.
        if self.residuals.len() < self.m + 1 {
            self.residuals.push(residual.to_vec());
            self.iterates.push(x.to_vec());
        } else {
            let idx = self.count % (self.m + 1);
            self.residuals[idx].copy_from_slice(residual);
            self.iterates[idx].copy_from_slice(x);
        }
        self.count += 1;

        let k = self.residuals.len().min(self.count);

        // If we don't have enough history, just do simple fixed-point step.
        if k < 2 {
            let mut result = vec![0.0; self.n];
            for i in 0..self.n {
                result[i] = x[i] + residual[i];
            }
            return result;
        }

        // Solve least-squares for mixing coefficients.
        // We want to minimize ||R_k - sum_i alpha_i * dR_i||^2
        // where dR_i = R_{k} - R_{k-i} for i = 1..m_k.
        let m_k = k - 1; // number of difference vectors

        // Build difference vectors dR_i = R_newest - R_{newest-i}
        let newest = (self.count - 1) % k;
        let r_newest = &self.residuals[newest];

        // Build the Gram matrix G[i][j] = <dR_i, dR_j> and vector b[i] = <dR_i, R_newest>
        let mut dr: Vec<Vec<f64>> = Vec::with_capacity(m_k);
        for i in 1..=m_k {
            let idx = ((self.count as isize - 1 - i as isize).rem_euclid(k as isize)) as usize;
            let r_old = &self.residuals[idx];
            let diff: Vec<f64> = (0..self.n).map(|j| r_newest[j] - r_old[j]).collect();
            dr.push(diff);
        }

        // Solve G * alpha = b via normal equations (small m_k x m_k system).
        let mm = dr.len();
        let mut gram = vec![0.0; mm * mm];
        let mut rhs = vec![0.0; mm];

        for i in 0..mm {
            for j in 0..mm {
                gram[i * mm + j] = dot(&dr[i], &dr[j]);
            }
            rhs[i] = dot(&dr[i], r_newest);
        }

        // Solve via Cholesky-like approach (regularized for stability).
        let alpha = solve_small_ls(&mut gram, &mut rhs, mm);

        // Compute mixed iterate:
        // x_new = (1 - sum_alpha) * g(x_k) + sum_i alpha_i * g(x_{k-i})
        // where g(x) = x + r
        let mut result = vec![0.0; self.n];
        let sum_alpha: f64 = alpha.iter().sum();

        // Start with (1 - sum_alpha) * g(x_newest) = (1 - sum_alpha) * (x + residual)
        let x_newest = &self.iterates[newest];
        for j in 0..self.n {
            result[j] = (1.0 - sum_alpha) * (x_newest[j] + r_newest[j]);
        }

        // Add alpha_i * g(x_{k-i})
        for i in 0..mm {
            let idx =
                ((self.count as isize - 1 - (i + 1) as isize).rem_euclid(k as isize)) as usize;
            let x_old = &self.iterates[idx];
            let r_old = &self.residuals[idx];
            for j in 0..self.n {
                result[j] += alpha[i] * (x_old[j] + r_old[j]);
            }
        }

        result
    }

    /// Most recent residual norm ||g(x) - x||_2.
    pub fn residual_norm(&self) -> f64 {
        self.last_residual_norm
    }
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Solve a small m x m linear system G * alpha = rhs via Gaussian elimination
/// with partial pivoting and Tikhonov regularization for stability.
fn solve_small_ls(gram: &mut [f64], rhs: &mut [f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }

    // Tikhonov regularization: add small epsilon to diagonal.
    let trace: f64 = (0..m).map(|i| gram[i * m + i]).sum();
    let reg = 1e-10 * (trace / m as f64).max(1e-15);
    for i in 0..m {
        gram[i * m + i] += reg;
    }

    // Gaussian elimination with partial pivoting.
    let mut alpha = rhs.to_vec();
    let mut a = gram.to_vec();

    for col in 0..m {
        // Partial pivoting.
        let mut max_val = a[col * m + col].abs();
        let mut max_row = col;
        for row in (col + 1)..m {
            let v = a[row * m + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            alpha[col] = 0.0;
            continue;
        }
        if max_row != col {
            for j in 0..m {
                a.swap(col * m + j, max_row * m + j);
            }
            alpha.swap(col, max_row);
        }

        let pivot = a[col * m + col];
        for row in (col + 1)..m {
            let factor = a[row * m + col] / pivot;
            for j in col..m {
                a[row * m + j] -= factor * a[col * m + j];
            }
            alpha[row] -= factor * alpha[col];
        }
    }

    // Back substitution.
    for col in (0..m).rev() {
        let diag = a[col * m + col];
        if diag.abs() < 1e-15 {
            alpha[col] = 0.0;
            continue;
        }
        for j in (col + 1)..m {
            alpha[col] -= a[col * m + j] * alpha[j];
        }
        alpha[col] /= diag;
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_fixed_point() {
        // x = g(x) where g(x) = 0.5 * x + 1 => fixed point at x = 2.
        let mut aa = AndersonAccelerator::new(3, 1);

        let mut x = vec![0.0];
        for _ in 0..20 {
            let gx = 0.5 * x[0] + 1.0;
            let residual = vec![gx - x[0]];
            x = aa.step(&x, &residual);
        }
        assert!((x[0] - 2.0).abs() < 1e-6, "x = {}", x[0]);
    }

    #[test]
    fn two_dimensional() {
        // g(x, y) = (0.3*x + 0.1*y + 1, 0.1*x + 0.4*y + 2)
        // Fixed point: solve (0.7x - 0.1y = 1, -0.1x + 0.6y = 2)
        // => x ~ 1.56, y ~ 3.59
        let mut aa = AndersonAccelerator::new(3, 2);

        let mut x = vec![0.0, 0.0];
        for _ in 0..30 {
            let gx = vec![0.3 * x[0] + 0.1 * x[1] + 1.0, 0.1 * x[0] + 0.4 * x[1] + 2.0];
            let residual = vec![gx[0] - x[0], gx[1] - x[1]];
            x = aa.step(&x, &residual);
        }

        let expected_x = (1.0 * 0.6 + 0.1 * 2.0) / (0.7 * 0.6 - 0.01);
        let expected_y = (0.7 * 2.0 + 0.1 * 1.0) / (0.7 * 0.6 - 0.01);
        assert!(
            (x[0] - expected_x).abs() < 1e-4,
            "x[0] = {}, expected {}",
            x[0],
            expected_x
        );
        assert!(
            (x[1] - expected_y).abs() < 1e-4,
            "x[1] = {}, expected {}",
            x[1],
            expected_y
        );
    }

    #[test]
    fn residual_norm_decreases() {
        let mut aa = AndersonAccelerator::new(3, 1);
        let mut x = vec![0.0];
        let mut norms = Vec::new();
        for _ in 0..10 {
            let gx = 0.5 * x[0] + 1.0;
            let residual = vec![gx - x[0]];
            x = aa.step(&x, &residual);
            norms.push(aa.residual_norm());
        }
        assert!(norms.last().unwrap() < norms.first().unwrap());
    }
}
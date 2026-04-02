//! L-BFGS quasi-Newton optimizer for analytical placement.
//!
//! Approximates the inverse Hessian from the last `m` gradient/step pairs,
//! giving curvature-adapted step directions that converge faster than
//! gradient descent on ill-conditioned energy landscapes.
//!
//! The two-loop recursion computes H_k * grad without forming H_k explicitly.
//! Memory cost: O(m * n) for m history pairs of dimension n.

use std::collections::VecDeque;

/// L-BFGS optimizer with bounded history.
pub struct LbfgsOptimizer {
    /// History depth (number of correction pairs to keep).
    m: usize,
    /// Position differences: s_i = x_{i+1} - x_i.
    s_history: VecDeque<Vec<f64>>,
    /// Gradient differences: y_i = grad_{i+1} - grad_i.
    y_history: VecDeque<Vec<f64>>,
    /// Cached rho_i = 1 / (y_i^T s_i) for each pair.
    rho_history: VecDeque<f64>,
    /// Previous position (for computing s_i).
    prev_x: Option<Vec<f64>>,
    /// Previous gradient (for computing y_i).
    prev_grad: Option<Vec<f64>>,
}

impl LbfgsOptimizer {
    /// Create a new L-BFGS optimizer with history depth `m`.
    ///
    /// Typical values: m = 5..10. Larger m gives better Hessian approximation
    /// but costs more memory and per-step compute.
    pub fn new(m: usize) -> Self {
        Self {
            m: m.max(1),
            s_history: VecDeque::with_capacity(m),
            y_history: VecDeque::with_capacity(m),
            rho_history: VecDeque::with_capacity(m),
            prev_x: None,
            prev_grad: None,
        }
    }

    /// Reset all history (e.g., when changing multigrid levels).
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.prev_x = None;
        self.prev_grad = None;
    }

    /// Compute L-BFGS descent step from current position and gradient.
    ///
    /// Returns the step vector d such that x_new = x + d moves toward the minimum.
    /// On the first call (no history), returns -grad (steepest descent).
    pub fn step(&mut self, x: &[f64], grad: &[f64]) -> Vec<f64> {
        let n = grad.len();
        debug_assert_eq!(x.len(), n);

        // Update history with the new (s, y) pair.
        if let (Some(prev_x), Some(prev_grad)) = (&self.prev_x, &self.prev_grad) {
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            for i in 0..n {
                s[i] = x[i] - prev_x[i];
                y[i] = grad[i] - prev_grad[i];
            }

            let ys: f64 = y.iter().zip(&s).map(|(yi, si)| yi * si).sum();

            // Skip update if curvature condition fails (ys must be positive
            // for the Hessian approximation to remain positive definite).
            if ys > 1e-16 {
                let rho = 1.0 / ys;

                if self.s_history.len() >= self.m {
                    self.s_history.pop_front();
                    self.y_history.pop_front();
                    self.rho_history.pop_front();
                }
                self.s_history.push_back(s);
                self.y_history.push_back(y);
                self.rho_history.push_back(rho);
            }
        }

        // Save current state for next iteration.
        match &mut self.prev_x {
            Some(px) => px.copy_from_slice(x),
            None => self.prev_x = Some(x.to_vec()),
        }
        match &mut self.prev_grad {
            Some(pg) => pg.copy_from_slice(grad),
            None => self.prev_grad = Some(grad.to_vec()),
        }

        // Two-loop recursion to compute H_k * grad.
        let k = self.s_history.len();

        if k == 0 {
            // No history yet: steepest descent.
            return grad.iter().map(|g| -g).collect();
        }

        // Forward loop: q = grad, then subtract projections.
        let mut q: Vec<f64> = grad.to_vec();
        let mut alpha = vec![0.0; k];

        for i in (0..k).rev() {
            let s = &self.s_history[i];
            let rho = self.rho_history[i];
            let a: f64 = rho * s.iter().zip(&q).map(|(si, qi)| si * qi).sum::<f64>();
            alpha[i] = a;
            let y = &self.y_history[i];
            for j in 0..n {
                q[j] -= a * y[j];
            }
        }

        // Initial Hessian approximation: H0 = gamma * I.
        // gamma = s_k^T y_k / y_k^T y_k (most recent pair).
        let s_last = &self.s_history[k - 1];
        let y_last = &self.y_history[k - 1];
        let yy: f64 = y_last.iter().map(|yi| yi * yi).sum();
        let sy: f64 = s_last.iter().zip(y_last).map(|(si, yi)| si * yi).sum();
        let gamma = if yy > 1e-30 { sy / yy } else { 1.0 };

        // r = H0 * q = gamma * q
        let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

        // Backward loop.
        for i in 0..k {
            let y = &self.y_history[i];
            let rho = self.rho_history[i];
            let beta: f64 = rho * y.iter().zip(&r).map(|(yi, ri)| yi * ri).sum::<f64>();
            let s = &self.s_history[i];
            for j in 0..n {
                r[j] += s[j] * (alpha[i] - beta);
            }
        }

        // Return -r (descent direction).
        for v in &mut r {
            *v = -*v;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steepest_descent_first_step() {
        let mut opt = LbfgsOptimizer::new(5);
        let x = vec![1.0, 2.0];
        let grad = vec![0.5, -0.3];
        let d = opt.step(&x, &grad);
        assert!((d[0] - (-0.5)).abs() < 1e-12);
        assert!((d[1] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn converges_on_quadratic() {
        let mut opt = LbfgsOptimizer::new(5);
        let mut x = vec![10.0, 10.0];

        for _ in 0..50 {
            let grad = vec![x[0], 100.0 * x[1]];
            let d = opt.step(&x, &grad);
            let mut t = 1.0;
            let f0 = 0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]);
            loop {
                let x_new = vec![x[0] + t * d[0], x[1] + t * d[1]];
                let f_new = 0.5 * (x_new[0] * x_new[0] + 100.0 * x_new[1] * x_new[1]);
                if f_new < f0 || t < 1e-15 {
                    x = x_new;
                    break;
                }
                t *= 0.5;
            }
        }

        assert!(x[0].abs() < 0.1, "x[0] = {} not near 0", x[0]);
        assert!(x[1].abs() < 0.01, "x[1] = {} not near 0", x[1]);
    }

    #[test]
    fn history_bounded() {
        let mut opt = LbfgsOptimizer::new(3);
        let mut x = vec![5.0];

        for _ in 0..10 {
            let grad = vec![2.0 * x[0]];
            let d = opt.step(&x, &grad);
            x[0] += 0.1 * d[0];
        }

        assert!(opt.s_history.len() <= 3);
    }

    #[test]
    fn reset_clears_state() {
        let mut opt = LbfgsOptimizer::new(5);
        let x = vec![1.0];
        let grad = vec![2.0];
        let _ = opt.step(&x, &grad);
        assert!(opt.prev_x.is_some());

        opt.reset();
        assert!(opt.prev_x.is_none());
        assert!(opt.s_history.is_empty());
    }
}

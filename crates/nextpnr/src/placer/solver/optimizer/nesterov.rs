//! Nesterov accelerated gradient descent solver.
//!
//! Uses the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) momentum
//! sequence for O(1/k^2) convergence on smooth convex objectives. Includes
//! Lipschitz-based step size estimation and adaptive restart for non-convex
//! regions (ePlace/DREAMPlace/OpenROAD style).

/// Minimum gradient norm denominator to avoid division by zero.
const GRAD_EPSILON: f64 = 1e-30;

/// Nesterov accelerated gradient descent solver with FISTA momentum.
pub struct NesterovSolver {
    x: Vec<f64>,
    x_prev: Vec<f64>,
    step_size: f64,
    iter: usize,
    /// FISTA sequence parameter (starts at 1.0, grows each step).
    a: f64,
}

impl NesterovSolver {
    /// Create a new solver with `n` variables and the given initial step size.
    pub fn new(n: usize, step_size: f64) -> Self {
        Self {
            x: vec![0.0; n],
            x_prev: vec![0.0; n],
            step_size,
            iter: 0,
            a: 1.0,
        }
    }

    /// Perform one Nesterov step given the gradient evaluated at the look-ahead point.
    ///
    /// The caller should:
    /// 1. Compute the look-ahead point via `look_ahead()`
    /// 2. Evaluate the gradient at that point
    /// 3. Call `step()` with that gradient
    ///
    /// Returns the L2 norm of the step (for convergence checking).
    pub fn step(&mut self, grad: &[f64]) -> f64 {
        let n = self.x.len();
        debug_assert_eq!(grad.len(), n);

        self.iter += 1;

        // FISTA momentum sequence
        let a_next = (1.0 + (1.0 + 4.0 * self.a * self.a).sqrt()) / 2.0;
        let momentum = (self.a - 1.0) / a_next;
        self.a = a_next;

        let mut step_norm_sq = 0.0;
        for i in 0..n {
            let x_old = self.x[i];
            let y = self.x[i] + momentum * (self.x[i] - self.x_prev[i]);
            let x_new = y - self.step_size * grad[i];
            self.x_prev[i] = x_old;
            self.x[i] = x_new;
            let delta = x_new - x_old;
            step_norm_sq += delta * delta;
        }

        step_norm_sq.sqrt()
    }

    /// Clamp all internal positions (current and previous) to [lo, hi].
    pub fn clamp_positions_range(&mut self, lo: f64, hi: f64) {
        for i in 0..self.x.len() {
            self.x[i] = self.x[i].clamp(lo, hi);
            self.x_prev[i] = self.x_prev[i].clamp(lo, hi);
        }
    }

    /// Current FISTA momentum coefficient.
    fn momentum(&self) -> f64 {
        if self.iter == 0 {
            0.0
        } else {
            let a_next = (1.0 + (1.0 + 4.0 * self.a * self.a).sqrt()) / 2.0;
            (self.a - 1.0) / a_next
        }
    }

    /// Get the look-ahead point where the gradient should be evaluated.
    ///
    /// Returns a newly allocated vector. For zero-allocation usage, use
    /// `look_ahead_into()`.
    pub fn look_ahead(&self) -> Vec<f64> {
        let mut y = vec![0.0; self.x.len()];
        self.look_ahead_into(&mut y);
        y
    }

    /// Write the look-ahead point into the provided buffer.
    pub fn look_ahead_into(&self, y: &mut [f64]) {
        debug_assert_eq!(y.len(), self.x.len());
        let momentum = self.momentum();
        for i in 0..self.x.len() {
            y[i] = self.x[i] + momentum * (self.x[i] - self.x_prev[i]);
        }
    }

    /// Current positions (solution vector).
    pub fn positions(&self) -> &[f64] {
        &self.x
    }

    /// Mutable access to current positions.
    pub fn positions_mut(&mut self) -> &mut [f64] {
        &mut self.x
    }

    /// Set positions directly (also resets x_prev to the same values).
    pub fn set_positions(&mut self, x: &[f64]) {
        debug_assert_eq!(x.len(), self.x.len());
        self.x.copy_from_slice(x);
        self.x_prev.copy_from_slice(x);
    }

    /// Adaptive restart: reset momentum if gradient opposes the step direction.
    ///
    /// If grad . (x - x_prev) > 0, the momentum is pushing against the gradient,
    /// so we restart by setting x_prev = x and resetting both the iteration counter
    /// and the FISTA parameter `a`.
    pub fn adaptive_restart(&mut self, grad: &[f64]) {
        let n = self.x.len();
        debug_assert_eq!(grad.len(), n);

        let mut dot = 0.0;
        for i in 0..n {
            dot += grad[i] * (self.x[i] - self.x_prev[i]);
        }

        if dot > 0.0 {
            self.x_prev.copy_from_slice(&self.x);
            self.iter = 0;
            self.a = 1.0;
        }
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Set the step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// Current iteration count.
    pub fn iter_count(&self) -> usize {
        self.iter
    }

    /// Number of variables.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the solver has zero variables.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Estimate step size using Lipschitz constant from consecutive gradients.
    ///
    /// Computes step = ||x - x_prev|| / ||grad - prev_grad||, which approximates
    /// the inverse Lipschitz constant of the gradient (1/L). This is the theoretically
    /// optimal step size for FISTA on L-smooth functions.
    pub fn lipschitz_step_size(&self, prev_grad: &[f64], curr_grad: &[f64]) -> f64 {
        let n = self.x.len();
        debug_assert_eq!(prev_grad.len(), n);
        debug_assert_eq!(curr_grad.len(), n);

        let mut dx_norm_sq = 0.0;
        let mut dg_norm_sq = 0.0;
        for i in 0..n {
            let dx = self.x[i] - self.x_prev[i];
            let dg = curr_grad[i] - prev_grad[i];
            dx_norm_sq += dx * dx;
            dg_norm_sq += dg * dg;
        }

        let dg_norm = dg_norm_sq.sqrt();
        if dg_norm > GRAD_EPSILON {
            dx_norm_sq.sqrt() / dg_norm
        } else {
            self.step_size
        }
    }

    /// Estimate step size using the Barzilai-Borwein method.
    ///
    /// Given the previous gradient and current gradient, estimates a good step size:
    /// step = |dx . dx| / |dx . dg| (BB1 formula)
    ///
    /// Returns the estimated step size, or None if the estimate is degenerate.
    pub fn bb_step_size(&self, prev_grad: &[f64], curr_grad: &[f64]) -> Option<f64> {
        let n = self.x.len();
        debug_assert_eq!(prev_grad.len(), n);
        debug_assert_eq!(curr_grad.len(), n);

        let mut dx_dx = 0.0;
        let mut dx_dg = 0.0;
        for i in 0..n {
            let dx = self.x[i] - self.x_prev[i];
            let dg = curr_grad[i] - prev_grad[i];
            dx_dx += dx * dx;
            dx_dg += dx * dg;
        }

        if dx_dg.abs() < GRAD_EPSILON {
            None
        } else {
            Some((dx_dx / dx_dg).abs())
        }
    }

    /// Current FISTA parameter value.
    pub fn fista_param(&self) -> f64 {
        self.a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_minimization() {
        // Minimize f(x) = 0.5 * (x0^2 + x1^2), gradient = (x0, x1)
        // Optimal: (0, 0)
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[5.0, 3.0]);

        for _ in 0..500 {
            let y = solver.look_ahead();
            let grad = vec![y[0], y[1]]; // grad f = (x0, x1)
            solver.step(&grad);
        }

        let pos = solver.positions();
        assert!(pos[0].abs() < 1e-3, "x0 = {}", pos[0]);
        assert!(pos[1].abs() < 1e-3, "x1 = {}", pos[1]);
    }

    #[test]
    fn convergence_faster_than_gd() {
        // Compare Nesterov vs plain GD on f(x) = 0.5 * sum(x_i^2)
        let n = 10;
        let iters = 50;
        let step = 0.05;

        // Nesterov
        let mut nesterov = NesterovSolver::new(n, step);
        let init: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        nesterov.set_positions(&init);

        for _ in 0..iters {
            let y = nesterov.look_ahead();
            nesterov.step(&y); // grad = position for this objective
        }

        let nesterov_dist: f64 = nesterov.positions().iter().map(|x| x * x).sum();

        // Plain GD (use NesterovSolver but reset momentum each step to simulate GD)
        let mut gd = NesterovSolver::new(n, step);
        gd.set_positions(&init);

        for _ in 0..iters {
            let grad: Vec<f64> = gd.positions().to_vec();
            // Manual GD step: x_new = x - step * grad
            let new_pos: Vec<f64> = gd.positions().iter().zip(grad.iter())
                .map(|(x, g)| x - step * g)
                .collect();
            gd.set_positions(&new_pos);
        }

        let gd_dist: f64 = gd.positions().iter().map(|x| x * x).sum();

        assert!(
            nesterov_dist < gd_dist,
            "Nesterov ({}) should converge faster than GD ({})",
            nesterov_dist,
            gd_dist
        );
    }

    #[test]
    fn adaptive_restart_resets_momentum() {
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[1.0, 1.0]);
        // Do a step to build momentum
        solver.step(&[0.1, 0.1]);
        assert!(solver.iter_count() > 0);
        assert!(solver.fista_param() > 1.0);

        // Gradient opposing the direction -> should restart
        // restart happens when grad . (x - x_prev) > 0
        let disp = [
            solver.positions()[0] - solver.x_prev[0],
            solver.positions()[1] - solver.x_prev[1],
        ];
        // Use same direction as displacement so dot > 0
        solver.adaptive_restart(&disp);
        assert_eq!(solver.iter_count(), 0);
        assert!((solver.fista_param() - 1.0).abs() < 1e-10, "FISTA param should reset to 1.0");
    }

    #[test]
    fn rosenbrock_converges() {
        // Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        // Optimal: (1, 1)
        let mut solver = NesterovSolver::new(2, 1e-4);
        solver.set_positions(&[-1.0, 1.0]);

        for _ in 0..50_000 {
            let y = solver.look_ahead();
            let x0 = y[0];
            let x1 = y[1];
            // grad f = (-2(1-x) - 400x(y-x^2), 200(y-x^2))
            let grad = vec![
                -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
                200.0 * (x1 - x0 * x0),
            ];
            solver.adaptive_restart(&grad);
            solver.step(&grad);
        }

        let pos = solver.positions();
        assert!(
            (pos[0] - 1.0).abs() < 0.2 && (pos[1] - 1.0).abs() < 0.2,
            "Rosenbrock: got ({}, {}), expected near (1, 1)",
            pos[0],
            pos[1]
        );
    }

    #[test]
    fn bb_step_size_estimate() {
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[2.0, 3.0]);

        let prev_grad = vec![2.0, 3.0];
        solver.step(&prev_grad);
        let curr_grad: Vec<f64> = solver.positions().to_vec();

        let bb = solver.bb_step_size(&prev_grad, &curr_grad);
        assert!(bb.is_some());
        assert!(bb.unwrap() > 0.0);
    }

    #[test]
    fn lipschitz_step_size_estimate() {
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[2.0, 3.0]);

        let prev_grad = vec![2.0, 3.0];
        solver.step(&prev_grad);
        let curr_grad: Vec<f64> = solver.positions().to_vec();

        let lip = solver.lipschitz_step_size(&prev_grad, &curr_grad);
        assert!(lip > 0.0, "Lipschitz step size should be positive, got {}", lip);
    }

    #[test]
    fn lipschitz_step_size_degenerate() {
        // When gradients are identical, should fall back to current step_size
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[1.0, 2.0]);
        solver.step(&[0.5, 0.5]);

        let grad = vec![1.0, 1.0];
        let lip = solver.lipschitz_step_size(&grad, &grad);
        assert!((lip - 0.1).abs() < 1e-10, "Should return current step_size for identical grads");
    }

    #[test]
    fn fista_momentum_grows() {
        // FISTA momentum should start at 0 and grow over iterations
        let mut solver = NesterovSolver::new(2, 0.1);
        solver.set_positions(&[1.0, 1.0]);

        let mut prev_a = solver.fista_param();
        assert!((prev_a - 1.0).abs() < 1e-10);

        for _ in 0..10 {
            solver.step(&[0.1, 0.1]);
            let curr_a = solver.fista_param();
            assert!(curr_a > prev_a, "FISTA param should grow: {} -> {}", prev_a, curr_a);
            prev_a = curr_a;
        }
    }

    #[test]
    fn zero_variables() {
        let solver = NesterovSolver::new(0, 0.1);
        assert!(solver.is_empty());
        assert_eq!(solver.len(), 0);
    }
}

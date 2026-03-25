//! Adam optimizer for non-smooth placement objectives.
//!
//! Maintains per-variable running averages of gradient (momentum) and
//! squared gradient (adaptive step scale). Well-suited for non-stationary
//! objectives where the gradient landscape changes each iteration
//! (e.g., Kirchhoff pressure fields that depend on cell positions).
//!
//! Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2014.

/// Adam optimizer with per-variable adaptive step sizes.
pub struct AdamSolver {
    x: Vec<f64>,
    m: Vec<f64>,  // First moment (gradient EMA)
    v: Vec<f64>,  // Second moment (squared gradient EMA)
    beta1: f64,
    beta2: f64,
    alpha: f64,   // Base learning rate
    eps: f64,
    t: usize,     // Timestep (for bias correction)
}

impl AdamSolver {
    /// Create a new Adam solver with `n` variables.
    ///
    /// Default hyperparameters from the paper: beta1=0.9, beta2=0.999, eps=1e-8.
    pub fn new(n: usize, alpha: f64) -> Self {
        Self {
            x: vec![0.0; n],
            m: vec![0.0; n],
            v: vec![0.0; n],
            beta1: 0.9,
            beta2: 0.999,
            alpha,
            eps: 1e-8,
            t: 0,
        }
    }

    /// Set beta1 (gradient momentum decay). Default 0.9.
    pub fn set_beta1(&mut self, beta1: f64) {
        self.beta1 = beta1;
    }

    /// Set beta2 (squared gradient decay). Default 0.999.
    pub fn set_beta2(&mut self, beta2: f64) {
        self.beta2 = beta2;
    }

    /// Set the base learning rate.
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Perform one Adam step given the gradient.
    ///
    /// Returns the L2 norm of the position change (for convergence checking).
    pub fn step(&mut self, grad: &[f64]) -> f64 {
        let n = self.x.len();
        debug_assert_eq!(grad.len(), n);

        self.t += 1;

        // Bias correction factors.
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut step_norm_sq = 0.0;
        for i in 0..n {
            // Update moments.
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            // Bias-corrected estimates.
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Update position.
            let delta = self.alpha * m_hat / (v_hat.sqrt() + self.eps);
            self.x[i] -= delta;
            step_norm_sq += delta * delta;
        }

        step_norm_sq.sqrt()
    }

    /// Reset moments (m, v) without changing positions.
    /// Use when the objective function changes significantly (e.g., density penalty ramp).
    pub fn reset_moments(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }

    /// Current positions.
    pub fn positions(&self) -> &[f64] {
        &self.x
    }

    /// Set positions directly (also resets moments).
    pub fn set_positions(&mut self, x: &[f64]) {
        debug_assert_eq!(x.len(), self.x.len());
        self.x.copy_from_slice(x);
        self.reset_moments();
    }

    /// Clamp all positions to [lo, hi].
    pub fn clamp_positions_range(&mut self, lo: f64, hi: f64) {
        for x in &mut self.x {
            *x = x.clamp(lo, hi);
        }
    }

    /// Current base learning rate.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Current timestep.
    pub fn timestep(&self) -> usize {
        self.t
    }

    /// Number of variables.
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_minimization() {
        // Minimize f(x) = 0.5 * (x0^2 + x1^2), gradient = (x0, x1)
        let mut solver = AdamSolver::new(2, 0.1);
        solver.set_positions(&[5.0, 3.0]);

        for _ in 0..500 {
            let grad = vec![solver.positions()[0], solver.positions()[1]];
            solver.step(&grad);
        }

        let pos = solver.positions();
        assert!(pos[0].abs() < 0.1, "x0 = {}", pos[0]);
        assert!(pos[1].abs() < 0.1, "x1 = {}", pos[1]);
    }

    #[test]
    fn non_stationary_convergence() {
        // Objective that changes: f(x) = 0.5 * (x - target)^2
        // Target shifts every 50 iterations.
        let mut solver = AdamSolver::new(1, 0.05);
        solver.set_positions(&[0.0]);

        let targets = [3.0, -2.0, 1.0, 4.0];
        for (phase, &target) in targets.iter().enumerate() {
            for _ in 0..100 {
                let grad = vec![solver.positions()[0] - target];
                solver.step(&grad);
            }
            let err = (solver.positions()[0] - target).abs();
            assert!(err < 1.0, "Phase {}: pos={}, target={}, err={}", phase, solver.positions()[0], target, err);
        }
    }
}

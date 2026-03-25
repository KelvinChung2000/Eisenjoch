//! Damped velocity field solver for physics-based placement optimization.
//!
//! Unlike Adam (which applies per-coordinate adaptive rescaling that distorts
//! gradient geometry), this solver preserves the direction and relative magnitude
//! of the combined force field. Two channels:
//!
//! - Main step: `v = α·v - η·g`, `x += v`  (flow + density + anchor, after viscosity)
//! - Helmholtz step: `v -= η_h·∇φ`, `x -= η_h·∇φ`  (de-clustering, bypasses viscosity)
//!
//! The Helmholtz force has its own step size η_h, truly decoupled from the
//! flow optimization. This is operator splitting: transport and density correction
//! compose as independent velocity contributions.

pub struct VelocityFieldSolver {
    x: Vec<f64>,
    v: Vec<f64>,
    eta: f64,
    eta_h: f64,
    alpha: f64,
}

impl VelocityFieldSolver {
    pub fn new(n: usize, eta: f64) -> Self {
        Self {
            x: vec![0.0; n],
            v: vec![0.0; n],
            eta,
            eta_h: eta * 0.3,
            alpha: 0.85,
        }
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn set_eta_helmholtz(&mut self, eta_h: f64) {
        self.eta_h = eta_h;
    }

    pub fn set_positions(&mut self, x: &[f64]) {
        debug_assert_eq!(x.len(), self.x.len());
        self.x.copy_from_slice(x);
        self.v.fill(0.0);
    }

    pub fn positions(&self) -> &[f64] {
        &self.x
    }

    /// Main gradient step: damped velocity update.
    /// v = α·v - η·grad, x += v
    pub fn step(&mut self, grad: &[f64]) {
        debug_assert_eq!(grad.len(), self.x.len());
        for i in 0..self.x.len() {
            self.v[i] = self.alpha * self.v[i] - self.eta * grad[i];
            self.x[i] += self.v[i];
        }
    }

    /// Decoupled Helmholtz de-clustering step.
    /// Applies -η_h·∇φ as a velocity impulse only — position updates via the
    /// next main step's `x += v`. This is consistent operator splitting:
    /// Helmholtz acts as a force (modifying velocity), not a direct displacement.
    pub fn step_helmholtz(&mut self, helmholtz_grad: &[f64]) {
        debug_assert_eq!(helmholtz_grad.len(), self.x.len());
        for i in 0..self.x.len() {
            self.v[i] -= self.eta_h * helmholtz_grad[i];
        }
    }

    /// Clamp positions to [lo, hi]. Zero velocity at boundaries to prevent
    /// momentum from pushing cells into walls repeatedly.
    pub fn clamp_positions_range(&mut self, lo: f64, hi: f64) {
        for i in 0..self.x.len() {
            if self.x[i] < lo {
                self.x[i] = lo;
                self.v[i] = self.v[i].max(0.0); // only allow outward velocity
            } else if self.x[i] > hi {
                self.x[i] = hi;
                self.v[i] = self.v[i].min(0.0);
            }
        }
    }

    /// Scale velocity by factor (for periodic cooling without full reset).
    pub fn dampen(&mut self, factor: f64) {
        for v in &mut self.v {
            *v *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_to_minimum() {
        // Minimize f(x) = x^2, gradient = 2x, minimum at x=0.
        let mut solver = VelocityFieldSolver::new(1, 0.1);
        solver.set_positions(&[5.0]);
        for _ in 0..100 {
            let grad = vec![2.0 * solver.positions()[0]];
            solver.step(&grad);
        }
        assert!(solver.positions()[0].abs() < 0.1);
    }

    #[test]
    fn helmholtz_decoupled() {
        let mut solver = VelocityFieldSolver::new(2, 0.1);
        solver.set_positions(&[0.0, 0.0]);
        // Only Helmholtz step, no main gradient.
        solver.step_helmholtz(&[1.0, -1.0]);
        let pos = solver.positions();
        // Cell 0 should move in -∇φ direction (negative), cell 1 positive.
        assert!(pos[0] < 0.0);
        assert!(pos[1] > 0.0);
    }

    #[test]
    fn boundary_reflection() {
        let mut solver = VelocityFieldSolver::new(1, 1.0);
        solver.set_positions(&[0.5]);
        // Large step pushes past boundary.
        solver.step(&[-10.0]); // grad=-10 → v += 10 → x goes to 10.5
        solver.clamp_positions_range(0.0, 5.0);
        assert_eq!(solver.positions()[0], 5.0);
        // Velocity should be clamped to non-positive (at upper boundary).
        solver.step(&[0.0]); // v = α * v_clamped + 0
        assert!(solver.positions()[0] <= 5.0);
    }
}

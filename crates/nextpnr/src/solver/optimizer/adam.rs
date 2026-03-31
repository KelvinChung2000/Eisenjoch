//! Adam optimizer.

/// Adam optimizer state for a flat parameter vector.
pub struct AdamOptimizer {
    m: Vec<f64>,
    v: Vec<f64>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    lr: f64,
    t: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer with default beta/epsilon hyperparameters.
    pub fn new(n_params: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr,
            t: 0,
        }
    }

    /// Compute one Adam update from gradient and write step into `out`.
    pub fn step(&mut self, grad: &[f64], out: &mut [f64]) {
        debug_assert_eq!(grad.len(), self.m.len());
        debug_assert_eq!(out.len(), self.m.len());

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let one_minus_beta1 = 1.0 - self.beta1;
        let one_minus_beta2 = 1.0 - self.beta2;

        for i in 0..grad.len() {
            let g = grad[i];
            self.m[i] = self.beta1.mul_add(self.m[i], one_minus_beta1 * g);
            self.v[i] = self.beta2.mul_add(self.v[i], one_minus_beta2 * g * g);
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            out[i] = -self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_opposite_gradient() {
        let mut adam = AdamOptimizer::new(3, 0.1);
        let grad = [1.0, -2.0, 0.5];
        let mut out = [0.0; 3];

        adam.step(&grad, &mut out);

        assert!(out[0] < 0.0);
        assert!(out[1] > 0.0);
        assert!(out[2] < 0.0);
    }

    #[test]
    fn zero_gradient_produces_zero_step() {
        let mut adam = AdamOptimizer::new(4, 0.1);
        let grad = [0.0; 4];
        let mut out = [1.0; 4];

        adam.step(&grad, &mut out);

        assert_eq!(out, [0.0; 4]);
    }
}

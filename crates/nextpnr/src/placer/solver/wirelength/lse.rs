//! Log-Sum-Exp smooth differentiable HPWL approximation.
//!
//! Provides a smooth approximation to half-perimeter wirelength (HPWL) using
//! the log-sum-exp function. As γ → 0, LSE approaches the exact HPWL.
//!
//! Formula per axis:
//!   W = γ × [log Σ exp(x_i/γ) + log Σ exp(-x_i/γ)]
//!
//! Numerically stable implementation subtracts the max before exponentiating.

/// Compute the LSE wirelength approximation for a set of pin positions.
///
/// Each position is (x, y). The total wirelength is the sum of the x-axis
/// and y-axis LSE approximations.
///
/// `gamma` controls smoothness: larger γ = smoother but less accurate,
/// smaller γ = tighter but harder to optimize.
pub fn lse_wirelength(positions: &[(f64, f64)], gamma: f64) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }

    lse_axis(positions.iter().map(|p| p.0), gamma)
        + lse_axis(positions.iter().map(|p| p.1), gamma)
}

/// LSE approximation for a single axis.
///
/// W = γ × [log Σ exp(x_i/γ) + log Σ exp(-x_i/γ)]
fn lse_axis(coords: impl Iterator<Item = f64> + Clone, gamma: f64) -> f64 {
    let inv_gamma = 1.0 / gamma;

    let max_val = coords.clone().fold(f64::NEG_INFINITY, f64::max);
    let min_val = coords.clone().fold(f64::INFINITY, f64::min);

    if max_val == f64::NEG_INFINITY {
        return 0.0;
    }

    let sum_exp_pos: f64 = coords
        .clone()
        .map(|x| ((x - max_val) * inv_gamma).exp())
        .sum();

    let sum_exp_neg: f64 = coords
        .map(|x| ((-x + min_val) * inv_gamma).exp())
        .sum();

    gamma * (max_val * inv_gamma + sum_exp_pos.ln() - min_val * inv_gamma + sum_exp_neg.ln())
}

/// Compute the gradient of LSE wirelength w.r.t. each pin position.
///
/// For each pin i on axis x:
///   ∂W/∂x_i = exp(x_i/γ) / Σ exp(x_j/γ) - exp(-x_i/γ) / Σ exp(-x_j/γ)
///
/// Gradients are accumulated (added to) the `grad` slice.
pub fn lse_gradient(positions: &[(f64, f64)], gamma: f64, grad: &mut [(f64, f64)]) {
    let n = positions.len();
    debug_assert_eq!(grad.len(), n);

    if n < 2 {
        return;
    }

    lse_axis_gradient(
        positions.iter().map(|p| p.0),
        gamma,
        &mut grad.iter_mut().map(|g| &mut g.0).collect::<Vec<_>>(),
    );

    lse_axis_gradient(
        positions.iter().map(|p| p.1),
        gamma,
        &mut grad.iter_mut().map(|g| &mut g.1).collect::<Vec<_>>(),
    );
}

/// Core gradient computation for a single axis.
///
/// Returns softmax weights: (exp_pos[i] / sum_pos - exp_neg[i] / sum_neg) for each i.
fn lse_axis_weights(coords: &[f64], gamma: f64) -> Vec<f64> {
    let inv_gamma = 1.0 / gamma;

    let max_val = coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = coords.iter().cloned().fold(f64::INFINITY, f64::min);

    let exp_pos: Vec<f64> = coords.iter().map(|&x| ((x - max_val) * inv_gamma).exp()).collect();
    let exp_neg: Vec<f64> = coords.iter().map(|&x| ((-x + min_val) * inv_gamma).exp()).collect();

    let inv_sum_pos = 1.0 / exp_pos.iter().sum::<f64>();
    let inv_sum_neg = 1.0 / exp_neg.iter().sum::<f64>();

    (0..coords.len())
        .map(|i| exp_pos[i] * inv_sum_pos - exp_neg[i] * inv_sum_neg)
        .collect()
}

/// Compute gradient for a single axis using (x,y) pair indirection.
fn lse_axis_gradient(coords: impl Iterator<Item = f64>, gamma: f64, grad_out: &mut [&mut f64]) {
    let xs: Vec<f64> = coords.collect();
    let weights = lse_axis_weights(&xs, gamma);
    for (i, w) in weights.iter().enumerate() {
        *grad_out[i] += w;
    }
}

/// Compute LSE wirelength for a single net on one axis given coordinate values.
///
/// Useful when you have coordinates in separate arrays rather than as (x,y) pairs.
pub fn lse_axis_value(coords: &[f64], gamma: f64) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }
    lse_axis(coords.iter().copied(), gamma)
}

/// Compute LSE gradient for a single net on one axis.
///
/// Accumulates gradients into `grad_out`.
pub fn lse_axis_grad(coords: &[f64], gamma: f64, grad_out: &mut [f64]) {
    if coords.len() < 2 {
        return;
    }

    let weights = lse_axis_weights(coords, gamma);
    for (i, w) in weights.iter().enumerate() {
        grad_out[i] += w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_points_lse_approximates_hpwl() {
        // HPWL of two points at (0,0) and (3,4) is |3| + |4| = 7
        let positions = vec![(0.0, 0.0), (3.0, 4.0)];

        // With small gamma, should be close to 7
        let wl = lse_wirelength(&positions, 0.1);
        assert!((wl - 7.0).abs() < 0.2, "LSE = {}, expected ~7.0", wl);

        // With larger gamma, overestimates
        let wl_smooth = lse_wirelength(&positions, 5.0);
        assert!(wl_smooth >= 7.0 - 0.01, "LSE should be >= HPWL: {}", wl_smooth);
    }

    #[test]
    fn gamma_to_zero_approaches_hpwl() {
        let positions = vec![(0.0, 0.0), (5.0, 0.0), (2.0, 3.0)];
        // HPWL = (5 - 0) + (3 - 0) = 8

        let wl_large = lse_wirelength(&positions, 10.0);
        let wl_medium = lse_wirelength(&positions, 1.0);
        let wl_small = lse_wirelength(&positions, 0.01);

        // As gamma shrinks, should approach 8
        assert!((wl_small - 8.0).abs() < (wl_medium - 8.0).abs());
        assert!((wl_medium - 8.0).abs() < (wl_large - 8.0).abs());
    }

    #[test]
    fn gradient_finite_differences() {
        let positions = vec![(1.0, 2.0), (4.0, 1.0), (2.0, 5.0)];
        let gamma = 2.0;
        let eps = 1e-6;

        let mut grad = vec![(0.0, 0.0); 3];
        lse_gradient(&positions, gamma, &mut grad);

        // Check each component with finite differences
        for i in 0..3 {
            for axis in 0..2 {
                let mut pos_plus = positions.clone();
                let mut pos_minus = positions.clone();

                if axis == 0 {
                    pos_plus[i].0 += eps;
                    pos_minus[i].0 -= eps;
                } else {
                    pos_plus[i].1 += eps;
                    pos_minus[i].1 -= eps;
                }

                let fd = (lse_wirelength(&pos_plus, gamma) - lse_wirelength(&pos_minus, gamma))
                    / (2.0 * eps);
                let analytic = if axis == 0 { grad[i].0 } else { grad[i].1 };

                assert!(
                    (fd - analytic).abs() < 1e-4,
                    "Gradient mismatch at pin {} axis {}: fd={}, analytic={}",
                    i,
                    axis,
                    fd,
                    analytic
                );
            }
        }
    }

    #[test]
    fn single_point_is_zero() {
        assert_eq!(lse_wirelength(&[(1.0, 2.0)], 1.0), 0.0);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(lse_wirelength(&[], 1.0), 0.0);
    }

    #[test]
    fn collinear_points() {
        // Points along x-axis: HPWL = max - min = 10
        let positions = vec![(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)];
        let wl = lse_wirelength(&positions, 0.1);
        assert!((wl - 10.0).abs() < 0.3, "LSE = {}, expected ~10.0", wl);
    }

    #[test]
    fn axis_value_and_grad() {
        let coords = vec![1.0, 4.0, 2.0];
        let gamma = 2.0;

        let val = lse_axis_value(&coords, gamma);
        assert!(val > 0.0);

        let mut grad = vec![0.0; 3];
        lse_axis_grad(&coords, gamma, &mut grad);

        // Finite difference check
        let eps = 1e-6;
        for i in 0..3 {
            let mut c_plus = coords.clone();
            let mut c_minus = coords.clone();
            c_plus[i] += eps;
            c_minus[i] -= eps;
            let fd = (lse_axis_value(&c_plus, gamma) - lse_axis_value(&c_minus, gamma)) / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() < 1e-4,
                "Axis grad mismatch at {}: fd={}, analytic={}",
                i,
                fd,
                grad[i]
            );
        }
    }
}

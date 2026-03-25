//! Weighted-Average (WA) smooth differentiable HPWL approximation.
//!
//! Used by the ElectroPlace analytical placer (aligned with placer_static.cc).
//! Unlike LSE, WA uses a fixed `wl_coeff` parameter (no gamma annealing).
//!
//! Formula per axis:
//!   WA = (x_max_weighted / sum_max_exp) - (x_min_weighted / sum_min_exp)
//!
//! Where per-pin exponential weights:
//!   max_exp_i = exp(wl_coeff * (x_i - center))
//!   min_exp_i = exp(wl_coeff * (center - x_i))

/// Minimum exponent argument to prevent underflow.
const EXP_CLAMP_MIN: f64 = -3000.0;

/// Compute WA wirelength for a single axis.
pub fn wa_axis_value(coords: &[f64], wl_coeff: f64) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let center = coords.iter().sum::<f64>() / coords.len() as f64;

    let mut sum_max_exp = 0.0;
    let mut x_max_weighted = 0.0;
    let mut sum_min_exp = 0.0;
    let mut x_min_weighted = 0.0;

    for &x in coords {
        let max_arg = (wl_coeff * (x - center)).max(EXP_CLAMP_MIN);
        let max_exp = max_arg.exp();
        sum_max_exp += max_exp;
        x_max_weighted += x * max_exp;

        let min_arg = (wl_coeff * (center - x)).max(EXP_CLAMP_MIN);
        let min_exp = min_arg.exp();
        sum_min_exp += min_exp;
        x_min_weighted += x * min_exp;
    }

    if sum_max_exp < 1e-30 || sum_min_exp < 1e-30 {
        return 0.0;
    }

    x_max_weighted / sum_max_exp - x_min_weighted / sum_min_exp
}

/// Compute WA gradient for a single axis, accumulating into `grad_out`.
///
/// Uses the quotient rule on WA = (x_max_weighted / sum_max) - (x_min_weighted / sum_min).
pub fn wa_axis_grad(coords: &[f64], wl_coeff: f64, grad_out: &mut [f64]) {
    let n = coords.len();
    if n < 2 {
        return;
    }
    debug_assert_eq!(grad_out.len(), n);

    let center = coords.iter().sum::<f64>() / n as f64;

    let mut max_exps = vec![0.0; n];
    let mut min_exps = vec![0.0; n];
    let mut sum_max = 0.0;
    let mut sum_min = 0.0;
    let mut x_max_sum = 0.0;
    let mut x_min_sum = 0.0;

    for (i, &x) in coords.iter().enumerate() {
        let max_arg = (wl_coeff * (x - center)).max(EXP_CLAMP_MIN);
        max_exps[i] = max_arg.exp();
        sum_max += max_exps[i];
        x_max_sum += x * max_exps[i];

        let min_arg = (wl_coeff * (center - x)).max(EXP_CLAMP_MIN);
        min_exps[i] = min_arg.exp();
        sum_min += min_exps[i];
        x_min_sum += x * min_exps[i];
    }

    if sum_max < 1e-30 || sum_min < 1e-30 {
        return;
    }

    let inv_sum_max = 1.0 / sum_max;
    let inv_sum_min = 1.0 / sum_min;
    let inv_sum_max2 = inv_sum_max * inv_sum_max;
    let inv_sum_min2 = inv_sum_min * inv_sum_min;

    for (i, &x) in coords.iter().enumerate() {
        // d(max_term)/d(x_i) via quotient rule:
        // numerator: max_exp_i * sum_max + wl_coeff * max_exp_i * (x_i * sum_max - x_max_sum)
        // denom: sum_max^2
        let d_max = max_exps[i] * inv_sum_max
            + wl_coeff * max_exps[i] * (x * sum_max - x_max_sum) * inv_sum_max2;

        // d(min_term)/d(x_i) via quotient rule (note the negative wl_coeff for min):
        let d_min = min_exps[i] * inv_sum_min
            - wl_coeff * min_exps[i] * (x * sum_min - x_min_sum) * inv_sum_min2;

        grad_out[i] += d_max - d_min;
    }
}

/// Compute 2D WA wirelength for a set of pin positions.
pub fn wa_wirelength(positions: &[(f64, f64)], wl_coeff: f64) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }

    let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();

    wa_axis_value(&xs, wl_coeff) + wa_axis_value(&ys, wl_coeff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_points_approximates_hpwl() {
        let positions = vec![(0.0, 0.0), (3.0, 4.0)];
        let wl = wa_wirelength(&positions, 0.5);
        // HPWL = |3| + |4| = 7. WA underestimates for small wl_coeff.
        assert!(
            wl > 0.0 && wl < 8.0,
            "WA = {}, expected positive and reasonable",
            wl
        );

        // With higher wl_coeff, should be closer to HPWL.
        let wl_sharp = wa_wirelength(&positions, 5.0);
        assert!(
            (wl_sharp - 7.0).abs() < 1.0,
            "WA(coeff=5) = {}, expected ~7.0",
            wl_sharp
        );
    }

    #[test]
    fn single_point_is_zero() {
        assert_eq!(wa_wirelength(&[(1.0, 2.0)], 0.5), 0.0);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(wa_wirelength(&[], 0.5), 0.0);
    }

    #[test]
    fn gradient_finite_differences() {
        let coords = vec![1.0, 4.0, 2.0, 6.0];
        let wl_coeff = 0.5;
        let eps = 1e-6;

        let mut grad = vec![0.0; 4];
        wa_axis_grad(&coords, wl_coeff, &mut grad);

        for i in 0..coords.len() {
            let mut c_plus = coords.clone();
            let mut c_minus = coords.clone();
            c_plus[i] += eps;
            c_minus[i] -= eps;
            let fd = (wa_axis_value(&c_plus, wl_coeff) - wa_axis_value(&c_minus, wl_coeff))
                / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() < 1e-4,
                "Gradient mismatch at pin {}: fd={}, analytic={}",
                i,
                fd,
                grad[i]
            );
        }
    }

    #[test]
    fn gradient_2d_finite_differences() {
        let positions = vec![(1.0, 2.0), (4.0, 1.0), (2.0, 5.0)];
        let wl_coeff = 0.5;
        let eps = 1e-6;

        let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();
        let mut grad_x = vec![0.0; 3];
        let mut grad_y = vec![0.0; 3];
        wa_axis_grad(&xs, wl_coeff, &mut grad_x);
        wa_axis_grad(&ys, wl_coeff, &mut grad_y);

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

                let fd = (wa_wirelength(&pos_plus, wl_coeff)
                    - wa_wirelength(&pos_minus, wl_coeff))
                    / (2.0 * eps);
                let analytic = if axis == 0 { grad_x[i] } else { grad_y[i] };

                assert!(
                    (fd - analytic).abs() < 1e-4,
                    "2D gradient mismatch at pin {} axis {}: fd={}, analytic={}",
                    i,
                    axis,
                    fd,
                    analytic
                );
            }
        }
    }

    #[test]
    fn higher_wl_coeff_is_sharper() {
        let positions = vec![(0.0, 0.0), (5.0, 0.0), (2.0, 3.0)];
        // HPWL = 5 + 3 = 8
        let wl_low = wa_wirelength(&positions, 0.1);
        let wl_high = wa_wirelength(&positions, 2.0);
        // Higher coeff should be closer to true HPWL
        assert!(
            (wl_high - 8.0).abs() <= (wl_low - 8.0).abs() + 0.5,
            "Higher coeff ({}) should be closer to HPWL (8) than lower ({})",
            wl_high,
            wl_low
        );
    }
}

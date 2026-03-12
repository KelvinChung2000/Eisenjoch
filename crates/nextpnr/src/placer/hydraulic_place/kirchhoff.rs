//! Kirchhoff resistive network solver for congestion-driven placement.
//!
//! Solves L * P = d where L is a conductance-weighted Laplacian and d is
//! net demand injected at junctions. Flow Q = (P_from - P_to) / R.
//!
//! Turbulence: R_eff = R_base * (1 + beta * (Q/C)^2). Re-solved via Newton
//! iteration with warm-started CG.
//!
//! The resistive energy E = sum(d[j] * P[j]) = sum(R * Q^2) is the unified
//! objective: it captures both wirelength (pressure drop) and congestion
//! (turbulent resistance).

use super::network::{Pipe, PipeNetwork};
use crate::placer::solver::cg::conjugate_gradient;

pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
    /// Resistive energy E = d^T * P = sum(R * Q^2).
    pub energy: f64,
}

/// Effective resistance with turbulence: R_eff = R_base * (1 + beta * (Q/C)^2).
///
/// On the first Newton iteration (before any flow is computed), uses base resistance only.
fn effective_resistance(pipe: &Pipe, turbulence_beta: f64, use_turbulence: bool) -> f64 {
    let r_base = pipe.resistance.max(1e-12);
    if !use_turbulence {
        return r_base;
    }
    let util = if pipe.capacity > 0.0 {
        pipe.flow.abs() / pipe.capacity
    } else {
        0.0
    };
    r_base * (1.0 + turbulence_beta * util * util)
}

/// Solve the Kirchhoff system on the pipe network.
///
/// 1. Build conductance-weighted Laplacian from pipe resistances
/// 2. Pin junction 0 as pressure reference
/// 3. CG solve L * P = demand (warm-started from previous pressure)
/// 4. Newton loop: compute flows, update R_eff with turbulence, re-solve
/// 5. Store pressure in junctions, flow in pipes
/// 6. Return energy = sum(demand[j] * P[j])
pub fn kirchhoff_solve(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    cg_max_iters: usize,
    cg_tol: f64,
) -> SolveResult {
    let n_j = network.num_junctions();
    let n_p = network.num_pipes();

    if n_j == 0 || n_p == 0 {
        return SolveResult {
            converged: true,
            iterations: 0,
            energy: 0.0,
        };
    }

    let mut pressure: Vec<f64> = network.junctions.iter().map(|j| j.pressure).collect();
    let mut total_iters = 0;

    // Newton iterations for nonlinear resistance.
    // Iteration 0 uses base resistance; subsequent iterations update R_eff from flow.
    let num_solves = newton_iters.max(1);
    for newton_iter in 0..num_solves {
        let use_turbulence = newton_iter > 0;

        // Build Laplacian from pipe conductances.
        let mut diag = vec![0.0; n_j];
        let mut off_diag: Vec<(usize, usize, f64)> = Vec::with_capacity(n_p);

        for pipe in &network.pipes {
            let conductance = 1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence);
            let i = pipe.from;
            let j = pipe.to;
            diag[i] += conductance;
            diag[j] += conductance;
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            off_diag.push((lo, hi, -conductance));
        }

        // Pin junction 0 as pressure reference (P[0] = 0).
        diag[0] = 1e10;
        let mut rhs = demand.to_vec();
        rhs[0] = 0.0;

        let iters = conjugate_gradient(&diag, &off_diag, &rhs, &mut pressure, cg_tol, cg_max_iters);
        total_iters += iters;

        // Compute flows: Q = (P[from] - P[to]) / R_eff.
        for pipe in &mut network.pipes {
            let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence);
            pipe.flow = (pressure[pipe.from] - pressure[pipe.to]) / r_eff;
        }
    }

    // Store pressure in junctions.
    for (j, p) in pressure.iter().enumerate() {
        network.junctions[j].pressure = *p;
    }
    network.junctions[0].pressure = 0.0;

    let energy: f64 = demand.iter().zip(pressure.iter()).map(|(d, p)| d * p).sum();

    SolveResult {
        converged: true,
        iterations: total_iters,
        energy,
    }
}

/// Transit time: tau = 1 + beta * max(0, |Q|/C - 1)^2.
pub fn transit_time(flow: f64, capacity: f64, beta: f64) -> f64 {
    if capacity <= 0.0 {
        return 1.0;
    }
    let excess = (flow.abs() / capacity - 1.0).max(0.0);
    1.0 + beta * excess * excess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transit_time_below_capacity() {
        let tau = transit_time(5.0, 10.0, 1.0);
        assert!((tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transit_time_above_capacity() {
        let tau = transit_time(15.0, 10.0, 1.0);
        assert!(tau > 1.0);
    }
}

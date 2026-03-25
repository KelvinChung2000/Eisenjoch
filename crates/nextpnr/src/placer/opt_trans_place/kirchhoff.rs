//! Kirchhoff solver for minimum-energy electrical flow on the pipe network.
//!
//! Solves the Kirchhoff system LP = S where L is the conductance-weighted
//! Laplacian, P is junction potential, and S is net demand.
//! Newton iterations update nonlinear resistance from flow-dependent turbulence.
//!
//! Turbulence: R_eff = R_base * (1 + beta * tanh((Q/C)^2)).
//! Pump energy: E = Σ |P_driver - P_sinks| per net.

use super::config::KirchhoffSolver;
use super::network::{Pipe, PipeNetwork};
use crate::placer::solver::FaerDirectSolver;

pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
}

/// Solver resources passed through to the Kirchhoff solve dispatch.
/// Created once before the main loop, reused every iteration.
pub struct SolverCtx<'a> {
    pub solver_type: KirchhoffSolver,
    pub cg_max_iters: usize,
    pub cg_tol: f64,
    pub direct: Option<&'a mut FaerDirectSolver>,
    /// Placeholder for removed AMG preconditioner. Kept for struct compatibility.
    pub amg: Option<&'a mut ()>,
}

// ---- Internal helpers ----

/// Density penalty for a pipe: max overflow of its two endpoint tiles.
#[inline]
fn pipe_density_penalty(pipe: &Pipe, density_overflow: &[f64]) -> f64 {
    if density_overflow.is_empty() {
        return 0.0;
    }
    let of = density_overflow.get(pipe.from / 4).copied().unwrap_or(0.0);
    let ot = density_overflow.get(pipe.to / 4).copied().unwrap_or(0.0);
    of.max(ot)
}

/// Effective resistance with Beckmann aggregate flow coupling and density feedback.
fn effective_resistance(
    pipe: &Pipe,
    turbulence_beta: f64,
    use_turbulence: bool,
    density_penalty: f64,
    agg_flow_scale: f64,
) -> f64 {
    let r_base = pipe.resistance.max(1e-12);
    let mut r_eff = r_base;

    if use_turbulence {
        if pipe.capacity > 0.0 {
            let raw_util = pipe.aggregate_flow / pipe.capacity;
            let util = raw_util / agg_flow_scale;
            let penalty = (util * util).min(100.0);
            r_eff *= 1.0 + turbulence_beta * penalty;
        }
        if pipe.capacity > 0.0 {
            let raw_inst = pipe.flow.abs() / pipe.capacity;
            let inst_util = raw_inst / agg_flow_scale;
            let inst_penalty = (inst_util * inst_util).min(100.0);
            r_eff *= 1.0 + 0.5 * turbulence_beta * inst_penalty;
        }
    }

    r_eff *= 1.0 + density_penalty.min(10.0);
    r_eff
}

/// Regularize the Laplacian and pin junction 0 as pressure reference.
///
/// Adds ε·I to prevent floating potentials from turbulence, then pins
/// junction 0 to ground with a large diagonal entry.
fn regularize_and_pin(diag: &mut [f64], rhs: &mut [f64]) {
    let n = diag.len();
    let avg_diag = diag.iter().sum::<f64>() / n as f64;
    let epsilon = 0.01 * avg_diag.max(1e-6);
    for d in diag.iter_mut() {
        *d += epsilon;
    }
    diag[0] += 1e10;
    rhs[0] = 0.0;
}

/// Dispatch to the configured solver. Unified for both full and cropped solves.
fn dispatch_solve(
    ctx: &mut SolverCtx<'_>,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    pressure: &mut [f64],
) -> usize {
    match ctx.solver_type {
        KirchhoffSolver::Direct => {
            if let Some(ref mut ds) = ctx.direct {
                ds.solve(diag, off_diag, rhs, pressure);
                1
            } else {
                crate::placer::solver::faer_cg(
                    diag, off_diag, rhs, pressure, ctx.cg_tol, ctx.cg_max_iters,
                )
            }
        }
        KirchhoffSolver::CG => {
            crate::placer::solver::faer_cg(
                diag, off_diag, rhs, pressure, ctx.cg_tol, ctx.cg_max_iters,
            )
        }
    }
}

// ---- Public API ----

/// Solve the Kirchhoff system on the pipe network.
pub fn kirchhoff_solve(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    density_overflow: &[f64],
    solver: &mut SolverCtx<'_>,
) -> SolveResult {
    let n_j = network.num_junctions();
    let n_p = network.num_pipes();

    if n_j == 0 || n_p == 0 {
        return SolveResult { converged: true, iterations: 0 };
    }

    let mut pressure: Vec<f64> = network.junctions.iter().map(|j| j.pressure).collect();
    let mut total_iters = 0;
    let agg_scale = network.agg_flow_scale;

    let num_solves = newton_iters.max(1);
    for newton_iter in 0..num_solves {
        let has_flow_data = newton_iter > 0
            || network.pipes.iter().any(|p| p.flow.abs() > 1e-15);
        let use_turbulence = turbulence_beta > 1e-6 && has_flow_data;

        // Build Laplacian from pipe conductances.
        let mut diag = vec![0.0; n_j];
        let mut off_diag: Vec<(usize, usize, f64)> = Vec::with_capacity(n_p);

        for pipe in &network.pipes {
            let dp = pipe_density_penalty(pipe, density_overflow);
            let conductance = 1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence, dp, agg_scale);
            let i = pipe.from;
            let j = pipe.to;
            diag[i] += conductance;
            diag[j] += conductance;
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            off_diag.push((lo, hi, -conductance));
        }

        let mut rhs = demand.to_vec();
        regularize_and_pin(&mut diag, &mut rhs);

        total_iters += dispatch_solve(
            solver, &diag, &off_diag, &rhs, &mut pressure,
        );

        // Compute flows: Q = (P[from] - P[to]) / R_eff.
        for pipe in &mut network.pipes {
            let dp = pipe_density_penalty(pipe, density_overflow);
            let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence, dp, agg_scale);
            pipe.flow = (pressure[pipe.from] - pressure[pipe.to]) / r_eff;
        }
    }

    for (j, &p) in pressure.iter().enumerate() {
        network.junctions[j].pressure = p;
    }
    network.junctions[0].pressure = 0.0;

    SolveResult { converged: true, iterations: total_iters }
}

/// Transit time: tau = 1 + beta * max(0, |Q|/C - 1)^2.
pub fn transit_time(flow: f64, capacity: f64, beta: f64) -> f64 {
    if capacity <= 0.0 {
        return 1.0;
    }
    let excess = (flow.abs() / capacity - 1.0).max(0.0);
    1.0 + beta * excess * excess
}

/// A cropped region of the grid for faster Kirchhoff solves.
pub struct CroppedRegion {
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    pub junction_map: Vec<Option<usize>>,
    pub cropped_to_full: Vec<usize>,
}

impl CroppedRegion {
    pub fn from_cells(cell_x: &[f64], cell_y: &[f64], width: i32, height: i32) -> Self {
        let n_cells = cell_x.len();
        if n_cells == 0 {
            return Self::from_bounds(0, width - 1, 0, height - 1, width, height);
        }

        let margin = 5i32.max((n_cells as f64).sqrt().ceil() as i32);

        let mut x_min = i32::MAX;
        let mut x_max = i32::MIN;
        let mut y_min = i32::MAX;
        let mut y_max = i32::MIN;

        for i in 0..n_cells {
            let tx = cell_x[i].round() as i32;
            let ty = cell_y[i].round() as i32;
            x_min = x_min.min(tx);
            x_max = x_max.max(tx);
            y_min = y_min.min(ty);
            y_max = y_max.max(ty);
        }

        Self::from_bounds(x_min - margin, x_max + margin, y_min - margin, y_max + margin, width, height)
    }

    pub fn from_bounds(x_min: i32, x_max: i32, y_min: i32, y_max: i32, width: i32, height: i32) -> Self {
        let x_min = x_min.max(0);
        let x_max = x_max.min(width - 1);
        let y_min = y_min.max(0);
        let y_max = y_max.min(height - 1);

        let n_full = (width * height * 4) as usize;
        let mut junction_map = vec![None; n_full];
        let mut cropped_to_full = Vec::new();

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                for port in 0..4 {
                    let full_idx = ((y * width + x) * 4 + port) as usize;
                    junction_map[full_idx] = Some(cropped_to_full.len());
                    cropped_to_full.push(full_idx);
                }
            }
        }

        Self { x_min, x_max, y_min, y_max, junction_map, cropped_to_full }
    }

    pub fn num_junctions(&self) -> usize { self.cropped_to_full.len() }
    pub fn grid_width(&self) -> usize { (self.x_max - self.x_min + 1) as usize }
    pub fn grid_height(&self) -> usize { (self.y_max - self.y_min + 1) as usize }
}

/// Solve the Kirchhoff system on a cropped subregion of the pipe network.
pub fn kirchhoff_solve_cropped(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    region: &CroppedRegion,
    density_overflow: &[f64],
    solver: &mut SolverCtx<'_>,
) -> SolveResult {
    let n_cropped = region.num_junctions();

    if n_cropped == 0 {
        return SolveResult { converged: true, iterations: 0 };
    }

    let mut cropped_demand = vec![0.0; n_cropped];
    let mut pressure = vec![0.0; n_cropped];
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        cropped_demand[ci] = demand[fi];
        pressure[ci] = network.junctions[fi].pressure;
    }

    let mut total_iters = 0;
    let num_solves = newton_iters.max(1);
    let agg_scale = network.agg_flow_scale;

    for newton_iter in 0..num_solves {
        let has_flow_data = newton_iter > 0
            || network.pipes.iter().any(|p| p.flow.abs() > 1e-15);
        let use_turbulence = turbulence_beta > 1e-6 && has_flow_data;

        let mut diag = vec![0.0; n_cropped];
        let mut off_diag: Vec<(usize, usize, f64)> = Vec::with_capacity(network.num_pipes());

        for pipe in &network.pipes {
            let ci_from = region.junction_map[pipe.from];
            let ci_to = region.junction_map[pipe.to];

            if let (Some(cf), Some(ct)) = (ci_from, ci_to) {
                let dp = pipe_density_penalty(pipe, density_overflow);
                let conductance =
                    1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence, dp, agg_scale);
                diag[cf] += conductance;
                diag[ct] += conductance;
                let (lo, hi) = if cf < ct { (cf, ct) } else { (ct, cf) };
                off_diag.push((lo, hi, -conductance));
            }
        }

        regularize_and_pin(&mut diag, &mut cropped_demand);

        total_iters += dispatch_solve(
            solver, &diag, &off_diag, &cropped_demand, &mut pressure,
        );

        for pipe in &mut network.pipes {
            let p_from = region.junction_map[pipe.from]
                .map(|ci| pressure[ci]).unwrap_or(0.0);
            let p_to = region.junction_map[pipe.to]
                .map(|ci| pressure[ci]).unwrap_or(0.0);
            let dp = pipe_density_penalty(pipe, density_overflow);
            let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence, dp, agg_scale);
            pipe.flow = (p_from - p_to) / r_eff;
        }
    }

    for j in &mut network.junctions {
        j.pressure = 0.0;
    }
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        network.junctions[fi].pressure = pressure[ci];
    }
    if let Some(&fi) = region.cropped_to_full.first() {
        network.junctions[fi].pressure = 0.0;
    }

    SolveResult { converged: true, iterations: total_iters }
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

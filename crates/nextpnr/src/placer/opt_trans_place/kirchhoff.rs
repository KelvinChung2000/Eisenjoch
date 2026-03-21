//! Kirchhoff solver for minimum-energy electrical flow on the pipe network.
//!
//! Solves the Kirchhoff system LP = S where L is the conductance-weighted
//! Laplacian, P is junction potential, and S is net demand.
//! Newton iterations update nonlinear resistance from flow-dependent turbulence.
//!
//! Turbulence: R_eff = R_base * (1 + beta * tanh((Q/C)^2)).
//! Pump energy: E = Σ |P_driver - P_sinks| per net.

use super::network::{Pipe, PipeNetwork};
use crate::placer::solver::cg::multigrid_preconditioned_cg;

pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
}

/// Density penalty for a pipe: max overflow of its two endpoint tiles.
///
/// Returns 0.0 when density_overflow is empty (iter 0 or disabled).
/// Junction index / 4 = tile index (4 ports per tile).
#[inline]
fn pipe_density_penalty(pipe: &Pipe, density_overflow: &[f64]) -> f64 {
    if density_overflow.is_empty() {
        return 0.0;
    }
    let of = density_overflow.get(pipe.from / 4).copied().unwrap_or(0.0);
    let ot = density_overflow.get(pipe.to / 4).copied().unwrap_or(0.0);
    of.max(ot)
}

/// Effective resistance with congestion and density feedback.
///
/// R_eff = R_base * (1 + beta * min((Q/C)^2, 100)) * (1 + density_penalty)
///
/// Quadratic flow-based growth with cap at 100× to prevent numerical instability.
/// Density penalty increases resistance on pipes adjacent to overcrowded tiles,
/// steering the Kirchhoff solver around congested regions (Benamou-Brenier coupling).
fn effective_resistance(pipe: &Pipe, turbulence_beta: f64, use_turbulence: bool, density_penalty: f64) -> f64 {
    let r_base = pipe.resistance.max(1e-12);
    let mut r_eff = r_base;
    if use_turbulence {
        let util = if pipe.capacity > 0.0 {
            pipe.flow.abs() / pipe.capacity
        } else {
            0.0
        };
        let penalty = (util * util).min(100.0);
        r_eff *= 1.0 + turbulence_beta * penalty;
    }
    // Density coupling: overcrowded tiles make adjacent pipes more expensive.
    r_eff *= 1.0 + density_penalty.min(10.0);
    r_eff
}

/// Solve the Kirchhoff system on the pipe network.
///
/// 1. Build conductance-weighted Laplacian from pipe resistances
/// 2. Pin junction 0 as pressure reference
/// 3. CG solve L * P = demand (warm-started from previous pressure)
/// 4. Newton loop: compute flows, update R_eff with turbulence, re-solve
/// 5. Store pressure in junctions, flow in pipes
/// 6. Return solve result (energy computed separately via pump-cost model)
pub fn kirchhoff_solve(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    cg_max_iters: usize,
    cg_tol: f64,
    density_overflow: &[f64],
) -> SolveResult {
    let n_j = network.num_junctions();
    let n_p = network.num_pipes();

    if n_j == 0 || n_p == 0 {
        return SolveResult {
            converged: true,
            iterations: 0,
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
            let dp = pipe_density_penalty(pipe, density_overflow);
            let conductance = 1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence, dp);
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

        let iters = multigrid_preconditioned_cg(
            &diag,
            &off_diag,
            &rhs,
            &mut pressure,
            cg_tol,
            cg_max_iters,
            network.width as usize,
            network.height as usize,
        );
        total_iters += iters;

        // Compute flows: Q = (P[from] - P[to]) / R_eff.
        for pipe in &mut network.pipes {
            let dp = pipe_density_penalty(pipe, density_overflow);
            let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence, dp);
            pipe.flow = (pressure[pipe.from] - pressure[pipe.to]) / r_eff;
        }
    }

    // Store pressure in junctions.
    for (j, &p) in pressure.iter().enumerate() {
        network.junctions[j].pressure = p;
    }
    network.junctions[0].pressure = 0.0;

    SolveResult {
        converged: true,
        iterations: total_iters,
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

/// A cropped region of the grid for faster Kirchhoff solves.
///
/// Only junctions within the bounding box (expanded by a margin) are included.
/// Junctions outside get P=0 (Dirichlet boundary).
pub struct CroppedRegion {
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    /// Maps full junction index -> cropped index (None if outside).
    pub junction_map: Vec<Option<usize>>,
    /// Inverse map: cropped index -> full junction index.
    pub cropped_to_full: Vec<usize>,
}

impl CroppedRegion {
    /// Compute a cropped region from cell positions.
    ///
    /// The region is the bounding box of all cells, expanded by a margin.
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

    /// Create a cropped region from explicit bounds.
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

    /// Number of junctions in the cropped region.
    pub fn num_junctions(&self) -> usize {
        self.cropped_to_full.len()
    }

    /// Width of the cropped grid in tiles.
    pub fn grid_width(&self) -> usize {
        (self.x_max - self.x_min + 1) as usize
    }

    /// Height of the cropped grid in tiles.
    pub fn grid_height(&self) -> usize {
        (self.y_max - self.y_min + 1) as usize
    }
}

/// Solve the Kirchhoff system on a cropped subregion of the pipe network.
///
/// Only pipes with BOTH endpoints inside the region are included.
/// Junctions outside get P=0 (Dirichlet boundary).
pub fn kirchhoff_solve_cropped(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    cg_max_iters: usize,
    cg_tol: f64,
    region: &CroppedRegion,
    density_overflow: &[f64],
) -> SolveResult {
    let n_cropped = region.num_junctions();

    if n_cropped == 0 {
        return SolveResult {
            converged: true,
            iterations: 0,
        };
    }

    // Map demand and initial pressure to cropped indices.
    let mut cropped_demand = vec![0.0; n_cropped];
    let mut pressure = vec![0.0; n_cropped];
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        cropped_demand[ci] = demand[fi];
        pressure[ci] = network.junctions[fi].pressure;
    }

    let mut total_iters = 0;
    let num_solves = newton_iters.max(1);

    for newton_iter in 0..num_solves {
        let use_turbulence = newton_iter > 0;

        let mut diag = vec![0.0; n_cropped];
        let mut off_diag: Vec<(usize, usize, f64)> = Vec::with_capacity(network.num_pipes());

        for pipe in &network.pipes {
            let ci_from = region.junction_map[pipe.from];
            let ci_to = region.junction_map[pipe.to];

            // Only include pipes with BOTH endpoints in the cropped region.
            if let (Some(cf), Some(ct)) = (ci_from, ci_to) {
                let dp = pipe_density_penalty(pipe, density_overflow);
                let conductance =
                    1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence, dp);
                diag[cf] += conductance;
                diag[ct] += conductance;
                let (lo, hi) = if cf < ct { (cf, ct) } else { (ct, cf) };
                off_diag.push((lo, hi, -conductance));
            }
        }

        // Pin junction 0 (in cropped space) as reference.
        diag[0] = 1e10;
        cropped_demand[0] = 0.0;

        let iters = multigrid_preconditioned_cg(
            &diag,
            &off_diag,
            &cropped_demand,
            &mut pressure,
            cg_tol,
            cg_max_iters,
            region.grid_width(),
            region.grid_height(),
        );
        total_iters += iters;

        // Compute flows for ALL pipes (not just cropped).
        for pipe in &mut network.pipes {
            let p_from = region
                .junction_map[pipe.from]
                .map(|ci| pressure[ci])
                .unwrap_or(0.0);
            let p_to = region
                .junction_map[pipe.to]
                .map(|ci| pressure[ci])
                .unwrap_or(0.0);
            let dp = pipe_density_penalty(pipe, density_overflow);
            let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence, dp);
            pipe.flow = (p_from - p_to) / r_eff;
        }
    }

    // Write pressures back: cropped region gets solved values, outside gets 0.
    for j in &mut network.junctions {
        j.pressure = 0.0;
    }
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        network.junctions[fi].pressure = pressure[ci];
    }
    // Pin reference junction.
    if let Some(&fi) = region.cropped_to_full.first() {
        network.junctions[fi].pressure = 0.0;
    }

    SolveResult {
        converged: true,
        iterations: total_iters,
    }
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

//! Configuration for the Hydraulic placer (Kirchhoff resistive network model).

#[derive(Clone)]
pub struct HydraulicPlacerCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Maximum turbulence coefficient (default: 4.0).
    /// Controls nonlinear pipe resistance: R_eff = R_base * (1 + beta * (Q/C)^2).
    /// Ramped from 0 to this value over the first half of iterations.
    pub turbulence_beta: f64,
    /// Newton iterations per pressure solve for nonlinear resistance updates (default: 2).
    pub newton_iters: usize,
    /// Maximum CG solver iterations per pressure solve (default: 500).
    pub cg_max_iters: usize,
    /// CG convergence tolerance (default: 1e-6).
    pub cg_tolerance: f64,
    /// Initial Nesterov step size (default: 0.1).
    pub nesterov_step_size: f64,
    /// Maximum outer loop iterations (default: 500).
    pub max_outer_iters: usize,
    /// Legalize every N outer iterations (default: 5).
    pub legalize_interval: usize,
    /// Transit time weight in combined objective (default: 0.0).
    pub timing_weight: f64,
}

impl Default for HydraulicPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            turbulence_beta: 4.0,
            newton_iters: 2,
            cg_max_iters: 500,
            cg_tolerance: 1e-6,
            nesterov_step_size: 0.1,
            max_outer_iters: 500,
            legalize_interval: 5,
            timing_weight: 0.0,
        }
    }
}

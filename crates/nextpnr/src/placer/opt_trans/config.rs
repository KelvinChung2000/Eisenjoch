//! Configuration for the Beckmann optimal transport placer.

use crate::netlist::NetId;
use rustc_hash::FxHashMap;

/// Preconditioner for the CG solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerType {
    /// Diagonal (Jacobi) preconditioner. Simple, always works.
    Jacobi,
    /// Algebraic Multigrid. Fast convergence on regular grids (~40 iters vs ~200 Jacobi).
    Amg,
}

impl Default for PreconditionerType {
    fn default() -> Self {
        Self::Jacobi
    }
}

/// Initialization strategy for cell positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitStrategy {
    /// Random BEL assignment (default).
    RandomBel,
    /// Centroid of connected fixed cells.
    Centroid,
    /// Uniform grid distribution.
    Uniform,
}

impl Default for InitStrategy {
    fn default() -> Self {
        Self::RandomBel
    }
}

/// Configuration for the Beckmann optimal transport placer.
#[derive(Debug, Clone)]
pub struct OptTransPlacerCfg {
    /// Random seed.
    pub seed: u64,
    /// Maximum outer iterations.
    pub max_iters: usize,
    /// CG maximum iterations per Kirchhoff solve.
    pub cg_max_iters: usize,
    /// CG relative tolerance.
    pub cg_tol: f64,
    /// Number of right-hand sides solved together in per-net Kirchhoff solves.
    /// 1 preserves original single-net solve behavior.
    pub cg_batch_size: usize,
    /// Beckmann congestion exponent: alpha in (|J|/C)^alpha.
    pub congestion_exponent: f64,
    /// Weight for flow interference spreading.
    pub interference_weight: f64,
    /// Weight for timing-critical path resistance.
    pub timing_weight: f64,
    /// Optional per-net timing criticality map (0.0..1.0).
    pub timing_criticality: FxHashMap<NetId, f32>,
    /// IO net demand amplification factor.
    pub io_boost: f64,
    /// Exponent used to normalize sink demand by fanout.
    pub fanout_norm_exp: f64,
    /// Apply sqrt(fanout) scaling to IO demand for large nets.
    pub fanout_weight_sqrt: bool,
    /// Anderson mixing depth m.
    pub anderson_depth: usize,
    /// Report every N iterations.
    pub report_interval: usize,
    /// Maximum cells for legalization.
    pub lap_max_cells: usize,
    /// Initialization strategy.
    pub init_strategy: InitStrategy,
    /// Subtile resolution: each tile is decomposed into N×N subtile nodes.
    pub subtile_resolution: usize,
    /// Preconditioner for the Kirchhoff CG solve.
    pub preconditioner: PreconditionerType,
    /// Number of Rayon worker threads for per-net solves.
    pub num_threads: usize,

    /// Legalization strategy: "ring", "sorted", "bipartite", "greedy".
    pub legalization: String,

    // --- Iteration control ---
    /// Use Anderson acceleration (true) or direct gradient step (false).
    pub use_anderson: bool,
    /// Step scale multiplier for direct gradient step mode.
    pub step_scale: f64,
    /// Stagnation step reduction factor (multiply step_limit by this).
    pub step_decay: f64,
    /// Don't check stagnation before this iteration.
    pub stagnation_warmup: usize,
    /// Rollback after this many consecutive non-improving iterations.
    pub stagnation_patience: usize,
}

impl Default for OptTransPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            max_iters: 500,
            cg_max_iters: 500,
            cg_tol: 1e-2,
            cg_batch_size: 4,
            congestion_exponent: 0.0,
            interference_weight: 0.0,
            timing_weight: 0.0,
            timing_criticality: FxHashMap::default(),
            io_boost: 1.0,
            fanout_norm_exp: 1.0,
            fanout_weight_sqrt: false,
            anderson_depth: 3,
            report_interval: 5,
            lap_max_cells: 10000,
            init_strategy: InitStrategy::RandomBel,
            subtile_resolution: 1,
            preconditioner: PreconditionerType::Amg,
            num_threads: 8,
            legalization: "ring".to_string(),
            // Iteration control.
            use_anderson: false,
            step_scale: 0.5,
            step_decay: 0.7,
            stagnation_warmup: 20,
            stagnation_patience: 5,
        }
    }
}

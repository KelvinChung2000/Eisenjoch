//! Configuration for the Beckmann optimal transport placer.

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

/// Per-cell displacement normalization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellNormalization {
    /// Keep raw per-cell magnitudes (no normalization).
    Raw,
    /// Normalize each cell's displacement to unit vector.
    Unit,
    /// RMS-normalize each component independently before superposition (default).
    Rms,
}

impl Default for CellNormalization {
    fn default() -> Self {
        Self::Rms
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
    /// Beckmann congestion exponent: alpha in (|J|/C)^alpha.
    pub congestion_exponent: f64,
    /// Weight for flow interference spreading.
    pub interference_weight: f64,
    /// Weight for timing-critical path resistance.
    pub timing_weight: f64,
    /// IO net demand amplification factor.
    pub io_boost: f64,
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
    /// Weight for congestion repulsion force.
    pub congestion_repulsion_weight: f64,

    // --- Displacement model ---

    /// Weight for -grad(P) flow field component.
    pub grad_weight: f64,
    /// Weight for dp/dist pin attraction component.
    pub attraction_weight: f64,
    /// Per-cell normalization strategy.
    pub cell_normalization: CellNormalization,

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
            max_iters: 150,
            cg_max_iters: 200,
            cg_tol: 1e-4,
            congestion_exponent: 2.0,
            interference_weight: 0.0,
            timing_weight: 0.0,
            io_boost: 1.0,
            anderson_depth: 3,
            report_interval: 5,
            lap_max_cells: 10000,
            init_strategy: InitStrategy::default(),
            subtile_resolution: 2,
            preconditioner: PreconditionerType::Amg,
            congestion_repulsion_weight: 0.0,
            // Displacement model: equal weight superposition with RMS normalization.
            grad_weight: 1.0,
            attraction_weight: 1.0,
            cell_normalization: CellNormalization::Rms,
            // Iteration control.
            use_anderson: true,
            step_scale: 5.0,
            step_decay: 0.7,
            stagnation_warmup: 20,
            stagnation_patience: 5,
        }
    }
}

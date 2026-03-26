//! Configuration for the Beckmann optimal transport placer.

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
    /// Maximum Anderson fixed-point iterations.
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
}

impl Default for OptTransPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            max_iters: 30,
            cg_max_iters: 200,
            cg_tol: 1e-4,
            congestion_exponent: 2.0,
            interference_weight: 1.0,
            timing_weight: 0.0,
            io_boost: 3.0,
            anderson_depth: 3,
            report_interval: 5,
            lap_max_cells: 10000,
            init_strategy: InitStrategy::default(),
        }
    }
}

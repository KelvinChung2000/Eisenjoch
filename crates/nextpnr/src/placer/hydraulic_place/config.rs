//! Configuration for the Hydraulic placer (Kirchhoff resistive network model).

/// How to initialize cell positions before the optimization loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStrategy {
    /// Uniform grid: cells evenly distributed across the chip.
    Uniform,
    /// Center: all cells start at the grid center.
    Centroid,
    /// Random BEL: read positions from the random BEL assignment by initial_placement.
    RandomBel,
}

#[derive(Clone)]
pub struct HydraulicPlacerCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Maximum turbulence coefficient (default: 4.0).
    /// Controls nonlinear pipe resistance: R_eff = R_base * (1 + beta * tanh((Q/C)^2)).
    /// Exponentially ramped from 0 toward this value over the iteration schedule.
    pub turbulence_beta: f64,
    /// Newton iterations per pressure solve for nonlinear resistance updates (default: 2).
    pub newton_iters: usize,
    /// Maximum CG solver iterations per pressure solve (default: 500).
    pub cg_max_iters: usize,
    /// CG convergence tolerance (default: 1e-6).
    pub cg_tolerance: f64,
    /// CFL number: max cell displacement per iteration in tiles (default: 0.5).
    /// From the Courant-Friedrichs-Lewy stability condition.
    pub cfl_number: f64,
    /// Maximum outer loop iterations (default: 500).
    pub max_outer_iters: usize,
    /// Report metrics (with diagnostic legalization) every N outer iterations (default: 5).
    pub report_interval: usize,
    /// Timing viscosity: critical cells get up to (1 + timing_weight) × drag (default: 0.0).
    /// Higher values make critical nets resist spreading, keeping timing paths short.
    /// Also controls transit-time-based setup/hold violation checking.
    pub timing_weight: f64,
    /// Initial gas temperature for density spreading (default: 0.0).
    /// Starts hot (strong spreading to break clustering), anneals to 0.
    /// Higher = more initial spreading. Set to 0.0 to disable.
    pub gas_temperature: f64,
    /// Initial cell placement strategy (default: Centroid).
    pub init_strategy: InitStrategy,
    /// Maximum cells per type group for bipartite legalization (default: 10000).
    pub lap_max_cells: usize,

    // === Force blend weights (all independently configurable) ===

    /// Weight for WA star model force (default: 1.0). Set 0.0 to disable.
    pub star_weight: f64,

    /// Weight for gas hydraulic pressure gradient (default: 0.0 at start).
    /// Ramps from `pressure_weight_start` to `pressure_weight_end` over iterations.
    pub pressure_weight_start: f64,

    /// Final weight for pressure gradient (default: 2.0).
    pub pressure_weight_end: f64,

    /// IO demand boost factor for nets with fixed pins (default: 4.0).
    pub io_boost: f64,

    // === Optimizer ===

    /// Nesterov initial step size (default: 0.1).
    pub nesterov_step_size: f64,

    /// Nesterov momentum coefficient override. None = use FISTA automatic (default: None).
    pub momentum: Option<f64>,

    /// Legalize every N iterations for convergence tracking (default: 5).
    pub legalize_interval: usize,

    /// WA wirelength smoothing coefficient (default: 0.5).
    pub wl_coeff: f64,

    /// Enable expanding bounding box (default: true). Set false to disable.
    pub enable_expanding_box: bool,

    /// Pump gain for dynamic demand amplification of timing-violating nets (default: 10.0).
    /// Nets with high criticality get demand scaled by (1 + pump_gain * crit^2).
    pub pump_gain: f64,
}

impl Default for HydraulicPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            turbulence_beta: 4.0,
            newton_iters: 2,
            cg_max_iters: 500,
            cg_tolerance: 1e-6,
            cfl_number: 0.5,
            max_outer_iters: 500,
            report_interval: 5,
            timing_weight: 0.0,
            gas_temperature: 1.0,
            init_strategy: InitStrategy::Centroid,
            lap_max_cells: 10000,
            star_weight: 1.0,
            pressure_weight_start: 0.0,
            pressure_weight_end: 2.0,
            io_boost: 4.0,
            nesterov_step_size: 0.1,
            momentum: None,
            legalize_interval: 5,
            wl_coeff: 0.5,
            enable_expanding_box: true,
            pump_gain: 10.0,
        }
    }
}

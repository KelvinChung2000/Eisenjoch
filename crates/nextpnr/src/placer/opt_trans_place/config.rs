//! Configuration for the optimal transport placer (Kirchhoff resistive network model).

/// How to initialize cell positions before the optimization loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStrategy {
    /// Uniform grid: cells evenly distributed across the chip.
    Uniform,
    /// Center: all cells start at the grid center.
    Centroid,
    /// Random BEL: read positions from the random BEL assignment by initial_placement.
    RandomBel,
    /// Radial capacity: spread from IO centroid, filling tiles up to BEL capacity.
    RadialCapacity,
}

#[derive(Clone)]
pub struct OptTransPlacerCfg {
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
    /// Unused. Kept for Python API backward compatibility.
    pub cfl_number: f64,
    /// Maximum outer loop iterations (default: 500).
    pub max_outer_iters: usize,
    /// Report metrics (with diagnostic legalization) every N outer iterations (default: 5).
    pub report_interval: usize,
    /// Timing viscosity: critical cells get up to (1 + timing_weight) × drag (default: 0.0).
    /// Higher values make critical nets resist spreading, keeping timing paths short.
    /// Also controls transit-time-based setup/hold violation checking.
    pub timing_weight: f64,
    /// Unused. Replaced by per-tile AL multipliers. Kept for Python API backward compatibility.
    pub gas_temperature: f64,
    /// Initial cell placement strategy (default: Centroid).
    pub init_strategy: InitStrategy,
    /// Maximum cells per type group for bipartite legalization (default: 10000).
    pub lap_max_cells: usize,

    // === Unused force blend weights (kept for Python API backward compatibility) ===

    /// Unused. Kept for Python API backward compatibility.
    pub star_weight: f64,

    /// Unused. Kept for Python API backward compatibility.
    pub pressure_weight_start: f64,

    /// Unused. Kept for Python API backward compatibility.
    pub pressure_weight_end: f64,

    /// IO demand boost factor for nets with fixed pins (default: 4.0).
    pub io_boost: f64,

    // === Optimizer ===

    /// Nesterov initial step size (default: 0.1).
    pub nesterov_step_size: f64,

    /// Unused. Kept for Python API backward compatibility.
    pub momentum: Option<f64>,

    /// Unused. Kept for Python API backward compatibility.
    pub wl_coeff: f64,

    /// Enable expanding bounding box (default: true). Set false to disable.
    pub enable_expanding_box: bool,

    /// Pump gain for dynamic demand amplification of timing-violating nets (default: 10.0).
    /// Nets with high criticality get demand scaled by (1 + pump_gain * crit^2).
    pub pump_gain: f64,

    /// Velocity field momentum decay (default: 0.85). Range [0, 1).
    pub velocity_alpha: f64,

    /// Helmholtz de-clustering step size relative to main step (default: 0.3).
    pub helmholtz_eta_ratio: f64,
}

impl Default for OptTransPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            turbulence_beta: 4.0,
            newton_iters: 1,
            cg_max_iters: 500,
            cg_tolerance: 1e-3,
            cfl_number: 0.5,
            max_outer_iters: 200,
            report_interval: 5,
            timing_weight: 0.0,
            gas_temperature: 1.0,
            init_strategy: InitStrategy::Uniform,
            lap_max_cells: 10000,
            star_weight: 1.0,
            pressure_weight_start: 0.0,
            pressure_weight_end: 2.0,
            io_boost: 4.0,
            nesterov_step_size: 0.1,
            momentum: None,
            wl_coeff: 0.5,
            enable_expanding_box: true,
            pump_gain: 10.0,
            velocity_alpha: 0.7,
            helmholtz_eta_ratio: 0.15,
        }
    }
}

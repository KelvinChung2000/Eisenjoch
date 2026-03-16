//! Configuration for the ElectroPlace (RePlAce-style) placer.

/// Configuration for the ElectroPlace analytical placer.
#[derive(Clone)]
pub struct ElectroPlaceCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// WA wirelength coefficient (fixed, no annealing). Default 0.5.
    pub wl_coeff: f64,
    /// Target utilization for spacer insertion. Default 0.7.
    pub target_util: f64,
    /// Enable timing-driven net weighting. Default false.
    pub timing_driven: bool,
    /// Target density (typically 0.9-1.0).
    pub target_density: f64,
    /// Timing penalty weight.
    pub timing_weight: f64,
    /// Initial Nesterov step size.
    pub nesterov_step_size: f64,
    /// Maximum outer iterations (safety limit).
    pub max_iters: usize,
    /// Legalize every N iterations.
    pub legalize_interval: usize,
}

impl Default for ElectroPlaceCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            wl_coeff: 0.5,
            target_util: 0.7,
            timing_driven: false,
            target_density: 1.0,
            timing_weight: 0.0,
            nesterov_step_size: 0.1,
            max_iters: 500,
            legalize_interval: 5,
        }
    }
}

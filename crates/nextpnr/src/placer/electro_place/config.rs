//! Configuration for the ElectroPlace (ePlace-style) placer.

/// Configuration for the ElectroPlace analytical placer.
#[derive(Clone)]
pub struct ElectroPlaceCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Initial LSE smoothing parameter (0 = auto).
    pub gamma_init: f64,
    /// Gamma annealing rate.
    pub gamma_decay: f64,
    /// Minimum gamma.
    pub gamma_min: f64,
    /// Density penalty weight (λ). Set to 0 for auto-computation via
    /// gradient norm ratio (wirelength_norm / density_norm).
    pub density_weight: f64,
    /// Target density (typically 0.9-1.0).
    pub target_density: f64,
    /// Timing penalty weight.
    pub timing_weight: f64,
    /// Initial Nesterov step size.
    pub nesterov_step_size: f64,
    /// Maximum outer iterations.
    pub max_iters: usize,
    /// Legalize every N iterations.
    pub legalize_interval: usize,
}

impl Default for ElectroPlaceCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            gamma_init: 0.0,
            gamma_decay: 0.9,
            gamma_min: 1.0,
            density_weight: 0.0,
            target_density: 1.0,
            timing_weight: 0.0,
            nesterov_step_size: 0.1,
            max_iters: 500,
            legalize_interval: 5,
        }
    }
}

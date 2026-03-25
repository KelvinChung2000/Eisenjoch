/// Configuration for the HeAP placer.
#[derive(Debug, Clone)]
pub struct PlacerHeapCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Weight for timing cost (currently reserved, unused).
    pub timing_weight: f64,
    /// Initial spreading force multiplier. Grows by 1.5x each iteration.
    pub alpha: f64,
    /// Weight for net connections in the quadratic system.
    pub beta: f64,
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Quality threshold at which spreading is considered good enough.
    pub spreading_threshold: f64,
    /// Conjugate gradient solver convergence tolerance.
    pub solver_tolerance: f64,
    /// Maximum CG solver iterations.
    pub max_solver_iters: usize,
    /// Weight for congestion-aware forces (0.0 = no congestion awareness).
    pub congestion_weight: f64,
}

impl Default for PlacerHeapCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            timing_weight: 0.5,
            alpha: 0.1,
            beta: 1.0,
            max_iterations: 20,
            spreading_threshold: 0.95,
            solver_tolerance: 1e-5,
            max_solver_iters: 100,
            congestion_weight: 0.5,
        }
    }
}

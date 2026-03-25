/// Configuration for the simulated annealing placer.
#[derive(Debug, Clone)]
pub struct PlacerSaCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Cooling rate per outer iteration (e.g. 0.995).
    pub cooling_rate: f64,
    /// Inner loop iterations as a multiple of the cell count.
    pub inner_iters_per_cell: i32,
    /// Factor for computing the initial temperature from the initial cost.
    pub initial_temp_factor: f64,
    /// Temperature at which the annealing loop stops.
    pub min_temp: f64,
    /// Weight for timing cost (0.0 = pure HPWL, 1.0 = pure timing).
    pub timing_weight: f64,
    /// Enable slack redistribution (currently unused, reserved for future).
    pub slack_redistribution: bool,
    /// Weight for congestion cost relative to HPWL (0.0 = no congestion awareness).
    pub congestion_weight: f64,
}

impl Default for PlacerSaCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            cooling_rate: 0.995,
            inner_iters_per_cell: 10,
            initial_temp_factor: 1.5,
            min_temp: 1e-6,
            timing_weight: 0.5,
            slack_redistribution: true,
            congestion_weight: 0.0,
        }
    }
}

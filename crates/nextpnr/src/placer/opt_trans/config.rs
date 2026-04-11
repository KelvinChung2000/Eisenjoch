//! Configuration for the Beckmann optimal transport placer.

use crate::netlist::NetId;
use rustc_hash::FxHashMap;

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
    /// Number of nets solved together in the per-net path solver.
    pub net_parallel_batch_size: usize,
    /// Weight for timing-critical nets inside the attractive flux objective.
    pub timing_weight: f64,
    /// Optional per-net timing criticality map (0.0..1.0).
    pub timing_criticality: FxHashMap<NetId, f32>,
    /// IO net demand amplification factor.
    pub io_boost: f64,
    /// Exponent used to normalize sink demand by fanout.
    pub fanout_norm_exp: f64,
    /// Apply sqrt(fanout) scaling to IO demand for large nets.
    pub fanout_weight_sqrt: bool,
    /// Report every N iterations.
    pub report_interval: usize,
    /// Maximum cells for legalization.
    pub lap_max_cells: usize,
    /// Initialization strategy.
    pub init_strategy: InitStrategy,
    /// Number of Rayon worker threads for per-net solves.
    pub num_threads: usize,

    /// Legalization strategy: "ring", "sorted", "bipartite", "greedy".
    pub legalization: String,

    // --- Adam/EMA step control ---
    /// Global gain applied to the Adam learning rate computed from energy progress.
    pub adam_lr_gain: f64,
    /// EMA beta for relative energy progress used to adapt Adam's learning rate.
    pub energy_progress_ema_beta: f64,
}

impl Default for OptTransPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            max_iters: 500,
            net_parallel_batch_size: 4,
            timing_weight: 0.0,
            timing_criticality: FxHashMap::default(),
            io_boost: 1.0,
            fanout_norm_exp: 1.0,
            fanout_weight_sqrt: false,
            report_interval: 5,
            lap_max_cells: 10000,
            init_strategy: InitStrategy::RandomBel,
            num_threads: 8,
            legalization: "ring".to_string(),
            adam_lr_gain: 1.0,
            energy_progress_ema_beta: 0.8,
        }
    }
}

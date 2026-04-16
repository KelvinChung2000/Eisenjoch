//! Configuration for the Beckmann optimal transport placer.

use crate::netlist::NetId;
use rustc_hash::FxHashMap;

/// Sweep strategy for the DCD optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepMode {
    /// Jacobi: every cell scans the full grid against a frozen dist_cache, then
    /// all winning moves are applied simultaneously with damping `jacobi_alpha`.
    JacobiFullscan,
    /// Sequential: cells processed in topo order, each uses log-depth bisection
    /// in x then y, commits immediately, and refreshes dist_cache for nets it
    /// drives so the next cell sees the live field.
    SequentialBisection,
    /// Pure parallel bisection: every cell runs the 2D quadtree (with K×K seed)
    /// against a frozen dist_cache via rayon par_iter; all winning moves are
    /// applied simultaneously. The outer loop refreshes the field between sweeps.
    JacobiBisection,
    /// Best-first branch-and-bound over a per-net region-min pyramid. For each
    /// cell, the search is guaranteed to return the same argmin that an
    /// exhaustive fullscan would find under the frozen `dist_cache`, typically
    /// touching far fewer nodes than fullscan.
    JacobiBB,
}

impl Default for SweepMode {
    fn default() -> Self {
        Self::JacobiFullscan
    }
}

/// Path/load model used to build DCD costs and fractional pipe usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathModel {
    /// Shortest-label Dial-logit assignment over the pipe graph.
    DialLogit,
    /// Sparse Bresenham/dogleg route-template logit assignment.
    BresenhamLogit,
}

impl Default for PathModel {
    fn default() -> Self {
        Self::DialLogit
    }
}

/// Graph/cost representation used by the path solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphModel {
    /// Current flat per-pipe cost arrays.
    FlatPipe,
    /// Cache costs by source tile type, span delta, usage, and capacity/base
    /// buckets, while retaining the existing pipe graph and usage accounting.
    TwoLevelSpan,
    /// Reserved for a future chipdb PIP-level tile transit model.
    TwoLevelPip,
    /// Shift-invariant displacement table keyed by (col_src, col_dst, dy, dx).
    /// Uniform (src, sink) pairs served by O(1) lookup into D_ref/W_ref; all
    /// others fall back to per-net bucket-Dial. No congestion correction.
    DisplacementPure,
    /// DisplacementPure + per-hot-pipe sparse correction applied to nets whose
    /// bounding box touches any pipe with R_eff > 1.2 * base.
    DisplacementSparse,
    /// Limit the per-net bucket-Dial search to the (src, sinks) bounding box
    /// plus a fixed margin, early-terminating when all sinks are reached.
    /// Leaves the cost model unchanged.
    CorridorPrune,
    /// CorridorPrune + use D_ref as a priority upper bound to prune the Dial
    /// frontier further; the real bucket-Dial still runs and refines within
    /// the corridor.
    CorridorWarmstart,
}

impl Default for GraphModel {
    fn default() -> Self {
        Self::FlatPipe
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
    /// Number of nets solved together in the per-net path solver.
    pub net_parallel_batch_size: usize,
    /// Weight for timing-critical nets inside the attractive flux objective.
    pub timing_weight: f64,
    /// Optional per-net timing criticality map (0.0..1.0).
    pub timing_criticality: FxHashMap<NetId, f32>,
    /// IO net demand amplification factor.
    pub io_boost: f64,
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

    // --- DCD optimizer ---
    /// Maximum freeze-and-refresh DCD outer iterations.
    pub max_outer_iters: usize,
    /// Trisection depth per coordinate for each cell in each outer iteration.
    pub dcd_iters_per_cell: usize,
    /// Damping factor for Jacobi-style simultaneous position update.
    /// 1.0 = full jump to individual optimum (overshoots under conflicts),
    /// 0.0 = no movement. 0.5 is a reasonable starting point.
    pub jacobi_alpha: f64,
    /// Which sweep strategy to use inside the DCD outer loop.
    pub sweep_mode: SweepMode,
    /// Which path model to use for per-net cost and usage refresh.
    pub path_model: PathModel,
    /// Which graph/cost representation to use for path solves.
    pub graph_model: GraphModel,
    /// Number of uniform seed samples taken before bisection refines, per axis.
    /// Purpose: pick the correct basin of a multi-modal cost surface before
    /// the log-depth search locks in on a local minimum.
    pub bisect_seed_k: usize,
    /// Region size (in tiles) for skipping per-move dist_cache refresh in the
    /// sequential bisection sweep. A move is treated as "in the same region"
    /// when `(gx/R, gy/R)` is unchanged; in that case the Dial refresh is
    /// skipped and the cached distances are reused for subsequent cells.
    /// Set to `1` to always refresh (conservative). Larger values trade some
    /// staleness inside a sweep for fewer per-net path solves.
    pub bisect_refresh_region: i32,

    // --- Scarcity-scaled pipe cost (always on) ---
    /// Slope of the linear scarcity penalty applied at pipe creation:
    ///     factor = 1 + k * max(0, 1 - cap / median_cap(span))
    ///     base   = sqrt(span) * factor
    /// At `cap = median_cap(span)` factor=1 (no penalty). At `cap=0`, factor=1+k.
    /// Penalises narrow (low-cap) pipes relative to their own span bucket, so
    /// IOB/NULL/CLK boundary wires don't get used as general routing. Default 10.0.
    pub scarcity_k: f64,

    /// EMA old-usage blend for updating fractional `pipe.net_count` between
    /// outer iterations: `net_count = α * net_count + (1-α) * new_usage`.
    /// Default 0.5.
    pub blend_alpha: f64,

    // --- Softmin position update ---
    /// If true, the Jacobi commit step picks a softmin-weighted expected
    /// position over the cell's probe set instead of the hard argmin. This
    /// matches the logit path assignment philosophy: a cell at iter N does
    /// not collapse onto one tile but spreads mass over nearby tiles in
    /// proportion to `exp(-theta * cost)`. With theta → ∞ the update reduces
    /// to argmin.
    pub softmin_enabled: bool,
    /// Starting softmin stiffness (low = broad, exploratory).
    /// Default 0.25, matching `path_solver::LOGIT_THETA`.
    pub softmin_theta_start: f64,
    /// Ending softmin stiffness (high = argmin-like).
    /// Default 5.0 — `exp(-5)` ≈ 0.007 on a 1-unit cost gap.
    pub softmin_theta_end: f64,
    /// If true, constant (GND/VCC) nets are excluded from the solve set.
    pub skip_constants: bool,
    /// If true, clock-like nets (name contains "clk"/"clock") are excluded.
    pub skip_clocks: bool,
    /// If true, both constants AND clock-like nets are excluded (convenience
    /// toggle that ORs into both filters above).
    pub exclude_globals: bool,
}

impl Default for OptTransPlacerCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            net_parallel_batch_size: 4,
            timing_weight: 0.0,
            timing_criticality: FxHashMap::default(),
            io_boost: 1.0,
            report_interval: 5,
            lap_max_cells: 10000,
            init_strategy: InitStrategy::RandomBel,
            num_threads: 8,
            legalization: "ring".to_string(),
            max_outer_iters: 50,
            dcd_iters_per_cell: 8,
            jacobi_alpha: 1.0,
            sweep_mode: SweepMode::JacobiFullscan,
            path_model: PathModel::DialLogit,
            graph_model: GraphModel::FlatPipe,
            bisect_seed_k: 8,
            bisect_refresh_region: 1,
            scarcity_k: 10.0,
            blend_alpha: 0.5,
            skip_constants: true,
            skip_clocks: false,
            exclude_globals: false,
            softmin_enabled: true,
            softmin_theta_start: 0.25,
            softmin_theta_end: 5.0,
        }
    }
}

impl OptTransPlacerCfg {
    /// Override config fields from `NPNR_OT_*` env vars. Reads every known
    /// behavioural flag once, at placer entry, instead of re-reading per
    /// outer iteration or per cell.
    pub fn apply_env_overrides(&mut self) {
        use std::env;
        if let Some(v) = env::var("NPNR_OT_BLEND_ALPHA")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
        {
            self.blend_alpha = v.clamp(0.0, 1.0);
        }
        // NPNR_OT_INCLUDE_CONSTANTS=1 disables the default skip_constants.
        if env::var("NPNR_OT_INCLUDE_CONSTANTS").ok().as_deref() == Some("1") {
            self.skip_constants = false;
        }
        if env::var("NPNR_OT_EXCLUDE_CLOCKS").ok().as_deref() == Some("1") {
            self.skip_clocks = true;
        }
        if env::var("NPNR_OT_EXCLUDE_GLOBALS").ok().as_deref() == Some("1") {
            self.exclude_globals = true;
            self.skip_clocks = true;
        }
        if env::var("NPNR_OT_BRESENHAM_LOGIT").ok().as_deref() == Some("1") {
            self.path_model = PathModel::BresenhamLogit;
        }
        if let Some(v) = env::var("NPNR_OT_GRAPH_MODEL").ok() {
            self.graph_model = match v.as_str() {
                "flat" | "FlatPipe" | "FLAT" => GraphModel::FlatPipe,
                "span" | "twolevel_span" | "TwoLevelSpan" | "TWOLEVEL_SPAN" => {
                    GraphModel::TwoLevelSpan
                }
                "pip" | "twolevel_pip" | "TwoLevelPip" | "TWOLEVEL_PIP" => GraphModel::TwoLevelPip,
                "disp" | "disp_pure" | "DisplacementPure" | "DISP_PURE" => {
                    GraphModel::DisplacementPure
                }
                "disp_sparse" | "DisplacementSparse" | "DISP_SPARSE" => {
                    GraphModel::DisplacementSparse
                }
                "corridor" | "corridor_prune" | "CorridorPrune" | "CORRIDOR" => {
                    GraphModel::CorridorPrune
                }
                "corridor_warm" | "CorridorWarmstart" | "CORRIDOR_WARM" => {
                    GraphModel::CorridorWarmstart
                }
                _ => self.graph_model,
            };
        }
        if let Some(v) = env::var("NPNR_OT_TWOLEVEL").ok() {
            self.graph_model = match v.as_str() {
                "0" | "false" | "FALSE" => GraphModel::FlatPipe,
                "1" | "true" | "TRUE" => GraphModel::TwoLevelSpan,
                _ => self.graph_model,
            };
        }
        if let Some(v) = env::var("NPNR_OT_SOFTMIN").ok() {
            self.softmin_enabled = matches!(v.as_str(), "1" | "true" | "TRUE");
        }
        if let Some(v) = env::var("NPNR_OT_SOFTMIN_THETA_START")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
        {
            self.softmin_theta_start = v.max(1e-6);
        }
        if let Some(v) = env::var("NPNR_OT_SOFTMIN_THETA_END")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
        {
            self.softmin_theta_end = v.max(1e-6);
        }
    }
}

//! Configuration for the placer1 refinement pass.

/// `Placer1Cfg`, restricted to the settings the refinement path reads.
///
/// Defaults are upstream's (`placer1.cc:1233`). `net_share_weight` is absent
/// because it defaults to 0 there, which makes the whole `nets_by_tile`
/// machinery multiply out to nothing; if it is ever wanted it has to come back
/// with `update_nets_by_tile`.
#[derive(Clone, Debug)]
pub struct RefineCfg {
    /// Penalty on cluster-constraint distance, scaled by `1/temp`.
    pub constraint_weight: f64,
    /// Exponent applied to criticality, so near-critical arcs dominate.
    pub crit_exp: f64,
    /// Blend between timing (`lambda`) and wirelength (`1 - lambda`).
    pub lambda: f64,
    pub hpwl_scale_x: i32,
    pub hpwl_scale_y: i32,
    /// Below this many bels of a type, pick from the whole fabric instead of a
    /// window around the cell.
    pub min_bels_for_grid_pick: i32,
    pub timing_driven: bool,
    /// Nets at or above this fanout carry no timing cost. Upstream sets it to
    /// `INT_MAX`, i.e. off.
    pub timing_fanout_thresh: usize,
    /// Starting search radius. Refinement starts tight (`placer1.cc:234`).
    pub diameter: i32,
    /// Starting temperature. `1e-7` makes acceptance almost, but not exactly,
    /// downhill-only -- the loop's own exit test keys on this same value.
    pub temp: f64,
    /// Passes over every movable cell per outer iteration.
    pub inner_iters: usize,
}

impl Default for RefineCfg {
    fn default() -> Self {
        Self {
            constraint_weight: 10.0,
            crit_exp: 8.0,
            lambda: 0.5,
            hpwl_scale_x: 1,
            hpwl_scale_y: 1,
            min_bels_for_grid_pick: 64,
            timing_driven: true,
            timing_fanout_thresh: usize::MAX,
            diameter: 3,
            temp: 1e-7,
            inner_iters: 15,
        }
    }
}

/// What a refinement run did, for logging and for tests that need to prove the
/// pass was not a no-op.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    pub timing: RefineTiming,
    /// Unclustered movable cells the pass was allowed to move.
    pub autoplaced: usize,
    /// Cluster roots the pass was allowed to move as chains.
    pub chain_basis: usize,
    pub iterations: usize,
    pub moves_tried: u64,
    pub moves_accepted: u64,
    pub wirelen_before: i64,
    pub wirelen_after: i64,
}

/// Timing side of a refinement run, kept separate because it is the half that
/// silently degrades: if the analyser reports no criticality the timing term
/// vanishes and the pass quietly becomes wirelength-only.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineTiming {
    pub cost_before: f64,
    pub cost_after: f64,
    /// Arcs with nonzero criticality at setup. Zero means the timing term is
    /// inert and the run is not timing-driven, whatever the config says.
    pub critical_arcs: usize,
}

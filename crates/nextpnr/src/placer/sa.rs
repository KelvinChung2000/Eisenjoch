//! Simulated Annealing (SA) placer for FPGA cell placement.
//!
//! This implements the Placer1/SA algorithm: cells are initially placed at random
//! valid BELs, then iteratively improved by proposing random swap moves and
//! accepting or rejecting them via the Metropolis criterion. The cost function
//! combines HPWL (Half-Perimeter Wire Length) with optional congestion-awareness
//! via edge-based demand tracking and optional timing-driven weighting via net
//! criticality.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::metrics::{bresenham_line, net_hpwl, total_hpwl};
use crate::netlist::{CellId, NetId};
use log::{debug, info};
use rustc_hash::FxHashMap;

use super::common;
use super::common::initial_placement;
use super::PlacerError;

/// Simulated annealing placer.
pub struct PlacerSa;

impl super::Placer for PlacerSa {
    type Config = PlacerSaCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_sa(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), super::PlacerError> {
        use rustc_hash::FxHashSet;
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        super::common::with_locked_others(ctx, &cells_set, |ctx| place_sa(ctx, cfg))
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the simulated annealing placer.
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

// ---------------------------------------------------------------------------
// Congestion cache
// ---------------------------------------------------------------------------

/// Cached congestion state for incremental updates during SA.
///
/// Maintains per-edge demand and capacity grids that can be incrementally
/// updated as nets are moved. Only over-capacity edges contribute to the
/// congestion cost penalty.
///
/// Uses a per-net Bresenham point cache to avoid recomputing line segments
/// when removing demand (the common revert path). When adding demand the
/// Bresenham lines are computed fresh (positions may have changed) and
/// cached for future removal.
///
/// The congestion cost is tracked fully incrementally: each edge update
/// adjusts `cached_cost` by the delta in that edge's over-capacity penalty,
/// making `total_congestion_cost()` O(1).
pub struct CongestionCache {
    /// East-edge demand grid [y][x].
    h_demand: Vec<Vec<f64>>,
    /// South-edge demand grid [y][x].
    v_demand: Vec<Vec<f64>>,
    /// East-edge capacity grid [y][x].
    h_capacity: Vec<Vec<f64>>,
    /// South-edge capacity grid [y][x].
    v_capacity: Vec<Vec<f64>>,
    /// Grid width.
    grid_w: i32,
    /// Grid height.
    grid_h: i32,
    /// Cached Bresenham point lists per net (one Vec<(i32,i32)> per driver→sink pair).
    net_points: FxHashMap<NetId, Vec<Vec<(i32, i32)>>>,
    /// Running congestion cost (sum of max(0, ratio-1) for over-capacity edges).
    cached_cost: f64,
}

impl CongestionCache {
    /// Build capacity grids from chipdb and initialize demand by tracing all placed nets.
    pub fn new(ctx: &Context) -> Self {
        let grid_w = ctx.chipdb().width();
        let grid_h = ctx.chipdb().height();
        let wu = grid_w as usize;
        let hu = grid_h as usize;

        let mut h_capacity = vec![vec![0.0f64; wu]; hu];
        let mut v_capacity = vec![vec![0.0f64; wu]; hu];
        for ty in 0..grid_h {
            for tx in 0..grid_w {
                let tile_idx = ty * grid_w + tx;
                let tt = ctx.chipdb().tile_type(tile_idx);
                let nwires = tt.wires.get().len() as f64;
                let cap = (nwires / 4.0).max(1.0);
                h_capacity[ty as usize][tx as usize] = cap;
                v_capacity[ty as usize][tx as usize] = cap;
            }
        }

        let mut cache = Self {
            h_demand: vec![vec![0.0; wu]; hu],
            v_demand: vec![vec![0.0; wu]; hu],
            h_capacity,
            v_capacity,
            grid_w,
            grid_h,
            net_points: FxHashMap::default(),
            cached_cost: 0.0,
        };

        // Initialize demand from all current nets.
        for (net_idx, _) in ctx.design.iter_alive_nets() {
            cache.add_net_demand(ctx, net_idx, 1.0);
        }

        cache
    }

    /// O(1) congestion cost lookup.
    pub fn total_congestion_cost(&self) -> f64 {
        self.cached_cost
    }

    /// Update a single edge's demand and adjust cached_cost incrementally.
    #[inline]
    fn update_edge(demand: &mut f64, capacity: f64, sign: f64, cached_cost: &mut f64) {
        let old_ratio = *demand / capacity;
        let old_penalty = if old_ratio > 1.0 { old_ratio - 1.0 } else { 0.0 };

        *demand = (*demand + sign).max(0.0);

        let new_ratio = *demand / capacity;
        let new_penalty = if new_ratio > 1.0 { new_ratio - 1.0 } else { 0.0 };

        *cached_cost += new_penalty - old_penalty;
    }

    /// Apply edge crossings from a Bresenham point list, updating demand and cost.
    fn apply_crossings(&mut self, points: &[(i32, i32)], sign: f64) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        for window in points.windows(2) {
            let (x1, y1) = window[0];
            let (x2, y2) = window[1];
            let dx = x2 - x1;
            let dy = y2 - y1;

            if dx != 0 {
                let ex = if dx > 0 { x1 } else { x2 };
                let ey = y1;
                if ex >= 0 && ex < gw - 1 && ey >= 0 && ey < gh {
                    Self::update_edge(
                        &mut self.h_demand[ey as usize][ex as usize],
                        self.h_capacity[ey as usize][ex as usize],
                        sign,
                        &mut self.cached_cost,
                    );
                }
            }
            if dy != 0 {
                let ex = x1;
                let ey = if dy > 0 { y1 } else { y2 };
                if ex >= 0 && ex < gw && ey >= 0 && ey < gh - 1 {
                    Self::update_edge(
                        &mut self.v_demand[ey as usize][ex as usize],
                        self.v_capacity[ey as usize][ex as usize],
                        sign,
                        &mut self.cached_cost,
                    );
                }
            }
        }
    }

    /// Add or remove demand for a net.
    ///
    /// `sign` should be +1.0 to add demand or -1.0 to remove it.
    ///
    /// When removing (`sign < 0`): uses cached Bresenham points so no line
    /// recomputation is needed. When adding (`sign > 0`): computes fresh
    /// Bresenham lines from current cell positions and caches them.
    pub fn add_net_demand(&mut self, ctx: &Context, net_idx: NetId, sign: f64) {
        let net = ctx.design.net(net_idx);
        if !net.alive {
            return;
        }

        if sign < 0.0 {
            // Remove demand using cached points (no Bresenham recomputation).
            if let Some(point_lists) = self.net_points.remove(&net_idx) {
                for points in &point_lists {
                    self.apply_crossings(points, sign);
                }
            }
        } else {
            // Add demand: compute fresh Bresenham lines and cache them.
            let driver = match net.driver() {
                Some(pin) => pin,
                None => return,
            };
            let driver_bel = match ctx.design.cell(driver.cell).bel {
                Some(bel) => bel,
                None => return,
            };
            let driver_loc = ctx.chipdb().bel_loc(driver_bel);

            let mut point_lists = Vec::with_capacity(net.users().len());

            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let sink_bel = match ctx.design.cell(user.cell).bel {
                    Some(bel) => bel,
                    None => continue,
                };
                let sink_loc = ctx.chipdb().bel_loc(sink_bel);

                let points =
                    bresenham_line(driver_loc.x, driver_loc.y, sink_loc.x, sink_loc.y);
                self.apply_crossings(&points, sign);
                point_lists.push(points);
            }

            self.net_points.insert(net_idx, point_lists);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect all nets that are touched by a given cell (both as driver and as user).
fn nets_for_cell(ctx: &Context, cell_idx: CellId) -> Vec<NetId> {
    let cell = ctx.cell(cell_idx);
    let mut nets = Vec::new();
    for pin in cell.ports() {
        if let Some(net_idx) = pin.view(ctx).net_id() {
            nets.push(net_idx);
        }
    }
    nets.sort_unstable();
    nets.dedup();
    nets
}

/// Compute HPWL for a set of nets (used for incremental delta computation).
///
/// Uses parallel iteration for large net lists (> 16 nets) to amortize rayon overhead.
fn hpwl_for_nets(ctx: &Context, net_indices: &[NetId]) -> f64 {
    if net_indices.len() > 16 {
        use rayon::prelude::*;
        net_indices
            .par_iter()
            .map(|&idx| net_hpwl(ctx, idx))
            .sum()
    } else {
        net_indices.iter().map(|&idx| net_hpwl(ctx, idx)).sum()
    }
}

/// Collect all live cell indices that are not locked.
fn moveable_cells(ctx: &Context) -> Vec<CellId> {
    let mut result = Vec::new();
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        if !cell.bel_strength.is_locked() {
            result.push(cell_idx);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Swap move
// ---------------------------------------------------------------------------

/// Result of a proposed swap move.
pub struct SwapResult {
    /// The delta in HPWL cost (negative = improvement).
    pub delta_cost: f64,
    /// The delta in congestion cost (negative = improvement).
    pub delta_congestion: f64,
    /// Whether the move was actually performed on the context.
    pub performed: bool,
    /// The nets affected by this swap (needed for congestion revert).
    pub affected_nets: Vec<NetId>,
}

impl SwapResult {
    /// A no-op result: no move was performed, no cost change.
    fn noop() -> Self {
        Self {
            delta_cost: 0.0,
            delta_congestion: 0.0,
            performed: false,
            affected_nets: Vec::new(),
        }
    }
}

/// Attempt a swap move: move `cell` to `target_bel`.
///
/// If `target_bel` is occupied by another cell, the two cells are swapped.
/// The function computes the delta cost by measuring HPWL of affected nets
/// before and after the move. If a congestion cache is provided, it also
/// computes the delta congestion by removing and re-adding demand for
/// affected nets.
///
/// The move is always performed (bind/unbind) so the caller can decide whether
/// to accept or revert. If the caller rejects, it must call `revert_swap`.
pub fn try_swap(
    ctx: &mut Context,
    cell_idx: CellId,
    target_bel: BelId,
    mut congestion: Option<&mut CongestionCache>,
) -> SwapResult {
    let cell = ctx.cell(cell_idx);
    let old_bel = cell.bel().map(|b| b.id());

    // If we are already at the target, no-op.
    if old_bel == Some(target_bel) {
        return SwapResult::noop();
    }

    let old_bel = match old_bel {
        Some(bel) => bel,
        None => return SwapResult::noop(),
    };

    // Determine if there is a cell at the target bel.
    let other_cell_idx = ctx.bel(target_bel).bound_cell().map(|c| c.id());

    // Check that the other cell (if any) is moveable.
    if let Some(oci) = other_cell_idx {
        let other_cell = ctx.cell(oci);
        if other_cell.bel_strength().is_locked() {
            return SwapResult::noop();
        }
    }

    // Collect affected nets before the move.
    let mut affected_nets = nets_for_cell(ctx, cell_idx);
    if let Some(oci) = other_cell_idx {
        let mut other_nets = nets_for_cell(ctx, oci);
        affected_nets.append(&mut other_nets);
        affected_nets.sort_unstable();
        affected_nets.dedup();
    }

    // Compute HPWL cost before.
    let cost_before = hpwl_for_nets(ctx, &affected_nets);

    // Remove congestion demand for affected nets (at old positions).
    let congestion_before = if let Some(ref mut cache) = congestion {
        for &net in &affected_nets {
            cache.add_net_demand(ctx, net, -1.0);
        }
        cache.total_congestion_cost()
    } else {
        0.0
    };

    // Unbind both cells.
    ctx.unbind_bel(old_bel);
    if other_cell_idx.is_some() {
        ctx.unbind_bel(target_bel);
    }

    // Bind cell to target_bel.
    ctx.bind_bel(target_bel, cell_idx, PlaceStrength::Placer);

    // Bind other cell to old_bel (if swap).
    if let Some(oci) = other_cell_idx {
        ctx.bind_bel(old_bel, oci, PlaceStrength::Placer);
    }

    // Compute HPWL cost after.
    let cost_after = hpwl_for_nets(ctx, &affected_nets);

    // Add congestion demand for affected nets (at new positions).
    let delta_congestion = if let Some(cache) = congestion {
        for &net in &affected_nets {
            cache.add_net_demand(ctx, net, 1.0);
        }
        let congestion_after = cache.total_congestion_cost();
        congestion_after - congestion_before
    } else {
        0.0
    };

    SwapResult {
        delta_cost: cost_after - cost_before,
        delta_congestion,
        performed: true,
        affected_nets,
    }
}

/// Revert a swap move (undo the last try_swap that was performed).
pub fn revert_swap(
    ctx: &mut Context,
    cell_idx: CellId,
    old_bel: BelId,
    other_cell_idx: Option<CellId>,
    current_bel: BelId,
) {
    // Unbind current positions.
    ctx.unbind_bel(current_bel);
    if let Some(oci) = other_cell_idx {
        ctx.unbind_bel(old_bel);
        // Restore other cell to its original position (current_bel was its old bel).
        ctx.bind_bel(current_bel, oci, PlaceStrength::Placer);
    }
    // Restore cell to old_bel.
    ctx.bind_bel(old_bel, cell_idx, PlaceStrength::Placer);
}

// ---------------------------------------------------------------------------
// Main SA loop
// ---------------------------------------------------------------------------

/// Run the SA placer on the given context.
///
/// Steps:
/// 1. Initial placement: assign all unplaced cells to random valid BELs.
/// 2. Compute initial cost (total HPWL of all nets).
/// 3. SA loop: propose random swap moves, accept via Metropolis criterion.
/// 4. Cool the temperature each outer iteration.
/// 5. Stop when temperature falls below `min_temp`.
pub fn place_sa(ctx: &mut Context, cfg: &PlacerSaCfg) -> Result<(), PlacerError> {
    // Seed the RNG.
    ctx.reseed_rng(cfg.seed);

    // Step 1: initial placement.
    info!("SA Placer: performing initial placement...");
    initial_placement(ctx)?;

    // Gather moveable cells.
    let moveable = moveable_cells(ctx);
    let num_cells = moveable.len();
    if num_cells == 0 {
        info!("SA Placer: no moveable cells, nothing to do.");
        return Ok(());
    }
    info!("SA Placer: {} moveable cells.", num_cells);

    // Pre-cache BELs per cell type to avoid re-collecting every inner iteration.
    let mut bel_cache: FxHashMap<IdString, Vec<BelId>> = FxHashMap::default();
    for &ci in &moveable {
        let cell = ctx.cell(ci);
        let cell_type = cell.cell_type_id();
        bel_cache.entry(cell_type).or_insert_with(|| {
            ctx.bels_for_bucket(cell_type)
                .map(|bel| bel.id())
                .collect()
        });
    }

    // Pre-cache region-filtered BELs for cells with region constraints.
    let mut region_bel_cache: FxHashMap<(IdString, u32), Vec<BelId>> = FxHashMap::default();
    for &ci in &moveable {
        let cell = ctx.cell(ci);
        let cell_type = cell.cell_type_id();
        if let Some(rid) = ctx.design.cell(ci).region {
            region_bel_cache
                .entry((cell_type, rid))
                .or_insert_with(|| {
                    bel_cache
                        .get(&cell_type)
                        .map(|bels| {
                            bels.iter()
                                .copied()
                                .filter(|&b| ctx.is_bel_in_region(b, rid))
                                .collect()
                        })
                        .unwrap_or_default()
                });
        }
    }

    // Step 2: compute initial cost.
    let mut current_cost = total_hpwl(ctx);
    info!("SA Placer: initial HPWL cost = {:.2}", current_cost);

    // Build congestion cache if congestion weight is enabled.
    let mut congestion_cache = if cfg.congestion_weight > 0.0 {
        let cache = CongestionCache::new(ctx);
        info!(
            "SA Placer: congestion weight = {:.2}, initial congestion cost = {:.2}",
            cfg.congestion_weight,
            cache.total_congestion_cost()
        );
        Some(cache)
    } else {
        None
    };
    let mut congestion_cost = congestion_cache
        .as_ref()
        .map_or(0.0, |c| c.total_congestion_cost());

    // Compute initial temperature.
    let mut temperature = current_cost * cfg.initial_temp_factor / num_cells as f64;
    if temperature < cfg.min_temp {
        temperature = cfg.min_temp * 10.0;
    }
    info!("SA Placer: initial temperature = {:.6}", temperature);

    let inner_iters = (num_cells as i32 * cfg.inner_iters_per_cell).max(1) as u32;
    let mut n_accept = 0u64;
    let mut n_moves = 0u64;
    let mut iteration = 0u32;

    // Step 3: SA outer loop.
    while temperature > cfg.min_temp {
        let mut iter_accept = 0u32;

        for _ in 0..inner_iters {
            // Pick a random moveable cell.
            let cell_idx = moveable[ctx.rng_mut().next_range(num_cells as u32) as usize];
            let cell = ctx.cell(cell_idx);
            let cell_type = cell.cell_type_id();
            let old_bel = cell.bel().map(|b| b.id());

            let old_bel = match old_bel {
                Some(bel) => bel,
                None => continue,
            };

            // Get candidate BELs from cache, filtered by cell's region constraint.
            let cell_region = ctx.design.cell(cell_idx).region;
            let bucket_bels = if let Some(rid) = cell_region {
                match region_bel_cache.get(&(cell_type, rid)) {
                    Some(bels) => bels.as_slice(),
                    None => continue,
                }
            } else {
                match bel_cache.get(&cell_type) {
                    Some(bels) => bels.as_slice(),
                    None => continue,
                }
            };
            let num_bucket_bels = bucket_bels.len();
            if num_bucket_bels == 0 {
                continue;
            }

            // Pick a random valid BEL of the same bucket.
            let rand_idx = ctx.rng_mut().next_range(num_bucket_bels as u32) as usize;
            let target_bel = bucket_bels[rand_idx];

            // Determine the other cell (if any) for potential revert.
            let other_cell_idx = ctx.bel(target_bel).bound_cell().map(|c| c.id());

            // Check that the other cell type is compatible with old_bel.
            if let Some(oci) = other_cell_idx {
                let other_cell = ctx.cell(oci);
                if !ctx
                    .bel(old_bel)
                    .is_valid_for_cell_type(other_cell.cell_type_id())
                {
                    continue;
                }
                // Check that the other cell's region constraint allows old_bel.
                if let Some(other_rid) = ctx.design.cell(oci).region {
                    if !ctx.is_bel_in_region(old_bel, other_rid) {
                        continue;
                    }
                }
            }

            // Attempt the swap.
            let result = try_swap(ctx, cell_idx, target_bel, congestion_cache.as_mut());
            n_moves += 1;

            if !result.performed {
                continue;
            }

            // Compute combined delta including congestion penalty.
            let total_delta =
                result.delta_cost + cfg.congestion_weight * result.delta_congestion;

            // Metropolis criterion: accept if delta < 0 or with probability exp(-delta/T).
            let accept = if total_delta <= 0.0 {
                true
            } else {
                let prob = (-total_delta / temperature).exp();
                let rand_val = (ctx.rng_mut().next_u32() as f64) / (u32::MAX as f64);
                rand_val < prob
            };

            if accept {
                current_cost += result.delta_cost;
                congestion_cost += result.delta_congestion;
                n_accept += 1;
                iter_accept += 1;
            } else {
                // Revert congestion cache, then revert the move.
                if let Some(ref mut cache) = congestion_cache {
                    for &net in &result.affected_nets {
                        cache.add_net_demand(ctx, net, -1.0);
                    }
                }
                revert_swap(ctx, cell_idx, old_bel, other_cell_idx, target_bel);
                if let Some(ref mut cache) = congestion_cache {
                    for &net in &result.affected_nets {
                        cache.add_net_demand(ctx, net, 1.0);
                    }
                }
            }
        }

        // Cool the temperature.
        temperature *= cfg.cooling_rate;
        iteration += 1;

        if iteration % 50 == 0 {
            debug!(
                "SA Placer: iter={}, temp={:.6}, cost={:.2}, congestion={:.2}, accept_rate={:.2}%",
                iteration,
                temperature,
                current_cost,
                congestion_cost,
                (iter_accept as f64 / inner_iters as f64) * 100.0
            );
        }
    }

    info!(
        "SA Placer: finished after {} iterations, {} moves, {} accepted ({:.1}%).",
        iteration,
        n_moves,
        n_accept,
        if n_moves > 0 {
            (n_accept as f64 / n_moves as f64) * 100.0
        } else {
            0.0
        }
    );
    info!("SA Placer: final HPWL cost = {:.2}", current_cost);
    if cfg.congestion_weight > 0.0 {
        info!("SA Placer: final congestion cost = {:.2}", congestion_cost);
    }

    // Step 4: final validation -- check all alive cells are placed and region constraints hold.
    common::validate_all_placed(ctx)?;
    common::validate_region_constraints(ctx)?;

    Ok(())
}

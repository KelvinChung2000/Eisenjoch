//! Simulated Annealing (SA) placer for FPGA cell placement.
//!
//! This implements the Placer1/SA algorithm: cells are initially placed at random
//! valid BELs, then iteratively improved by proposing random swap moves and
//! accepting or rejecting them via the Metropolis criterion. The cost function
//! combines HPWL (Half-Perimeter Wire Length) with optional timing-driven
//! weighting via net criticality.

use crate::chipdb::BelId;
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::{CellId, NetId};
use log::{debug, info};

use super::common;
use super::common::{initial_placement, net_hpwl, total_hpwl};
use super::PlacerError;

/// Simulated annealing placer.
pub struct PlacerSa;

impl super::Placer for PlacerSa {
    type Config = PlacerSaCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_sa(ctx, cfg)
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
fn hpwl_for_nets(ctx: &Context, net_indices: &[NetId]) -> f64 {
    net_indices.iter().map(|&idx| net_hpwl(ctx, idx)).sum()
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
    /// Whether the move was actually performed on the context.
    pub performed: bool,
}

/// Attempt a swap move: move `cell` to `target_bel`.
///
/// If `target_bel` is occupied by another cell, the two cells are swapped.
/// The function computes the delta cost by measuring HPWL of affected nets
/// before and after the move.
///
/// The move is always performed (bind/unbind) so the caller can decide whether
/// to accept or revert. If the caller rejects, it must call `revert_swap`.
pub fn try_swap(ctx: &mut Context, cell_idx: CellId, target_bel: BelId) -> SwapResult {
    let cell = ctx.cell(cell_idx);
    let old_bel = cell.bel().map(|b| b.id());

    // If we are already at the target, no-op.
    if old_bel == Some(target_bel) {
        return SwapResult {
            delta_cost: 0.0,
            performed: false,
        };
    }

    let old_bel = match old_bel {
        Some(bel) => bel,
        None => {
            return SwapResult {
                delta_cost: 0.0,
                performed: false,
            }
        }
    };

    // Determine if there is a cell at the target bel.
    let other_cell_idx = ctx.bel(target_bel).bound_cell().map(|c| c.id());

    // Check that the other cell (if any) is moveable.
    if let Some(oci) = other_cell_idx {
        let other_cell = ctx.cell(oci);
        if other_cell.bel_strength().is_locked() {
            return SwapResult {
                delta_cost: 0.0,
                performed: false,
            };
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

    // Compute cost before.
    let cost_before = hpwl_for_nets(ctx, &affected_nets);

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

    // Compute cost after.
    let cost_after = hpwl_for_nets(ctx, &affected_nets);

    SwapResult {
        delta_cost: cost_after - cost_before,
        performed: true,
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

    // Step 2: compute initial cost.
    let mut current_cost = total_hpwl(ctx);
    info!("SA Placer: initial HPWL cost = {:.2}", current_cost);

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

            let bucket_bels: Vec<_> = ctx.bels_for_bucket(cell_type).map(|bel| bel.id()).collect();
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
            }

            // Attempt the swap.
            let result = try_swap(ctx, cell_idx, target_bel);
            n_moves += 1;

            if !result.performed {
                continue;
            }

            // Metropolis criterion: accept if delta < 0 or with probability exp(-delta/T).
            let accept = if result.delta_cost <= 0.0 {
                true
            } else {
                let prob = (-result.delta_cost / temperature).exp();
                let rand_val = (ctx.rng_mut().next_u32() as f64) / (u32::MAX as f64);
                rand_val < prob
            };

            if accept {
                current_cost += result.delta_cost;
                n_accept += 1;
                iter_accept += 1;
            } else {
                // Revert the move.
                revert_swap(ctx, cell_idx, old_bel, other_cell_idx, target_bel);
            }
        }

        // Cool the temperature.
        temperature *= cfg.cooling_rate;
        iteration += 1;

        if iteration % 50 == 0 {
            debug!(
                "SA Placer: iter={}, temp={:.6}, cost={:.2}, accept_rate={:.2}%",
                iteration,
                temperature,
                current_cost,
                if inner_iters > 0 {
                    (iter_accept as f64 / inner_iters as f64) * 100.0
                } else {
                    0.0
                }
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

    // Step 4: final validation -- check all alive cells are placed.
    common::validate_all_placed(ctx)?;

    Ok(())
}


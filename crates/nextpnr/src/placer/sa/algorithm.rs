//! Main SA placement loop.

use crate::common::IdString;
use crate::chipdb::BelId;
use crate::context::Context;
use crate::metrics::total_hpwl;
use crate::netlist::CellId;
use log::{debug, info};
use rustc_hash::FxHashMap;

use super::config::PlacerSaCfg;
use super::congestion::CongestionCache;
use super::swap::{try_swap, revert_swap};
use crate::placer::PlacerError;
use crate::placer::PlacerPipeline;

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

/// Run the SA placer on the given context.
///
/// Steps:
/// 1. Initial placement: assign all unplaced cells to random valid BELs.
/// 2. Compute initial cost (total HPWL of all nets).
/// 3. SA loop: propose random swap moves, accept via Metropolis criterion.
/// 4. Cool the temperature each outer iteration.
/// 5. Stop when temperature falls below `min_temp`.
pub fn place_sa(ctx: &mut Context, cfg: &PlacerSaCfg) -> Result<(), PlacerError> {
    // Step 1: seed RNG + initial placement.
    info!("SA Placer: performing initial placement...");
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

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
    PlacerPipeline::validate(ctx)?;

    Ok(())
}

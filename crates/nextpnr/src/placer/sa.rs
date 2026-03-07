//! Simulated Annealing (SA) placer for FPGA cell placement.
//!
//! This implements the Placer1/SA algorithm: cells are initially placed at random
//! valid BELs, then iteratively improved by proposing random swap moves and
//! accepting or rejecting them via the Metropolis criterion. The cost function
//! combines HPWL (Half-Perimeter Wire Length) with optional timing-driven
//! weighting via net criticality.

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use crate::types::{BelId, PlaceStrength};
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
    for port_info in cell.ports().values() {
        if let Some(net_idx) = port_info.net {
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
pub(crate) struct SwapResult {
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
pub(crate) fn try_swap(ctx: &mut Context, cell_idx: CellId, target_bel: BelId) -> SwapResult {
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
pub(crate) fn revert_swap(
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

            let bucket_bels: Vec<_> = ctx
                .bels_for_bucket(cell_type)
                .map(|bel| bel.id())
                .collect();
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

#[cfg(test)]
#[cfg(feature = "test-utils")]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::context::Context;
    use crate::netlist::{CellId, PortRef};
    use crate::types::{BelId, IdString, PlaceStrength, PortType};

    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    fn make_context_with_cells(n: usize) -> Context {
        assert!(n <= 4, "synthetic chipdb only has 4 BELs");
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT4");
        let mut cell_names = Vec::new();

        for i in 0..n {
            let name = ctx.id(&format!("cell_{}", i));
            ctx.design.add_cell(name, cell_type);
            cell_names.push(name);
        }

        if n >= 2 {
            let net_name = ctx.id("net_0");
            let net_idx = ctx.design.add_net(net_name);
            let q_port = ctx.id("Q");
            let a_port = ctx.id("A");

            let cell0_idx = ctx.design.cell_by_name(cell_names[0]).unwrap();
            ctx.design
                .cell_edit(cell0_idx)
                .add_port(q_port, PortType::Out);
            ctx.design
                .cell_edit(cell0_idx)
                .set_port_net(q_port, Some(net_idx), None);

            ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
                cell: Some(cell0_idx),
                port: q_port,
                budget: 0,
            });

            for i in 1..n {
                let cell_idx = ctx.design.cell_by_name(cell_names[i]).unwrap();
                ctx.design
                    .cell_edit(cell_idx)
                    .add_port(a_port, PortType::In);
                let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
                    cell: Some(cell_idx),
                    port: a_port,
                    budget: 0,
                });
                ctx.design
                    .cell_edit(cell_idx)
                    .set_port_net(a_port, Some(net_idx), Some(user_idx));
            }
        }

        ctx
    }

    // HPWL tests

    #[test]
    fn hpwl_no_driver_is_zero() {
        let mut ctx = make_context();
        let net_name = ctx.id("floating");
        let net_idx = ctx.design.add_net(net_name);
        assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
    }

    #[test]
    fn hpwl_no_users_is_zero() {
        let mut ctx = make_context();
        let cell_type = ctx.id("LUT4");
        let cell_name = ctx.id("drv");
        let cell_idx = ctx.design.add_cell(cell_name, cell_type);
        let net_name = ctx.id("n0");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        ctx.design
            .cell_edit(cell_idx)
            .add_port(q_port, PortType::Out);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(cell_idx),
            port: q_port,
            budget: 0,
        });
        assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
    }

    #[test]
    fn hpwl_adjacent_tiles() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();
        let cell_type = ctx.id("LUT4");
        let drv_name = ctx.id("drv");
        let usr_name = ctx.id("usr");
        let drv_idx = ctx.design.add_cell(drv_name, cell_type);
        let usr_idx = ctx.design.add_cell(usr_name, cell_type);
        ctx.bind_bel(BelId::new(0, 0), drv_idx, PlaceStrength::Placer);
        ctx.bind_bel(BelId::new(1, 0), usr_idx, PlaceStrength::Placer);
        let net_name = ctx.id("n");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");
        ctx.design
            .cell_edit(drv_idx)
            .add_port(q_port, PortType::Out);
        ctx.design
            .cell_edit(drv_idx)
            .set_port_net(q_port, Some(net_idx), None);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(drv_idx),
            port: q_port,
            budget: 0,
        });
        ctx.design.cell_edit(usr_idx).add_port(a_port, PortType::In);
        let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
            cell: Some(usr_idx),
            port: a_port,
            budget: 0,
        });
        ctx.design
            .cell_edit(usr_idx)
            .set_port_net(a_port, Some(net_idx), Some(user_idx));
        assert_eq!(net_hpwl(&ctx, net_idx), 1.0);
    }

    #[test]
    fn hpwl_diagonal_placement() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();
        let cell_type = ctx.id("LUT4");
        let drv_name = ctx.id("drv");
        let usr_name = ctx.id("usr");
        let drv_idx = ctx.design.add_cell(drv_name, cell_type);
        let usr_idx = ctx.design.add_cell(usr_name, cell_type);
        ctx.bind_bel(BelId::new(0, 0), drv_idx, PlaceStrength::Placer);
        ctx.bind_bel(BelId::new(3, 0), usr_idx, PlaceStrength::Placer);
        let net_name = ctx.id("n");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");
        ctx.design
            .cell_edit(drv_idx)
            .add_port(q_port, PortType::Out);
        ctx.design
            .cell_edit(drv_idx)
            .set_port_net(q_port, Some(net_idx), None);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(drv_idx),
            port: q_port,
            budget: 0,
        });
        ctx.design.cell_edit(usr_idx).add_port(a_port, PortType::In);
        let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
            cell: Some(usr_idx),
            port: a_port,
            budget: 0,
        });
        ctx.design
            .cell_edit(usr_idx)
            .set_port_net(a_port, Some(net_idx), Some(user_idx));
        assert_eq!(net_hpwl(&ctx, net_idx), 2.0);
    }

    #[test]
    fn total_hpwl_sums_all_nets() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();
        let cell_type = ctx.id("LUT4");
        let names: Vec<IdString> = (0..4).map(|i| ctx.id(&format!("c{}", i))).collect();
        let cell_indices: Vec<CellId> = names
            .iter()
            .map(|&n| ctx.design.add_cell(n, cell_type))
            .collect();
        for (i, &ci) in cell_indices.iter().enumerate() {
            ctx.bind_bel(BelId::new(i as i32, 0), ci, PlaceStrength::Placer);
        }
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");
        {
            let net_name = ctx.id("net_a");
            let net_idx = ctx.design.add_net(net_name);
            ctx.design
                .cell_edit(cell_indices[0])
                .add_port(q_port, PortType::Out);
            ctx.design
                .cell_edit(cell_indices[0])
                .set_port_net(q_port, Some(net_idx), None);
            ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
                cell: Some(cell_indices[0]),
                port: q_port,
                budget: 0,
            });
            ctx.design
                .cell_edit(cell_indices[3])
                .add_port(a_port, PortType::In);
            let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
                cell: Some(cell_indices[3]),
                port: a_port,
                budget: 0,
            });
            ctx.design.cell_edit(cell_indices[3]).set_port_net(
                a_port,
                Some(net_idx),
                Some(user_idx),
            );
        }
        {
            let b_port = ctx.id("B");
            let net_name = ctx.id("net_b");
            let net_idx = ctx.design.add_net(net_name);
            ctx.design
                .cell_edit(cell_indices[1])
                .add_port(b_port, PortType::Out);
            ctx.design
                .cell_edit(cell_indices[1])
                .set_port_net(b_port, Some(net_idx), None);
            ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
                cell: Some(cell_indices[1]),
                port: b_port,
                budget: 0,
            });
            let c_port = ctx.id("C");
            ctx.design
                .cell_edit(cell_indices[2])
                .add_port(c_port, PortType::In);
            let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
                cell: Some(cell_indices[2]),
                port: c_port,
                budget: 0,
            });
            ctx.design.cell_edit(cell_indices[2]).set_port_net(
                c_port,
                Some(net_idx),
                Some(user_idx),
            );
        }
        assert_eq!(total_hpwl(&ctx), 4.0);
    }

    // Initial placement tests

    #[test]
    fn initial_placement_places_all_cells() {
        let mut ctx = make_context_with_cells(3);
        initial_placement(&mut ctx).expect("should succeed");
        for (cell_idx, cell) in ctx.design.iter_alive_cells() {
            assert!(
                cell.bel.is_some(),
                "cell {} should be placed",
                cell_idx.raw()
            );
        }
    }

    #[test]
    fn initial_placement_no_duplicate_bels() {
        let mut ctx = make_context_with_cells(4);
        initial_placement(&mut ctx).expect("should succeed");
        let mut used_bels = std::collections::HashSet::new();
        for (_cell_idx, cell) in ctx.design.iter_alive_cells() {
            let bel = cell.bel.expect("alive cell should be placed");
            assert!(used_bels.insert(bel));
        }
    }

    #[test]
    fn initial_placement_too_many_cells_fails() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();
        let cell_type = ctx.id("LUT4");
        for i in 0..5 {
            let name = ctx.id(&format!("cell_{}", i));
            ctx.design.add_cell(name, cell_type);
        }
        assert!(initial_placement(&mut ctx).is_err());
    }

    #[test]
    fn initial_placement_skips_already_placed() {
        let mut ctx = make_context_with_cells(2);
        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        let bel = BelId::new(0, 0);
        ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer);
        initial_placement(&mut ctx).expect("should succeed");
        assert_eq!(ctx.design.cell(cell_idx).bel, Some(bel));
        let cell_name_1 = ctx.id("cell_1");
        let cell_idx_1 = ctx.design.cell_by_name(cell_name_1).unwrap();
        assert!(ctx.design.cell(cell_idx_1).bel.is_some());
        assert_ne!(ctx.design.cell(cell_idx_1).bel, Some(bel));
    }

    // Swap mechanics tests

    #[test]
    fn swap_to_empty_bel() {
        let mut ctx = make_context_with_cells(1);
        initial_placement(&mut ctx).expect("should succeed");
        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        let old_bel = ctx.design.cell(cell_idx).bel.unwrap();
        let bels: Vec<BelId> = ctx.bels().map(|b| b.id()).collect();
        let empty_bel = bels.iter().find(|&&b| b != old_bel).copied().unwrap();
        let result = try_swap(&mut ctx, cell_idx, empty_bel);
        assert!(result.performed);
        assert_eq!(ctx.design.cell(cell_idx).bel, Some(empty_bel));
        assert!(ctx.bel(old_bel).is_available());
    }

    #[test]
    fn swap_two_cells() {
        let mut ctx = make_context_with_cells(2);
        initial_placement(&mut ctx).expect("should succeed");
        let cell0_name = ctx.id("cell_0");
        let cell1_name = ctx.id("cell_1");
        let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
        let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();
        let bel0 = ctx.design.cell(cell0_idx).bel.unwrap();
        let bel1 = ctx.design.cell(cell1_idx).bel.unwrap();
        let result = try_swap(&mut ctx, cell0_idx, bel1);
        assert!(result.performed);
        assert_eq!(ctx.design.cell(cell0_idx).bel, Some(bel1));
        assert_eq!(ctx.design.cell(cell1_idx).bel, Some(bel0));
    }

    #[test]
    fn swap_same_bel_is_noop() {
        let mut ctx = make_context_with_cells(1);
        initial_placement(&mut ctx).expect("should succeed");
        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        let bel = ctx.design.cell(cell_idx).bel.unwrap();
        let result = try_swap(&mut ctx, cell_idx, bel);
        assert!(!result.performed);
        assert_eq!(result.delta_cost, 0.0);
    }

    #[test]
    fn revert_swap_restores_state() {
        let mut ctx = make_context_with_cells(2);
        initial_placement(&mut ctx).expect("should succeed");
        let cell0_name = ctx.id("cell_0");
        let cell1_name = ctx.id("cell_1");
        let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
        let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();
        let bel0 = ctx.design.cell(cell0_idx).bel.unwrap();
        let bel1 = ctx.design.cell(cell1_idx).bel.unwrap();
        let _result = try_swap(&mut ctx, cell0_idx, bel1);
        revert_swap(&mut ctx, cell0_idx, bel0, Some(cell1_idx), bel1);
        assert_eq!(ctx.design.cell(cell0_idx).bel, Some(bel0));
        assert_eq!(ctx.design.cell(cell1_idx).bel, Some(bel1));
    }
}

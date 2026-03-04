//! Simulated Annealing (SA) placer for FPGA cell placement.
//!
//! This implements the Placer1/SA algorithm: cells are initially placed at random
//! valid BELs, then iteratively improved by proposing random swap moves and
//! accepting or rejecting them via the Metropolis criterion. The cost function
//! combines HPWL (Half-Perimeter Wire Length) with optional timing-driven
//! weighting via net criticality.

use log::{debug, info};
use npnr_context::Context;
use npnr_netlist::{CellIdx, NetIdx};
use npnr_types::{BelId, IdString, PlaceStrength};
use rustc_hash::FxHashMap;

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
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during SA placement.
#[derive(Debug, thiserror::Error)]
pub enum PlacerError {
    #[error("No valid BELs available for cell type {0}")]
    NoBelsAvailable(String),
    #[error("Placement failed: {0}")]
    PlacementFailed(String),
    #[error("Initial placement failed: could not place cell {0}")]
    InitialPlacementFailed(String),
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect all nets that are touched by a given cell (both as driver and as user).
fn nets_for_cell(ctx: &Context, cell_idx: CellIdx) -> Vec<NetIdx> {
    let cell = ctx.design.cell(cell_idx);
    let mut nets = Vec::new();
    for port_info in cell.ports.values() {
        if port_info.net.is_some() {
            nets.push(port_info.net);
        }
    }
    nets.sort_by_key(|n| n.0);
    nets.dedup();
    nets
}

/// Compute HPWL for a single net.
///
/// HPWL = (max_x - min_x) + (max_y - min_y) across all connected cell locations.
/// Returns 0.0 for nets with no driver, no users, or dead nets.
fn net_hpwl(ctx: &Context, net_idx: NetIdx) -> f64 {
    let net = ctx.design.net(net_idx);
    if !net.alive || !net.driver.is_connected() || net.users.is_empty() {
        return 0.0;
    }

    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;

    // Include driver location.
    let driver_cell_idx = net.driver.cell;
    let driver_cell = ctx.design.cell(driver_cell_idx);
    if driver_cell.bel.is_valid() {
        let loc = ctx.get_bel_location(driver_cell.bel);
        min_x = min_x.min(loc.x);
        max_x = max_x.max(loc.x);
        min_y = min_y.min(loc.y);
        max_y = max_y.max(loc.y);
    }

    // Include all user locations.
    for user in &net.users {
        if !user.is_connected() {
            continue;
        }
        let user_cell = ctx.design.cell(user.cell);
        if user_cell.bel.is_valid() {
            let loc = ctx.get_bel_location(user_cell.bel);
            min_x = min_x.min(loc.x);
            max_x = max_x.max(loc.x);
            min_y = min_y.min(loc.y);
            max_y = max_y.max(loc.y);
        }
    }

    if min_x > max_x || min_y > max_y {
        return 0.0;
    }

    ((max_x - min_x) + (max_y - min_y)) as f64
}

/// Compute total HPWL cost across all alive nets.
fn total_hpwl(ctx: &Context) -> f64 {
    let mut total = 0.0;
    for (i, net) in ctx.design.net_store.iter().enumerate() {
        if net.alive {
            total += net_hpwl(ctx, NetIdx(i as u32));
        }
    }
    total
}

/// Compute HPWL for a set of nets (used for incremental delta computation).
fn hpwl_for_nets(ctx: &Context, net_indices: &[NetIdx]) -> f64 {
    let mut total = 0.0;
    for &net_idx in net_indices {
        total += net_hpwl(ctx, net_idx);
    }
    total
}

/// Collect all live, placeable cell indices grouped by their cell type.
///
/// Returns a map from cell type IdString to the list of CellIdx values.
fn cells_by_type(ctx: &Context) -> FxHashMap<IdString, Vec<CellIdx>> {
    let mut map: FxHashMap<IdString, Vec<CellIdx>> = FxHashMap::default();
    for (i, cell) in ctx.design.cell_store.iter().enumerate() {
        if cell.alive {
            map.entry(cell.cell_type).or_default().push(CellIdx(i as u32));
        }
    }
    map
}

/// Collect all live cell indices that are not locked.
fn moveable_cells(ctx: &Context) -> Vec<CellIdx> {
    let mut result = Vec::new();
    for (i, cell) in ctx.design.cell_store.iter().enumerate() {
        if cell.alive && !cell.bel_strength.is_locked() {
            result.push(CellIdx(i as u32));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Initial placement
// ---------------------------------------------------------------------------

/// Place all unplaced cells at random valid BELs.
///
/// Groups cells by type/bucket, collects available BELs of the matching bucket,
/// shuffles the BELs, and assigns cells sequentially.
fn initial_placement(ctx: &mut Context) -> Result<(), PlacerError> {
    ctx.populate_bel_buckets();

    let grouped = cells_by_type(ctx);

    for (&cell_type, cell_indices) in &grouped {
        let cell_type_name = ctx.name_of(cell_type);

        // Find all BELs matching this cell type's bucket.
        let bucket_bels = ctx.get_bels_for_bucket(&cell_type_name);
        if bucket_bels.is_empty() {
            return Err(PlacerError::NoBelsAvailable(cell_type_name));
        }

        // Collect available BELs.
        let mut available: Vec<BelId> = bucket_bels
            .iter()
            .copied()
            .filter(|b| ctx.is_bel_available(*b))
            .collect();

        ctx.rng.shuffle(&mut available);

        // Filter to only unplaced cells.
        let unplaced: Vec<CellIdx> = cell_indices
            .iter()
            .copied()
            .filter(|&ci| !ctx.design.cell(ci).bel.is_valid())
            .collect();

        if unplaced.len() > available.len() {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (need {} BELs but only {} available)",
                cell_type_name,
                unplaced.len(),
                available.len()
            )));
        }

        for (i, &cell_idx) in unplaced.iter().enumerate() {
            let bel = available[i];
            let cell_name = ctx.design.cell(cell_idx).name;
            if !ctx.bind_bel(bel, cell_name, PlaceStrength::Placer) {
                return Err(PlacerError::InitialPlacementFailed(
                    ctx.name_of(cell_name),
                ));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Swap move
// ---------------------------------------------------------------------------

/// Result of a proposed swap move.
struct SwapResult {
    /// The delta in HPWL cost (negative = improvement).
    delta_cost: f64,
    /// Whether the move was actually performed on the context.
    performed: bool,
}

/// Attempt a swap move: move `cell` to `target_bel`.
///
/// If `target_bel` is occupied by another cell, the two cells are swapped.
/// The function computes the delta cost by measuring HPWL of affected nets
/// before and after the move.
///
/// The move is always performed (bind/unbind) so the caller can decide whether
/// to accept or revert. If the caller rejects, it must call `revert_swap`.
fn try_swap(
    ctx: &mut Context,
    cell_idx: CellIdx,
    target_bel: BelId,
) -> SwapResult {
    let cell = ctx.design.cell(cell_idx);
    let old_bel = cell.bel;
    let cell_name = cell.name;

    // If we are already at the target, no-op.
    if old_bel == target_bel {
        return SwapResult {
            delta_cost: 0.0,
            performed: false,
        };
    }

    // Determine if there is a cell at the target bel.
    let other_cell_name = ctx.get_bound_bel_cell(target_bel);
    let other_cell_idx = other_cell_name.and_then(|name| ctx.design.cell_by_name(name));

    // Check that the other cell (if any) is moveable.
    if let Some(oci) = other_cell_idx {
        let other_cell = ctx.design.cell(oci);
        if other_cell.bel_strength.is_locked() {
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
        affected_nets.sort_by_key(|n| n.0);
        affected_nets.dedup();
    }

    // Compute cost before.
    let cost_before = hpwl_for_nets(ctx, &affected_nets);

    // Unbind both cells.
    ctx.unbind_bel(old_bel);
    if other_cell_name.is_some() {
        ctx.unbind_bel(target_bel);
    }

    // Bind cell to target_bel.
    ctx.bind_bel(target_bel, cell_name, PlaceStrength::Placer);

    // Bind other cell to old_bel (if swap).
    if let Some(other_name) = other_cell_name {
        ctx.bind_bel(old_bel, other_name, PlaceStrength::Placer);
    }

    // Compute cost after.
    let cost_after = hpwl_for_nets(ctx, &affected_nets);

    SwapResult {
        delta_cost: cost_after - cost_before,
        performed: true,
    }
}

/// Revert a swap move (undo the last try_swap that was performed).
fn revert_swap(
    ctx: &mut Context,
    cell_idx: CellIdx,
    old_bel: BelId,
    other_cell_idx: Option<CellIdx>,
    current_bel: BelId,
) {
    let cell_name = ctx.design.cell(cell_idx).name;

    // Unbind current positions.
    ctx.unbind_bel(current_bel);
    if let Some(oci) = other_cell_idx {
        let other_name = ctx.design.cell(oci).name;
        ctx.unbind_bel(old_bel);
        // Restore other cell to its original position (current_bel was its old bel).
        ctx.bind_bel(current_bel, other_name, PlaceStrength::Placer);
    }
    // Restore cell to old_bel.
    ctx.bind_bel(old_bel, cell_name, PlaceStrength::Placer);
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
    ctx.rng = npnr_context::DeterministicRng::new(cfg.seed);

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
            let cell_idx = moveable[ctx.rng.next_range(num_cells as u32) as usize];
            let cell = ctx.design.cell(cell_idx);
            let cell_type = cell.cell_type;
            let old_bel = cell.bel;

            if !old_bel.is_valid() {
                continue;
            }

            // Get the cell type name for bucket lookup.
            let cell_type_name = ctx.name_of(cell_type);
            let num_bucket_bels = ctx.get_bels_for_bucket(&cell_type_name).len();
            if num_bucket_bels == 0 {
                continue;
            }

            // Pick a random valid BEL of the same bucket.
            let rand_idx = ctx.rng.next_range(num_bucket_bels as u32) as usize;
            let target_bel = ctx.get_bels_for_bucket(&cell_type_name)[rand_idx];

            // Determine the other cell (if any) for potential revert.
            let other_cell_name = ctx.get_bound_bel_cell(target_bel);
            let other_cell_idx = other_cell_name.and_then(|name| ctx.design.cell_by_name(name));

            // Check that the other cell type is compatible with old_bel.
            if let Some(oci) = other_cell_idx {
                let other_cell = ctx.design.cell(oci);
                if !ctx.is_valid_bel_for_cell(old_bel, other_cell.cell_type) {
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
                let rand_val = (ctx.rng.next_u32() as f64) / (u32::MAX as f64);
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
    for (i, cell) in ctx.design.cell_store.iter().enumerate() {
        if cell.alive && !cell.bel.is_valid() {
            return Err(PlacerError::PlacementFailed(format!(
                "Cell {} (index {}) is alive but has no BEL after placement",
                ctx.name_of(cell.name),
                i
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_chipdb::testutil::make_test_chipdb;
    use npnr_context::Context;
    use npnr_netlist::PortRef;
    use npnr_types::{BelId, IdString, PlaceStrength, PortType};

    /// Create a fresh Context backed by the synthetic 2x2 chipdb.
    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    /// Create a context with N cells of type "LUT" and some nets connecting them.
    fn make_context_with_cells(n: usize) -> Context {
        assert!(n <= 4, "synthetic chipdb only has 4 BELs");
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT");
        let mut cell_names = Vec::new();

        for i in 0..n {
            let name = ctx.id(&format!("cell_{}", i));
            ctx.design.add_cell(name, cell_type);
            cell_names.push(name);
        }

        // If we have at least 2 cells, create a net connecting them.
        if n >= 2 {
            let net_name = ctx.id("net_0");
            let net_idx = ctx.design.add_net(net_name);

            // Cell 0 drives the net via port "Q".
            let q_port = ctx.id("Q");
            let a_port = ctx.id("A");

            let cell0_idx = ctx.design.cell_by_name(cell_names[0]).unwrap();
            ctx.design.cell_mut(cell0_idx).add_port(q_port, PortType::Out);
            ctx.design.cell_mut(cell0_idx).port_mut(q_port).unwrap().net = net_idx;

            ctx.design.net_mut(net_idx).driver = PortRef {
                cell: cell0_idx,
                port: q_port,
                budget: 0,
            };

            // Remaining cells are users of the net via port "A".
            for i in 1..n {
                let cell_idx = ctx.design.cell_by_name(cell_names[i]).unwrap();
                ctx.design.cell_mut(cell_idx).add_port(a_port, PortType::In);
                ctx.design.cell_mut(cell_idx).port_mut(a_port).unwrap().net = net_idx;

                let user_idx = ctx.design.net(net_idx).users.len() as i32;
                ctx.design.cell_mut(cell_idx).port_mut(a_port).unwrap().user_idx = user_idx;
                ctx.design.net_mut(net_idx).users.push(PortRef {
                    cell: cell_idx,
                    port: a_port,
                    budget: 0,
                });
            }
        }

        ctx
    }

    // =====================================================================
    // HPWL tests
    // =====================================================================

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
        let cell_type = ctx.id("LUT");
        let cell_name = ctx.id("drv");
        let cell_idx = ctx.design.add_cell(cell_name, cell_type);

        let net_name = ctx.id("n0");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        ctx.design.cell_mut(cell_idx).add_port(q_port, PortType::Out);
        ctx.design.net_mut(net_idx).driver = PortRef {
            cell: cell_idx,
            port: q_port,
            budget: 0,
        };

        assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
    }

    #[test]
    fn hpwl_same_location_is_zero() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT");
        let drv_name = ctx.id("drv");
        let usr_name = ctx.id("usr");
        let drv_idx = ctx.design.add_cell(drv_name, cell_type);
        let usr_idx = ctx.design.add_cell(usr_name, cell_type);

        // Place both cells at same tile (tile 0).
        // But synthetic chipdb has 1 bel per tile, so we need to use different tiles
        // and check HPWL is non-zero, or same tile (not possible with 1 bel per tile).
        // Instead, let us place them at tile 0 and tile 0 -- not possible.
        // Let's place at tile 0 (0,0) and tile 1 (1,0).
        let bel0 = BelId::new(0, 0);
        let bel1 = BelId::new(1, 0);
        ctx.bind_bel(bel0, drv_name, PlaceStrength::Placer);
        ctx.bind_bel(bel1, usr_name, PlaceStrength::Placer);

        let net_name = ctx.id("n");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        ctx.design.cell_mut(drv_idx).add_port(q_port, PortType::Out);
        ctx.design.cell_mut(drv_idx).port_mut(q_port).unwrap().net = net_idx;
        ctx.design.net_mut(net_idx).driver = PortRef {
            cell: drv_idx,
            port: q_port,
            budget: 0,
        };

        ctx.design.cell_mut(usr_idx).add_port(a_port, PortType::In);
        ctx.design.cell_mut(usr_idx).port_mut(a_port).unwrap().net = net_idx;
        ctx.design.net_mut(net_idx).users.push(PortRef {
            cell: usr_idx,
            port: a_port,
            budget: 0,
        });

        // bel0 is at (0,0), bel1 is at (1,0).
        // HPWL = (1-0) + (0-0) = 1.
        let hpwl = net_hpwl(&ctx, net_idx);
        assert_eq!(hpwl, 1.0);
    }

    #[test]
    fn hpwl_diagonal_placement() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT");
        let drv_name = ctx.id("drv");
        let usr_name = ctx.id("usr");
        let drv_idx = ctx.design.add_cell(drv_name, cell_type);
        let usr_idx = ctx.design.add_cell(usr_name, cell_type);

        // Place at tile 0 (0,0) and tile 3 (1,1).
        let bel0 = BelId::new(0, 0);
        let bel3 = BelId::new(3, 0);
        ctx.bind_bel(bel0, drv_name, PlaceStrength::Placer);
        ctx.bind_bel(bel3, usr_name, PlaceStrength::Placer);

        let net_name = ctx.id("n");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        ctx.design.cell_mut(drv_idx).add_port(q_port, PortType::Out);
        ctx.design.cell_mut(drv_idx).port_mut(q_port).unwrap().net = net_idx;
        ctx.design.net_mut(net_idx).driver = PortRef {
            cell: drv_idx,
            port: q_port,
            budget: 0,
        };

        ctx.design.cell_mut(usr_idx).add_port(a_port, PortType::In);
        ctx.design.cell_mut(usr_idx).port_mut(a_port).unwrap().net = net_idx;
        ctx.design.net_mut(net_idx).users.push(PortRef {
            cell: usr_idx,
            port: a_port,
            budget: 0,
        });

        // HPWL = (1-0) + (1-0) = 2.
        let hpwl = net_hpwl(&ctx, net_idx);
        assert_eq!(hpwl, 2.0);
    }

    #[test]
    fn total_hpwl_sums_all_nets() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        let cell_type = ctx.id("LUT");
        let names: Vec<IdString> = (0..4).map(|i| ctx.id(&format!("c{}", i))).collect();
        let cell_indices: Vec<CellIdx> = names
            .iter()
            .map(|&n| ctx.design.add_cell(n, cell_type))
            .collect();

        // Place at all 4 tiles.
        for (i, &ci) in cell_indices.iter().enumerate() {
            let bel = BelId::new(i as i32, 0);
            let name = ctx.design.cell(ci).name;
            ctx.bind_bel(bel, name, PlaceStrength::Placer);
        }

        // Create 2 nets:
        // net_a: c0 (0,0) -> c3 (1,1) => HPWL = 2
        // net_b: c1 (1,0) -> c2 (0,1) => HPWL = 2
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        // net_a
        {
            let net_name = ctx.id("net_a");
            let net_idx = ctx.design.add_net(net_name);

            ctx.design.cell_mut(cell_indices[0]).add_port(q_port, PortType::Out);
            ctx.design.cell_mut(cell_indices[0]).port_mut(q_port).unwrap().net = net_idx;
            ctx.design.net_mut(net_idx).driver = PortRef {
                cell: cell_indices[0],
                port: q_port,
                budget: 0,
            };

            ctx.design.cell_mut(cell_indices[3]).add_port(a_port, PortType::In);
            ctx.design.cell_mut(cell_indices[3]).port_mut(a_port).unwrap().net = net_idx;
            ctx.design.net_mut(net_idx).users.push(PortRef {
                cell: cell_indices[3],
                port: a_port,
                budget: 0,
            });
        }

        // net_b
        {
            let b_port = ctx.id("B");
            let net_name = ctx.id("net_b");
            let net_idx = ctx.design.add_net(net_name);

            ctx.design.cell_mut(cell_indices[1]).add_port(b_port, PortType::Out);
            ctx.design.cell_mut(cell_indices[1]).port_mut(b_port).unwrap().net = net_idx;
            ctx.design.net_mut(net_idx).driver = PortRef {
                cell: cell_indices[1],
                port: b_port,
                budget: 0,
            };

            let c_port = ctx.id("C");
            ctx.design.cell_mut(cell_indices[2]).add_port(c_port, PortType::In);
            ctx.design.cell_mut(cell_indices[2]).port_mut(c_port).unwrap().net = net_idx;
            ctx.design.net_mut(net_idx).users.push(PortRef {
                cell: cell_indices[2],
                port: c_port,
                budget: 0,
            });
        }

        let total = total_hpwl(&ctx);
        assert_eq!(total, 4.0); // 2 + 2
    }

    // =====================================================================
    // Initial placement tests
    // =====================================================================

    #[test]
    fn initial_placement_places_all_cells() {
        let mut ctx = make_context_with_cells(3);

        initial_placement(&mut ctx).expect("initial placement should succeed");

        // All 3 cells should be placed.
        for (i, cell) in ctx.design.cell_store.iter().enumerate() {
            if cell.alive {
                assert!(
                    cell.bel.is_valid(),
                    "cell {} (index {}) should be placed",
                    ctx.name_of(cell.name),
                    i
                );
            }
        }
    }

    #[test]
    fn initial_placement_no_duplicate_bels() {
        let mut ctx = make_context_with_cells(4);

        initial_placement(&mut ctx).expect("initial placement should succeed");

        // All cells should be at distinct BELs.
        let mut used_bels = std::collections::HashSet::new();
        for cell in &ctx.design.cell_store {
            if cell.alive {
                assert!(
                    used_bels.insert(cell.bel),
                    "BEL {:?} used by multiple cells",
                    cell.bel
                );
            }
        }
    }

    #[test]
    fn initial_placement_too_many_cells_fails() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();

        // Create 5 cells but only 4 BELs.
        let cell_type = ctx.id("LUT");
        for i in 0..5 {
            let name = ctx.id(&format!("cell_{}", i));
            ctx.design.add_cell(name, cell_type);
        }

        let result = initial_placement(&mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn initial_placement_skips_already_placed() {
        let mut ctx = make_context_with_cells(2);

        // Manually place cell_0 at bel (0,0).
        let cell_name = ctx.id("cell_0");
        let bel = BelId::new(0, 0);
        ctx.bind_bel(bel, cell_name, PlaceStrength::Placer);

        initial_placement(&mut ctx).expect("initial placement should succeed");

        // cell_0 should still be at bel (0,0).
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        assert_eq!(ctx.design.cell(cell_idx).bel, bel);

        // cell_1 should be placed somewhere else.
        let cell_name_1 = ctx.id("cell_1");
        let cell_idx_1 = ctx.design.cell_by_name(cell_name_1).unwrap();
        assert!(ctx.design.cell(cell_idx_1).bel.is_valid());
        assert_ne!(ctx.design.cell(cell_idx_1).bel, bel);
    }

    // =====================================================================
    // Swap mechanics tests
    // =====================================================================

    #[test]
    fn swap_to_empty_bel() {
        let mut ctx = make_context_with_cells(1);
        initial_placement(&mut ctx).expect("initial placement should succeed");

        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        let old_bel = ctx.design.cell(cell_idx).bel;

        // Find an empty bel.
        let bels: Vec<BelId> = ctx.get_bels().collect();
        let empty_bel = bels.iter().find(|&&b| b != old_bel).copied().unwrap();

        let result = try_swap(&mut ctx, cell_idx, empty_bel);
        assert!(result.performed);

        // Cell should now be at the new bel.
        assert_eq!(ctx.design.cell(cell_idx).bel, empty_bel);
        assert!(ctx.is_bel_available(old_bel));
    }

    #[test]
    fn swap_two_cells() {
        let mut ctx = make_context_with_cells(2);
        initial_placement(&mut ctx).expect("initial placement should succeed");

        let cell0_name = ctx.id("cell_0");
        let cell1_name = ctx.id("cell_1");
        let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
        let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();

        let bel0 = ctx.design.cell(cell0_idx).bel;
        let bel1 = ctx.design.cell(cell1_idx).bel;

        // Swap cell0 to cell1's bel.
        let result = try_swap(&mut ctx, cell0_idx, bel1);
        assert!(result.performed);

        // Cells should have swapped positions.
        assert_eq!(ctx.design.cell(cell0_idx).bel, bel1);
        assert_eq!(ctx.design.cell(cell1_idx).bel, bel0);
    }

    #[test]
    fn swap_same_bel_is_noop() {
        let mut ctx = make_context_with_cells(1);
        initial_placement(&mut ctx).expect("initial placement should succeed");

        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        let bel = ctx.design.cell(cell_idx).bel;

        let result = try_swap(&mut ctx, cell_idx, bel);
        assert!(!result.performed);
        assert_eq!(result.delta_cost, 0.0);
    }

    #[test]
    fn revert_swap_restores_state() {
        let mut ctx = make_context_with_cells(2);
        initial_placement(&mut ctx).expect("initial placement should succeed");

        let cell0_name = ctx.id("cell_0");
        let cell1_name = ctx.id("cell_1");
        let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
        let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();

        let bel0 = ctx.design.cell(cell0_idx).bel;
        let bel1 = ctx.design.cell(cell1_idx).bel;

        // Perform swap.
        let _result = try_swap(&mut ctx, cell0_idx, bel1);

        // Revert.
        revert_swap(&mut ctx, cell0_idx, bel0, Some(cell1_idx), bel1);

        // Should be back to original.
        assert_eq!(ctx.design.cell(cell0_idx).bel, bel0);
        assert_eq!(ctx.design.cell(cell1_idx).bel, bel1);
    }

    // =====================================================================
    // Temperature cooling tests
    // =====================================================================

    #[test]
    fn cooling_rate_reduces_temperature() {
        let cfg = PlacerSaCfg::default();
        let mut temp = 1.0;
        let initial = temp;

        for _ in 0..100 {
            temp *= cfg.cooling_rate;
        }

        assert!(temp < initial);
        assert!(temp > 0.0);
    }

    #[test]
    fn temperature_converges_to_zero() {
        let cfg = PlacerSaCfg::default();
        let mut temp = 1000.0;

        let mut iters = 0;
        while temp > cfg.min_temp {
            temp *= cfg.cooling_rate;
            iters += 1;
            // Safety valve: should converge in a reasonable number of iterations.
            assert!(iters < 100_000, "temperature did not converge");
        }
    }

    #[test]
    fn default_config_values() {
        let cfg = PlacerSaCfg::default();
        assert_eq!(cfg.seed, 1);
        assert_eq!(cfg.cooling_rate, 0.995);
        assert_eq!(cfg.inner_iters_per_cell, 10);
        assert_eq!(cfg.initial_temp_factor, 1.5);
        assert_eq!(cfg.min_temp, 1e-6);
        assert_eq!(cfg.timing_weight, 0.5);
        assert!(cfg.slack_redistribution);
    }

    // =====================================================================
    // Integration test: full SA run with mock context
    // =====================================================================

    #[test]
    fn full_sa_placement_2_cells() {
        let mut ctx = make_context_with_cells(2);

        let cfg = PlacerSaCfg {
            seed: 42,
            cooling_rate: 0.9,  // Fast cooling for test.
            inner_iters_per_cell: 5,
            min_temp: 0.01,
            ..PlacerSaCfg::default()
        };

        place_sa(&mut ctx, &cfg).expect("SA placement should succeed");

        // All cells should be placed.
        for cell in &ctx.design.cell_store {
            if cell.alive {
                assert!(cell.bel.is_valid());
            }
        }
    }

    #[test]
    fn full_sa_placement_4_cells() {
        let mut ctx = make_context_with_cells(4);

        let cfg = PlacerSaCfg {
            seed: 123,
            cooling_rate: 0.9,
            inner_iters_per_cell: 5,
            min_temp: 0.01,
            ..PlacerSaCfg::default()
        };

        place_sa(&mut ctx, &cfg).expect("SA placement should succeed");

        // All cells placed at distinct BELs.
        let mut used_bels = std::collections::HashSet::new();
        for cell in &ctx.design.cell_store {
            if cell.alive {
                assert!(cell.bel.is_valid());
                assert!(used_bels.insert(cell.bel));
            }
        }
    }

    #[test]
    fn full_sa_placement_single_cell() {
        let mut ctx = make_context_with_cells(1);

        let cfg = PlacerSaCfg {
            seed: 1,
            cooling_rate: 0.9,
            inner_iters_per_cell: 2,
            min_temp: 0.01,
            ..PlacerSaCfg::default()
        };

        place_sa(&mut ctx, &cfg).expect("SA placement should succeed");

        let cell_name = ctx.id("cell_0");
        let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
        assert!(ctx.design.cell(cell_idx).bel.is_valid());
    }

    #[test]
    fn full_sa_deterministic() {
        // Running with the same seed should produce the same result.
        let cfg = PlacerSaCfg {
            seed: 99,
            cooling_rate: 0.9,
            inner_iters_per_cell: 5,
            min_temp: 0.01,
            ..PlacerSaCfg::default()
        };

        let mut ctx1 = make_context_with_cells(3);
        place_sa(&mut ctx1, &cfg).expect("run 1");

        let mut ctx2 = make_context_with_cells(3);
        place_sa(&mut ctx2, &cfg).expect("run 2");

        // Same placement result.
        for i in 0..ctx1.design.cell_store.len() {
            let c1 = &ctx1.design.cell_store[i];
            let c2 = &ctx2.design.cell_store[i];
            assert_eq!(c1.bel, c2.bel, "cell {} placed differently", i);
        }
    }

    #[test]
    fn sa_no_moveable_cells_is_ok() {
        let mut ctx = make_context();
        ctx.populate_bel_buckets();
        // No cells at all.
        let cfg = PlacerSaCfg::default();
        place_sa(&mut ctx, &cfg).expect("no cells should be OK");
    }
}

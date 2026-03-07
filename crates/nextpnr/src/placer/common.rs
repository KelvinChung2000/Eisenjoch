//! Shared helper functions used by multiple placer implementations.

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use crate::types::{BelId, IdString, PlaceStrength};
use rustc_hash::FxHashMap;

use super::PlacerError;

/// Collect all live, placeable cell indices grouped by their cell type.
///
/// Returns a map from cell type IdString to the list of CellIdx values.
pub(crate) fn cells_by_type(ctx: &Context) -> FxHashMap<IdString, Vec<CellId>> {
    let mut map: FxHashMap<IdString, Vec<CellId>> = FxHashMap::default();
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        map.entry(cell.cell_type).or_default().push(cell_idx);
    }
    map
}

/// Compute HPWL for a single net.
///
/// HPWL = (max_x - min_x) + (max_y - min_y) across all connected cell locations.
/// Returns 0.0 for nets with no driver, no users, or dead nets.
pub fn net_hpwl(ctx: &Context, net_idx: NetId) -> f64 {
    let net = ctx.net(net_idx);
    if !net.is_alive() || net.driver().is_none() || net.users().is_empty() {
        return 0.0;
    }

    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;

    // Include driver location.
    let driver_cell_idx = match net.driver_cell_port() {
        Some(pin) => pin.cell,
        None => return 0.0,
    };
    let driver_cell = ctx.cell(driver_cell_idx);
    if let Some(bel) = driver_cell.bel() {
        let loc = bel.loc();
        min_x = min_x.min(loc.x);
        max_x = max_x.max(loc.x);
        min_y = min_y.min(loc.y);
        max_y = max_y.max(loc.y);
    }

    // Include all user locations.
    for user in net.users() {
        if !user.is_valid() {
            continue;
        }
        let user_cell_idx = user.cell;
        let user_cell = ctx.cell(user_cell_idx);
        if let Some(bel) = user_cell.bel() {
            let loc = bel.loc();
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
pub fn total_hpwl(ctx: &Context) -> f64 {
    let mut total = 0.0;
    for (net_idx, _) in ctx.design.iter_alive_nets() {
        total += net_hpwl(ctx, net_idx);
    }
    total
}

/// Place all unplaced cells at random valid BELs.
///
/// Groups cells by type/bucket, collects available BELs of the matching bucket,
/// shuffles the BELs, and assigns cells sequentially.
pub fn initial_placement(ctx: &mut Context) -> Result<(), PlacerError> {
    ctx.populate_bel_buckets();

    let grouped = cells_by_type(ctx);

    for (&cell_type, cell_indices) in &grouped {
        let cell_type_name = ctx.name_of(cell_type).to_owned();

        // Find all BELs matching this cell type's bucket.
        let bucket_bels: Vec<_> = ctx.bels_for_bucket(cell_type).map(|bel| bel.id()).collect();
        if bucket_bels.is_empty() {
            return Err(PlacerError::NoBelsAvailable(cell_type_name));
        }

        // Collect available BELs.
        let mut available: Vec<BelId> = bucket_bels
            .iter()
            .copied()
            .filter(|b| ctx.bel(*b).is_available())
            .collect();

        ctx.rng_mut().shuffle(&mut available);

        // Filter to only unplaced cells.
        let unplaced: Vec<CellId> = cell_indices
            .iter()
            .copied()
            .filter(|&ci| ctx.cell(ci).bel().is_none())
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
            if !ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer) {
                let cell_name = ctx.cell(cell_idx).name_id();
                return Err(PlacerError::InitialPlacementFailed(
                    ctx.name_of(cell_name).to_owned(),
                ));
            }
        }
    }

    Ok(())
}

/// Validate that all alive cells have been placed on a BEL.
pub(crate) fn validate_all_placed(ctx: &Context) -> Result<(), PlacerError> {
    for (cell_idx, cell) in ctx.design.iter_alive_cells() {
        if cell.bel.is_none() {
            return Err(PlacerError::PlacementFailed(format!(
                "Cell {} (index {}) is alive but has no BEL after placement",
                ctx.name_of(cell.name),
                cell_idx.slot()
            )));
        }
    }
    Ok(())
}

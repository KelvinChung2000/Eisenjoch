//! Type-aware snap legalization: snap + spread + BEL assignment in one pass.
//!
//! A complete Legalizer that:
//! 1. Snaps each cell to the nearest tile with compatible BELs (TypeAwarePlacement)
//! 2. Spreads overcrowded tiles to respect per-type BEL capacity
//! 3. Unbinds movable cells and assigns them to specific BELs at their snapped positions
//! 4. Places cluster children relative to root BELs

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use rustc_hash::FxHashMap;

use super::Legalizer;

/// Type-aware snap legalizer: snaps cells to valid tiles, spreads overcrowding,
/// and assigns BELs in a single pass.
pub struct SnapLegalizer;

impl Legalizer for SnapLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
        type_aware: &TypeAwarePlacement,
    ) -> Result<f64, PlacerError> {
        let t_start = std::time::Instant::now();
        let n = cell_x.len();

        // 1. Snap positions to valid tiles for each cell's type.
        let cell_buckets: Vec<IdString> = idx_to_cell
            .iter()
            .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
            .collect();

        let mut snapped_x = cell_x.to_vec();
        let mut snapped_y = cell_y.to_vec();
        let mut snap_max_disp = 0.0f64;
        let mut snap_total_disp = 0.0f64;

        for i in 0..n {
            let bucket = cell_buckets[i];
            let ox = snapped_x[i];
            let oy = snapped_y[i];
            snapped_x[i] = type_aware.snap_x(bucket, ox);
            let sx = snapped_x[i] as i32;
            snapped_y[i] = type_aware.snap_y(bucket, sx, oy);
            let d = ((snapped_x[i] - ox).powi(2) + (snapped_y[i] - oy).powi(2)).sqrt();
            snap_total_disp += d;
            snap_max_disp = snap_max_disp.max(d);
        }

        let snap_avg = snap_total_disp / n.max(1) as f64;

        // 2. Spread overcrowded tiles.
        let spread_count = spread_overcrowded(
            type_aware, &cell_buckets, &mut snapped_x, &mut snapped_y,
        );

        // 3. Unbind movable cells.
        unbind_movable_cells(ctx, idx_to_cell);

        // 4. Assign BELs at snapped positions.
        let mut total_displacement = 0.0f64;
        let mut assigned = 0usize;

        for i in 0..n {
            let cell_id = idx_to_cell[i];
            let bucket = cell_buckets[i];
            let tx = snapped_x[i].round() as i32;
            let ty = snapped_y[i].round() as i32;

            // Find an available BEL of the right type at or near (tx, ty).
            let bel = find_nearest_bel(ctx, bucket, tx, ty);
            match bel {
                Some(bel_id) => {
                    let loc = ctx.bel(bel_id).loc();
                    let dx = loc.x as f64 - cell_x[i];
                    let dy = loc.y as f64 - cell_y[i];
                    total_displacement += dx * dx + dy * dy;

                    ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer);
                    place_cluster_children(ctx, cell_id, bel_id)?;
                    assigned += 1;
                }
                None => {
                    return Err(PlacerError::NoBelsAvailable(format!(
                        "{} at ({}, {})",
                        ctx.name_of(bucket), tx, ty,
                    )));
                }
            }
        }

        let elapsed = t_start.elapsed().as_millis();
        eprintln!(
            "Snap legalize: {} cells, snap avg={:.1} max={:.1}, spread={}, assign={}, {}ms",
            n, snap_avg, snap_max_disp, spread_count, assigned, elapsed,
        );

        Ok(total_displacement)
    }
}

/// Find the nearest available BEL of the given type, expanding Manhattan rings from (tx, ty).
fn find_nearest_bel(ctx: &Context, bucket: IdString, tx: i32, ty: i32) -> Option<BelId> {
    // Collect all available BELs for this type, grouped by (x, y).
    // This is O(n_bels) per call — for small designs it's fine.
    // For large designs, TypeAwarePlacement could cache BEL lists per tile.
    let max_radius: i32 = 20;
    for radius in 0..=max_radius {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                if dx.abs() + dy.abs() != radius { continue; } // Manhattan ring
                let bx = tx + dx;
                let by = ty + dy;
                // Check BELs at this tile.
                for bel in ctx.bels_for_bucket(bucket) {
                    let loc = bel.loc();
                    if loc.x == bx && loc.y == by && bel.is_available() {
                        return Some(bel.id());
                    }
                }
            }
        }
    }
    None
}

/// Spread overcrowded tiles. Returns number of cells moved.
fn spread_overcrowded(
    type_aware: &TypeAwarePlacement,
    cell_buckets: &[IdString],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
) -> usize {
    let n = cell_x.len();

    let mut type_tile_cells: FxHashMap<(IdString, i32, i32), Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        let tx = cell_x[i].round() as i32;
        let ty = cell_y[i].round() as i32;
        type_tile_cells
            .entry((cell_buckets[i], tx, ty))
            .or_default()
            .push(i);
    }

    let mut remaining_cap: FxHashMap<(IdString, i32, i32), u32> = FxHashMap::default();
    for (bucket, cap_map) in &type_aware.tile_capacity {
        for (&(vx, vy), &cap) in cap_map {
            remaining_cap.insert((*bucket, vx, vy), cap);
        }
    }

    let mut spread_count = 0usize;

    let mut groups: Vec<_> = type_tile_cells.into_iter().collect();
    groups.sort_by_key(|(_, cells)| std::cmp::Reverse(cells.len()));

    for ((bucket, tx, ty), cells) in &groups {
        let cap = remaining_cap
            .get(&(*bucket, *tx, *ty))
            .copied()
            .unwrap_or(0) as usize;

        if cells.len() <= cap {
            if let Some(c) = remaining_cap.get_mut(&(*bucket, *tx, *ty)) {
                *c = c.saturating_sub(cells.len() as u32);
            }
            continue;
        }

        let mut cell_dists: Vec<(usize, f64)> = cells
            .iter()
            .map(|&ci| {
                let dx = cell_x[ci] - *tx as f64;
                let dy = cell_y[ci] - *ty as f64;
                (ci, dx * dx + dy * dy)
            })
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        remaining_cap.insert((*bucket, *tx, *ty), 0);

        for &(ci, _) in cell_dists.iter().skip(cap) {
            let mut best: Option<(i32, i32, f64)> = None;
            for (&(bt, bx, by), &cap_left) in &remaining_cap {
                if bt != *bucket || cap_left == 0 { continue; }
                let dx = (bx - tx) as f64;
                let dy = (by - ty) as f64;
                let d = dx * dx + dy * dy;
                if best.is_none() || d < best.unwrap().2 {
                    best = Some((bx, by, d));
                }
            }
            if let Some((bx, by, _)) = best {
                cell_x[ci] = bx as f64;
                cell_y[ci] = by as f64;
                *remaining_cap.get_mut(&(*bucket, bx, by)).unwrap() -= 1;
                spread_count += 1;
            }
        }
    }

    spread_count
}

// Keep the old function signature for backward compatibility.
pub fn snap_to_clb_grid(
    ctx: &Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    type_aware: &TypeAwarePlacement,
) -> (Vec<f64>, Vec<f64>) {
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();

    let mut snapped_x = cell_x.to_vec();
    let mut snapped_y = cell_y.to_vec();

    for i in 0..cell_x.len() {
        let bucket = cell_buckets[i];
        snapped_x[i] = type_aware.snap_x(bucket, snapped_x[i]);
        let sx = snapped_x[i] as i32;
        snapped_y[i] = type_aware.snap_y(bucket, sx, snapped_y[i]);
    }

    spread_overcrowded(type_aware, &cell_buckets, &mut snapped_x, &mut snapped_y);

    (snapped_x, snapped_y)
}

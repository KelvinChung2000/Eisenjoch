//! Snap continuous positions to nearest valid tile for each cell's type.
//!
//! Uses TypeAwarePlacement for per-cell-type BEL compatibility: each cell
//! snaps to the nearest tile that has compatible BELs. Overcrowded tiles
//! are spread using per-type capacity from the chipdb.

use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;

use rustc_hash::FxHashMap;

/// Snap continuous physical positions to the nearest valid tile for each cell's type.
///
/// Returns (snapped_x, snapped_y) in physical coordinates.
/// Also spreads overcrowded tiles using per-type BEL capacity.
pub fn snap_to_clb_grid(
    ctx: &Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = cell_x.len();
    let mut snapped_x = cell_x.to_vec();
    let mut snapped_y = cell_y.to_vec();

    // Build type-aware placement info (x0=0, y0=0 since positions are physical).
    let type_aware = TypeAwarePlacement::build(ctx, 0, 0);

    // Resolve each cell's bucket type.
    let cell_buckets: Vec<_> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();

    // Snap each cell to nearest valid (x, y) for its type.
    let mut snap_max_disp = 0.0f64;
    let mut snap_total_disp = 0.0f64;
    let mut n_valid_columns = 0usize;

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

    // Count valid columns for reporting.
    for xs in type_aware.valid_xs.values() {
        n_valid_columns = n_valid_columns.max(xs.len());
    }

    let snap_avg = snap_total_disp / n.max(1) as f64;
    eprintln!(
        "Snapped {} cells (type-aware, {} columns, avg_disp={:.1}, max_disp={:.1})",
        n, n_valid_columns, snap_avg, snap_max_disp,
    );

    // Spread overcrowded tiles using per-type capacity.
    spread_overcrowded(&type_aware, &cell_buckets, &mut snapped_x, &mut snapped_y);

    (snapped_x, snapped_y)
}

/// Move excess cells from overcrowded tiles to the nearest tile with capacity.
///
/// Uses TypeAwarePlacement's per-type tile capacity for accurate overflow detection.
fn spread_overcrowded(
    type_aware: &TypeAwarePlacement,
    cell_buckets: &[crate::common::IdString],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
) {
    let n = cell_x.len();

    // Group cells by (bucket, tile_x, tile_y).
    let mut type_tile_cells: FxHashMap<(crate::common::IdString, i32, i32), Vec<usize>> =
        FxHashMap::default();
    for i in 0..n {
        let tx = cell_x[i].round() as i32;
        let ty = cell_y[i].round() as i32;
        type_tile_cells
            .entry((cell_buckets[i], tx, ty))
            .or_default()
            .push(i);
    }

    // Build remaining capacity from TypeAwarePlacement.
    let mut remaining_cap: FxHashMap<(crate::common::IdString, i32, i32), u32> =
        FxHashMap::default();
    for (bucket, cap_map) in &type_aware.tile_capacity {
        for (&(vx, vy), &cap) in cap_map {
            remaining_cap.insert((*bucket, vx, vy), cap);
        }
    }

    let mut spread_count = 0usize;
    let mut spread_max_disp = 0.0f64;

    // Process most crowded tiles first.
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

        // Keep `cap` closest cells, move the rest.
        let mut cell_dists: Vec<(usize, f64)> = cells
            .iter()
            .map(|&ci| {
                let dx = cell_x[ci] - *tx as f64;
                let dy = cell_y[ci] - *ty as f64;
                (ci, dx * dx + dy * dy)
            })
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Reserve capacity for kept cells.
        remaining_cap.insert((*bucket, *tx, *ty), 0);

        for &(ci, _) in cell_dists.iter().skip(cap) {
            // Find nearest tile with remaining capacity for this type.
            let mut best: Option<(i32, i32, f64)> = None;
            for (&(bt, bx, by), &cap_left) in &remaining_cap {
                if bt != *bucket || cap_left == 0 {
                    continue;
                }
                let dx = (bx - tx) as f64;
                let dy = (by - ty) as f64;
                let d = dx * dx + dy * dy;
                if best.is_none() || d < best.unwrap().2 {
                    best = Some((bx, by, d));
                }
            }
            if let Some((bx, by, d)) = best {
                cell_x[ci] = bx as f64;
                cell_y[ci] = by as f64;
                *remaining_cap.get_mut(&(*bucket, bx, by)).unwrap() -= 1;
                spread_count += 1;
                spread_max_disp = spread_max_disp.max(d.sqrt());
            }
        }
    }

    if spread_count > 0 {
        eprintln!(
            "Spread {} cells from overcrowded tiles (max_disp={:.1})",
            spread_count, spread_max_disp,
        );
    }
}

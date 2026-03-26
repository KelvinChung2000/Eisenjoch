//! BEL legalization for the Beckmann optimal transport placer.
//!
//! - `RingLegalizer`: greedy nearest-available-BEL with expanding Manhattan rings
//! - `snap_to_clb_grid`: snap continuous positions to nearest CLB column/row

use std::collections::HashSet;

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use rustc_hash::FxHashMap;

use super::network::PipeNetwork;

/// Manhattan ring search legalization.
pub struct RingLegalizer {
    pub x_offset: f64,
    pub y_offset: f64,
}

impl crate::placer::legalize::Legalizer for RingLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
    ) -> Result<f64, PlacerError> {
        let phys_x: Vec<f64> = cell_x.iter().map(|&x| x + self.x_offset).collect();
        let phys_y: Vec<f64> = cell_y.iter().map(|&y| y + self.y_offset).collect();
        legalize_ring(ctx, idx_to_cell, &phys_x, &phys_y)
    }
}

/// Greedy ring search legalization with physical coordinates.
///
/// Assigns each cell to nearest available BEL using expanding Manhattan-distance
/// rings. `phys_x` and `phys_y` must be in physical (chip) coordinates.
pub fn legalize_ring(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    phys_x: &[f64],
    phys_y: &[f64],
) -> Result<f64, PlacerError> {
    let t_start = std::time::Instant::now();

    unbind_movable_cells(ctx, idx_to_cell);

    // Group cells by type.
    let mut groups: FxHashMap<IdString, Vec<usize>> = FxHashMap::default();
    for (solver_idx, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        groups.entry(cell_type).or_default().push(solver_idx);
    }

    let mut total_displacement = 0.0;

    for (&cell_type, cell_indices) in &groups {
        let n_cells = cell_indices.len();

        let mut bels_by_pos: FxHashMap<(i32, i32), Vec<BelId>> = FxHashMap::default();
        for b in ctx.bels_for_bucket(cell_type).filter(|b| b.is_available()) {
            let loc = b.loc();
            bels_by_pos.entry((loc.x, loc.y)).or_default().push(b.id());
        }

        let n_bels: usize = bels_by_pos.values().map(|v| v.len()).sum();
        if n_bels < n_cells {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (need {} BELs but only {} available)",
                ctx.name_of(cell_type),
                n_cells,
                n_bels,
            )));
        }

        let mut all_positions: Vec<(i32, i32)> = bels_by_pos.keys().copied().collect();
        all_positions.sort_unstable();

        // Sort cells by distance to nearest BEL (closest first).
        let mut sorted_cells: Vec<(usize, f64)> = cell_indices
            .iter()
            .map(|&si| {
                let tx = phys_x[si].round() as i32;
                let ty = phys_y[si].round() as i32;
                let min_dist = all_positions
                    .iter()
                    .map(|&(bx, by)| {
                        let dx = (bx - tx) as f64;
                        let dy = (by - ty) as f64;
                        dx * dx + dy * dy
                    })
                    .fold(f64::MAX, f64::min);
                (si, min_dist)
            })
            .collect();
        sorted_cells.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let max_search_radius = 200;
        for &(solver_idx, _) in &sorted_cells {
            let tx = phys_x[solver_idx].round() as i32;
            let ty = phys_y[solver_idx].round() as i32;
            let cell_id = idx_to_cell[solver_idx];

            let mut best_bel: Option<(BelId, i32, i32, f64)> = None;

            'search: for radius in 0..=max_search_radius {
                for dx in -(radius as i32)..=(radius as i32) {
                    let dy_abs = radius as i32 - dx.abs();
                    for &dy in &[dy_abs, -dy_abs] {
                        let px = tx + dx;
                        let py = ty + dy;
                        if let Some(bel_list) = bels_by_pos.get(&(px, py)) {
                            if !bel_list.is_empty() {
                                let dist = (px as f64 - phys_x[solver_idx]).powi(2)
                                    + (py as f64 - phys_y[solver_idx]).powi(2);
                                if best_bel.is_none() || dist < best_bel.as_ref().unwrap().3 {
                                    best_bel =
                                        Some((*bel_list.last().unwrap(), px, py, dist));
                                }
                            }
                        }
                        if dy_abs == 0 {
                            break;
                        }
                    }
                }
                if best_bel.is_some() {
                    break 'search;
                }
            }

            let (bel_id, bx, by, _dist) = best_bel.ok_or_else(|| {
                PlacerError::PlacementFailed(format!(
                    "No available BEL found for cell {} of type {} near ({}, {})",
                    ctx.name_of(ctx.design.cell(cell_id).name),
                    ctx.name_of(cell_type),
                    tx,
                    ty,
                ))
            })?;

            let bel_list = bels_by_pos.get_mut(&(bx, by)).unwrap();
            bel_list.pop();

            if !ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer) {
                let cell_name = ctx.design.cell(cell_id).name;
                return Err(PlacerError::PlacementFailed(format!(
                    "Failed to bind cell {} during ring legalization",
                    ctx.name_of(cell_name),
                )));
            }

            let dx = bx as f64 - phys_x[solver_idx];
            let dy = by as f64 - phys_y[solver_idx];
            total_displacement += dx * dx + dy * dy;

            place_cluster_children(ctx, cell_id, bel_id)?;
        }
    }

    eprintln!(
        "  Legalization: {:.0}ms",
        t_start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(total_displacement)
}

/// Snap continuous positions to the nearest CLB column and row.
///
/// The continuous placer works on the full tile grid, but BELs only exist
/// on CLB tiles. Snapping reduces legalization displacement significantly.
///
/// Also performs per-type capacity spreading to avoid overcrowding at
/// individual CLB tiles.
pub fn snap_to_clb_grid(
    ctx: &Context,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    idx_to_cell: &[CellId],
    network: &PipeNetwork,
) {
    let chipdb = ctx.chipdb();
    let net_w = network.width;
    let net_h = network.height;
    let x0 = network.x0;
    let y0 = network.y0;

    // Build sorted list of CLB x-columns and y-rows.
    let mut clb_xs: Vec<f64> = Vec::new();
    let mut clb_ys: Vec<f64> = Vec::new();
    let mut seen_x = HashSet::new();
    let mut seen_y = HashSet::new();
    for vy in 0..net_h {
        for vx in 0..net_w {
            let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
            let bel_count = chipdb.tile_type(tile).bels.len();
            if bel_count >= 4 {
                if seen_x.insert(vx) {
                    clb_xs.push(vx as f64);
                }
                if seen_y.insert(vy) {
                    clb_ys.push(vy as f64);
                }
            }
        }
    }
    clb_xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    clb_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if clb_xs.is_empty() || clb_ys.is_empty() {
        return;
    }

    let n = cell_x.len();
    let mut snap_max_disp = 0.0f64;
    let mut snap_total_disp = 0.0f64;

    for i in 0..n {
        let cx = cell_x[i];
        let best_x = clb_xs
            .iter()
            .copied()
            .min_by(|a, b| (a - cx).abs().partial_cmp(&(b - cx).abs()).unwrap())
            .unwrap();
        cell_x[i] = best_x;

        let cy = cell_y[i];
        let best_y = clb_ys
            .iter()
            .copied()
            .min_by(|a, b| (a - cy).abs().partial_cmp(&(b - cy).abs()).unwrap())
            .unwrap();
        cell_y[i] = best_y;

        let d = ((best_x - cx).powi(2) + (best_y - cy).powi(2)).sqrt();
        snap_total_disp += d;
        snap_max_disp = snap_max_disp.max(d);
    }

    let snap_avg = snap_total_disp / n.max(1) as f64;
    eprintln!(
        "Snapped {} cells to {} CLB columns x {} CLB rows (avg_disp={:.1}, max_disp={:.1})",
        n,
        clb_xs.len(),
        clb_ys.len(),
        snap_avg,
        snap_max_disp,
    );

    // Per-type capacity spreading: ensure no tile has more cells than BELs.
    spread_overcrowded_cells(ctx, cell_x, cell_y, idx_to_cell, network, &clb_xs, &clb_ys);
}

/// Move excess cells from overcrowded tiles to the nearest tile with capacity.
fn spread_overcrowded_cells(
    ctx: &Context,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    idx_to_cell: &[CellId],
    network: &PipeNetwork,
    clb_xs: &[f64],
    clb_ys: &[f64],
) {
    use rustc_hash::FxHashSet;

    let chipdb = ctx.chipdb();
    let x0 = network.x0;
    let y0 = network.y0;
    let n = cell_x.len();

    // Count distinct cell types.
    let mut cell_types_present: FxHashSet<IdString> = FxHashSet::default();
    for i in 0..n {
        let ct = ctx.design.cell(idx_to_cell[i]).cell_type;
        cell_types_present.insert(ct);
    }
    let n_types = cell_types_present.len().max(1);

    // Per-tile capacity = total BELs / number of cell types (fair share).
    let mut tile_cap: FxHashMap<(i32, i32), usize> = FxHashMap::default();
    for &cx in clb_xs {
        for &cy in clb_ys {
            let vx = cx as i32;
            let vy = cy as i32;
            let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
            let total_bels = chipdb.tile_type(tile).bels.len();
            tile_cap.insert((vx, vy), total_bels / n_types);
        }
    }

    // Group cells by (type, snap position).
    let mut type_tile_cells: FxHashMap<(IdString, i32, i32), Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        let ct = ctx.design.cell(idx_to_cell[i]).cell_type;
        let tx = cell_x[i].round() as i32;
        let ty = cell_y[i].round() as i32;
        type_tile_cells.entry((ct, tx, ty)).or_default().push(i);
    }

    // Per-type remaining capacity.
    let mut remaining_cap: FxHashMap<(IdString, i32, i32), usize> = FxHashMap::default();
    for &ct in &cell_types_present {
        for (&(vx, vy), &cap) in &tile_cap {
            remaining_cap.insert((ct, vx, vy), cap);
        }
    }

    let mut spread_count = 0usize;
    let mut spread_max_disp = 0.0f64;

    let mut groups: Vec<_> = type_tile_cells.into_iter().collect();
    groups.sort_by_key(|(_, cells)| std::cmp::Reverse(cells.len()));

    for ((ct, tx, ty), cells) in &groups {
        let cap = remaining_cap.get(&(*ct, *tx, *ty)).copied().unwrap_or(0);
        if cells.len() <= cap {
            if let Some(c) = remaining_cap.get_mut(&(*ct, *tx, *ty)) {
                *c = c.saturating_sub(cells.len());
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

        remaining_cap.insert((*ct, *tx, *ty), 0);

        for &(ci, _) in cell_dists.iter().skip(cap) {
            let mut best: Option<(i32, i32, f64)> = None;
            for (&(bt, bx, by), cap_left) in &remaining_cap {
                if bt != *ct || *cap_left == 0 {
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
                *remaining_cap.get_mut(&(*ct, bx, by)).unwrap() -= 1;
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

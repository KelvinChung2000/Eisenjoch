//! BEL legalization for the optimal transport placer.
//!
//! Uses greedy nearest-available-BEL assignment after CLB snapping.
//! With low overflow and pre-snapped positions, most cells land on their
//! nearest BEL directly. Conflicts are resolved by displacement search
//! in expanding rings around the target position.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::placer::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;

use rustc_hash::FxHashMap;

use super::state::OptTransState;

/// Greedy legalization: assign each cell to nearest available BEL.
///
/// Cells are processed in priority order (most constrained first: cells
/// closest to their target get placed first to avoid displacement).
/// For each cell, search expanding rings around its snapped position
/// for an available BEL of the matching type.
pub fn legalize_opt_trans(
    ctx: &mut Context,
    state: &OptTransState,
    _lap_max_cells: usize,
) -> Result<f64, PlacerError> {
    let t_start = std::time::Instant::now();

    // Convert virtual grid positions to physical for BEL matching.
    let x0 = state.network.x0 as f64;
    let y0 = state.network.y0 as f64;
    let phys_x: Vec<f64> = state.cell_x.iter().map(|&x| x + x0).collect();
    let phys_y: Vec<f64> = state.cell_y.iter().map(|&y| y + y0).collect();

    unbind_movable_cells(ctx, &state.idx_to_cell);

    // Group cells by type.
    let mut groups: FxHashMap<IdString, Vec<usize>> = FxHashMap::default();
    for (solver_idx, &cell_id) in state.idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        groups.entry(cell_type).or_default().push(solver_idx);
    }

    let mut total_displacement = 0.0;

    for (&cell_type, cell_indices) in &groups {
        let n_cells = cell_indices.len();

        // Collect all available BELs for this cell type, indexed by (x, y).
        let mut bels_by_pos: FxHashMap<(i32, i32), Vec<BelId>> = FxHashMap::default();
        for b in ctx.bels_for_bucket(cell_type).filter(|b| b.is_available()) {
            let loc = b.loc();
            bels_by_pos.entry((loc.x, loc.y)).or_default().push(b.id());
        }

        let n_bels: usize = bels_by_pos.values().map(|v| v.len()).sum();
        if n_bels < n_cells {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (need {} BELs but only {} available)",
                ctx.name_of(cell_type), n_cells, n_bels,
            )));
        }

        // Collect all BEL positions for spatial search.
        let mut all_positions: Vec<(i32, i32)> = bels_by_pos.keys().copied().collect();
        all_positions.sort_unstable();

        // Sort cells closest-to-target first. Cells that are already near a
        // BEL get placed first, preserving the continuous solver's quality.
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

        // Greedy assignment: for each cell, find nearest available BEL.
        let max_search_radius = 200; // tiles
        for &(solver_idx, _) in &sorted_cells {
            let tx = phys_x[solver_idx].round() as i32;
            let ty = phys_y[solver_idx].round() as i32;
            let cell_id = state.idx_to_cell[solver_idx];

            // Search in expanding Manhattan-distance rings.
            let mut best_bel: Option<(BelId, i32, i32, f64)> = None;

            'search: for radius in 0..=max_search_radius {
                // Check all positions at exactly this Manhattan distance.
                for dx in -(radius as i32)..=(radius as i32) {
                    let dy_abs = radius as i32 - dx.abs();
                    for &dy in &[dy_abs, -dy_abs] {
                        let px = tx + dx;
                        let py = ty + dy;
                        if let Some(bel_list) = bels_by_pos.get(&(px, py)) {
                            if !bel_list.is_empty() {
                                let dist = (px as f64 - phys_x[solver_idx]).powi(2)
                                    + (py as f64 - phys_y[solver_idx]).powi(2);
                                if best_bel.is_none()
                                    || dist < best_bel.as_ref().unwrap().3
                                {
                                    best_bel =
                                        Some((*bel_list.last().unwrap(), px, py, dist));
                                }
                            }
                        }
                        // Avoid double-processing when dy_abs == 0.
                        if dy_abs == 0 {
                            break;
                        }
                    }
                }
                // If we found a BEL at this radius, no need to search further.
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

            // Remove this BEL from the available pool.
            let bel_list = bels_by_pos.get_mut(&(bx, by)).unwrap();
            bel_list.pop(); // We used last(), so pop removes it.

            if !ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer) {
                let cell_name = ctx.design.cell(cell_id).name;
                return Err(PlacerError::PlacementFailed(format!(
                    "Failed to bind cell {} during greedy legalization",
                    ctx.name_of(cell_name),
                )));
            }

            let dx = bx as f64 - phys_x[solver_idx];
            let dy = by as f64 - phys_y[solver_idx];
            let d = (dx * dx + dy * dy).sqrt();
            total_displacement += dx * dx + dy * dy;

            if d > 5.0 {
                eprintln!(
                    "    OUTLIER: {} disp={:.1} target=({:.1},{:.1}) bel=({},{})",
                    ctx.name_of(ctx.design.cell(cell_id).name),
                    d, phys_x[solver_idx], phys_y[solver_idx], bx, by,
                );
            }

            place_cluster_children(ctx, cell_id, bel_id)?;
        }

        // Per-type displacement stats.
        let mut type_max_disp = 0.0f64;
        let mut type_total_disp = 0.0f64;
        for &(si, _) in &sorted_cells {
            let bx = phys_x[si].round() as i32;
            let by = phys_y[si].round() as i32;
            if let Some(bel_id) = ctx.cell(state.idx_to_cell[si]).bel_id() {
                let loc = ctx.chipdb().bel_loc(bel_id);
                let dx = (loc.x - bx) as f64;
                let dy = (loc.y - by) as f64;
                let d = (dx * dx + dy * dy).sqrt();
                type_total_disp += d;
                type_max_disp = type_max_disp.max(d);
            }
        }
        let type_avg = if n_cells > 0 { type_total_disp / n_cells as f64 } else { 0.0 };

        let t_group = std::time::Instant::now();
        eprintln!(
            "  Legalize {}: {} cells → {} BELs in {:.0}ms (avg_disp={:.1}, max_disp={:.1})",
            ctx.name_of(cell_type),
            n_cells,
            n_bels,
            (t_group - t_start).as_secs_f64() * 1000.0,
            type_avg,
            type_max_disp,
        );
    }

    let t_end = std::time::Instant::now();
    eprintln!(
        "  Legalization total: {:.0}ms",
        (t_end - t_start).as_secs_f64() * 1000.0,
    );

    Ok(total_displacement)
}

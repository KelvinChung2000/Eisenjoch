//! Manhattan ring search legalization.
//!
//! Greedy nearest-available-BEL with expanding Manhattan rings.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::legalize::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::common::TypeAwarePlacement;
use crate::placer::PlacerError;

use rustc_hash::FxHashMap;

use super::Legalizer;

/// Manhattan ring search legalization.
pub struct RingLegalizer {
    pub x_offset: f64,
    pub y_offset: f64,
}

impl Legalizer for RingLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
        _type_aware: &TypeAwarePlacement,
    ) -> Result<f64, PlacerError> {
        let phys_x: Vec<f64> = cell_x.iter().map(|&x| x + self.x_offset).collect();
        let phys_y: Vec<f64> = cell_y.iter().map(|&y| y + self.y_offset).collect();
        legalize_ring(ctx, idx_to_cell, &phys_x, &phys_y, _type_aware)
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
    type_aware: &TypeAwarePlacement,
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
        let bucket = ctx.resolve_bucket(cell_type);

        // Use tile_capacity from TypeAwarePlacement for the capacity check.
        let total_capacity: u32 = type_aware
            .tile_capacity
            .get(&bucket)
            .map(|cap_map| cap_map.values().sum())
            .unwrap_or(0);
        if (total_capacity as usize) < n_cells {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (need {} BELs but only {} total capacity)",
                ctx.name_of(cell_type),
                n_cells,
                total_capacity,
            )));
        }

        // Build per-position BEL lists for the ring search assignment.
        let mut bels_by_pos: FxHashMap<(i32, i32), Vec<BelId>> = FxHashMap::default();
        for b in ctx.bels_for_bucket(cell_type).filter(|b| b.is_available()) {
            let loc = b.loc();
            bels_by_pos.entry((loc.x, loc.y)).or_default().push(b.id());
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

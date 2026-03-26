//! Parallel-sorted nearest-BEL legalization with region support.
//!
//! Phase A (parallel): pre-sort BEL candidates by distance per cell.
//! Phase B (sequential): greedy assignment, outer cells first.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::legalize::common::{place_cluster_children, unbind_movable_cells};
use crate::placer::PlacerError;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::Legalizer;

/// Parallel-sorted nearest-BEL legalization with region support.
///
/// Phase A (parallel): pre-sort BEL candidates by distance per cell.
/// Phase B (sequential): greedy assignment, outer cells first.
pub struct SortedLegalizer;

impl Legalizer for SortedLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
    ) -> Result<f64, PlacerError> {
        sorted_legalize(ctx, idx_to_cell, cell_x, cell_y)
    }
}

/// Per-cell info gathered before the parallel sort phase.
struct CellLegalizeInfo {
    cell_idx: CellId,
    cell_type_id: IdString,
    cell_type_name: String,
    cell_name: String,
    cell_region: Option<u32>,
    target_x: f64,
    target_y: f64,
    /// True when this cell is a cluster child (placed by its root).
    is_cluster_child: bool,
}

/// Parallel-sorted nearest-BEL legalization.
///
/// 1. Unbinds all movable cells.
/// 2. Builds BEL candidate lists per cell type.
/// 3. Sorts candidates by distance to target position (parallel, per cell).
/// 4. Assigns cells sequentially (outer-first), skipping cluster children.
/// 5. Calls `place_cluster_children` for cluster roots.
///
/// Returns total squared displacement.
pub fn sorted_legalize(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> Result<f64, PlacerError> {
    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Unbind all movable cells.
    unbind_movable_cells(ctx, idx_to_cell);

    // Pre-collect BEL data per cell type into plain data (BelId, x, y)
    // so we can share across rayon threads without lifetime issues.
    let mut bel_data_cache: FxHashMap<IdString, Vec<(BelId, i32, i32)>> = FxHashMap::default();
    for &cell_idx in idx_to_cell {
        let cell_type_id = ctx.cell(cell_idx).cell_type_id();
        bel_data_cache.entry(cell_type_id).or_insert_with(|| {
            ctx.bels_for_bucket(cell_type_id)
                .map(|bel| {
                    let loc = bel.loc();
                    (bel.id(), loc.x, loc.y)
                })
                .collect()
        });
    }

    // Sort movable cells by distance from center (place outer cells first).
    let grid_w = ctx.chipdb().width();
    let grid_h = ctx.chipdb().height();
    let cx = (grid_w as f64 - 1.0) / 2.0;
    let cy = (grid_h as f64 - 1.0) / 2.0;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let da = (cell_x[a] - cx).powi(2) + (cell_y[a] - cy).powi(2);
        let db = (cell_x[b] - cx).powi(2) + (cell_y[b] - cy).powi(2);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Gather per-cell info for parallel phase.
    // Read region from the cell directly via ctx.design.
    let cell_infos: Vec<CellLegalizeInfo> = order
        .iter()
        .map(|&idx| {
            let cell_idx = idx_to_cell[idx];
            let cell = ctx.cell(cell_idx);
            let is_cluster_child = cell
                .cluster()
                .map_or(false, |root_id| root_id != cell_idx);
            CellLegalizeInfo {
                cell_idx,
                cell_type_id: cell.cell_type_id(),
                cell_type_name: cell.cell_type().to_owned(),
                cell_name: cell.name().to_owned(),
                cell_region: ctx.design.cell(cell_idx).region,
                target_x: cell_x[idx],
                target_y: cell_y[idx],
                is_cluster_child,
            }
        })
        .collect();

    // Pre-compute region BEL sets for region-constrained cells.
    let region_bel_sets: FxHashMap<u32, FxHashSet<BelId>> = {
        let mut map = FxHashMap::default();
        for info in &cell_infos {
            if let Some(rid) = info.cell_region {
                map.entry(rid).or_insert_with(|| {
                    let region = ctx.design.region(rid);
                    let mut set = FxHashSet::default();
                    if let Some(bbox) = region.bounding_box() {
                        for bel in ctx.bels() {
                            let loc = bel.loc();
                            if region.contains(loc.x, loc.y)
                                && loc.x >= bbox.x0
                                && loc.x <= bbox.x1
                                && loc.y >= bbox.y0
                                && loc.y <= bbox.y1
                            {
                                set.insert(bel.id());
                            }
                        }
                    }
                    set
                });
            }
        }
        map
    };

    // Phase A (parallel): compute distance-sorted BEL candidate lists.
    let sorted_candidates: Vec<Vec<BelId>> = cell_infos
        .par_iter()
        .map(|info| {
            let bels = match bel_data_cache.get(&info.cell_type_id) {
                Some(b) => b,
                None => return Vec::new(),
            };

            let mut candidates: Vec<(BelId, f64)> = bels
                .iter()
                .filter(|&&(bel_id, _, _)| {
                    if let Some(rid) = info.cell_region {
                        region_bel_sets
                            .get(&rid)
                            .map_or(false, |s| s.contains(&bel_id))
                    } else {
                        true
                    }
                })
                .map(|&(bel_id, bx, by)| {
                    let dx = bx as f64 - info.target_x;
                    let dy = by as f64 - info.target_y;
                    (bel_id, dx * dx + dy * dy)
                })
                .collect();

            candidates
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.into_iter().map(|(id, _)| id).collect()
        })
        .collect();

    // Phase B (sequential): assign cells to nearest available BEL.
    let mut total_displacement = 0.0;

    for (i, info) in cell_infos.iter().enumerate() {
        if info.is_cluster_child {
            continue;
        }

        let candidates = &sorted_candidates[i];

        if candidates.is_empty() {
            return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
        }

        let mut bound = false;
        for &bel in candidates {
            if ctx.bel(bel).is_available() {
                if !ctx.bind_bel(bel, info.cell_idx, PlaceStrength::Placer) {
                    return Err(PlacerError::PlacementFailed(format!(
                        "Failed to bind cell {} to BEL {}",
                        info.cell_name, bel,
                    )));
                }
                let loc = ctx.bel(bel).loc();
                let dx = loc.x as f64 - info.target_x;
                let dy = loc.y as f64 - info.target_y;
                total_displacement += dx * dx + dy * dy;
                place_cluster_children(ctx, info.cell_idx, bel)?;
                bound = true;
                break;
            }
        }

        if !bound {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (no available BELs for cell {})",
                info.cell_type_name, info.cell_name,
            )));
        }
    }

    Ok(total_displacement)
}

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::metrics::{compute_net_demand, compute_tile_capacities};
use crate::netlist::CellId;
use crate::placer::legalize::common::place_cluster_children;
use crate::placer::PlacerError;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::state::HeapState;

impl HeapState {
    /// Legalize the placement: assign each movable cell to the nearest
    /// available BEL of matching bucket type.
    ///
    /// Two-phase approach:
    ///   Phase A (parallel): compute distance-sorted BEL candidate lists per cell
    ///   Phase B (sequential): assign cells to BELs, skipping already-taken ones
    pub(super) fn legalize(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(());
        }

        // First, unbind all movable cells.
        for &cell_idx in &self.movable_cells {
            if let Some(bel_id) = ctx.cell(cell_idx).bel().map(|b| b.id()) {
                ctx.unbind_bel(bel_id);
            }
        }

        // Pre-collect BEL data per cell type into plain data (BelId, x, y)
        // so we can share across rayon threads without lifetime issues.
        let mut bel_data_cache: FxHashMap<IdString, Vec<(BelId, i32, i32)>> =
            FxHashMap::default();
        for &cell_idx in &self.movable_cells {
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
        let cx = (self.grid_w as f64 - 1.0) / 2.0;
        let cy = (self.grid_h as f64 - 1.0) / 2.0;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let da = (self.cell_x[a] - cx).powi(2) + (self.cell_y[a] - cy).powi(2);
            let db = (self.cell_x[b] - cx).powi(2) + (self.cell_y[b] - cy).powi(2);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Gather per-cell info for parallel phase.
        struct CellLegalizeInfo {
            idx: usize,
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

        let cell_infos: Vec<CellLegalizeInfo> = order
            .iter()
            .map(|&idx| {
                let cell_idx = self.movable_cells[idx];
                let cell = ctx.cell(cell_idx);
                let is_cluster_child = cell
                    .cluster()
                    .map_or(false, |root_id| root_id != cell_idx);
                CellLegalizeInfo {
                    idx,
                    cell_idx,
                    cell_type_id: cell.cell_type_id(),
                    cell_type_name: cell.cell_type().to_owned(),
                    cell_name: cell.name().to_owned(),
                    cell_region: self.cell_region[idx],
                    target_x: self.cell_x[idx],
                    target_y: self.cell_y[idx],
                    is_cluster_child,
                }
            })
            .collect();

        // Phase A (parallel): compute distance-sorted BEL candidate lists.
        // Each candidate list is sorted by distance to the cell's target position.
        // Region filtering is applied here using precomputed region data.
        let region_bel_sets: FxHashMap<u32, FxHashSet<BelId>> = {
            let mut map = FxHashMap::default();
            for info in &cell_infos {
                if let Some(rid) = info.cell_region {
                    map.entry(rid).or_insert_with(|| {
                        let region = ctx.design.region(rid);
                        let mut set = FxHashSet::default();
                        if let Some(bbox) = region.bounding_box() {
                            // Collect all BELs in the region.
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

        let sorted_candidates: Vec<Vec<BelId>> = cell_infos
            .par_iter()
            .map(|info| {
                let bels = match bel_data_cache.get(&info.cell_type_id) {
                    Some(b) => b,
                    None => return Vec::new(),
                };

                // Filter by region if needed, then sort by distance.
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

                candidates.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.into_iter().map(|(id, _)| id).collect()
            })
            .collect();

        // Phase B (sequential): assign cells to nearest available BEL.
        // Cluster children are skipped here; they are placed by
        // `place_cluster_children` after their root is bound.
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
                    self.cell_x[info.idx] = loc.x as f64;
                    self.cell_y[info.idx] = loc.y as f64;
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

        Ok(())
    }

    /// Compute congestion-aware displacement targets for each movable cell.
    ///
    /// For each cell, estimates the local congestion gradient from the edge-demand
    /// grid and computes a target position shifted away from congested edges.
    /// Returns (target_x, target_y, force_weight) for each movable cell.
    pub(super) fn compute_congestion_targets(&self, ctx: &Context) -> Vec<(f64, f64, f64)> {
        let n = self.movable_cells.len();
        let grid_w = self.grid_w;
        let grid_h = self.grid_h;
        let wu = grid_w as usize;
        let hu = grid_h as usize;
        let congestion_weight = self.cfg.congestion_weight;

        // Build capacity grids (total_wires / 4 per direction).
        let (h_capacity, v_capacity) = compute_tile_capacities(ctx);

        // Build demand grids by iterating all alive nets and tracing Bresenham lines.
        let (h_demand, v_demand) = compute_net_demand(ctx);

        // Compute per-cell congestion displacement targets.
        let max_x = (grid_w - 1) as f64;
        let max_y = (grid_h - 1) as f64;

        (0..n)
            .map(|i| {
                let cx = self.cell_x[i];
                let cy = self.cell_y[i];
                let ix = cx.round() as i32;
                let iy = cy.round() as i32;

                // Get congestion at surrounding edges (0.0 if at boundary).
                let east_c =
                    if ix >= 0 && (ix as usize) + 1 < wu && iy >= 0 && (iy as usize) < hu {
                        h_demand[iy as usize][ix as usize]
                            / h_capacity[iy as usize][ix as usize]
                    } else {
                        0.0
                    };
                let west_c =
                    if ix > 0 && (ix as usize) < wu && iy >= 0 && (iy as usize) < hu {
                        h_demand[iy as usize][(ix - 1) as usize]
                            / h_capacity[iy as usize][(ix - 1) as usize]
                    } else {
                        0.0
                    };
                let south_c =
                    if iy >= 0 && (iy as usize) + 1 < hu && ix >= 0 && (ix as usize) < wu {
                        v_demand[iy as usize][ix as usize]
                            / v_capacity[iy as usize][ix as usize]
                    } else {
                        0.0
                    };
                let north_c =
                    if iy > 0 && (iy as usize) < hu && ix >= 0 && (ix as usize) < wu {
                        v_demand[(iy - 1) as usize][ix as usize]
                            / v_capacity[(iy - 1) as usize][ix as usize]
                    } else {
                        0.0
                    };

                // Displacement: push away from congested edges.
                let dx = west_c - east_c; // positive = push east (away from west congestion)
                let dy = north_c - south_c; // positive = push south (away from north congestion)
                let max_c = east_c.max(west_c).max(south_c).max(north_c);

                // Only apply force if congestion is above 1.0 (over-capacity).
                if max_c > 1.0 {
                    let target_x = (cx + dx).clamp(0.0, max_x);
                    let target_y = (cy + dy).clamp(0.0, max_y);
                    let force = self.alpha * congestion_weight * (max_c - 1.0);
                    (target_x, target_y, force)
                } else {
                    (cx, cy, 0.0)
                }
            })
            .collect()
    }
}

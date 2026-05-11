//! Manhattan ring search legalization.
//!
//! Greedy nearest-available-BEL with expanding Manhattan rings.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::common::{
    build_bel_by_loc, build_cell_legality_cache, cluster_is_legal_cached,
    place_cluster_children, unbind_movable_cells, CellLegalityCache, DriverNodeRegistry,
};
use crate::placer::legalize::snap::snap_to_clb_grid;
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

    // Pre-spread: snap to valid bucket positions and distribute overflow tiles
    // before ring search. On collapsed DCD output (FPGA01: cells co-located),
    // this drops shared-mux rejections by orders of magnitude — the ring
    // search starts at a tile that already has free capacity instead of at
    // the over-saturated centroid.  Disable with NPNR_RING_PRESPREAD=0.
    let t_phase = std::time::Instant::now();
    let prespread_enabled = std::env::var("NPNR_RING_PRESPREAD")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    let (phys_x_owned, phys_y_owned, prespread_used) = if prespread_enabled {
        let (sx, sy) = snap_to_clb_grid(ctx, idx_to_cell, phys_x, phys_y, type_aware);
        (sx, sy, true)
    } else {
        (phys_x.to_vec(), phys_y.to_vec(), false)
    };
    let phys_x: &[f64] = &phys_x_owned;
    let phys_y: &[f64] = &phys_y_owned;
    let t_prespread_ms = t_phase.elapsed().as_millis();

    let t_phase = std::time::Instant::now();
    unbind_movable_cells(ctx, idx_to_cell);
    let t_unbind_ms = t_phase.elapsed().as_millis();

    // (bucket, x, y, z) -> BelId index built once; `cluster_is_legal` consults
    // it for each cluster child. Without this, every cluster legality probe
    // did `bels_for_bucket(child_type).find(...)` on a 500k-1M-element bucket
    // iterator, dominating ring-legalize wall time.
    let t_phase = std::time::Instant::now();
    let bel_by_loc = build_bel_by_loc(ctx);
    let t_belloc_ms = t_phase.elapsed().as_millis();

    // Seed shared-mux registry from already-bound cells (packer-placed
    // fixed cells: BUFG, IO, clock buffers). Movable cells were just
    // unbound so they don't contribute.
    let t_phase = std::time::Instant::now();
    let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
    let t_seed_ms = t_phase.elapsed().as_millis();

    // Per-cell pin-port templates built once, reused for every BEL candidate.
    // Pre-fix this was an O(n_ports) cell.ports walk *per* `cluster_is_legal`
    // call (~800 M calls on FPGA01); now it's an O(1) HashMap fetch.
    let t_phase = std::time::Instant::now();
    let cell_caches: rustc_hash::FxHashMap<CellId, CellLegalityCache> = idx_to_cell
        .iter()
        .copied()
        .map(|cid| (cid, build_cell_legality_cache(ctx, cid)))
        .collect();
    let t_cache_ms = t_phase.elapsed().as_millis();

    let mut rejected_for_shared_mux: u64 = 0;
    let mut probe_islegal_us: u128 = 0;
    let mut probe_record_us: u128 = 0;
    let mut probe_ring_search_us: u128 = 0;
    let mut probe_islegal_calls: u64 = 0;
    let mut probe_record_calls: u64 = 0;

    // Group cells by type.
    let mut groups: FxHashMap<IdString, Vec<usize>> = FxHashMap::default();
    for (solver_idx, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        groups.entry(cell_type).or_default().push(solver_idx);
    }

    let mut total_displacement = 0.0;
    let t_assign_start = std::time::Instant::now();
    let mut probe_assign_phase_ms: u128 = 0;
    let mut probe_islegal_only_us: u128 = 0;
    let mut probe_islegal_sample_count: u64 = 0;

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
            // VCC_DRV/GND_DRV are switch-matrix constants with no BEL bucket
            // entries in `tile_capacity`; the router pulls them from PIP
            // tieoffs. Skip placement for them; only error when other types
            // truly lack capacity.
            if total_capacity == 0 {
                continue;
            }
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

        // Sort cells by distance to nearest BEL (closest first). Computing
        // the true nearest-BEL distance is O(cells × positions), which is
        // billions of ops on FPGA01. Bucket all_positions onto a coarse
        // grid and probe only nearby buckets — orders of magnitude faster
        // and the sort key only needs to be approximately monotone.
        let mut sorted_cells: Vec<(usize, f64)> = {
            const GRID: i32 = 8;
            let mut grid: FxHashMap<(i32, i32), Vec<(i32, i32)>> = FxHashMap::default();
            for &(bx, by) in &all_positions {
                grid.entry((bx / GRID, by / GRID)).or_default().push((bx, by));
            }
            cell_indices
                .iter()
                .map(|&si| {
                    let tx = phys_x[si].round() as i32;
                    let ty = phys_y[si].round() as i32;
                    let cx = tx / GRID;
                    let cy = ty / GRID;
                    let mut best = f64::MAX;
                    'r: for r in 0i32..=8 {
                        for dx in -r..=r {
                            for dy in -r..=r {
                                if dx.abs() != r && dy.abs() != r {
                                    continue;
                                }
                                if let Some(bucket) = grid.get(&(cx + dx, cy + dy)) {
                                    for &(bx, by) in bucket {
                                        let ddx = (bx - tx) as f64;
                                        let ddy = (by - ty) as f64;
                                        let d = ddx * ddx + ddy * ddy;
                                        if d < best {
                                            best = d;
                                        }
                                    }
                                }
                            }
                        }
                        if best.is_finite() {
                            break 'r;
                        }
                    }
                    (si, best)
                })
                .collect()
        };
        sorted_cells.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let max_search_radius = 200;
        for &(solver_idx, _) in &sorted_cells {
            let tx = phys_x[solver_idx].round() as i32;
            let ty = phys_y[solver_idx].round() as i32;
            let cell_id = idx_to_cell[solver_idx];

            // Best candidate across the whole search: (bel, idx_in_bel_list, x, y, sq_dist).
            let mut best: Option<(BelId, usize, i32, i32, f64)> = None;

            'search: for radius in 0..=max_search_radius {
                for dx in -(radius as i32)..=(radius as i32) {
                    let dy_abs = radius as i32 - dx.abs();
                    for &dy in &[dy_abs, -dy_abs] {
                        let px = tx + dx;
                        let py = ty + dy;
                        if let Some(bel_list) = bels_by_pos.get(&(px, py)) {
                            // Try each BEL at this position (newest first so
                            // the popped list semantics match the old code),
                            // keeping the first that is legal w.r.t. the
                            // shared-mux constraint.
                            let mut legal_pick: Option<(BelId, usize)> = None;
                            // Sample-based timing: time only every 4096th call
                            // so per-call Instant::now() overhead doesn't
                            // distort the measurement. Each call is sub-µs
                            // and 800M calls × 100ns sampler = 80s of pure
                            // timing overhead if we measured every call.
                            let cache = cell_caches.get(&cell_id);
                            for (i, &bid) in bel_list.iter().enumerate().rev() {
                                probe_islegal_calls += 1;
                                let sample = (probe_islegal_calls & 0xFFF) == 0;
                                let legal = if let Some(cache) = cache {
                                    if sample {
                                        let t = std::time::Instant::now();
                                        let r = cluster_is_legal_cached(
                                            ctx, &registry, &bel_by_loc, cache, bid,
                                        );
                                        probe_islegal_only_us +=
                                            t.elapsed().as_nanos() as u128;
                                        probe_islegal_sample_count += 1;
                                        r
                                    } else {
                                        cluster_is_legal_cached(
                                            ctx, &registry, &bel_by_loc, cache, bid,
                                        )
                                    }
                                } else {
                                    false
                                };
                                if legal {
                                    legal_pick = Some((bid, i));
                                    break;
                                } else {
                                    rejected_for_shared_mux += 1;
                                }
                            }
                            if let Some((bid, bi)) = legal_pick {
                                let dist = (px as f64 - phys_x[solver_idx]).powi(2)
                                    + (py as f64 - phys_y[solver_idx]).powi(2);
                                if best.is_none() || dist < best.as_ref().unwrap().4 {
                                    best = Some((bid, bi, px, py, dist));
                                }
                            }
                        }
                        if dy_abs == 0 {
                            break;
                        }
                    }
                }
                if best.is_some() {
                    break 'search;
                }
            }

            let (bel_id, bi, bx, by, _dist) = best.ok_or_else(|| {
                PlacerError::PlacementFailed(format!(
                    "No legal BEL found for cell {} of type {} near ({}, {}) \
                     (shared-mux constraints: {} rejections)",
                    ctx.name_of(ctx.design.cell(cell_id).name),
                    ctx.name_of(cell_type),
                    tx,
                    ty,
                    rejected_for_shared_mux,
                ))
            })?;

            let bel_list = bels_by_pos.get_mut(&(bx, by)).unwrap();
            bel_list.swap_remove(bi);

            if !ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer) {
                let cell_name = ctx.design.cell(cell_id).name;
                return Err(PlacerError::PlacementFailed(format!(
                    "Failed to bind cell {} during ring legalization",
                    ctx.name_of(cell_name),
                )));
            }
            probe_record_calls += 1;
            registry.record(ctx, cell_id, bel_id);

            let dx = bx as f64 - phys_x[solver_idx];
            let dy = by as f64 - phys_y[solver_idx];
            total_displacement += dx * dx + dy * dy;

            place_cluster_children(ctx, &bel_by_loc, cell_id, bel_id)?;

            // Cluster children are bound via place_cluster_children; register
            // their driver wires too so subsequent cells in the outer loop
            // see the updated occupancy.
            if let Some(cluster) = ctx.design.clusters.get(&cell_id) {
                let children: Vec<CellId> = cluster.constr_children.clone();
                for child_id in children {
                    if let Some(child_bel) = ctx.design.cell(child_id).bel {
                        probe_record_calls += 1;
                        registry.record(ctx, child_id, child_bel);
                    }
                }
            }
        }
    }

    probe_assign_phase_ms = t_assign_start.elapsed().as_millis();
    let n = idx_to_cell.len() as f64;
    let rms_disp = (total_displacement / n.max(1.0)).sqrt();
    let _max_disp = total_displacement;
    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  Legalization: {:.0}ms, rms_disp={:.2}, total_sq_disp={:.1}, shared_mux_rejects={}",
        total_ms, rms_disp, total_displacement, rejected_for_shared_mux,
    );
    // Estimated total cluster_is_legal time = avg sampled call ns × total calls.
    let est_islegal_ns = if probe_islegal_sample_count == 0 {
        0u128
    } else {
        (probe_islegal_only_us / probe_islegal_sample_count as u128) * probe_islegal_calls as u128
    };
    eprintln!(
        "  ring_phase_ms: prespread={} unbind={} bel_by_loc={} seed={} cache={} assign_loop={} (total={}); est_islegal_ms={} prespread_used={}",
        t_prespread_ms,
        t_unbind_ms,
        t_belloc_ms,
        t_seed_ms,
        t_cache_ms,
        probe_assign_phase_ms,
        total_ms as u128,
        est_islegal_ns / 1_000_000,
        prespread_used,
    );
    eprintln!(
        "  ring_probe: islegal_calls={} record_calls={} avg_sampled_islegal_ns={}",
        probe_islegal_calls,
        probe_record_calls,
        if probe_islegal_sample_count == 0 { 0 } else { probe_islegal_only_us / probe_islegal_sample_count as u128 },
    );
    let _ = probe_ring_search_us;
    let _ = probe_islegal_us;
    let _ = probe_record_us;

    Ok(total_displacement)
}

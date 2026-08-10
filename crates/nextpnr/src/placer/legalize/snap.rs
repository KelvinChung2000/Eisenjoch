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
use crate::placer::legalize::common::{
    build_bel_by_loc, cluster_is_legal, place_cluster_children, unbind_movable_cells,
    DriverNodeRegistry,
};
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

        // 2. Spread overcrowded tiles, routing evictions toward each cell's
        //    pin centroid so cells land aligned with their net structure
        //    rather than at arbitrary first-free ring slots.
        let spread_count = spread_overcrowded(
            ctx,
            idx_to_cell,
            type_aware,
            &cell_buckets,
            &mut snapped_x,
            &mut snapped_y,
        );

        // 3. Unbind movable cells.
        unbind_movable_cells(ctx, idx_to_cell);

        // (bucket, x, y, z) -> BelId index for cluster_is_legal child slot
        // resolution.
        let bel_by_loc = build_bel_by_loc(ctx);

        // Seed shared-mux registry from cells still bound (packer-fixed
        // BUFG/IO/clock buffers). Movable cells were just unbound so they
        // don't contribute to seed claims.
        let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
        let mut rejected_for_shared_mux: u64 = 0;

        // 4. Build a per-bucket index of available BELs grouped by tile.
        // Walk every active bucket once instead of `bels_for_bucket()`-per-cell.
        let mut active_buckets: rustc_hash::FxHashSet<IdString> = rustc_hash::FxHashSet::default();
        for &b in &cell_buckets {
            active_buckets.insert(b);
        }
        let mut bels_by_tile: FxHashMap<(IdString, i32, i32), Vec<BelId>> = FxHashMap::default();
        for &bucket in &active_buckets {
            for bel in ctx.bels_for_bucket(bucket).filter(|b| b.is_available()) {
                let loc = bel.loc();
                bels_by_tile
                    .entry((bucket, loc.x, loc.y))
                    .or_default()
                    .push(bel.id());
            }
        }

        // 5. Assign BELs at snapped positions.
        // Pass A: cluster roots — pick a tile/z where all constrained child
        // slots are also free, otherwise the post-bind `place_cluster_children`
        // call cannot satisfy the offset constraint.
        // Pass B: non-cluster cells — pop any available BEL at the snapped tile.
        let mut total_displacement = 0.0f64;
        let mut assigned = 0usize;
        let max_radius: i32 = 200;

        let order: Vec<usize> = {
            let mut roots: Vec<usize> = Vec::new();
            let mut leaves: Vec<usize> = Vec::new();
            for i in 0..n {
                let cid = idx_to_cell[i];
                let cell = ctx.design.cell(cid);
                // Cluster child: bound later by `place_cluster_children`
                // when its root is processed. Skip here.
                if let Some(root_id) = cell.cluster {
                    if root_id != cid {
                        continue;
                    }
                }
                if ctx
                    .design
                    .clusters
                    .get(&cid)
                    .map_or(false, |c| !c.constr_children.is_empty())
                {
                    roots.push(i);
                } else {
                    leaves.push(i);
                }
            }
            roots.extend(leaves);
            roots
        };

        for i in order {
            let cell_id = idx_to_cell[i];
            let bucket = cell_buckets[i];
            let tx = snapped_x[i].round() as i32;
            let ty = snapped_y[i].round() as i32;

            let has_cluster_children = ctx
                .design
                .clusters
                .get(&cell_id)
                .map_or(false, |c| !c.constr_children.is_empty());

            // Pop the first available BEL on the smallest Manhattan ring,
            // honoring cluster-child slot reservation when applicable.
            let mut found: Option<(BelId, i32, i32)> = None;
            'r: for radius in 0..=max_radius {
                for dx in -radius..=radius {
                    let dy_abs = radius - dx.abs();
                    for &dy in &[dy_abs, -dy_abs] {
                        let bx = tx + dx;
                        let by = ty + dy;
                        if let Some(list) = bels_by_tile.get_mut(&(bucket, bx, by)) {
                            // Scan the BEL list at this tile for a candidate
                            // that is (a) still available, (b) shared-mux legal
                            // against already-placed cells, and (c) for cluster
                            // roots, has every child slot available and legal.
                            let mut pick_idx: Option<usize> = None;
                            for (idx, &bid) in list.iter().enumerate() {
                                if !ctx.bel(bid).is_available() {
                                    continue;
                                }
                                if has_cluster_children {
                                    if !cluster_is_legal(ctx, &registry, &bel_by_loc, cell_id, bid)
                                    {
                                        rejected_for_shared_mux += 1;
                                        continue;
                                    }
                                } else {
                                    if !registry.is_legal(ctx, cell_id, bid) {
                                        rejected_for_shared_mux += 1;
                                        continue;
                                    }
                                }
                                pick_idx = Some(idx);
                                break;
                            }
                            if let Some(idx) = pick_idx {
                                let bid = list.swap_remove(idx);
                                found = Some((bid, bx, by));
                                break 'r;
                            }
                        }
                        if dy_abs == 0 {
                            break;
                        }
                    }
                }
            }

            match found {
                Some((bel_id, bx, by)) => {
                    let dx = bx as f64 - cell_x[i];
                    let dy = by as f64 - cell_y[i];
                    total_displacement += dx * dx + dy * dy;

                    if !ctx.bind_bel(bel_id, cell_id, PlaceStrength::Placer) {
                        let cell_name = ctx.design.cell(cell_id).name;
                        return Err(PlacerError::PlacementFailed(format!(
                            "Failed to bind cell {} during snap legalization",
                            ctx.name_of(cell_name),
                        )));
                    }
                    registry.record(ctx, cell_id, bel_id);
                    place_cluster_children(ctx, &bel_by_loc, cell_id, bel_id)?;
                    // Record cluster children that were just bound so that
                    // later cells see their pin-wire claims.
                    if let Some(cluster) = ctx.design.clusters.get(&cell_id) {
                        let children: Vec<CellId> = cluster.constr_children.clone();
                        for child_id in children {
                            if let Some(child_bel) = ctx.design.cell(child_id).bel {
                                registry.record(ctx, child_id, child_bel);
                            }
                        }
                    }
                    assigned += 1;
                }
                None => {
                    return Err(PlacerError::NoBelsAvailable(format!(
                        "{} at ({}, {}) (shared-mux rejections: {})",
                        ctx.name_of(bucket),
                        tx,
                        ty,
                        rejected_for_shared_mux,
                    )));
                }
            }
        }

        let elapsed = t_start.elapsed().as_millis();
        let rms_disp = (total_displacement / n.max(1) as f64).sqrt();
        eprintln!(
            "Snap legalize: {} cells, snap avg={:.1} max={:.1}, spread={}, assign={}, rms_disp={:.1}, shared_mux_rejects={}, {}ms",
            n, snap_avg, snap_max_disp, spread_count, assigned, rms_disp, rejected_for_shared_mux, elapsed,
        );

        Ok(total_displacement)
    }
}

/// Per-cell "pin centroid": average position of the other pins on every net
/// the cell is on. Fixed cells (already bound to a BEL — IOs, BUFGs) appear
/// at their BEL position and serve as anchors that diversify the centroids
/// of cells that DCD has piled together. Movable cells without a cell_to_idx
/// entry but with a cluster root fall back to the root's snapped position
/// plus their constraint offset.
///
/// Returns (centroid_x, centroid_y, has_centroid). For cells with no usable
/// neighbour positions, `has_centroid` is false and the caller should fall
/// back to the snapped position.
fn compute_pin_centroids(
    ctx: &Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let n = idx_to_cell.len();
    let cell_to_idx: FxHashMap<CellId, usize> = idx_to_cell
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let pin_pos = |cid: CellId| -> Option<(f64, f64)> {
        if let Some(&idx) = cell_to_idx.get(&cid) {
            return Some((cell_x[idx], cell_y[idx]));
        }
        let cell = ctx.design.cell(cid);
        if let Some(root_id) = cell.cluster {
            if root_id != cid {
                if let Some(&ridx) = cell_to_idx.get(&root_id) {
                    return Some((
                        cell_x[ridx] + cell.constr_x as f64,
                        cell_y[ridx] + cell.constr_y as f64,
                    ));
                }
            }
        }
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            return Some((loc.x as f64, loc.y as f64));
        }
        None
    };

    let mut sum_x = vec![0.0f64; n];
    let mut sum_y = vec![0.0f64; n];
    let mut count = vec![0u32; n];

    let mut pins: Vec<(CellId, f64, f64)> = Vec::new();
    for (_, net) in ctx.design.iter_alive_nets() {
        if net.users().is_empty() {
            continue;
        }
        pins.clear();
        if let Some(dp) = net.driver() {
            if let Some((x, y)) = pin_pos(dp.cell) {
                pins.push((dp.cell, x, y));
            }
        }
        for u in net.users() {
            if !u.is_valid() {
                continue;
            }
            if let Some((x, y)) = pin_pos(u.cell) {
                pins.push((u.cell, x, y));
            }
        }
        if pins.len() < 2 {
            continue;
        }

        let nsx: f64 = pins.iter().map(|(_, x, _)| *x).sum();
        let nsy: f64 = pins.iter().map(|(_, _, y)| *y).sum();
        let ncnt = pins.len() as u32;

        for &(cid, x, y) in &pins {
            if let Some(&idx) = cell_to_idx.get(&cid) {
                sum_x[idx] += nsx - x;
                sum_y[idx] += nsy - y;
                count[idx] += ncnt - 1;
            }
        }
    }

    let mut cx = vec![0.0f64; n];
    let mut cy = vec![0.0f64; n];
    let mut has = vec![false; n];
    for i in 0..n {
        if count[i] > 0 {
            let c = count[i] as f64;
            cx[i] = sum_x[i] / c;
            cy[i] = sum_y[i] / c;
            has[i] = true;
        }
    }
    (cx, cy, has)
}

/// Spread overcrowded tiles. Returns number of cells moved.
///
/// Net-aware: each over-capacity (bucket, tile) group keeps the `cap` cells
/// whose pin centroid sits closest to the tile, and evicts the rest. Each
/// evicted cell ring-searches outward starting from ITS OWN pin centroid
/// rather than the pile tile, so cells land aligned with their net structure
/// even when DCD has piled them at one tile.
fn spread_overcrowded(
    ctx: &Context,
    idx_to_cell: &[CellId],
    type_aware: &TypeAwarePlacement,
    cell_buckets: &[IdString],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
) -> usize {
    let n = cell_x.len();

    let (centroid_x, centroid_y, has_centroid) =
        compute_pin_centroids(ctx, idx_to_cell, cell_x, cell_y);

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
    let mut spread_dist_sum = 0i64;
    let mut spread_dist_max = 0i32;

    let mut groups: Vec<_> = type_tile_cells.into_iter().collect();
    groups.sort_by_key(|(_, cells)| std::cmp::Reverse(cells.len()));

    if let Some(((_bucket, tx, ty), cells)) = groups.first() {
        let total_overflow: usize = groups.iter().map(|(_, c)| c.len()).sum();
        let n_groups = groups.len();
        let big_groups = groups.iter().take(5);
        let top: Vec<(i32, i32, usize)> =
            big_groups.map(|((_, x, y), c)| (*x, *y, c.len())).collect();
        eprintln!(
            "  pile distribution: groups={} total_cells={} biggest=({},{}, size={}) top5={:?}",
            n_groups,
            total_overflow,
            tx,
            ty,
            cells.len(),
            top,
        );
    }

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

        // Score each cell by squared distance from its pin centroid to this
        // tile. Cells whose net structure agrees with this tile (low score)
        // stay; cells pulled elsewhere (high score) get evicted first and
        // route to their own pin centroid.
        let tx_f = *tx as f64;
        let ty_f = *ty as f64;
        let mut cell_dists: Vec<(usize, f64)> = cells
            .iter()
            .map(|&ci| {
                let (ex, ey) = if has_centroid[ci] {
                    (centroid_x[ci], centroid_y[ci])
                } else {
                    (cell_x[ci], cell_y[ci])
                };
                let dx = ex - tx_f;
                let dy = ey - ty_f;
                (ci, dx * dx + dy * dy)
            })
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        remaining_cap.insert((*bucket, *tx, *ty), 0);

        // Expand Manhattan rings outward from the SOURCE PILE TILE (preserves
        // the geometric minimum eviction radius — disk fills cap-by-cap).
        // Within each ring, gather every (bx, by) with remaining capacity and
        // pick the one closest to the cell's pin centroid. This bounds
        // displacement by the pile-fill geometry while still routing each
        // cell toward its actual net structure (HPWL-optimal among slots at
        // the same Manhattan distance from the pile).
        const MAX_SPREAD_RADIUS: i32 = 200;
        let src_tx = *tx;
        let src_ty = *ty;
        for &(ci, _) in cell_dists.iter().skip(cap) {
            let (anchor_cx, anchor_cy) = if has_centroid[ci] {
                (centroid_x[ci], centroid_y[ci])
            } else {
                (cell_x[ci], cell_y[ci])
            };

            let mut ring_candidates: Vec<(i32, i32)> = Vec::new();
            let mut placed = false;
            'r: for radius in 1..=MAX_SPREAD_RADIUS {
                ring_candidates.clear();
                for dx in -radius..=radius {
                    let dy_abs = radius - dx.abs();
                    for &dy in &[dy_abs, -dy_abs] {
                        let bx = src_tx + dx;
                        let by = src_ty + dy;
                        if let Some(&c) = remaining_cap.get(&(*bucket, bx, by)) {
                            if c > 0 {
                                ring_candidates.push((bx, by));
                            }
                        }
                        if dy_abs == 0 {
                            break;
                        }
                    }
                }
                if ring_candidates.is_empty() {
                    continue;
                }
                let (best_bx, best_by) = *ring_candidates
                    .iter()
                    .min_by(|a, b| {
                        let da =
                            (a.0 as f64 - anchor_cx).powi(2) + (a.1 as f64 - anchor_cy).powi(2);
                        let db =
                            (b.0 as f64 - anchor_cx).powi(2) + (b.1 as f64 - anchor_cy).powi(2);
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap();
                if let Some(c) = remaining_cap.get_mut(&(*bucket, best_bx, best_by)) {
                    *c -= 1;
                }
                cell_x[ci] = best_bx as f64;
                cell_y[ci] = best_by as f64;
                spread_count += 1;
                let d = (best_bx - src_tx).abs() + (best_by - src_ty).abs();
                spread_dist_sum += d as i64;
                if d > spread_dist_max {
                    spread_dist_max = d;
                }
                placed = true;
                break 'r;
            }
            let _ = placed;
        }
    }

    if spread_count > 0 {
        eprintln!(
            "  spread stats: moved={} avg_manhattan={:.1} max_manhattan={}",
            spread_count,
            spread_dist_sum as f64 / spread_count as f64,
            spread_dist_max,
        );
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

    spread_overcrowded(
        ctx,
        idx_to_cell,
        type_aware,
        &cell_buckets,
        &mut snapped_x,
        &mut snapped_y,
    );

    (snapped_x, snapped_y)
}

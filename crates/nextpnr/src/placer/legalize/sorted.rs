//! Nearest-BEL legalization with region support.
//!
//! Phase A: one BEL pool per bucket, borrowed from the context.
//! Phase B (sequential): greedy assignment, outer cells first.

use crate::chipdb::{BelId, Loc};
use crate::common::{IdString, PlaceStrength};
use crate::context::{BucketGrid, Context};
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::bel_grid::RingCandidates;
use crate::placer::legalize::common::{
    build_cell_pin_template, place_cluster_children, unbind_movable_cells, DriverNodeRegistry,
};
use crate::placer::PlacerError;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::{leg_timers, Legalizer};

/// Nearest-BEL legalization with region support.
///
/// Phase A: one BEL pool per bucket, borrowed from the context.
/// Phase B (sequential): greedy assignment, outer cells first.
pub struct SortedLegalizer;

impl Legalizer for SortedLegalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
        _type_aware: &TypeAwarePlacement,
    ) -> Result<f64, PlacerError> {
        sorted_legalize(ctx, idx_to_cell, cell_x, cell_y, _type_aware)
    }
}

/// One bucket's BELs, and which of them are bound.
struct Pool {
    bels: Arc<[BelId]>,
    /// One flag per position in `bels`.
    taken: Vec<bool>,
}

/// Phase A: one pool per bucket, so aliases share a list and its flags.
///
/// `taken` is seeded from what is already bound, so the ring walk never
/// offers a BEL the caller would reject. Phase B is the only thing binding
/// from here on, so the flags stay exact.
fn build_pools(ctx: &Context, cell_infos: &[CellLegalizeInfo]) -> FxHashMap<IdString, Pool> {
    let mut pools: FxHashMap<IdString, Pool> = FxHashMap::default();
    for info in cell_infos {
        pools.entry(info.bucket).or_insert_with(|| {
            let bels = ctx.bucket_bel_list(info.bucket);
            let taken = bels.iter().map(|&b| !ctx.bel(b).is_available()).collect();
            Pool { bels, taken }
        });
    }
    pools
}

/// Per-cell info gathered before assignment.
struct CellLegalizeInfo {
    cell_idx: CellId,
    /// Resolved, so aliases of one bucket share its pool.
    bucket: IdString,
    cell_type_name: String,
    cell_name: String,
    cell_region: Option<u32>,
    target_x: f64,
    target_y: f64,
    /// True when this cell is a cluster child (placed by its root).
    is_cluster_child: bool,
    /// Cluster footprint constraints for cluster roots; empty otherwise.
    /// Each entry is (child_id, child_type, constr_x, constr_y, constr_z, abs_z).
    cluster_children: Vec<(CellId, IdString, i32, i32, i32, bool)>,
}

/// The BELs at `positions` of `info`'s bucket, sorted by
/// `(dist_sq, enum_index)`.
///
/// Distance ties are broken on enumeration order because they are the common
/// case -- a tile holds many slots at identical x/y -- and the order among
/// them decides which BEL Phase B binds. `RingCandidates` reproduces this
/// exact sequence lazily; this function is the region-constrained path and
/// the oracle that test proves it against.
fn candidate_list(
    info: &CellLegalizeInfo,
    grid: BucketGrid<'_>,
    positions: impl Iterator<Item = usize>,
) -> Vec<BelId> {
    // Carry each candidate's position in the BEL enumeration and break
    // distance ties on it. Making the comparison total is what lets
    // `RingCandidates` reproduce this sequence from a heap keyed on
    // `(dist_sq.to_bits(), enum_index)` -- an unstable sort and a heap agree
    // only when the order is total.
    let mut candidates: Vec<(BelId, f64, usize)> = positions
        .map(|order| {
            let bel_id = grid.bels()[order];
            let (bx, by) = grid.xy(bel_id);
            let dx = bx as f64 - info.target_x;
            let dy = by as f64 - info.target_y;
            (bel_id, dx * dx + dy * dy, order)
        })
        .collect();

    let cmp = |a: &(BelId, f64, usize), b: &(BelId, f64, usize)| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    };

    candidates.sort_unstable_by(cmp);

    // Iterate by reference, NOT `into_iter()`. Consuming the vec would hit the
    // in-place collect specialization, which reuses the SOURCE allocation: a
    // 24-byte-per-element buffer kept alive to hold 8-byte ids. Building a
    // fresh, exactly-sized vec is what bounds it.
    candidates.iter().map(|&(id, _, _)| id).collect()
}

/// Walk `candidates` in distance order and bind the cell to the first BEL
/// that is available, whose cluster footprint fits, that does not collide on
/// a shared mux node, and that the architecture accepts.
///
/// Returns the squared displacement of the BEL it bound, or `None` if every
/// candidate was rejected.
///
/// Takes an iterator rather than a slice so the candidate sequence can be
/// produced lazily: the common case rejects a few hundred BELs and binds,
/// having materialized nothing.
#[allow(clippy::too_many_arguments)]
fn try_bind_cell(
    ctx: &mut Context,
    info: &CellLegalizeInfo,
    registry: &mut DriverNodeRegistry,
    // `(BelId, enumeration index)`. The region-constrained path has no
    // enumeration index and passes `u32::MAX`.
    candidates: impl Iterator<Item = (BelId, u32)>,
    cluster_footprint_rejects: &mut u64,
    shared_mux_rejects: &mut u64,
    arch_validity_rejects: &mut u64,
) -> Result<Option<(f64, u32)>, PlacerError> {
    // The cell and its children are fixed across the candidate loop, so their
    // pin ports are resolved once. `is_legal` rebuilt them per candidate, and
    // with 1.8 million rejected candidates on FPGA01 that was two vector
    // allocations and a port walk each time.
    let root_template = build_cell_pin_template(ctx, info.cell_idx);
    let child_templates: Vec<_> = info
        .cluster_children
        .iter()
        .map(|&(child_id, ..)| build_cell_pin_template(ctx, child_id))
        .collect();

    'outer: for (bel, enum_index) in candidates {
        let bel_view = ctx.bel(bel);
        if !bel_view.is_available() {
            continue;
        }
        let root_loc = bel_view.loc();

        // Cluster footprint check: every constr_child must have an
        // available BEL at the constrained slot. Skip otherwise.
        let mut child_bels: Vec<(CellId, BelId)> = Vec::new();
        for &(child_id, child_type, cx_off, cy_off, cz_off, abs_z) in &info.cluster_children {
            let want_x = root_loc.x + cx_off;
            let want_y = root_loc.y + cy_off;
            let want_z = if abs_z { cz_off } else { root_loc.z + cz_off };
            let Some(child_bel) = ctx.bucket_bel_at(child_type, Loc::new(want_x, want_y, want_z))
            else {
                *cluster_footprint_rejects += 1;
                continue 'outer;
            };
            if !ctx.bel(child_bel).is_available() {
                *cluster_footprint_rejects += 1;
                continue 'outer;
            }
            child_bels.push((child_id, child_bel));
        }

        // Shared-mux check: root and every child's pin wires must not
        // collide on a routing node already claimed by another net.
        if !registry.is_legal_template(ctx, &root_template, bel) {
            *shared_mux_rejects += 1;
            continue 'outer;
        }
        // `child_bels` is filled in `cluster_children` order and the loop above
        // bails on the first missing child, so the two stay aligned.
        for (i, &(_, child_bel)) in child_bels.iter().enumerate() {
            if !registry.is_legal_template(ctx, &child_templates[i], child_bel) {
                *shared_mux_rejects += 1;
                continue 'outer;
            }
        }

        // All cheap checks passed — commit, then ask the architecture.
        //
        // Strength matters: refinement only moves cells bound at or below
        // STRONG (`placer1.cc:228`). Binding at PLACER would leave every
        // legalised cell frozen and make the refiner a silent no-op.
        // Upstream's strict legalisation binds plain cells WEAK
        // (`placer_heap.cc:1242`) and cluster-constrained ones STRONG
        // (`:1431`, matching `placer1.cc:580`); mirror that split.
        let root_strength = match ctx.design.cell(info.cell_idx).cluster {
            Some(_) => PlaceStrength::Strong,
            None => PlaceStrength::Weak,
        };
        if !ctx.bind_bel(bel, info.cell_idx, root_strength) {
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to bind cell {} to BEL {}",
                info.cell_name, bel,
            )));
        }

        // `isBelLocationValid` is a *tile-level* rule -- it reads whatever
        // is currently bound around the candidate -- so the cell has to be
        // bound before we can ask. Roll back and try the next bel if the
        // arch rejects it. Checked last because it is the costliest test
        // and the previous ones have already pruned most candidates.
        if !ctx.is_bel_location_valid(bel) {
            ctx.unbind_bel(bel);
            *arch_validity_rejects += 1;
            continue 'outer;
        }
        // place_cluster_children binds each child to its constrained
        // slot. The child slots we resolved above must match exactly.
        place_cluster_children(ctx, info.cell_idx, bel)?;

        // Re-check with the children bound. This is not redundant: a lone
        // LUT in a slice is valid, and it is binding the FF beside it that
        // can make the slice illegal. Validate the root's bel again too,
        // since the children share its tile.
        let child_bound: Vec<BelId> = child_bels
            .iter()
            .map(|&(child_id, _)| {
                ctx.design
                    .cell(child_id)
                    .bel
                    .expect("place_cluster_children must bind child")
            })
            .collect();
        if !std::iter::once(bel)
            .chain(child_bound.iter().copied())
            .all(|b| ctx.is_bel_location_valid(b))
        {
            for &b in &child_bound {
                ctx.unbind_bel(b);
            }
            ctx.unbind_bel(bel);
            *arch_validity_rejects += 1;
            continue 'outer;
        }

        registry.record(ctx, info.cell_idx, bel);
        for (&(child_id, _), &bound_bel) in child_bels.iter().zip(&child_bound) {
            registry.record(ctx, child_id, bound_bel);
        }

        let loc = ctx.bel(bel).loc();
        let dx = loc.x as f64 - info.target_x;
        let dy = loc.y as f64 - info.target_y;
        return Ok(Some((dx * dx + dy * dy, enum_index)));
    }

    Ok(None)
}

/// Nearest-BEL legalization.
///
/// 1. Unbinds all movable cells.
/// 2. Takes each bucket's BEL list from the context.
/// 3. Assigns cells sequentially (outer-first), skipping cluster children,
///    walking each cell's BELs nearest-first until one is legal.
/// 4. Calls `place_cluster_children` for cluster roots.
///
/// Returns total squared displacement.
pub fn sorted_legalize(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    _type_aware: &TypeAwarePlacement,
) -> Result<f64, PlacerError> {
    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Unbind all movable cells.
    unbind_movable_cells(ctx, idx_to_cell);

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

    // Gather per-cell info up front.
    // Read region from the cell directly via ctx.design.
    let t_leg = std::time::Instant::now();
    let cell_infos: Vec<CellLegalizeInfo> = order
        .iter()
        .map(|&idx| {
            let cell_idx = idx_to_cell[idx];
            let cell = ctx.cell(cell_idx);
            let is_cluster_child = cell.cluster().is_some_and(|root_id| root_id != cell_idx);
            let cluster_children: Vec<(CellId, IdString, i32, i32, i32, bool)> = ctx
                .design
                .clusters
                .get(&cell_idx)
                .map(|cluster| {
                    cluster
                        .constr_children
                        .iter()
                        .map(|&child_id| {
                            let child = ctx.design.cell(child_id);
                            (
                                child_id,
                                child.cell_type,
                                child.constr_x,
                                child.constr_y,
                                child.constr_z,
                                child.constr_abs_z,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            CellLegalizeInfo {
                cell_idx,
                bucket: ctx.resolve_bucket(cell.cell_type_id()),
                cell_type_name: cell.cell_type().to_owned(),
                cell_name: cell.name().to_owned(),
                cell_region: ctx.design.cell(cell_idx).region,
                target_x: cell_x[idx],
                target_y: cell_y[idx],
                is_cluster_child,
                cluster_children,
            }
        })
        .collect();

    if leg_timers() {
        eprintln!("  LEG_T cell_infos: {:.3}s", t_leg.elapsed().as_secs_f64());
    }
    let mut pools = build_pools(ctx, &cell_infos);

    if leg_timers() {
        eprintln!("  LEG_T pools: {:.3}s", t_leg.elapsed().as_secs_f64());
    }
    // Seed the shared-mux registry from already-bound cells (packer-placed
    // fixed cells: BUFG, IO, clock buffers). Movable cells were just unbound
    // so they don't contribute.
    let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
    if leg_timers() {
        eprintln!(
            "  LEG_T registry_seed: {:.3}s",
            t_leg.elapsed().as_secs_f64()
        );
    }
    let mut shared_mux_rejects: u64 = 0;
    let mut arch_validity_rejects: u64 = 0;
    let mut cluster_footprint_rejects: u64 = 0;

    // Phase B (sequential): assign cells to nearest available BEL that
    //   1. is itself available,
    //   2. for cluster roots, has every constrained child slot also available,
    //   3. doesn't collide with already-placed cells on a shared-mux node.
    let mut total_displacement = 0.0;

    for info in &cell_infos {
        if info.is_cluster_child {
            continue;
        }

        // Region-constrained cells keep the materialized path. Ring-walking a
        // cell whose region sits far from its target would scan outward across
        // the whole device before finding anything; bounding rings by the
        // region bbox is a separate geometry problem and a documented
        // follow-up. FPGA01 has no region-constrained cells.
        let pool = &pools[&info.bucket];
        let grid = ctx.bucket_grid(&pool.bels);
        let disp = if let Some(rid) = info.cell_region {
            let candidates =
                candidate_list(info, grid, ctx.bucket_positions_in_region(&pool.bels, rid));
            if candidates.is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }
            try_bind_cell(
                ctx,
                info,
                &mut registry,
                candidates.into_iter().map(|b| (b, u32::MAX)),
                &mut cluster_footprint_rejects,
                &mut shared_mux_rejects,
                &mut arch_validity_rejects,
            )?
        } else {
            if pool.bels.is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }
            // Exhaustive by construction, so there is no shortlist to widen
            // from: "try the near ones, then re-scan for the rest" collapses
            // into this one call.
            try_bind_cell(
                ctx,
                info,
                &mut registry,
                RingCandidates::new(grid, &pool.taken, info.target_x, info.target_y),
                &mut cluster_footprint_rejects,
                &mut shared_mux_rejects,
                &mut arch_validity_rejects,
            )?
        };

        match disp {
            Some((d, enum_index)) => {
                total_displacement += d;
                if enum_index != u32::MAX {
                    pools
                        .get_mut(&info.bucket)
                        .expect("pool exists for this bucket")
                        .taken[enum_index as usize] = true;
                }
            }
            None => {
                return Err(PlacerError::NoBelsAvailable(format!(
                    "{} (no available BELs for cell {} after {} cluster-footprint, {} shared-mux and {} arch-validity rejections)",
                    info.cell_type_name,
                    info.cell_name,
                    cluster_footprint_rejects,
                    shared_mux_rejects,
                    arch_validity_rejects,
                )))
            }
        }
    }

    if leg_timers() {
        eprintln!(
            "  LEG_T phase_b_done: {:.3}s",
            t_leg.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "  SortedLegalizer: cluster_rejects={} shared_mux_rejects={} arch_validity_rejects={}",
        cluster_footprint_rejects, shared_mux_rejects, arch_validity_rejects,
    );

    Ok(total_displacement)
}

#[cfg(test)]
mod candidate_list_tests {
    use super::*;

    /// A grid of BELs with `per_tile` slots at every (x, y), so equidistant
    /// candidates are the common case rather than the exception.
    fn grid_bels(dim: i32, per_tile: i32) -> Vec<BelId> {
        (0..dim * dim)
            .flat_map(|tile| (0..per_tile).map(move |z| BelId::new(tile, z)))
            .collect()
    }

    fn info_at(target_x: f64, target_y: f64) -> CellLegalizeInfo {
        CellLegalizeInfo {
            cell_idx: CellId::NONE,
            bucket: IdString(1),
            cell_type_name: "SLICE".to_string(),
            cell_name: "c0".to_string(),
            cell_region: None,
            target_x,
            target_y,
            is_cluster_child: false,
            cluster_children: Vec::new(),
        }
    }

    /// A region-constrained cell sees only the positions it is handed, nearest
    /// first, even when its region sits far from its target.
    #[test]
    fn a_region_constrained_cell_sees_only_its_regions_bels() {
        let bels = grid_bels(6, 2);
        let grid = BucketGrid::new(&bels, 6, 6);

        // The region is the single tile (5, 5) -- the far corner from the
        // target, so an unfiltered list would put other BELs first.
        let in_region: Vec<usize> = grid.span(5, 5).collect();
        assert_eq!(in_region.len(), 2, "fixture: (5,5) should hold 2 BELs");

        let got = candidate_list(&info_at(0.0, 0.0), grid, in_region.iter().copied());
        let want: Vec<BelId> = in_region.iter().map(|&i| bels[i]).collect();
        assert_eq!(got, want);

        // And with every position, the same cell sees everything, nearest first.
        let unconstrained = candidate_list(&info_at(0.0, 0.0), grid, 0..bels.len());
        assert_eq!(unconstrained.len(), bels.len());
        assert_ne!(
            unconstrained[0], got[0],
            "fixture is not discriminating: nearest BEL is already in the region"
        );
    }

    /// Cells of many aliases of one bucket share one pool, which borrows the
    /// context's list: the only per-call storage is one flag per BEL.
    #[test]
    fn pools_do_not_scale_with_alias_count() {
        let mut ctx = crate::context::example_context();
        let list = ctx.bucket_bel_list(ctx.id("DFF"));
        let mut infos = Vec::new();
        for i in 0..50 {
            let alias = format!("ALIAS{i}");
            ctx.add_cell_type_alias(&alias, "DFF");
            let mut info = info_at(0.0, 0.0);
            info.bucket = ctx.resolve_bucket(ctx.id(alias.as_str()));
            infos.push(info);
        }
        let pools = build_pools(&ctx, &infos);
        assert_eq!(pools.len(), 1);
        let pool = pools.values().next().expect("one pool");
        assert!(Arc::ptr_eq(&pool.bels, &list));
        assert_eq!(pool.taken.len(), list.len());
    }

    /// Sparse islands: most tiles empty, so the walk actually expands.
    fn sparse_bels(dim: i32, stride: i32, per_tile: i32) -> Vec<BelId> {
        let mut bels = Vec::new();
        for y in (0..dim).step_by(stride as usize) {
            for x in (0..dim).step_by(stride as usize) {
                for z in 0..per_tile {
                    bels.push(BelId::new(y * dim + x, z));
                }
            }
        }
        bels
    }

    /// The walk must emit exactly the sequence `candidate_list` builds --
    /// every element, in order, not merely the same first pick.
    ///
    /// This is what makes the ring query a data-structure change rather than
    /// a heuristic one. It has to be full-sequence because Phase B walks
    /// deep: FPGA01 averages ~285 rejected candidates per cell, so the 200th
    /// element matters as much as the first.
    #[test]
    fn the_walk_emits_candidate_lists_exact_sequence() {
        let layouts: Vec<(&str, i32, Vec<BelId>)> = vec![
            ("dense 10x10 x4", 10, grid_bels(10, 4)),
            ("dense 7x7 x1", 7, grid_bels(7, 1)),
            ("dense 20x20 x8", 20, grid_bels(20, 8)),
            ("sparse stride 3", 21, sparse_bels(21, 3, 2)),
            ("sparse stride 7", 28, sparse_bels(28, 7, 1)),
        ];
        for (label, dim, bels) in &layouts {
            let grid = BucketGrid::new(bels, *dim, *dim);
            for &(tx, ty) in &[
                (0.0, 0.0),
                (4.5, 4.5),
                (3.0, 7.0),
                (2.25, 6.75),
                (*dim as f64 - 1.0, *dim as f64 - 1.0),
                (-1.5, 2.0),
                (*dim as f64 + 1.5, *dim as f64 + 0.5),
            ] {
                let want = candidate_list(&info_at(tx, ty), grid, 0..bels.len());
                let free = vec![false; bels.len()];
                let got: Vec<BelId> = RingCandidates::new(grid, &free, tx, ty)
                    .map(|(id, _)| id)
                    .collect();
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{label} target=({tx},{ty}): walk emitted {} of {} candidates",
                    got.len(),
                    want.len()
                );
                assert_eq!(got, want, "{label} target=({tx},{ty}): order diverged");
            }
        }
    }
}

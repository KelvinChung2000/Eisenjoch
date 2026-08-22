//! Nearest-BEL legalization with region support.
//!
//! Phase A: index each cell type's BELs by tile.
//! Phase B (sequential): greedy assignment, outer cells first.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::bel_grid::{BelGrid, RingCandidates};
use crate::placer::legalize::common::{
    build_cell_pin_template, place_cluster_children, unbind_movable_cells, DriverNodeRegistry,
};
use crate::placer::PlacerError;
use rustc_hash::{FxHashMap, FxHashSet};

use super::Legalizer;

/// Nearest-BEL legalization with region support.
///
/// Phase A: index each cell type's BELs by tile.
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

/// Per-cell info gathered before assignment.
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
    /// Cluster footprint constraints for cluster roots; empty otherwise.
    /// Each entry is (child_id, child_type, constr_x, constr_y, constr_z, abs_z).
    cluster_children: Vec<(CellId, IdString, i32, i32, i32, bool)>,
}

/// Every BEL of `info`'s type, sorted by `(dist_sq, enum_index)`.
///
/// Distance ties are broken on enumeration order because they are the common
/// case -- a tile holds many slots at identical x/y -- and the order among
/// them decides which BEL Phase B binds. `RingCandidates` reproduces this
/// exact sequence lazily; this function is the region-constrained path and
/// the oracle that test proves it against.
fn candidate_list(
    info: &CellLegalizeInfo,
    bel_data_cache: &FxHashMap<IdString, Vec<(BelId, i32, i32, i32)>>,
    region_bel_sets: &FxHashMap<u32, FxHashSet<BelId>>,
) -> Vec<BelId> {
    let Some(bels) = bel_data_cache.get(&info.cell_type_id) else {
        return Vec::new();
    };

    // Carry each candidate's position in the BEL enumeration and break
    // distance ties on it. Equidistant BELs are common (a tile holds many
    // slots at identical x/y) and the order among them decides which one
    // Phase B binds. Making the comparison total is what lets `RingCandidates`
    // reproduce this sequence from a heap keyed on
    // `(dist_sq.to_bits(), enum_index)` -- an unstable sort and a heap agree
    // only when the order is total.
    let mut candidates: Vec<(BelId, f64, usize)> = bels
        .iter()
        .enumerate()
        .filter(|(_, &(bel_id, _, _, _))| {
            if let Some(rid) = info.cell_region {
                region_bel_sets
                    .get(&rid)
                    .map_or(false, |s| s.contains(&bel_id))
            } else {
                true
            }
        })
        .map(|(order, &(bel_id, bx, by, _bz))| {
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
    bel_by_loc: &FxHashMap<(IdString, i32, i32, i32), BelId>,
    registry: &mut DriverNodeRegistry,
    candidates: impl Iterator<Item = BelId>,
    cluster_footprint_rejects: &mut u64,
    shared_mux_rejects: &mut u64,
    arch_validity_rejects: &mut u64,
) -> Result<Option<f64>, PlacerError> {
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

    'outer: for bel in candidates {
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
            let Some(&child_bel) = bel_by_loc.get(&(child_type, want_x, want_y, want_z)) else {
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
        place_cluster_children(ctx, &bel_by_loc, info.cell_idx, bel)?;

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
        return Ok(Some(dx * dx + dy * dy));
    }

    Ok(None)
}

/// Nearest-BEL legalization.
///
/// 1. Unbinds all movable cells.
/// 2. Builds BEL data and a tile index per cell type.
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

    // Pre-collect BEL data per cell type into plain data (BelId, x, y, z).
    // Detaching it from `ctx` is what lets Phase B hold it while binding, and
    // its enumeration order is the tie-break both candidate paths sort on.
    let mut bel_data_cache: FxHashMap<IdString, Vec<(BelId, i32, i32, i32)>> = FxHashMap::default();
    for &cell_idx in idx_to_cell {
        let cell_type_id = ctx.cell(cell_idx).cell_type_id();
        bel_data_cache.entry(cell_type_id).or_insert_with(|| {
            ctx.bels_for_bucket(cell_type_id)
                .map(|bel| {
                    let loc = bel.loc();
                    (bel.id(), loc.x, loc.y, loc.z)
                })
                .collect()
        });
    }

    // Cluster-child types may not appear among `idx_to_cell` (children are
    // not in cell_to_idx). Pre-fetch their BEL pools too so we can probe
    // child slots when scoring cluster-root candidates.
    let mut child_types: FxHashSet<IdString> = FxHashSet::default();
    for &cell_idx in idx_to_cell {
        if let Some(cluster) = ctx.design.clusters.get(&cell_idx) {
            for &child_id in &cluster.constr_children {
                child_types.insert(ctx.design.cell(child_id).cell_type);
            }
        }
    }
    for child_type in child_types {
        bel_data_cache.entry(child_type).or_insert_with(|| {
            ctx.bels_for_bucket(child_type)
                .map(|bel| {
                    let loc = bel.loc();
                    (bel.id(), loc.x, loc.y, loc.z)
                })
                .collect()
        });
    }

    // Index by exact (cell_type, x, y, z) so cluster-root candidate scoring
    // can look up "is the child slot available?" in O(1).
    let bel_by_loc: FxHashMap<(IdString, i32, i32, i32), BelId> = {
        let mut idx = FxHashMap::default();
        for (cell_type_id, bels) in &bel_data_cache {
            for &(bel_id, x, y, z) in bels {
                idx.insert((*cell_type_id, x, y, z), bel_id);
            }
        }
        idx
    };

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
    let cell_infos: Vec<CellLegalizeInfo> = order
        .iter()
        .map(|&idx| {
            let cell_idx = idx_to_cell[idx];
            let cell = ctx.cell(cell_idx);
            let is_cluster_child = cell.cluster().map_or(false, |root_id| root_id != cell_idx);
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
                cell_type_id: cell.cell_type_id(),
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

    // Phase A: one spatial index per cell type, replacing the per-cell
    // shortlists.
    //
    // The old Phase A held a distance-sorted list per cell: O(cells * bels),
    // bounded to 1024 entries only by capping, which bounded work and not
    // legality -- 25% of FPGA01's cells exhausted the cap and paid a full
    // re-scan and re-sort of up to 1.08M entries. Indexing per *type* instead
    // is O(bels) once, and the walk keeps one cell's candidates live.
    //
    // Built from `bel_data_cache`, so the candidate universe is identical to
    // `candidate_list`'s by construction: same alias resolution via
    // `bels_for_bucket`, same filtering, same enumeration order.
    let bel_grids: FxHashMap<IdString, BelGrid> = bel_data_cache
        .iter()
        .map(|(&type_id, bels)| (type_id, BelGrid::build(bels, grid_w, grid_h)))
        .collect();

    // Seed the shared-mux registry from already-bound cells (packer-placed
    // fixed cells: BUFG, IO, clock buffers). Movable cells were just unbound
    // so they don't contribute.
    let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
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
        let disp = if info.cell_region.is_some() {
            let candidates = candidate_list(info, &bel_data_cache, &region_bel_sets);
            if candidates.is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }
            try_bind_cell(
                ctx,
                info,
                &bel_by_loc,
                &mut registry,
                candidates.into_iter(),
                &mut cluster_footprint_rejects,
                &mut shared_mux_rejects,
                &mut arch_validity_rejects,
            )?
        } else {
            let (Some(grid), Some(bels)) = (
                bel_grids.get(&info.cell_type_id),
                bel_data_cache.get(&info.cell_type_id),
            ) else {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            };
            if grid.is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }
            // Exhaustive by construction, so there is no shortlist to widen
            // from: "try the near ones, then re-scan for the rest" collapses
            // into this one call.
            try_bind_cell(
                ctx,
                info,
                &bel_by_loc,
                &mut registry,
                RingCandidates::new(grid, bels, info.target_x, info.target_y),
                &mut cluster_footprint_rejects,
                &mut shared_mux_rejects,
                &mut arch_validity_rejects,
            )?
        };

        match disp {
            Some(d) => total_displacement += d,
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
    fn grid_bels(dim: i32, per_tile: i32) -> Vec<(BelId, i32, i32, i32)> {
        let mut bels = Vec::new();
        let mut tile = 0;
        for x in 0..dim {
            for y in 0..dim {
                for z in 0..per_tile {
                    bels.push((BelId::new(tile, z), x, y, z));
                }
                tile += 1;
            }
        }
        bels
    }

    fn info_at(target_x: f64, target_y: f64) -> CellLegalizeInfo {
        CellLegalizeInfo {
            cell_idx: CellId::NONE,
            cell_type_id: IdString(1),
            cell_type_name: "SLICE".to_string(),
            cell_name: "c0".to_string(),
            cell_region: None,
            target_x,
            target_y,
            is_cluster_child: false,
            cluster_children: Vec::new(),
        }
    }

    fn cache(bels: Vec<(BelId, i32, i32, i32)>) -> FxHashMap<IdString, Vec<(BelId, i32, i32, i32)>> {
        let mut m = FxHashMap::default();
        m.insert(IdString(1), bels);
        m
    }

    /// Region-constrained cells must see only their region's BELs.
    ///
    /// This is the one path the ring walk does NOT take -- walking outward from
    /// a cell whose region sits far from its target would scan the device
    /// before finding anything -- so it stays on `candidate_list`, and this is
    /// what holds that filter in place. Every other test here leaves
    /// `cell_region` unset, so without this the filter was unguarded.
    #[test]
    fn a_region_constrained_cell_sees_only_its_regions_bels() {
        let bels = grid_bels(6, 2);
        let c = cache(bels.clone());

        // The region is the single tile (5, 5) -- the far corner from the
        // target, so an unfiltered list would put other BELs first.
        let in_region: FxHashSet<BelId> = bels
            .iter()
            .filter(|&&(_, x, y, _)| (x, y) == (5, 5))
            .map(|&(id, _, _, _)| id)
            .collect();
        assert_eq!(in_region.len(), 2, "fixture: (5,5) should hold 2 BELs");

        let mut regions = FxHashMap::default();
        regions.insert(7u32, in_region.clone());

        let mut info = info_at(0.0, 0.0);
        info.cell_region = Some(7);

        let got = candidate_list(&info, &c, &regions);
        assert_eq!(got.len(), 2, "region cell got {} candidates", got.len());
        for id in &got {
            assert!(in_region.contains(id), "candidate {id} is outside the region");
        }

        // And with no region set, the same cell sees everything, nearest first.
        let unconstrained = candidate_list(&info_at(0.0, 0.0), &c, &regions);
        assert_eq!(unconstrained.len(), bels.len());
        assert_ne!(
            unconstrained[0], got[0],
            "fixture is not discriminating: nearest BEL is already in the region"
        );
    }

    /// Sparse islands: most tiles empty, so the walk actually expands.
    fn sparse_bels(dim: i32, stride: i32, per_tile: i32) -> Vec<(BelId, i32, i32, i32)> {
        let mut bels = Vec::new();
        let mut tile = 0;
        let mut x = 0;
        while x < dim {
            let mut y = 0;
            while y < dim {
                for z in 0..per_tile {
                    bels.push((BelId::new(tile, z), x, y, z));
                }
                tile += 1;
                y += stride;
            }
            x += stride;
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
    ///
    /// It is also the only thing that certifies the swap. Nothing on this
    /// branch is bit-reproducible (`e3efbf3` is not merged), so a live run
    /// cannot tell a legalizer regression from ordinary run-to-run spread.
    #[test]
    fn the_walk_emits_candidate_lists_exact_sequence() {
        let regions = FxHashMap::default();
        let layouts: Vec<(&str, i32, Vec<(BelId, i32, i32, i32)>)> = vec![
            ("dense 10x10 x4", 10, grid_bels(10, 4)),
            ("dense 7x7 x1", 7, grid_bels(7, 1)),
            ("dense 20x20 x8", 20, grid_bels(20, 8)),
            ("sparse stride 3", 21, sparse_bels(21, 3, 2)),
            ("sparse stride 7", 28, sparse_bels(28, 7, 1)),
        ];
        for (label, dim, bels) in &layouts {
            let c = cache(bels.clone());
            let grid = BelGrid::build(bels, *dim, *dim);
            for &(tx, ty) in &[
                (0.0, 0.0),
                (4.5, 4.5),
                (3.0, 7.0),
                (2.25, 6.75),
                (*dim as f64 - 1.0, *dim as f64 - 1.0),
                (-1.5, 2.0),
                (*dim as f64 + 1.5, *dim as f64 + 0.5),
            ] {
                let info = info_at(tx, ty);
                let want = candidate_list(&info, &c, &regions);
                let got: Vec<BelId> = RingCandidates::new(&grid, bels, tx, ty).collect();
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

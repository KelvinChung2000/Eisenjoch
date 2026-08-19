//! Parallel-sorted nearest-BEL legalization with region support.
//!
//! Phase A (parallel): pre-sort BEL candidates by distance per cell.
//! Phase B (sequential): greedy assignment, outer cells first.

use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::legalize::common::{
    place_cluster_children, unbind_movable_cells, DriverNodeRegistry,
};
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
        _type_aware: &TypeAwarePlacement,
    ) -> Result<f64, PlacerError> {
        sorted_legalize(ctx, idx_to_cell, cell_x, cell_y, _type_aware)
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
    /// Cluster footprint constraints for cluster roots; empty otherwise.
    /// Each entry is (child_id, child_type, constr_x, constr_y, constr_z, abs_z).
    cluster_children: Vec<(CellId, IdString, i32, i32, i32, bool)>,
}

/// How many nearest BELs Phase A keeps per cell.
///
/// Phase B walks the list in distance order and stops at the first legal BEL,
/// so it almost always binds within the first handful. The shortlist only has
/// to be deep enough that exhausting it is rare, because exhausting it costs
/// one full re-scan for that cell. It must NOT be sized to "always enough" --
/// that is the O(cells * bels) blowup this exists to avoid.
const CAND_SHORTLIST: usize = 1024;

/// Distance-sorted BEL candidates for one cell.
///
/// With `limit = Some(k)` this returns the k nearest, selected in O(m) and
/// then sorted among themselves. With `limit = None` it returns every
/// candidate, fully sorted. Both orderings agree on their common prefix,
/// which is what lets Phase B widen from one to the other without changing
/// which BEL it picks.
fn candidate_list(
    info: &CellLegalizeInfo,
    bel_data_cache: &FxHashMap<IdString, Vec<(BelId, i32, i32, i32)>>,
    region_bel_sets: &FxHashMap<u32, FxHashSet<BelId>>,
    limit: Option<usize>,
) -> Vec<BelId> {
    let Some(bels) = bel_data_cache.get(&info.cell_type_id) else {
        return Vec::new();
    };

    // Carry each candidate's position in the BEL enumeration and break
    // distance ties on it. Equidistant BELs are common (a tile holds many
    // slots at identical x/y) and the order among them decides which one
    // Phase B binds. The previous full `sort_by` was stable, so it preserved
    // enumeration order for free; `select_nth_unstable_by` does not. Making
    // the comparison total is what keeps the shortlist and the full list
    // agreeing on their common prefix -- and keeps both agreeing with the
    // original stable sort.
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

    match limit {
        Some(k) if k < candidates.len() => {
            // Partition so the k nearest are in front, then order just those.
            candidates.select_nth_unstable_by(k, cmp);
            candidates.truncate(k);
            candidates.sort_unstable_by(cmp);
        }
        _ => candidates.sort_unstable_by(cmp),
    }

    // Iterate by reference, NOT `into_iter()`. Consuming the vec would hit the
    // in-place collect specialization, which reuses the SOURCE allocation --
    // one buffer sized for every BEL of the type (17MB for DFF on this device)
    // kept alive per cell to hold at most `CAND_SHORTLIST` ids. `truncate`
    // sets a vec's length and never its capacity, so capping the shortlist
    // alone does not bound the memory; building a fresh, exactly-sized vec is
    // what does.
    candidates.iter().map(|&(id, _, _)| id).collect()
}

/// Walk `candidates` in distance order and bind the cell to the first BEL
/// that is available, whose cluster footprint fits, that does not collide on
/// a shared mux node, and that the architecture accepts.
///
/// Returns the squared displacement of the BEL it bound, or `None` if every
/// candidate was rejected. Split out of the Phase B loop so the same checks
/// can run against a cell's shortlist and then, only if that is exhausted,
/// against the remainder of its full candidate list.
#[allow(clippy::too_many_arguments)]
fn try_bind_cell(
    ctx: &mut Context,
    info: &CellLegalizeInfo,
    bel_by_loc: &FxHashMap<(IdString, i32, i32, i32), BelId>,
    registry: &mut DriverNodeRegistry,
    candidates: &[BelId],
    cluster_footprint_rejects: &mut u64,
    shared_mux_rejects: &mut u64,
    arch_validity_rejects: &mut u64,
) -> Result<Option<f64>, PlacerError> {
    'outer: for &bel in candidates {
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
        if !registry.is_legal(ctx, info.cell_idx, bel) {
            *shared_mux_rejects += 1;
            continue 'outer;
        }
        for &(child_id, child_bel) in &child_bels {
            if !registry.is_legal(ctx, child_id, child_bel) {
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
    _type_aware: &TypeAwarePlacement,
) -> Result<f64, PlacerError> {
    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Unbind all movable cells.
    unbind_movable_cells(ctx, idx_to_cell);

    // Pre-collect BEL data per cell type into plain data (BelId, x, y, z)
    // so we can share across rayon threads without lifetime issues.
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

    // Gather per-cell info for parallel phase.
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

    // Phase A (parallel): distance-sorted BEL candidate shortlists.
    //
    // Bounded to CAND_SHORTLIST per cell. The unbounded version held, for
    // every cell, a sorted list of every BEL of that cell's type -- all of
    // them live at once. On xc7_large a DFF has 1,077,536 BELs and BelId is 8
    // bytes, so that is 8.6MB per cell over ~105k cells: O(cells * bels), and
    // ~900GB even with an exactly-sized allocation. Capping the length and
    // fixing the allocation are both required.
    let sorted_candidates: Vec<Vec<BelId>> = cell_infos
        .par_iter()
        .map(|info| {
            candidate_list(
                info,
                &bel_data_cache,
                &region_bel_sets,
                Some(CAND_SHORTLIST),
            )
        })
        .collect();

    // Seed the shared-mux registry from already-bound cells (packer-placed
    // fixed cells: BUFG, IO, clock buffers). Movable cells were just unbound
    // so they don't contribute.
    let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
    let mut shared_mux_rejects: u64 = 0;
    let mut arch_validity_rejects: u64 = 0;
    let mut cluster_footprint_rejects: u64 = 0;
    let mut widened_rescans: u64 = 0;

    // Phase B (sequential): assign cells to nearest available BEL that
    //   1. is itself available,
    //   2. for cluster roots, has every constrained child slot also available,
    //   3. doesn't collide with already-placed cells on a shared-mux node.
    let mut total_displacement = 0.0;

    for (i, info) in cell_infos.iter().enumerate() {
        if info.is_cluster_child {
            continue;
        }

        let shortlist = &sorted_candidates[i];

        if shortlist.is_empty() {
            return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
        }

        let mut disp = try_bind_cell(
            ctx,
            info,
            &bel_by_loc,
            &mut registry,
            shortlist,
            &mut cluster_footprint_rejects,
            &mut shared_mux_rejects,
            &mut arch_validity_rejects,
        )?;

        // The shortlist bounds work, not legality: if all k nearest BELs were
        // rejected the cell may still have a legal home further out. Rebuild
        // its full list and try the remainder. Both lists share one total
        // order, so the full list's first `shortlist.len()` entries are
        // exactly the ones already tried.
        if disp.is_none() {
            widened_rescans += 1;
            let full = candidate_list(info, &bel_data_cache, &region_bel_sets, None);
            if full.len() > shortlist.len() {
                disp = try_bind_cell(
                    ctx,
                    info,
                    &bel_by_loc,
                    &mut registry,
                    &full[shortlist.len()..],
                    &mut cluster_footprint_rejects,
                    &mut shared_mux_rejects,
                    &mut arch_validity_rejects,
                )?;
            }
        }

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
        "  SortedLegalizer: cluster_rejects={} shared_mux_rejects={} arch_validity_rejects={} widened_rescans={}",
        cluster_footprint_rejects, shared_mux_rejects, arch_validity_rejects, widened_rescans,
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

    /// The shortlist must be a prefix of the full list.
    ///
    /// Phase B widens by trying the shortlist and then the full list's
    /// remainder, skipping `shortlist.len()` entries it believes it already
    /// tried. If the two disagree on that prefix, widening silently skips
    /// untried BELs and binds a different one than an uncapped run would.
    /// `select_nth_unstable_by` is not stable, so this holds only because
    /// the comparison breaks distance ties on enumeration order.
    #[test]
    fn shortlist_is_a_prefix_of_the_full_list() {
        let regions = FxHashMap::default();
        for &(dim, per_tile) in &[(10, 4), (20, 8), (7, 1)] {
            let c = cache(grid_bels(dim, per_tile));
            let total = (dim * dim * per_tile) as usize;
            for &(tx, ty) in &[(0.0, 0.0), (4.5, 4.5), (3.0, 7.0)] {
                let info = info_at(tx, ty);
                let full = candidate_list(&info, &c, &regions, None);
                assert_eq!(full.len(), total, "full list must keep every candidate");

                for &k in &[1usize, 17, 256, total, total + 100] {
                    let short = candidate_list(&info, &c, &regions, Some(k));
                    let want = k.min(total);
                    assert_eq!(short.len(), want, "dim={dim} per_tile={per_tile} k={k}");
                    assert_eq!(
                        short[..],
                        full[..want],
                        "shortlist diverged from the full list's prefix \
                         (dim={dim} per_tile={per_tile} target=({tx},{ty}) k={k})"
                    );
                }
            }
        }
    }

    /// The capped list must not carry the oversized source allocation.
    ///
    /// `truncate` sets a vec's length and never its capacity, so a shortlist
    /// built by consuming the candidate vec would hold one buffer sized for
    /// every BEL of the type -- the FPGA01 legalization OOM.
    #[test]
    fn shortlist_allocation_is_bounded_by_the_cap() {
        let regions = FxHashMap::default();
        let c = cache(grid_bels(40, 8)); // 12,800 candidates
        let info = info_at(20.0, 20.0);
        let short = candidate_list(&info, &c, &regions, Some(64));
        assert_eq!(short.len(), 64);
        assert!(
            short.capacity() < 256,
            "shortlist kept an oversized buffer: capacity {} for 64 ids",
            short.capacity()
        );
    }
}

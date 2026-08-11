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
use crate::placer::{process_rss_kb, PlacerError};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::Legalizer;

/// Report RSS at a phase boundary when `NPNR_LEG_DIAG=1`.
///
/// Peak RSS sampled from outside the process says only that the legalizer
/// allocated; it cannot say which structure did. These do.
fn diag_rss(stage: &str) {
    if std::env::var("NPNR_LEG_DIAG").ok().as_deref() == Some("1") {
        eprintln!(
            "    leg_rss[{stage}]: {:.0}MB",
            process_rss_kb() as f64 / 1024.0
        );
    }
}

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

/// How many nearest BELs Phase A keeps per cell.
///
/// Phase B almost always binds within the first handful: it walks the list in
/// distance order and stops at the first legal BEL. The shortlist only has to
/// be deep enough that exhausting it is rare, because exhausting it costs one
/// full re-scan for that cell. It must NOT be sized to "always enough" — that
/// is the O(cells * bels) blowup this exists to avoid.
const CAND_SHORTLIST: usize = 1024;

/// Distance-sorted BEL candidates for one cell.
///
/// With `limit = Some(k)` this returns the k nearest, selected in O(m) and
/// then sorted among themselves. With `limit = None` it returns every
/// candidate, fully sorted — the exact list the pre-shortlist code built.
/// Both orderings agree on their common prefix, which is what lets Phase B
/// widen from one to the other without changing its choice.
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
    // slots at identical x/y), and the ordering among them decides which one
    // Phase B binds. A stable sort would preserve enumeration order for free,
    // but select_nth_unstable_by does not, so making the comparison total is
    // what keeps the shortlist and the full list agreeing on their common
    // prefix -- and keeps both agreeing with the original full stable sort.
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
    // in-place collect specialization, which reuses the source allocation --
    // one buffer sized for every BEL of the type (30MB on xc7_large) kept alive
    // per cell to hold `CAND_SHORTLIST` ids. `truncate` sets the length, never
    // the capacity, so the shortlist alone does not bound the memory; building
    // a fresh, exactly-sized vec is what does.
    candidates.iter().map(|&(id, _, _)| id).collect()
}

/// Walk `candidates` in order and bind the cell to the first BEL that is
/// available, whose cluster footprint fits, and which does not collide on a
/// shared mux node.
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

        // All checks passed — commit.
        if !ctx.bind_bel(bel, info.cell_idx, PlaceStrength::Placer) {
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to bind cell {} to BEL {}",
                info.cell_name, bel,
            )));
        }
        registry.record(ctx, info.cell_idx, bel);

        let loc = ctx.bel(bel).loc();
        let dx = loc.x as f64 - info.target_x;
        let dy = loc.y as f64 - info.target_y;

        // place_cluster_children binds each child to its constrained
        // slot. The child slots we resolved above must match exactly.
        place_cluster_children(ctx, bel_by_loc, info.cell_idx, bel)?;
        for &(child_id, _) in &child_bels {
            let bound_bel = ctx
                .design
                .cell(child_id)
                .bel
                .expect("place_cluster_children must bind child");
            registry.record(ctx, child_id, bound_bel);
        }

        return Ok(Some(dx * dx + dy * dy));
    }

    Ok(None)
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
    diag_rss("entry");

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

    if std::env::var("NPNR_LEG_DIAG").ok().as_deref() == Some("1") {
        let total: usize = bel_data_cache.values().map(|v| v.len()).sum();
        eprintln!(
            "    leg_bels: types={} total_bel_entries={} largest_type={}",
            bel_data_cache.len(),
            total,
            bel_data_cache.values().map(|v| v.len()).max().unwrap_or(0),
        );
    }
    diag_rss("bel_data_cache");

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

    diag_rss("bel_by_loc");

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

    diag_rss("cell_infos");

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

    // Phase A (parallel): distance-sorted BEL candidate SHORTLISTS.
    //
    // Only the `CAND_SHORTLIST` nearest BELs per cell are kept. Materializing
    // every BEL of a cell's type for every cell is O(cells * bels): on FPGA01
    // that is 76,660 cells against ~110k candidate BELs each, about 33GB, and
    // it OOM-killed the process while still inside this phase. Selecting the K
    // nearest is also asymptotically cheaper than sorting all of them, O(m)
    // versus O(m log m) per cell.
    //
    // If a cell exhausts its shortlist in Phase B it re-derives the FULL
    // sorted list for that one cell and carries on past the shortlist, so the
    // BEL finally chosen is exactly the one the old code would have chosen.
    // The shortlist bounds memory; it does not change the answer.
    diag_rss("region_bel_sets");
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
    if std::env::var("NPNR_LEG_DIAG").ok().as_deref() == Some("1") {
        let entries: usize = sorted_candidates.iter().map(|v| v.len()).sum();
        let capacity: usize = sorted_candidates.iter().map(|v| v.capacity()).sum();
        eprintln!(
            "    leg_phaseA: lists={} total_len={} len_mb={} total_capacity={} cap_mb={}",
            sorted_candidates.len(),
            entries,
            entries * std::mem::size_of::<BelId>() / (1024 * 1024),
            capacity,
            capacity * std::mem::size_of::<BelId>() / (1024 * 1024),
        );
    }
    diag_rss("phase_a");

    // Seed the shared-mux registry from already-bound cells (packer-placed
    // fixed cells: BUFG, IO, clock buffers). Movable cells were just unbound
    // so they don't contribute.
    let mut registry = DriverNodeRegistry::seed_from_bound(ctx);
    let mut shared_mux_rejects: u64 = 0;
    // Cells that exhausted their shortlist and needed the full rescan. Should
    // stay near zero; a large count means CAND_SHORTLIST is too shallow.
    let mut widened_rescans: u64 = 0;
    let mut cluster_footprint_rejects: u64 = 0;

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

        let mut displacement = try_bind_cell(
            ctx,
            info,
            &bel_by_loc,
            &mut registry,
            shortlist,
            &mut cluster_footprint_rejects,
            &mut shared_mux_rejects,
        )?;

        // Shortlist exhausted. It can only be missing candidates if it was
        // truncated, i.e. it came back full. Re-derive the complete list for
        // this one cell and resume past the part already tried, which lands on
        // exactly the BEL an untruncated list would have reached.
        if displacement.is_none() && shortlist.len() >= CAND_SHORTLIST {
            let full = candidate_list(info, &bel_data_cache, &region_bel_sets, None);
            if full.len() > shortlist.len() {
                widened_rescans += 1;
                displacement = try_bind_cell(
                    ctx,
                    info,
                    &bel_by_loc,
                    &mut registry,
                    &full[shortlist.len()..],
                    &mut cluster_footprint_rejects,
                    &mut shared_mux_rejects,
                )?;
            }
        }

        if let Some(d) = displacement {
            total_displacement += d;
        } else {
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (no available BELs for cell {} after {} cluster-footprint and {} shared-mux rejections)",
                info.cell_type_name,
                info.cell_name,
                cluster_footprint_rejects,
                shared_mux_rejects,
            )));
        }
    }

    diag_rss("phase_b");

    eprintln!(
        "  SortedLegalizer: cluster_rejects={} shared_mux_rejects={} shortlist={} widened_rescans={}",
        cluster_footprint_rejects, shared_mux_rejects, CAND_SHORTLIST, widened_rescans,
    );

    Ok(total_displacement)
}

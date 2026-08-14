//! The placer1 refinement loop.
//!
//! Port of `SAPlacer::place(refine = true)` (`common/place/placer1.cc:137`),
//! which is what `placer1_refine` runs and what HeAP unconditionally calls at
//! the end of placement (`placer_heap.cc:402-412`). Only the refinement path is
//! ported; the from-scratch annealing path is a separate job.
//!
//! Refinement differs from a cold anneal in four ways, all load-bearing:
//! `require_legal` is false, the search radius starts at 3 rather than the
//! fabric width, the temperature starts at `1e-7`, and the loop exits after a
//! *single* iteration that fails to improve.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::chipdb::{BelId, Loc};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::PlacerError;
use crate::timing::TimingAnalyser;

use super::config::{RefineCfg, RefineStats};
use super::cost::CostModel;

/// Bels of one cell type, bucketed by location, owned so the search does not
/// hold a borrow of the context across a bind.
struct BelGrid {
    /// `[x][y] -> bels`, upstream's `FastBelsData`.
    by_loc: Vec<Vec<Vec<BelId>>>,
    /// Total bels of this type, against `min_bels_for_grid_pick`.
    count: i32,
}

/// `placer1_refine` -- improve an existing placement in place.
///
/// The placement must already be legal and bound at strength `STRONG` or
/// below; anything stronger is treated as pinned and never moved.
pub fn refine_placement(
    ctx: &mut Context,
    cfg: &RefineCfg,
) -> Result<RefineStats, PlacerError> {
    let mut autoplaced: Vec<CellId> = Vec::new();
    let mut chain_basis: Vec<CellId> = Vec::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        // Refinement moves what is already placed; strength above STRONG is
        // the arch's or the user's choice and is not ours to revisit.
        if cell.bel.is_none() || cell.bel_strength > PlaceStrength::Strong {
            continue;
        }
        match cell.cluster {
            // Non-root cluster members move only with their root.
            Some(root) if root != cell_id => continue,
            Some(_) => chain_basis.push(cell_id),
            None => autoplaced.push(cell_id),
        }
    }

    let mut stats = RefineStats {
        autoplaced: autoplaced.len(),
        chain_basis: chain_basis.len(),
        ..RefineStats::default()
    };
    if autoplaced.is_empty() && chain_basis.is_empty() {
        return Ok(stats);
    }

    let grids = build_bel_grids(ctx, &autoplaced, &chain_basis);
    let (max_x, max_y) = fabric_extent(ctx);

    let mut tmg = TimingAnalyser::new();
    tmg.setup_and_run(ctx);

    let mut cost = CostModel::build(ctx);
    cost.refresh(ctx, &tmg, cfg);
    stats.wirelen_before = cost.wirelen;
    stats.timing.cost_before = cost.timing;
    stats.timing.critical_arcs = cost.critical_arcs();

    let mut temp = cfg.temp;
    let mut diameter = cfg.diameter;
    let mut avg_wirelen = cost.wirelen;
    let mut min_wirelen = cost.wirelen;
    let mut n_no_progress = 0usize;

    for iter in 1.. {
        let mut n_move = 0u64;
        let mut n_accept = 0u64;
        let mut improved = false;

        for _ in 0..cfg.inner_iters {
            for &cell in &autoplaced {
                let Some(target) =
                    random_bel_for_cell(ctx, &grids, cell, diameter, cfg, None)
                else {
                    continue;
                };
                if Some(target) == ctx.design.cell(cell).bel {
                    continue;
                }
                n_move += 1;
                if try_swap_position(ctx, &mut cost, &tmg, cfg, cell, target, temp) {
                    n_accept += 1;
                }
            }
            for &root in &chain_basis {
                let root_bel = ctx.design.cell(root).bel.expect("chain root is placed");
                let base_z = ctx.chipdb().bel_loc(root_bel).z;
                let Some(target) =
                    random_bel_for_cell(ctx, &grids, root, diameter, cfg, Some(base_z))
                else {
                    continue;
                };
                if target == root_bel {
                    continue;
                }
                n_move += 1;
                if try_swap_chain(ctx, &mut cost, &tmg, cfg, root, target, temp) {
                    n_accept += 1;
                }
            }
        }

        stats.moves_tried += n_move;
        stats.moves_accepted += n_accept;
        stats.iterations = iter;

        if cost.wirelen < min_wirelen {
            min_wirelen = cost.wirelen;
            improved = true;
        }
        if improved {
            n_no_progress = 0;
        } else {
            n_no_progress += 1;
        }

        // Refinement stops the first time an iteration fails to improve;
        // a cold anneal would allow five.
        if temp <= 1e-7 && n_no_progress >= 1 {
            break;
        }

        let r_accept = if n_move == 0 {
            0.0
        } else {
            n_accept as f64 / n_move as f64
        };
        let m = max_x.max(max_y) + 1;
        if cost.wirelen < (95 * avg_wirelen) / 100 && cost.wirelen > 0 {
            avg_wirelen = (8 * avg_wirelen + 2 * cost.wirelen) / 10;
        } else {
            let diam_next = diameter as f64 * (1.0 - 0.44 + r_accept);
            diameter = ((diam_next + 0.5) as i32).clamp(1, m);
            if r_accept > 0.96 {
                temp *= 0.5;
            } else if r_accept > 0.8 {
                temp *= 0.9;
            } else if r_accept > 0.15 && diameter > 1 {
                temp *= 0.95;
            } else {
                temp *= 0.8;
            }
        }

        if cfg.timing_driven {
            tmg.run(ctx);
        }
        // Rebuild from scratch each iteration: criticalities have moved, and
        // upstream recomputes the totals here specifically so incremental
        // rounding cannot accumulate.
        cost.refresh(ctx, &tmg, cfg);
    }

    stats.wirelen_after = cost.wirelen;
    stats.timing.cost_after = cost.timing;
    Ok(stats)
}

/// Bels within `diameter` of the cell, drawn uniformly; `force_z` pins the
/// sub-tile slot so a chain keeps its shape.
fn random_bel_for_cell(
    ctx: &mut Context,
    grids: &FxHashMap<IdString, BelGrid>,
    cell: CellId,
    diameter: i32,
    cfg: &RefineCfg,
    force_z: Option<i32>,
) -> Option<BelId> {
    let cell_type = ctx.design.cell(cell).cell_type;
    let grid = grids.get(&cell_type)?;
    let curr = ctx.chipdb().bel_loc(ctx.design.cell(cell).bel?);
    let region = ctx.design.cell(cell).region;

    // Upstream spins until it finds a candidate. Bound it instead: a cell whose
    // window holds nothing placeable would otherwise hang the placer.
    for _ in 0..64 {
        let (nx, ny) = if cfg.min_bels_for_grid_pick >= 0 && grid.count < cfg.min_bels_for_grid_pick
        {
            (0usize, 0usize)
        } else {
            let dx = ctx.rng_mut().rng_n(2 * diameter + 1) + (curr.x - diameter).max(0);
            let dy = ctx.rng_mut().rng_n(2 * diameter + 1) + (curr.y - diameter).max(0);
            (dx as usize, dy as usize)
        };
        let Some(col) = grid.by_loc.get(nx) else {
            continue;
        };
        let Some(bucket) = col.get(ny) else {
            continue;
        };
        if bucket.is_empty() {
            continue;
        }
        let bel = bucket[ctx.rng_mut().rng_n(bucket.len() as i32) as usize];
        if let Some(z) = force_z {
            if ctx.chipdb().bel_loc(bel).z != z {
                continue;
            }
        }
        if let Some(rid) = region {
            if !ctx.is_bel_in_region(bel, rid) {
                continue;
            }
        }
        return Some(bel);
    }
    None
}

/// `try_swap_position` -- move one unclustered cell, swapping with whatever
/// occupies the target.
fn try_swap_position(
    ctx: &mut Context,
    cost: &mut CostModel,
    tmg: &TimingAnalyser,
    cfg: &RefineCfg,
    cell: CellId,
    new_bel: BelId,
    temp: f64,
) -> bool {
    // Clustered cells are the chain swap's business.
    if ctx.design.cell(cell).cluster.is_some() {
        return false;
    }
    let old_bel = ctx.design.cell(cell).bel.expect("movable cell is placed");
    let other = ctx.conflicting_bel_cell(new_bel);
    if let Some(oc) = other {
        let info = ctx.design.cell(oc);
        // A swap partner must itself be free to move, and free to move alone.
        if info.cluster.is_some() || info.bel_strength > PlaceStrength::Weak {
            return false;
        }
    }

    let cell_type = ctx.design.cell(cell).cell_type;
    if !ctx.bel(new_bel).is_valid_for_cell_type(cell_type) {
        return false;
    }
    if let Some(oc) = other {
        let other_type = ctx.design.cell(oc).cell_type;
        if !ctx.bel(old_bel).is_valid_for_cell_type(other_type) {
            return false;
        }
    }

    ctx.unbind_bel(old_bel);
    if other.is_some() {
        ctx.unbind_bel(new_bel);
    }
    ctx.bind_bel(new_bel, cell, PlaceStrength::Weak);
    if let Some(oc) = other {
        ctx.bind_bel(old_bel, oc, PlaceStrength::Weak);
    }

    let restore = |ctx: &mut Context| {
        ctx.unbind_bel(new_bel);
        if other.is_some() {
            ctx.unbind_bel(old_bel);
        }
        ctx.bind_bel(old_bel, cell, PlaceStrength::Weak);
        if let Some(oc) = other {
            ctx.bind_bel(new_bel, oc, PlaceStrength::Weak);
        }
    };

    // Both ends, not just the destination: ripping a cell out can invalidate
    // the tile it left, e.g. by freeing a dedicated path its neighbour needed.
    if !ctx.is_bel_location_valid(new_bel) || !ctx.is_bel_location_valid(old_bel) {
        restore(ctx);
        return false;
    }

    let mut moved = vec![cell];
    moved.extend(other);
    let staged = cost.stage(ctx, tmg, cfg, &moved);
    // The constraint-distance term upstream adds here is identically zero in
    // refinement: both cells are known unclustered by the guards above.
    let delta = weighted_delta(cost, &staged, cfg);
    if accept(ctx, delta, temp) {
        cost.commit(staged);
        true
    } else {
        restore(ctx);
        false
    }
}

/// `try_swap_chain` -- move a whole cluster to a new base, displacing whatever
/// stands in the way (recursively, if a displaced cell is itself clustered).
fn try_swap_chain(
    ctx: &mut Context,
    cost: &mut CostModel,
    tmg: &TimingAnalyser,
    cfg: &RefineCfg,
    root: CellId,
    new_base: BelId,
    temp: f64,
) -> bool {
    // Original bel of every cell this move disturbs, for rollback.
    let mut moved: FxHashMap<CellId, BelId> = FxHashMap::default();
    let mut queue: Vec<(CellId, BelId)> = vec![(root, new_base)];
    let mut ok = true;

    'outer: while let Some((cluster, base)) = queue.pop() {
        let Some(dest) = ctx.cluster_placement(cluster, base) else {
            ok = false;
            break;
        };
        // Rip the cluster up first so its own bels count as free below.
        for &(cid, _) in &dest {
            if let Some(bel) = ctx.design.cell(cid).bel {
                moved.entry(cid).or_insert(bel);
                ctx.unbind_bel(bel);
            }
        }
        for &(cid, target) in &dest {
            let old_bel = moved.get(&cid).copied();
            let bound = ctx.conflicting_bel_cell(target);
            if let Some(b) = bound {
                let Some(old_bel) = old_bel else {
                    ok = false;
                    break 'outer;
                };
                if !ctx.bel(old_bel).is_available() {
                    ok = false;
                    break 'outer;
                }
                if moved.contains_key(&b) || ctx.design.cell(b).bel_strength > PlaceStrength::Strong
                {
                    // Already moved this pass, or pinned: give up rather than
                    // move a cell twice.
                    ok = false;
                    break 'outer;
                }
                match ctx.design.cell(b).cluster {
                    Some(other_root) => {
                        // Displace the whole cluster, keeping its shape, by the
                        // same vector that moves `b` into the vacated slot.
                        let b_bel = ctx.design.cell(b).bel.expect("bound cell is placed");
                        let root_bel = ctx
                            .design
                            .cell(ctx.cluster_root_cell(other_root))
                            .bel
                            .expect("cluster root is placed");
                        let old_loc = ctx.chipdb().bel_loc(old_bel);
                        let b_loc = ctx.chipdb().bel_loc(b_bel);
                        let r_loc = ctx.chipdb().bel_loc(root_bel);
                        let new_loc = Loc {
                            x: old_loc.x + (r_loc.x - b_loc.x),
                            y: old_loc.y + (r_loc.y - b_loc.y),
                            z: old_loc.z + (r_loc.z - b_loc.z),
                        };
                        let Some(new_root_bel) = ctx.bel_by_location(new_loc) else {
                            ok = false;
                            break 'outer;
                        };
                        let members: Vec<CellId> = ctx
                            .design
                            .clusters
                            .get(&other_root)
                            .map(|c| c.members.clone())
                            .unwrap_or_default();
                        for m in members {
                            if let Some(mb) = ctx.design.cell(m).bel {
                                moved.entry(m).or_insert(mb);
                                ctx.unbind_bel(mb);
                            }
                        }
                        queue.push((other_root, new_root_bel));
                    }
                    None => {
                        moved.entry(b).or_insert(b_bel_of(ctx, b));
                        if let Some(bb) = ctx.design.cell(b).bel {
                            ctx.unbind_bel(bb);
                        }
                        ctx.bind_bel(old_bel, b, PlaceStrength::Weak);
                    }
                }
            } else if !ctx.bel(target).is_available() {
                ok = false;
                break 'outer;
            }
            let strength = match ctx.design.cell(cid).cluster {
                Some(_) => PlaceStrength::Strong,
                None => PlaceStrength::Weak,
            };
            if !ctx.bind_bel(target, cid, strength) {
                ok = false;
                break 'outer;
            }
        }
    }

    let rollback = |ctx: &mut Context, moved: &FxHashMap<CellId, BelId>| {
        for (&cid, _) in moved.iter() {
            if let Some(bel) = ctx.design.cell(cid).bel {
                ctx.unbind_bel(bel);
            }
        }
        for (&cid, &bel) in moved.iter() {
            let strength = match ctx.design.cell(cid).cluster {
                Some(_) => PlaceStrength::Strong,
                None => PlaceStrength::Weak,
            };
            ctx.bind_bel(bel, cid, strength);
        }
    };

    if !ok {
        rollback(ctx, &moved);
        return false;
    }

    // Every disturbed cell must land somewhere legal and inside its region.
    let disturbed: Vec<CellId> = moved.keys().copied().collect();
    for &cid in &disturbed {
        let Some(bel) = ctx.design.cell(cid).bel else {
            rollback(ctx, &moved);
            return false;
        };
        if !ctx.is_bel_location_valid(bel) {
            rollback(ctx, &moved);
            return false;
        }
        if let Some(rid) = ctx.design.cell(cid).region {
            if !ctx.is_bel_in_region(bel, rid) {
                rollback(ctx, &moved);
                return false;
            }
        }
    }

    let staged = cost.stage(ctx, tmg, cfg, &disturbed);
    let delta = weighted_delta(cost, &staged, cfg);
    if accept(ctx, delta, temp) {
        cost.commit(staged);
        true
    } else {
        rollback(ctx, &moved);
        false
    }
}

fn b_bel_of(ctx: &Context, cell: CellId) -> BelId {
    ctx.design.cell(cell).bel.expect("bound cell is placed")
}

/// The blended objective: timing and wirelength deltas each normalised by their
/// own current total, so neither unit dominates the other.
fn weighted_delta(cost: &CostModel, staged: &super::cost::Staged, cfg: &RefineCfg) -> f64 {
    const EPSILON: f64 = 1e-20;
    cfg.lambda * (staged.timing_delta / cost.timing.max(EPSILON))
        + (1.0 - cfg.lambda) * (staged.wirelen_delta as f64 / (cost.wirelen as f64).max(EPSILON))
}

/// Metropolis acceptance. At the refinement temperature this is downhill-only
/// in all but a vanishing fraction of cases, but it is not *exactly* greedy and
/// upstream's `temp > 1e-8` guard is what decides that.
fn accept(ctx: &mut Context, delta: f64, temp: f64) -> bool {
    if delta < 0.0 {
        return true;
    }
    if temp <= 1e-8 {
        return false;
    }
    let r = ctx.rng_mut().rng() as f64 / 0x3fff_ffff as f64;
    r <= (-delta / temp).exp()
}

fn fabric_extent(ctx: &Context) -> (i32, i32) {
    (
        (ctx.chipdb().width() as i32 - 1).max(1),
        (ctx.chipdb().height() as i32 - 1).max(1),
    )
}

/// Bucket every candidate bel by type and location once, up front.
fn build_bel_grids(
    ctx: &Context,
    autoplaced: &[CellId],
    chain_basis: &[CellId],
) -> FxHashMap<IdString, BelGrid> {
    let types: FxHashSet<IdString> = autoplaced
        .iter()
        .chain(chain_basis)
        .map(|&c| ctx.design.cell(c).cell_type)
        .collect();

    let w = ctx.chipdb().width() as usize;
    let h = ctx.chipdb().height() as usize;
    let mut grids: FxHashMap<IdString, BelGrid> = types
        .into_iter()
        .map(|t| {
            (
                t,
                BelGrid {
                    by_loc: vec![vec![Vec::new(); h]; w],
                    count: 0,
                },
            )
        })
        .collect();

    for bel in ctx.chipdb().bels() {
        let loc = ctx.chipdb().bel_loc(bel);
        for (&ty, grid) in grids.iter_mut() {
            if ctx.bel(bel).is_valid_for_cell_type(ty) {
                grid.by_loc[loc.x as usize][loc.y as usize].push(bel);
                grid.count += 1;
            }
        }
    }
    grids
}

//! Generic A* / Dijkstra search over the chipdb routing graph.
//!
//! This module is the single shared kernel used by every A* caller in the
//! router tree (maze, router2, raster::segment_astar, lookahead). Each caller
//! supplies a [`PathCostModel`] that defines the per-pip cost, the
//! wire-arrival penalty, and the heuristic; the kernel owns the heap, the
//! visited map, the node-as-vertex expansion and first-pop termination.
//!
//! # Graph model
//!
//! Vertices are *routing nodes* — one electrical wire that may span multiple
//! tiles. All [`WireId`] entries returned by
//! [`ChipDb::node_wires_cb`](crate::chipdb::ChipDb::node_wires_cb) for a given
//! wire belong to the same node. Moving between node peers is free (hop cost
//! 0): peers share a single electrical conductor.
//!
//! Edges are PIPs. Each PIP lives in a single tile and connects two wires in
//! that tile; one of those wires may be a node peer in a different tile.
//!
//! # Why node-as-vertex
//!
//! Treating each tile-wire as a separate vertex would have the algorithm
//! push every peer of a large node onto the heap each time any member is
//! popped, which is wasteful — all peers have the same g-cost as their
//! first-popped member. A 65k-peer clock-distribution node would generate
//! 65k redundant heap operations per pop; total work would be O(N²).
//!
//! Instead, `astar_search` expands each node exactly once: the first member
//! to pop triggers a walk of every member's `pips_downhill`, marks all
//! members visited at that cost, and marks the node id in
//! `nodes_expanded` so subsequent pops of peer wires skip the walk.
//!
//! # Admissibility
//!
//! For the first-pop-optimal break to be sound the heuristic must be
//! admissible (`h(w) ≤ true_cost(w, dst)`). The current
//! [`crate::router::lookahead::Lookahead`] returns
//! `manhattan × per_tile_cost` where `per_tile_cost` is the chipdb's
//! cheapest pip class, which is a true lower bound on any uncongested
//! route's cost.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{hash_map::Entry, BinaryHeap};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::chipdb::{ChipDb, PipId, WireId};
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::timing::DelayT;

// ---------------------------------------------------------------------------
// Cost model trait
// ---------------------------------------------------------------------------

/// User-supplied cost + heuristic for an A* search. Implementors must ensure
/// the heuristic is admissible with respect to `pip_cost` + `wire_penalty`;
/// otherwise [`astar_search`] may return a suboptimal path.
pub trait PathCostModel {
    /// Cost of traversing `pip` (edge cost from its source wire to its dst
    /// wire). Must be non-negative.
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT;

    /// Additional cost paid on arriving at `wire`, e.g. congestion penalty.
    /// Returns 0 by default.
    #[inline]
    fn wire_penalty(&self, _ctx: &Context, _wire: WireId) -> DelayT {
        0
    }

    /// Admissible lower-bound on the remaining cost from `wire` to `dst`.
    /// Return `0` to reduce the search to Dijkstra.
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT;

    /// Return `true` to exclude `wire` from expansion entirely. Used by
    /// routing cost models to hard-block wires bound to a different net: a
    /// soft penalty (e.g. `DelayT::MAX / 4`) still lets A* waste its visit
    /// budget exploring the reduced graph one-pop-at-a-time, whereas a hard
    /// skip causes A* to drain the heap quickly when no unblocked path
    /// exists so the caller can trigger rip-up. Defaults to `false`.
    #[inline]
    fn is_blocked(&self, _ctx: &Context, _wire: WireId) -> bool {
        false
    }

    /// Tiles lying outside **every** box in this slice are rejected. An
    /// empty slice means no spatial pruning. Multiple boxes express a
    /// union — e.g. a corridor-shaped region covered by one small box per
    /// raster path point. The kernel calls this before dereferencing PIP
    /// destinations, so the scan must be cheap; callers with a single
    /// rectangular bbox can return `std::slice::from_ref(&bb)`.
    #[inline]
    fn bboxes(&self) -> &[BoundingBox] {
        &[]
    }
}

// ---------------------------------------------------------------------------
// Search options, result, and trace
// ---------------------------------------------------------------------------

/// Optional parameters for [`astar_search`].
#[derive(Clone, Copy, Debug)]
pub struct AStarOptions {
    /// Maximum number of wire pops before the search aborts. `None` means
    /// use the kernel default of `grid_area * 10`.
    pub visit_limit: Option<usize>,
    /// Stop-on-dst disabled: run until the heap drains or the visit limit
    /// fires. Intended for building lookup tables (Dijkstra mode).
    pub exhaustive: bool,
    /// Retain the full visited map in the returned trace. Callers that only
    /// need `path`/exit status can disable this to reduce per-wire state.
    pub retain_trace: bool,
    /// Stop the moment `dst_wire` is first inserted into the visited map,
    /// without waiting for it to pop off the heap. Intended for weighted
    /// (non-admissible) A* / greedy-best-first searches where the caller
    /// explicitly accepts suboptimality in exchange for speed. Ignored when
    /// `exhaustive` is set.
    pub stop_on_first_touch: bool,
}

impl Default for AStarOptions {
    fn default() -> Self {
        Self {
            visit_limit: None,
            exhaustive: false,
            retain_trace: true,
            stop_on_first_touch: false,
        }
    }
}

/// Why the search loop exited.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AStarExit {
    /// `dst` was popped (or was a peer of a popped node). Optimal under an
    /// admissible heuristic.
    Reached,
    /// Heap was drained without reaching `dst`.
    HeapDrained,
    /// Visit-count budget was exhausted before `dst` was reached.
    VisitLimit,
}

/// Full diagnostic output of a search.
pub struct AStarTrace {
    /// Total pops of the heap, including stale entries.
    pub visit_count: usize,
    /// Budget in effect at exit time.
    pub max_visits: usize,
    /// Best `cost + penalty` at `dst`, or `DelayT::MAX` if never reached.
    pub best_score: DelayT,
    /// Reason the loop exited.
    pub exit: AStarExit,
    /// Full visited map: `wire -> (cost, penalty, parent_pip, came_from_wire)`.
    /// `parent_pip` is `None` when the wire was reached by a free node hop
    /// (the parent pip belongs to `came_from_wire`'s traversal). Used for
    /// path reconstruction and post-mortem diagnostics.
    pub visited: FxHashMap<WireId, (DelayT, DelayT, Option<PipId>, WireId)>,
}

/// Result of a call to [`astar_search`].
pub struct AStarResult {
    /// List of PIPs from source to destination, empty if `dst` was itself a
    /// source wire, `None` if `dst` was never reached.
    pub path: Option<Vec<PipId>>,
    /// Detailed trace of the search, always populated.
    pub trace: AStarTrace,
}

#[inline]
fn bbox_area(bb: &BoundingBox) -> usize {
    let width = (bb.x1 - bb.x0 + 1).max(0) as usize;
    let height = (bb.y1 - bb.y0 + 1).max(0) as usize;
    width.saturating_mul(height)
}

#[inline]
fn default_visit_limit_for_search(
    grid_area: usize,
    min_manhattan: usize,
    region_tiles: Option<usize>,
) -> usize {
    let distance_budget = min_manhattan
        .saturating_add(8)
        .saturating_pow(2)
        .saturating_mul(128)
        .max(10_000);
    let region_budget = region_tiles
        .unwrap_or(grid_area)
        .saturating_mul(256)
        .max(10_000);
    distance_budget
        .min(region_budget)
        .min(grid_area.saturating_mul(10).max(100_000))
}

enum SearchRegion<'a> {
    Unbounded,
    Single(&'a BoundingBox),
    Mask(Vec<u8>),
}

impl<'a> SearchRegion<'a> {
    #[inline]
    fn contains_tile(&self, chipdb: &ChipDb, tile: i32) -> bool {
        match self {
            Self::Unbounded => true,
            Self::Single(bb) => {
                let (x, y) = chipdb.tile_xy(tile);
                bb.contains(x, y)
            }
            Self::Mask(mask) => usize::try_from(tile)
                .ok()
                .and_then(|idx| mask.get(idx))
                .is_some_and(|&bit| bit != 0),
        }
    }
}

fn build_search_region<'a>(
    chipdb: &ChipDb,
    bboxes: &'a [BoundingBox],
) -> (SearchRegion<'a>, Option<usize>) {
    match bboxes {
        [] => (SearchRegion::Unbounded, None),
        [bb] => (SearchRegion::Single(bb), Some(bbox_area(bb))),
        many => {
            let num_tiles = chipdb.num_tiles().max(0) as usize;
            let mut mask = vec![0u8; num_tiles];
            let mut covered_tiles = 0usize;
            for bb in many {
                for y in bb.y0..=bb.y1 {
                    for x in bb.x0..=bb.x1 {
                        let tile = chipdb.tile_by_xy(x, y);
                        let idx = tile as usize;
                        if mask[idx] == 0 {
                            mask[idx] = 1;
                            covered_tiles += 1;
                        }
                    }
                }
            }
            (SearchRegion::Mask(mask), Some(covered_tiles))
        }
    }
}

type TraceVisit = (DelayT, DelayT, Option<PipId>, WireId);
type ParentVisit = (Option<PipId>, WireId);

enum VisitState {
    Trace(FxHashMap<WireId, TraceVisit>),
    Fast {
        scores: FxHashMap<WireId, DelayT>,
        parents: FxHashMap<WireId, ParentVisit>,
    },
}

impl VisitState {
    fn with_capacity(cap: usize, retain_trace: bool) -> Self {
        if retain_trace {
            Self::Trace(FxHashMap::with_capacity_and_hasher(cap, Default::default()))
        } else {
            Self::Fast {
                scores: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
                parents: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
            }
        }
    }

    #[inline]
    fn score(&self, wire: WireId) -> Option<DelayT> {
        match self {
            Self::Trace(visited) => visited
                .get(&wire)
                .map(|&(cost, penalty, _, _)| cost + penalty),
            Self::Fast { scores, .. } => scores.get(&wire).copied(),
        }
    }

    fn insert_source(&mut self, wire: WireId) {
        match self {
            Self::Trace(visited) => {
                visited.insert(wire, (0, 0, None, wire));
            }
            Self::Fast { scores, parents } => {
                scores.insert(wire, 0);
                parents.insert(wire, (None, wire));
            }
        }
    }

    fn update_peer(&mut self, wire: WireId, cost: DelayT, penalty: DelayT, from: WireId) -> bool {
        let score = cost + penalty;
        match self {
            Self::Trace(visited) => match visited.entry(wire) {
                Entry::Occupied(mut entry) => {
                    let &(prev_cost, prev_penalty, _, _) = entry.get();
                    if prev_cost + prev_penalty <= score {
                        false
                    } else {
                        entry.insert((cost, penalty, None, from));
                        true
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert((cost, penalty, None, from));
                    true
                }
            },
            Self::Fast { scores, parents } => match scores.entry(wire) {
                Entry::Occupied(mut entry) => {
                    if *entry.get() <= score {
                        false
                    } else {
                        entry.insert(score);
                        parents.insert(wire, (None, from));
                        true
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(score);
                    parents.insert(wire, (None, from));
                    true
                }
            },
        }
    }

    fn update_step(
        &mut self,
        wire: WireId,
        cost: DelayT,
        penalty: DelayT,
        pip: PipId,
        from: WireId,
    ) -> bool {
        let score = cost + penalty;
        match self {
            Self::Trace(visited) => match visited.entry(wire) {
                Entry::Occupied(mut entry) => {
                    let &(prev_cost, prev_penalty, _, _) = entry.get();
                    if prev_cost + prev_penalty <= score {
                        false
                    } else {
                        entry.insert((cost, penalty, Some(pip), from));
                        true
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert((cost, penalty, Some(pip), from));
                    true
                }
            },
            Self::Fast { scores, parents } => match scores.entry(wire) {
                Entry::Occupied(mut entry) => {
                    if *entry.get() <= score {
                        false
                    } else {
                        entry.insert(score);
                        parents.insert(wire, (Some(pip), from));
                        true
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(score);
                    parents.insert(wire, (Some(pip), from));
                    true
                }
            },
        }
    }

    fn reconstruct_path(&self, chipdb: &ChipDb, end: WireId) -> Option<Vec<PipId>> {
        match self {
            Self::Trace(visited) => reconstruct_path_to(chipdb, visited, end),
            Self::Fast { parents, .. } => reconstruct_path_from_parents(chipdb, parents, end),
        }
    }

    fn into_trace_visited(self) -> FxHashMap<WireId, TraceVisit> {
        match self {
            Self::Trace(visited) => visited,
            Self::Fast { .. } => FxHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Priority queue entry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct QueueEntry {
    wire: WireId,
    cost: DelayT,
    penalty: DelayT,
    estimate: DelayT,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}
impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower estimate pops first, break ties by lower cost so
        // paths closer to the source win when f-values are equal.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run an A* search (or Dijkstra when the model's heuristic is zero) from
/// any wire in `src_wires` to `dst_wire`.
///
/// The path reconstruction in the returned [`AStarResult::path`] is a list
/// of PIPs. Consecutive PIPs whose dst-wire and next src-wire belong to the
/// same routing node are connected by an implicit free node hop; the
/// validator and `apply_route_plan` already handle this.
pub fn astar_search<M: PathCostModel>(
    ctx: &Context,
    model: &M,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    opts: &AStarOptions,
) -> AStarResult {
    let chipdb = ctx.chipdb();

    // Fast-path: `dst` is already a source (trivial zero-length route). Only
    // valid in non-exhaustive mode — exhaustive callers (e.g. the lookahead
    // builder) pass `src_wire` as a placeholder `dst_wire` to run Dijkstra
    // to budget exhaustion, and must not short-circuit here.
    if !opts.exhaustive && src_wires.contains(&dst_wire) {
        return AStarResult {
            path: Some(Vec::new()),
            trace: AStarTrace {
                visit_count: 0,
                max_visits: 0,
                best_score: 0,
                exit: AStarExit::Reached,
                visited: if opts.retain_trace {
                    let mut visited = FxHashMap::default();
                    visited.insert(dst_wire, (0, 0, None, dst_wire));
                    visited
                } else {
                    FxHashMap::default()
                },
            },
        };
    }

    let init_cap = src_wires.len().saturating_mul(8).max(16);
    let mut heap: BinaryHeap<QueueEntry> = BinaryHeap::with_capacity(init_cap);
    let mut visited = VisitState::with_capacity(init_cap, opts.retain_trace);
    let mut nodes_expanded: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(init_cap, Default::default());

    let grid_area = (chipdb.width() as usize) * (chipdb.height() as usize);
    let bboxes = model.bboxes();
    let (search_region, region_tiles) = build_search_region(chipdb, bboxes);
    let (dst_x, dst_y) = chipdb.tile_xy(dst_wire.tile());
    let min_manhattan = src_wires
        .iter()
        .map(|&src| {
            let (sx, sy) = chipdb.tile_xy(src.tile());
            ((sx - dst_x).abs() + (sy - dst_y).abs()) as usize
        })
        .min()
        .unwrap_or(0);
    let max_visits = opts
        .visit_limit
        .unwrap_or_else(|| default_visit_limit_for_search(grid_area, min_manhattan, region_tiles));

    // Whether dst is ever a peer of the node currently being expanded.
    let dst_node_id = chipdb.node_id(dst_wire);

    // Seed: push each source wire onto the heap. The first-pop handler
    // fires the same dst-peer short-circuit as any other node expansion,
    // so we don't need to walk every peer of every source up front. That
    // matters on mega-nodes where a source's peer set is in the tens of
    // thousands — walking it here would cost O(peers × sources) at seed
    // time, for no benefit beyond what the first pop already does.
    for &src in src_wires {
        let hval = model.heuristic(ctx, src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            penalty: 0,
            estimate: hval,
        });
        visited.insert_source(src);
    }

    let mut visit_count: usize = 0;
    let mut best_score: DelayT = DelayT::MAX;
    let mut hit_visit_limit = false;
    let mut reached = false;
    let stop_on_touch = opts.stop_on_first_touch && !opts.exhaustive;

    while let Some(entry) = heap.pop() {
        visit_count += 1;
        if visit_count > max_visits {
            hit_visit_limit = true;
            break;
        }

        // Stale skip: if we already reached this wire at a strictly lower
        // score, the current entry is obsolete.
        if let Some(prev_score) = visited.score(entry.wire) {
            if entry.cost + entry.penalty > prev_score {
                continue;
            }
        }

        // Destination reached: first pop is optimal under admissible h
        // when node hops and pip costs are non-negative.
        if !opts.exhaustive && entry.wire == dst_wire {
            best_score = entry.cost + entry.penalty;
            reached = true;
            break;
        }

        // Node-as-vertex expansion. Walk every member's `pips_downhill`
        // exactly once per node.
        let nid = chipdb.node_id(entry.wire);
        let should_expand_node = match nid {
            Some(id) => nodes_expanded.insert(id),
            None => true, // tile-local: not a multi-tile node, always expand
        };

        if should_expand_node {
            // Early short-circuit: if `dst` is a peer of this node, there's
            // no routing work to do — BUFG_O and its clock-distribution
            // members are electrically one wire. Mark `dst` reached via a
            // free node hop (no pip) and break. Avoids walking tens of
            // thousands of peers on mega-nodes just to reach this same
            // conclusion after the fact.
            if !opts.exhaustive && nid.is_some() && dst_node_id == nid {
                visited.update_peer(dst_wire, entry.cost, entry.penalty, entry.wire);
                best_score = entry.cost + entry.penalty;
                reached = true;
                break;
            }

            // Shared flag across the closure and the outer loop. The closure
            // captures `&touched_dst` (Cell's interior mutability lets it
            // write through a shared borrow), so the outer `touched_dst.get()`
            // check after each call observes the latest value without
            // conflicting with the closure's other mutable captures.
            let touched_dst: Cell<bool> = Cell::new(false);

            // Enumerate members (the popped wire itself + every peer) so we
            // can iterate once without allocating a Vec.
            let mut expand_member = |member: WireId| {
                if touched_dst.get() {
                    return;
                }
                // Mark the member as reached at the current cost. A peer
                // reached via a free node hop stores None for the pip and
                // records the popped wire as came_from.
                if member != entry.wire {
                    visited.update_peer(member, entry.cost, entry.penalty, entry.wire);
                }

                // Expand pip_downhill for this member.
                let wire_info = chipdb.wire_info(member);
                for &pip_idx in wire_info.pips_downhill.get() {
                    let pip = PipId::new(member.tile(), pip_idx);
                    let next_wire = chipdb.pip_dst_wire(pip);

                    if !search_region.contains_tile(chipdb, next_wire.tile()) {
                        continue;
                    }

                    // Hard-block wires the cost model marks as impassable
                    // (e.g. bound to a different net). Skipping here keeps
                    // them out of the heap entirely so A* drains the heap
                    // quickly when no unblocked path exists — a soft
                    // penalty just wastes the visit budget.
                    if model.is_blocked(ctx, next_wire) {
                        continue;
                    }

                    let pip_cost = model.pip_cost(ctx, pip);
                    let penalty = model.wire_penalty(ctx, next_wire);
                    let new_cost = entry.cost + pip_cost;
                    let new_pen = entry.penalty + penalty;
                    let new_score = new_cost + new_pen;

                    if !visited.update_step(next_wire, new_cost, new_pen, pip, member) {
                        continue;
                    }

                    if stop_on_touch && next_wire == dst_wire {
                        touched_dst.set(true);
                        return;
                    }

                    let hval = model.heuristic(ctx, next_wire, dst_wire);
                    heap.push(QueueEntry {
                        wire: next_wire,
                        cost: new_cost,
                        penalty: new_pen,
                        estimate: new_score + hval,
                    });
                }
            };

            // Expand the popped wire itself.
            expand_member(entry.wire);

            // Then its peers, if any.
            if nid.is_some() && !touched_dst.get() {
                // node_wires_cb excludes `entry.wire` already.
                chipdb.node_wires_cb(entry.wire, &mut expand_member);
            }

            drop(expand_member);
            if touched_dst.get() {
                best_score = visited.score(dst_wire).unwrap_or(DelayT::MAX);
                reached = true;
                break;
            }
        }
    }

    let exit = if reached {
        AStarExit::Reached
    } else if hit_visit_limit {
        AStarExit::VisitLimit
    } else {
        AStarExit::HeapDrained
    };

    // Reconstruct path if dst was reached. Walk backwards through
    // (parent_pip, came_from_wire). A free node hop appears as `pip = None`;
    // we step to `came_from` without emitting a pip.
    let path = if reached {
        visited.reconstruct_path(chipdb, dst_wire)
    } else {
        None
    };

    AStarResult {
        path,
        trace: AStarTrace {
            visit_count,
            max_visits,
            best_score,
            exit,
            visited: visited.into_trace_visited(),
        },
    }
}

/// Canonical pip cost used by the router and the lookahead: raw pip delay
/// plus a unit step so every PIP edge is strictly positive. Extracted here
/// so every PathCostModel that wants the default behaviour can delegate to
/// this single function — keeps the router and lookahead in lockstep.
#[inline]
pub fn default_pip_cost(ctx: &Context, pip: PipId) -> DelayT {
    ctx.pip(pip).delay().max_delay() + 1
}

/// Reconstruct the PIP chain ending at `end` from a populated `visited`
/// map. Used by callers whose real goal is "any wire in a target set" —
/// they run [`astar_search`] with a representative wire as anchor, then
/// scan `trace.visited` for the best-scoring goal wire and reconstruct a
/// path to it. Returns `None` if `end` is not in `visited`.
pub fn reconstruct_path_to(
    chipdb: &ChipDb,
    visited: &FxHashMap<WireId, (DelayT, DelayT, Option<PipId>, WireId)>,
    end: WireId,
) -> Option<Vec<PipId>> {
    if !visited.contains_key(&end) {
        return None;
    }
    let mut pips: Vec<PipId> = Vec::new();
    let mut cursor = end;
    loop {
        let Some(&(_, _, pip, from)) = visited.get(&cursor) else {
            break;
        };
        match pip {
            Some(p) => {
                pips.push(p);
                cursor = chipdb.pip_src_wire(p);
            }
            None => {
                if from == cursor {
                    break;
                }
                cursor = from;
            }
        }
    }
    pips.reverse();
    Some(pips)
}

fn reconstruct_path_from_parents(
    chipdb: &ChipDb,
    parents: &FxHashMap<WireId, ParentVisit>,
    end: WireId,
) -> Option<Vec<PipId>> {
    if !parents.contains_key(&end) {
        return None;
    }
    let mut pips = Vec::new();
    let mut cursor = end;
    loop {
        let Some(&(pip, from)) = parents.get(&cursor) else {
            break;
        };
        match pip {
            Some(p) => {
                pips.push(p);
                cursor = chipdb.pip_src_wire(p);
            }
            None => {
                if from == cursor {
                    break;
                }
                cursor = from;
            }
        }
    }
    pips.reverse();
    Some(pips)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial manhattan-per-tile model used by integration tests that have
    // a real ChipDb; pure-algorithm tests below don't instantiate it.

    #[test]
    fn queue_entry_orders_by_estimate_then_cost() {
        let a = QueueEntry {
            wire: WireId::new(0, 0),
            cost: 10,
            penalty: 0,
            estimate: 100,
        };
        let b = QueueEntry {
            wire: WireId::new(0, 1),
            cost: 5,
            penalty: 0,
            estimate: 100,
        };
        // Both have estimate=100; b has lower cost so b should pop first in
        // a min-heap (i.e. b > a under the Ord impl that reverses order).
        assert!(b > a);

        let c = QueueEntry {
            wire: WireId::new(0, 2),
            cost: 1000,
            penalty: 0,
            estimate: 50,
        };
        // c has a strictly lower estimate, so c should pop before both.
        assert!(c > a);
        assert!(c > b);
    }

    #[test]
    fn default_visit_limit_scales_with_distance() {
        let grid_area = 10_000;
        let short = default_visit_limit_for_search(grid_area, 4, None);
        let long = default_visit_limit_for_search(grid_area, 80, None);
        assert!(short >= 10_000);
        assert!(long > short);
        assert!(long <= grid_area.saturating_mul(10));
    }

    #[test]
    fn default_visit_limit_respects_pruned_region() {
        let grid_area = 1_000_000;
        let unconstrained = default_visit_limit_for_search(grid_area, 60, None);
        let pruned = default_visit_limit_for_search(grid_area, 60, Some(64));
        assert!(pruned >= 10_000);
        assert!(pruned < unconstrained);
    }

    #[test]
    fn astar_options_default_retains_trace() {
        let opts = AStarOptions::default();
        assert!(opts.retain_trace);
        assert!(!opts.exhaustive);
        assert!(!opts.stop_on_first_touch);
        assert!(opts.visit_limit.is_none());
    }
}

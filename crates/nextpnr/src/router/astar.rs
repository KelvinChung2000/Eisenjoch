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
//! tiles — identified by their canonical wire (see
//! [`ChipDb::canonical_wire`](crate::chipdb::ChipDb::canonical_wire)). Edges
//! are PIPs. Each PIP lives in one tile, and expanding a node walks the pips
//! driven from every tile the node crosses.
//!
//! A node is one vertex with one visited entry however many tiles it spans,
//! so search state grows with the nodes explored, not with their length. A
//! chip-spanning clock node is one pop, not tens of thousands.
//!
//! Each queue entry also carries the tile wire the search entered the node
//! through. Position-dependent decisions — the search region and the
//! heuristic — use that, because a long node's root can be far from where
//! the search actually is.
//!
//! # Admissibility
//!
//! For the first-pop-optimal break to be sound the heuristic must be
//! admissible (`h(w) ≤ true_cost(w, dst)`). The current
//! [`crate::router::lookahead::Lookahead`] returns
//! `manhattan × per_tile_cost` where `per_tile_cost` is the chipdb's
//! cheapest pip class, which is a true lower bound on any uncongested
//! route's cost.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, BinaryHeap};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::chipdb::{ChipDb, PipId, WireId};
use crate::common::DelayT;
use crate::context::Context;
use crate::metrics::BoundingBox;

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
    /// Smallest manhattan tile distance to `dst` over every wire popped. A
    /// failure that got within a tile or two of the sink and still exhausted
    /// its budget was walled at the pin rather than starved en route, and the
    /// two want opposite fixes. `usize::MAX` when nothing was popped.
    pub min_approach: usize,
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
    Mask(RegionMask),
}

/// Tiles covered by a union of boxes, stored over the union's bounding
/// rectangle.
struct RegionMask {
    x0: i32,
    y0: i32,
    width: usize,
    height: usize,
    bits: Vec<bool>,
}

impl RegionMask {
    /// The union of `boxes` clipped to a `w` x `h` grid, and how many tiles
    /// it covers. Storage is the union's bounding rectangle: a corridor
    /// search costs its corridor, whatever the device.
    fn from_boxes(boxes: &[BoundingBox], w: i32, h: i32) -> (Self, usize) {
        let x0 = boxes.iter().map(|b| b.x0).min().unwrap_or(0).max(0);
        let y0 = boxes.iter().map(|b| b.y0).min().unwrap_or(0).max(0);
        let x1 = boxes.iter().map(|b| b.x1).max().unwrap_or(-1).min(w - 1);
        let y1 = boxes.iter().map(|b| b.y1).max().unwrap_or(-1).min(h - 1);
        let width = (x1 - x0 + 1).max(0) as usize;
        let height = (y1 - y0 + 1).max(0) as usize;
        let mut bits = vec![false; width * height];
        let mut covered = 0usize;
        for bb in boxes {
            for y in bb.y0.max(y0)..=bb.y1.min(y1) {
                for x in bb.x0.max(x0)..=bb.x1.min(x1) {
                    let idx = (y - y0) as usize * width + (x - x0) as usize;
                    if !bits[idx] {
                        bits[idx] = true;
                        covered += 1;
                    }
                }
            }
        }
        let mask = Self {
            x0,
            y0,
            width,
            height,
            bits,
        };
        (mask, covered)
    }

    #[inline]
    fn contains(&self, x: i32, y: i32) -> bool {
        let (dx, dy) = (x - self.x0, y - self.y0);
        dx >= 0
            && dy >= 0
            && (dx as usize) < self.width
            && (dy as usize) < self.height
            && self.bits[dy as usize * self.width + dx as usize]
    }
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
            Self::Mask(mask) => {
                let (x, y) = chipdb.tile_xy(tile);
                mask.contains(x, y)
            }
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
            let (mask, covered_tiles) =
                RegionMask::from_boxes(many, chipdb.width(), chipdb.height());
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
    /// The node, as its canonical wire.
    wire: WireId,
    /// The tile wire the search entered `wire`'s node through. A node can
    /// span the chip; this is where the search actually is.
    at: WireId,
    cost: DelayT,
    penalty: DelayT,
    estimate: DelayT,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
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
        // Min-heap: lower estimate pops first, then lower cost so paths closer
        // to the source win. The wire keys make the order total, so the pop
        // order never depends on the hash order the sources were pushed in.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| other.penalty.cmp(&self.penalty))
            .then_with(|| other.wire.raw().cmp(&self.wire.raw()))
            .then_with(|| other.at.raw().cmp(&self.at.raw()))
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
                min_approach: 0,
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
    let mut nodes_expanded: FxHashSet<WireId> =
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

    for &src in src_wires {
        let hval = model.heuristic(ctx, src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            at: src,
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
    let mut min_approach = usize::MAX;
    let stop_on_touch = opts.stop_on_first_touch && !opts.exhaustive;

    while let Some(entry) = heap.pop() {
        visit_count += 1;
        if visit_count > max_visits {
            hit_visit_limit = true;
            break;
        }

        let (ex, ey) = chipdb.tile_xy(entry.at.tile());
        min_approach = min_approach.min(((ex - dst_x).abs() + (ey - dst_y).abs()) as usize);

        // Stale skip: if we already reached this node at a strictly lower
        // score, the current entry is obsolete.
        if let Some(prev_score) = visited.score(entry.wire) {
            if entry.cost + entry.penalty > prev_score {
                continue;
            }
        }

        // Destination reached: first pop is optimal under admissible h
        // when pip costs are non-negative.
        if !opts.exhaustive && entry.wire == dst_wire {
            best_score = entry.cost + entry.penalty;
            reached = true;
            break;
        }

        // Expand each node once, from every tile it spans: the pips any
        // member drives all leave the same conductor.
        if !nodes_expanded.insert(entry.wire) {
            continue;
        }

        let mut touched_dst = false;
        chipdb.pips_downhill_cb(entry.wire, |pip| {
            if touched_dst {
                return;
            }
            // A pip sits in one tile; that is where the search would be.
            if !search_region.contains_tile(chipdb, pip.tile()) {
                return;
            }
            let next_wire = chipdb.pip_dst_wire(pip);

            // Hard-block wires the cost model marks as impassable (e.g.
            // bound to a different net). Skipping here keeps them out of the
            // heap entirely so A* drains the heap quickly when no unblocked
            // path exists — a soft penalty just wastes the visit budget.
            if model.is_blocked(ctx, next_wire) {
                return;
            }

            let pip_cost = model.pip_cost(ctx, pip);
            let penalty = model.wire_penalty(ctx, next_wire);
            let new_cost = entry.cost + pip_cost;
            let new_pen = entry.penalty + penalty;
            let new_score = new_cost + new_pen;

            if !visited.update_step(next_wire, new_cost, new_pen, pip, entry.wire) {
                return;
            }

            if stop_on_touch && next_wire == dst_wire {
                touched_dst = true;
                return;
            }

            let at = chipdb.pip_dst_tile_wire(pip);
            let hval = model.heuristic(ctx, at, dst_wire);
            heap.push(QueueEntry {
                wire: next_wire,
                at,
                cost: new_cost,
                penalty: new_pen,
                estimate: new_score + hval,
            });
        });

        if touched_dst {
            best_score = visited.score(dst_wire).unwrap_or(DelayT::MAX);
            reached = true;
            break;
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
            min_approach,
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
    while let Some(&(_, _, pip, from)) = visited.get(&cursor) {
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
    while let Some(&(pip, from)) = parents.get(&cursor) {
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
    fn region_mask_covers_the_union_and_is_sized_by_it() {
        let bb = |x0, y0, x1, y1| BoundingBox { x0, y0, x1, y1 };
        let boxes = [bb(10, 10, 11, 11), bb(11, 11, 13, 12), bb(-5, 12, 10, 12)];
        for (w, h) in [(20, 20), (10_000, 10_000)] {
            let (mask, covered) = RegionMask::from_boxes(&boxes, w, h);
            // x 0..=13, y 10..=12 whatever the grid: the device size must
            // not reach the per-search allocation.
            assert_eq!((mask.width, mask.height), (14, 3));
            assert_eq!(covered, 4 + 6 - 1 + 11);
            assert!(mask.contains(10, 10) && mask.contains(13, 12) && mask.contains(0, 12));
            assert!(!mask.contains(12, 10) && !mask.contains(0, 11) && !mask.contains(14, 12));
            assert!(!mask.contains(-1, 12) && !mask.contains(5, 9));
        }
    }

    #[test]
    fn queue_entry_orders_by_estimate_then_cost() {
        let a = QueueEntry {
            wire: WireId::new(0, 0),
            at: WireId::new(0, 0),
            cost: 10,
            penalty: 0,
            estimate: 100,
        };
        let b = QueueEntry {
            wire: WireId::new(0, 1),
            at: WireId::new(0, 1),
            cost: 5,
            penalty: 0,
            estimate: 100,
        };
        // Both have estimate=100; b has lower cost so b should pop first in
        // a min-heap (i.e. b > a under the Ord impl that reverses order).
        assert!(b > a);

        let c = QueueEntry {
            wire: WireId::new(0, 2),
            at: WireId::new(0, 2),
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

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
//! admissible (`h(w) ≤ true_cost(w, dst)`) *and* the cost model used by the
//! heuristic must match the cost model used by this kernel. In particular
//! the lookahead's node hops must cost 0 — see
//! [`crate::router::lookahead::Lookahead::build`] for the matching Dijkstra.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::chipdb::{PipId, WireId};
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
#[derive(Clone, Copy, Debug, Default)]
pub struct AStarOptions {
    /// Maximum number of wire pops before the search aborts. `None` means
    /// use the kernel default of `grid_area * 10`.
    pub visit_limit: Option<usize>,
    /// Stop-on-dst disabled: run until the heap drains or the visit limit
    /// fires. Intended for building lookup tables (Dijkstra mode).
    pub exhaustive: bool,
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

    if src_wires.contains(&dst_wire) {
        let mut visited = FxHashMap::default();
        visited.insert(dst_wire, (0, 0, None, dst_wire));
        return AStarResult {
            path: Some(Vec::new()),
            trace: AStarTrace {
                visit_count: 0,
                max_visits: 0,
                best_score: 0,
                exit: AStarExit::Reached,
                visited,
            },
        };
    }

    let init_cap = src_wires.len().saturating_mul(8).max(16);
    let mut heap: BinaryHeap<QueueEntry> = BinaryHeap::with_capacity(init_cap);
    let mut visited: FxHashMap<WireId, (DelayT, DelayT, Option<PipId>, WireId)> =
        FxHashMap::with_capacity_and_hasher(init_cap, Default::default());
    let mut nodes_expanded: FxHashSet<u64> = FxHashSet::default();

    let grid_area = (chipdb.width() as usize) * (chipdb.height() as usize);
    let max_visits = opts
        .visit_limit
        .unwrap_or_else(|| grid_area.saturating_mul(10).max(100_000));

    // Whether dst is ever a peer of the node currently being expanded.
    let dst_node_id = chipdb.node_id(dst_wire);
    let bboxes = model.bboxes();
    let tile_fits = |w: WireId| -> bool {
        if bboxes.is_empty() {
            return true;
        }
        let (x, y) = chipdb.tile_xy(w.tile());
        bboxes.iter().any(|bb| bb.contains(x, y))
    };

    // Seed: push each source wire and eagerly mark every node peer as
    // visited at cost 0. Peers share the electrical wire; moving between
    // them is free. We still push one heap entry per source so the pop
    // loop runs the usual expansion step.
    for &src in src_wires {
        let hval = model.heuristic(ctx, src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            penalty: 0,
            estimate: hval,
        });
        visited.insert(src, (0, 0, None, src));
        // Eagerly mark peers as visited so repeated source checks and
        // cheap-early-exit paths see them. The actual pip expansion for
        // each peer happens on the first pop that expands this node.
        if chipdb.node_id(src).is_some() {
            chipdb.node_wires_cb(src, |nw| {
                if !visited.contains_key(&nw) {
                    visited.insert(nw, (0, 0, None, src));
                }
            });
        }
    }

    let mut visit_count: usize = 0;
    let mut best_score: DelayT = DelayT::MAX;
    let mut hit_visit_limit = false;
    let mut reached = false;

    while let Some(entry) = heap.pop() {
        visit_count += 1;
        if visit_count > max_visits {
            hit_visit_limit = true;
            break;
        }

        // Stale skip: if we already reached this wire at a strictly lower
        // score, the current entry is obsolete.
        if let Some(&(pc, pp, _, _)) = visited.get(&entry.wire) {
            if entry.cost + entry.penalty > pc + pp {
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
            // Enumerate members (the popped wire itself + every peer) so we
            // can iterate once without allocating a Vec.
            let mut expand_member = |member: WireId| {
                // Mark the member as reached at the current cost. A peer
                // reached via a free node hop stores None for the pip and
                // records the popped wire as came_from.
                if member != entry.wire {
                    let key = (entry.cost, entry.penalty, None, entry.wire);
                    match visited.get(&member) {
                        Some(&(pc, pp, _, _)) if pc + pp <= entry.cost + entry.penalty => {}
                        _ => {
                            visited.insert(member, key);
                        }
                    }
                }

                // Expand pip_downhill for this member.
                let wire_info = chipdb.wire_info(member);
                for &pip_idx in wire_info.pips_downhill.get() {
                    let pip = PipId::new(member.tile(), pip_idx);
                    let next_wire = chipdb.pip_dst_wire(pip);

                    if !tile_fits(next_wire) {
                        continue;
                    }

                    let pip_cost = model.pip_cost(ctx, pip);
                    let penalty = model.wire_penalty(ctx, next_wire);
                    let new_cost = entry.cost + pip_cost;
                    let new_pen = entry.penalty + penalty;

                    if let Some(&(pc, pp, _, _)) = visited.get(&next_wire) {
                        if pc + pp <= new_cost + new_pen {
                            continue;
                        }
                    }

                    visited.insert(
                        next_wire,
                        (new_cost, new_pen, Some(pip), member),
                    );

                    let hval = model.heuristic(ctx, next_wire, dst_wire);
                    heap.push(QueueEntry {
                        wire: next_wire,
                        cost: new_cost,
                        penalty: new_pen,
                        estimate: new_cost + new_pen + hval,
                    });
                }
            };

            // Expand the popped wire itself.
            expand_member(entry.wire);

            // Then its peers, if any.
            if nid.is_some() {
                // node_wires_cb excludes `entry.wire` already.
                chipdb.node_wires_cb(entry.wire, &mut expand_member);

                // dst may be a peer of this node — check once at the end.
                if !opts.exhaustive && dst_node_id == nid {
                    if let Some(&(pc, pp, _, _)) = visited.get(&dst_wire) {
                        best_score = pc + pp;
                        reached = true;
                        break;
                    }
                }
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
        let mut pips: Vec<PipId> = Vec::new();
        let mut cursor = dst_wire;
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
            visited,
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
}

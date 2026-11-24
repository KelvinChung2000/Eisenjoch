//! Nearest-first walk over one bucket's BELs.

use crate::chipdb::BelId;
use crate::context::BucketGrid;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Nearest-BEL walk for one cell, in `(dist_sq, enum_index)` order.
///
/// Emits exactly the sequence `candidate_list` produces for the same inputs,
/// but lazily: a cell that binds to its second candidate never looks at the
/// rest of its type. That is the whole point -- the materialized list was
/// `O(cells * bels)`, this is `O(bels)` with one cell's worth live at a time.
pub(crate) struct RingCandidates<'a> {
    /// The bucket's BELs; `enum_index` is a position in `grid.bels()`.
    grid: BucketGrid<'a>,
    /// One flag per enumeration index, true once the BEL is bound. Bound BELs
    /// are never pushed, so the heap holds only candidates the caller can
    /// actually use. On FPGA01 the walk pushed 3.08 million BELs that were
    /// already taken, 62 % of everything it emitted, for the caller to reject
    /// with `is_available`.
    taken: &'a [bool],
    target_x: f64,
    target_y: f64,
    cx: i32,
    cy: i32,
    /// Rings `0..=radius` have been pushed; `-1` before the first expansion.
    radius: i32,
    /// Smallest `dist_sq` any not-yet-pushed BEL can have. Recomputed once
    /// per expansion, not per `next()`.
    guard: f64,
    /// `radius` already covers the whole grid, so nothing more can arrive.
    exhausted: bool,
    /// `(dist_sq bits, enum_index)`. `BelId` is deliberately absent: it has no
    /// `Ord` (`chipdb/ids.rs:30`), and `enum_index` is already unique, so the
    /// pair is a total order on its own.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl<'a> RingCandidates<'a> {
    pub(crate) fn new(
        grid: BucketGrid<'a>,
        taken: &'a [bool],
        target_x: f64,
        target_y: f64,
    ) -> Self {
        assert_eq!(
            taken.len(),
            grid.bels().len(),
            "one taken flag per enumeration index"
        );
        let mut walk = Self {
            grid,
            taken,
            target_x,
            target_y,
            cx: target_x.round() as i32,
            cy: target_y.round() as i32,
            radius: -1,
            guard: 0.0,
            exhausted: false,
            heap: BinaryHeap::new(),
        };
        walk.guard = walk.ring_min_dist_sq(0);
        walk
    }

    /// How many candidates are buffered but not yet emitted. Test-facing:
    /// it is how laziness is asserted.
    #[cfg(test)]
    pub(crate) fn buffered(&self) -> usize {
        self.heap.len()
    }

    /// The one distance expression in this file. Identical to the sorted
    /// path's, so ties compare *equal* rather than nearly-equal and the two
    /// orderings agree bitwise.
    #[inline]
    fn dist_sq(&self, x: i32, y: i32) -> f64 {
        let dx = x as f64 - self.target_x;
        let dy = y as f64 - self.target_y;
        dx * dx + dy * dy
    }

    /// Lower bound on `dist_sq` for every cell at Chebyshev radius `>= r`.
    ///
    /// Ring `r`'s closest cells to the target are the midpoints of its four
    /// sides -- the closest point of a square ring is the perpendicular foot
    /// on its nearest side, and `|cx - target_x| <= 0.5` puts the nearest
    /// integer cell to that foot at the midpoint. Evaluating those four with
    /// `dist_sq` makes the bound exact in f64 instead of an epsilon argument:
    /// rounding is weakly monotone componentwise, and any cell at radius
    /// `>= r` dominates the same-side axis point in both `|dx|` and `|dy|`.
    fn ring_min_dist_sq(&self, r: i32) -> f64 {
        let (cx, cy) = (self.cx, self.cy);
        [
            self.dist_sq(cx + r, cy),
            self.dist_sq(cx - r, cy),
            self.dist_sq(cx, cy + r),
            self.dist_sq(cx, cy - r),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }

    fn push_tile(&mut self, x: i32, y: i32) {
        let entries = self.grid.span(x, y);
        if entries.is_empty() {
            return;
        }
        let bits = self.dist_sq(x, y).to_bits();
        for e in entries {
            if self.taken[e] {
                continue;
            }
            self.heap.push(Reverse((bits, e as u32)));
        }
    }

    /// Push the next ring outward and re-derive the guard.
    fn expand(&mut self) {
        self.radius += 1;
        let r = self.radius;
        let (cx, cy) = (self.cx, self.cy);
        if r == 0 {
            self.push_tile(cx, cy);
        } else {
            for x in (cx - r)..=(cx + r) {
                self.push_tile(x, cy - r);
                self.push_tile(x, cy + r);
            }
            for y in (cy - r + 1)..=(cy + r - 1) {
                self.push_tile(cx - r, y);
                self.push_tile(cx + r, y);
            }
        }
        self.guard = self.ring_min_dist_sq(r + 1);
        self.exhausted = cx - r <= 0
            && cy - r <= 0
            && cx + r >= self.grid.width() - 1
            && cy + r >= self.grid.height() - 1;
    }
}

impl Iterator for RingCandidates<'_> {
    /// `(BelId, enumeration index)`. The caller needs the index to mark the
    /// BEL taken once it binds there.
    type Item = (BelId, u32);

    fn next(&mut self) -> Option<(BelId, u32)> {
        loop {
            match self.heap.peek() {
                // Strictly `<`. On an exact tie an unexplored BEL could carry
                // a lower `enum_index` and has to be pushed before we commit
                // -- and computed ties are the common case, since a whole tile
                // shares one `dist_sq`.
                Some(&Reverse((bits, _)))
                    if self.exhausted || f64::from_bits(bits) < self.guard =>
                {
                    let Reverse((_, e)) = self.heap.pop().expect("just peeked");
                    return Some((self.grid.bels()[e as usize], e));
                }
                None if self.exhausted => return None,
                _ => self.expand(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `per_tile` BELs at every (x, y) of a `dim x dim` grid, in the order a
    /// bucket list holds them: tile index `y * dim + x`, then slot.
    fn grid_bels(dim: i32, per_tile: i32) -> Vec<BelId> {
        (0..dim * dim)
            .flat_map(|tile| (0..per_tile).map(move |z| BelId::new(tile, z)))
            .collect()
    }

    /// One BEL in each listed tile, in bucket order.
    fn bels_at(dim: i32, tiles: &[(i32, i32)]) -> Vec<BelId> {
        let mut bels: Vec<BelId> = tiles
            .iter()
            .map(|&(x, y)| BelId::new(y * dim + x, 0))
            .collect();
        bels.sort_by_key(|b| b.tile());
        bels
    }

    fn xy(bel: BelId, dim: i32) -> (i32, i32) {
        (bel.tile() % dim, bel.tile() / dim)
    }

    /// Every BEL must land in its own tile, and nowhere else.
    #[test]
    fn spans_cover_each_tile_exactly() {
        let bels = grid_bels(6, 3);
        let grid = BucketGrid::new(&bels, 6, 6);
        let mut seen = 0;
        for x in 0..6 {
            for y in 0..6 {
                let entries = grid.at(x, y);
                assert_eq!(entries.len(), 3, "tile ({x},{y})");
                for &b in entries {
                    assert_eq!(xy(b, 6), (x, y), "{b} filed under the wrong tile");
                }
                seen += entries.len();
            }
        }
        assert_eq!(seen, bels.len(), "grid dropped or duplicated entries");
    }

    /// Sparse layouts are the interesting case: most tiles hold nothing.
    #[test]
    fn empty_tiles_return_an_empty_span() {
        let bels = vec![BelId::new(3 * 8 + 2, 0), BelId::new(3 * 8 + 2, 1)];
        let grid = BucketGrid::new(&bels, 8, 8);
        assert_eq!(grid.at(2, 3).len(), 2);
        assert!(grid.at(0, 0).is_empty());
        assert!(grid.at(7, 7).is_empty());
    }

    /// The walk probes ring cells that fall off the edge of the device; the
    /// lookup must answer those without a bounds panic.
    #[test]
    fn out_of_bounds_lookups_are_empty_not_a_panic() {
        let bels = grid_bels(4, 2);
        let grid = BucketGrid::new(&bels, 4, 4);
        for (x, y) in [(-1, 0), (0, -1), (4, 0), (0, 4), (-7, 9)] {
            assert!(grid.span(x, y).is_empty(), "({x},{y})");
        }
    }

    /// Reference ordering: exactly what `candidate_list` computes, done the
    /// slow obvious way. Kept here so the geometry tests are self-contained;
    /// `sorted.rs` checks the walk against the real `candidate_list` too.
    fn expected_order(bels: &[BelId], dim: i32, tx: f64, ty: f64) -> Vec<BelId> {
        let mut v: Vec<(f64, usize, BelId)> = bels
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let (bx, by) = xy(id, dim);
                let dx = bx as f64 - tx;
                let dy = by as f64 - ty;
                (dx * dx + dy * dy, i, id)
            })
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        v.into_iter().map(|(_, _, id)| id).collect()
    }

    fn walk(bels: &[BelId], dim: i32, tx: f64, ty: f64) -> Vec<BelId> {
        let free = vec![false; bels.len()];
        RingCandidates::new(BucketGrid::new(bels, dim, dim), &free, tx, ty)
            .map(|(id, _)| id)
            .collect()
    }

    /// Square rings under a Euclidean metric: a BEL in ring `r` is not
    /// necessarily nearer than one in ring `r+1`. `(3,0)` sits in ring 3 at
    /// distance 3; `(2,2)` sits in ring 2 at distance ~2.83 -- but `(2,2)`'s
    /// ring is explored first, and `(1,3)` (ring 3, distance ~3.16) must NOT
    /// overtake `(3,0)`. Ordering has to come from the heap, not the rings.
    #[test]
    fn ring_order_is_not_distance_order() {
        let bels = bels_at(8, &[(3, 0), (2, 2), (1, 3), (0, 1)]);
        assert_eq!(walk(&bels, 8, 0.0, 0.0), expected_order(&bels, 8, 0.0, 0.0));
    }

    /// The tie the guard exists for: equal `dist_sq`, with the winner -- the
    /// lower enumeration index -- one ring FURTHER OUT.
    ///
    /// A fractional target is what makes this reachable. From `(1.5, 0.0)` the
    /// rings centre on `cx = round(1.5) = 2`, so `(3,0)` is ring 1 and `(0,0)`
    /// is ring 2 -- yet both are exactly 1.5 away. Releasing a candidate as
    /// soon as its ring is explored returns them backwards.
    #[test]
    fn an_exact_tie_is_won_by_the_lower_index_in_the_outer_ring() {
        // Index 0 at (0,0), ring 2; index 1 at (3,0), ring 1; both d^2 = 2.25.
        let bels = bels_at(5, &[(0, 0), (3, 0)]);
        let got = walk(&bels, 5, 1.5, 0.0);
        assert_eq!(got, expected_order(&bels, 5, 1.5, 0.0));
        assert_eq!(
            got,
            vec![BelId::new(0, 0), BelId::new(3, 0)],
            "tie must resolve on enumeration index, not on ring order"
        );
    }

    /// Fractional targets are the norm -- solver positions are continuous --
    /// and they are what makes `round()` and the `|cx - tx| <= 0.5` premise
    /// load-bearing.
    #[test]
    fn fractional_targets_match_the_reference_order() {
        let bels = grid_bels(9, 3);
        for &(tx, ty) in &[
            (0.0, 0.0),
            (4.5, 4.5),
            (4.4999, 4.5001),
            (3.0, 7.0),
            (8.5, 0.25),
            (-0.5, 8.5),
        ] {
            assert_eq!(
                walk(&bels, 9, tx, ty),
                expected_order(&bels, 9, tx, ty),
                "target ({tx}, {ty})"
            );
        }
    }

    /// Sparse grids are where multi-ring expansion actually happens: a dense
    /// grid binds in ring 0 and never exercises the guard.
    #[test]
    fn sparse_grids_match_the_reference_order() {
        // Four islands, far apart, plus one lone BEL in the far corner.
        let mut bels: Vec<BelId> = [(1, 1), (1, 18), (18, 1), (18, 18), (9, 10)]
            .iter()
            .flat_map(|&(x, y)| (0..3).map(move |z| BelId::new(y * 20 + x, z)))
            .collect();
        bels.push(BelId::new(19 * 20 + 19, 0));
        bels.sort_by_key(|b| (b.tile(), b.index()));
        for &(tx, ty) in &[
            (0.0, 0.0),
            (9.5, 9.5),
            (10.5, 10.5),
            (19.0, 0.0),
            (5.5, 12.25),
        ] {
            assert_eq!(
                walk(&bels, 20, tx, ty),
                expected_order(&bels, 20, tx, ty),
                "target ({tx}, {ty})"
            );
        }
    }

    /// Targets off the edge of the device: the first rings find nothing and
    /// the walk must keep expanding rather than terminate empty.
    #[test]
    fn targets_outside_the_grid_still_emit_everything() {
        let bels = grid_bels(6, 2);
        for &(tx, ty) in &[(-2.5, 3.0), (7.5, 7.5), (3.0, -1.5), (-1.0, -1.0)] {
            let got = walk(&bels, 6, tx, ty);
            assert_eq!(got.len(), bels.len(), "target ({tx}, {ty}) lost candidates");
            assert_eq!(got, expected_order(&bels, 6, tx, ty), "target ({tx}, {ty})");
        }
    }

    /// A single BEL, and an empty type: the two degenerate walks.
    #[test]
    fn degenerate_walks_terminate() {
        let one = bels_at(6, &[(3, 3)]);
        assert_eq!(walk(&one, 6, 0.0, 0.0), one);
        assert!(walk(&[], 6, 0.0, 0.0).is_empty());
    }

    /// A taken BEL is never emitted, and the rest keep their order.
    #[test]
    fn taken_bels_are_skipped_without_disturbing_the_order() {
        let bels = grid_bels(8, 2);
        let grid = BucketGrid::new(&bels, 8, 8);
        let all: Vec<BelId> = {
            let free = vec![false; bels.len()];
            RingCandidates::new(grid, &free, 3.0, 3.0)
                .map(|(id, _)| id)
                .collect()
        };
        let mut taken = vec![false; bels.len()];
        for i in (0..bels.len()).step_by(3) {
            taken[i] = true;
        }
        let kept: Vec<BelId> = RingCandidates::new(grid, &taken, 3.0, 3.0)
            .map(|(id, _)| id)
            .collect();
        let expected: Vec<BelId> = all
            .into_iter()
            .filter(|id| {
                let e = bels.iter().position(|b| b == id).expect("known bel");
                !taken[e]
            })
            .collect();
        assert_eq!(kept, expected);
    }

    /// Laziness is the point: binding to the nearest BEL must not touch the
    /// rest of the type. Taking one candidate may expand at most a ring or
    /// two, never the whole grid.
    #[test]
    fn taking_one_candidate_does_not_walk_the_grid() {
        let bels = grid_bels(64, 4); // 16,384 candidates
        let free = vec![false; bels.len()];
        let mut it = RingCandidates::new(BucketGrid::new(&bels, 64, 64), &free, 32.0, 32.0);
        let (first, _) = it.next().expect("a candidate");
        assert_eq!(xy(first, 64), (32, 32));
        assert!(
            it.buffered() < 100,
            "walk buffered {} candidates to emit one",
            it.buffered()
        );
    }
}

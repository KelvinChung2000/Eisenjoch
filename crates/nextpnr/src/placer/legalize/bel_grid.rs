//! Spatial index over one cell type's BELs, and a nearest-first walk over it.

use crate::chipdb::BelId;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Tile `(x, y)` -> the BELs of one cell type in that tile, as CSR.
///
/// Stores only each BEL's *enumeration index* -- its position in the type's
/// `bel_data_cache` entry -- because that is both the key the walk's tie-break
/// needs and the way the `BelId` is recovered. Storing the `BelId` too would
/// double the index for nothing.
pub(crate) struct BelGrid {
    width: i32,
    height: i32,
    /// Prefix sums into `entries`; `len() == width * height + 1`.
    offsets: Vec<u32>,
    /// Enumeration indices, grouped by tile, ascending within each tile.
    entries: Vec<u32>,
}

impl BelGrid {
    /// Index one cell type's `(BelId, x, y, z)` cache entries by tile.
    ///
    /// Panics if any BEL sits outside `width x height`: such a BEL would be
    /// unreachable by the ring walk yet visible to `candidate_list`, which is
    /// exactly the silent divergence this whole design exists to rule out.
    pub(crate) fn build(bels: &[(BelId, i32, i32, i32)], width: i32, height: i32) -> Self {
        assert!(width > 0 && height > 0, "degenerate grid {width}x{height}");
        assert!(
            bels.len() <= u32::MAX as usize,
            "{} BELs overflows the u32 enumeration index",
            bels.len()
        );
        let n_cells = (width as usize) * (height as usize);
        let cell_of = |x: i32, y: i32| -> usize {
            assert!(
                x >= 0 && x < width && y >= 0 && y < height,
                "BEL at ({x}, {y}) is outside the {width}x{height} grid"
            );
            (y as usize) * (width as usize) + (x as usize)
        };

        let mut offsets = vec![0u32; n_cells + 1];
        for &(_, x, y, _) in bels {
            offsets[cell_of(x, y) + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        // Counting-sort fill. Walking `bels` in enumeration order is what
        // leaves each tile's entries ascending.
        let mut cursor = offsets.clone();
        let mut entries = vec![0u32; bels.len()];
        for (enum_index, &(_, x, y, _)) in bels.iter().enumerate() {
            let c = cell_of(x, y);
            entries[cursor[c] as usize] = enum_index as u32;
            cursor[c] += 1;
        }

        Self {
            width,
            height,
            offsets,
            entries,
        }
    }

    /// Enumeration indices in tile `(x, y)`, ascending. Empty off-grid: the
    /// walk probes ring cells that fall past the device edge.
    #[inline]
    pub(crate) fn at(&self, x: i32, y: i32) -> &[u32] {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return &[];
        }
        let c = (y as usize) * (self.width as usize) + (x as usize);
        &self.entries[self.offsets[c] as usize..self.offsets[c + 1] as usize]
    }

    #[inline]
    pub(crate) fn width(&self) -> i32 {
        self.width
    }

    #[inline]
    pub(crate) fn height(&self) -> i32 {
        self.height
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Nearest-BEL walk for one cell, in `(dist_sq, enum_index)` order.
///
/// Emits exactly the sequence `candidate_list` produces for the same inputs,
/// but lazily: a cell that binds to its second candidate never looks at the
/// rest of its type. That is the whole point -- the materialized list was
/// `O(cells * bels)`, this is `O(bels)` with one cell's worth live at a time.
pub(crate) struct RingCandidates<'a> {
    grid: &'a BelGrid,
    /// The type's `bel_data_cache` entry; `enum_index` indexes it.
    bels: &'a [(BelId, i32, i32, i32)],
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
        grid: &'a BelGrid,
        bels: &'a [(BelId, i32, i32, i32)],
        target_x: f64,
        target_y: f64,
    ) -> Self {
        let mut walk = Self {
            grid,
            bels,
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
        // Rebind the grid first: `at` borrows it for as long as the slice
        // lives, and the loop body needs `&mut self` for the heap.
        let grid = self.grid;
        let entries = grid.at(x, y);
        if entries.is_empty() {
            return;
        }
        let bits = self.dist_sq(x, y).to_bits();
        for &e in entries {
            self.heap.push(Reverse((bits, e)));
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
    type Item = BelId;

    fn next(&mut self) -> Option<BelId> {
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
                    return Some(self.bels[e as usize].0);
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

    /// `per_tile` BELs at every (x, y) of a `dim x dim` grid, enumerated in
    /// the order `bels_for_bucket` yields them: tile-major, then slot.
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

    /// Every BEL must land in its own tile, and nowhere else.
    #[test]
    fn build_places_every_bel_in_its_own_tile() {
        let bels = grid_bels(6, 3);
        let grid = BelGrid::build(&bels, 6, 6);
        let mut seen = 0;
        for x in 0..6 {
            for y in 0..6 {
                let entries = grid.at(x, y);
                assert_eq!(entries.len(), 3, "tile ({x},{y})");
                for &e in entries {
                    let (_, bx, by, _) = bels[e as usize];
                    assert_eq!((bx, by), (x, y), "entry {e} filed under the wrong tile");
                }
                seen += entries.len();
            }
        }
        assert_eq!(seen, bels.len(), "grid dropped or duplicated entries");
    }

    /// A tile's entries must stay in enumeration order.
    ///
    /// Every BEL in a tile shares one `dist_sq`, so this ordering *is* the
    /// tie-break the walk relies on to match `candidate_list`. A counting-sort
    /// fill that iterated the source out of order would silently break it.
    #[test]
    fn tile_entries_are_in_enumeration_order() {
        let bels = grid_bels(5, 4);
        let grid = BelGrid::build(&bels, 5, 5);
        for x in 0..5 {
            for y in 0..5 {
                let entries = grid.at(x, y);
                assert!(
                    entries.windows(2).all(|w| w[0] < w[1]),
                    "tile ({x},{y}) entries out of order: {entries:?}"
                );
            }
        }
    }

    /// Sparse layouts are the interesting case: most tiles hold nothing.
    #[test]
    fn empty_tiles_return_an_empty_slice() {
        let bels = vec![(BelId::new(0, 0), 2, 3, 0), (BelId::new(1, 0), 2, 3, 1)];
        let grid = BelGrid::build(&bels, 8, 8);
        assert_eq!(grid.at(2, 3).len(), 2);
        assert!(grid.at(0, 0).is_empty());
        assert!(grid.at(7, 7).is_empty());
    }

    /// The walk probes ring cells that fall off the edge of the device;
    /// `at` must answer those without a bounds panic.
    #[test]
    fn out_of_bounds_lookups_are_empty_not_a_panic() {
        let grid = BelGrid::build(&grid_bels(4, 2), 4, 4);
        for (x, y) in [(-1, 0), (0, -1), (4, 0), (0, 4), (-7, 9)] {
            assert!(grid.at(x, y).is_empty(), "({x},{y})");
        }
    }

    #[test]
    fn dimensions_are_reported() {
        let grid = BelGrid::build(&grid_bels(4, 2), 4, 9);
        assert_eq!((grid.width(), grid.height()), (4, 9));
        assert!(!grid.is_empty());
        assert!(BelGrid::build(&[], 4, 9).is_empty());
    }

    /// Fail loudly. A BEL outside the declared grid would be unreachable by
    /// the walk but present in `candidate_list`, i.e. a silent divergence.
    #[test]
    #[should_panic(expected = "outside the")]
    fn a_bel_outside_the_grid_panics() {
        BelGrid::build(&[(BelId::new(0, 0), 9, 0, 0)], 4, 4);
    }

    /// Reference ordering: exactly what `candidate_list` computes, done the
    /// slow obvious way. Kept here so the geometry tests are self-contained;
    /// `sorted.rs` checks the walk against the real `candidate_list` too.
    fn expected_order(bels: &[(BelId, i32, i32, i32)], tx: f64, ty: f64) -> Vec<BelId> {
        let mut v: Vec<(f64, usize, BelId)> = bels
            .iter()
            .enumerate()
            .map(|(i, &(id, bx, by, _))| {
                let dx = bx as f64 - tx;
                let dy = by as f64 - ty;
                (dx * dx + dy * dy, i, id)
            })
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        v.into_iter().map(|(_, _, id)| id).collect()
    }

    fn walk(bels: &[(BelId, i32, i32, i32)], dim: i32, tx: f64, ty: f64) -> Vec<BelId> {
        let grid = BelGrid::build(bels, dim, dim);
        RingCandidates::new(&grid, bels, tx, ty).collect()
    }

    /// Square rings under a Euclidean metric: a BEL in ring `r` is not
    /// necessarily nearer than one in ring `r+1`. `(3,0)` sits in ring 3 at
    /// distance 3; `(2,2)` sits in ring 2 at distance ~2.83 -- but `(2,2)`'s
    /// ring is explored first, and `(1,3)` (ring 3, distance ~3.16) must NOT
    /// overtake `(3,0)`. Ordering has to come from the heap, not the rings.
    #[test]
    fn ring_order_is_not_distance_order() {
        let bels = vec![
            (BelId::new(0, 0), 3, 0, 0), // ring 3, d^2 = 9
            (BelId::new(1, 0), 2, 2, 0), // ring 2, d^2 = 8
            (BelId::new(2, 0), 1, 3, 0), // ring 3, d^2 = 10
            (BelId::new(3, 0), 0, 1, 0), // ring 1, d^2 = 1
        ];
        assert_eq!(walk(&bels, 8, 0.0, 0.0), expected_order(&bels, 0.0, 0.0));
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
        let bels = vec![
            (BelId::new(0, 0), 0, 0, 0), // index 0, ring 2, d^2 = 2.25
            (BelId::new(1, 0), 3, 0, 0), // index 1, ring 1, d^2 = 2.25
        ];
        let got = walk(&bels, 5, 1.5, 0.0);
        assert_eq!(got, expected_order(&bels, 1.5, 0.0));
        assert_eq!(
            got,
            vec![BelId::new(0, 0), BelId::new(1, 0)],
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
                expected_order(&bels, tx, ty),
                "target ({tx}, {ty})"
            );
        }
    }

    /// Sparse grids are where multi-ring expansion actually happens: a dense
    /// grid binds in ring 0 and never exercises the guard.
    #[test]
    fn sparse_grids_match_the_reference_order() {
        // Four islands, far apart, plus one lone BEL in the far corner.
        let mut bels = Vec::new();
        let mut tile = 0;
        for &(x, y) in &[(1, 1), (1, 18), (18, 1), (18, 18), (9, 10)] {
            for z in 0..3 {
                bels.push((BelId::new(tile, z), x, y, z));
            }
            tile += 1;
        }
        bels.push((BelId::new(tile, 0), 19, 19, 0));
        for &(tx, ty) in &[
            (0.0, 0.0),
            (9.5, 9.5),
            (10.5, 10.5),
            (19.0, 0.0),
            (5.5, 12.25),
        ] {
            assert_eq!(
                walk(&bels, 20, tx, ty),
                expected_order(&bels, tx, ty),
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
            assert_eq!(got, expected_order(&bels, tx, ty), "target ({tx}, {ty})");
        }
    }

    /// A single BEL, and an empty type: the two degenerate walks.
    #[test]
    fn degenerate_walks_terminate() {
        let one = vec![(BelId::new(0, 0), 3, 3, 0)];
        assert_eq!(walk(&one, 6, 0.0, 0.0), vec![BelId::new(0, 0)]);
        assert!(walk(&[], 6, 0.0, 0.0).is_empty());
    }

    /// Laziness is the point: binding to the nearest BEL must not touch the
    /// rest of the type. Taking one candidate may expand at most a ring or
    /// two, never the whole grid.
    #[test]
    fn taking_one_candidate_does_not_walk_the_grid() {
        let bels = grid_bels(64, 4); // 16,384 candidates
        let grid = BelGrid::build(&bels, 64, 64);
        let mut it = RingCandidates::new(&grid, &bels, 32.0, 32.0);
        let first = it.next().expect("a candidate");
        let (_, bx, by, _) = bels.iter().find(|b| b.0 == first).copied().unwrap();
        assert_eq!((bx, by), (32, 32));
        assert!(
            it.buffered() < 100,
            "walk buffered {} candidates to emit one",
            it.buffered()
        );
    }
}

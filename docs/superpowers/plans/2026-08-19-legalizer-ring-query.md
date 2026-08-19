# Legalizer Ring-Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legalizer's per-cell materialized BEL shortlist (and its 27,467 full re-sorts) with a lazy nearest-first walk over a per-type spatial grid, without changing which BEL any cell binds to.

**Architecture:** A new `legalize/bel_grid.rs` holds two pure-data types: `BelGrid`, a CSR `(x, y) -> [enum_index]` index built from the `bel_data_cache` the legalizer already assembles, and `RingCandidates`, an `Iterator<Item = BelId>` that expands Chebyshev rings around the target and emits candidates in ascending `(dist_sq, enum_index)` — exactly `candidate_list`'s order. `sorted.rs` then drops Phase A, `CAND_SHORTLIST`, and the widen block, and hands `try_bind_cell` an iterator instead of a slice. `candidate_list` survives, unlimited, for region-constrained cells and as the property-test oracle.

**Tech Stack:** Rust 2021, `std::collections::BinaryHeap`, `rustc_hash::FxHashMap`, `cargo test`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-19-legalizer-ring-query-design.md`

## Global Constraints

- **Branch/worktree:** `npnr-faithful-port` at `/home/kelvin/side-project/eisenjoch/.claude/worktrees/npnr-faithful-port`. All paths below are relative to it.
- **`cargo` needs the sandbox disabled** — `target/` is outside the writable set, and a sandboxed `cargo` fails with `Read-only file system (os error 30)` on `.cargo-lock`.
- **Baseline to preserve:** `cargo test -p nextpnr --lib` = 132 passed, 1 ignored. Two tests are deliberately deleted in Task 4, so the post-Task-4 floor is 130 plus whatever this plan adds.
- **`try_bind_cell`'s body does not change.** Only its `candidates` parameter type changes. Its shared-mux / cluster-footprint / arch-validity logic is what made `fc82eb8` un-cherry-pickable; this plan does not reopen it.
- **`dist_sq` is always the same expression**, everywhere, with no algebraic rearrangement:
  ```rust
  let dx = x as f64 - target_x;
  let dy = y as f64 - target_y;
  dx * dx + dy * dy
  ```
  Equivalence with `candidate_list` is bitwise, not approximate. Rearranging this expression breaks the plan's central guarantee.
- **Ordering contract:** candidates are emitted in ascending `(dist_sq, enum_index)`, where `enum_index` is the BEL's position in `bel_data_cache[type]`. Heap key is `(dist_sq.to_bits(), enum_index)`; `BelId` is **not** in the key (it derives no `Ord`, `chipdb/ids.rs:30`) and is recovered by indexing.
- **FPGA01 runs are wrapped**, always: `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect`. Unwrapped it takes the box down.
- Rustdoc comments explain *why*, not *what*, matching the density already in `sorted.rs`.

---

## Deviation from the spec (already applied to it)

The spec's emit guard is `dist_sq < (R + 0.5)^2`. That bound is derived from *true* distances, but the values being compared are *computed* f64s, so a candidate could clear it by an ulp and be emitted ahead of an equal-distance BEL with a lower `enum_index`. This plan uses an exact bound instead: the minimum computed `dist_sq` over ring `R+1`'s four axis points, evaluated with the identical expression. Same idea, no epsilon reasoning. Task 0 applied this to the spec, so the authoritative document does not keep the superseded formula.

---

### Task 0: Amend the spec's emit-guard section — DONE

Executed while writing this plan, so the authoritative spec never carried the
superseded formula. Recorded here for the audit trail; there is nothing to do.

**Files:** Modified `docs/superpowers/specs/2026-08-19-legalizer-ring-query-design.md`.

- [x] "Ring geometry and the emit guard" rewritten: the bound is now the
  minimum `dist_sq` over ring `R+1`'s four axis points, evaluated with the
  candidates' own expression, with the componentwise-domination argument that
  makes it exact in f64 and the monotonicity that makes equality-expand
  terminate.
- [x] The iterator sketch's `TypeGrid`/`(BelId, u32)` line replaced with
  `BelGrid` CSR, and a `guard: f64` field added.
- [x] Approach A's payload corrected to `[x][y] -> [enum_index]`, with a
  paragraph on why the `BelId` is not stored twice.
- [x] Test 2 rewritten: its old failure conditions named the superseded bound.
  It now also requires sparse layouts, and names the two mutations that must
  break it (see Task 2 Step 5).
- [x] Test 1's snippet updated for `candidate_list`'s dropped `limit` argument.
- [x] Risk-table row updated.
- [x] Verified: `grep -n "R + 0.5\|TypeGrid\|(BelId, u32)"` leaves only prose
  that deliberately discusses the rejected bound and the pre-change code.

---

### Task 1: `BelGrid` — the CSR spatial index

**Files:**
- Create: `crates/nextpnr/src/placer/legalize/bel_grid.rs`
- Modify: `crates/nextpnr/src/placer/legalize/mod.rs` (add `mod bel_grid;` beside the existing `pub mod` lines)
- Test: same file, `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `crate::chipdb::BelId`.
- Produces:
  - `pub(crate) struct BelGrid`
  - `pub(crate) fn BelGrid::build(bels: &[(BelId, i32, i32, i32)], width: i32, height: i32) -> BelGrid`
  - `pub(crate) fn BelGrid::at(&self, x: i32, y: i32) -> &[u32]` — enum indices in that tile, ascending; empty slice when out of bounds
  - `pub(crate) fn BelGrid::width(&self) -> i32`, `pub(crate) fn BelGrid::height(&self) -> i32`
  - `pub(crate) fn BelGrid::is_empty(&self) -> bool`

**Note on the module name:** `legalize/ring.rs` already exists and is a *different* thing (`RingLegalizer`, a Manhattan-ring `Legalizer` impl). Do not add to it, and do not name the new module `ring.rs`.

- [ ] **Step 1: Write the failing tests**

Create `crates/nextpnr/src/placer/legalize/bel_grid.rs` with only this test module plus the imports it needs:

```rust
//! Spatial index over one cell type's BELs, and a nearest-first walk over it.

use crate::chipdb::BelId;

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
}
```

- [ ] **Step 2: Register the module and run the tests to verify they fail**

In `crates/nextpnr/src/placer/legalize/mod.rs`, add after `pub mod bipartite;`:

```rust
mod bel_grid;
```

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: FAIL to compile, `cannot find type BelGrid in this scope`.

- [ ] **Step 3: Write the implementation**

Insert above the `#[cfg(test)]` module in `bel_grid.rs`:

```rust
/// Tile `(x, y)` -> the BELs of one cell type in that tile, as CSR.
///
/// Stores only each BEL's *enumeration index* — its position in the type's
/// `bel_data_cache` entry — because that is both the key the walk's tie-break
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

        Self { width, height, offsets, entries }
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/legalize/bel_grid.rs crates/nextpnr/src/placer/legalize/mod.rs
git commit -m "feat(legalize): add BelGrid, a CSR tile index over one type's BELs

Stores enumeration indices rather than BelIds: the index is the walk's
tie-break key and the way the BelId is recovered, so keeping both would
double the structure for nothing. The counting-sort fill walks the source
in enumeration order, which is what leaves each tile's entries ascending
-- every BEL in a tile shares one dist_sq, so that order IS the tie-break."
```

---

### Task 2: `RingCandidates` — the lazy nearest-first walk

**Files:**
- Modify: `crates/nextpnr/src/placer/legalize/bel_grid.rs` (add the iterator and its tests)

**Interfaces:**
- Consumes: `BelGrid::{at, width, height}` from Task 1.
- Produces:
  - `pub(crate) struct RingCandidates<'a>`
  - `pub(crate) fn RingCandidates::new(grid: &'a BelGrid, bels: &'a [(BelId, i32, i32, i32)], target_x: f64, target_y: f64) -> RingCandidates<'a>`
  - `impl Iterator for RingCandidates<'_> { type Item = BelId; }`

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` in `bel_grid.rs`:

```rust
    /// Reference ordering: exactly what `candidate_list` computes, done the
    /// slow obvious way. Kept here so the geometry tests are self-contained;
    /// Task 3 checks the walk against the real `candidate_list` too.
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
        for &(tx, ty) in &[(0.0, 0.0), (9.5, 9.5), (10.5, 10.5), (19.0, 0.0), (5.5, 12.25)] {
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: FAIL to compile, `cannot find type RingCandidates in this scope`.

- [ ] **Step 3: Write the implementation**

Change the import line at the top of `bel_grid.rs` to:

```rust
use crate::chipdb::BelId;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
```

and add below the `impl BelGrid` block:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: PASS, 13 tests.

- [ ] **Step 5: Prove the guard is load-bearing**

A guard is only meaningful if a weakened one is caught. Apply each mutation, run, revert, confirm green again.

**Mutation A — guard too permissive.** In `expand`, change `self.guard = self.ring_min_dist_sq(r + 1);` to `self.ring_min_dist_sq(r + 2);`.

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: FAIL, at least `fractional_targets_match_the_reference_order`. (Concretely, on a dense grid from `(4.5, 4.5)`, ring 2's `(3,5)` is nearer at `d^2 = 2.5` than ring 1's `(6,6)` at `4.5`; the loosened bound emits `(6,6)` first.)

**Mutation B — equality stops expanding.** Change the emit condition from `<` to `<=`.

Run: `cargo test -p nextpnr --lib bel_grid`
Expected: FAIL, `an_exact_tie_is_won_by_the_lower_index_in_the_outer_ring`.

**Do not** try `ring_min_dist_sq(r)` as a mutation: `guard` is monotone nondecreasing in `r`, so that bound is *stricter* than the real one. It is safe, merely slower, and the suite stays green — which is correct, not a gap.

If either real mutation leaves the suite green, the tests do not cover the guard — stop and add a case that does before continuing.

- [ ] **Step 6: Commit**

```bash
git add crates/nextpnr/src/placer/legalize/bel_grid.rs
git commit -m "feat(legalize): add RingCandidates, a lazy nearest-BEL walk

Rings are Chebyshev but the metric is Euclidean, so ring order is not
distance order and candidates have to be buffered. They are released only
once nothing unexplored can beat them, where the bound is the minimum
dist_sq over the next ring's four axis points -- evaluated with the same
dx*dx+dy*dy as the candidates, so it is exact in f64 rather than an
epsilon argument. Equality expands one more ring: a whole tile shares one
dist_sq, so ties are the common case and the lower enum_index wins."
```

---

### Task 3: Certify the walk against `candidate_list`

**Files:**
- Modify: `crates/nextpnr/src/placer/legalize/sorted.rs` (add to `mod candidate_list_tests`)

**Interfaces:**
- Consumes: `RingCandidates::new`, `BelGrid::build` (Task 1, 2); `candidate_list`, `CellLegalizeInfo` (existing, unchanged).
- Produces: the equivalence guarantee Task 4 relies on. Nothing else consumes it.

**Why this is its own task:** it is the load-bearing gate. Nothing on this branch is bit-reproducible (`e3efbf3` is not merged; sv3 spans ~0.6% run-to-run at fixed seed), so the FPGA01 run in Task 5 *cannot* certify that the legalizer still picks the same BELs. This test is what does — and it must pass before the old path is deleted, not after.

- [ ] **Step 1: Write the failing test**

Add to `mod candidate_list_tests` in `sorted.rs`:

```rust
    use crate::placer::legalize::bel_grid::{BelGrid, RingCandidates};

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
```

Note: this test calls `candidate_list` with **three** arguments. The `limit` parameter is removed in Task 4 Step 1, so write that step's change first if you prefer a compiling intermediate; otherwise expect this step's failure to include an arity error, which is the intended signal.

- [ ] **Step 2: Make `bel_grid` reachable and drop the `limit` parameter**

In `crates/nextpnr/src/placer/legalize/mod.rs`, change `mod bel_grid;` to:

```rust
pub(crate) mod bel_grid;
```

In `sorted.rs`, change `candidate_list`'s signature from four parameters to three by deleting `limit: Option<usize>`, and replace its whole `match limit { .. }` block with:

```rust
    candidates.sort_unstable_by(cmp);
```

Update its doc comment to:

```rust
/// Every BEL of `info`'s type, sorted by `(dist_sq, enum_index)`.
///
/// Distance ties are broken on enumeration order because they are the common
/// case -- a tile holds many slots at identical x/y -- and the order among
/// them decides which BEL Phase B binds. `RingCandidates` reproduces this
/// exact sequence lazily; this function is the region-constrained path and
/// the oracle that test proves it against.
```

Then fix the two existing call sites in `sorted_legalize` (Phase A's `Some(CAND_SHORTLIST)` and the widen block's `None`) by deleting that argument — both call sites are removed outright in Task 4, so a mechanical fix here is enough to compile.

Also delete the now-uncompilable test `shortlist_is_a_prefix_of_the_full_list` and `shortlist_allocation_is_bounded_by_the_cap` from `mod candidate_list_tests`. They cover the capping mechanism, which no longer exists.

- [ ] **Step 3: Run the test to verify it fails for the right reason**

Run: `cargo test -p nextpnr --lib the_walk_emits_candidate_lists_exact_sequence`

If it PASSES immediately, that is the expected outcome — Tasks 1 and 2 already implement the behaviour, and this task's job is to *prove* it. Confirm the proof is real by breaking it: in `bel_grid.rs`, change the heap key push from `Reverse((bits, e))` to `Reverse((bits, u32::MAX - e))` and re-run.
Expected: FAIL with "order diverged".
Then revert and confirm PASS.

- [ ] **Step 4: Run the whole suite**

Run: `cargo test -p nextpnr --lib`
Expected: PASS. Count is 132 − 2 deleted + 13 (Task 1/2) + 1 (this task) = 144 passed, 1 ignored.

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/legalize/sorted.rs crates/nextpnr/src/placer/legalize/mod.rs
git commit -m "test(legalize): prove RingCandidates emits candidate_list's exact sequence

Full-sequence equality over dense and sparse layouts, fractional and
off-grid targets. Full-sequence rather than first-pick because Phase B
walks deep -- FPGA01 averages ~285 rejected candidates per cell.

This is what certifies the swap. Nothing on this branch is bit-reproducible
(e3efbf3 is not merged), so a live run cannot tell a legalizer regression
from ordinary run-to-run spread; the oracle can.

candidate_list loses its limit parameter with the last capped caller."
```

---

### Task 4: Swap Phase B onto the walk and delete the shortlist

**Files:**
- Modify: `crates/nextpnr/src/placer/legalize/sorted.rs` (`CAND_SHORTLIST`, `try_bind_cell` signature, `sorted_legalize` Phase A and Phase B)

**Interfaces:**
- Consumes: `BelGrid::build`, `RingCandidates::new` (Tasks 1–2); the oracle guarantee (Task 3).
- Produces: `try_bind_cell(..., candidates: impl Iterator<Item = BelId>, ...)`. No other signature changes; `sorted_legalize`'s own signature is untouched, so `heap/legalize.rs:19` and `example_arch_tests.rs:1259` keep compiling unmodified.

- [ ] **Step 1: Delete `CAND_SHORTLIST`**

Remove the `const CAND_SHORTLIST: usize = 1024;` declaration and the whole doc comment above it.

- [ ] **Step 2: Change `try_bind_cell` to take an iterator**

Change the parameter:

```rust
    candidates: &[BelId],
```

to:

```rust
    candidates: impl Iterator<Item = BelId>,
```

and the loop header:

```rust
    'outer: for &bel in candidates {
```

to:

```rust
    'outer: for bel in candidates {
```

Nothing else inside the function changes. Then replace its doc comment's last paragraph — the sentence beginning "Split out of the Phase B loop" — with:

```rust
/// Takes an iterator rather than a slice so the candidate sequence can be
/// produced lazily: the common case rejects a few hundred BELs and binds,
/// having materialized nothing.
```

- [ ] **Step 3: Replace Phase A with the grid build**

Delete the entire `let sorted_candidates: Vec<Vec<BelId>> = cell_infos.par_iter()...collect();` statement and its comment block. In its place put:

```rust
    // One spatial index per cell type, replacing the per-cell shortlists.
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
    let grid_w = ctx.chipdb().width();
    let grid_h = ctx.chipdb().height();
    let bel_grids: FxHashMap<IdString, BelGrid> = bel_data_cache
        .iter()
        .map(|(&type_id, bels)| (type_id, BelGrid::build(bels, grid_w, grid_h)))
        .collect();
```

`grid_w` and `grid_h` are already bound earlier in `sorted_legalize` (the outer-cells-first sort uses them). Reuse those bindings rather than rebinding: delete the two `let grid_w`/`let grid_h` lines from the snippet above and keep the existing ones. Verify with `grep -n "let grid_w" crates/nextpnr/src/placer/legalize/sorted.rs` — there must be exactly one.

Add to the imports at the top of the file:

```rust
use crate::placer::legalize::bel_grid::{BelGrid, RingCandidates};
```

and delete `use rayon::prelude::*;` — Phase A was its only user. Confirm with `grep -n "par_iter\|par_bridge\|into_par" crates/nextpnr/src/placer/legalize/sorted.rs` returning nothing.

- [ ] **Step 4: Replace the Phase B body**

Delete `let mut widened_rescans: u64 = 0;`. Then replace everything from `let shortlist = &sorted_candidates[i];` down to the end of the widen `if disp.is_none() { ... }` block with:

```rust
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
```

The `match disp { Some(d) => ..., None => Err(..) }` block below it is unchanged.

Because `disp` is now immutable and bound by `let`, delete the `mut` from what was `let mut disp`. The loop no longer uses its index, so change

```rust
    for (i, info) in cell_infos.iter().enumerate() {
```

to

```rust
    for info in &cell_infos {
```

- [ ] **Step 5: Drop `widened_rescans` from the summary line**

Change the `eprintln!` to:

```rust
    eprintln!(
        "  SortedLegalizer: cluster_rejects={} shared_mux_rejects={} arch_validity_rejects={}",
        cluster_footprint_rejects, shared_mux_rejects, arch_validity_rejects,
    );
```

- [ ] **Step 6: Run the whole suite**

Run: `cargo test -p nextpnr 2>&1 | tail -20`
Expected: PASS. Lib: 144 passed, 1 ignored. Integration tests including `sorted_legalize_honours_the_arch_validity_rule` also pass.

- [ ] **Step 7: Confirm the old mechanism is gone**

Run: `grep -n "CAND_SHORTLIST\|sorted_candidates\|widened_rescans\|shortlist" crates/nextpnr/src/placer/legalize/sorted.rs`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add crates/nextpnr/src/placer/legalize/sorted.rs
git commit -m "perf(legalize): walk BELs nearest-first instead of materializing shortlists

Phase A built a distance-sorted candidate list per cell, capped at 1024.
The cap bounded work but not legality, so exhausting it fell back to a
full scan and sort of every BEL of the type: 27467 times on FPGA01, at up
to 1.08M entries each. Indexing per type and walking outward removes both
the 599 MiB of shortlists and the re-sorts.

try_bind_cell now takes an iterator; its body is untouched. Region cells
keep the materialized path -- ring-walking a cell whose region sits far
from its target would scan the device. candidate_list stays as that path
and as the oracle proving the walk emits its exact sequence."
```

---

### Task 5: Measure on FPGA01

**Files:**
- Create: `measurements/mem_probe/fpga01_ring_query.log`

**Interfaces:**
- Consumes: the legalizer from Task 4.
- Produces: measured peak RSS and legalization wall time, to compare against 4217 MB / 1167 s.

**Why the run cannot certify correctness:** the branch is not bit-reproducible, so post-legalization HPWL will not match 14234563 and is not expected to. Task 3 is the correctness gate; this is a performance measurement. Judge HPWL only against the ~0.6% run-to-run spread.

- [ ] **Step 1: Locate the FPGA01 invocation used for the baseline**

Run: `head -40 measurements/mem_probe/fpga01_legalize_fix.log`

Reuse whatever command that log records, unchanged except for the output path. Do not invent a new invocation — a changed command makes the comparison meaningless.

- [ ] **Step 2: Build release**

Run: `cargo build --release -p nextpnr`
Expected: success. (Sandbox disabled — see Global Constraints.)

- [ ] **Step 3: Run FPGA01, backgrounded and memory-capped**

Legalization alone took ~20 minutes at baseline, so this must not run in the foreground. Wrap it exactly as below; the cap is standing policy and being killed at it is a usable result.

```bash
nohup systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  <baseline command from Step 1, with NPNR_OT_MAX_ITERS=1> \
  > measurements/mem_probe/fpga01_ring_query.log 2>&1 &
```

- [ ] **Step 4: Extract the comparison**

Once it finishes, pull the three numbers that matter:

```bash
grep -n "SortedLegalizer:\|Pre-legalization\|Post-legalization\|RSS_SAMPLE\|Killed" \
  measurements/mem_probe/fpga01_ring_query.log | tail -40
```

Record, against the baseline in the spec's measurement table:
- peak RSS (baseline 4217 MB; ~600 MiB of shortlist should be gone, ~20 MB of grid added)
- legalization wall time (baseline 1167 s; no figure is predicted)
- `shared_mux_rejects` (baseline 21,841,944 — this is *expected to be unchanged*, and a large move in it means the candidate order changed, which would contradict Task 3)

- [ ] **Step 5: Commit the measurement**

```bash
git add measurements/mem_probe/fpga01_ring_query.log
git commit -m "evidence(legalize): FPGA01 under the ring query -- <peak> MB, <time> s

Against 4217 MB / 1167 s. shared_mux_rejects <changed/unchanged> at <n>
vs 21841944; unchanged is the expected result, since Task 3 fixes the
candidate order. HPWL is not compared -- the branch is not bit-reproducible."
```

---

## Self-Review

**Spec coverage.** Problem → Tasks 1–4. Goals: shortlist and re-sorts removed → Task 4; output identical → Task 3; `try_bind_cell` body untouched → Task 4 Step 2 (signature only). Approach A → Tasks 1–2, built from `bel_data_cache`. Ordering contract → Global Constraints + Tasks 1–2. Ring geometry / emit guard → Task 0 (amended) + Task 2. The iterator → Task 2. Phase B integration → Task 4. Region-constrained cells → Task 4 Step 4. Empty-candidate detection → Task 4 Step 4 (three `NoBelsAvailable` sites). Testing 1 → Task 3; 2 → Task 2 Steps 1 and 5; 3 → Task 3 Step 4 and Task 4 Step 6; 4 → Task 5. Non-goals stay out: no task touches `shared_mux_rejects`, cell ordering, or displacement accumulation.

**Deviations from the spec, both applied to the spec in Task 0.** (1) The emit guard is the minimum computed `dist_sq` over ring `R+1`'s axis points, not `(R+0.5)²` — exact in f64 rather than epsilon-dependent. (2) The grid stores bare `u32` enum indices, not `(BelId, u32)`; the spec's own text already says the `BelId` is recovered by indexing, so the pair was redundant. Neither changes design intent.

**Placeholder scan.** No TBD/TODO. Every code step carries the literal code. The one deliberate blank is Task 5's baseline command, which Step 1 reads out of the existing log rather than guessing — inventing it would silently break the comparison.

**Type consistency.** `BelGrid::build(&[(BelId, i32, i32, i32)], i32, i32)`, `at(i32, i32) -> &[u32]`, `width()/height() -> i32`, `is_empty() -> bool` are used with those exact types in Tasks 2–4. `RingCandidates::new(&BelGrid, &[(BelId, i32, i32, i32)], f64, f64)` matches Task 4's call. `candidate_list` is three-argument from Task 3 Step 2 onward, and Task 4 Step 4 calls it with three. `try_bind_cell` takes `impl Iterator<Item = BelId>` from Task 4 Step 2; both call sites pass an iterator (`Vec::into_iter`, `RingCandidates`). `buffered()` is `#[cfg(test)]` and used only in Task 2's tests.

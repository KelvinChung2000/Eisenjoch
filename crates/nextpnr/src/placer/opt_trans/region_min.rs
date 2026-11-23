//! Per-net region-minimum pyramid over a net's dist_cache row.
//!
//! On a `width × height` grid (node index `y*width + x`), level `k` cell
//! `(cx, cy)` holds the minimum label over the rectangle
//! `[2^k·cx, 2^k·(cx+1)) × [2^k·cy, 2^k·(cy+1))`, clamped to the grid. Level 0
//! is the row itself and is not stored here — callers read `dist_cache`.
//!
//! The pyramid is used by the branch-and-bound search in `bb_2d` to prune
//! rectangles whose lower bound already exceeds the current best cost.
//!
//! Levels are sparse like the row: a cell no row node falls in is absent and
//! reads as `INFINITY`, which is exactly what a dense level would have held
//! there. Each level therefore stores at most one entry per row entry, so a
//! pyramid costs `O(row_len · levels)` however large the chip is. Dense levels
//! cost `n_nodes / 3` floats per net, ~9 GB per sweep at 100k nets.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub(crate) struct PyrLevel {
    pub w: i32,
    pub h: i32,
    /// `cy * w + cx` -> minimum label inside that cell.
    data: FxHashMap<u32, f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct RegionMinPyramid {
    /// `levels[k-1]` is pyramid level `k` (level 0 is the dist grid itself).
    pub levels: Vec<PyrLevel>,
}

impl RegionMinPyramid {
    /// Highest level index (inclusive). At `max_level()` there is a single
    /// cell covering the whole grid.
    #[inline]
    pub fn max_level(&self) -> i32 {
        self.levels.len() as i32
    }

    /// Minimum over the rectangle covered by pyramid cell `(level, cx, cy)`.
    /// Precondition: `level >= 1`.
    #[inline]
    pub fn at(&self, level: i32, cx: i32, cy: i32) -> f32 {
        debug_assert!(level >= 1);
        let lvl = &self.levels[(level - 1) as usize];
        debug_assert!(cx >= 0 && cx < lvl.w && cy >= 0 && cy < lvl.h);
        lvl.data
            .get(&((cy * lvl.w + cx) as u32))
            .copied()
            .unwrap_or(f32::INFINITY)
    }

    /// Build a pyramid from a sparse row of `node -> label` on a
    /// `width × height` grid; absent nodes are `INFINITY`.
    pub fn build(row: &FxHashMap<u32, f32>, width: i32, height: i32) -> Self {
        let mut levels: Vec<PyrLevel> = Vec::new();
        let (mut cur_w, mut cur_h) = (width, height);
        // Level 1 always exists, even on a 1×1 grid, so `max_level() >= 1`
        // whenever the grid is non-empty.
        loop {
            let w = (cur_w + 1) / 2;
            let h = (cur_h + 1) / 2;
            let (src, src_w) = match levels.last() {
                Some(prev) => (&prev.data, prev.w),
                None => (row, width),
            };
            let mut data = FxHashMap::with_capacity_and_hasher(src.len(), Default::default());
            for (&idx, &v) in src {
                let (x, y) = (idx as i32 % src_w, idx as i32 / src_w);
                let m = data
                    .entry(((y / 2) * w + x / 2) as u32)
                    .or_insert(f32::INFINITY);
                if v < *m {
                    *m = v;
                }
            }
            data.shrink_to_fit();
            levels.push(PyrLevel { w, h, data });
            (cur_w, cur_h) = (w, h);
            if cur_w <= 1 && cur_h <= 1 {
                break;
            }
        }
        Self { levels }
    }

    /// Bytes held on the heap.
    pub fn heap_bytes(&self) -> usize {
        // hashbrown: one (key, value) slot plus one control byte per bucket.
        let slot = std::mem::size_of::<(u32, f32)>() + 1;
        self.levels.capacity() * std::mem::size_of::<PyrLevel>()
            + self
                .levels
                .iter()
                .map(|l| l.data.capacity() * slot)
                .sum::<usize>()
    }
}

/// Build pyramids for every net in parallel from sparse per-net rows.
pub(crate) fn build_all(
    rows: &[FxHashMap<u32, f32>],
    width: i32,
    height: i32,
) -> Vec<RegionMinPyramid> {
    rows.par_iter()
        .map(|row| RegionMinPyramid::build(row, width, height))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense reference: the minimum over each level cell's clamped rectangle.
    fn dense_min(
        row: &FxHashMap<u32, f32>,
        width: i32,
        height: i32,
        level: i32,
        cx: i32,
        cy: i32,
    ) -> f32 {
        let s = 1 << level;
        let mut m = f32::INFINITY;
        for y in (cy * s)..((cy + 1) * s).min(height) {
            for x in (cx * s)..((cx + 1) * s).min(width) {
                if let Some(&v) = row.get(&((y * width + x) as u32)) {
                    if v < m {
                        m = v;
                    }
                }
            }
        }
        m
    }

    fn disc_row(width: i32, cx: i32, cy: i32, r: i32) -> FxHashMap<u32, f32> {
        let mut row = FxHashMap::default();
        for y in (cy - r)..=(cy + r) {
            for x in (cx - r)..=(cx + r) {
                let d = (x - cx).abs() + (y - cy).abs();
                if d <= r && x >= 0 && y >= 0 {
                    row.insert((y * width + x) as u32, d as f32 + 0.5);
                }
            }
        }
        row
    }

    #[test]
    fn sparse_levels_match_dense_region_minimum() {
        let (w, h) = (37, 23);
        let mut row = disc_row(w, 5, 7, 4);
        row.extend(disc_row(w, 30, 18, 3));
        let pyr = RegionMinPyramid::build(&row, w, h);
        assert_eq!(pyr.levels.last().map(|l| (l.w, l.h)), Some((1, 1)));
        for level in 1..=pyr.max_level() {
            let lvl = &pyr.levels[(level - 1) as usize];
            for cy in 0..lvl.h {
                for cx in 0..lvl.w {
                    assert_eq!(
                        pyr.at(level, cx, cy),
                        dense_min(&row, w, h, level, cx, cy),
                        "level {level} cell ({cx},{cy})"
                    );
                }
            }
        }
    }

    #[test]
    fn single_tile_grid_still_has_one_level() {
        let row: FxHashMap<u32, f32> = [(0, 3.0)].into_iter().collect();
        let pyr = RegionMinPyramid::build(&row, 1, 1);
        assert_eq!(pyr.max_level(), 1);
        assert_eq!(pyr.at(1, 0, 0), 3.0);
    }

    /// A row holds only nodes near the net's pins, so its pyramid must cost
    /// the same on a 12x12 grid as on a 311x223 one.
    #[test]
    fn storage_does_not_scale_with_grid_size() {
        let small = RegionMinPyramid::build(&disc_row(12, 6, 6, 5), 12, 12);
        let large = RegionMinPyramid::build(&disc_row(311, 6, 6, 5), 311, 223);
        // The larger grid only adds levels, each holding the disc in one cell.
        let extra = large.levels.len() - small.levels.len();
        assert!(
            large.heap_bytes() <= small.heap_bytes() + extra * 256,
            "{} B on 311x223 vs {} B on 12x12",
            large.heap_bytes(),
            small.heap_bytes()
        );
        // The dense pyramid this replaced: a third of the grid in f32.
        assert!(large.heap_bytes() * 20 < 311 * 223 * 4 / 3);
    }
}

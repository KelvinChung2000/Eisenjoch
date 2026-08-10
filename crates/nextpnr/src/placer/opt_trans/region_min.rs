//! Per-net region-minimum pyramid over a net's dist_cache row.
//!
//! Given a dist grid of size `width × height` (row-major, node index
//! `y*width + x`), this builds levels 1..=max_level where each level-k cell at
//! pyramid coordinate `(cx, cy)` stores the minimum value of the dist grid
//! over the rectangle `[2^k·cx, 2^k·(cx+1)) × [2^k·cy, 2^k·(cy+1))`, clamped
//! to `[0, width) × [0, height)`. Level 0 is the raw dist grid (not stored
//! here — callers access `dist_cache` directly).
//!
//! The pyramid is used by the branch-and-bound search in `bb_2d` to prune
//! rectangles whose lower bound already exceeds the current best cost.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub(crate) struct PyrLevel {
    pub w: i32,
    pub h: i32,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct RegionMinPyramid {
    pub width: i32,
    pub height: i32,
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
        lvl.data[(cy * lvl.w + cx) as usize]
    }

    /// Build a pyramid from a dist grid. `dist` has `width*height` entries,
    /// row-major.
    pub fn build(dist: &[f32], width: i32, height: i32) -> Self {
        assert_eq!(dist.len(), (width as usize) * (height as usize));
        let mut levels: Vec<PyrLevel> = Vec::new();

        // Level 1 directly from raw dist (already f32 in the cache).
        let mut cur_w = width;
        let mut cur_h = height;
        {
            let w1 = (cur_w + 1) / 2;
            let h1 = (cur_h + 1) / 2;
            let mut data = vec![f32::INFINITY; (w1 * h1) as usize];
            for cy in 0..h1 {
                for cx in 0..w1 {
                    let mut m = f32::INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let gx = 2 * cx + dx;
                            let gy = 2 * cy + dy;
                            if gx < cur_w && gy < cur_h {
                                let v = dist[(gy * cur_w + gx) as usize];
                                if v < m {
                                    m = v;
                                }
                            }
                        }
                    }
                    data[(cy * w1 + cx) as usize] = m;
                }
            }
            levels.push(PyrLevel { w: w1, h: h1, data });
            cur_w = w1;
            cur_h = h1;
        }

        // Levels 2.. from previous pyramid level.
        while cur_w > 1 || cur_h > 1 {
            let w_next = (cur_w + 1) / 2;
            let h_next = (cur_h + 1) / 2;
            let prev = levels.last().unwrap();
            let prev_w = prev.w;
            let prev_h = prev.h;
            let prev_data = &prev.data;
            let mut data = vec![f32::INFINITY; (w_next * h_next) as usize];
            for cy in 0..h_next {
                for cx in 0..w_next {
                    let mut m = f32::INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let px = 2 * cx + dx;
                            let py = 2 * cy + dy;
                            if px < prev_w && py < prev_h {
                                let v = prev_data[(py * prev_w + px) as usize];
                                if v < m {
                                    m = v;
                                }
                            }
                        }
                    }
                    data[(cy * w_next + cx) as usize] = m;
                }
            }
            levels.push(PyrLevel {
                w: w_next,
                h: h_next,
                data,
            });
            cur_w = w_next;
            cur_h = h_next;
        }

        Self {
            width,
            height,
            levels,
        }
    }

    /// Total stored bytes (for diagnostics).
    pub fn bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|l| l.data.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Build pyramids for every net in parallel from sparse per-net rows.
/// `rows[ni]` is the settled-node set for net `ni`; absent nodes are
/// treated as `INFINITY`. `n_nodes` must equal `width * height`.
///
/// Each worker thread reuses a single dense `Vec<f32>` scratch (size
/// `n_nodes`) across all nets it processes — `map_init` ensures the
/// allocation is paid once per thread, not once per net.
pub(crate) fn build_all(
    rows: &[FxHashMap<u32, f32>],
    n_nodes: usize,
    width: i32,
    height: i32,
) -> Vec<RegionMinPyramid> {
    let stride = (width as usize) * (height as usize);
    assert_eq!(stride, n_nodes);
    rows.par_iter()
        .map_init(
            || vec![f32::INFINITY; stride],
            |scratch, row| {
                scratch.fill(f32::INFINITY);
                for (&node, &v) in row {
                    scratch[node as usize] = v;
                }
                RegionMinPyramid::build(scratch, width, height)
            },
        )
        .collect()
}

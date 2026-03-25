use crate::context::Context;
use crate::placer::PlacerError;
use log::debug;

use super::state::HeapState;

/// A rectangular region used during the spreading phase.
pub(super) struct Region {
    pub(super) x0: i32,
    pub(super) y0: i32,
    pub(super) x1: i32,
    pub(super) y1: i32,
    /// Indices into HeapState::movable_cells (the movable cell index, not CellIdx).
    pub(super) cells: Vec<usize>,
    /// Number of available BELs in this region.
    pub(super) bel_count: usize,
}

pub(super) struct BelPrefixGrid {
    width: i32,
    height: i32,
    prefix: Vec<usize>,
}

impl BelPrefixGrid {
    pub(super) fn build(ctx: &Context) -> Self {
        let width = ctx.chipdb().width().max(0);
        let height = ctx.chipdb().height().max(0);
        let stride = (width + 1) as usize;
        let mut prefix = vec![0usize; ((height + 1) as usize) * stride];

        for bel in ctx.bels() {
            let loc = bel.loc();
            if loc.x < 0 || loc.y < 0 || loc.x >= width || loc.y >= height {
                continue;
            }
            let idx = ((loc.y + 1) as usize) * stride + (loc.x + 1) as usize;
            prefix[idx] += 1;
        }

        for y in 1..=height as usize {
            for x in 1..=width as usize {
                let idx = y * stride + x;
                let left = y * stride + (x - 1);
                let up = (y - 1) * stride + x;
                let up_left = (y - 1) * stride + (x - 1);
                prefix[idx] += prefix[left] + prefix[up] - prefix[up_left];
            }
        }

        Self {
            width,
            height,
            prefix,
        }
    }

    #[inline]
    pub(super) fn count_in_region(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
        if self.width <= 0 || self.height <= 0 {
            return 0;
        }

        let xa = x0.clamp(0, self.width - 1);
        let ya = y0.clamp(0, self.height - 1);
        let xb = x1.clamp(0, self.width - 1);
        let yb = y1.clamp(0, self.height - 1);

        if xa > xb || ya > yb {
            return 0;
        }

        let stride = (self.width + 1) as usize;
        let x0p = xa as usize;
        let y0p = ya as usize;
        let x1p = (xb + 1) as usize;
        let y1p = (yb + 1) as usize;

        let a = self.prefix[y1p * stride + x1p];
        let b = self.prefix[y0p * stride + x1p];
        let c = self.prefix[y1p * stride + x0p];
        let d = self.prefix[y0p * stride + x0p];
        a - b - c + d
    }
}

impl HeapState {
    /// Spread cells via recursive bisection to reduce overlap.
    ///
    /// Returns a quality metric in [0, 1] where 1.0 means no overlap.
    pub fn spread(&mut self, ctx: &Context) -> Result<f64, PlacerError> {
        let n = self.movable_cells.len();
        if n == 0 {
            return Ok(1.0);
        }

        let bel_grid = BelPrefixGrid::build(ctx);

        let total_bels = bel_grid.count_in_region(0, 0, self.grid_w - 1, self.grid_h - 1);

        let initial_region = Region {
            x0: 0,
            y0: 0,
            x1: self.grid_w - 1,
            y1: self.grid_h - 1,
            cells: (0..n).collect(),
            bel_count: total_bels,
        };

        let mut leaf_regions: Vec<Region> = Vec::new();
        let mut stack: Vec<Region> = vec![initial_region];

        while let Some(region) = stack.pop() {
            if region.cells.is_empty() {
                continue;
            }

            if region.cells.len() <= region.bel_count {
                leaf_regions.push(region);
                continue;
            }

            let width = region.x1 - region.x0;
            let height = region.y1 - region.y0;

            if width <= 0 && height <= 0 {
                leaf_regions.push(region);
                continue;
            }

            let split_horizontal = width >= height;

            // Compute the split midpoint along the chosen axis.
            let (lo, hi) = if split_horizontal {
                (region.x0, region.x1)
            } else {
                (region.y0, region.y1)
            };
            let mid = (lo + hi) / 2;

            if mid == lo && hi > lo {
                leaf_regions.push(region);
                continue;
            }

            // Sort cells along the split axis.
            let positions = if split_horizontal {
                &self.cell_x
            } else {
                &self.cell_y
            };
            let mut sorted_cells = region.cells.clone();
            sorted_cells.sort_by(|&a, &b| {
                positions[a]
                    .partial_cmp(&positions[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Count BELs in each sub-region.
            let (lo_bels, hi_bels) = if split_horizontal {
                (
                    bel_grid.count_in_region(region.x0, region.y0, mid, region.y1),
                    bel_grid.count_in_region(mid + 1, region.y0, region.x1, region.y1),
                )
            } else {
                (
                    bel_grid.count_in_region(region.x0, region.y0, region.x1, mid),
                    bel_grid.count_in_region(region.x0, mid + 1, region.x1, region.y1),
                )
            };

            // Split cells proportionally to BEL counts.
            let total_bels_here = lo_bels + hi_bels;
            let lo_capacity = if total_bels_here > 0 {
                (sorted_cells.len() * lo_bels) / total_bels_here
            } else {
                sorted_cells.len() / 2
            };
            let lo_capacity = lo_capacity.min(sorted_cells.len());

            let hi_cells = sorted_cells.split_off(lo_capacity);
            let lo_cells = sorted_cells;

            // Clamp cell positions into their assigned sub-region.
            let positions = if split_horizontal {
                &mut self.cell_x
            } else {
                &mut self.cell_y
            };
            for &idx in &lo_cells {
                positions[idx] = positions[idx].clamp(lo as f64, mid as f64);
            }
            for &idx in &hi_cells {
                positions[idx] = positions[idx].clamp((mid + 1) as f64, hi as f64);
            }

            // Push the two sub-regions.
            let (lo_region, hi_region) = if split_horizontal {
                (
                    Region {
                        x0: region.x0,
                        y0: region.y0,
                        x1: mid,
                        y1: region.y1,
                        cells: lo_cells,
                        bel_count: lo_bels,
                    },
                    Region {
                        x0: mid + 1,
                        y0: region.y0,
                        x1: region.x1,
                        y1: region.y1,
                        cells: hi_cells,
                        bel_count: hi_bels,
                    },
                )
            } else {
                (
                    Region {
                        x0: region.x0,
                        y0: region.y0,
                        x1: region.x1,
                        y1: mid,
                        cells: lo_cells,
                        bel_count: lo_bels,
                    },
                    Region {
                        x0: region.x0,
                        y0: mid + 1,
                        x1: region.x1,
                        y1: region.y1,
                        cells: hi_cells,
                        bel_count: hi_bels,
                    },
                )
            };
            stack.push(lo_region);
            stack.push(hi_region);
        }

        // Quality: ratio of cells that fit into their leaf regions.
        let cells_fitting: usize = leaf_regions
            .iter()
            .map(|r| r.cells.len().min(r.bel_count))
            .sum();
        let quality = cells_fitting as f64 / n as f64;

        debug!("HeAP: spreading quality = {:.4}", quality);
        Ok(quality)
    }
}

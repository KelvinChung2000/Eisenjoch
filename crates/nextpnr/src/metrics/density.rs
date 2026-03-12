//! Spatial placement density metric.

use crate::context::Context;

/// Placement density report.
#[derive(Debug, Clone)]
pub struct DensityReport {
    /// Highest regional density (0.0-1.0).
    pub max_density: f64,
    /// Average regional density.
    pub avg_density: f64,
    /// (x, y) tile coords of the densest region's top-left corner.
    pub hotspot: (i32, i32),
    /// Count of regions above 50% density.
    pub hot_regions: usize,
    /// List of (x, y, density) for regions above 50%.
    pub grid: Vec<(i32, i32, f64)>,
}

/// Compute spatial placement density using a sliding window.
///
/// Divides the chip into regions of `window` x `window` tiles and computes
/// the fraction of BELs occupied in each region.
pub fn placement_density(ctx: &Context, window: i32) -> DensityReport {
    let chipdb = ctx.chipdb();
    let grid_w = chipdb.width();
    let grid_h = chipdb.height();
    let n_tiles = (grid_w * grid_h) as usize;

    let mut tile_placed = vec![0u32; n_tiles];
    let mut tile_capacity = vec![0u32; n_tiles];

    for tile_idx in 0..n_tiles {
        let tt = chipdb.tile_type(tile_idx as i32);
        tile_capacity[tile_idx] = tt.bels.get().len() as u32;
    }

    for cell in ctx.cells() {
        if !cell.is_alive() {
            continue;
        }
        if let Some(bel_id) = cell.bel_id() {
            let tile = bel_id.tile() as usize;
            if tile < n_tiles {
                tile_placed[tile] += 1;
            }
        }
    }

    compute_sliding_window_density(&tile_placed, &tile_capacity, grid_w, grid_h, window)
}

/// Compute sliding window density from pre-computed tile arrays.
///
/// `tile_placed[idx]` = number of placed cells in tile idx.
/// `tile_capacity[idx]` = number of BELs in tile idx.
/// Tile index = y * grid_w + x.
pub fn compute_sliding_window_density(
    tile_placed: &[u32],
    tile_capacity: &[u32],
    grid_w: i32,
    grid_h: i32,
    window: i32,
) -> DensityReport {
    let mut max_density = 0.0f64;
    let mut density_sum = 0.0f64;
    let mut region_count = 0usize;
    let mut hotspot = (0i32, 0i32);
    let mut hot_regions: Vec<(i32, i32, f64)> = Vec::new();

    let step = std::cmp::max(1, window / 2);
    for wy in (0..grid_h).step_by(step as usize) {
        for wx in (0..grid_w).step_by(step as usize) {
            let mut placed = 0u32;
            let mut capacity = 0u32;
            for dy in 0..window {
                for dx in 0..window {
                    let tx = wx + dx;
                    let ty = wy + dy;
                    if tx < grid_w && ty < grid_h {
                        let idx = (ty * grid_w + tx) as usize;
                        placed += tile_placed[idx];
                        capacity += tile_capacity[idx];
                    }
                }
            }
            if capacity > 0 {
                let density = placed as f64 / capacity as f64;
                density_sum += density;
                region_count += 1;
                if density > max_density {
                    max_density = density;
                    hotspot = (wx, wy);
                }
                if density > 0.5 {
                    hot_regions.push((wx, wy, density));
                }
            }
        }
    }

    let avg_density = if region_count > 0 {
        density_sum / region_count as f64
    } else {
        0.0
    };

    DensityReport {
        max_density,
        avg_density,
        hotspot,
        hot_regions: hot_regions.len(),
        grid: hot_regions,
    }
}

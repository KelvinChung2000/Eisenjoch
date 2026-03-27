//! Net demand computation and bilinear interpolation for the Beckmann placer.
//!
//! All interpolation operates on the subtile grid: a regular (W·N) × (H·N) lattice
//! where N is the subtile resolution. Cell positions in tile coordinates are mapped
//! to subtile coordinates for demand injection and pressure gradient extraction.

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

use super::config::OptTransPlacerCfg;
use super::network::PipeNetwork;

/// Convert tile-coordinate position to subtile-grid coordinate.
///
/// Subtile node (gx, gy) has its center at tile position ((gx+0.5)/N, (gy+0.5)/N).
/// Inverting: subtile_coord = tile_coord * N - 0.5.
#[inline]
fn to_subtile_coord(tile_pos: f64, resolution: usize) -> f64 {
    tile_pos * resolution as f64 - 0.5
}

/// Clamp and compute bilinear interpolation coordinates on the subtile grid.
///
/// Input: position in subtile coordinates.
/// Returns (gx0, gy0, fx, fy) where gx0/gy0 are lower-left subtile indices
/// and fx/fy are fractional offsets in [0, 1].
#[inline]
fn bilinear_cell(sx: f64, sy: f64, sw: usize, sh: usize) -> (usize, usize, f64, f64) {
    let max_x = (sw - 1) as f64;
    let max_y = (sh - 1) as f64;
    let sx = sx.clamp(0.0, max_x);
    let sy = sy.clamp(0.0, max_y);

    let gx0 = (sx.floor() as usize).min(sw - 2);
    let gy0 = (sy.floor() as usize).min(sh - 2);
    let fx = sx - gx0 as f64;
    let fy = sy - gy0 as f64;
    (gx0, gy0, fx, fy)
}

/// Bilinear weights on subtile grid: maps continuous position to 4 surrounding
/// subtile nodes with weights. Returns [(gx, gy, weight); 4].
#[inline]
fn bilinear_weights(sx: f64, sy: f64, sw: usize, sh: usize) -> [(usize, usize, f64); 4] {
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);
    [
        (gx0, gy0, (1.0 - fx) * (1.0 - fy)),
        (gx0 + 1, gy0, fx * (1.0 - fy)),
        (gx0, gy0 + 1, (1.0 - fx) * fy),
        (gx0 + 1, gy0 + 1, fx * fy),
    ]
}

/// Bilinear gradient of a scalar field at fractional position.
///
/// Given 4 corner values (f00, f10, f01, f11) and fractional offsets (fx, fy):
///   df/dx = (1-fy)*(f10-f00) + fy*(f11-f01)
///   df/dy = (1-fx)*(f01-f00) + fx*(f11-f10)
#[inline]
fn bilinear_gradient(fx: f64, fy: f64, f00: f64, f10: f64, f01: f64, f11: f64) -> (f64, f64) {
    let gx = (1.0 - fy) * (f10 - f00) + fy * (f11 - f01);
    let gy = (1.0 - fx) * (f01 - f00) + fx * (f11 - f10);
    (gx, gy)
}

/// Subtile grid index from subtile-grid coordinates (gx, gy).
///
/// The subtile grid is stored in tile-major order:
///   node_index(tx, ty, sx, sy) = (ty*W + tx) * N² + sy*N + sx
///
/// Given global subtile coords (gx, gy):
///   tx = gx / N, sx = gx % N, ty = gy / N, sy = gy % N
#[inline]
fn subtile_grid_index(gx: usize, gy: usize, tile_w: usize, n: usize) -> usize {
    let tx = gx / n;
    let sx = gx % n;
    let ty = gy / n;
    let sy = gy % n;
    (ty * tile_w + tx) * (n * n) + sy * n + sx
}

/// Get position of a pin (movable or fixed) in virtual tile coordinates.
fn pin_pos(
    ctx: &Context,
    cell_id: CellId,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (f64, f64) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        (cell_x[idx], cell_y[idx])
    } else {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            (
                (loc.x - network.x0) as f64,
                (loc.y - network.y0) as f64,
            )
        } else {
            (network.width as f64 / 2.0, network.height as f64 / 2.0)
        }
    }
}

/// Build node demand vector from cell positions.
///
/// For each net, the driver injects +demand and each sink extracts -demand/fanout,
/// both bilinearly interpolated on the subtile grid.
pub fn build_demands(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let n_nodes = network.num_nodes();
    let mut demand = vec![0.0; n_nodes];
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };

        let mut sink_positions: Vec<(f64, f64)> = Vec::new();
        let mut has_fixed_pin = ctx.design.cell(dp.cell).bel_strength.is_locked();

        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            has_fixed_pin |= ctx.design.cell(user.cell).bel_strength.is_locked();
            sink_positions.push(pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network));
        }
        if sink_positions.is_empty() {
            continue;
        }

        let (dx, dy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let fanout = sink_positions.len() as f64;
        let io_factor = if has_fixed_pin { cfg.io_boost } else { 1.0 };

        // Driver injects +demand, bilinearly spread across 4 subtile nodes.
        let dsx = to_subtile_coord(dx, n);
        let dsy = to_subtile_coord(dy, n);
        for (gx, gy, bw) in bilinear_weights(dsx, dsy, sw, sh) {
            let ni = subtile_grid_index(gx, gy, tile_w, n);
            demand[ni] += io_factor * bw;
        }

        // Each sink extracts -demand/fanout.
        let sink_weight = io_factor / fanout;
        for &(sx, sy) in &sink_positions {
            let ssx = to_subtile_coord(sx, n);
            let ssy = to_subtile_coord(sy, n);
            for (gx, gy, bw) in bilinear_weights(ssx, ssy, sw, sh) {
                let ni = subtile_grid_index(gx, gy, tile_w, n);
                demand[ni] -= sink_weight * bw;
            }
        }
    }

    demand
}

/// Compute cell displacement from -grad(P) at each cell position.
///
/// Cells move along the flow direction. When congestion reroutes flow,
/// the gradient changes and cells follow the rerouted path.
pub fn compute_displacement_gradient(
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (Vec<f64>, Vec<f64>) {
    let num_cells = cell_x.len();
    let mut dx = vec![0.0; num_cells];
    let mut dy = vec![0.0; num_cells];

    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    for i in 0..num_cells {
        let sx = to_subtile_coord(cell_x[i], n);
        let sy = to_subtile_coord(cell_y[i], n);
        let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);

        let p00 = network.nodes[subtile_grid_index(gx0, gy0, tile_w, n)].pressure;
        let p10 = network.nodes[subtile_grid_index(gx0 + 1, gy0, tile_w, n)].pressure;
        let p01 = network.nodes[subtile_grid_index(gx0, gy0 + 1, tile_w, n)].pressure;
        let p11 = network.nodes[subtile_grid_index(gx0 + 1, gy0 + 1, tile_w, n)].pressure;

        let (gx, gy) = bilinear_gradient(fx, fy, p00, p10, p01, p11);

        // -grad(P) in tile coords (scale by N for subtile→tile).
        dx[i] = -gx * n as f64;
        dy[i] = -gy * n as f64;
    }

    (dx, dy)
}

/// Interpolate pressure at a tile-coordinate position using bilinear interpolation
/// on the subtile grid.
fn pressure_at(x: f64, y: f64, network: &PipeNetwork) -> f64 {
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    let sx = to_subtile_coord(x, n);
    let sy = to_subtile_coord(y, n);
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);

    let p00 = network.nodes[subtile_grid_index(gx0, gy0, tile_w, n)].pressure;
    let p10 = network.nodes[subtile_grid_index(gx0 + 1, gy0, tile_w, n)].pressure;
    let p01 = network.nodes[subtile_grid_index(gx0, gy0 + 1, tile_w, n)].pressure;
    let p11 = network.nodes[subtile_grid_index(gx0 + 1, gy0 + 1, tile_w, n)].pressure;

    (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11
}

/// Interpolate cell density at a tile-coordinate position.
/// Uses the pipe cell_density field, averaged across pipes incident to the
/// nearest subtile nodes.
fn density_at(x: f64, y: f64, network: &PipeNetwork) -> f64 {
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    let sx = to_subtile_coord(x, n);
    let sy = to_subtile_coord(y, n);
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);

    // Average cell_density of pipes touching the 4 surrounding nodes.
    let mut total = 0.0;
    let mut count = 0usize;
    for &(gx, gy) in &[(gx0, gy0), (gx0 + 1, gy0), (gx0, gy0 + 1), (gx0 + 1, gy0 + 1)] {
        let ni = subtile_grid_index(gx, gy, tile_w, n);
        if ni < network.node_pipes.len() {
            for &pi in &network.node_pipes[ni] {
                total += network.pipes[pi].cell_density;
                count += 1;
            }
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

/// Compute cell displacement from pressure differences between connected pins.
///
/// For each net, the pressure difference between the cell and each other pin
/// tells the cell which direction to move through the network. The displacement
/// is the sum over all connected pins of:
///   (P_other - P_self) * unit_direction_to_other
///
/// This avoids self-interaction: a cell's own demand doesn't affect its
/// displacement because P_self cancels out in the difference.
pub fn compute_displacement(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (Vec<f64>, Vec<f64>) {
    let num_cells = cell_x.len();
    let mut dx = vec![0.0; num_cells];
    let mut dy = vec![0.0; num_cells];

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };
        if net.num_users() == 0 {
            continue;
        }

        // Collect all pin positions on this net.
        let mut pins: Vec<(f64, f64, Option<usize>)> = Vec::new(); // (x, y, movable_idx)

        let (dpx, dpy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let dp_idx = cell_to_idx.get(&dp.cell).copied();
        pins.push((dpx, dpy, dp_idx));

        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let (ux, uy) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
            let u_idx = cell_to_idx.get(&user.cell).copied();
            pins.push((ux, uy, u_idx));
        }

        if pins.len() < 2 {
            continue;
        }

        // For each movable pin, compute displacement from pressure differences
        // with all other pins on this net.
        for i in 0..pins.len() {
            let Some(ci) = pins[i].2 else { continue }; // skip fixed pins
            let (xi, yi, _) = pins[i];
            let p_self = pressure_at(xi, yi, network);
            let d_self = density_at(xi, yi, network);

            for j in 0..pins.len() {
                if i == j {
                    continue;
                }
                let (xj, yj, _) = pins[j];
                let dir_x = xj - xi;
                let dir_y = yj - yi;
                let dist = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if dist < 0.1 {
                    continue; // pins coincident, no force
                }

                let p_other = pressure_at(xj, yj, network);
                let dp = p_other - p_self;

                // In dense regions, damp the pressure difference to prevent
                // congestion-amplified pull. The damping scales with density²
                // so it only activates when significantly overcrowded.
                let density_factor = 1.0 + 0.5 * d_self * d_self;
                let weight = dp / (dist * density_factor);
                dx[ci] += weight * dir_x;
                dy[ci] += weight * dir_y;
            }
        }
    }

    (dx, dy)
}

/// Compute congestion repulsion: push cells away from high-utilization regions.
///
/// Builds a utilization density field on the subtile grid from pipe flows,
/// then computes -grad(utilization) at each cell position. Cells are pushed
/// down the utilization gradient, away from congested areas.
pub fn compute_congestion_repulsion(
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (Vec<f64>, Vec<f64>) {
    let num_cells = cell_x.len();
    let mut dx = vec![0.0; num_cells];
    let mut dy = vec![0.0; num_cells];

    let n = network.resolution;
    let n_nodes = network.num_nodes();
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    // Build utilization density at each subtile node.
    // Each pipe's utilization (|flow|/capacity) is distributed to its endpoint nodes.
    let mut util_field = vec![0.0f64; n_nodes];
    let mut node_degree = vec![0u32; n_nodes];

    for pipe in &network.pipes {
        let util = (pipe.flow.abs() / pipe.capacity.max(1.0)).min(10.0);
        util_field[pipe.from] += util;
        util_field[pipe.to] += util;
        node_degree[pipe.from] += 1;
        node_degree[pipe.to] += 1;
    }

    // Normalize by degree to get average utilization at each node.
    for i in 0..n_nodes {
        if node_degree[i] > 0 {
            util_field[i] /= node_degree[i] as f64;
        }
    }

    // Compute -grad(utilization) at each cell position via bilinear interpolation.
    for i in 0..num_cells {
        let sx = to_subtile_coord(cell_x[i], n);
        let sy = to_subtile_coord(cell_y[i], n);
        let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);

        let u00 = util_field[subtile_grid_index(gx0, gy0, tile_w, n)];
        let u10 = util_field[subtile_grid_index(gx0 + 1, gy0, tile_w, n)];
        let u01 = util_field[subtile_grid_index(gx0, gy0 + 1, tile_w, n)];
        let u11 = util_field[subtile_grid_index(gx0 + 1, gy0 + 1, tile_w, n)];

        let (gx, gy) = bilinear_gradient(fx, fy, u00, u10, u01, u11);

        // Push away from congestion: -grad(util), scaled by N for subtile→tile coords.
        dx[i] = -gx * n as f64;
        dy[i] = -gy * n as f64;
    }

    (dx, dy)
}

/// Update net_count on pipes based on current cell positions.
///
/// For each net, determines which pipes its bounding box covers and
/// increments net_count.
pub fn update_net_counts(
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &mut PipeNetwork,
    ctx: &Context,
) {
    for pipe in &mut network.pipes {
        pipe.net_count = 0;
    }

    let w = network.width;

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };
        if net.num_users() == 0 {
            continue;
        }

        let (dxx, dyy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let (mut min_x, mut max_x) = (dxx, dxx);
        let (mut min_y, mut max_y) = (dyy, dyy);

        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let (ux, uy) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
            min_x = min_x.min(ux);
            max_x = max_x.max(ux);
            min_y = min_y.min(uy);
            max_y = max_y.max(uy);
        }

        let x0 = (min_x.floor() as i32).max(0);
        let x1 = (max_x.ceil() as i32).min(w - 1);
        let y0 = (min_y.floor() as i32).max(0);
        let y1 = (max_y.ceil() as i32).min(network.height - 1);

        let n_per_tile = network.nodes_per_tile();

        // Increment net_count for all pipes in tiles within the bounding box.
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let base = ((ty * w + tx) as usize) * n_per_tile;
                for offset in 0..n_per_tile {
                    let ni = base + offset;
                    if ni < network.node_pipes.len() {
                        for &pipe_idx in &network.node_pipes[ni] {
                            if network.pipes[pipe_idx].from == ni {
                                network.pipes[pipe_idx].net_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Update cell_density on pipes from current cell positions.
///
/// Each movable cell contributes density 1.0 to the pipes near its position
/// (bilinearly interpolated on the subtile grid). Density is normalized by
/// the pipe's placement capacity (BELs) so density=1.0 means fully utilized.
pub fn update_cell_density(
    cell_x: &[f64],
    cell_y: &[f64],
    network: &mut PipeNetwork,
) {
    // Reset all densities.
    for pipe in &mut network.pipes {
        pipe.cell_density = 0.0;
    }

    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    // Accumulate density at subtile nodes from cell positions.
    let n_nodes = network.num_nodes();
    let mut node_density = vec![0.0f64; n_nodes];

    for i in 0..cell_x.len() {
        let sx = to_subtile_coord(cell_x[i], n);
        let sy = to_subtile_coord(cell_y[i], n);
        for (gx, gy, bw) in bilinear_weights(sx, sy, sw, sh) {
            let ni = subtile_grid_index(gx, gy, tile_w, n);
            node_density[ni] += bw;
        }
    }

    // Transfer node density to pipes: each pipe gets the average of its endpoints.
    // Normalize by capacity so density=1.0 means at placement capacity.
    for pipe in &mut network.pipes {
        let d_from = node_density[pipe.from];
        let d_to = node_density[pipe.to];
        let avg = (d_from + d_to) * 0.5;
        // Normalize by capacity (BEL-based).
        pipe.cell_density = avg / pipe.capacity.max(0.25);
    }
}

/// Compute continuous HPWL from cell positions.
pub fn continuous_hpwl(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> f64 {
    let mut total = 0.0;
    for (_, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };

        let (dxx, dyy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let (mut min_x, mut max_x) = (dxx, dxx);
        let (mut min_y, mut max_y) = (dyy, dyy);

        let mut has_valid_sink = false;
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            has_valid_sink = true;
            let (x, y) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if !has_valid_sink {
            continue;
        }

        total += (max_x - min_x) + (max_y - min_y);
    }
    total
}

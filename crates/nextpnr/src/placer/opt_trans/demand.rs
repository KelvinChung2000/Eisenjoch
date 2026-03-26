//! Net demand computation and bilinear interpolation for the Beckmann placer.
//!
//! - `build_demands`: injects flow demand at junction nodes from cell positions
//! - `compute_displacement`: computes cell movement from pressure gradients
//! - `update_net_counts`: tracks number of distinct nets per pipe for interference

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

use super::config::OptTransPlacerCfg;
use super::network::{PipeNetwork, Port};

const ALL_PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

/// Clamp and compute bilinear interpolation coordinates.
///
/// Returns (x0, y0, fx, fy) where x0/y0 are lower-left tile indices
/// and fx/fy are fractional offsets in [0, 1].
#[inline]
fn bilinear_cell(x: f64, y: f64, width: i32, height: i32) -> (i32, i32, f64, f64) {
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);

    let x0 = (x.floor() as i32).clamp(0, width - 2);
    let y0 = (y.floor() as i32).clamp(0, height - 2);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    (x0, y0, fx, fy)
}

/// Bilinear weights: maps continuous (x, y) to 4 surrounding tiles with weights.
/// Returns [(tile_x, tile_y, weight); 4].
#[inline]
fn bilinear_weights(x: f64, y: f64, width: i32, height: i32) -> [(i32, i32, f64); 4] {
    let (x0, y0, fx, fy) = bilinear_cell(x, y, width, height);
    [
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x0 + 1, y0, fx * (1.0 - fy)),
        (x0, y0 + 1, (1.0 - fx) * fy),
        (x0 + 1, y0 + 1, fx * fy),
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

/// Get position of a pin (movable or fixed) in virtual grid coordinates.
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

/// Build junction demand vector from cell positions.
///
/// For each net, the driver injects +demand at its position and each sink
/// extracts -demand/fanout, both bilinearly interpolated across the 4
/// surrounding tile junctions.
pub fn build_demands(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let n_j = network.num_junctions();
    let mut demand = vec![0.0; n_j];

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
        let port_share = 0.25 * io_factor;

        // Driver injects +demand, bilinearly spread across 4 junction ports.
        for (tx, ty, bw) in bilinear_weights(dx, dy, network.width, network.height) {
            let share = port_share * bw;
            for &port in &ALL_PORTS {
                demand[network.junction_index(tx, ty, port)] += share;
            }
        }

        // Each sink extracts -demand/fanout.
        let sink_share = port_share / fanout;
        for &(sx, sy) in &sink_positions {
            for (tx, ty, bw) in bilinear_weights(sx, sy, network.width, network.height) {
                let share = sink_share * bw;
                for &port in &ALL_PORTS {
                    demand[network.junction_index(tx, ty, port)] -= share;
                }
            }
        }
    }

    demand
}

/// Compute cell displacement from the pressure field.
///
/// For each cell, computes -grad(P) at the cell position using bilinear
/// interpolation of junction pressures. Returns (dx, dy) per cell.
pub fn compute_displacement(
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (Vec<f64>, Vec<f64>) {
    let n = cell_x.len();
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; n];

    // Build a tile-level pressure field by averaging the 4 port pressures per tile.
    let w = network.width as usize;
    let h = network.height as usize;
    let mut tile_pressure = vec![0.0; w * h];
    for ty in 0..h {
        for tx in 0..w {
            let mut sum = 0.0;
            for &port in &ALL_PORTS {
                let ji = network.junction_index(tx as i32, ty as i32, port);
                sum += network.junctions[ji].pressure;
            }
            tile_pressure[ty * w + tx] = sum * 0.25;
        }
    }

    // For each cell, compute -grad(P) via bilinear interpolation.
    for i in 0..n {
        let (x0, y0, fx, fy) = bilinear_cell(cell_x[i], cell_y[i], network.width, network.height);
        let col0 = x0 as usize;
        let col1 = (x0 + 1) as usize;
        let row0 = y0 as usize * w;
        let row1 = (y0 + 1) as usize * w;

        let f00 = tile_pressure[row0 + col0];
        let f10 = tile_pressure[row0 + col1];
        let f01 = tile_pressure[row1 + col0];
        let f11 = tile_pressure[row1 + col1];

        let (gx, gy) = bilinear_gradient(fx, fy, f00, f10, f01, f11);
        // Cells move along -grad(P).
        dx[i] = -gx;
        dy[i] = -gy;
    }

    (dx, dy)
}

/// Update net_count on pipes based on current cell positions.
///
/// For each net, determines which pipes its bounding box covers and
/// increments net_count. This is an approximation: a net "uses" all
/// pipes within the bounding box of its pins.
pub fn update_net_counts(
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &mut PipeNetwork,
    ctx: &Context,
) {
    // Reset all net counts.
    for pipe in &mut network.pipes {
        pipe.net_count = 0;
    }

    let w = network.width;

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };
        if net.num_users() == 0 {
            continue;
        }

        // Compute bounding box of the net in virtual coords.
        let (dx, dy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let (mut min_x, mut max_x) = (dx, dx);
        let (mut min_y, mut max_y) = (dy, dy);

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

        // Increment net_count for all pipes in tiles within the bounding box.
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let base = ((ty * w + tx) * 4) as usize;
                // All junction pipes for this tile's 4 ports.
                for port_offset in 0..4 {
                    let ji = base + port_offset;
                    if ji < network.junction_pipes.len() {
                        for &pipe_idx in &network.junction_pipes[ji] {
                            // Avoid double-counting: only increment from the
                            // lower junction to avoid counting each pipe twice.
                            if network.pipes[pipe_idx].from == ji {
                                network.pipes[pipe_idx].net_count += 1;
                            }
                        }
                    }
                }
            }
        }
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

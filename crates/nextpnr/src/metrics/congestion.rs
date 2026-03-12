//! Edge-based directional congestion estimation.
//!
//! Estimates routing congestion by tracking per-edge demand using Bresenham
//! lines from net drivers to sinks. Each tile boundary has two edges (East
//! and South). West/North edges are the East/South edges of adjacent tiles.

use crate::context::Context;
use crate::netlist::NetId;
use rayon::prelude::*;

/// Direction of a tile-boundary edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// East-West boundary (horizontal routing crossing a vertical edge).
    Horizontal,
    /// North-South boundary (vertical routing crossing a horizontal edge).
    Vertical,
}

/// Congestion estimation report.
#[derive(Debug, Clone)]
pub struct CongestionReport {
    /// East-edge demand grid [y][x] -- number of nets crossing the east boundary of tile (x,y).
    pub h_demand: Vec<Vec<f64>>,
    /// South-edge demand grid [y][x] -- number of nets crossing the south boundary of tile (x,y).
    pub v_demand: Vec<Vec<f64>>,
    /// Horizontal congestion ratio (h_demand / h_capacity).
    pub h_congestion: Vec<Vec<f64>>,
    /// Vertical congestion ratio (v_demand / v_capacity).
    pub v_congestion: Vec<Vec<f64>>,
    /// Maximum congestion ratio across all edges.
    pub max_congestion: f64,
    /// Average congestion ratio across all edges.
    pub avg_congestion: f64,
    /// Tile coordinate of the most congested edge.
    pub hotspot: (i32, i32),
    /// Axis of the most congested edge.
    pub hotspot_axis: Axis,
    /// Edges with congestion above the given threshold: (x, y, axis, congestion).
    pub hot_edges: Vec<(i32, i32, Axis, f64)>,
}

/// Compute points along a Bresenham line from (x0, y0) to (x1, y1).
///
/// Returns all integer grid points along the line, including both endpoints.
/// Handles all octants (steep, shallow, positive and negative directions).
pub fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut points = Vec::new();

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut x = x0;
    let mut y = y0;

    if dx >= dy {
        // Shallow line (more horizontal)
        let mut err = dx / 2;
        for _ in 0..=dx {
            points.push((x, y));
            err -= dy;
            if err < 0 {
                y += sy;
                err += dx;
            }
            x += sx;
        }
    } else {
        // Steep line (more vertical)
        let mut err = dy / 2;
        for _ in 0..=dy {
            points.push((x, y));
            err -= dx;
            if err < 0 {
                x += sx;
                err += dy;
            }
            y += sy;
        }
    }

    points
}

/// Estimate routing congestion using edge-based demand with Bresenham lines.
///
/// For each alive net with a placed driver and sinks, draws Bresenham lines
/// from driver to each sink. Each consecutive tile pair in the line increments
/// the appropriate edge demand counter.
///
/// Edge capacity is estimated as total_wires / 4 per direction (since we don't
/// have per-direction wire type info in the generic chipdb).
///
/// Uses rayon for parallel demand accumulation across nets.
pub fn estimate_congestion(ctx: &Context, threshold: f64) -> CongestionReport {
    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    let wu = w as usize;
    let hu = h as usize;

    // Build capacity grids.
    // Use total_wires / 4 as per-direction capacity estimate.
    let mut h_capacity = vec![vec![0.0f64; wu]; hu];
    let mut v_capacity = vec![vec![0.0f64; wu]; hu];
    for ty in 0..h {
        for tx in 0..w {
            let tile_idx = ty * w + tx;
            let tt = ctx.chipdb().tile_type(tile_idx);
            let nwires = tt.wires.get().len() as f64;
            let cap = (nwires / 4.0).max(1.0);
            h_capacity[ty as usize][tx as usize] = cap;
            v_capacity[ty as usize][tx as usize] = cap;
        }
    }

    // Collect net indices for parallel iteration.
    let net_indices: Vec<NetId> = ctx
        .design
        .iter_alive_nets()
        .map(|(idx, _)| idx)
        .collect();

    // Parallel demand accumulation with per-thread grids, then reduce.
    let (h_demand, v_demand) = net_indices
        .par_iter()
        .fold(
            || (vec![vec![0.0f64; wu]; hu], vec![vec![0.0f64; wu]; hu]),
            |(mut local_h, mut local_v), &net_idx| {
                accumulate_net_demand(ctx, net_idx, w, h, &mut local_h, &mut local_v);
                (local_h, local_v)
            },
        )
        .reduce(
            || (vec![vec![0.0f64; wu]; hu], vec![vec![0.0f64; wu]; hu]),
            |(mut ah, mut av), (bh, bv)| {
                for y in 0..hu {
                    for x in 0..wu {
                        ah[y][x] += bh[y][x];
                        av[y][x] += bv[y][x];
                    }
                }
                (ah, av)
            },
        );

    let ratios = compute_congestion_ratios(&h_demand, &v_demand, &h_capacity, &v_capacity, threshold);

    CongestionReport {
        h_demand,
        v_demand,
        h_congestion: ratios.h_congestion,
        v_congestion: ratios.v_congestion,
        max_congestion: ratios.max_congestion,
        avg_congestion: ratios.avg_congestion,
        hotspot: ratios.hotspot,
        hotspot_axis: ratios.hotspot_axis,
        hot_edges: ratios.hot_edges,
    }
}

/// Result of computing congestion ratios from demand and capacity grids.
pub struct CongestionRatios {
    pub h_congestion: Vec<Vec<f64>>,
    pub v_congestion: Vec<Vec<f64>>,
    pub max_congestion: f64,
    pub avg_congestion: f64,
    pub hotspot: (i32, i32),
    pub hotspot_axis: Axis,
    pub hot_edges: Vec<(i32, i32, Axis, f64)>,
}

/// Compute congestion ratios from demand and capacity grids.
pub fn compute_congestion_ratios(
    h_demand: &[Vec<f64>],
    v_demand: &[Vec<f64>],
    h_capacity: &[Vec<f64>],
    v_capacity: &[Vec<f64>],
    threshold: f64,
) -> CongestionRatios {
    let hu = h_demand.len();
    let wu = if hu > 0 { h_demand[0].len() } else { 0 };

    let mut result = CongestionRatios {
        h_congestion: vec![vec![0.0f64; wu]; hu],
        v_congestion: vec![vec![0.0f64; wu]; hu],
        max_congestion: 0.0,
        avg_congestion: 0.0,
        hotspot: (0, 0),
        hotspot_axis: Axis::Horizontal,
        hot_edges: Vec::new(),
    };
    let mut congestion_sum = 0.0f64;
    let mut edge_count = 0u64;

    for y in 0..hu {
        for x in 0..wu {
            if x + 1 < wu {
                let ratio = h_demand[y][x] / h_capacity[y][x];
                result.h_congestion[y][x] = ratio;
                congestion_sum += ratio;
                edge_count += 1;
                if ratio > result.max_congestion {
                    result.max_congestion = ratio;
                    result.hotspot = (x as i32, y as i32);
                    result.hotspot_axis = Axis::Horizontal;
                }
                if ratio > threshold {
                    result.hot_edges.push((x as i32, y as i32, Axis::Horizontal, ratio));
                }
            }

            if y + 1 < hu {
                let ratio = v_demand[y][x] / v_capacity[y][x];
                result.v_congestion[y][x] = ratio;
                congestion_sum += ratio;
                edge_count += 1;
                if ratio > result.max_congestion {
                    result.max_congestion = ratio;
                    result.hotspot = (x as i32, y as i32);
                    result.hotspot_axis = Axis::Vertical;
                }
                if ratio > threshold {
                    result.hot_edges.push((x as i32, y as i32, Axis::Vertical, ratio));
                }
            }
        }
    }

    result.avg_congestion = if edge_count > 0 {
        congestion_sum / edge_count as f64
    } else {
        0.0
    };

    result
}

/// Accumulate demand from a single net into local demand grids.
fn accumulate_net_demand(
    ctx: &Context,
    net_idx: NetId,
    grid_w: i32,
    grid_h: i32,
    h_demand: &mut [Vec<f64>],
    v_demand: &mut [Vec<f64>],
) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }

    let Some(driver_pin) = net.driver() else {
        return;
    };
    let Some(driver_bel) = ctx.cell(driver_pin.cell).bel() else {
        return;
    };
    let driver_loc = driver_bel.loc();

    for user in net.users() {
        if !user.is_valid() {
            continue;
        }
        let Some(sink_bel) = ctx.cell(user.cell).bel() else {
            continue;
        };
        let sink_loc = sink_bel.loc();

        let points = bresenham_line(driver_loc.x, driver_loc.y, sink_loc.x, sink_loc.y);
        accumulate_edge_crossings(&points, grid_w, grid_h, h_demand, v_demand, 1.0);
    }
}

/// Given a Bresenham line (sequence of grid points), accumulate edge crossings
/// into demand grids. Each consecutive pair of points updates the appropriate
/// edge counter based on the direction of movement.
///
/// `weight` controls the increment: use +1.0 to add demand, -1.0 to remove it.
/// When subtracting, demand is clamped to zero to avoid negative values.
///
/// `h_demand[y][x]` tracks east-edge crossings at tile (x,y).
/// `v_demand[y][x]` tracks south-edge crossings at tile (x,y).
pub fn accumulate_edge_crossings(
    points: &[(i32, i32)],
    grid_w: i32,
    grid_h: i32,
    h_demand: &mut [Vec<f64>],
    v_demand: &mut [Vec<f64>],
    weight: f64,
) {
    for pair in points.windows(2) {
        let (x1, y1) = pair[0];
        let (x2, y2) = pair[1];
        let dx = x2 - x1;
        let dy = y2 - y1;

        if dx > 0 && x1 >= 0 && x1 < grid_w && y1 >= 0 && y1 < grid_h {
            let slot = &mut h_demand[y1 as usize][x1 as usize];
            *slot = (*slot + weight).max(0.0);
        } else if dx < 0 && x2 >= 0 && x2 < grid_w && y2 >= 0 && y2 < grid_h {
            let slot = &mut h_demand[y2 as usize][x2 as usize];
            *slot = (*slot + weight).max(0.0);
        }

        if dy > 0 && x1 >= 0 && x1 < grid_w && y1 >= 0 && y1 < grid_h {
            let slot = &mut v_demand[y1 as usize][x1 as usize];
            *slot = (*slot + weight).max(0.0);
        } else if dy < 0 && x2 >= 0 && x2 < grid_w && y2 >= 0 && y2 < grid_h {
            let slot = &mut v_demand[y2 as usize][x2 as usize];
            *slot = (*slot + weight).max(0.0);
        }
    }
}

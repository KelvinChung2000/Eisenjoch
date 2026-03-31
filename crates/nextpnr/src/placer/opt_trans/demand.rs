//! Net demand computation and bilinear interpolation for the Beckmann placer.
//!
//! All interpolation operates on the subtile grid: a regular (W·N) × (H·N) lattice
//! where N is the subtile resolution. Cell positions in tile coordinates are mapped
//! to subtile coordinates for demand injection and pressure gradient extraction.

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::CellId;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::config::OptTransPlacerCfg;
use super::network::PipeNetwork;

// ---------------------------------------------------------------------------
// Per-net Kirchhoff solve infrastructure
// ---------------------------------------------------------------------------

/// Information about a single net for per-net Kirchhoff solving.
pub struct NetSolveInfo {
    /// Pin positions and optional movable cell index.
    /// First element is always the driver.
    pub pins: Vec<(f64, f64, Option<usize>)>,
    /// Whether each pin is fixed (locked). Same indexing as `pins`.
    pub pin_is_fixed: Vec<bool>,
    /// Whether this net has at least one fixed/IO pin.
    pub has_fixed_pin: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridParams {
    pub resolution: usize,
    pub subtile_width: usize,
    pub subtile_height: usize,
    pub tile_width: usize,
}

impl GridParams {
    #[inline]
    pub(crate) fn from_network(network: &PipeNetwork) -> Self {
        Self {
            resolution: network.resolution,
            subtile_width: network.subtile_width(),
            subtile_height: network.subtile_height(),
            tile_width: network.width as usize,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DemandFactors {
    pub io_factor: f64,
    pub fanout: f64,
    pub sink_weight: f64,
}

#[inline]
pub(crate) fn demand_factors(info: &NetSolveInfo, cfg: &OptTransPlacerCfg) -> DemandFactors {
    let io_factor = if info.has_fixed_pin {
        cfg.io_boost
    } else {
        1.0
    };
    let fanout = (info.pins.len() - 1) as f64;
    let sink_weight = -io_factor / fanout;
    DemandFactors {
        io_factor,
        fanout,
        sink_weight,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BilinearJacobianStencil {
    pub nodes: [usize; 4],
    pub dw_dx: [f64; 4],
    pub dw_dy: [f64; 4],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BilinearCorners {
    pub nodes: [usize; 4],
    pub fx: f64,
    pub fy: f64,
}

#[inline]
pub(crate) fn bilinear_corners(tile_x: f64, tile_y: f64, params: &GridParams) -> BilinearCorners {
    let sx = to_subtile_coord(tile_x, params.resolution);
    let sy = to_subtile_coord(tile_y, params.resolution);
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, params.subtile_width, params.subtile_height);

    let nodes = [
        subtile_grid_index(gx0, gy0, params.tile_width, params.resolution),
        subtile_grid_index(gx0 + 1, gy0, params.tile_width, params.resolution),
        subtile_grid_index(gx0, gy0 + 1, params.tile_width, params.resolution),
        subtile_grid_index(gx0 + 1, gy0 + 1, params.tile_width, params.resolution),
    ];

    BilinearCorners { nodes, fx, fy }
}

#[inline]
pub(crate) fn bilinear_jacobian_stencil(
    tile_x: f64,
    tile_y: f64,
    params: &GridParams,
) -> BilinearJacobianStencil {
    let corners = bilinear_corners(tile_x, tile_y, params);
    let n_f = params.resolution as f64;

    let dw_dx = [
        -(1.0 - corners.fy) * n_f,
        (1.0 - corners.fy) * n_f,
        -corners.fy * n_f,
        corners.fy * n_f,
    ];
    let dw_dy = [
        -(1.0 - corners.fx) * n_f,
        -corners.fx * n_f,
        (1.0 - corners.fx) * n_f,
        corners.fx * n_f,
    ];

    BilinearJacobianStencil {
        nodes: corners.nodes,
        dw_dx,
        dw_dy,
    }
}

#[inline]
pub(crate) fn pressure_gradient_at(
    pressure: &[f64],
    tile_x: f64,
    tile_y: f64,
    params: &GridParams,
) -> (f64, f64) {
    let corners = bilinear_corners(tile_x, tile_y, params);

    let p00 = pressure[corners.nodes[0]];
    let p10 = pressure[corners.nodes[1]];
    let p01 = pressure[corners.nodes[2]];
    let p11 = pressure[corners.nodes[3]];

    bilinear_gradient(corners.fx, corners.fy, p00, p10, p01, p11)
}

#[inline]
pub(crate) fn for_each_movable_pin_with_coeff<F>(
    net_info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    mut f: F,
) where
    F: FnMut(usize, f64, f64, f64),
{
    let factors = demand_factors(net_info, cfg);
    for (k, &(px, py, ci_opt)) in net_info.pins.iter().enumerate() {
        let Some(ci) = ci_opt else { continue };
        let q = if k == 0 {
            factors.io_factor
        } else {
            -factors.io_factor / factors.fanout
        };
        f(ci, px, py, q);
    }
}

#[inline]
pub(crate) fn scatter_net_jacobian_vec(
    net_info: &NetSolveInfo,
    v: &[f64],
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    params: &GridParams,
    out: &mut [f64],
) {
    for_each_movable_pin_with_coeff(net_info, cfg, |ci, px, py, q| {
        let dvx = v[ci];
        let dvy = v[n_cells + ci];
        let stencil = bilinear_jacobian_stencil(px, py, params);
        for j in 0..4 {
            out[stencil.nodes[j]] += q * (stencil.dw_dx[j] * dvx + stencil.dw_dy[j] * dvy);
        }
    });
}

#[inline]
pub(crate) fn gather_net_jacobian_transpose(
    net_info: &NetSolveInfo,
    z: &[f64],
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    params: &GridParams,
    out: &mut [f64],
) {
    for_each_movable_pin_with_coeff(net_info, cfg, |ci, px, py, q| {
        let stencil = bilinear_jacobian_stencil(px, py, params);
        for j in 0..4 {
            out[ci] += q * stencil.dw_dx[j] * z[stencil.nodes[j]];
            out[n_cells + ci] += q * stencil.dw_dy[j] * z[stencil.nodes[j]];
        }
    });
}

/// Collect nets that need per-net Kirchhoff solves.
///
/// Returns all nets with at least one movable pin (those are the cells
/// we need to move). ALL pins inject demand — the flow-vector extraction
/// handles self-cancellation via Gauss's law. Sorted largest-first.
pub fn collect_nets_for_solve(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> Vec<NetSolveInfo> {
    let net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();

    // Parallel: build NetSolveInfo for each net independently
    let mut nets: Vec<NetSolveInfo> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            let net = ctx.design.net(net_id);
            let Some(dp) = net.driver() else {
                return None;
            };
            if net.num_users() == 0 {
                return None;
            }

            let mut pins = Vec::new();
            let mut pin_is_fixed = Vec::new();
            let mut has_fixed = false;
            let mut has_movable = false;

            // Driver.
            let dc = ctx.design.cell(dp.cell);
            let (dx, dy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
            let ci = cell_to_idx.get(&dp.cell).copied();
            let is_fixed = dc.bel_strength.is_locked();
            if is_fixed {
                has_fixed = true;
            }
            if ci.is_some() {
                has_movable = true;
            }
            pins.push((dx, dy, ci));
            pin_is_fixed.push(is_fixed);

            // Sinks.
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let uc = ctx.design.cell(user.cell);
                let (ux, uy) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
                let ci = cell_to_idx.get(&user.cell).copied();
                let is_fixed = uc.bel_strength.is_locked();
                if is_fixed {
                    has_fixed = true;
                }
                if ci.is_some() {
                    has_movable = true;
                }
                pins.push((ux, uy, ci));
                pin_is_fixed.push(is_fixed);
            }

            // Need at least one movable pin and 2+ total pins.
            if has_movable && pins.len() >= 2 {
                Some(NetSolveInfo {
                    pins,
                    pin_is_fixed,
                    has_fixed_pin: has_fixed,
                })
            } else {
                None
            }
        })
        .collect();

    // Largest nets first (sequential sort is necessary for determinism)
    nets.sort_by(|a, b| b.pins.len().cmp(&a.pins.len()));
    nets
}

/// Build RHS vector for a single net — all pins inject demand.
///
/// Driver injects +1, each sink extracts −1/fanout. Both fixed and
/// movable pins contribute. The flow-vector force extraction handles
/// self-cancellation via Gauss's law on the discrete grid.
#[allow(dead_code)]
pub fn build_net_rhs(
    info: &NetSolveInfo,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let n_nodes = network.num_nodes();
    let mut rhs = vec![0.0; n_nodes];
    let params = GridParams::from_network(network);
    let factors = demand_factors(info, cfg);

    // Driver (first pin): inject +demand.
    let (dx, dy, _) = info.pins[0];
    let dsx = to_subtile_coord(dx, params.resolution);
    let dsy = to_subtile_coord(dy, params.resolution);
    for (gx, gy, bw) in bilinear_weights(dsx, dsy, params.subtile_width, params.subtile_height) {
        let ni = subtile_grid_index(gx, gy, params.tile_width, params.resolution);
        rhs[ni] += factors.io_factor * bw;
    }

    // All sinks: extract −demand/fanout.
    for k in 1..info.pins.len() {
        let (sx, sy, _) = info.pins[k];
        let ssx = to_subtile_coord(sx, params.resolution);
        let ssy = to_subtile_coord(sy, params.resolution);
        for (gx, gy, bw) in bilinear_weights(ssx, ssy, params.subtile_width, params.subtile_height)
        {
            let ni = subtile_grid_index(gx, gy, params.tile_width, params.resolution);
            rhs[ni] += factors.sink_weight * bw;
        }
    }

    rhs
}

/// Build RHS vector with DRIVER demand only (no sink demand).
///
/// This produces the pressure field from the driver alone. Evaluating the
/// gradient of this pressure at sink positions gives the pure transport cost
/// gradient, free from sink self-interaction.
pub fn build_driver_rhs(
    info: &NetSolveInfo,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let n_nodes = network.num_nodes();
    let mut rhs = vec![0.0; n_nodes];
    let params = GridParams::from_network(network);
    let factors = demand_factors(info, cfg);

    // Only the driver injects demand.
    let (dx, dy, _) = info.pins[0];
    let dsx = to_subtile_coord(dx, params.resolution);
    let dsy = to_subtile_coord(dy, params.resolution);
    for (gx, gy, bw) in bilinear_weights(dsx, dsy, params.subtile_width, params.subtile_height) {
        let ni = subtile_grid_index(gx, gy, params.tile_width, params.resolution);
        rhs[ni] += factors.io_factor * bw;
    }

    rhs
}

/// Compute per-net Beckmann energy: E_k = S_k^T · P_k.
///
/// This is the dot product of the net's demand vector with the pressure
/// solution. For a 2-pin net, E_k equals the effective resistance between
/// driver and sink, which grows as ~(1/2π)·ln(d) on a 2D grid.
#[allow(dead_code)]
pub fn net_energy(_info: &NetSolveInfo, rhs: &[f64], pressure: &[f64]) -> f64 {
    rhs.iter().zip(pressure.iter()).map(|(s, p)| s * p).sum()
}

/// Accumulate linearized Beckmann energy gradient from a per-net pressure field.
///
/// The raw Beckmann gradient has magnitude ~1/d (weak for long nets).
/// To linearize: weight by exp(c · E_k) where E_k is the per-net energy
/// and c = 2π. This makes the gradient magnitude distance-independent,
/// equivalent to optimizing exp(c·E) instead of E.
///
/// Only SINK pins receive force (driver self-field corrupts its gradient).
pub fn accumulate_energy_gradient(
    info: &NetSolveInfo,
    pressure: &[f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    energy_weight: f64,
    dx: &mut [f64],
    dy: &mut [f64],
) {
    let params = GridParams::from_network(network);
    let factors = demand_factors(info, cfg);

    for (k, &(px, py, ci_opt)) in info.pins.iter().enumerate() {
        let Some(ci) = ci_opt else { continue };
        if info.pin_is_fixed[k] {
            continue;
        }

        // Skip driver — its bilinear gradient is corrupted by self-field.
        if k == 0 {
            continue;
        }

        let demand_coeff = factors.sink_weight;
        let (grad_x, grad_y) = pressure_gradient_at(pressure, px, py, &params);

        // Scale by N for subtile→tile coordinate transform.
        let scale = params.resolution as f64;

        // Energy gradient: ∂E/∂x = demand_coeff × ∇P × scale.
        // Adam (or other optimizer) applies the negation for descent.
        dx[ci] += demand_coeff * grad_x * scale * energy_weight;
        dy[ci] += demand_coeff * grad_y * scale * energy_weight;
    }
}

// ---------------------------------------------------------------------------
// Bilinear interpolation primitives
// ---------------------------------------------------------------------------

/// Convert tile-coordinate position to subtile-grid coordinate.
///
/// Subtile node (gx, gy) has its center at tile position ((gx+0.5)/N, (gy+0.5)/N).
/// Inverting: subtile_coord = tile_coord * N - 0.5.
#[inline(always)]
pub(crate) fn to_subtile_coord(tile_pos: f64, resolution: usize) -> f64 {
    tile_pos * resolution as f64 - 0.5
}

/// Clamp and compute bilinear interpolation coordinates on the subtile grid.
///
/// Input: position in subtile coordinates.
/// Returns (gx0, gy0, fx, fy) where gx0/gy0 are lower-left subtile indices
/// and fx/fy are fractional offsets in [0, 1].
#[inline(always)]
pub(crate) fn bilinear_cell(sx: f64, sy: f64, sw: usize, sh: usize) -> (usize, usize, f64, f64) {
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
#[inline(always)]
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
#[inline(always)]
fn bilinear_gradient(fx: f64, fy: f64, f00: f64, f10: f64, f01: f64, f11: f64) -> (f64, f64) {
    let gx = (1.0 - fy).mul_add(f10 - f00, fy * (f11 - f01));
    let gy = (1.0 - fx).mul_add(f01 - f00, fx * (f11 - f10));
    (gx, gy)
}

/// Subtile grid index from subtile-grid coordinates (gx, gy).
///
/// The subtile grid is stored in tile-major order:
///   node_index(tx, ty, sx, sy) = (ty*W + tx) * N² + sy*N + sx
///
/// Given global subtile coords (gx, gy):
///   tx = gx / N, sx = gx % N, ty = gy / N, sy = gy % N
///
/// NOTE: If resolution is power-of-2, divisions are strength-reduced by compiler.
#[inline(always)]
pub(crate) fn subtile_grid_index(gx: usize, gy: usize, tile_w: usize, n: usize) -> usize {
    if n.is_power_of_two() {
        let shift = n.trailing_zeros() as usize;
        let mask = n - 1;
        let tx = gx >> shift;
        let sx = gx & mask;
        let ty = gy >> shift;
        let sy = gy & mask;
        return (ty * tile_w + tx) * (n * n) + sy * n + sx;
    }

    let tx = gx / n;
    let sx = gx % n;
    let ty = gy / n;
    let sy = gy % n;
    (ty * tile_w + tx) * (n * n) + sy * n + sx
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
            ((loc.x - network.x0) as f64, (loc.y - network.y0) as f64)
        } else {
            (network.width as f64 / 2.0, network.height as f64 / 2.0)
        }
    }
}

#[derive(Clone, Copy)]
struct TileBBox {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
}

fn net_tile_bbox(
    ctx: &Context,
    net_id: crate::netlist::NetId,
    driver_cell: CellId,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> TileBBox {
    let net = ctx.design.net(net_id);
    let (dxx, dyy) = pin_pos(ctx, driver_cell, cell_to_idx, cell_x, cell_y, network);
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

    TileBBox {
        x0: (min_x.floor() as i32).clamp(0, network.width - 1),
        x1: (max_x.ceil() as i32).clamp(0, network.width - 1),
        y0: (min_y.floor() as i32).clamp(0, network.height - 1),
        y1: (max_y.ceil() as i32).clamp(0, network.height - 1),
    }
}

fn accumulate_pipe_counts_in_bbox(
    local_counts: &mut [u32],
    bbox: TileBBox,
    node_pipes: &[Vec<usize>],
    pipe_from: &[usize],
    tile_width: i32,
    n_per_tile: usize,
) {
    for ty in bbox.y0..=bbox.y1 {
        for tx in bbox.x0..=bbox.x1 {
            let base = ((ty * tile_width + tx) as usize) * n_per_tile;
            for ni in base..(base + n_per_tile) {
                if ni >= node_pipes.len() {
                    continue;
                }
                for &pipe_idx in &node_pipes[ni] {
                    if pipe_from[pipe_idx] == ni {
                        local_counts[pipe_idx] = local_counts[pipe_idx].saturating_add(1);
                    }
                }
            }
        }
    }
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
    let n_pipes = network.pipes.len();
    if n_pipes == 0 {
        return;
    }

    // Parallel: reset net_count on all pipes.
    network
        .pipes
        .par_iter_mut()
        .for_each(|pipe| pipe.net_count = 0);

    let pipe_from: Vec<usize> = network.pipes.iter().map(|p| p.from).collect();
    let node_pipes = &network.node_pipes;
    let w = network.width;
    let n_per_tile = network.nodes_per_tile();

    let net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();

    let counts = net_ids
        .par_iter()
        .fold(
            || vec![0u32; n_pipes],
            |mut local_counts, &net_id| {
                let net = ctx.design.net(net_id);
                let Some(dp) = net.driver() else {
                    return local_counts;
                };
                if net.num_users() == 0 {
                    return local_counts;
                }

                let bbox =
                    net_tile_bbox(ctx, net_id, dp.cell, cell_to_idx, cell_x, cell_y, network);
                accumulate_pipe_counts_in_bbox(
                    &mut local_counts,
                    bbox,
                    node_pipes,
                    &pipe_from,
                    w,
                    n_per_tile,
                );

                local_counts
            },
        )
        .reduce(
            || vec![0u32; n_pipes],
            |mut a, b| {
                for (ai, &bi) in a.iter_mut().zip(b.iter()) {
                    *ai = ai.saturating_add(bi);
                }
                a
            },
        );

    network
        .pipes
        .par_iter_mut()
        .zip(counts.par_iter())
        .for_each(|(pipe, &count)| {
            pipe.net_count = count;
        });
}

/// Compute continuous HPWL from cell positions.
pub fn continuous_hpwl(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> f64 {
    let net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();

    net_ids
        .par_iter()
        .map(|&net_id| {
            let net = ctx.design.net(net_id);
            let Some(dp) = net.driver() else {
                return 0.0;
            };

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
                return 0.0;
            }

            (max_x - min_x) + (max_y - min_y)
        })
        .reduce(|| 0.0, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demand_factors_apply_io_boost_and_sink_weight() {
        let cfg = OptTransPlacerCfg {
            io_boost: 2.5,
            ..Default::default()
        };

        let info = NetSolveInfo {
            pins: vec![
                (0.0, 0.0, Some(0)),
                (1.0, 1.0, Some(1)),
                (2.0, 2.0, Some(2)),
            ],
            pin_is_fixed: vec![false, false, false],
            has_fixed_pin: true,
        };

        let factors = demand_factors(&info, &cfg);
        assert_eq!(factors.io_factor, 2.5);
        assert_eq!(factors.fanout, 2.0);
        assert_eq!(factors.sink_weight, -1.25);
    }

    #[test]
    fn bilinear_jacobian_stencil_matches_expected_corner_weights() {
        let params = GridParams {
            resolution: 4,
            subtile_width: 8,
            subtile_height: 8,
            tile_width: 2,
        };

        // Pick a position whose subtile coords are exactly (2.5, 1.5): fx=0.5, fy=0.5.
        let stencil = bilinear_jacobian_stencil(0.75, 0.5, &params);

        assert_eq!(stencil.dw_dx, [-2.0, 2.0, -2.0, 2.0]);
        assert_eq!(stencil.dw_dy, [-2.0, -2.0, 2.0, 2.0]);
        assert_eq!(stencil.nodes, [6, 7, 10, 11]);
    }
}

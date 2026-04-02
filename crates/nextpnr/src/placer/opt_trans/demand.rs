//! Net demand computation and bilinear interpolation for the Beckmann placer.
//!
//! All interpolation operates on the subtile grid: a regular (W·N) × (H·N) lattice
//! where N is the subtile resolution. Cell positions in tile coordinates are mapped
//! to subtile coordinates for demand injection and pressure gradient extraction.

use rustc_hash::{FxHashMap, FxHashSet};
use std::env;
use crate::context::Context;
use crate::netlist::{CellId, NetId};

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::metrics::congestion::bresenham_line;

use super::config::OptTransPlacerCfg;
use super::network::PipeNetwork;

// ---------------------------------------------------------------------------
// Per-net Kirchhoff solve infrastructure
// ---------------------------------------------------------------------------

/// Information about a single net for per-net Kirchhoff solving.
#[derive(Clone, Debug)]
pub struct NetPinData {
    pub cell_idx: Option<usize>,
    pub is_fixed: bool,
    pub is_driver: bool,
    pub nodes: [usize; 4],
    pub weights: [f64; 4],
    pub dw_dx: [f64; 4],
    pub dw_dy: [f64; 4],
}

pub struct NetSolveInfo {
    /// Original design net id for timing weighting.
    pub net_id: NetId,
    /// Debug label for tracing per-net energy contributions.
    pub debug_name: String,
    /// Pin positions and optional movable cell index.
    /// First element is always the driver.
    pub pins: Vec<(f64, f64, Option<usize>)>,
    /// Whether each pin is fixed (locked). Same indexing as `pins`.
    pub pin_is_fixed: Vec<bool>,
    /// Cached interpolation/Jacobian data for hot-path demand assembly.
    pub pin_data: Vec<NetPinData>,
    /// Unique node indices touched by the cached pin data.
    pub touched_nodes: Vec<usize>,
    /// Whether this net has at least one fixed/IO pin.
    pub has_fixed_pin: bool,
}

impl NetSolveInfo {
    pub fn from_pins(
        net_id: NetId,
        debug_name: impl Into<String>,
        pins: Vec<(f64, f64, Option<usize>)>,
        pin_is_fixed: Vec<bool>,
        has_fixed_pin: bool,
        network: &PipeNetwork,
    ) -> Self {
        let mut info = Self {
            net_id,
            debug_name: debug_name.into(),
            pins,
            pin_is_fixed,
            pin_data: Vec::new(),
            touched_nodes: Vec::new(),
            has_fixed_pin,
        };
        precompute_net_pin_data(&mut info, network);
        info
    }
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
    let base_io_factor = if info.has_fixed_pin {
        cfg.io_boost
    } else {
        1.0
    };
    let fanout = (info.pins.len() - 1) as f64;
    let fanout_norm_exp = cfg.fanout_norm_exp.clamp(0.0, 1.0);
    let fanout_scale = if cfg.fanout_weight_sqrt {
        fanout.sqrt().max(1.0)
    } else {
        1.0
    };
    let io_factor = base_io_factor * fanout_scale;
    let sink_weight = -io_factor / fanout.powf(fanout_norm_exp).max(1.0);
    DemandFactors {
        io_factor,
        fanout,
        sink_weight,
    }
}

#[inline]
fn precompute_net_pin_data(info: &mut NetSolveInfo, network: &PipeNetwork) {
    let params = GridParams::from_network(network);
    info.pin_data.clear();
    info.pin_data.reserve(info.pins.len());
    info.touched_nodes.clear();
    let mut touched = Vec::with_capacity(info.pins.len() * 4);

    for (k, &(px, py, ci_opt)) in info.pins.iter().enumerate() {
        let corners = bilinear_corners(px, py, &params);
        let stencil = bilinear_jacobian_stencil(px, py, &params);
        let weights = [
            (1.0 - corners.fx) * (1.0 - corners.fy),
            corners.fx * (1.0 - corners.fy),
            (1.0 - corners.fx) * corners.fy,
            corners.fx * corners.fy,
        ];
        touched.extend_from_slice(&corners.nodes);
        info.pin_data.push(NetPinData {
            cell_idx: ci_opt,
            is_fixed: info.pin_is_fixed.get(k).copied().unwrap_or(false),
            is_driver: k == 0,
            nodes: corners.nodes,
            weights,
            dw_dx: stencil.dw_dx,
            dw_dy: stencil.dw_dy,
        });
    }

    touched.sort_unstable();
    touched.dedup();
    info.touched_nodes = touched;
}

#[inline]
pub(crate) fn transformed_net_energy(raw_energy: f64) -> (f64, f64) {
    (raw_energy, 1.0)
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

    // For 1D grids, clamp the +1 index to stay in bounds.
    let gx1 = if params.subtile_width <= 1 {
        gx0
    } else {
        gx0 + 1
    };
    let gy1 = if params.subtile_height <= 1 {
        gy0
    } else {
        gy0 + 1
    };

    let nodes = [
        subtile_grid_index(gx0, gy0, params.tile_width, params.resolution),
        subtile_grid_index(gx1, gy0, params.tile_width, params.resolution),
        subtile_grid_index(gx0, gy1, params.tile_width, params.resolution),
        subtile_grid_index(gx1, gy1, params.tile_width, params.resolution),
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
pub(crate) fn for_each_movable_pin_with_coeff<F>(
    net_info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    mut f: F,
) where
    F: FnMut(usize, f64, f64, f64),
{
    let factors = demand_factors(net_info, cfg);
    for pin in &net_info.pin_data {
        let Some(ci) = pin.cell_idx else { continue };
        let q = if pin.is_driver {
            factors.io_factor
        } else {
            factors.sink_weight
        };
        f(ci, 0.0, 0.0, q);
    }
}

#[inline]
pub(crate) fn scatter_net_jacobian_vec(
    net_info: &NetSolveInfo,
    v: &[f64],
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    _params: &GridParams,
    out: &mut [f64],
) {
    let factors = demand_factors(net_info, cfg);
    for pin in &net_info.pin_data {
        let Some(ci) = pin.cell_idx else { continue };
        let dvx = v[ci];
        let dvy = v[n_cells + ci];
        let q = if pin.is_driver {
            factors.io_factor
        } else {
            factors.sink_weight
        };
        for j in 0..4 {
            out[pin.nodes[j]] += q * (pin.dw_dx[j] * dvx + pin.dw_dy[j] * dvy);
        }
    }
}

#[inline]
pub(crate) fn gather_net_jacobian_transpose(
    net_info: &NetSolveInfo,
    z: &[f64],
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    _params: &GridParams,
    out: &mut [f64],
) {
    let factors = demand_factors(net_info, cfg);
    for pin in &net_info.pin_data {
        let Some(ci) = pin.cell_idx else { continue };
        let q = if pin.is_driver {
            factors.io_factor
        } else {
            factors.sink_weight
        };
        for j in 0..4 {
            out[ci] += q * pin.dw_dx[j] * z[pin.nodes[j]];
            out[n_cells + ci] += q * pin.dw_dy[j] * z[pin.nodes[j]];
        }
    }
}

/// Collect nets that need per-net Kirchhoff solves.
///
/// Returns all nets with at least one movable pin (those are the cells
/// we need to move). ALL pins inject demand — the flow-vector extraction
/// handles self-cancellation via Gauss's law. Sorted largest-first.
pub fn collect_nets_for_solve(
    ctx: &Context,
    net_ids: &[NetId],
    include_debug_names: bool,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> Vec<NetSolveInfo> {
    let coord_scale = network.coarsen as f64;
    let exclude_globals = env::var("NPNR_OT_EXCLUDE_GLOBALS").ok().as_deref() == Some("1");

    // Parallel: build NetSolveInfo for each net independently
    let mut nets: Vec<NetSolveInfo> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            let net = ctx.design.net(net_id);
            let net_name = ctx.name_of(net.name);
            if exclude_globals {
                let lower = net_name.to_ascii_lowercase();
                let is_const = net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET";
                let is_clockish = lower.contains("clk") || lower.contains("clock");
                if is_const || is_clockish {
                    return None;
                }
            }
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
            let (dx, dy) = pin_pos(
                ctx,
                dp.cell,
                cell_to_idx,
                cell_x,
                cell_y,
                network,
                coord_scale,
            );
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
                let (ux, uy) = pin_pos(
                    ctx,
                    user.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
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
                let mut info = NetSolveInfo {
                    net_id,
                    debug_name: if include_debug_names {
                        ctx.name_of(net.name).to_string()
                    } else {
                        String::new()
                    },
                    pins,
                    pin_is_fixed,
                    pin_data: Vec::new(),
                    touched_nodes: Vec::new(),
                    has_fixed_pin: has_fixed,
                };
                precompute_net_pin_data(&mut info, network);
                Some(info)
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
    build_net_rhs_into(info, network, cfg, &mut rhs);
    rhs
}

#[inline]
pub fn build_net_rhs_into(
    info: &NetSolveInfo,
    _network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    rhs: &mut [f64],
) {
    rhs.fill(0.0);
    accumulate_net_rhs_into(info, cfg, rhs);
}

#[inline]
pub fn build_net_rhs_reuse_into(
    info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    rhs: &mut [f64],
    prev_touched: &mut Vec<usize>,
) {
    for &node in prev_touched.iter() {
        rhs[node] = 0.0;
    }
    for &node in &info.touched_nodes {
        rhs[node] = 0.0;
    }
    accumulate_net_rhs_into(info, cfg, rhs);
    prev_touched.clear();
    prev_touched.extend_from_slice(&info.touched_nodes);
}

#[inline]
pub fn accumulate_net_rhs_into(
    info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    rhs: &mut [f64],
) {
    let factors = demand_factors(info, cfg);
    for pin in &info.pin_data {
        let q = if pin.is_driver {
            factors.io_factor
        } else {
            factors.sink_weight
        };
        for j in 0..4 {
            rhs[pin.nodes[j]] += q * pin.weights[j];
        }
    }
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

/// Physical network dissipation energy for one net on a fixed resistive graph.
///
/// This is the standard quadratic dissipation law:
///   E = 1/2 * b^T phi
/// where `b` is the balanced source/sink injection vector and `phi` is the
/// solved potential field for that net on the current resistive network.
#[inline]
pub fn physical_net_energy(info: &NetSolveInfo, rhs: &[f64], pressure: &[f64]) -> f64 {
    0.5 * net_energy(info, rhs, pressure)
}

/// Accumulate the gradient of the physical network dissipation energy.
///
/// For fixed resistance during a single optimization step:
///   E(x) = 1/2 * b(x)^T L^{-1} b(x)
/// and the gradient is:
///   dE/dx = J(x)^T phi
/// where J = db/dx and phi solves L phi = b.
pub fn accumulate_physical_energy_gradient(
    info: &NetSolveInfo,
    pressure: &[f64],
    _network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    energy_weight: f64,
    dx: &mut [f64],
    dy: &mut [f64],
) {
    let factors = demand_factors(info, cfg);
    for pin in &info.pin_data {
        let Some(ci) = pin.cell_idx else { continue };
        if pin.is_fixed {
            continue;
        }
        let q = if pin.is_driver {
            factors.io_factor
        } else {
            factors.sink_weight
        };
        for j in 0..4 {
            let p = pressure[pin.nodes[j]];
            dx[ci] += energy_weight * q * pin.dw_dx[j] * p;
            dy[ci] += energy_weight * q * pin.dw_dy[j] * p;
        }
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
    // Handle degenerate 1D cases: when a dimension has only 1 node,
    // clamp to index 0 with fraction 0 so interpolation degenerates to linear/point.
    let (gx0, fx) = if sw <= 1 {
        (0usize, 0.0f64)
    } else {
        let max_x = (sw - 1) as f64;
        let sx = sx.clamp(0.0, max_x);
        let gx0 = (sx.floor() as usize).min(sw - 2);
        (gx0, sx - gx0 as f64)
    };

    let (gy0, fy) = if sh <= 1 {
        (0usize, 0.0f64)
    } else {
        let max_y = (sh - 1) as f64;
        let sy = sy.clamp(0.0, max_y);
        let gy0 = (sy.floor() as usize).min(sh - 2);
        (gy0, sy - gy0 as f64)
    };

    (gx0, gy0, fx, fy)
}

/// Bilinear weights on subtile grid: maps continuous position to 4 surrounding
/// subtile nodes with weights. Returns [(gx, gy, weight); 4].
#[inline(always)]
fn bilinear_weights(sx: f64, sy: f64, sw: usize, sh: usize) -> [(usize, usize, f64); 4] {
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);
    // For 1D grids, clamp the +1 index to stay in bounds.
    let gx1 = if sw <= 1 { gx0 } else { gx0 + 1 };
    let gy1 = if sh <= 1 { gy0 } else { gy0 + 1 };
    [
        (gx0, gy0, (1.0 - fx) * (1.0 - fy)),
        (gx1, gy0, fx * (1.0 - fy)),
        (gx0, gy1, (1.0 - fx) * fy),
        (gx1, gy1, fx * fy),
    ]
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
    coord_scale: f64,
) -> (f64, f64) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        (cell_x[idx], cell_y[idx])
    } else {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            (
                (loc.x - network.x0) as f64 / coord_scale,
                (loc.y - network.y0) as f64 / coord_scale,
            )
        } else {
            (
                network.width as f64 / (2.0 * coord_scale.max(1.0)),
                network.height as f64 / (2.0 * coord_scale.max(1.0)),
            )
        }
    }
}

#[inline]
fn clamp_net_tile(x: f64, y: f64, network: &PipeNetwork) -> (i32, i32) {
    (
        (x.round() as i32).clamp(0, network.width - 1),
        (y.round() as i32).clamp(0, network.height - 1),
    )
}

#[inline]
fn pipe_key(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

#[inline]
fn mark_pipe_step(
    from: (i32, i32),
    to: (i32, i32),
    pipe_lookup: &FxHashMap<u64, usize>,
    network: &PipeNetwork,
    local_hits: &mut FxHashSet<usize>,
) {
    let res = network.resolution;
    let mid = res / 2;
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;

    if dx != 0 {
        let from_sx = if dx > 0 { res - 1 } else { 0 };
        let to_sx = if dx > 0 { 0 } else { res - 1 };
        for sy in [mid, 0, res.saturating_sub(1)] {
            let from_node = network.node_index(from.0, from.1, from_sx, sy.min(res - 1));
            let to_node = network.node_index(to.0, to.1, to_sx, sy.min(res - 1));
            if let Some(&pipe_idx) = pipe_lookup.get(&pipe_key(from_node, to_node)) {
                local_hits.insert(pipe_idx);
            }
        }
    } else if dy != 0 {
        let from_sy = if dy > 0 { res - 1 } else { 0 };
        let to_sy = if dy > 0 { 0 } else { res - 1 };
        for sx in [mid, 0, res.saturating_sub(1)] {
            let from_node = network.node_index(from.0, from.1, sx.min(res - 1), from_sy);
            let to_node = network.node_index(to.0, to.1, sx.min(res - 1), to_sy);
            if let Some(&pipe_idx) = pipe_lookup.get(&pipe_key(from_node, to_node)) {
                local_hits.insert(pipe_idx);
            }
        }
    }
}

fn mark_corridor_pipes(
    start: (i32, i32),
    goal: (i32, i32),
    pipe_lookup: &FxHashMap<u64, usize>,
    network: &PipeNetwork,
    local_hits: &mut FxHashSet<usize>,
) {
    let line = bresenham_line(start.0, start.1, goal.0, goal.1);
    for window in line.windows(2) {
        let from = window[0];
        let to = window[1];
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;

        if dx != 0 {
            let mid = (from.0 + dx.signum(), from.1);
            mark_pipe_step(from, mid, pipe_lookup, network, local_hits);
            if dy != 0 {
                mark_pipe_step(mid, to, pipe_lookup, network, local_hits);
            }
        } else if dy != 0 {
            mark_pipe_step(from, to, pipe_lookup, network, local_hits);
        }
    }
}

/// Update net_count on pipes based on current cell positions.
///
/// For each net, traces a thin driver-to-sink corridor over the coarse routing
/// graph and increments `net_count` for the pipes the net is likely to use.
pub fn update_net_counts(
    net_ids: &[NetId],
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
    let coord_scale = network.coarsen as f64;

    // Parallel: reset net_count on all pipes.
    network
        .pipes
        .par_iter_mut()
        .for_each(|pipe| pipe.net_count = 0);

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

                let driver_pos = pin_pos(
                    ctx,
                    dp.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
                let driver_tile = clamp_net_tile(driver_pos.0, driver_pos.1, network);
                let mut pipe_hits = FxHashSet::default();

                for user in net.users() {
                    if !user.is_valid() {
                        continue;
                    }
                    let sink_pos = pin_pos(
                        ctx,
                        user.cell,
                        cell_to_idx,
                        cell_x,
                        cell_y,
                        network,
                        coord_scale,
                    );
                    let sink_tile = clamp_net_tile(sink_pos.0, sink_pos.1, network);
                    mark_corridor_pipes(
                        driver_tile,
                        sink_tile,
                        &network.pipe_lookup,
                        network,
                        &mut pipe_hits,
                    );
                }

                for pipe_idx in pipe_hits {
                    local_counts[pipe_idx] = local_counts[pipe_idx].saturating_add(1);
                }

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
    let coord_scale = 1.0;
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

            let (dxx, dyy) = pin_pos(
                ctx,
                dp.cell,
                cell_to_idx,
                cell_x,
                cell_y,
                network,
                coord_scale,
            );
            let (mut min_x, mut max_x) = (dxx, dxx);
            let (mut min_y, mut max_y) = (dyy, dyy);

            let mut has_valid_sink = false;
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                has_valid_sink = true;
                let (x, y) = pin_pos(
                    ctx,
                    user.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
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

/// Compute a continuous analogue of the line estimate from current floating-point positions.
pub fn continuous_line_estimate(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> f64 {
    let coord_scale = 1.0;
    let width = ctx.chipdb().width();
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
            if net.num_users() == 0 {
                return 0.0;
            }

            let (dx, dy) = pin_pos(
                ctx,
                dp.cell,
                cell_to_idx,
                cell_x,
                cell_y,
                network,
                coord_scale,
            );
            let driver = (dx.round() as i32, dy.round() as i32);
            let mut edges = FxHashSet::default();

            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let (sx, sy) = pin_pos(
                    ctx,
                    user.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
                let sink = (sx.round() as i32, sy.round() as i32);
                let points = bresenham_line(driver.0, driver.1, sink.0, sink.1);
                for pair in points.windows(2) {
                    let (x1, y1) = pair[0];
                    let (x2, y2) = pair[1];
                    let a = (y1 * width + x1) as u32;
                    let b = (y2 * width + x2) as u32;
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    edges.insert(((lo as u64) << 32) | (hi as u64));
                }
            }

            edges.len() as f64
        })
        .reduce(|| 0.0, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Node, PipeNetwork};
    use crate::solver::optimizer::LbfgsOptimizer;

    fn simple_network() -> PipeNetwork {
        PipeNetwork {
            nodes: vec![
                Node {
                    tile_x: 0,
                    tile_y: 0,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 1,
                    tile_y: 0,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 0,
                    tile_y: 1,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 1,
                    tile_y: 1,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
            ],
            pipes: Vec::new(),
            node_pipes: vec![Vec::new(); 4],
            width: 2,
            height: 2,
            x0: 0,
            y0: 0,
            resolution: 1,
            zero_bel_tiles: 0,
            coarsen: 1,
        }
    }

    fn one_sink_info(x: f64, y: f64) -> NetSolveInfo {
        NetSolveInfo::from_pins(
            NetId::from_raw(0),
            "one_sink",
            vec![(0.0, 0.0, None), (x, y, Some(0))],
            vec![true, false],
            false,
            &simple_network(),
        )
    }

    fn scalar_and_grad(
        network: &PipeNetwork,
        pressure: &[f64],
        cfg: &OptTransPlacerCfg,
        x: f64,
        y: f64,
    ) -> (f64, [f64; 2]) {
        let info = one_sink_info(x, y);
        let rhs = build_net_rhs(&info, network, cfg);
        let energy = physical_net_energy(&info, &rhs, pressure);
        let mut grad_x = [0.0];
        let mut grad_y = [0.0];
        accumulate_physical_energy_gradient(
            &info,
            pressure,
            network,
            cfg,
            1.0,
            &mut grad_x,
            &mut grad_y,
        );
        (energy, [grad_x[0], grad_y[0]])
    }

    #[test]
    fn demand_factors_apply_io_boost_and_sink_weight() {
        let cfg = OptTransPlacerCfg {
            io_boost: 2.5,
            ..Default::default()
        };

        let info = NetSolveInfo::from_pins(
            NetId::from_raw(1),
            "factors",
            vec![
                (0.0, 0.0, Some(0)),
                (1.0, 1.0, Some(1)),
                (2.0, 2.0, Some(2)),
            ],
            vec![false, false, false],
            true,
            &simple_network(),
        );

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

    #[test]
    fn movable_driver_gets_equal_and_opposite_reaction_force() {
        let cfg = OptTransPlacerCfg::default();
        let network = simple_network();
        let pressure = vec![0.0, 1.0, 2.0, 4.0];
        let info = NetSolveInfo::from_pins(
            NetId::from_raw(2),
            "movable_driver",
            vec![(0.0, 0.0, Some(0)), (0.35, 0.60, Some(1))],
            vec![false, false],
            false,
            &network,
        );

        let mut grad_x = [0.0, 0.0];
        let mut grad_y = [0.0, 0.0];
        accumulate_physical_energy_gradient(
            &info,
            &pressure,
            &network,
            &cfg,
            1.0,
            &mut grad_x,
            &mut grad_y,
        );

        assert!((grad_x[0] + grad_x[1]).abs() < 1e-9);
        assert!((grad_y[0] + grad_y[1]).abs() < 1e-9);
    }

    #[test]
    fn physical_energy_gradient_matches_finite_difference() {
        let cfg = OptTransPlacerCfg::default();
        let network = simple_network();
        let pressure = vec![0.0, 1.0, 2.0, 4.0];
        let x = 0.35;
        let y = 0.60;
        let eps = 1e-6;

        let (_, grad) = scalar_and_grad(&network, &pressure, &cfg, x, y);

        let e_xp = scalar_and_grad(&network, &pressure, &cfg, x + eps, y).0;
        let e_xm = scalar_and_grad(&network, &pressure, &cfg, x - eps, y).0;
        let e_yp = scalar_and_grad(&network, &pressure, &cfg, x, y + eps).0;
        let e_ym = scalar_and_grad(&network, &pressure, &cfg, x, y - eps).0;
        let num_dx = (e_xp - e_xm) / (2.0 * eps);
        let num_dy = (e_yp - e_ym) / (2.0 * eps);

        assert!(
            (grad[0] - num_dx).abs() + (grad[1] - num_dy).abs() < 1e-5,
            "analytic=({:.6},{:.6}) numeric=({:.6},{:.6})",
            grad[0],
            grad[1],
            num_dx,
            num_dy
        );
    }

    #[test]
    fn lbfgs_descends_simple_ot_energy_quickly() {
        let cfg = OptTransPlacerCfg::default();
        let network = simple_network();
        let pressure = vec![0.0, 1.0, 2.0, 4.0];
        let mut opt = LbfgsOptimizer::new(5);
        let mut pos = vec![0.8, 0.8];
        let mut energy_trace = Vec::new();

        for _ in 0..8 {
            let (energy, grad) = scalar_and_grad(&network, &pressure, &cfg, pos[0], pos[1]);
            energy_trace.push(energy);

            let step = opt.step(&pos, &grad);
            let directional_derivative = grad[0] * step[0] + grad[1] * step[1];
            assert!(
                directional_derivative < 0.0,
                "step is not a descent direction: grad={grad:?} step={step:?}"
            );

            let mut t = 1.0;
            let mut accepted = false;
            while t >= 1e-6 {
                let trial = vec![
                    (pos[0] + t * step[0]).clamp(0.0, 1.0),
                    (pos[1] + t * step[1]).clamp(0.0, 1.0),
                ];
                let (trial_energy, _) =
                    scalar_and_grad(&network, &pressure, &cfg, trial[0], trial[1]);
                if trial_energy < energy {
                    pos = trial;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }

            assert!(accepted, "line search failed to find a lower-energy trial step");
        }

        let (final_energy, _) = scalar_and_grad(&network, &pressure, &cfg, pos[0], pos[1]);
        assert!(
            final_energy < energy_trace[0] * 0.2,
            "final energy {final_energy} did not drop enough from {}",
            energy_trace[0]
        );
        assert!(pos[0] < 0.1, "x did not converge near low-pressure corner: {}", pos[0]);
        assert!(pos[1] < 0.1, "y did not converge near low-pressure corner: {}", pos[1]);
    }
}

//! Net demand computation and bilinear interpolation for the Beckmann placer.
//!
//! Interpolation operates on the tile or coarsened tile grid used by the
//! path solver. Cell positions are mapped to four neighboring network
//! nodes, and the distance-field interpolation Jacobian provides the gradient.

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use rustc_hash::{FxHashMap, FxHashSet};
use std::env;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::metrics::congestion::bresenham_line;

use super::config::OptTransPlacerCfg;
use super::network::PipeNetwork;

// ---------------------------------------------------------------------------
// Per-net path-solve infrastructure
// ---------------------------------------------------------------------------

/// Information about a single net for per-net path solving.
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
    pub grid_width: usize,
    pub grid_height: usize,
    pub tile_width: usize,
}

impl GridParams {
    #[inline]
    pub(crate) fn from_network(network: &PipeNetwork) -> Self {
        Self {
            grid_width: network.width as usize,
            grid_height: network.height as usize,
            tile_width: network.width as usize,
        }
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct BilinearJacobianStencil {
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
    let sx = tile_x - 0.5;
    let sy = tile_y - 0.5;
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, params.grid_width, params.grid_height);

    // For 1D grids, clamp the +1 index to stay in bounds.
    let gx1 = if params.grid_width <= 1 { gx0 } else { gx0 + 1 };
    let gy1 = if params.grid_height <= 1 {
        gy0
    } else {
        gy0 + 1
    };

    let nodes = [
        grid_index(gx0, gy0, params.tile_width),
        grid_index(gx1, gy0, params.tile_width),
        grid_index(gx0, gy1, params.tile_width),
        grid_index(gx1, gy1, params.tile_width),
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

    let dw_dx = [
        -(1.0 - corners.fy),
        (1.0 - corners.fy),
        -corners.fy,
        corners.fy,
    ];
    let dw_dy = [
        -(1.0 - corners.fx),
        -corners.fx,
        (1.0 - corners.fx),
        corners.fx,
    ];

    BilinearJacobianStencil { dw_dx, dw_dy }
}

/// Collect nets that need per-net path solves.
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
            let (dx, dy, ci) = pin_pos(
                ctx,
                dp.cell,
                cell_to_idx,
                cell_x,
                cell_y,
                network,
                coord_scale,
            );
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
                let (ux, uy, ci) = pin_pos(
                    ctx,
                    user.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
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

#[inline]
pub fn net_timing_weight(info: &NetSolveInfo, cfg: &OptTransPlacerCfg) -> f64 {
    let crit = cfg
        .timing_criticality
        .get(&info.net_id)
        .copied()
        .unwrap_or(0.0) as f64;
    let timing = 1.0 + cfg.timing_weight.max(0.0) * crit.clamp(0.0, 1.0);
    let locked = if info.has_fixed_pin {
        cfg.locked_pin_weight.max(0.0)
    } else {
        1.0
    };
    timing * locked
}

// ---------------------------------------------------------------------------
// Bilinear interpolation primitives
// ---------------------------------------------------------------------------

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

/// Grid index from coordinates (gx, gy).
///
/// One node per tile: index = gy * tile_w + gx.
#[inline(always)]
pub(crate) fn grid_index(gx: usize, gy: usize, tile_w: usize) -> usize {
    gy * tile_w + gx
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pin position in network coords plus the index of the cell whose motion
/// drives this pin. Cluster children move rigidly with their root, so their
/// position is `root_pos + (constr_x, constr_y)` and the driving index is
/// the root's. Returns `None` for the third value only when the pin belongs
/// to a non-movable already-placed cell.
fn pin_pos(
    ctx: &Context,
    cell_id: CellId,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    coord_scale: f64,
) -> (f64, f64, Option<usize>) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        return (cell_x[idx], cell_y[idx], Some(idx));
    }

    let cell = ctx.design.cell(cell_id);

    if let Some(root_id) = cell.cluster {
        if root_id != cell_id {
            if let Some(&root_idx) = cell_to_idx.get(&root_id) {
                return (
                    cell_x[root_idx] + cell.constr_x as f64 / coord_scale,
                    cell_y[root_idx] + cell.constr_y as f64 / coord_scale,
                    Some(root_idx),
                );
            }
            let root_cell = ctx.design.cell(root_id);
            if let Some(root_bel) = root_cell.bel {
                let loc = ctx.bel(root_bel).loc();
                return (
                    (loc.x + cell.constr_x - network.x0) as f64 / coord_scale,
                    (loc.y + cell.constr_y - network.y0) as f64 / coord_scale,
                    None,
                );
            }
        }
    }

    if let Some(bel) = cell.bel {
        let loc = ctx.bel(bel).loc();
        return (
            (loc.x - network.x0) as f64 / coord_scale,
            (loc.y - network.y0) as f64 / coord_scale,
            None,
        );
    }

    panic!(
        "demand::pin_pos: cell {} is neither movable, a placed cluster child, nor bound to a BEL",
        ctx.name_of(cell.name),
    );
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
    let skip_constants = env::var("NPNR_OT_INCLUDE_CONSTANTS").ok().as_deref() != Some("1");
    let skip_clocks = env::var("NPNR_OT_EXCLUDE_CLOCKS").ok().as_deref() == Some("1")
        || env::var("NPNR_OT_EXCLUDE_GLOBALS").ok().as_deref() == Some("1");
    let net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();

    net_ids
        .par_iter()
        .map(|&net_id| {
            let net = ctx.design.net(net_id);
            let net_name = ctx.name_of(net.name);
            let is_const = net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET";
            if skip_constants && is_const {
                return 0.0;
            }
            if skip_clocks {
                let lower = net_name.to_ascii_lowercase();
                if lower.contains("clk") || lower.contains("clock") {
                    return 0.0;
                }
            }
            let Some(dp) = net.driver() else {
                return 0.0;
            };
            if net.num_users() == 0 {
                return 0.0;
            }

            let (dx, dy, _) = pin_pos(
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
                let (sx, sy, _) = pin_pos(
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

    #[test]
    fn bilinear_jacobian_stencil_matches_expected_corner_weights() {
        let params = GridParams {
            grid_width: 4,
            grid_height: 4,
            tile_width: 4,
        };

        // Position (1.5, 1.5): grid coord = (1.0, 1.0) → gx0=1, gy0=1, fx=0.0, fy=0.0.
        let stencil = bilinear_jacobian_stencil(1.5, 1.5, &params);

        // At fx=0, fy=0: dw_dx = [-1, 1, 0, 0], dw_dy = [-1, 0, 1, 0]
        assert_eq!(stencil.dw_dx, [-1.0, 1.0, 0.0, 0.0]);
        assert_eq!(stencil.dw_dy, [-1.0, 0.0, 1.0, 0.0]);
    }
}

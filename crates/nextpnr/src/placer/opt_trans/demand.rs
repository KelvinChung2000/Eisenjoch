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
    let mut nets = Vec::new();

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };
        if net.num_users() == 0 { continue; }

        let mut pins = Vec::new();
        let mut pin_is_fixed = Vec::new();
        let mut has_fixed = false;
        let mut has_movable = false;

        // Driver.
        let dc = ctx.design.cell(dp.cell);
        let (dx, dy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let ci = cell_to_idx.get(&dp.cell).copied();
        let is_fixed = dc.bel_strength.is_locked();
        if is_fixed { has_fixed = true; }
        if ci.is_some() { has_movable = true; }
        pins.push((dx, dy, ci));
        pin_is_fixed.push(is_fixed);

        // Sinks.
        for user in net.users() {
            if !user.is_valid() { continue; }
            let uc = ctx.design.cell(user.cell);
            let (ux, uy) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
            let ci = cell_to_idx.get(&user.cell).copied();
            let is_fixed = uc.bel_strength.is_locked();
            if is_fixed { has_fixed = true; }
            if ci.is_some() { has_movable = true; }
            pins.push((ux, uy, ci));
            pin_is_fixed.push(is_fixed);
        }

        // Need at least one fixed pin (for directional signal) and one movable pin.
        if has_fixed && has_movable && pins.len() >= 2 {
            nets.push(NetSolveInfo { pins, pin_is_fixed, has_fixed_pin: has_fixed });
        }
    }

    // Largest nets first.
    nets.sort_by(|a, b| b.pins.len().cmp(&a.pins.len()));
    nets
}

/// Build RHS vector for a single net — all pins inject demand.
///
/// Driver injects +1, each sink extracts −1/fanout. Both fixed and
/// movable pins contribute. The flow-vector force extraction handles
/// self-cancellation via Gauss's law on the discrete grid.
pub fn build_net_rhs(
    info: &NetSolveInfo,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let n_nodes = network.num_nodes();
    let mut rhs = vec![0.0; n_nodes];
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    let io_factor = if info.has_fixed_pin { cfg.io_boost } else { 1.0 };
    let fanout = (info.pins.len() - 1) as f64;

    // Driver (first pin): inject +demand.
    let (dx, dy, _) = info.pins[0];
    let dsx = to_subtile_coord(dx, n);
    let dsy = to_subtile_coord(dy, n);
    for (gx, gy, bw) in bilinear_weights(dsx, dsy, sw, sh) {
        let ni = subtile_grid_index(gx, gy, tile_w, n);
        rhs[ni] += io_factor * bw;
    }

    // All sinks: extract −demand/fanout.
    let sink_weight = io_factor / fanout;
    for k in 1..info.pins.len() {
        let (sx, sy, _) = info.pins[k];
        let ssx = to_subtile_coord(sx, n);
        let ssy = to_subtile_coord(sy, n);
        for (gx, gy, bw) in bilinear_weights(ssx, ssy, sw, sh) {
            let ni = subtile_grid_index(gx, gy, tile_w, n);
            rhs[ni] -= sink_weight * bw;
        }
    }

    rhs
}

/// Accumulate dp/dist force with energy-gradient role sign.
///
/// For each movable cell, computes force from pressure differences to
/// connected pins. The role sign ensures energy-minimizing direction:
///   Sink:   F = +dp / (R_local × dist) × direction  (toward driver)
///   Driver: F = −dp / (R_local × dist) × direction  (toward sinks)
pub fn accumulate_dp_dist_force(
    info: &NetSolveInfo,
    pressure: &[f64],
    network: &PipeNetwork,
    dx: &mut [f64],
    dy: &mut [f64],
) {
    let pin_pressures: Vec<f64> = info.pins.iter()
        .map(|&(px, py, _)| pressure_from_vec(px, py, pressure, network))
        .collect();

    for (i, &(xi, yi, ci_opt)) in info.pins.iter().enumerate() {
        let Some(ci) = ci_opt else { continue };
        if info.pin_is_fixed[i] { continue; }

        // Role sign: sinks (+1) move toward high P, drivers (-1) toward low P.
        let role_sign: f64 = if i == 0 { -1.0 } else { 1.0 };
        let p_self = pin_pressures[i];
        let r_local = congestion_at(xi, yi, network);

        for (j, &(xj, yj, _)) in info.pins.iter().enumerate() {
            if i == j { continue; }
            let dir_x = xj - xi;
            let dir_y = yj - yi;
            let dist = (dir_x * dir_x + dir_y * dir_y).sqrt();
            if dist < 0.1 { continue; }

            let dp = pin_pressures[j] - p_self;
            let weight = role_sign * dp / (r_local * dist);
            dx[ci] += weight * dir_x / dist;
            dy[ci] += weight * dir_y / dist;
        }
    }
}

/// Accumulate flow-vector force from a per-net pressure field.
///
/// For each movable cell, computes the net flow vector at its position
/// from the per-net pressure solution: F = Σ_pipes ΔP × g × direction.
///
/// This is "pressure × area = force" (Gauss's law). The self-field from
/// the cell's own demand is symmetric and cancels in the vector sum,
/// leaving only the external field (force from other pins' demand).
///
/// Role sign: sinks move against the flow (toward driver),
///            drivers move with the flow (toward sinks).
/// This comes from the energy gradient: dE/dx = ±∇P depending on role.
pub fn accumulate_flow_force(
    info: &NetSolveInfo,
    pressure: &[f64],
    network: &PipeNetwork,
    dx: &mut [f64],
    dy: &mut [f64],
) {
    let res = network.resolution;

    for (k, &(px, py, ci_opt)) in info.pins.iter().enumerate() {
        let Some(ci) = ci_opt else { continue };
        if info.pin_is_fixed[k] { continue; }

        // Role sign: driver (k==0) moves WITH flow, sinks move AGAINST.
        let role_sign: f64 = if k == 0 { 1.0 } else { -1.0 };

        // Find the nearest subtile node to the cell position.
        let sx = to_subtile_coord(px, res);
        let sy = to_subtile_coord(py, res);
        let sw = network.subtile_width();
        let sh = network.subtile_height();
        let tile_w = network.width as usize;

        let gx = (sx.round() as usize).min(sw - 1);
        let gy = (sy.round() as usize).min(sh - 1);
        let node_i = subtile_grid_index(gx, gy, tile_w, res);

        // Cell's demand at this specific node (bilinear weight at nearest node).
        let base_demand: f64 = if k == 0 { 1.0 } else { -1.0 / (info.pins.len() - 1) as f64 };
        // Bilinear weight at the nearest node: the fraction of demand at node_i.
        let bw = bilinear_weights(sx, sy, sw, sh);
        let self_weight = bw.iter()
            .filter(|&&(bgx, bgy, _)| subtile_grid_index(bgx, bgy, tile_w, res) == node_i)
            .map(|&(_, _, w)| w)
            .sum::<f64>();
        let self_demand = base_demand * self_weight;

        // Sum of effective conductances at this node (matches Laplacian diagonal).
        let mut g_total = 0.0f64;
        if node_i < network.node_pipes.len() {
            for &pipe_idx in &network.node_pipes[node_i] {
                g_total += network.pipes[pipe_idx].eff_conductance;
            }
        }
        // Add regularization (ε·I from CG setup: ε = 1e-4 × avg_diag).
        g_total += 1e-4 * g_total.max(1.0);

        // Sum flow vectors with self-field subtraction.
        let mut fx = 0.0f64;
        let mut fy = 0.0f64;

        if node_i < network.node_pipes.len() {
            for &pipe_idx in &network.node_pipes[node_i] {
                let pipe = &network.pipes[pipe_idx];
                let other = if pipe.from == node_i { pipe.to } else { pipe.from };
                let node_self = &network.nodes[node_i];
                let node_other = &network.nodes[other];

                // Total flow on this pipe from per-net pressure (using same g as Laplacian).
                let p_from = pressure[pipe.from];
                let p_to = pressure[pipe.to];
                let g_pipe = pipe.eff_conductance;
                let flow = (p_from - p_to) * g_pipe;

                // Self-field subtraction: cell's demand distributes ∝ conductance.
                let self_flow = self_demand * g_pipe / g_total;

                // Outward flow from node_i (positive = away from node_i).
                let outward_sign = if pipe.from == node_i { 1.0 } else { -1.0 };
                let corrected_flow = outward_sign * flow - self_flow;

                // Direction: from self toward other (in tile coordinates).
                let dir_x = node_other.center_x(res) - node_self.center_x(res);
                let dir_y = node_other.center_y(res) - node_self.center_y(res);

                fx += corrected_flow * dir_x;
                fy += corrected_flow * dir_y;
            }
        }

        // Apply role sign: sinks go against flow, drivers go with flow.
        dx[ci] += role_sign * fx;
        dy[ci] += role_sign * fy;
    }
}

/// Interpolate pressure from a given pressure vector at a tile-coordinate position.
fn pressure_from_vec(x: f64, y: f64, pressure: &[f64], network: &PipeNetwork) -> f64 {
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    let sx = to_subtile_coord(x, n);
    let sy = to_subtile_coord(y, n);
    let (gx0, gy0, fx, fy) = bilinear_cell(sx, sy, sw, sh);

    let p00 = pressure[subtile_grid_index(gx0, gy0, tile_w, n)];
    let p10 = pressure[subtile_grid_index(gx0 + 1, gy0, tile_w, n)];
    let p01 = pressure[subtile_grid_index(gx0, gy0 + 1, tile_w, n)];
    let p11 = pressure[subtile_grid_index(gx0 + 1, gy0 + 1, tile_w, n)];

    (1.0 - fx) * (1.0 - fy) * p00
        + fx * (1.0 - fy) * p10
        + (1.0 - fx) * fy * p01
        + fx * fy * p11
}

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
/// both bilinearly interpolated on the subtile grid. All pins contribute (fixed and movable).
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

    // Utilization-aware demand scaling: at low utilization, demand is dilute
    // and produces weak pressure gradients. Scale demand by 1/sqrt(util)
    // so the Kirchhoff solve produces proportionally stronger fields.
    let n_tiles = (network.width * network.height) as usize;
    let n_clb_tiles = (n_tiles - network.num_zero_bel_tiles()) as f64;
    let n_movable = cell_to_idx.len() as f64;
    let util = (n_movable / n_clb_tiles.max(1.0)).max(0.01);
    let demand_scale = (1.0 / util).powf(0.25).min(3.0);
    eprintln!("  demand_scale={:.2} (util={:.4})", demand_scale, util);

    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(dp) = net.driver() else { continue };

        let mut sink_positions: Vec<(f64, f64)> = Vec::new();
        let mut has_fixed_pin = ctx.design.cell(dp.cell).bel_strength.is_locked();

        for user in net.users() {
            if !user.is_valid() { continue; }
            has_fixed_pin |= ctx.design.cell(user.cell).bel_strength.is_locked();
            sink_positions.push(pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network));
        }
        if sink_positions.is_empty() { continue; }

        let (dx, dy) = pin_pos(ctx, dp.cell, cell_to_idx, cell_x, cell_y, network);
        let fanout = sink_positions.len() as f64;
        let io_factor = if has_fixed_pin { cfg.io_boost } else { 1.0 };
        let net_demand = io_factor * demand_scale;

        // Driver injects +demand.
        let dsx = to_subtile_coord(dx, n);
        let dsy = to_subtile_coord(dy, n);
        for (gx, gy, bw) in bilinear_weights(dsx, dsy, sw, sh) {
            let ni = subtile_grid_index(gx, gy, tile_w, n);
            demand[ni] += net_demand * bw;
        }

        // Each sink extracts -demand/fanout.
        let sink_weight = net_demand / fanout;
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

/// Sample congestion resistance (1 + util²) at a tile-coordinate position.
/// Returns the average congestion of pipes near the position.
fn congestion_at(x: f64, y: f64, network: &PipeNetwork) -> f64 {
    let n = network.resolution;
    let sw = network.subtile_width();
    let sh = network.subtile_height();
    let tile_w = network.width as usize;

    let sx = to_subtile_coord(x, n);
    let sy = to_subtile_coord(y, n);
    let (gx0, gy0, _fx, _fy) = bilinear_cell(sx, sy, sw, sh);

    let mut r_sum = 0.0;
    let mut r_count = 0usize;
    for &(gx, gy) in &[(gx0, gy0), (gx0 + 1, gy0), (gx0, gy0 + 1), (gx0 + 1, gy0 + 1)] {
        let ni = subtile_grid_index(gx, gy, tile_w, n);
        if ni < network.node_pipes.len() {
            for &pi in &network.node_pipes[ni] {
                let pipe = &network.pipes[pi];
                let util = (pipe.flow.abs() / pipe.capacity.max(1.0)).min(10.0);
                r_sum += 1.0 + util * util;
                r_count += 1;
            }
        }
    }
    if r_count > 0 { r_sum / r_count as f64 } else { 1.0 }
}

/// Compute bottleneck congestion resistance along a straight line
/// from (x0,y0) to (x1,y1). Returns the MAXIMUM congestion sampled
/// at N_SAMPLES points — the choke point determines flow rate.
fn path_congestion(x0: f64, y0: f64, x1: f64, y1: f64, network: &PipeNetwork) -> f64 {
    const N_SAMPLES: usize = 5;
    let w = (network.width - 1) as f64;
    let h = (network.height - 1) as f64;
    let mut max_r = 1.0f64;
    for k in 0..N_SAMPLES {
        let t = (k as f64 + 0.5) / N_SAMPLES as f64;
        let px = (x0 + t * (x1 - x0)).clamp(0.0, w);
        let py = (y0 + t * (y1 - y0)).clamp(0.0, h);
        max_r = max_r.max(congestion_at(px, py, network));
    }
    max_r
}

/// Compute cell displacement from congestion-aware pressure differences.
///
/// For each cell, the force from each connected pin is:
///   F = (P_pin - P_cell) / R_local / dist * direction
///
/// Dividing by R_local (average congestion resistance near the cell) converts
/// the pressure drop into a flow-proportional force. Congested regions have
/// high R_local, which WEAKENS the pull — cells are repelled from congestion
/// rather than attracted to it.
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

    // Compute per-cell displacement from connected pins,
    // divided by path-averaged congestion resistance.
    for (_net_id, net) in ctx.design.iter_alive_nets() {
        let Some(drv) = net.driver() else { continue };
        if net.num_users() == 0 { continue; }

        let mut pins: Vec<(f64, f64, Option<usize>)> = Vec::new();
        let (dpx, dpy) = pin_pos(ctx, drv.cell, cell_to_idx, cell_x, cell_y, network);
        pins.push((dpx, dpy, cell_to_idx.get(&drv.cell).copied()));
        for user in net.users() {
            if !user.is_valid() { continue; }
            let (ux, uy) = pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
            pins.push((ux, uy, cell_to_idx.get(&user.cell).copied()));
        }
        if pins.len() < 2 { continue; }

        for i in 0..pins.len() {
            let Some(ci) = pins[i].2 else { continue };
            let (xi, yi, _) = pins[i];
            let p_self = pressure_at(xi, yi, network);

            for j in 0..pins.len() {
                if i == j { continue; }
                let (xj, yj, _) = pins[j];
                let dir_x = xj - xi;
                let dir_y = yj - yi;
                let dist = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if dist < 0.1 { continue; }

                let p_other = pressure_at(xj, yj, network);
                let dp = p_other - p_self;

                // Path-averaged congestion: sample R_cong along the straight
                // line from cell to pin. The entire corridor's congestion
                // damps the pull, not just the cell's local congestion.
                let r_path = path_congestion(xi, yi, xj, yj, network);
                let weight = dp / (r_path * dist);
                dx[ci] += weight * dir_x;
                dy[ci] += weight * dir_y;
            }
        }
    }

    (dx, dy)
}

/// Eulerian-Lagrangian velocity projection: enforce mass conservation between
/// the continuous Kirchhoff flow field and the discrete cell displacements.
///
/// For each tile containing cells, compute the residual between the continuous
/// fluid velocity (from pipe flows) and the sum of cell intentions (dx, dy).
/// Distribute the residual equally among cells in that tile. This ensures
/// cells carry the exact volume the solver computed — pileups are flushed
/// by the outward flow their own demand creates.
///
/// v_i = u_i + (Q_tile - sum(u_k)) / N
///
/// where u_i is the Bottleneck-R intention, Q_tile is the continuous flow
/// vector at the tile, and N is the number of cells in the tile.
pub fn project_velocity(
    cell_x: &[f64],
    cell_y: &[f64],
    dx: &mut [f64],
    dy: &mut [f64],
    network: &PipeNetwork,
) {
    let num_cells = cell_x.len();
    let tw = network.width as usize;
    let th = network.height as usize;
    let n = network.resolution;
    let n_per_tile = n * n;

    // 1. Accumulate cell intentions per tile and count cells per tile.
    let mut tile_sum_dx = vec![0.0f64; tw * th];
    let mut tile_sum_dy = vec![0.0f64; tw * th];
    let mut tile_count = vec![0u32; tw * th];
    let mut cell_tile = vec![0usize; num_cells];

    for i in 0..num_cells {
        let tx = (cell_x[i].round() as usize).min(tw - 1);
        let ty = (cell_y[i].round() as usize).min(th - 1);
        let ti = ty * tw + tx;
        tile_sum_dx[ti] += dx[i];
        tile_sum_dy[ti] += dy[i];
        tile_count[ti] += 1;
        cell_tile[i] = ti;
    }

    // 2. Build cell density grid and per-tile BEL capacity.
    let mut tile_density = vec![0.0f64; tw * th];
    let mut tile_bel_cap = vec![1.0f64; tw * th];

    for ti in 0..(tw * th) {
        tile_density[ti] = tile_count[ti] as f64;
    }
    for ty in 0..th {
        for tx in 0..tw {
            let ni = network.node_index(tx as i32, ty as i32, 0, 0);
            if ni < network.node_pipes.len() && !network.node_pipes[ni].is_empty() {
                let pi = network.node_pipes[ni][0];
                tile_bel_cap[ty * tw + tx] = network.pipes[pi].capacity.max(0.25);
            }
        }
    }

    // 3. Smooth the density field with a Gaussian kernel.
    //    Radius chosen to span across non-CLB columns (typically 3-7 tiles gap).
    //    This creates a continuous slope that bridges void columns, preventing
    //    the oscillation trap where cells bounce between CLB and non-CLB tiles.
    let smooth_density = {
        // Compute sigma from average CLB column spacing.
        // Scan the middle row to find which columns have BELs.
        let mid_y = th / 2;
        let mut clb_xs: Vec<usize> = Vec::new();
        for tx in 0..tw {
            let ni = network.node_index(tx as i32, mid_y as i32, 0, 0);
            if ni < network.node_pipes.len() && !network.node_pipes[ni].is_empty() {
                let pi = network.node_pipes[ni][0];
                if network.pipes[pi].capacity > 1.0 {
                    clb_xs.push(tx);
                }
            }
        }
        let sigma = if clb_xs.len() >= 2 {
            let mut total_gap = 0.0;
            for i in 1..clb_xs.len() {
                total_gap += (clb_xs[i] - clb_xs[i - 1]) as f64;
            }
            (total_gap / (clb_xs.len() - 1) as f64).max(1.0)
        } else {
            3.0 // reasonable default
        };
        let radius = (sigma * 2.5) as usize;
        let mut smoothed = vec![0.0f64; tw * th];
        for ty in 0..th {
            for tx in 0..tw {
                let ti = ty * tw + tx;
                if tile_density[ti] == 0.0 { continue; }
                let weight = tile_density[ti];
                // Spread this cell's density contribution over the kernel.
                let x0 = tx.saturating_sub(radius);
                let x1 = (tx + radius + 1).min(tw);
                let y0 = ty.saturating_sub(radius);
                let y1 = (ty + radius + 1).min(th);
                for sy in y0..y1 {
                    let dy2 = (sy as f64 - ty as f64) * (sy as f64 - ty as f64);
                    for sx in x0..x1 {
                        let dx2 = (sx as f64 - tx as f64) * (sx as f64 - tx as f64);
                        let g = (-(dx2 + dy2) / (2.0 * sigma * sigma)).exp();
                        smoothed[sy * tw + sx] += weight * g;
                    }
                }
            }
        }
        smoothed
    };

    // 4. Compute radial gradient of SMOOTHED density: -grad(D_smooth).
    //    Central differences on the smooth field. The gradient now spans
    //    across non-CLB voids, providing a continuous slope to guide cells
    //    from x=10 through x=9 (void) to x=13 (next CLB).
    let mut grad_x = vec![0.0f64; tw * th];
    let mut grad_y = vec![0.0f64; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let ti = ty * tw + tx;
            let d_left = if tx > 0 { smooth_density[ti - 1] } else { smooth_density[ti] };
            let d_right = if tx + 1 < tw { smooth_density[ti + 1] } else { smooth_density[ti] };
            let d_up = if ty > 0 { smooth_density[ti - tw] } else { smooth_density[ti] };
            let d_down = if ty + 1 < th { smooth_density[ti + tw] } else { smooth_density[ti] };
            grad_x[ti] = -(d_right - d_left) * 0.5;
            grad_y[ti] = -(d_down - d_up) * 0.5;
        }
    }

    // 4. Radial density expansion, scaled by inverse utilization.
    //    activation = max(0, N/capacity - 1): zero when legal, scales with pileup
    //    direction = -grad(cell_density): radial push away from density center
    //    util_scale = 1/utilization: at low utilization, push harder to overcome
    //    the dilute Bottleneck-R attraction.
    let total_cells = num_cells as f64;
    let total_bel_cap: f64 = tile_bel_cap.iter().sum();
    let util = (total_cells / total_bel_cap.max(1.0)).max(0.01);
    let util_scale = (1.0 / util).sqrt();

    for i in 0..num_cells {
        let ti = cell_tile[i];
        let nc = tile_count[ti] as f64;
        let activation = (nc / tile_bel_cap[ti] - 1.0).max(0.0);
        if activation > 0.0 {
            let gx = grad_x[ti];
            let gy = grad_y[ti];
            let mag = (gx * gx + gy * gy).sqrt();
            if mag > 1e-6 {
                dx[i] += activation * util_scale * gx / mag;
                dy[i] += activation * util_scale * gy / mag;
            }
        }
    }
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

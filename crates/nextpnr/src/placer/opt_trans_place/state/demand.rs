//! Net demand computation and cell demand sign/viscosity.

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::NetId;

use super::super::network::Port;
use super::OptTransState;

const ALL_PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

impl OptTransState {
    pub fn compute_net_demands(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        timing_weight: f64,
        io_boost: f64,
        pump_gain: f64,
    ) -> Vec<f64> {
        let n_j = self.network.num_junctions();
        let mut demand = vec![0.0; n_j];
        let grid_span = (self.network.width + self.network.height) as f64;

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };

            let mut has_fixed_sink = false;
            let mut sink_positions: Vec<(f64, f64)> = Vec::new();
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                has_fixed_sink |= ctx.design.cell(user.cell).bel_strength.is_locked();
                sink_positions.push(self.pin_pos(ctx, user.cell));
            }
            if sink_positions.is_empty() {
                continue;
            }

            let (dx, dy) = self.pin_pos(ctx, dp.cell);
            let fanout = sink_positions.len() as f64;

            // IO boost: nets with a fixed (locked) pin pump harder.
            let has_fixed_pin =
                ctx.design.cell(dp.cell).bel_strength.is_locked() || has_fixed_sink;
            let io_factor = if has_fixed_pin { io_boost } else { 1.0 };

            // Sink centroid determines net span.
            let (sum_x, sum_y) = sink_positions
                .iter()
                .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
            let (cx, cy) = (sum_x / fanout, sum_y / fanout);

            // Span factor: sqrt for sublinear scaling.
            let span = (dx - cx).abs() + (dy - cy).abs();
            let span_factor = 1.0 + (span / grid_span).sqrt();

            // Criticality factor: viscous nets pump harder.
            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            let crit_factor = 1.0 + crit * timing_weight;

            // Dynamic pump: nets violating timing get amplified demand.
            // Quadratic ramp steepens local gradients for stuck long-distance nets.
            let transit_factor = 1.0 + pump_gain * crit.powi(2);

            // Combined scale: IO boost x span x criticality x pump.
            let port_share = 0.25 * io_factor * span_factor * crit_factor * transit_factor;

            // Driver injects +scale, bilinearly spread across 4 ports.
            for (tx, ty, bw) in self.bilinear_weights(dx, dy) {
                let share = port_share * bw;
                for &port in &ALL_PORTS {
                    demand[self.network.junction_index(tx, ty, port)] += share;
                }
            }

            // Each sink extracts scale/fanout, bilinearly spread across 4 ports.
            let sink_share = port_share / fanout;
            for &(sx, sy) in &sink_positions {
                for (tx, ty, bw) in self.bilinear_weights(sx, sy) {
                    let share = sink_share * bw;
                    for &port in &ALL_PORTS {
                        demand[self.network.junction_index(tx, ty, port)] -= share;
                    }
                }
            }
        }

        demand
    }

    /// Per-cell demand sign for asymmetric pressure gradient.
    ///
    /// Returns a smooth sign factor per cell: positive for net sinks, negative
    /// for net sources. Used to flip the pressure gradient direction:
    /// - Source cells (drivers): negative sign -> force = -∇P (toward sinks)
    /// - Sink cells: positive sign -> force = +∇P (toward drivers)
    ///
    /// Uses tanh for smooth transition. The result is in [-1, 1].
    pub fn compute_cell_demand_sign(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        timing_weight: f64,
        io_boost: f64,
    ) -> Vec<f64> {
        let n = self.num_cells();
        let mut cell_demand = vec![0.0; n];

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let Some(dp) = net.driver() else { continue };
            let users = net.users();

            let has_fixed_sink = users.iter().any(|u| {
                u.is_valid() && ctx.design.cell(u.cell).bel_strength.is_locked()
            });
            let has_fixed_pin =
                ctx.design.cell(dp.cell).bel_strength.is_locked() || has_fixed_sink;
            let io_factor = if has_fixed_pin { io_boost } else { 1.0 };

            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            let crit_factor = 1.0 + crit * timing_weight;
            let weight = io_factor * crit_factor;

            let fanout = users.iter().filter(|u| u.is_valid()).count() as f64;
            if fanout == 0.0 {
                continue;
            }

            // Driver contributes positive demand (source).
            if let Some(&idx) = self.cell_to_idx.get(&dp.cell) {
                cell_demand[idx] += weight;
            }
            // Sinks contribute negative demand (extraction).
            let sink_weight = weight / fanout;
            for user in users {
                if !user.is_valid() {
                    continue;
                }
                if let Some(&idx) = self.cell_to_idx.get(&user.cell) {
                    cell_demand[idx] -= sink_weight;
                }
            }
        }

        // Smooth sign: tanh(demand / scale).
        // Scale by median absolute value to normalize.
        let mut abs_vals: Vec<f64> = cell_demand.iter().map(|d| d.abs()).filter(|&a| a > 1e-12).collect();
        let scale = if abs_vals.is_empty() {
            1.0
        } else {
            abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            abs_vals[abs_vals.len() / 2].max(1e-6)
        };

        // Negate and smooth: sources (d>0) get sign<0, sinks (d<0) get sign>0.
        // Used as: grad = sign * ∇P, so sources move down-gradient, sinks up-gradient.
        cell_demand.iter().map(|&d| -(d / scale).tanh()).collect()
    }

    /// Per-cell viscosity from net criticality.
    ///
    /// Viscosity = 1 + alpha * max_criticality across all nets touching this cell.
    /// Critical cells (high viscosity) move slowly and settle first.
    /// Non-critical cells (low viscosity) flow freely around them.
    pub fn compute_cell_viscosity(
        &self,
        ctx: &Context,
        criticality: &FxHashMap<NetId, f64>,
        alpha: f64,
    ) -> Vec<f64> {
        let n = self.num_cells();
        let mut max_crit = vec![0.0_f64; n];

        for (net_id, net) in ctx.design.iter_alive_nets() {
            let crit = criticality.get(&net_id).copied().unwrap_or(0.0);
            if crit <= 0.0 {
                continue;
            }
            let Some(dp) = net.driver() else { continue };

            if let Some(&idx) = self.cell_to_idx.get(&dp.cell) {
                max_crit[idx] = max_crit[idx].max(crit);
            }
            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                if let Some(&idx) = self.cell_to_idx.get(&user.cell) {
                    max_crit[idx] = max_crit[idx].max(crit);
                }
            }
        }

        max_crit.iter().map(|&c| 1.0 + alpha * c).collect()
    }
}

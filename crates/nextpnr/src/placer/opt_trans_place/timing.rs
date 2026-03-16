//! Fluid timing model: arrival time propagation through solved pipe flows.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::NetId;

use super::kirchhoff::transit_time;
use super::network::Port;
use super::state::OptTransState;

pub struct FluidTimingResult {
    pub net_criticality: FxHashMap<NetId, f64>,
}

/// Compute arrival times through the pipe network using solved flows.
///
/// For each net: BFS from driver tile's junctions through pipes to sink junctions.
/// Transit time at each pipe models fluid velocity (congestion penalty).
/// Net criticality = max_sink_arrival / target_period, clamped to [0, 1].
pub fn compute_fluid_timing(
    ctx: &Context,
    state: &OptTransState,
    target_period: f64,
    turbulence_beta: f64,
) -> FluidTimingResult {
    let mut net_criticality = FxHashMap::default();

    if target_period <= 0.0 {
        return FluidTimingResult { net_criticality };
    }

    let network = &state.network;
    let n_j = network.num_junctions();
    let ports = [Port::North, Port::East, Port::South, Port::West];

    // Pre-allocate BFS buffers outside the net loop to avoid per-net allocation.
    let mut arrival = vec![f64::INFINITY; n_j];
    let mut queue = VecDeque::new();

    for (net_id, net) in ctx.design.iter_alive_nets() {
        let driver = net.driver();
        let users = net.users();
        if users.is_empty() {
            continue;
        }
        let Some(dp) = driver else { continue };

        // Reset BFS buffers
        arrival.fill(f64::INFINITY);
        queue.clear();

        // Seed driver junctions
        let (dx, dy) = state.pin_tile(ctx, dp.cell);
        for &port in &ports {
            let j = network.junction_index(dx, dy, port);
            arrival[j] = 0.0;
            queue.push_back(j);
        }

        // BFS relaxation through pipe network
        while let Some(j) = queue.pop_front() {
            let current_arrival = arrival[j];

            for &pipe_idx in &network.junction_pipes[j] {
                let pipe = &network.pipes[pipe_idx];
                let neighbor = if pipe.from == j { pipe.to } else { pipe.from };

                let tau = transit_time(pipe.flow, pipe.capacity, turbulence_beta);
                let new_arrival = current_arrival + tau;

                if new_arrival < arrival[neighbor] {
                    arrival[neighbor] = new_arrival;
                    queue.push_back(neighbor);
                }
            }
        }

        // Compute max arrival at sink junctions
        let mut max_sink_arrival = 0.0f64;
        for user in users {
            if !user.is_valid() {
                continue;
            }
            let (sx, sy) = state.pin_tile(ctx, user.cell);
            for &port in &ports {
                let sink_j = network.junction_index(sx, sy, port);
                if arrival[sink_j] < f64::INFINITY {
                    max_sink_arrival = max_sink_arrival.max(arrival[sink_j]);
                }
            }
        }

        let criticality = (max_sink_arrival / target_period).clamp(0.0, 1.0);
        net_criticality.insert(net_id, criticality);
    }

    FluidTimingResult { net_criticality }
}

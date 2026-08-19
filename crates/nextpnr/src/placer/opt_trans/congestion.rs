//! Congestion metrics derived from pipe usage and resistance.
//!
//! Two quantities:
//! - **Congestion pressure** (per-node): mean excess resistance of adjacent
//!   pipes, used for diagnostics and logging.
//! - **Friction energy** (scalar): total dissipation from congestion,
//!   `Σ_pipe usage × (R_eff - R_base)`. Added to the objective energy for
//!   step acceptance so the optimizer penalizes congestion-increasing moves
//!   without altering the gradient direction.

use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Compute per-node congestion pressure from pipe excess resistance.
///
/// `P_cong[node] = mean_{adj pipes} (R_eff(pipe) - R_base(pipe))`
///
/// Returns a Vec<f64> of length `network.num_nodes()`.
pub fn compute_congestion_pressure(
    network: &PipeNetwork,
    resistance_model: &ResistanceModel,
) -> Vec<f64> {
    let n = network.num_nodes();
    let mut pressure = vec![0.0f64; n];
    let mut degree = vec![0u32; n];

    for pipe in &network.pipes {
        let r_eff = resistance_model.effective_resistance(pipe);
        let r_excess = r_eff - pipe.base_resistance;
        pressure[pipe.from] += r_excess;
        pressure[pipe.to] += r_excess;
        degree[pipe.from] += 1;
        degree[pipe.to] += 1;
    }

    for i in 0..n {
        if degree[i] > 0 {
            pressure[i] /= degree[i] as f64;
        }
    }

    pressure
}

/// Compute the friction energy: mean congestion ratio squared across all pipes.
///
/// `E_friction = Σ_pipe (usage / capacity)²`
///
/// Each term is in [0, 1] for non-overloaded pipes, giving a scale-matched
/// diagnostic alongside the attraction energy. Long-range pipes with high
/// capacity naturally produce lower ratios unless genuinely congested.
pub fn compute_friction_energy(network: &PipeNetwork) -> f64 {
    let mut friction = 0.0f64;
    for pipe in &network.pipes {
        let cap = pipe.capacity;
        if cap <= 0.0 {
            continue;
        }
        let ratio = pipe.net_count / cap;
        friction += ratio * ratio;
    }
    friction
}

/// Beckmann's potential — the functional whose stationary point IS the
/// user-equilibrium assignment:
///
/// `Φ_total = Σ_pipe ∫₀^u t(w) dw = Σ base·(u + α·u^(β+1) / ((β+1)·c_eff^β))`
///
/// The placer's own `energy` is `Σ demand·label`, i.e. `u·t(u)` — the same
/// integrand evaluated at the endpoint rather than integrated, so it misses
/// the `β+1` divisor and overstates the congestion part by a factor of 5 at
/// the default β=4. Worse, its `u` comes from the new loading while its `t`
/// comes from the previous iteration's frozen usage, so it is a Lyapunov
/// function of nothing.
///
/// This is monitor-only: it is the scalar that *should* descend, reported so
/// a run can be judged against the right quantity instead of against a
/// mixed one.
pub fn compute_beckmann_potential(network: &PipeNetwork) -> f64 {
    use super::resistance::{bpr_alpha, bpr_beta, effective_capacity};
    let alpha = bpr_alpha();
    let beta = bpr_beta();
    let mut total = 0.0f64;
    for pipe in &network.pipes {
        let u = pipe.net_count.max(0.0);
        if u <= 0.0 {
            continue;
        }
        let eff_cap = effective_capacity(pipe);
        let congested = if eff_cap > 0.0 {
            alpha * u.powf(beta + 1.0) / ((beta + 1.0) * eff_cap.powf(beta))
        } else {
            0.0
        };
        total += pipe.base_resistance * (u + congested);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, Node, Pipe, PipeNetwork, PipeType};
    use rustc_hash::FxHashMap;

    fn make_pipe(from: usize, to: usize, base: f64, capacity: f64, net_count: f64) -> Pipe {
        Pipe {
            from,
            to,
            base_resistance: base,
            capacity,
            flow: 0.0,
            net_count,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0,
            pipe_type: PipeType::InterTile(Direction::East),
        }
    }

    fn make_network(nodes: Vec<Node>, pipes: Vec<Pipe>) -> PipeNetwork {
        let n = nodes.len();
        let mut node_pipes = vec![Vec::new(); n];
        for (i, pipe) in pipes.iter().enumerate() {
            node_pipes[pipe.from].push(i);
            node_pipes[pipe.to].push(i);
        }
        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|p| 1.0 / p.eff_conductance.max(1e-12))
            .collect();
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * crate::placer::opt_trans::network::DIST_SCALE).round() as u32).max(1))
            .collect();
        let tile_grid = crate::placer::opt_trans::network::TileGrid::build(&pipes, &nodes, 2, 2);
        let flat_adjacency =
            crate::placer::opt_trans::network::FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = std::sync::Arc::new(Vec::new());
        let n_nodes = nodes.len();
        let n_pipes = pipes.len();
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table: crate::placer::opt_trans::tile_cache::SpanCostTable::disabled(n_pipes),
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n_nodes],
            width: 2,
            height: 2,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: 0,
            coarsen: 1,
        }
    }

    #[test]
    fn zero_usage_gives_zero_pressure() {
        let nodes = vec![
            Node {
                tile_x: 0,
                tile_y: 0,
                pressure: 0.0,
            },
            Node {
                tile_x: 1,
                tile_y: 0,
                pressure: 0.0,
            },
        ];
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 0.0)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        assert_eq!(pressure.len(), 2);
        assert!(pressure[0].abs() < 1e-12, "expected 0, got {}", pressure[0]);
        assert!(pressure[1].abs() < 1e-12, "expected 0, got {}", pressure[1]);
    }

    #[test]
    fn half_saturated_pipe_gives_expected_bpr_pressure() {
        // Span-1 pipe with borrow_slack(1) = 1.25 applied to eff_capacity.
        // u/c_eff = 5 / (10 * 1.25) = 0.4 under BPR(α=0.05, β=4):
        //   R_eff = base · (1 + 0.05 · 0.4^4)
        //   R_excess = 0.05 · 0.4^4 · base
        let nodes = vec![
            Node {
                tile_x: 0,
                tile_y: 0,
                pressure: 0.0,
            },
            Node {
                tile_x: 1,
                tile_y: 0,
                pressure: 0.0,
            },
        ];
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 5.0)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        let expected = 0.05 * (5.0f64 / 12.5).powi(4);
        assert!((pressure[0] - expected).abs() < 1e-9, "got {}", pressure[0]);
        assert!((pressure[1] - expected).abs() < 1e-9, "got {}", pressure[1]);
    }

    #[test]
    fn node_with_multiple_pipes_averages_bpr() {
        // Node 1 sits between an empty pipe (excess=0) and a half-saturated
        // span-1 pipe (excess = 0.05 · (5/12.5)^4). Mean = half of that.
        // Node 2 only touches the half-saturated pipe.
        let nodes = vec![
            Node {
                tile_x: 0,
                tile_y: 0,
                pressure: 0.0,
            },
            Node {
                tile_x: 1,
                tile_y: 0,
                pressure: 0.0,
            },
            Node {
                tile_x: 2,
                tile_y: 0,
                pressure: 0.0,
            },
        ];
        let pipes = vec![
            make_pipe(0, 1, 1.0, 10.0, 0.0),
            make_pipe(1, 2, 1.0, 10.0, 5.0),
        ];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        let per_pipe_excess = 0.05 * (5.0f64 / 12.5).powi(4);
        assert!(pressure[0].abs() < 1e-12);
        assert!(
            (pressure[1] - per_pipe_excess / 2.0).abs() < 1e-9,
            "got {}",
            pressure[1]
        );
        assert!(
            (pressure[2] - per_pipe_excess).abs() < 1e-9,
            "got {}",
            pressure[2]
        );
    }
}

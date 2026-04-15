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
        let ratio = pipe.net_count as f64 / cap;
        friction += ratio * ratio;
    }
    friction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, Node, Pipe, PipeNetwork, PipeType};
    use rustc_hash::FxHashMap;

    fn make_pipe(from: usize, to: usize, base: f64, capacity: f64, net_count: u32) -> Pipe {
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
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_lookup: FxHashMap::default(),
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
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
        ];
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 0)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        assert_eq!(pressure.len(), 2);
        assert!(pressure[0].abs() < 1e-12, "expected 0, got {}", pressure[0]);
        assert!(pressure[1].abs() < 1e-12, "expected 0, got {}", pressure[1]);
    }

    #[test]
    fn half_saturated_gives_base_resistance_pressure() {
        let nodes = vec![
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
        ];
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 5)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        assert!((pressure[0] - 1.0).abs() < 1e-9, "got {}", pressure[0]);
        assert!((pressure[1] - 1.0).abs() < 1e-9, "got {}", pressure[1]);
    }

    #[test]
    fn node_with_multiple_pipes_averages() {
        let nodes = vec![
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 2, tile_y: 0, pressure: 0.0 },
        ];
        let pipes = vec![
            make_pipe(0, 1, 1.0, 10.0, 0),
            make_pipe(1, 2, 1.0, 10.0, 5),
        ];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        assert!(pressure[0].abs() < 1e-12);
        assert!((pressure[1] - 0.5).abs() < 1e-9, "got {}", pressure[1]);
        assert!((pressure[2] - 1.0).abs() < 1e-9, "got {}", pressure[2]);
    }
}

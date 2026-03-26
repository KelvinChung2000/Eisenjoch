//! Optimal transport pipe network model for FPGA placement (Beckmann formulation).
//!
//! Models the FPGA tile grid as a network of pipes and junctions:
//! - Each tile has 4 junction nodes (N, E, S, W boundary ports)
//! - Tiles connected by inter-tile pipes in cardinal directions
//! - Intra-tile pipes from Schur-condensed internal switch matrix
//! - Pipe resistance derived from chipdb wire count per direction
//!
//! Simplified from old opt_trans_place/network.rs:
//! - Removed aggregate_flow, agg_flow_scale fields
//! - Added net_count field for interference approximation
//! - Stores base_resistance separately from effective resistance

use crate::context::Context;

/// Direction of an inter-tile pipe between two adjacent tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    East,
    South,
}

/// Identifies the specific boundary port within a single tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Port {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

/// Distinguishes between global routing wires and internal switch matrix paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipeType {
    InterTile(Direction),
    IntraTile,
}

/// A junction node in the pipe network, representing one port on a tile.
#[derive(Debug, Clone)]
pub struct Junction {
    pub x: i32,
    pub y: i32,
    pub port: Port,
    pub pressure: f64,
}

/// A pipe connecting two junctions.
#[derive(Debug, Clone)]
pub struct Pipe {
    pub from: usize,
    pub to: usize,
    /// Base resistance (from chipdb geometry, constant).
    pub base_resistance: f64,
    /// Routing capacity (wire count or BEL count).
    pub capacity: f64,
    /// Current flow through this pipe (updated each Kirchhoff solve).
    pub flow: f64,
    /// Number of distinct nets using this pipe (for interference).
    pub net_count: u32,
    pub pipe_type: PipeType,
}

/// The pipe network: junctions + pipes over the FPGA tile grid.
pub struct PipeNetwork {
    pub junctions: Vec<Junction>,
    pub pipes: Vec<Pipe>,
    pub junction_pipes: Vec<Vec<usize>>,
    pub width: i32,
    pub height: i32,
    /// Grid origin offset: the virtual grid starts at (x0, y0) in physical coordinates.
    pub x0: i32,
    pub y0: i32,
    /// Per tile-type Schur condensation matrices.
    pub schur_matrices: Vec<[[f64; 4]; 4]>,
}

const ALL_PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

impl PipeNetwork {
    /// Build a pipe network from the chip database.
    pub fn from_context(ctx: &Context) -> Self {
        let w = ctx.chipdb().width();
        let h = ctx.chipdb().height();
        let n = (w * h) as usize;

        let x0 = 0;
        let y0 = 0;

        // 4 junctions per tile (N, E, S, W).
        let mut junctions = Vec::with_capacity(n * 4);
        for tile in 0..n {
            let x = (tile as i32) % w;
            let y = (tile as i32) / w;
            for &port in &ALL_PORTS {
                junctions.push(Junction {
                    x,
                    y,
                    port,
                    pressure: 0.0,
                });
            }
        }

        let mut pipes = Vec::new();
        let mut junction_pipes = vec![Vec::new(); n * 4];

        let num_tile_types = ctx.chipdb().num_tile_types();
        let schur_matrices = compute_schur_matrices(ctx, num_tile_types);

        // 1. Intra-tile pipes (internal switch matrix via Schur condensation).
        for y in 0..h {
            for x in 0..w {
                let tile = ctx.chipdb().tile_by_xy(x + x0, y + y0);
                let tt_idx = ctx.chipdb().tile_type_index(tile) as usize;
                let tt = ctx.chipdb().tile_type(tile);
                let n_bels = tt.bels.len() as f64;

                let matrix = &schur_matrices[tt_idx];

                for i in 0..4 {
                    for j in (i + 1)..4 {
                        let conductance = -matrix[i][j];
                        if conductance > 1e-9 {
                            let from = ((y * w + x) * 4 + i as i32) as usize;
                            let to = ((y * w + x) * 4 + j as i32) as usize;
                            let pipe_idx = pipes.len();
                            pipes.push(Pipe {
                                from,
                                to,
                                base_resistance: 1.0 / conductance,
                                capacity: n_bels.max(1.0),
                                flow: 0.0,
                                net_count: 0,
                                pipe_type: PipeType::IntraTile,
                            });
                            junction_pipes[from].push(pipe_idx);
                            junction_pipes[to].push(pipe_idx);
                        }
                    }
                }
            }
        }

        // 2. Inter-tile pipes (East direction).
        for y in 0..h {
            for x in 0..(w - 1) {
                let from = ((y * w + x) * 4 + Port::East as i32) as usize;
                let to = ((y * w + x + 1) * 4 + Port::West as i32) as usize;

                let wire_count = estimate_wire_count(ctx, x + x0, y + y0, Direction::East);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    base_resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
                    net_count: 0,
                    pipe_type: PipeType::InterTile(Direction::East),
                });
                junction_pipes[from].push(pipe_idx);
                junction_pipes[to].push(pipe_idx);
            }
        }

        // 3. Inter-tile pipes (South direction).
        for y in 0..(h - 1) {
            for x in 0..w {
                let from = ((y * w + x) * 4 + Port::South as i32) as usize;
                let to = (((y + 1) * w + x) * 4 + Port::North as i32) as usize;

                let wire_count = estimate_wire_count(ctx, x + x0, y + y0, Direction::South);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    base_resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
                    net_count: 0,
                    pipe_type: PipeType::InterTile(Direction::South),
                });
                junction_pipes[from].push(pipe_idx);
                junction_pipes[to].push(pipe_idx);
            }
        }

        Self {
            junctions,
            pipes,
            junction_pipes,
            width: w,
            height: h,
            x0,
            y0,
            schur_matrices,
        }
    }

    /// Index of junction at virtual tile (x, y) with given port.
    #[inline]
    pub fn junction_index(&self, x: i32, y: i32, port: Port) -> usize {
        ((y * self.width + x) * 4 + port as i32) as usize
    }

    /// Number of junctions in the network.
    pub fn num_junctions(&self) -> usize {
        self.junctions.len()
    }

    /// Number of pipes in the network.
    pub fn num_pipes(&self) -> usize {
        self.pipes.len()
    }

    /// Reset all dynamic state (pressures, flows, net counts).
    pub fn reset(&mut self) {
        for j in &mut self.junctions {
            j.pressure = 0.0;
        }
        for p in &mut self.pipes {
            p.flow = 0.0;
            p.net_count = 0;
        }
    }

    /// Maximum utilization ratio |flow|/capacity across all pipes.
    pub fn max_utilization(&self) -> f64 {
        self.pipes
            .iter()
            .filter(|p| p.capacity > 0.0)
            .map(|p| p.flow.abs() / p.capacity)
            .fold(0.0, f64::max)
    }
}

/// Estimate wire count between adjacent tiles in the given direction.
fn estimate_wire_count(ctx: &Context, x: i32, y: i32, direction: Direction) -> usize {
    let tile = ctx.chipdb().tile_by_xy(x, y);
    let tt = ctx.chipdb().tile_type(tile);
    let total_pips = tt.pips.len();
    let bels = tt.bels.len();

    let (nx, ny) = match direction {
        Direction::East => (x + 1, y),
        Direction::South => (x, y + 1),
    };

    let neighbor_tile = ctx.chipdb().tile_by_xy(nx, ny);
    let ntt = ctx.chipdb().tile_type(neighbor_tile);
    let neighbor_pips = ntt.pips.len();
    let neighbor_bels = ntt.bels.len();

    if bels == 0 || neighbor_bels == 0 {
        return 1;
    }
    let min_pips = total_pips.min(neighbor_pips);
    (min_pips / 4).max(1)
}

/// Pipe resistance: 1 / n_wires.
fn compute_resistance(wire_count: usize) -> f64 {
    1.0 / (wire_count as f64).max(1.0)
}

/// Schur condensation of internal BEL-to-port sub-networks into 4x4
/// port-to-port conductance matrices (N, E, S, W) per tile type.
fn compute_schur_matrices(ctx: &Context, num_tile_types: usize) -> Vec<[[f64; 4]; 4]> {
    let mut matrices = Vec::with_capacity(num_tile_types);

    for tt_idx in 0..num_tile_types {
        let tt = ctx.chipdb().tile_type_by_index(tt_idx as i32);
        let n_bels = tt.bels.len();

        if n_bels == 0 {
            let g_pass = 0.001;
            let mut m = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    if i == j {
                        m[i][j] = 3.0 * g_pass;
                    } else {
                        m[i][j] = -g_pass;
                    }
                }
            }
            matrices.push(m);
            continue;
        }

        let g_off = 0.1 + 0.4 * (n_bels as f64 / 24.0).min(1.0);
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    m[i][j] = 3.0 * g_off;
                } else {
                    m[i][j] = -g_off;
                }
            }
        }
        matrices.push(m);
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resistance_decreases_with_more_wires() {
        let r1 = compute_resistance(1);
        let r10 = compute_resistance(10);
        assert!(r10 < r1);
    }

    #[test]
    fn resistance_positive() {
        for wc in 0..100 {
            assert!(compute_resistance(wc) > 0.0);
        }
    }
}

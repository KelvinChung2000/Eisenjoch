//! Hydraulic pipe network model for FPGA placement.
//!
//! Models the FPGA tile grid as a network of pipes and junctions:
//! - Each tile is a junction node with a pressure variable
//! - Tiles connected by pipes in 4 cardinal directions
//! - Pipe resistance derived from chipdb wire count per direction
//!
//! Schur complement condensation reduces internal BEL-to-port sub-networks
//! to 4×4 effective port-to-port conductance matrices.

use crate::context::Context;

/// Direction of a pipe between two junctions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    East,
    South,
}

/// A junction node (one per tile) in the pipe network.
#[derive(Debug, Clone)]
pub struct Junction {
    pub x: i32,
    pub y: i32,
    pub pressure: f64,
    /// Net flow demand (positive = source, negative = sink).
    pub demand: f64,
}

/// A pipe connecting two junctions.
#[derive(Debug, Clone)]
pub struct Pipe {
    pub from: usize,
    pub to: usize,
    /// r = base / n_wires² (fewer wires = higher resistance).
    pub resistance: f64,
    pub capacity: f64,
    pub flow: f64,
    pub direction: Direction,
}

pub struct PipeNetwork {
    pub junctions: Vec<Junction>,
    /// East and South pipes only; reverse direction for West/North.
    pub pipes: Vec<Pipe>,
    /// Junction index -> list of incident pipe indices.
    pub junction_pipes: Vec<Vec<usize>>,
    pub width: i32,
    pub height: i32,
    /// Per tile-type Schur condensation matrices (4x4: N, E, S, W).
    pub schur_matrices: Vec<[[f64; 4]; 4]>,
}

impl PipeNetwork {
    /// Build a pipe network from the chip database.
    ///
    /// Creates one junction per tile and pipes between adjacent tiles.
    /// Resistance is inversely proportional to wire count at tile boundaries.
    pub fn from_context(ctx: &Context) -> Self {
        let w = ctx.chipdb().width();
        let h = ctx.chipdb().height();
        let n = (w * h) as usize;

        let junctions: Vec<Junction> = (0..n)
            .map(|tile| Junction {
                x: (tile as i32) % w,
                y: (tile as i32) / w,
                pressure: 0.0,
                demand: 0.0,
            })
            .collect();

        let mut pipes = Vec::new();
        let mut junction_pipes = vec![Vec::new(); n];

        for y in 0..h {
            for x in 0..(w - 1) {
                let from = (y * w + x) as usize;
                let to = (y * w + x + 1) as usize;

                let wire_count = estimate_wire_count(ctx, x, y, Direction::East);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
                    direction: Direction::East,
                });
                junction_pipes[from].push(pipe_idx);
                junction_pipes[to].push(pipe_idx);
            }
        }

        for y in 0..(h - 1) {
            for x in 0..w {
                let from = (y * w + x) as usize;
                let to = ((y + 1) * w + x) as usize;

                let wire_count = estimate_wire_count(ctx, x, y, Direction::South);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
                    direction: Direction::South,
                });
                junction_pipes[from].push(pipe_idx);
                junction_pipes[to].push(pipe_idx);
            }
        }

        let num_tile_types = ctx.chipdb().num_tile_types();
        let schur_matrices = compute_schur_matrices(ctx, num_tile_types);

        Self {
            junctions,
            pipes,
            junction_pipes,
            width: w,
            height: h,
            schur_matrices,
        }
    }

    #[inline]
    pub fn junction_index(&self, x: i32, y: i32) -> usize {
        (y * self.width + x) as usize
    }

    pub fn reset(&mut self) {
        for j in &mut self.junctions {
            j.pressure = 0.0;
            j.demand = 0.0;
        }
        for p in &mut self.pipes {
            p.flow = 0.0;
        }
    }

    pub fn num_junctions(&self) -> usize {
        self.junctions.len()
    }

    pub fn num_pipes(&self) -> usize {
        self.pipes.len()
    }

    /// Maximum utilization ratio |Q|/C across all pipes.
    pub fn max_utilization(&self) -> f64 {
        self.pipes
            .iter()
            .filter(|p| p.capacity > 0.0)
            .map(|p| p.flow.abs() / p.capacity)
            .fold(0.0, f64::max)
    }
}

/// Estimate wire count between adjacent tiles in the given direction.
///
/// Uses the average PIP count of source and neighbor tiles divided by 4 directions
/// as a heuristic for directional routing capacity.
fn estimate_wire_count(ctx: &Context, x: i32, y: i32, direction: Direction) -> usize {
    let tile = ctx.chipdb().tile_by_xy(x, y);
    let total_pips = ctx.chipdb().tile_type(tile).pips.len();

    let (nx, ny) = match direction {
        Direction::East => (x + 1, y),
        Direction::South => (x, y + 1),
    };

    let neighbor_tile = ctx.chipdb().tile_by_xy(nx, ny);
    let neighbor_pips = ctx.chipdb().tile_type(neighbor_tile).pips.len();

    let avg_pips = (total_pips + neighbor_pips) / 2;
    (avg_pips / 4).max(1)
}

/// Pipe resistance: 1 / n_wires² (fewer wires = higher resistance).
fn compute_resistance(wire_count: usize) -> f64 {
    let n = wire_count as f64;
    1.0 / (n * n).max(1.0)
}

/// Schur condensation of internal BEL-to-port sub-networks into 4x4
/// port-to-port conductance matrices (N, E, S, W) per tile type.
fn compute_schur_matrices(ctx: &Context, num_tile_types: usize) -> Vec<[[f64; 4]; 4]> {
    let mut matrices = Vec::with_capacity(num_tile_types);

    for tt_idx in 0..num_tile_types {
        let tt = ctx.chipdb().tile_type_by_index(tt_idx as i32);
        let n_bels = tt.bels.len();

        if n_bels == 0 {
            let mut m = [[0.0; 4]; 4];
            for i in 0..4 {
                m[i][i] = 1.0;
            }
            matrices.push(m);
            continue;
        }

        // Schur complement with uniform BEL-to-port conductance g = 1/n_bels:
        // G_off = n_bels * g² / (4g) = 1/4
        let g_off = 0.25;

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

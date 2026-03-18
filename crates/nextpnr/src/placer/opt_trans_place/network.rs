//! Optimal transport pipe network model for FPGA placement.
//!
//! Models the FPGA tile grid as a network of pipes and junctions:
//! - Each tile is a junction node with a pressure variable
//! - Tiles connected by pipes in 4 cardinal directions
//! - Pipe resistance derived from chipdb wire count per direction
//!
//! Schur complement condensation reduces internal BEL-to-port sub-networks
//! to 4×4 effective port-to-port conductance matrices.

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

/// A pipe connecting two junctions, modelling either external wires or internal routing.
#[derive(Debug, Clone)]
pub struct Pipe {
    pub from: usize,
    pub to: usize,
    /// Resistance is $1 / conductance$. For intra-tile pipes, this comes from the Schur matrix.
    pub resistance: f64,
    pub capacity: f64,
    pub flow: f64,
    pub pipe_type: PipeType,
}

pub struct PipeNetwork {
    pub junctions: Vec<Junction>,
    pub pipes: Vec<Pipe>,
    pub junction_pipes: Vec<Vec<usize>>,
    pub width: i32,
    pub height: i32,
    /// Grid origin offset — the virtual grid starts at (x0, y0) in physical coordinates.
    pub x0: i32,
    pub y0: i32,
    /// Per tile-type Schur condensation matrices.
    pub schur_matrices: Vec<[[f64; 4]; 4]>,
}

impl PipeNetwork {
    /// Build a pipe network from the chip database.
    ///
    /// Creates one junction per tile and pipes between adjacent tiles.
    /// Resistance is inversely proportional to wire count at tile boundaries.
    pub fn from_context(ctx: &Context) -> Self {
        Self::from_context_with_bounds(ctx, None)
    }

    /// Build a pipe network, optionally cropped to a bounding box.
    /// `bounds`: Optional (center_x, center_y, half_size) in physical coords.
    pub fn from_context_with_bounds(ctx: &Context, bounds: Option<(i32, i32, i32)>) -> Self {
        let full_w = ctx.chipdb().width();
        let full_h = ctx.chipdb().height();

        let (min_x, min_y, max_x, max_y) = if let Some((cx, cy, half)) = bounds {
            // Crop to specified region with margin for growth
            let margin = half; // Allow 2x growth
            let min_x = (cx - half - margin).max(0);
            let min_y = (cy - half - margin).max(0);
            let max_x = (cx + half + margin).min(full_w - 1);
            let max_y = (cy + half + margin).min(full_h - 1);
            (min_x, min_y, max_x, max_y)
        } else {
            (0, 0, full_w - 1, full_h - 1)
        };

        let w = max_x - min_x + 1;
        let h = max_y - min_y + 1;
        if w != full_w || h != full_h {
            eprintln!(
                "PipeNetwork: cropped grid from {}x{} to {}x{} (bbox [{},{}]-[{},{}])",
                full_w, full_h, w, h, min_x, min_y, max_x, max_y,
            );
        }
        let n = (w * h) as usize;

        const PORTS: [Port; 4] = [Port::North, Port::East, Port::South, Port::West];

        let x0 = min_x;
        let y0 = min_y;

        // 4 junctions per tile (North, East, South, West).
        // Junction coordinates are in virtual (cropped) space.
        let mut junctions = Vec::with_capacity(n * 4);
        for tile in 0..n {
            let x = (tile as i32) % w;
            let y = (tile as i32) / w;
            for &port in &PORTS {
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

        // 1. Build Intra-Tile Pipes (Internal Switch Matrix)
        for y in 0..h {
            for x in 0..w {
                // Map virtual coords to physical for chipdb lookup
                let tile = ctx.chipdb().tile_by_xy(x + x0, y + y0);
                let tt_idx = ctx.chipdb().tile_type_index(tile) as usize;
                let tt = ctx.chipdb().tile_type(tile);
                let matrix = &schur_matrices[tt_idx];
                let n_bels = tt.bels.len() as f64;

                // Connect internal ports to each other.
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        let conductance = -matrix[i][j];
                        if conductance > 1e-9 {
                            let from = (((y * w) + x) * 4 + (i as i32)) as usize;
                            let to = (((y * w) + x) * 4 + (j as i32)) as usize;
                            let pipe_idx = pipes.len();
                            pipes.push(Pipe {
                                from,
                                to,
                                resistance: 1.0 / conductance,
                                capacity: n_bels.max(1.0),
                                flow: 0.0,
                                pipe_type: PipeType::IntraTile,
                            });
                            junction_pipes[from].push(pipe_idx);
                            junction_pipes[to].push(pipe_idx);
                        }
                    }
                }
            }
        }

        // 2. Build Inter-Tile Pipes (Global Routing Channels)
        for y in 0..h {
            for x in 0..(w - 1) {
                // East port of (x, y) to West port of (x+1, y)
                let from = (((y * w) + x) * 4 + (Port::East as i32)) as usize;
                let to = (((y * w) + x + 1) * 4 + (Port::West as i32)) as usize;

                let wire_count = estimate_wire_count(ctx, x + x0, y + y0, Direction::East);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
                    pipe_type: PipeType::InterTile(Direction::East),
                });
                junction_pipes[from].push(pipe_idx);
                junction_pipes[to].push(pipe_idx);
            }
        }

        for y in 0..(h - 1) {
            for x in 0..w {
                // South port of (x, y) to North port of (x, y+1)
                let from = (((y * w) + x) * 4 + (Port::South as i32)) as usize;
                let to = ((((y + 1) * w) + x) * 4 + (Port::North as i32)) as usize;

                let wire_count = estimate_wire_count(ctx, x + x0, y + y0, Direction::South);
                let pipe_idx = pipes.len();
                pipes.push(Pipe {
                    from,
                    to,
                    resistance: compute_resistance(wire_count),
                    capacity: wire_count as f64,
                    flow: 0.0,
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

    #[inline]
    pub fn junction_index(&self, x: i32, y: i32, port: Port) -> usize {
        ((y * self.width + x) * 4 + port as i32) as usize
    }

    pub fn reset(&mut self) {
        for j in &mut self.junctions {
            j.pressure = 0.0;
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

    // Use the MINIMUM of the two tiles' PIP counts (bottleneck model).
    // If either tile has no BELs (NULL tile), it has no real routing —
    // use just 1 wire (high resistance) so flow avoids that path.
    if bels == 0 || neighbor_bels == 0 {
        return 1; // minimal capacity through empty tiles
    }
    let min_pips = total_pips.min(neighbor_pips);
    (min_pips / 4).max(1)
}

/// Pipe resistance: 1 / n_wires (linear).
///
/// Fewer wires = narrower pipe = higher resistance = more pump energy.
/// Linear scaling keeps resistance high enough for meaningful pressure
/// gradients while reflecting actual routing availability.
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
            let mut m = [[0.0; 4]; 4];
            for i in 0..4 {
                m[i][i] = 1.0;
            }
            matrices.push(m);
            continue;
        }

        // Schur complement: internal conductance scales with BEL count.
        // More BELs = more internal routing paths = higher conductance
        // between ports.  Normalized so a tile with 1 BEL has g_off = 0.1
        // and a tile with 24 BELs has g_off ≈ 0.5.
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

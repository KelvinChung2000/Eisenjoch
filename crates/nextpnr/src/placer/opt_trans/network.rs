//! Subtile grid network model for FPGA placement (Beckmann formulation).
//!
//! Each tile is decomposed into an N×N grid of subtile nodes, where N is
//! the configurable `subtile_resolution`. This produces a regular 2D lattice
//! of (W·N) × (H·N) nodes that AMG can coarsen cleanly.
//!
//! Connections:
//! - Intra-tile: adjacent subtiles within the same tile (4-connected grid)
//! - Inter-tile: boundary subtiles of adjacent tiles connected across tile edges

use crate::context::Context;

/// Direction of an inter-tile pipe between two adjacent tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    East,
    South,
}

/// Distinguishes intra-tile (internal) from inter-tile (boundary) pipes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeType {
    InterTile(Direction),
    IntraTile,
}

/// A subtile node in the network.
#[derive(Debug, Clone)]
pub struct Node {
    /// Tile coordinate.
    pub tile_x: i32,
    pub tile_y: i32,
    /// Subtile coordinate within the tile (0..N).
    pub sub_x: usize,
    pub sub_y: usize,
    /// Pressure (solved by Kirchhoff system).
    pub pressure: f64,
}

impl Node {
    /// Physical center of this subtile in tile coordinates.
    #[inline]
    pub fn center_x(&self, resolution: usize) -> f64 {
        self.tile_x as f64 + (self.sub_x as f64 + 0.5) / resolution as f64
    }

    #[inline]
    pub fn center_y(&self, resolution: usize) -> f64 {
        self.tile_y as f64 + (self.sub_y as f64 + 0.5) / resolution as f64
    }
}

/// A pipe connecting two nodes.
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

/// The subtile pipe network over the FPGA tile grid.
pub struct PipeNetwork {
    pub nodes: Vec<Node>,
    pub pipes: Vec<Pipe>,
    pub node_pipes: Vec<Vec<usize>>,
    /// Tile grid dimensions.
    pub width: i32,
    pub height: i32,
    /// Grid origin offset: the virtual grid starts at (x0, y0) in physical coordinates.
    pub x0: i32,
    pub y0: i32,
    /// Subtile resolution: N×N subtiles per tile.
    pub resolution: usize,
    /// Number of tiles with zero BELs (routing-only, BRAM, etc.).
    pub zero_bel_tiles: usize,
}

impl PipeNetwork {
    /// Build a subtile network from the chip database.
    pub fn from_context(ctx: &Context, resolution: usize) -> Self {
        assert!(resolution >= 1, "subtile resolution must be >= 1");
        let w = ctx.chipdb().width();
        let h = ctx.chipdb().height();
        let n_tiles = (w * h) as usize;
        let n_per_tile = resolution * resolution;

        let x0 = 0;
        let y0 = 0;

        // Create nodes: N×N per tile.
        let mut nodes = Vec::with_capacity(n_tiles * n_per_tile);
        for tile in 0..n_tiles {
            let tx = (tile as i32) % w;
            let ty = (tile as i32) / w;
            for sy in 0..resolution {
                for sx in 0..resolution {
                    nodes.push(Node {
                        tile_x: tx,
                        tile_y: ty,
                        sub_x: sx,
                        sub_y: sy,
                        pressure: 0.0,
                    });
                }
            }
        }

        let total_nodes = nodes.len();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); total_nodes];

        // Helper: node index for tile (tx, ty), subtile (sx, sy).
        let idx = |tx: i32, ty: i32, sx: usize, sy: usize| -> usize {
            ((ty * w + tx) as usize) * n_per_tile + sy * resolution + sx
        };

        // 1. Intra-tile pipes: 4-connected grid within each tile.
        let mut zero_bel_tiles = 0usize;
        let mut total_bels = 0usize;
        // One-time: dump PIP/BEL counts along y=h/2 for debugging.
        let mid_y = h / 2;
        for ty in 0..h {
            for tx in 0..w {
                let tile = ctx.chipdb().tile_by_xy(tx + x0, ty + y0);
                let tt = ctx.chipdb().tile_type(tile);
                let n_bels = tt.bels.len();
                if n_bels == 0 { zero_bel_tiles += 1; }
                total_bels += n_bels;

                // Conductance scales with routing capacity: max(wires, pips).
                // Wires are pass-through tracks; PIPs are switchable connections.
                // Both contribute to how easily flow traverses the tile.
                let n_pips = tt.pips.len();
                let n_wires = tt.wires.len();
                let routing_capacity = n_wires.max(n_pips);
                if ty == mid_y && tx % 5 == 0 {
                    let g_intra_val = intra_tile_conductance(routing_capacity, resolution);
                    eprintln!("    tile({:3},{:3}): wires={:5} pips={:5} bels={:2} g_intra={:.4}", tx, ty, n_wires, n_pips, n_bels, g_intra_val);
                }
                let g_intra = intra_tile_conductance(routing_capacity, resolution);
                let capacity = (n_bels as f64).max(1.0) / n_per_tile as f64;

                // Horizontal edges within tile.
                for sy in 0..resolution {
                    for sx in 0..(resolution - 1) {
                        let from = idx(tx, ty, sx, sy);
                        let to = idx(tx, ty, sx + 1, sy);
                        add_pipe(
                            &mut pipes,
                            &mut node_pipes,
                            from,
                            to,
                            1.0 / g_intra,
                            capacity,
                            PipeType::IntraTile,
                        );
                    }
                }
                // Vertical edges within tile.
                for sy in 0..(resolution - 1) {
                    for sx in 0..resolution {
                        let from = idx(tx, ty, sx, sy);
                        let to = idx(tx, ty, sx, sy + 1);
                        add_pipe(
                            &mut pipes,
                            &mut node_pipes,
                            from,
                            to,
                            1.0 / g_intra,
                            capacity,
                            PipeType::IntraTile,
                        );
                    }
                }
            }
        }

        // 2. Inter-tile pipes East: right boundary of (tx,ty) ↔ left boundary of (tx+1,ty).
        for ty in 0..h {
            for tx in 0..(w - 1) {
                let wire_count = estimate_wire_count(ctx, tx + x0, ty + y0, Direction::East);
                let g_inter = inter_tile_conductance(wire_count, resolution);
                let capacity = wire_count as f64 / resolution as f64;

                for sy in 0..resolution {
                    let from = idx(tx, ty, resolution - 1, sy);
                    let to = idx(tx + 1, ty, 0, sy);
                    add_pipe(
                        &mut pipes,
                        &mut node_pipes,
                        from,
                        to,
                        1.0 / g_inter,
                        capacity,
                        PipeType::InterTile(Direction::East),
                    );
                }
            }
        }

        // 3. Inter-tile pipes South: bottom boundary of (tx,ty) ↔ top boundary of (tx,ty+1).
        for ty in 0..(h - 1) {
            for tx in 0..w {
                let wire_count = estimate_wire_count(ctx, tx + x0, ty + y0, Direction::South);
                let g_inter = inter_tile_conductance(wire_count, resolution);
                let capacity = wire_count as f64 / resolution as f64;

                for sx in 0..resolution {
                    let from = idx(tx, ty, sx, resolution - 1);
                    let to = idx(tx, ty + 1, sx, 0);
                    add_pipe(
                        &mut pipes,
                        &mut node_pipes,
                        from,
                        to,
                        1.0 / g_inter,
                        capacity,
                        PipeType::InterTile(Direction::South),
                    );
                }
            }
        }

        eprintln!(
            "  network: {}x{} = {} tiles, {} zero-BEL ({:.1}%), {} total BELs, {} nodes, {} pipes",
            w, h, n_tiles, zero_bel_tiles,
            100.0 * zero_bel_tiles as f64 / n_tiles as f64,
            total_bels, total_nodes, pipes.len(),
        );

        Self {
            nodes,
            pipes,
            node_pipes,
            width: w,
            height: h,
            x0,
            y0,
            resolution,
            zero_bel_tiles,
        }
    }

    /// Index of node at tile (tx, ty), subtile (sx, sy).
    #[inline]
    pub fn node_index(&self, tx: i32, ty: i32, sx: usize, sy: usize) -> usize {
        let n = self.resolution;
        ((ty * self.width + tx) as usize) * (n * n) + sy * n + sx
    }

    /// Number of nodes per tile (N²).
    #[inline]
    pub fn nodes_per_tile(&self) -> usize {
        self.resolution * self.resolution
    }

    /// Total grid dimensions in subtile units.
    #[inline]
    pub fn subtile_width(&self) -> usize {
        self.width as usize * self.resolution
    }

    #[inline]
    pub fn subtile_height(&self) -> usize {
        self.height as usize * self.resolution
    }

    /// Number of tiles with zero BELs.
    pub fn num_zero_bel_tiles(&self) -> usize {
        self.zero_bel_tiles
    }

    /// Number of nodes in the network.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of pipes in the network.
    pub fn num_pipes(&self) -> usize {
        self.pipes.len()
    }

    /// Reset all dynamic state (pressures, flows, net counts).
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.pressure = 0.0;
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

/// Add a pipe and update adjacency lists.
fn add_pipe(
    pipes: &mut Vec<Pipe>,
    node_pipes: &mut [Vec<usize>],
    from: usize,
    to: usize,
    base_resistance: f64,
    capacity: f64,
    pipe_type: PipeType,
) {
    let pipe_idx = pipes.len();
    pipes.push(Pipe {
        from,
        to,
        base_resistance,
        capacity,
        flow: 0.0,
        net_count: 0,
        pipe_type,
    });
    node_pipes[from].push(pipe_idx);
    node_pipes[to].push(pipe_idx);
}

/// Intra-tile conductance: scales with PIP count (routing richness).
/// Tiles with many PIPs have high internal conductance regardless of BEL count.
fn intra_tile_conductance(n_pips: usize, resolution: usize) -> f64 {
    // Scale conductance with sqrt(PIPs) to avoid extreme ratios.
    // A tile with 1000 PIPs gets g ≈ 1.0, a tile with 0 PIPs gets the minimum.
    let g_base = ((n_pips as f64).sqrt() / 32.0).max(0.01);
    g_base * resolution as f64
}

/// Inter-tile conductance from wire count, divided across N boundary subtiles.
fn inter_tile_conductance(wire_count: usize, resolution: usize) -> f64 {
    let total_g = (wire_count as f64).max(1.0);
    total_g / resolution as f64
}

/// Estimate routing capacity between adjacent tiles in the given direction.
///
/// Uses max(wires, pips) as proxy — wires represent pass-through tracks,
/// PIPs represent switchable connections. The minimum of the two tiles
/// determines the bottleneck capacity of the connection.
fn estimate_wire_count(ctx: &Context, x: i32, y: i32, direction: Direction) -> usize {
    let tile = ctx.chipdb().tile_by_xy(x, y);
    let tt = ctx.chipdb().tile_type(tile);
    let cap = tt.wires.len().max(tt.pips.len());

    let (nx, ny) = match direction {
        Direction::East => (x + 1, y),
        Direction::South => (x, y + 1),
    };

    let neighbor_tile = ctx.chipdb().tile_by_xy(nx, ny);
    let ntt = ctx.chipdb().tile_type(neighbor_tile);
    let ncap = ntt.wires.len().max(ntt.pips.len());

    let min_cap = cap.min(ncap);
    (min_cap / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_conductance_scales_with_resolution() {
        let g2 = intra_tile_conductance(8, 2);
        let g4 = intra_tile_conductance(8, 4);
        // Higher resolution → proportionally higher per-edge conductance
        // so total tile conductance stays consistent.
        assert!(g4 > g2);
        assert!((g4 / g2 - 2.0).abs() < 0.01);
    }

    #[test]
    fn inter_conductance_splits_across_boundary() {
        let g1 = inter_tile_conductance(100, 1);
        let g2 = inter_tile_conductance(100, 2);
        // Each boundary edge gets half the total conductance at N=2.
        assert!((g1 / g2 - 2.0).abs() < 0.01);
    }

    #[test]
    fn empty_tile_has_passthrough() {
        let g = intra_tile_conductance(0, 2);
        assert!(g > 0.0);
    }
}

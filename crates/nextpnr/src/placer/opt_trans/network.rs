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
use rayon::prelude::*;

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
    /// Cell density: number of cells near this pipe's endpoints.
    pub cell_density: f64,
    /// Effective conductance used in the current Laplacian (updated each iteration).
    pub eff_conductance: f64,
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
    /// Coarsening factor: C×C tiles grouped into one node.
    pub coarsen: usize,
}

impl PipeNetwork {
    /// Build a network at the given resolution scale.
    ///
    /// `scale` controls grid granularity:
    ///   - 0.0 → 1×1 grid (whole chip = 1 node)
    ///   - 0.5 → coarsened: ~74×105 nodes (groups of 2×2 tiles)
    ///   - 1.0 → tile-level: 148×209 nodes (one per tile)
    ///   - 2.0 → 2×2 subtiles per tile: 296×418 nodes
    ///
    /// For scale < 1.0: coarsen = max(1, round(1/scale)), groups C×C tiles.
    /// For scale >= 1.0: subtile_resolution = round(scale), N×N subtiles per tile.
    ///
    /// Cell positions are always in tile coordinates.
    pub fn from_context(ctx: &Context, scale: f64) -> Self {
        let full_w = ctx.chipdb().width();
        let full_h = ctx.chipdb().height();

        let (coarsen, resolution) = if scale < 1.0 {
            let c = if scale <= 0.0 {
                full_w.max(full_h) as usize // 1×1 grid
            } else {
                (1.0 / scale).round().max(1.0) as usize
            };
            (c, 1)
        } else {
            (1, scale.round().max(1.0) as usize)
        };

        let w = ((full_w as usize + coarsen - 1) / coarsen) as i32;
        let h = ((full_h as usize + coarsen - 1) / coarsen) as i32;
        let n_coarse = (w * h) as usize;
        let n_per_tile = 1; // one node per coarse cell

        let x0 = 0;
        let y0 = 0;

        // Create nodes: one per coarse cell.
        let mut nodes = Vec::with_capacity(n_coarse);
        for cy in 0..h {
            for cx in 0..w {
                nodes.push(Node {
                    tile_x: cx,
                    tile_y: cy,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                });
            }
        }

        let total_nodes = nodes.len();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); total_nodes];

        // Helper: node index for coarse cell (cx, cy).
        let idx = |cx: i32, cy: i32, _sx: usize, _sy: usize| -> usize {
            (cy * w + cx) as usize
        };

        // 1. Aggregate tile properties into coarse cells and build connectivity.
        let mut zero_bel_tiles = 0usize;
        let mut total_bels = 0usize;

        // Aggregate BEL count and routing capacity per coarse cell.
        let mut coarse_bels = vec![0usize; n_coarse];
        let mut coarse_routing = vec![0usize; n_coarse];
        for fy in 0..full_h {
            for fx in 0..full_w {
                let cx = (fx as usize / coarsen) as i32;
                let cy = (fy as usize / coarsen) as i32;
                let ci = (cy * w + cx) as usize;

                let tile = ctx.chipdb().tile_by_xy(fx, fy);
                let tt = ctx.chipdb().tile_type(tile);
                let n_bels = tt.bels.len();
                if n_bels == 0 { zero_bel_tiles += 1; }
                total_bels += n_bels;
                coarse_bels[ci] += n_bels;
                coarse_routing[ci] += tt.wires.len().max(tt.pips.len());
            }
        }

        // Build intra-cell and inter-cell pipes on the coarse grid.
        // Each coarse cell is a single node — no intra-cell pipes needed.
        // Coarse cell capacity = sum of BELs in the C×C block.
        for ci in 0..n_coarse {
            // No intra-cell pipes for coarsen > 1 (single node per coarse cell).
            // Capacity is set per-pipe on inter-cell edges below.
            let _ = coarse_bels[ci]; // used for capacity below
        }

        // 2. Inter-cell pipes on the coarse grid (4-connected).
        for cy in 0..h {
            for cx in 0..w {
                let ci = idx(cx, cy, 0, 0);
                let cap_here = coarse_bels[ci] as f64;

                // East neighbor.
                if cx + 1 < w {
                    let ni = idx(cx + 1, cy, 0, 0);
                    let mut total_wires = 0usize;
                    let boundary_x = ((cx + 1) as usize * coarsen).min(full_w as usize) as i32 - 1;
                    for fy in (cy as usize * coarsen)..((cy as usize + 1) * coarsen).min(full_h as usize) {
                        total_wires += estimate_wire_count(ctx, boundary_x, fy as i32, Direction::East);
                    }
                    let g = inter_tile_conductance(total_wires, 1);
                    let cap = (cap_here + coarse_bels[ni] as f64) / 2.0;
                    add_pipe(&mut pipes, &mut node_pipes, ci, ni, 1.0 / g, cap.max(1.0), PipeType::InterTile(Direction::East));
                }

                // South neighbor.
                if cy + 1 < h {
                    let ni = idx(cx, cy + 1, 0, 0);
                    let mut total_wires = 0usize;
                    let boundary_y = ((cy + 1) as usize * coarsen).min(full_h as usize) as i32 - 1;
                    for fx in (cx as usize * coarsen)..((cx as usize + 1) * coarsen).min(full_w as usize) {
                        total_wires += estimate_wire_count(ctx, fx as i32, boundary_y, Direction::South);
                    }
                    let g = inter_tile_conductance(total_wires, 1);
                    let cap = (cap_here + coarse_bels[ni] as f64) / 2.0;
                    add_pipe(&mut pipes, &mut node_pipes, ci, ni, 1.0 / g, cap.max(1.0), PipeType::InterTile(Direction::South));
                }
            }
        }

        eprintln!(
            "  network: {}x{} (coarsen={}) = {} nodes, {} pipes, {} zero-BEL, {} total BELs",
            w, h, coarsen, total_nodes, pipes.len(), zero_bel_tiles, total_bels,
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
            coarsen,
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
    /// Convert tile coordinate to network coordinate (accounts for coarsening).
    #[inline]
    pub fn tile_to_net(&self, tile_coord: f64) -> f64 {
        tile_coord / self.coarsen as f64
    }

    /// Convert network coordinate back to tile coordinate.
    #[inline]
    pub fn net_to_tile(&self, net_coord: f64) -> f64 {
        net_coord * self.coarsen as f64
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of pipes in the network.
    pub fn num_pipes(&self) -> usize {
        self.pipes.len()
    }

    /// Reset all dynamic state (pressures, flows, net counts).
    pub fn reset(&mut self) {
        self.nodes.par_iter_mut().for_each(|node| {
            node.pressure = 0.0;
        });
        self.pipes.par_iter_mut().for_each(|p| {
            p.flow = 0.0;
            p.net_count = 0;
            p.cell_density = 0.0;
        });
    }

    /// Maximum utilization ratio |flow|/capacity across all pipes.
    pub fn max_utilization(&self) -> f64 {
        self.pipes
            .par_iter()
            .filter(|p| p.capacity > 0.0)
            .map(|p| p.flow.abs() / p.capacity)
            .reduce(|| 0.0, f64::max)
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
        cell_density: 0.0,
        eff_conductance: 1.0 / base_resistance.max(1e-12),
        pipe_type,
    });
    node_pipes[from].push(pipe_idx);
    node_pipes[to].push(pipe_idx);
}

/// Intra-tile conductance: log-normalized.
///
/// Raw PIP/wire counts create 130x conductance ratios between adjacent tiles,
/// producing artificial pressure barriers. Using log compresses the range
/// while preserving the relative structure (routing-rich tiles still conduct
/// more, but the ratio is ~3x instead of 130x).
fn intra_tile_conductance(n_pips: usize, resolution: usize) -> f64 {
    let g_base = (1.0 + n_pips as f64).ln().max(0.1);
    g_base * resolution as f64
}

/// Inter-tile conductance: log-normalized from wire count.
fn inter_tile_conductance(wire_count: usize, resolution: usize) -> f64 {
    let g = (1.0 + wire_count as f64).ln().max(0.1);
    g * resolution as f64
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

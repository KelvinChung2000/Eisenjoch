//! Tile-level network model for FPGA placement (Beckmann formulation).
//!
//! One node per tile (or per coarsened group). Bilinear interpolation across
//! tile boundaries provides smooth gradients for the optimizer.
//!
//! Connections:
//! - Inter-tile: adjacent tiles connected across tile edges (4-connected)
//! - Long-range: multi-tile span pipes from chipdb routing analysis

use crate::chipdb::ChipDb;
use crate::context::Context;
use crate::read_packed;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Direction of an inter-tile pipe between two adjacent tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    East,
    South,
}

/// Pipe type: inter-tile or long-range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeType {
    InterTile(Direction),
    /// Long-range pipe spanning multiple coarse cells (dx, dy in coarse coords).
    LongRange {
        dx: i32,
        dy: i32,
    },
}

/// A node in the network (one per tile or coarsened group).
#[derive(Debug, Clone)]
pub struct Node {
    /// Tile coordinate.
    pub tile_x: i32,
    pub tile_y: i32,
    /// Diagnostic pressure-like field written from path costs.
    pub pressure: f64,
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
    /// Diagnostic flow-like field derived from the current effective conductance.
    pub flow: f64,
    /// Number of distinct nets using this pipe (for interference).
    pub net_count: u32,
    /// Raw continuous occupancy projected from the unified pin-demand field.
    pub raw_cell_density: f64,
    /// Normalized occupancy used by the passive congestion law.
    pub cell_density: f64,
    /// Effective conductance used in the current Laplacian (updated each iteration).
    pub eff_conductance: f64,
    pub pipe_type: PipeType,
}

/// The pipe network over the FPGA tile grid.
pub struct PipeNetwork {
    pub nodes: Vec<Node>,
    pub pipes: Vec<Pipe>,
    pub node_pipes: Vec<Vec<usize>>,
    pub pipe_lookup: FxHashMap<u64, usize>,
    /// Tile grid dimensions.
    pub width: i32,
    pub height: i32,
    /// Grid origin offset: the virtual grid starts at (x0, y0) in physical coordinates.
    pub x0: i32,
    pub y0: i32,
    /// Number of tiles with zero BELs (routing-only, BRAM, etc.).
    pub zero_bel_tiles: usize,
    /// Total BEL count across all tiles.
    pub total_bels: usize,
    /// Coarsening factor: C×C tiles grouped into one node.
    pub coarsen: usize,
}

impl PipeNetwork {
    #[inline]
    pub fn scale(&self) -> f64 {
        if self.coarsen > 1 {
            1.0 / self.coarsen as f64
        } else {
            1.0
        }
    }

    /// Build a network at the given resolution scale.
    ///
    /// `scale` controls grid granularity:
    ///   - 0.0 → 1×1 grid (whole chip = 1 node)
    ///   - 0.5 → coarsened: ~74×105 nodes (groups of 2×2 tiles)
    ///   - 1.0 → tile-level: 148×209 nodes (one per tile)
    ///
    /// For scale < 1.0: coarsen = max(1, round(1/scale)), groups C×C tiles.
    /// For scale >= 1.0: one node per tile.
    ///
    /// Cell positions are always in tile coordinates.
    pub fn from_context(ctx: &Context, scale: f64) -> Self {
        let full_w = ctx.chipdb().width();
        let full_h = ctx.chipdb().height();

        let coarsen = if scale < 1.0 {
            if scale <= 0.0 {
                full_w.max(full_h) as usize // 1×1 grid
            } else {
                (1.0 / scale).round().max(1.0) as usize
            }
        } else {
            1
        };

        let w = ((full_w as usize + coarsen - 1) / coarsen) as i32;
        let h = ((full_h as usize + coarsen - 1) / coarsen) as i32;
        let n_coarse = (w * h) as usize;

        let x0 = 0;
        let y0 = 0;

        // Create nodes: one per coarse cell.
        let mut nodes = Vec::with_capacity(n_coarse);
        for cy in 0..h {
            for cx in 0..w {
                nodes.push(Node {
                    tile_x: cx,
                    tile_y: cy,
                    pressure: 0.0,
                });
            }
        }

        let total_nodes = nodes.len();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); total_nodes];

        // Helper: node index for coarse cell (cx, cy).
        let idx = |cx: i32, cy: i32| -> usize { (cy * w + cx) as usize };

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
                if n_bels == 0 {
                    zero_bel_tiles += 1;
                }
                total_bels += n_bels;
                coarse_bels[ci] += n_bels;
                coarse_routing[ci] += tt.wires.len().max(tt.pips.len());
            }
        }

        // Inter-tile pipes on the coarse grid (4-connected).
        for cy in 0..h {
            for cx in 0..w {
                // East neighbor.
                if cx + 1 < w {
                    let mut total_wires = 0usize;
                    let boundary_x = ((cx + 1) as usize * coarsen).min(full_w as usize) as i32 - 1;
                    for fy in
                        (cy as usize * coarsen)..((cy as usize + 1) * coarsen).min(full_h as usize)
                    {
                        total_wires +=
                            estimate_wire_count(ctx, boundary_x, fy as i32, Direction::East);
                    }
                    let g = inter_tile_conductance(total_wires);
                    let cap = (total_wires as f64).max(1.0);
                    add_pipe(
                        &mut pipes,
                        &mut node_pipes,
                        idx(cx, cy),
                        idx(cx + 1, cy),
                        1.0 / g,
                        cap,
                        PipeType::InterTile(Direction::East),
                    );
                }

                // South neighbor.
                if cy + 1 < h {
                    let mut total_wires = 0usize;
                    let boundary_y = ((cy + 1) as usize * coarsen).min(full_h as usize) as i32 - 1;
                    for fx in
                        (cx as usize * coarsen)..((cx as usize + 1) * coarsen).min(full_w as usize)
                    {
                        total_wires +=
                            estimate_wire_count(ctx, fx as i32, boundary_y, Direction::South);
                    }
                    let g = inter_tile_conductance(total_wires);
                    let cap = (total_wires as f64).max(1.0);
                    add_pipe(
                        &mut pipes,
                        &mut node_pipes,
                        idx(cx, cy),
                        idx(cx, cy + 1),
                        1.0 / g,
                        cap,
                        PipeType::InterTile(Direction::South),
                    );
                }
            }
        }

        // 3. Long-range pipes from chipdb routing node analysis.
        // Histograms are in raw tile coordinates; we convert to coarse coords here.
        let span_histograms = build_span_histograms(ctx.chipdb());
        let pre_long_range = pipes.len();

        // Deduplicate: multiple raw (dx,dy) entries may map to the same coarse cell pair.
        // Aggregate wire counts for each unique coarse delta.
        let coarsen_i = coarsen as i32;
        for cy in 0..h {
            for cx in 0..w {
                // Use center tile of the coarse cell as representative.
                let repr_fx = (cx as usize * coarsen + coarsen / 2).min(full_w as usize - 1) as i32;
                let repr_fy = (cy as usize * coarsen + coarsen / 2).min(full_h as usize - 1) as i32;
                let repr_tile = ctx.chipdb().tile_by_xy(repr_fx, repr_fy);
                let tt_idx = ctx.chipdb().tile_type_index(repr_tile) as usize;

                if let Some(hist) = span_histograms.get(tt_idx) {
                    // Aggregate raw entries into coarse deltas.
                    let mut coarse_agg: FxHashMap<(i32, i32), usize> = FxHashMap::default();
                    for (&(raw_dx, raw_dy), &wire_count) in hist {
                        let cdx = raw_dx / coarsen_i;
                        let cdy = raw_dy / coarsen_i;
                        let coarse_span = cdx.abs() + cdy.abs();
                        if coarse_span < 2 {
                            continue;
                        }
                        *coarse_agg.entry((cdx, cdy)).or_insert(0) += wire_count;
                    }

                    for ((cdx, cdy), wire_count) in coarse_agg {
                        let nx = cx + cdx;
                        let ny = cy + cdy;
                        if nx < 0 || nx >= w || ny < 0 || ny >= h {
                            continue;
                        }
                        let span = (cdx.abs() + cdy.abs()) as usize;
                        let g = long_range_conductance(wire_count, span);
                        let cap = (wire_count as f64).max(1.0);
                        add_pipe(
                            &mut pipes,
                            &mut node_pipes,
                            idx(cx, cy),
                            idx(nx, ny),
                            1.0 / g,
                            cap,
                            PipeType::LongRange { dx: cdx, dy: cdy },
                        );
                    }
                }
            }
        }

        let n_long_range = pipes.len() - pre_long_range;
        log::debug!(
            "network: {}x{} (coarsen={}) = {} nodes, {} pipes ({} long-range)",
            w,
            h,
            coarsen,
            total_nodes,
            pipes.len(),
            n_long_range,
        );
        let pipe_lookup = build_pipe_lookup(&pipes);

        Self {
            nodes,
            pipes,
            node_pipes,
            pipe_lookup,
            width: w,
            height: h,
            x0,
            y0,
            zero_bel_tiles,
            total_bels,
            coarsen,
        }
    }

    /// Index of node at tile (tx, ty).
    #[inline]
    pub fn node_index(&self, tx: i32, ty: i32) -> usize {
        (ty * self.width + tx) as usize
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

    /// Build a 1D chain network for a single axis.
    ///
    /// `axis = 'x'`: chain of W nodes (one per column), connected left-to-right.
    ///   Grid shape: width=W, height=1. Positions mapped as cell_x → node index.
    ///
    /// `axis = 'y'`: chain of H nodes (one per row), connected top-to-bottom.
    ///   Grid shape: width=1, height=H. Positions mapped as cell_y → node index.
    ///
    /// Conductance between adjacent nodes = sum of wire counts across the boundary.
    /// Capacity = sum of BELs in each column (x) or row (y).
    ///
    /// On a 1D chain, effective resistance is LINEAR in distance (not logarithmic),
    /// producing a constant gradient that matches HPWL equilibrium.
    pub fn from_context_1d(ctx: &Context, axis: char, _scale: f64) -> Self {
        let full_w = ctx.chipdb().width();
        let full_h = ctx.chipdb().height();

        let (chain_len, grid_w, grid_h) = match axis {
            'x' => (full_w as usize, full_w, 1),
            'y' => (full_h as usize, 1, full_h),
            _ => panic!("axis must be 'x' or 'y'"),
        };

        // Create nodes: one per column (x-chain) or one per row (y-chain).
        let mut nodes = Vec::with_capacity(chain_len);
        for i in 0..chain_len {
            let (tx, ty) = match axis {
                'x' => (i as i32, 0),
                _ => (0, i as i32),
            };
            nodes.push(Node {
                tile_x: tx,
                tile_y: ty,
                pressure: 0.0,
            });
        }

        let total_nodes = nodes.len();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); total_nodes];

        // Aggregate BEL counts per column/row and build chain connectivity.
        let mut bels_per_node = vec![0usize; chain_len];
        let mut zero_bel_tiles = 0usize;

        for fy in 0..full_h {
            for fx in 0..full_w {
                let tile = ctx.chipdb().tile_by_xy(fx, fy);
                let tt = ctx.chipdb().tile_type(tile);
                let n_bels = tt.bels.len();
                if n_bels == 0 {
                    zero_bel_tiles += 1;
                }
                let node_idx = match axis {
                    'x' => fx as usize,
                    _ => fy as usize,
                };
                bels_per_node[node_idx] += n_bels;
            }
        }

        // Build chain pipes: node[i] <-> node[i+1].
        // Use UNIT conductance (g=1, R=1) for all pipes. On a 1D chain with
        // unit conductance, effective resistance between nodes i and j is |i-j|,
        // giving a constant pressure gradient -- exactly matching HPWL equilibrium.
        // Using physical wire counts would make the chain too conductive (all wires
        // across the orthogonal dimension are summed), producing tiny gradients.
        for i in 0..(chain_len - 1) {
            let g = 1.0;
            let cap = (bels_per_node[i] as f64 + bels_per_node[i + 1] as f64) / 2.0;
            let direction = match axis {
                'x' => Direction::East,
                _ => Direction::South,
            };
            add_pipe(
                &mut pipes,
                &mut node_pipes,
                i,
                i + 1,
                1.0 / g,
                cap.max(1.0),
                PipeType::InterTile(direction),
            );
        }

        log::debug!(
            "network_1d: axis={} {}x{} = {} nodes, {} pipes",
            axis,
            grid_w,
            grid_h,
            total_nodes,
            pipes.len(),
        );
        let pipe_lookup = build_pipe_lookup(&pipes);

        Self {
            nodes,
            pipes,
            node_pipes,
            pipe_lookup,
            width: grid_w,
            height: grid_h,
            x0: 0,
            y0: 0,
            zero_bel_tiles,
            total_bels: bels_per_node.iter().sum(),
            coarsen: 1,
        }
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
        raw_cell_density: 0.0,
        cell_density: 0.0,
        eff_conductance: 1.0 / base_resistance.max(1e-12),
        pipe_type,
    });
    node_pipes[from].push(pipe_idx);
    node_pipes[to].push(pipe_idx);
}

#[inline]
fn pipe_key(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

fn build_pipe_lookup(pipes: &[Pipe]) -> FxHashMap<u64, usize> {
    pipes
        .iter()
        .enumerate()
        .map(|(idx, pipe)| (pipe_key(pipe.from, pipe.to), idx))
        .collect()
}

/// Inter-tile conductance: log-normalized from wire count.
fn inter_tile_conductance(wire_count: usize) -> f64 {
    (1.0 + wire_count as f64).ln().max(0.1)
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

/// Long-range pipe conductance: log-normalized, resistance scales with Manhattan span.
fn long_range_conductance(wire_count: usize, span: usize) -> f64 {
    let g = (1.0 + wire_count as f64).ln().max(0.1);
    g / span.max(1) as f64
}

/// Per tile-type histogram: maps raw (dx, dy) in tile coords -> wire_count for long-range nodes.
type SpanHistogram = FxHashMap<(i32, i32), usize>;

/// Analyze chipdb routing node shapes to build per-tile-type span histograms.
///
/// For each unique tile type, picks one representative tile, iterates its root-node
/// wires, and counts how many routing nodes span each raw (dx, dy) distance.
/// Returns a Vec indexed by tile_type_index. The histogram is in raw tile coordinates;
/// conversion to coarse coordinates happens at pipe creation time.
fn build_span_histograms(chipdb: &ChipDb) -> Vec<SpanHistogram> {
    let num_tt = chipdb.num_tile_types();
    let num_tiles = chipdb.num_tiles();
    let mut histograms = vec![SpanHistogram::default(); num_tt];

    // Find one representative tile per tile type.
    let mut repr_tile: Vec<Option<i32>> = vec![None; num_tt];
    for tile in 0..num_tiles {
        let tt_idx = chipdb.tile_type_index(tile) as usize;
        if repr_tile[tt_idx].is_none() {
            repr_tile[tt_idx] = Some(tile);
        }
    }

    for tt_idx in 0..num_tt {
        let Some(tile) = repr_tile[tt_idx] else {
            continue;
        };
        let tt = chipdb.tile_type(tile);
        let n_wires = tt.wires.len();

        for wire_idx in 0..n_wires {
            let Some(ns) = chipdb.wire_node_shape(tile, wire_idx) else {
                continue;
            };
            let tile_wires = ns.tile_wires.get();
            for tw in tile_wires {
                let dx: i16 = unsafe { read_packed!(*tw, dx) };
                let dy: i16 = unsafe { read_packed!(*tw, dy) };
                let raw_span = (dx as i32).abs() + (dy as i32).abs();
                // Canonical half: only count positive direction to avoid double-counting.
                if dx < 0 || (dx == 0 && dy <= 0) {
                    continue;
                }
                // Skip same-tile (span 0) and nearest-neighbor (span 1).
                if raw_span <= 1 {
                    continue;
                }
                // Store raw tile coordinates; coarse conversion happens at pipe creation.
                *histograms[tt_idx]
                    .entry((dx as i32, dy as i32))
                    .or_insert(0) += 1;
            }
        }
    }

    // Log summary.
    let total_entries: usize = histograms.iter().map(|h| h.len()).sum();
    let max_span = histograms
        .iter()
        .flat_map(|h| h.keys())
        .map(|(dx, dy)| dx.abs() + dy.abs())
        .max()
        .unwrap_or(0);
    let total_wires: usize = histograms.iter().flat_map(|h| h.values()).sum();
    log::debug!(
        "span_histograms: {} tile types with long-range entries, {} unique (dx,dy) pairs, \
         max span={}, total long-range wires={}",
        histograms.iter().filter(|h| !h.is_empty()).count(),
        total_entries,
        max_span,
        total_wires,
    );

    histograms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inter_conductance_positive() {
        let g = inter_tile_conductance(100);
        assert!(g > 0.0);
    }

    #[test]
    fn inter_conductance_zero_wires() {
        let g = inter_tile_conductance(0);
        assert!(g >= 0.1);
    }
}

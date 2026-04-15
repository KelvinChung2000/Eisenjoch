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

use super::config::OptTransPlacerCfg;

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
    pub fn from_context(ctx: &Context, scale: f64, cfg: &OptTransPlacerCfg) -> Self {
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

        // Build span histograms first — one source of truth for both span-1
        // (inter-tile) and span-N (long-range) pipe capacities.
        let span_histograms = build_span_histograms(ctx.chipdb());

        // Unified pipe creation: one aggregation pass per coarse cell sums
        // `span_histograms` entries across ALL fine tiles inside the cell,
        // keyed by coarse (cdx, cdy). Each unique delta becomes one directed
        // outgoing pipe whose `capacity` reflects the real wires of that span
        // that ORIGINATE in the source coarse cell. A wire of delta (+dx, +dy)
        // at the source is physically distinct from a wire of delta (-dx, -dy)
        // at the destination, so both must be emitted to preserve capacity.
        //
        //   base_resistance = sqrt(|cdx| + |cdy|)   (r1 = 1; same model for all spans)
        //   capacity        = summed wire_count
        let coarsen_i = coarsen as i32;
        let mut n_span1 = 0usize;
        let mut n_long_range = 0usize;
        for cy in 0..h {
            for cx in 0..w {
                let mut coarse_agg: FxHashMap<(i32, i32), usize> = FxHashMap::default();
                let mut fallback_east = 0usize;
                let mut fallback_south = 0usize;
                for fy in (cy as usize * coarsen)
                    ..((cy as usize + 1) * coarsen).min(full_h as usize)
                {
                    for fx in (cx as usize * coarsen)
                        ..((cx as usize + 1) * coarsen).min(full_w as usize)
                    {
                        let tile = ctx.chipdb().tile_by_xy(fx as i32, fy as i32);
                        let tt_idx = ctx.chipdb().tile_type_index(tile) as usize;
                        if let Some(hist) = span_histograms.get(tt_idx) {
                            for (&(raw_dx, raw_dy), &wire_count) in hist {
                                let cdx = raw_dx / coarsen_i;
                                let cdy = raw_dy / coarsen_i;
                                if cdx == 0 && cdy == 0 {
                                    continue;
                                }
                                *coarse_agg.entry((cdx, cdy)).or_insert(0) += wire_count;
                            }
                        }
                        // Span-1 fallback: accumulate only for fine tiles that
                        // actually have an east/south neighbor (avoids OOB in
                        // estimate_wire_count).
                        if (fx as i32) + 1 < full_w {
                            fallback_east +=
                                estimate_wire_count(ctx, fx as i32, fy as i32, Direction::East);
                        }
                        if (fy as i32) + 1 < full_h {
                            fallback_south +=
                                estimate_wire_count(ctx, fx as i32, fy as i32, Direction::South);
                        }
                    }
                }
                if cx + 1 < w && !coarse_agg.contains_key(&(1, 0)) && fallback_east > 0 {
                    coarse_agg.insert((1, 0), fallback_east.max(1));
                }
                if cy + 1 < h && !coarse_agg.contains_key(&(0, 1)) && fallback_south > 0 {
                    coarse_agg.insert((0, 1), fallback_south.max(1));
                }

                for ((cdx, cdy), wire_count) in coarse_agg {
                    let nx = cx + cdx;
                    let ny = cy + cdy;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        continue;
                    }
                    let span = (cdx.abs() + cdy.abs()) as usize;
                    let base = (span as f64).sqrt();
                    let cap = (wire_count as f64).max(1.0);
                    let pipe_type = if span == 1 {
                        n_span1 += 1;
                        let dir = if cdx.abs() >= cdy.abs() {
                            Direction::East
                        } else {
                            Direction::South
                        };
                        PipeType::InterTile(dir)
                    } else {
                        n_long_range += 1;
                        PipeType::LongRange { dx: cdx, dy: cdy }
                    };
                    add_pipe(
                        &mut pipes,
                        &mut node_pipes,
                        idx(cx, cy),
                        idx(nx, ny),
                        base,
                        cap,
                        pipe_type,
                    );
                }
            }
        }
        let _ = n_span1;
        log::debug!(
            "network: {}x{} (coarsen={}) = {} nodes, {} pipes ({} long-range)",
            w,
            h,
            coarsen,
            total_nodes,
            pipes.len(),
            n_long_range,
        );

        // Per-span capacity summary: gives a clear read on what wire resources
        // are actually modeled for each span bucket. Useful to sanity-check
        // whether the span-1 pipes are being sized from the right histograms.
        {
            use rustc_hash::FxHashMap as _Map;
            let mut by_span: _Map<i32, (usize, f64, f64, f64, f64)> = _Map::default();
            // (count, sum_cap, min_cap, max_cap, sum_cap_sq)
            for pipe in &pipes {
                let span = match pipe.pipe_type {
                    PipeType::InterTile(_) => 1i32,
                    PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                };
                let e = by_span
                    .entry(span)
                    .or_insert((0usize, 0.0, f64::INFINITY, 0.0, 0.0));
                e.0 += 1;
                e.1 += pipe.capacity;
                if pipe.capacity < e.2 { e.2 = pipe.capacity; }
                if pipe.capacity > e.3 { e.3 = pipe.capacity; }
                e.4 += pipe.capacity * pipe.capacity;
            }
            let mut spans: Vec<i32> = by_span.keys().copied().collect();
            spans.sort();
            eprintln!("  Pipe capacity by span:");
            for s in spans {
                let (n, sum, mn, mx, _) = by_span[&s];
                let avg = sum / n as f64;
                eprintln!(
                    "    span={:3}: {} pipes, cap min={:.0} avg={:.1} max={:.0}",
                    s, n, mn, avg, mx,
                );
            }

            // Location of thin span-1 wires (cap<=2): tile-type pair histogram.
            // Answers "where are these bottlenecks on the device?"
            let mut thin_by_tt: _Map<(String, String), usize> = _Map::default();
            let mut n_thin = 0usize;
            let coarsen_i32 = coarsen as i32;
            for pipe in &pipes {
                let is_span1 = matches!(pipe.pipe_type, PipeType::InterTile(_));
                if !is_span1 || pipe.capacity > 2.0 {
                    continue;
                }
                n_thin += 1;
                let fn_ = &nodes[pipe.from];
                let tn_ = &nodes[pipe.to];
                let ffx = fn_.tile_x * coarsen_i32;
                let ffy = fn_.tile_y * coarsen_i32;
                let tfx = tn_.tile_x * coarsen_i32;
                let tfy = tn_.tile_y * coarsen_i32;
                if ffx < 0 || ffx >= full_w || ffy < 0 || ffy >= full_h {
                    continue;
                }
                if tfx < 0 || tfx >= full_w || tfy < 0 || tfy >= full_h {
                    continue;
                }
                let ft = ctx.chipdb().tile_by_xy(ffx, ffy);
                let tt = ctx.chipdb().tile_by_xy(tfx, tfy);
                let fname = ctx.chipdb().tile_type_name(ft).to_string();
                let tname = ctx.chipdb().tile_type_name(tt).to_string();
                // Normalize pair (a,b) with a<=b for deduplication.
                let key = if fname <= tname {
                    (fname, tname)
                } else {
                    (tname, fname)
                };
                *thin_by_tt.entry(key).or_insert(0) += 1;
            }
            let mut items: Vec<_> = thin_by_tt.into_iter().collect();
            items.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!(
                "  Thin span-1 pipes (cap<=2): {} total, by tile-type pair (top 20):",
                n_thin
            );
            for ((a, b), cnt) in items.iter().take(20) {
                eprintln!("    {} <-> {}: {}", a, b, cnt);
            }
        }

        // Unified sqrt pricing: base = r1 × sqrt(span) with r1 = 1. Set at
        // pipe-creation time for both InterTile (span=1, base=1) and LongRange
        // (span=N, base=sqrt(N)). Wire count is already baked into `capacity`;
        // congestion-dependent R_eff rides on top via `ResistanceModel`.
        eprintln!(
            "  Pipe cost: sqrt model (r1=1), span-1={:.3}, span-12={:.3}, 2×span-6={:.3}, 12×span-1={:.3}",
            1.0,
            12.0_f64.sqrt(),
            2.0 * 6.0_f64.sqrt(),
            12.0,
        );

        // Per-span scarcity scaling: multiply each pipe's base_resistance by
        // a linear penalty factor derived from per-span median capacity. Narrow
        // pipes pay extra; pipes at or above their span's median capacity pay
        // nothing. Keeps the sqrt(span) shortcut bias intact for legitimate
        // long-range wires while discouraging the placer from routing data
        // signals through IOB / CLK / NULL boundary pipes (cap 1-6 span-1 wires).
        {
            use rustc_hash::FxHashMap as _Map2;
            let mut caps_by_span: _Map2<i32, Vec<f64>> = _Map2::default();
            for pipe in &pipes {
                let span = match pipe.pipe_type {
                    PipeType::InterTile(_) => 1i32,
                    PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                };
                caps_by_span.entry(span).or_default().push(pipe.capacity);
            }
            let mut ref_cap_by_span: _Map2<i32, f64> = _Map2::default();
            let mut sorted_spans: Vec<i32> = caps_by_span.keys().copied().collect();
            sorted_spans.sort();
            for s in &sorted_spans {
                let v = caps_by_span.get_mut(s).unwrap();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = v.len() / 2;
                let median = if v.is_empty() { 1.0 } else { v[mid] };
                ref_cap_by_span.insert(*s, median.max(1.0));
            }

            let k = cfg.scarcity_k;
            let mut n_factor_1 = 0usize;
            let mut n_factor_1_2 = 0usize;
            let mut n_factor_2_10 = 0usize;
            let mut n_factor_gt10 = 0usize;
            for pipe in &mut pipes {
                let span = match pipe.pipe_type {
                    PipeType::InterTile(_) => 1i32,
                    PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                };
                let ref_cap = ref_cap_by_span.get(&span).copied().unwrap_or(1.0);
                let deficit = (1.0 - pipe.capacity / ref_cap).max(0.0);
                let factor = 1.0 + k * deficit;
                pipe.base_resistance *= factor;
                pipe.eff_conductance = 1.0 / pipe.base_resistance.max(1e-12);
                if factor <= 1.0 + 1e-9 {
                    n_factor_1 += 1;
                } else if factor < 2.0 {
                    n_factor_1_2 += 1;
                } else if factor < 10.0 {
                    n_factor_2_10 += 1;
                } else {
                    n_factor_gt10 += 1;
                }
            }
            eprintln!(
                "  Scarcity scaling: k={:.1}, factor distribution: 1x={} 1-2x={} 2-10x={} >10x={}",
                k, n_factor_1, n_factor_1_2, n_factor_2_10, n_factor_gt10,
            );
            let mut shown = 0;
            for s in sorted_spans.iter() {
                if shown >= 10 {
                    break;
                }
                eprintln!(
                    "    ref_cap(span={}) = {:.1}",
                    s, ref_cap_by_span[s]
                );
                shown += 1;
            }
        }

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

/// Estimate routing capacity between adjacent tiles in the given direction.
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
    if min_cap == 0 {
        return 0;
    }
    (min_cap / 4).max(1)
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
                // Skip same-tile wires (span 0).
                if raw_span <= 0 {
                    continue;
                }
                // `hist[(dx,dy)]` then counts wires that have a tile_wire at
                // offset (dx,dy) AND at (0,0) — i.e. wires tappable at T and
                // T+(dx,dy). That is the number of wires that can physically
                // serve a pipe of delta (dx,dy), whether the wire is a span-1
                // connecting exactly those two tiles or a longer wire that
                // can be tapped via intermediate pips. Keep this aggregation.
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


//! Tile-level network model for FPGA placement (Beckmann formulation).
//!
//! One node per tile (or per coarsened group). Bilinear interpolation across
//! tile boundaries provides smooth gradients for the optimizer.
//!
//! Connections:
//! - Inter-tile: adjacent tiles connected across tile edges (4-connected)
//! - Long-range: multi-tile span pipes from chipdb routing analysis

use crate::chipdb::{ChipDb, TileTypeTemplate};
use crate::context::Context;
use crate::read_packed;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::tile_cache::SpanCostTable;

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
    /// Expected net usage assigned to this pipe.
    pub net_count: f64,
    /// Raw continuous occupancy projected from the unified pin-demand field.
    pub raw_cell_density: f64,
    /// Normalized occupancy used by the passive congestion law.
    pub cell_density: f64,
    /// Effective conductance used in the current Laplacian (updated each iteration).
    pub eff_conductance: f64,
    pub pipe_type: PipeType,
}

/// The pipe network over the FPGA tile grid.
/// Integer-quantization scale factor for pipe costs (`int_cost = round(f64_cost * DIST_SCALE)`).
/// Chosen so 2-digit precision survives the quantization (BPR steps on r_eff
/// of 0.01 are distinguishable) and typical paths fit comfortably in u32.
pub const DIST_SCALE: f64 = 100.0;

/// Quantize a hardening-history value into the integer cost domain. Unlike
/// `pipe_costs_int` there is no `max(1)` floor: a pipe with no history must
/// add exactly zero, or every edge would silently gain a unit of cost.
#[inline]
pub fn hist_to_int(h: f64) -> u32 {
    if h <= 0.0 {
        return 0;
    }
    let scaled = (h * DIST_SCALE).round();
    if scaled >= u32::MAX as f64 {
        u32::MAX - 1
    } else {
        scaled as u32
    }
}

/// Grid-structured view over the pipe set used by the Fast Sweeping solver.
///
/// The pipe network is a regular WxH tile grid plus a sparse set of long-range
/// pipes. FSM needs O(1) access to the cardinal pipe index at each tile edge;
/// `pipe_e[y*w + x]` is the pipe from (x,y) to (x+1,y) (or `u32::MAX` if
/// absent), and `pipe_s[y*w + x]` is the pipe from (x,y) to (x,y+1).
///
/// Costs are read through `PipeNetwork::pipe_costs_int[pipe_idx]`, so BPR
/// refreshes flow through naturally without needing to rebuild this struct.
pub struct TileGrid {
    /// Pipe index for the East-going edge out of (x, y). `u32::MAX` means no
    /// such pipe (chip boundary or pruned connection). Size = `width * height`;
    /// the x = width-1 column is always `u32::MAX`.
    pub pipe_e: Vec<u32>,
    /// Pipe index for the South-going edge out of (x, y). `u32::MAX` means no
    /// such pipe. Size = `width * height`; the y = height-1 row is always
    /// `u32::MAX`.
    pub pipe_s: Vec<u32>,
    /// Indices into `PipeNetwork::pipes` of every `LongRange` pipe. FSM relaxes
    /// these as a sparse step between sweep rounds.
    pub long_pipes: Vec<u32>,
}

/// Build a `TileTypeTemplate` for every tile type in the chipdb.
///
/// Each type is represented by its first-seen tile instance; types with no
/// instance get an empty template. Empty types still occupy a slot so
/// `templates[tile_type_index(tile)]` is always in bounds.
pub(crate) fn build_tile_templates(chipdb: &ChipDb) -> Vec<TileTypeTemplate> {
    let num_tt = chipdb.num_tile_types();
    let num_tiles = chipdb.num_tiles();
    let mut repr = vec![-1i32; num_tt];
    for tile in 0..num_tiles {
        let tt = chipdb.tile_type_index(tile);
        if tt >= 0 && repr[tt as usize] < 0 {
            repr[tt as usize] = tile;
        }
    }
    let mut out = Vec::with_capacity(num_tt);
    for tt_idx in 0..num_tt {
        let rep = repr[tt_idx];
        if rep >= 0 {
            out.push(TileTypeTemplate::from_chipdb(chipdb, tt_idx as i32, rep));
        } else {
            out.push(TileTypeTemplate::empty());
        }
    }
    out
}

/// CSR-flattened adjacency view over `PipeNetwork::node_pipes`.
///
/// Built once at network construction. The hot Dijkstra/Dial loops stream
/// through this instead of the `Vec<Vec<usize>>` + `pipes[pipe_idx]` pair:
///
///   - `offsets[n+1]` gives node `n`'s directed-edge range in `neighbors` /
///     `pipe_idx`.
///   - `neighbors[e]` is the pre-resolved neighbor node, saving the 88-byte
///     random read of `pipes[pipe_idx]` just to recover `from`/`to`.
///   - `pipe_idx[e]` keeps the original pipe index so cost lookups via
///     `PipeNetwork::pipe_cost_int` and usage accumulation via
///     `record_edge_usage(pipe_idx, …)` keep working unchanged.
///
/// Memory: `2 * n_pipes * (4 + 4)` bytes (directed edges). On sv3 that's
/// ≈ 123 MB, a wash with the existing `node_pipes: Vec<Vec<usize>>` layout
/// (15.4 M usize entries) but streamed sequentially per source node.
pub struct FlatAdjacency {
    pub offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
    pub pipe_idx: Vec<u32>,
    /// Per-edge u32-quantised cost, refreshed in lockstep with
    /// `PipeNetwork::pipe_costs_int` / `SpanCostTable::costs_int`. Lives next
    /// to `neighbors` so the Dijkstra hot loop streams it sequentially instead
    /// of scattering into the pipe_costs_int array (9 MB) via `pipe_idx`.
    pub cost_int: Vec<u32>,
    /// Per-edge f32 cost for the link-likelihood weighting. f32 is sufficient
    /// at `DIST_SCALE=100` and halves the footprint vs f64 (18 MB vs 36 MB
    /// for 4.5 M edges), keeping the hot working set inside L3.
    pub cost_f: Vec<f32>,
}

impl FlatAdjacency {
    pub fn build(node_pipes: &[Vec<usize>], pipes: &[Pipe]) -> Self {
        let n_nodes = node_pipes.len();
        let total: usize = node_pipes.iter().map(|v| v.len()).sum();
        let mut offsets = Vec::with_capacity(n_nodes + 1);
        let mut neighbors = Vec::with_capacity(total);
        let mut pipe_idx = Vec::with_capacity(total);
        offsets.push(0u32);
        for (node, adj) in node_pipes.iter().enumerate() {
            for &pidx in adj {
                let pipe = &pipes[pidx];
                let neighbor = pipe.from ^ pipe.to ^ node;
                neighbors.push(neighbor as u32);
                pipe_idx.push(pidx as u32);
            }
            offsets.push(neighbors.len() as u32);
        }
        let cost_int = vec![0u32; total];
        let cost_f = vec![0.0f32; total];
        Self {
            offsets,
            neighbors,
            pipe_idx,
            cost_int,
            cost_f,
        }
    }

    pub fn empty(n_nodes: usize) -> Self {
        Self {
            offsets: vec![0u32; n_nodes + 1],
            neighbors: Vec::new(),
            pipe_idx: Vec::new(),
            cost_int: Vec::new(),
            cost_f: Vec::new(),
        }
    }

    #[inline]
    pub fn range(&self, node: usize) -> std::ops::Range<usize> {
        let s = self.offsets[node] as usize;
        let e = self.offsets[node + 1] as usize;
        s..e
    }
}

impl TileGrid {
    /// Build the cardinal / long-pipe index arrays from the final pipe list.
    pub(crate) fn build(pipes: &[Pipe], nodes: &[Node], width: i32, height: i32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;
        let mut pipe_e = vec![u32::MAX; n];
        let mut pipe_s = vec![u32::MAX; n];
        let mut long_pipes = Vec::new();

        for (idx, pipe) in pipes.iter().enumerate() {
            match pipe.pipe_type {
                PipeType::InterTile(dir) => {
                    let from = &nodes[pipe.from];
                    let to = &nodes[pipe.to];
                    let (src, dst) = if (to.tile_x + to.tile_y) > (from.tile_x + from.tile_y) {
                        (from, to)
                    } else {
                        (to, from)
                    };
                    if src.tile_x < 0 || src.tile_y < 0 {
                        continue;
                    }
                    let sx = src.tile_x as usize;
                    let sy = src.tile_y as usize;
                    if sx >= w || sy >= h {
                        continue;
                    }
                    let slot = sy * w + sx;
                    match dir {
                        Direction::East
                            if dst.tile_x == src.tile_x + 1 && dst.tile_y == src.tile_y =>
                        {
                            pipe_e[slot] = idx as u32;
                        }
                        Direction::South
                            if dst.tile_y == src.tile_y + 1 && dst.tile_x == src.tile_x =>
                        {
                            pipe_s[slot] = idx as u32;
                        }
                        // Test-only or otherwise-non-adjacent pipes fall through
                        // as long-range so FSM still relaxes them.
                        _ => long_pipes.push(idx as u32),
                    }
                }
                PipeType::LongRange { .. } => {
                    long_pipes.push(idx as u32);
                }
            }
        }

        Self {
            pipe_e,
            pipe_s,
            long_pipes,
        }
    }
}

pub struct PipeNetwork {
    pub nodes: Vec<Node>,
    pub pipes: Vec<Pipe>,
    pub node_pipes: Vec<Vec<usize>>,
    /// Per-pipe routing cost `1.0 / eff_conductance`. Refreshed alongside
    /// `eff_conductance` in `update_effective_conductance`. Hot Dijkstra loop
    /// reads this directly instead of recomputing `1.0 / g` per edge visit.
    pub pipe_costs: Vec<f64>,
    /// Per-pipe congestion **history** — PathFinder's `h` term. Grows by
    /// `step * base * overflow / eff_capacity` each outer iter for every pipe
    /// over capacity and **never decays**; that permanence is the whole point.
    /// The EMA on `net_count` smooths the *load*; this prices the *dual* of the
    /// capacity constraint, so present-cost-only limit cycles cannot re-form.
    /// Added to `pipe_cost`/`pipe_cost_int` on top of the BPR term, so it flows
    /// into the span-cost table path and the flat path alike. Zero everywhere
    /// when `hardening_step == 0.0`.
    pub pipe_history: Vec<f64>,
    /// u32-quantized copy of `pipe_costs` used by the bucket-Dial Dijkstra.
    /// `int = max(1, (pipe_costs * DIST_SCALE).round() as u32)`; the `max(1)`
    /// guarantees strictly-positive integer weights (required for Dial).
    pub pipe_costs_int: Vec<u32>,
    /// Cache-backed span-template costs. Disabled in the flat model.
    pub span_cost_table: SpanCostTable,
    /// CSR-flattened adjacency over `node_pipes`. Built once at construction;
    /// Dijkstra/Dial/FSM hot loops stream through this instead of hopping
    /// through `Vec<Vec<usize>>` + random `pipes[pipe_idx]` reads.
    pub flat_adjacency: FlatAdjacency,
    /// Per-tile-type internal PIP templates. Populated when the chipdb is
    /// available (production `from_context` path); empty in test fixtures.
    /// Indexed by `tile_type_index` returned by the chipdb.
    pub tile_templates: Arc<Vec<TileTypeTemplate>>,
    /// Grid-structured view for the Fast Sweeping solver. Derived from `pipes`
    /// at network construction; pipe costs are read live from `pipe_costs_int`
    /// so this struct does not need to be rebuilt between outer iters.
    pub tile_grid: TileGrid,
    pub pipe_lookup: FxHashMap<u64, usize>,
    /// Representative tile type for each network node.
    pub tile_type_by_node: Vec<u16>,
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
        // Helper: node index for coarse cell (cx, cy).
        let idx = |cx: i32, cy: i32| -> usize { (cy * w + cx) as usize };

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
        let mut tile_type_by_node = vec![0u16; n_coarse];
        for cy in 0..h {
            for cx in 0..w {
                let fx = ((cx as usize) * coarsen).min((full_w - 1) as usize) as i32;
                let fy = ((cy as usize) * coarsen).min((full_h - 1) as usize) as i32;
                let tile = ctx.chipdb().tile_by_xy(fx, fy);
                tile_type_by_node[idx(cx, cy)] = ctx.chipdb().tile_type_index(tile).max(0) as u16;
            }
        }

        let total_nodes = nodes.len();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); total_nodes];

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
        // A coarse cell is "truly NULL" iff every fine tile inside it has no
        // BELs AND no wires AND no pips. Such a cell represents dead silicon
        // (padding / corner gaps in the composite fabric) — the placer can
        // never land a cell there (CellValidityMask already disallows it), and
        // routing through it is meaningless. Skipping it as both pipe source
        // and destination prunes phantom edges without breaking connectivity,
        // because genuine pure-routing tiles (INT_L, HCLK, CLK_HROW) have
        // wires/pips and stay in the graph.
        let is_null_coarse = |ci: usize| coarse_bels[ci] == 0 && coarse_routing[ci] == 0;
        let n_null_coarse: usize = (0..n_coarse).filter(|&ci| is_null_coarse(ci)).count();

        let coarsen_i = coarsen as i32;
        // Pipe base-cost exponent: base = span^exp. exp=0.5 (sqrt) prices long
        // lines as a delay/resistance proxy (current default). exp=1.0 makes the
        // base cost linear in span, so the driver-star Dijkstra distance tracks
        // routed wirelength (matches an HPWL/wirelength objective like elfPlace).
        let pipe_cost_exp: f64 = std::env::var("NPNR_OT_PIPE_COST_EXP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let mut n_span1 = 0usize;
        let mut n_long_range = 0usize;
        let mut n_skipped_null_src = 0usize;
        let mut n_skipped_null_dst = 0usize;
        for cy in 0..h {
            for cx in 0..w {
                let src_ci = idx(cx, cy);
                if is_null_coarse(src_ci) {
                    n_skipped_null_src += 1;
                    continue;
                }
                let mut coarse_agg: FxHashMap<(i32, i32), usize> = FxHashMap::default();
                let mut fallback_east = 0usize;
                let mut fallback_south = 0usize;
                for fy in
                    (cy as usize * coarsen)..((cy as usize + 1) * coarsen).min(full_h as usize)
                {
                    for fx in
                        (cx as usize * coarsen)..((cx as usize + 1) * coarsen).min(full_w as usize)
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
                    let dst_ci = idx(nx, ny);
                    if is_null_coarse(dst_ci) {
                        n_skipped_null_dst += 1;
                        continue;
                    }
                    let span = (cdx.abs() + cdy.abs()) as usize;
                    let base = (span as f64).powf(pipe_cost_exp);
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
                        src_ci,
                        dst_ci,
                        base,
                        cap,
                        pipe_type,
                    );
                }
            }
        }
        let _ = n_span1;
        eprintln!(
            "  Skipped truly-null coarse cells: {} (of {} total); pruned pipes: src_null={} dst_null={}",
            n_null_coarse, n_coarse, n_skipped_null_src, n_skipped_null_dst,
        );
        log::debug!(
            "network: {}x{} (coarsen={}) = {} nodes ({} null), {} pipes ({} long-range)",
            w,
            h,
            coarsen,
            total_nodes,
            n_null_coarse,
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
                if pipe.capacity < e.2 {
                    e.2 = pipe.capacity;
                }
                if pipe.capacity > e.3 {
                    e.3 = pipe.capacity;
                }
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
            "  Pipe cost: span^{:.2} model (r1=1), span-1={:.3}, span-12={:.3}, 2×span-6={:.3}, 12×span-1={:.3}",
            pipe_cost_exp,
            1.0_f64.powf(pipe_cost_exp),
            12.0_f64.powf(pipe_cost_exp),
            2.0 * 6.0_f64.powf(pipe_cost_exp),
            12.0 * 1.0_f64.powf(pipe_cost_exp),
        );

        let pipe_lookup = build_pipe_lookup(&pipes);
        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|pipe| 1.0 / pipe.eff_conductance.max(1e-12))
            .collect();
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * DIST_SCALE).round() as u32).max(1))
            .collect();
        let span_cost_table = SpanCostTable::disabled(pipes.len());
        let tile_grid = TileGrid::build(&pipes, &nodes, w, h);
        let flat_adjacency = FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = Arc::new(build_tile_templates(ctx.chipdb()));

        let mut net = Self {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table,
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup,
            tile_type_by_node,
            width: w,
            height: h,
            x0,
            y0,
            zero_bel_tiles,
            total_bels,
            coarsen,
        };
        net.refresh_flat_edge_costs();
        net
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

    #[inline]
    pub fn pipe_delta(&self, pipe_idx: usize) -> (i32, i32) {
        let pipe = &self.pipes[pipe_idx];
        let from = &self.nodes[pipe.from];
        let to = &self.nodes[pipe.to];
        (to.tile_x - from.tile_x, to.tile_y - from.tile_y)
    }

    /// Hardening history for a pipe, or 0.0 when the term is disabled.
    /// Read through this so the span-cost table (which buckets pipes by
    /// *type*, not identity) cannot silently drop the per-pipe component.
    #[inline]
    pub fn pipe_hist(&self, pipe_idx: usize) -> f64 {
        self.pipe_history.get(pipe_idx).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn pipe_cost(&self, pipe_idx: usize) -> f64 {
        let bpr = if self.span_cost_table.enabled {
            let entry = self.span_cost_table.pipe_entry[pipe_idx] as usize;
            if entry < self.span_cost_table.costs.len() {
                self.span_cost_table.costs[entry]
            } else {
                self.pipe_costs[pipe_idx]
            }
        } else {
            self.pipe_costs[pipe_idx]
        };
        bpr + self.pipe_hist(pipe_idx)
    }

    #[inline]
    pub fn pipe_cost_int(&self, pipe_idx: usize) -> u32 {
        let bpr = if self.span_cost_table.enabled {
            let entry = self.span_cost_table.pipe_entry[pipe_idx] as usize;
            if entry < self.span_cost_table.costs_int.len() {
                self.span_cost_table.costs_int[entry]
            } else {
                self.pipe_costs_int[pipe_idx]
            }
        } else {
            self.pipe_costs_int[pipe_idx]
        };
        bpr.saturating_add(hist_to_int(self.pipe_hist(pipe_idx)))
    }

    /// Populate the per-edge cost arrays in `flat_adjacency` from the current
    /// `pipe_cost` / `pipe_cost_int` (which already routes through the span
    /// cost table when enabled). Call after every `update_effective_conductance`
    /// or `rebuild_span_cost_table*` so the Dijkstra hot path reads the packed
    /// per-edge costs sequentially instead of scattering through `pipe_idx`.
    pub fn refresh_flat_edge_costs(&mut self) {
        let n_edges = self.flat_adjacency.pipe_idx.len();
        // Split borrows so we can write into flat_adjacency while reading from
        // self.pipe_costs / pipe_costs_int / span_cost_table.
        let pipe_idx = self.flat_adjacency.pipe_idx.as_slice();
        let cost_int_dst = self.flat_adjacency.cost_int.as_mut_slice();
        let cost_f_dst = self.flat_adjacency.cost_f.as_mut_slice();
        let hist = self.pipe_history.as_slice();
        if self.span_cost_table.enabled {
            let entries = self.span_cost_table.pipe_entry.as_slice();
            let costs = self.span_cost_table.costs.as_slice();
            let costs_int = self.span_cost_table.costs_int.as_slice();
            let fallback_f = self.pipe_costs.as_slice();
            let fallback_int = self.pipe_costs_int.as_slice();
            for e in 0..n_edges {
                let pidx = pipe_idx[e] as usize;
                let entry = entries[pidx] as usize;
                let (cf, ci) = if entry < costs.len() {
                    (costs[entry], costs_int[entry])
                } else {
                    (fallback_f[pidx], fallback_int[pidx])
                };
                let h = hist[pidx];
                cost_int_dst[e] = ci.saturating_add(hist_to_int(h));
                cost_f_dst[e] = (cf + h) as f32;
            }
        } else {
            let src_f = self.pipe_costs.as_slice();
            let src_i = self.pipe_costs_int.as_slice();
            for e in 0..n_edges {
                let pidx = pipe_idx[e] as usize;
                let h = hist[pidx];
                cost_int_dst[e] = src_i[pidx].saturating_add(hist_to_int(h));
                cost_f_dst[e] = (src_f[pidx] + h) as f32;
            }
        }
    }

    /// Reset all dynamic state (pressures, flows, net counts).
    pub fn reset(&mut self) {
        self.nodes.par_iter_mut().for_each(|node| {
            node.pressure = 0.0;
        });
        self.pipes.par_iter_mut().for_each(|p| {
            p.flow = 0.0;
            p.net_count = 0.0;
            p.cell_density = 0.0;
        });
        self.pipe_history.iter_mut().for_each(|h| *h = 0.0);
    }

    /// Maximum utilization ratio |flow|/capacity across all pipes.
    pub fn max_utilization(&self) -> f64 {
        self.pipes
            .par_iter()
            .filter(|p| p.capacity > 0.0)
            .map(|p| p.flow.abs() / p.capacity)
            .reduce(|| 0.0, f64::max)
    }

    /// Print a per-span pipe utilization histogram. For each span bucket emits
    /// pipe count, total raw usage / capacity, and effective ratio percentiles
    /// (p50/p90/p99/max). `effective ratio` divides raw usage by `capacity *
    /// borrow_slack(span)` to match the BPR resistance model. Used to verify
    /// whether the placer is actually saturating short-span pipes.
    pub fn report_span_utilization(&self, label: &str) {
        use super::resistance::borrow_slack;
        let mut by_span: FxHashMap<i32, Vec<(f64, f64)>> = FxHashMap::default();
        for p in &self.pipes {
            if p.capacity <= 0.0 {
                continue;
            }
            let span = match p.pipe_type {
                PipeType::InterTile(_) => 1,
                PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
            };
            by_span
                .entry(span)
                .or_default()
                .push((p.net_count.max(0.0), p.capacity));
        }
        let mut spans: Vec<i32> = by_span.keys().copied().collect();
        spans.sort();
        eprintln!("Pipe span utilization ({label}):");
        eprintln!(
            "  span  pipes  cap_sum  use_sum  raw_avg   raw_p99  raw_max   eff_p99  eff_max  pct_over_eff_1"
        );
        for s in spans {
            let v = &by_span[&s];
            let n = v.len();
            let cap_sum: f64 = v.iter().map(|(_, c)| *c).sum();
            let use_sum: f64 = v.iter().map(|(u, _)| *u).sum();
            let slack = borrow_slack(s);
            let mut raw_ratios: Vec<f64> = v.iter().map(|(u, c)| u / c).collect();
            let mut eff_ratios: Vec<f64> = v.iter().map(|(u, c)| u / (c * slack)).collect();
            raw_ratios.sort_by(|a, b| a.total_cmp(b));
            eff_ratios.sort_by(|a, b| a.total_cmp(b));
            let pct = |v: &[f64], q: f64| -> f64 {
                if v.is_empty() {
                    0.0
                } else {
                    let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
                    v[idx]
                }
            };
            let raw_avg = if cap_sum > 0.0 {
                use_sum / cap_sum
            } else {
                0.0
            };
            let raw_p99 = pct(&raw_ratios, 0.99);
            let raw_max = *raw_ratios.last().unwrap_or(&0.0);
            let eff_p99 = pct(&eff_ratios, 0.99);
            let eff_max = *eff_ratios.last().unwrap_or(&0.0);
            let n_over = eff_ratios.iter().filter(|&&r| r > 1.0).count() as f64 * 100.0 / n as f64;
            eprintln!(
                "  {:>3}   {:>5}  {:>7.0}  {:>7.0}  {:>6.3}   {:>6.3}   {:>6.3}   {:>6.3}   {:>6.3}   {:>5.1}%",
                s, n, cap_sum, use_sum, raw_avg, raw_p99, raw_max, eff_p99, eff_max, n_over
            );
        }
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
        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|pipe| 1.0 / pipe.eff_conductance.max(1e-12))
            .collect();
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * DIST_SCALE).round() as u32).max(1))
            .collect();
        let span_cost_table = SpanCostTable::disabled(pipes.len());
        let tile_grid = TileGrid::build(&pipes, &nodes, grid_w, grid_h);
        let flat_adjacency = FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = Arc::new(build_tile_templates(ctx.chipdb()));

        let mut net = Self {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table,
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup,
            tile_type_by_node: vec![0; total_nodes],
            width: grid_w,
            height: grid_h,
            x0: 0,
            y0: 0,
            zero_bel_tiles,
            total_bels: bels_per_node.iter().sum(),
            coarsen: 1,
        };
        net.refresh_flat_edge_costs();
        net
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
        net_count: 0.0,
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

/// Per tile-type histogram: maps composite-grid (dx, dy) -> wire_count.
/// Exactly one entry is emitted per non-internal physical wire at its
/// max-reach composite delta (Model A: hypergraph wire approximated by its
/// longest physical end-to-end reach; short-span usage handled separately by
/// the BPR congestion function's mild-overflow tolerance).
type SpanHistogram = FxHashMap<(i32, i32), usize>;

/// Return true if the wire name is a global/clock network (GCLK, HCLK, VCC,
/// GND) that should be excluded from span histograms. These wires can span
/// the entire chip and would otherwise corrupt the coarse pipe capacity model
/// with pipes whose span exceeds anything that physical routing offers.
fn is_global_network_wire(name: &str) -> bool {
    name.contains("GCLK")
        || name.contains("HCLK")
        || name.contains("GND")
        || name.contains("VCC")
        || name.contains("CLK_HROW")
        || name.contains("CLK_BUFG")
}

/// Analyze chipdb routing node shapes to build per-tile-type span histograms.
///
/// The chipdb is already composite-compressed: each tile is one composite
/// column (CLBLL_L+INT_L+INT_R+CLBLM_R absorbed into one CLB tile type,
/// X-stride=1 in composite grid, Y-stride=1). A wire's node-shape
/// `tile_wires` list therefore gives tap offsets DIRECTLY in composite-grid
/// units relative to the home tile.
///
/// Under Model A (one-wire-one-entry, max reach):
///   * internal wires (all taps at dx=dy=0) are absorbed into the tile type
///     and contribute nothing to the fabric-level pipe graph.
///   * non-internal wires emit exactly one histogram entry at their farthest
///     composite tap (max Manhattan distance), canonicalized to the positive
///     half-plane so that a wire and its mirror at the destination tile
///     merge into the same key.
///
/// Short-span usage of long wires (hex wires serving span-3 nets, etc.) is
/// NOT modelled here; the BPR congestion function is tuned to accept mild
/// short-span overflow at a penalty, which is the placer's budgeted way of
/// representing that flexibility without double-counting physical wires.
///
/// Global-network wires (GCLK/HCLK/VCC/GND/CLK_HROW/CLK_BUFG) are filtered by
/// name: they are dedicated clock/power networks, not span-wire routing
/// resources.
///
/// Per-tile-type representative picker: for each tile type, pick the shape
/// with the most distinct non-zero reach keys (typically an interior shape
/// rather than a chip-boundary corner with missing neighbours).
fn build_span_histograms(chipdb: &ChipDb) -> Vec<SpanHistogram> {
    let num_tt = chipdb.num_tile_types();
    let num_tiles = chipdb.num_tiles();
    let mut histograms = vec![SpanHistogram::default(); num_tt];

    // Helper: compute a wire's canonical max-reach composite delta, or None if
    // the wire is a global-network wire, has no node shape, or is fully
    // internal to the home composite tile (absorbed, not exposed to fabric).
    let wire_reach = |tile: i32, wire_idx: usize| -> Option<(i32, i32)> {
        let wid = crate::chipdb::WireId::new(tile, wire_idx as i32);
        if is_global_network_wire(chipdb.wire_name(wid)) {
            return None;
        }
        let ns = chipdb.wire_node_shape(tile, wire_idx)?;
        let mut best: Option<(i32, i32, i32)> = None; // (|dx|+|dy|, dx, dy)
        for tw in ns.tile_wires.get() {
            let dx: i16 = unsafe { read_packed!(*tw, dx) };
            let dy: i16 = unsafe { read_packed!(*tw, dy) };
            let dx = dx as i32;
            let dy = dy as i32;
            let mag = dx.abs() + dy.abs();
            match best {
                None => best = Some((mag, dx, dy)),
                Some((m, _, _)) if mag > m => best = Some((mag, dx, dy)),
                _ => {}
            }
        }
        let (mag, dx, dy) = best?;
        if mag == 0 {
            return None; // fully internal → absorbed into the tile type
        }
        let (kdx, kdy) = if dx > 0 || (dx == 0 && dy > 0) {
            (dx, dy)
        } else {
            (-dx, -dy)
        };
        Some((kdx, kdy))
    };

    // Enumerate all unique (tile_type, tile_shape) combinations and pick the
    // richest shape per tile type. "Richest" = highest count of distinct
    // non-zero reach keys across all non-internal wires in that shape. This
    // avoids the single-representative bug where a corner tile with no east
    // neighbours dropped all horizontal connectivity.
    let mut seen: FxHashMap<(i32, i32), i32> = FxHashMap::default();
    for tile in 0..num_tiles {
        let tt = chipdb.tile_type_index(tile);
        if tt < 0 {
            continue;
        }
        let sh = chipdb.tile_shape_index(tile);
        seen.entry((tt, sh)).or_insert(tile);
    }

    let mut best_shape_per_tt: Vec<Option<(i32, usize)>> = vec![None; num_tt];
    for ((tt_idx, _sh_idx), tile) in &seen {
        let tt = chipdb.tile_type_by_index(*tt_idx);
        let mut distinct: FxHashMap<(i32, i32), ()> = FxHashMap::default();
        for wire_idx in 0..tt.wires.len() {
            if let Some(key) = wire_reach(*tile, wire_idx) {
                distinct.insert(key, ());
            }
        }
        let n = distinct.len();
        let slot = &mut best_shape_per_tt[*tt_idx as usize];
        match slot {
            Some((_, best_n)) if *best_n >= n => {}
            _ => *slot = Some((*tile, n)),
        }
    }

    for (tt_idx, entry) in best_shape_per_tt.iter().enumerate() {
        let Some((tile, _)) = entry else {
            continue;
        };
        let tt = chipdb.tile_type_by_index(tt_idx as i32);
        for wire_idx in 0..tt.wires.len() {
            if let Some(key) = wire_reach(*tile, wire_idx) {
                *histograms[tt_idx].entry(key).or_insert(0) += 1;
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
mod flat_adjacency_tests {
    use super::*;

    fn make_pipe(from: usize, to: usize) -> Pipe {
        Pipe {
            from,
            to,
            base_resistance: 1.0,
            capacity: 1.0,
            flow: 0.0,
            net_count: 0.0,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0,
            pipe_type: PipeType::InterTile(Direction::East),
        }
    }

    #[test]
    fn flat_adjacency_matches_node_pipes_enumeration() {
        // Simple diamond: 0 - 1, 0 - 2, 1 - 3, 2 - 3.
        let pipes = vec![
            make_pipe(0, 1),
            make_pipe(0, 2),
            make_pipe(1, 3),
            make_pipe(2, 3),
        ];
        let mut node_pipes: Vec<Vec<usize>> = vec![Vec::new(); 4];
        for (i, p) in pipes.iter().enumerate() {
            node_pipes[p.from].push(i);
            node_pipes[p.to].push(i);
        }
        let adj = FlatAdjacency::build(&node_pipes, &pipes);

        assert_eq!(adj.offsets.len(), 5);
        assert_eq!(*adj.offsets.last().unwrap() as usize, adj.neighbors.len());
        assert_eq!(adj.neighbors.len(), adj.pipe_idx.len());

        for node in 0..4 {
            let range = adj.range(node);
            let mut flat_pairs: Vec<(u32, u32)> = range
                .clone()
                .map(|e| (adj.neighbors[e], adj.pipe_idx[e]))
                .collect();
            let mut ref_pairs: Vec<(u32, u32)> = node_pipes[node]
                .iter()
                .map(|&pidx| {
                    let p = &pipes[pidx];
                    let n = p.from ^ p.to ^ node;
                    (n as u32, pidx as u32)
                })
                .collect();
            flat_pairs.sort();
            ref_pairs.sort();
            assert_eq!(flat_pairs, ref_pairs, "mismatch at node {}", node);
        }
    }

    #[test]
    fn flat_adjacency_empty_has_correct_shape() {
        let adj = FlatAdjacency::empty(7);
        assert_eq!(adj.offsets.len(), 8);
        assert!(adj.offsets.iter().all(|&x| x == 0));
        assert!(adj.neighbors.is_empty());
        assert!(adj.pipe_idx.is_empty());
        for node in 0..7 {
            assert!(adj.range(node).is_empty());
        }
    }
}

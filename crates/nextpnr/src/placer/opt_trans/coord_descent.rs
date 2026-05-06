//! Discrete coordinate descent placer over all movable cell coordinates.
//!
//! The DCD loop treats placement as a `2 * n_cells` dimensional discrete
//! optimization problem. Within a sweep the resistance field is frozen, trial
//! moves are evaluated with cached Dial-logit soft costs, and only the winning
//! moves update the cached distance fields. Pipe usage and effective
//! conductance are refreshed between sweeps.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::common::IdString;
use crate::context::Context;
use crate::metrics::congestion::bresenham_line;
use crate::netlist::{CellId, NetId};
use crate::placer::common;
use crate::placer::common::{CellValidityMask, MuxSlotTracker, TypeAwarePlacement};

use super::config::{GraphModel, OptTransPlacerCfg, PathModel};
use super::demand;
use super::diag::{self, BoundingBoxes, DiagCtx, MoveRecord, PlateauStat};
use super::network::PipeNetwork;
use super::path_solver::{self, PathSolverWorkspace, PathStats};
use super::region_min::{self, RegionMinPyramid};
use super::resistance::ResistanceModel;

#[derive(Clone, Debug)]
struct PinNode {
    node: usize,
    cell_idx: Option<usize>,
    is_driver: bool,
    is_fixed: bool,
}

#[derive(Clone, Debug)]
struct NetInfo {
    net_id: NetId,
    debug_name: String,
    pins: Vec<PinNode>,
    has_fixed_pin: bool,
}

/// Sparse per-net Dijkstra distance labels. `rows[net_idx]` holds only the
/// nodes that the per-net solve actually settled — absent nodes implicitly
/// have distance `INFINITY`. On a 105k-cell / 69k-node design, typical
/// per-net settle counts are 50–500, so the sparse representation is roughly
/// 100× smaller than the prior flat `n_nets × n_nodes` `Vec<f32>` (≈170 MB
/// vs ≈29 GB on FPGA01).
///
/// Sparse storage matches the access pattern: writers (Dijkstra fold) write
/// exactly the settled nodes via `ws.settle_order`; the dominant reader
/// (`get`) is a single-node lookup that maps absent → INFINITY.
///
/// `f32` storage gives 7 digits of precision — plenty for argmin selection
/// in the cost path; values are bounded by `chip_diameter × max_pipe_cost`.
///
/// Code paths that genuinely need a dense row (BB-pyramid build, the
/// Bresenham-surrogate writer) call `materialize_row` to copy the sparse
/// entries into a caller-supplied scratch buffer once per use.
struct DistCache {
    rows: Vec<FxHashMap<u32, f32>>,
    n_nets: usize,
    n_nodes: usize,
    /// 1-Steiner pseudo-node centroids in network grid coords. One entry per
    /// net. Zeros when `steiner_weight == 0.0` in config (term disabled).
    /// Refreshed by `compute_steiner_centroids` at the top of each outer iter.
    steiner_cx: Vec<f64>,
    steiner_cy: Vec<f64>,
    /// Per-net rectilinear MST adjacency. `mst_neighbors[net_idx][pin_idx]` is
    /// the list of pin indices that pin_idx shares an MST edge with on its
    /// net. Net-level cost is `Σ |p_i - p_j|` over MST edges; per-pin
    /// gradient pulls toward each MST neighbor (no central attractor).
    /// Empty when `mst_edge_weight == 0.0`. Refreshed per outer iter.
    mst_neighbors: Vec<Vec<Vec<u16>>>,
}

impl DistCache {
    fn new(n_nets: usize, n_nodes: usize) -> Self {
        Self {
            rows: (0..n_nets).map(|_| FxHashMap::default()).collect(),
            n_nets,
            n_nodes,
            steiner_cx: vec![0.0; n_nets],
            steiner_cy: vec![0.0; n_nets],
            mst_neighbors: Vec::new(),
        }
    }

    /// Reallocates if shape changed; otherwise leaves stored rows intact
    /// so the skip-refresh path can reuse per-net labels across iterations.
    /// Callers that drop the skip-mask path explicitly clear rows before
    /// re-writing them (sparse rows don't need a global INF fill).
    fn ensure_shape(&mut self, n_nets: usize, n_nodes: usize) {
        if self.n_nets != n_nets || self.n_nodes != n_nodes {
            *self = Self::new(n_nets, n_nodes);
        }
    }

    fn reset(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }

    #[inline]
    fn row(&self, net_idx: usize) -> &FxHashMap<u32, f32> {
        &self.rows[net_idx]
    }

    #[inline]
    fn row_mut(&mut self, net_idx: usize) -> &mut FxHashMap<u32, f32> {
        &mut self.rows[net_idx]
    }

    /// Read a single dist label. Returned as `f64` so callers in the cost
    /// path don't have to track the storage precision; absent → INFINITY.
    #[inline]
    fn get(&self, net_idx: usize, node: usize) -> f64 {
        match self.rows[net_idx].get(&(node as u32)) {
            Some(&v) => v as f64,
            None => f64::INFINITY,
        }
    }

    /// Copy this net's sparse row into a dense scratch buffer of length
    /// `n_nodes`. Absent entries become `f32::INFINITY`. Used by code paths
    /// that need contiguous random-access reads (BB-pyramid build).
    fn materialize_row(&self, net_idx: usize, scratch: &mut [f32]) {
        debug_assert_eq!(scratch.len(), self.n_nodes);
        scratch.fill(f32::INFINITY);
        for (&node, &v) in &self.rows[net_idx] {
            scratch[node as usize] = v;
        }
    }

    /// Diagnostic: total live entries and total reserved capacity across all
    /// rows. Returned `est_bytes` estimates hashbrown overhead (~9 B/slot:
    /// 8 B for `(K, V)` + 1 ctrl byte) plus 48 B per FxHashMap header.
    fn memory_stats(&self) -> (usize, usize, usize) {
        let mut total_entries = 0usize;
        let mut total_capacity = 0usize;
        for row in &self.rows {
            total_entries += row.len();
            total_capacity += row.capacity();
        }
        let est_bytes = total_capacity * 9 + self.rows.len() * 48;
        (total_entries, total_capacity, est_bytes)
    }
}

/// Read `VmRSS` (resident set size) from `/proc/self/status` in KiB. Returns
/// 0 if unavailable. Used in DCD diagnostics so we can compare reported
/// hashmap memory against actual process RSS each iter.
/// Manhattan radius (in tiles) around each pin within which DistCache
/// entries are kept. Distances to nodes farther than this radius from every
/// pin are computed by Dijkstra (Dial back-pass needs them) but never
/// queried by DCD, so we drop them at cache-write time. Default 12, tuned
/// to comfortably cover the bisection step (avg ~14, max ~356 manhattan
/// in FPGA01 iter 0 — outliers will pay an `INFINITY` cost and be skipped,
/// which is acceptable for placement-cost surrogate use). Override via
/// `NPNR_OT_CACHE_RADIUS`.
fn cache_radius_tiles() -> i32 {
    use std::sync::OnceLock;
    static RADIUS: OnceLock<i32> = OnceLock::new();
    *RADIUS.get_or_init(|| {
        std::env::var("NPNR_OT_CACHE_RADIUS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .map(|v| v.max(0))
            .unwrap_or(12)
    })
}

fn process_rss_kb() -> usize {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
        }
    }
    0
}

/// Recompute per-net Steiner pseudo-node centroids from current pin positions.
/// The centroid is the arithmetic mean of all pin positions (driver + sinks)
/// in network grid coords. Called at the top of each outer iter so the Jacobi
/// sweep sees a stable hub during inner evaluation.
fn compute_steiner_centroids(
    dist_cache: &mut DistCache,
    net_infos: &[NetInfo],
    network: &PipeNetwork,
) {
    let width = network.width;
    for (net_idx, info) in net_infos.iter().enumerate() {
        if info.pins.is_empty() {
            dist_cache.steiner_cx[net_idx] = 0.0;
            dist_cache.steiner_cy[net_idx] = 0.0;
            continue;
        }
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        for pin in &info.pins {
            let nx = (pin.node % width as usize) as f64;
            let ny = (pin.node / width as usize) as f64;
            sx += nx;
            sy += ny;
        }
        let n = info.pins.len() as f64;
        dist_cache.steiner_cx[net_idx] = sx / n;
        dist_cache.steiner_cy[net_idx] = sy / n;
    }
}

/// Recompute per-net rectilinear MST adjacency from current pin positions.
/// Prim's algorithm in O(n²) per net. The MST length is an upper bound on
/// the rectilinear Steiner tree (within 3/2) and a tight wirelength
/// approximation that distributes pull across pin-pair edges instead of a
/// single centroid attractor. Called per outer iter when
/// `cfg.mst_edge_weight > 0.0`.
fn compute_mst_neighbors(
    dist_cache: &mut DistCache,
    net_infos: &[NetInfo],
    network: &PipeNetwork,
) {
    let width = network.width as usize;
    dist_cache.mst_neighbors.clear();
    dist_cache.mst_neighbors.resize(net_infos.len(), Vec::new());

    for (net_idx, info) in net_infos.iter().enumerate() {
        let n = info.pins.len();
        let mut neighbors: Vec<Vec<u16>> = vec![Vec::new(); n];
        if n < 2 {
            dist_cache.mst_neighbors[net_idx] = neighbors;
            continue;
        }

        let pos: Vec<(f64, f64)> = info
            .pins
            .iter()
            .map(|p| ((p.node % width) as f64, (p.node / width) as f64))
            .collect();

        // Prim's: grow tree from pin 0, picking nearest non-tree pin each step.
        let mut in_tree = vec![false; n];
        let mut min_dist = vec![f64::INFINITY; n];
        let mut parent = vec![u16::MAX; n];
        in_tree[0] = true;
        for j in 1..n {
            let dx = (pos[0].0 - pos[j].0).abs();
            let dy = (pos[0].1 - pos[j].1).abs();
            min_dist[j] = dx + dy;
            parent[j] = 0;
        }

        for _ in 1..n {
            let mut best_j = usize::MAX;
            let mut best_d = f64::INFINITY;
            for j in 0..n {
                if !in_tree[j] && min_dist[j] < best_d {
                    best_d = min_dist[j];
                    best_j = j;
                }
            }
            if best_j == usize::MAX {
                break;
            }
            in_tree[best_j] = true;
            let p = parent[best_j] as usize;
            neighbors[best_j].push(p as u16);
            neighbors[p].push(best_j as u16);

            for j in 0..n {
                if !in_tree[j] {
                    let dx = (pos[best_j].0 - pos[j].0).abs();
                    let dy = (pos[best_j].1 - pos[j].1).abs();
                    let d = dx + dy;
                    if d < min_dist[j] {
                        min_dist[j] = d;
                        parent[j] = best_j as u16;
                    }
                }
            }
        }

        dist_cache.mst_neighbors[net_idx] = neighbors;
    }
}

struct CellNetMap {
    /// For each cell, the list of (net_idx, pin_idx) pairs it participates in.
    map: Vec<Vec<(usize, usize)>>,
    /// Topological cell order following driver → sink edges, with back edges
    /// broken by choosing the cell with smallest remaining in-degree.
    topo_order: Vec<usize>,
}

impl CellNetMap {
    fn build(net_infos: &[NetInfo], n_cells: usize) -> Self {
        let mut map = vec![Vec::new(); n_cells];
        for (net_idx, info) in net_infos.iter().enumerate() {
            for (pin_idx, pin) in info.pins.iter().enumerate() {
                if let Some(cell_idx) = pin.cell_idx {
                    if !pin.is_fixed && cell_idx < n_cells {
                        map[cell_idx].push((net_idx, pin_idx));
                    }
                }
            }
        }
        let topo_order = Self::build_topo_order(net_infos, n_cells);
        Self { map, topo_order }
    }

    /// Kahn's-style topological sort over driver → sink edges from `net_infos`.
    /// For cycles (back edges), relaxation picks the cell with the smallest
    /// remaining in-degree, breaking the cycle arbitrarily but deterministically.
    fn build_topo_order(net_infos: &[NetInfo], n_cells: usize) -> Vec<usize> {
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        let mut in_degree: Vec<u32> = vec![0; n_cells];

        for info in net_infos {
            let Some(drv) = info.pins.iter().find(|p| p.is_driver) else {
                continue;
            };
            let Some(drv_ci) = drv.cell_idx else { continue };
            if drv.is_fixed || drv_ci >= n_cells {
                continue;
            }

            for sink in &info.pins {
                if sink.is_driver {
                    continue;
                }
                let Some(sink_ci) = sink.cell_idx else {
                    continue;
                };
                if sink.is_fixed || sink_ci >= n_cells {
                    continue;
                }
                if sink_ci == drv_ci {
                    continue;
                }

                out_edges[drv_ci].push(sink_ci);
                in_degree[sink_ci] += 1;
            }
        }

        let mut order = Vec::with_capacity(n_cells);
        let mut placed = vec![false; n_cells];
        let mut deg = in_degree;

        loop {
            let mut best: Option<usize> = None;
            let mut best_deg = u32::MAX;
            for ci in 0..n_cells {
                if placed[ci] {
                    continue;
                }
                if deg[ci] < best_deg {
                    best_deg = deg[ci];
                    best = Some(ci);
                    if best_deg == 0 {
                        break;
                    }
                }
            }
            let Some(ci) = best else { break };

            order.push(ci);
            placed[ci] = true;

            for &next in &out_edges[ci] {
                if !placed[next] && deg[next] > 0 {
                    deg[next] -= 1;
                }
            }
        }

        for ci in 0..n_cells {
            if !placed[ci] {
                order.push(ci);
            }
        }

        order
    }
}

struct SolveAccum {
    edge_usage: Vec<f64>,
    energy: f64,
    stats: PathStats,
    /// Diagnostics: number of nets where the corridor attempt missed demand
    /// and fell through to a full-graph Dijkstra. Each fallback settles up
    /// to `n_nodes` nodes, ballooning the per-net `dist_cache` row.
    diag_corridor_fallback: u64,
    /// Diagnostic: number of nets where the sink-bounded distance cap
    /// actually armed (every unique sink settled before Dijkstra exhausted
    /// the corridor). The complement — nets where the cap never armed —
    /// pays the full `corridor_size` settle cost and is the dominant memory
    /// driver on chip-spanning / disconnected nets.
    diag_cap_armed: u64,
    diag_solves: u64,
    diag_settle_sum: u64,
    diag_settle_max: u32,
    /// Tally of how many times the rayon fold-init closure ran. With many
    /// splits each init pays a fresh `vec![0.0; n_pipes]` allocation, so
    /// this is the second possible source of large transient memory.
    diag_init_count: u32,
}

#[inline]
fn pipe_lookup_key(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

#[inline]
fn coord_to_node(network: &PipeNetwork, x: i32, y: i32) -> usize {
    let gx = x.clamp(0, network.width - 1);
    let gy = y.clamp(0, network.height - 1);
    network.node_index(gx, gy)
}

#[inline]
fn position_to_node(network: &PipeNetwork, x: f64, y: f64) -> usize {
    let gx = network.tile_to_net(x).round() as i32;
    let gy = network.tile_to_net(y).round() as i32;
    coord_to_node(network, gx, gy)
}

/// Pin position in tile coords plus the index of the cell whose motion
/// drives this pin. Cluster children move rigidly with their root, so their
/// pin position is `root_pos + (constr_x, constr_y)` and the gradient is
/// attributed to the root index — this is what tells the optimizer that
/// minimizing wirelength on a cluster-child net should move the root.
fn pin_pos(
    ctx: &Context,
    cell_id: CellId,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> (f64, f64, Option<usize>) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        return (cell_x[idx], cell_y[idx], Some(idx));
    }

    let cell = ctx.design.cell(cell_id);

    // Cluster child: derive position from root's current placement.
    if let Some(root_id) = cell.cluster {
        if root_id != cell_id {
            if let Some(&root_idx) = cell_to_idx.get(&root_id) {
                return (
                    cell_x[root_idx] + cell.constr_x as f64,
                    cell_y[root_idx] + cell.constr_y as f64,
                    Some(root_idx),
                );
            }
            // Root is not a movable cell: fall through to BEL location of
            // either the child (if already bound) or the root.
            let root_cell = ctx.design.cell(root_id);
            if let Some(root_bel) = root_cell.bel {
                let loc = ctx.bel(root_bel).loc();
                return (
                    (loc.x + cell.constr_x - network.x0) as f64,
                    (loc.y + cell.constr_y - network.y0) as f64,
                    None,
                );
            }
        }
    }

    if let Some(bel) = cell.bel {
        let loc = ctx.bel(bel).loc();
        return (
            (loc.x - network.x0) as f64,
            (loc.y - network.y0) as f64,
            None,
        );
    }

    panic!(
        "pin_pos: cell {} is neither movable, a placed cluster child, nor bound to a BEL",
        ctx.name_of(cell.name),
    );
}

fn collect_net_infos_simple(
    ctx: &Context,
    net_ids: &[NetId],
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<NetInfo> {
    let skip_constants = cfg.skip_constants;
    let skip_clocks = cfg.skip_clocks || cfg.exclude_globals;

    let mut nets: Vec<_> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            let net = ctx.design.net(net_id);
            let net_name = ctx.name_of(net.name);
            let is_const = net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET";
            if skip_constants && is_const {
                return None;
            }
            if skip_clocks {
                let lower = net_name.to_ascii_lowercase();
                let is_clockish = lower.contains("clk") || lower.contains("clock");
                if is_clockish {
                    return None;
                }
            }

            let driver = net.driver()?;
            if net.num_users() == 0 {
                return None;
            }

            let mut pins = Vec::with_capacity(net.num_users() + 1);
            let mut has_fixed_pin = false;
            let mut has_movable = false;

            let driver_cell = ctx.design.cell(driver.cell);
            let (dx, dy, driver_idx) =
                pin_pos(ctx, driver.cell, cell_to_idx, cell_x, cell_y, network);
            let driver_fixed = driver_cell.bel_strength.is_locked();
            has_fixed_pin |= driver_fixed || driver_idx.is_none();
            has_movable |= driver_idx.is_some() && !driver_fixed;
            pins.push(PinNode {
                node: position_to_node(network, dx, dy),
                cell_idx: driver_idx,
                is_driver: true,
                is_fixed: driver_fixed,
            });

            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let user_cell = ctx.design.cell(user.cell);
                let (ux, uy, user_idx) =
                    pin_pos(ctx, user.cell, cell_to_idx, cell_x, cell_y, network);
                let user_fixed = user_cell.bel_strength.is_locked();
                has_fixed_pin |= user_fixed || user_idx.is_none();
                has_movable |= user_idx.is_some() && !user_fixed;
                pins.push(PinNode {
                    node: position_to_node(network, ux, uy),
                    cell_idx: user_idx,
                    is_driver: false,
                    is_fixed: user_fixed,
                });
            }

            if has_movable && pins.len() >= 2 {
                Some(NetInfo {
                    net_id,
                    debug_name: net_name.to_string(),
                    pins,
                    has_fixed_pin,
                })
            } else {
                None
            }
        })
        .collect();

    nets.sort_by(|a, b| {
        b.pins
            .len()
            .cmp(&a.pins.len())
            .then_with(|| a.debug_name.cmp(&b.debug_name))
    });
    nets
}

fn net_path_weight(info: &NetInfo, cfg: &OptTransPlacerCfg) -> f64 {
    let crit = cfg
        .timing_criticality
        .get(&info.net_id)
        .copied()
        .unwrap_or(0.0) as f64;
    let timing = 1.0 + cfg.timing_weight.max(0.0) * crit.clamp(0.0, 1.0);
    let locked = if info.has_fixed_pin {
        cfg.locked_pin_weight.max(0.0)
    } else {
        1.0
    };
    timing * locked
}

fn source_node(info: &NetInfo) -> Option<usize> {
    info.pins
        .iter()
        .find(|pin| pin.is_driver)
        .map(|pin| pin.node)
}

/// Unsafe wrapper for parallel mutable access to disjoint hashmap rows of
/// the sparse `DistCache`. Each worker thread takes `&mut FxHashMap` for a
/// distinct `net_idx`; rayon's `par_chunks` guarantees disjointness via
/// `chunk_idx * batch_size + local_idx`, so the aliasing rule holds.
struct DistMapPtr {
    base: *mut FxHashMap<u32, f32>,
    n_nets: usize,
}
unsafe impl Send for DistMapPtr {}
unsafe impl Sync for DistMapPtr {}
impl DistMapPtr {
    fn row_mut(&self, idx: usize) -> &mut FxHashMap<u32, f32> {
        assert!(idx < self.n_nets);
        unsafe { &mut *self.base.add(idx) }
    }
}

fn solve_all_nets(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    collect_usage: bool,
    skip_mask: Option<&[bool]>,
) -> SolveAccum {
    solve_all_nets_with_displacement(
        network,
        net_infos,
        cfg,
        solve_pool,
        dist_cache,
        collect_usage,
        skip_mask,
        None,
    )
}

fn solve_all_nets_with_displacement(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    collect_usage: bool,
    skip_mask: Option<&[bool]>,
    displacement: Option<&super::displacement::DisplacementTable>,
) -> SolveAccum {
    if cfg.path_model == PathModel::BresenhamLogit {
        return solve_all_nets_bresenham_logit(
            network,
            net_infos,
            cfg,
            solve_pool,
            dist_cache,
            collect_usage,
        );
    }

    let n_nodes = network.num_nodes();
    let n_pipes = network.num_pipes();
    let batch_size = path_solver::auto_batch_size(
        cfg.net_parallel_batch_size,
        net_infos.len(),
        solve_pool,
    );

    let dist_access = DistMapPtr {
        base: dist_cache.rows.as_mut_ptr(),
        n_nets: dist_cache.n_nets,
    };
    solve_pool.install(|| {
        net_infos
            .par_chunks(batch_size)
            .enumerate()
            .fold(
                || {
                    (
                        SolveAccum {
                            edge_usage: vec![0.0; n_pipes],
                            energy: 0.0,
                            stats: PathStats::default(),
                            diag_corridor_fallback: 0,
                            diag_cap_armed: 0,
                            diag_solves: 0,
                            diag_settle_sum: 0,
                            diag_settle_max: 0,
                            diag_init_count: 1,
                        },
                        PathSolverWorkspace::new(n_nodes, n_pipes),
                    )
                },
                |(mut accum, mut ws), (chunk_idx, chunk)| {
                    let base_net_idx = chunk_idx * batch_size;
                    for (local_idx, info) in chunk.iter().enumerate() {
                        let net_idx = base_net_idx + local_idx;
                        // Skip-refresh: reuse the previous iter's dist_cache
                        // row for this net when the caller says its pin
                        // signature is unchanged. Pipe costs may have drifted
                        // slightly via BPR, but the cached labels are a
                        // good-enough approximation for the DCD sweep's
                        // argmin selection.
                        if skip_mask.map(|m| m[net_idx]).unwrap_or(false) {
                            continue;
                        }
                        let Some(source) = source_node(info) else {
                            accum.stats.failures += 1;
                            continue;
                        };
                        // Demand normalization: each NET represents ONE physical
                        // wire that the driver fans out to K sinks. Summed
                        // source-edge flow must therefore be 1 wire per net,
                        // not K. Splitting `1/K` across sinks gives the
                        // correct per-net source load; edge usage on shared
                        // common-path pipes accumulates to 1, which is the
                        // physical wire count.
                        let n_sinks = info.pins.iter().filter(|p| !p.is_driver).count();
                        if n_sinks == 0 {
                            accum.stats.failures += 1;
                            continue;
                        }
                        let per_sink = 1.0 / n_sinks as f64;
                        let sink_demands: Vec<(usize, f64)> = info
                            .pins
                            .iter()
                            .filter(|pin| !pin.is_driver)
                            .map(|pin| (pin.node, per_sink))
                            .collect();

                        ws.begin_net();
                        let result = if collect_usage {
                            path_solver::dial_logit_load_dispatch(
                                network,
                                source,
                                &sink_demands,
                                &mut ws,
                                Some(&mut accum.edge_usage),
                                displacement,
                                cfg.graph_model,
                            )
                        } else {
                            path_solver::dial_logit_load_dispatch(
                                network,
                                source,
                                &sink_demands,
                                &mut ws,
                                None,
                                displacement,
                                cfg.graph_model,
                            )
                        };

                        accum.stats.total_solves += 1;
                        accum.stats.total_heap_pops += result.heap_pops;
                        accum.stats.max_heap_pops = accum.stats.max_heap_pops.max(result.heap_pops);
                        if result.missing_demand > 0.0 {
                            accum.stats.failures += 1;
                        }

                        let settle_len = ws.settle_order.len() as u64;
                        accum.diag_solves += 1;
                        accum.diag_settle_sum += settle_len;
                        if settle_len as u32 > accum.diag_settle_max {
                            accum.diag_settle_max = settle_len as u32;
                        }
                        if result.corridor_fallback {
                            accum.diag_corridor_fallback += 1;
                        }
                        if result.cap_armed {
                            accum.diag_cap_armed += 1;
                        }

                        // Filter cache writes by "within R Manhattan tiles of
                        // any pin of this net". DCD's per-cell cost eval
                        // queries `dist_cache.get(net, candidate_pos)` only
                        // for candidate positions that a cell on this net
                        // might occupy — which by construction is a small
                        // bbox around each pin. Distances beyond that radius
                        // are computed by Dijkstra (needed for the Dial
                        // back-pass) but never read by DCD, so storing them
                        // is pure waste. On FPGA01 chip-spanning nets this
                        // is the dominant DistCache cost.
                        let r = cache_radius_tiles();
                        let width_n = network.width as usize;
                        ws.pin_xy_scratch.clear();
                        ws.pin_xy_scratch.reserve(info.pins.len());
                        for pin in &info.pins {
                            ws.pin_xy_scratch.push((
                                (pin.node % width_n) as i32,
                                (pin.node / width_n) as i32,
                            ));
                        }

                        let dist_out = dist_access.row_mut(net_idx);
                        dist_out.clear();
                        dist_out.reserve(ws.settle_order.len() / 4 + 8);
                        for &node in &ws.settle_order {
                            let nx = (node % width_n) as i32;
                            let ny = (node / width_n) as i32;
                            let mut near = false;
                            for &(px, py) in ws.pin_xy_scratch.iter() {
                                if (nx - px).abs() + (ny - py).abs() <= r {
                                    near = true;
                                    break;
                                }
                            }
                            if near {
                                dist_out.insert(node as u32, ws.dist[node] as f32);
                            }
                        }
                        // Reclaim capacity if a previous iter grew this row
                        // (e.g. the cell was near a high-fanout net then moved
                        // away). `clear()` keeps hashbrown's table allocation,
                        // so without this hashmaps that ever ballooned stay
                        // ballooned and total memory grows monotonically.
                        if dist_out.capacity() > 4 * dist_out.len().max(8) {
                            dist_out.shrink_to_fit();
                        }
                        accum.energy += net_path_weight(info, cfg) * result.energy;
                    }
                    (accum, ws)
                },
            )
            .map(|(accum, _)| accum)
            .reduce(
                || SolveAccum {
                    edge_usage: vec![0.0; n_pipes],
                    energy: 0.0,
                    stats: PathStats::default(),
                    diag_corridor_fallback: 0,
                    diag_cap_armed: 0,
                    diag_solves: 0,
                    diag_settle_sum: 0,
                    diag_settle_max: 0,
                    diag_init_count: 0,
                },
                |mut a, b| {
                    for (dst, src) in a.edge_usage.iter_mut().zip(b.edge_usage) {
                        *dst += src;
                    }
                    a.energy += b.energy;
                    a.stats.total_solves += b.stats.total_solves;
                    a.stats.total_heap_pops += b.stats.total_heap_pops;
                    a.stats.max_heap_pops = a.stats.max_heap_pops.max(b.stats.max_heap_pops);
                    a.stats.failures += b.stats.failures;
                    a.diag_corridor_fallback += b.diag_corridor_fallback;
                    a.diag_cap_armed += b.diag_cap_armed;
                    a.diag_solves += b.diag_solves;
                    a.diag_settle_sum += b.diag_settle_sum;
                    a.diag_settle_max = a.diag_settle_max.max(b.diag_settle_max);
                    a.diag_init_count += b.diag_init_count;
                    a
                },
            )
    })
}

#[derive(Default)]
struct TemplateAccum {
    edge_usage: Vec<f64>,
    energy: f64,
    failures: usize,
}

fn solve_all_nets_bresenham_logit(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    collect_usage: bool,
) -> SolveAccum {
    let n_pipes = network.num_pipes();
    let batch_size = path_solver::auto_batch_size(
        cfg.net_parallel_batch_size,
        net_infos.len(),
        solve_pool,
    );
    let dist_access = DistMapPtr {
        base: dist_cache.rows.as_mut_ptr(),
        n_nets: dist_cache.n_nets,
    };

    solve_pool
        .install(|| {
            net_infos
                .par_chunks(batch_size)
                .enumerate()
                .fold(
                    || TemplateAccum {
                        edge_usage: vec![0.0; n_pipes],
                        energy: 0.0,
                        failures: 0,
                    },
                    |mut accum, (chunk_idx, chunk)| {
                        let base_net_idx = chunk_idx * batch_size;
                        for (local_idx, info) in chunk.iter().enumerate() {
                            let net_idx = base_net_idx + local_idx;
                            let dist_out = dist_access.row_mut(net_idx);
                            // Bresenham writes every node, so the hashmap
                            // ends up dense — no memory win vs the legacy
                            // flat layout. Default Dijkstra path stays
                            // sparse; this branch is opt-in via PathModel.
                            dist_out.clear();
                            dist_out.reserve(network.num_nodes());

                            let Some(source) = source_node(info) else {
                                accum.failures += 1;
                                continue;
                            };
                            fill_bresenham_surrogate_field(network, source, dist_out);

                            let sinks: Vec<usize> = info
                                .pins
                                .iter()
                                .filter(|pin| !pin.is_driver)
                                .map(|pin| pin.node)
                                .collect();
                            if sinks.is_empty() {
                                accum.failures += 1;
                                continue;
                            }
                            let result = solve_net_bresenham_logit(
                                network,
                                source,
                                &sinks,
                                if collect_usage {
                                    Some(&mut accum.edge_usage)
                                } else {
                                    None
                                },
                            );
                            if result.failures > 0 {
                                accum.failures += result.failures;
                            }
                            accum.energy += net_path_weight(info, cfg) * result.energy;
                        }
                        accum
                    },
                )
                .reduce(
                    || TemplateAccum {
                        edge_usage: vec![0.0; n_pipes],
                        energy: 0.0,
                        failures: 0,
                    },
                    |mut a, b| {
                        for (dst, src) in a.edge_usage.iter_mut().zip(b.edge_usage) {
                            *dst += src;
                        }
                        a.energy += b.energy;
                        a.failures += b.failures;
                        a
                    },
                )
        })
        .into_solve_accum(net_infos.len())
}

fn solve_bresenham_usage_only(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
) -> SolveAccum {
    let n_pipes = network.num_pipes();
    let batch_size = path_solver::auto_batch_size(
        cfg.net_parallel_batch_size,
        net_infos.len(),
        solve_pool,
    );

    solve_pool
        .install(|| {
            net_infos
                .par_chunks(batch_size)
                .fold(
                    || TemplateAccum {
                        edge_usage: vec![0.0; n_pipes],
                        energy: 0.0,
                        failures: 0,
                    },
                    |mut accum, chunk| {
                        for info in chunk {
                            let Some(source) = source_node(info) else {
                                accum.failures += 1;
                                continue;
                            };
                            let sinks: Vec<usize> = info
                                .pins
                                .iter()
                                .filter(|pin| !pin.is_driver)
                                .map(|pin| pin.node)
                                .collect();
                            if sinks.is_empty() {
                                accum.failures += 1;
                                continue;
                            }
                            let result = solve_net_bresenham_logit(
                                network,
                                source,
                                &sinks,
                                Some(&mut accum.edge_usage),
                            );
                            accum.failures += result.failures;
                            accum.energy += net_path_weight(info, cfg) * result.energy;
                        }
                        accum
                    },
                )
                .reduce(
                    || TemplateAccum {
                        edge_usage: vec![0.0; n_pipes],
                        energy: 0.0,
                        failures: 0,
                    },
                    |mut a, b| {
                        for (dst, src) in a.edge_usage.iter_mut().zip(b.edge_usage) {
                            *dst += src;
                        }
                        a.energy += b.energy;
                        a.failures += b.failures;
                        a
                    },
                )
        })
        .into_solve_accum(net_infos.len())
}

impl TemplateAccum {
    fn into_solve_accum(self, total_solves: usize) -> SolveAccum {
        SolveAccum {
            edge_usage: self.edge_usage,
            energy: self.energy,
            stats: PathStats {
                total_solves,
                total_heap_pops: 0,
                max_heap_pops: 0,
                failures: self.failures,
            },
            diag_corridor_fallback: 0,
            diag_cap_armed: 0,
            diag_solves: 0,
            diag_settle_sum: 0,
            diag_settle_max: 0,
            diag_init_count: 0,
        }
    }
}

struct TemplateSolveResult {
    energy: f64,
    failures: usize,
}

fn solve_net_bresenham_logit(
    network: &PipeNetwork,
    source: usize,
    sinks: &[usize],
    mut edge_usage: Option<&mut [f64]>,
) -> TemplateSolveResult {
    let mut energy = 0.0;
    let mut failures = 0usize;
    for &sink in sinks {
        let templates = build_route_templates(network, source, sink);
        if templates.is_empty() {
            failures += 1;
            continue;
        }

        let mut min_cost = f64::INFINITY;
        let mut costs = Vec::with_capacity(templates.len());
        for path in &templates {
            let cost = template_path_cost(network, path);
            if cost.is_finite() {
                min_cost = min_cost.min(cost);
            }
            costs.push(cost);
        }
        if !min_cost.is_finite() {
            failures += 1;
            continue;
        }

        let mut z = 0.0f64;
        let mut weights = Vec::with_capacity(costs.len());
        for &cost in &costs {
            let w = if cost.is_finite() {
                (-path_solver::LOGIT_THETA * (cost - min_cost)).exp()
            } else {
                0.0
            };
            z += w;
            weights.push(w);
        }
        if z <= 0.0 || !z.is_finite() {
            failures += 1;
            continue;
        }

        // Canonical per-net energy: shortest template-path cost (demand=1 per
        // sink in the Bresenham surrogate). `z`/`weights` still drive the
        // logit edge-usage spread below, but the scalar we report/minimize is
        // the same physical "routed cost" as the dial-logit path uses.
        energy += min_cost;
        if let Some(usage) = edge_usage.as_deref_mut() {
            for (path, weight) in templates.iter().zip(weights) {
                if weight <= 0.0 {
                    continue;
                }
                let p = weight / z;
                for &pipe_idx in path {
                    usage[pipe_idx] += p;
                }
            }
        }
    }

    TemplateSolveResult { energy, failures }
}

fn template_path_cost(network: &PipeNetwork, path: &[usize]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }
    let mut cost = 0.0;
    for &pipe_idx in path {
        let pipe_cost = network.pipe_cost(pipe_idx);
        if !pipe_cost.is_finite() || pipe_cost <= 0.0 {
            return f64::INFINITY;
        }
        cost += pipe_cost;
    }
    cost
}

fn fill_bresenham_surrogate_field(
    network: &PipeNetwork,
    source: usize,
    dist_out: &mut FxHashMap<u32, f32>,
) {
    let src = &network.nodes[source];
    for (node_idx, node) in network.nodes.iter().enumerate() {
        let dx = (node.tile_x - src.tile_x).abs() as f64;
        let dy = (node.tile_y - src.tile_y).abs() as f64;
        let manhattan = dx + dy;
        // Sparse route templates provide the usage estimate. The DCD field is
        // deliberately a cheap smooth surrogate: route length plus a local
        // BPR-like pressure sample around the candidate node.
        dist_out.insert(
            node_idx as u32,
            (manhattan + local_node_pressure(network, node_idx)) as f32,
        );
    }
}

fn local_node_pressure(network: &PipeNetwork, node: usize) -> f64 {
    let model = ResistanceModel;
    let mut pressure = 0.0;
    let mut n = 0usize;
    for &pipe_idx in &network.node_pipes[node] {
        let pipe = &network.pipes[pipe_idx];
        if pipe.capacity <= 0.0 || pipe.base_resistance <= 0.0 {
            continue;
        }
        if pipe.net_count > 0.0 {
            let r_ratio = model.effective_resistance(pipe) / pipe.base_resistance;
            pressure += (r_ratio - 1.0).max(0.0);
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        pressure / n as f64
    }
}

fn build_route_templates(network: &PipeNetwork, source: usize, sink: usize) -> Vec<Vec<usize>> {
    let s = &network.nodes[source];
    let t = &network.nodes[sink];
    let mut templates = Vec::with_capacity(5);

    push_template_from_points(
        network,
        &bresenham_line(s.tile_x, s.tile_y, t.tile_x, t.tile_y),
        &mut templates,
    );
    push_template_from_points(
        network,
        &orthogonal_points(s.tile_x, s.tile_y, t.tile_x, t.tile_y, true),
        &mut templates,
    );
    push_template_from_points(
        network,
        &orthogonal_points(s.tile_x, s.tile_y, t.tile_x, t.tile_y, false),
        &mut templates,
    );

    let span = (t.tile_x - s.tile_x).abs() + (t.tile_y - s.tile_y).abs();
    let offset = (span / 8).clamp(2, 16);
    let (px, py) = perpendicular_step(t.tile_x - s.tile_x, t.tile_y - s.tile_y);
    for sign in [1, -1] {
        let ox = px * offset * sign;
        let oy = py * offset * sign;
        let a = (
            (s.tile_x + ox).clamp(0, network.width - 1),
            (s.tile_y + oy).clamp(0, network.height - 1),
        );
        let b = (
            (t.tile_x + ox).clamp(0, network.width - 1),
            (t.tile_y + oy).clamp(0, network.height - 1),
        );
        let mut points = Vec::new();
        append_points(&mut points, &bresenham_line(s.tile_x, s.tile_y, a.0, a.1));
        append_points(&mut points, &bresenham_line(a.0, a.1, b.0, b.1));
        append_points(&mut points, &bresenham_line(b.0, b.1, t.tile_x, t.tile_y));
        push_template_from_points(network, &points, &mut templates);
    }

    templates.sort();
    templates.dedup();
    templates
}

fn append_points(dst: &mut Vec<(i32, i32)>, src: &[(i32, i32)]) {
    for &p in src {
        if dst.last().copied() != Some(p) {
            dst.push(p);
        }
    }
}

fn perpendicular_step(dx: i32, dy: i32) -> (i32, i32) {
    if dx.abs() >= dy.abs() {
        (0, dy.signum().max(1))
    } else {
        (dx.signum().max(1), 0)
    }
}

fn orthogonal_points(x0: i32, y0: i32, x1: i32, y1: i32, x_first: bool) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    out.push((x0, y0));
    if x_first {
        step_axis(&mut out, x1, y0);
        step_axis(&mut out, x1, y1);
    } else {
        step_axis(&mut out, x0, y1);
        step_axis(&mut out, x1, y1);
    }
    out
}

fn step_axis(out: &mut Vec<(i32, i32)>, x1: i32, y1: i32) {
    let Some(&(mut x, mut y)) = out.last() else {
        return;
    };
    while x != x1 {
        x += (x1 - x).signum();
        out.push((x, y));
    }
    while y != y1 {
        y += (y1 - y).signum();
        out.push((x, y));
    }
}

fn push_template_from_points(
    network: &PipeNetwork,
    points: &[(i32, i32)],
    templates: &mut Vec<Vec<usize>>,
) {
    let mut path = Vec::new();
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x0 == x1 && y0 == y1 {
            continue;
        }
        if !append_orthogonal_step(network, x0, y0, x1, y1, &mut path) {
            return;
        }
    }
    templates.push(path);
}

fn append_orthogonal_step(
    network: &PipeNetwork,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    out: &mut Vec<usize>,
) -> bool {
    let dx = x1 - x0;
    let dy = y1 - y0;
    if dx.abs() + dy.abs() == 1 {
        return append_adjacent_pipe(network, x0, y0, x1, y1, out);
    }
    if dx.abs() == 1 && dy.abs() == 1 {
        let x_mid = (x1, y0);
        let y_mid = (x0, y1);
        let mut path_x = Vec::new();
        let mut path_y = Vec::new();
        let ok_x = append_adjacent_pipe(network, x0, y0, x_mid.0, x_mid.1, &mut path_x)
            && append_adjacent_pipe(network, x_mid.0, x_mid.1, x1, y1, &mut path_x);
        let ok_y = append_adjacent_pipe(network, x0, y0, y_mid.0, y_mid.1, &mut path_y)
            && append_adjacent_pipe(network, y_mid.0, y_mid.1, x1, y1, &mut path_y);
        match (ok_x, ok_y) {
            (true, true) => {
                if template_path_cost(network, &path_x) <= template_path_cost(network, &path_y) {
                    out.extend(path_x);
                } else {
                    out.extend(path_y);
                }
                true
            }
            (true, false) => {
                out.extend(path_x);
                true
            }
            (false, true) => {
                out.extend(path_y);
                true
            }
            (false, false) => false,
        }
    } else {
        false
    }
}

fn append_adjacent_pipe(
    network: &PipeNetwork,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    out: &mut Vec<usize>,
) -> bool {
    if x0 < 0 || x0 >= network.width || y0 < 0 || y0 >= network.height {
        return false;
    }
    if x1 < 0 || x1 >= network.width || y1 < 0 || y1 >= network.height {
        return false;
    }
    let a = network.node_index(x0, y0);
    let b = network.node_index(x1, y1);
    let Some(&pipe_idx) = network.pipe_lookup.get(&pipe_lookup_key(a, b)) else {
        return false;
    };
    out.push(pipe_idx);
    true
}

/// Dump percentiles of `R_eff / base` across the pipe graph for the current
/// iteration. Gated on `NEXTPNR_DIAG=1` so it only runs during sweeps. Keeps
/// everything on one line so the CSV harness can parse it with a grep.
fn report_reff_distribution(network: &PipeNetwork, resistance_model: &ResistanceModel, iter: usize) {
    if std::env::var("NEXTPNR_DIAG").ok().as_deref() != Some("1") {
        return;
    }
    let pipes = &network.pipes;
    if pipes.is_empty() {
        return;
    }
    let mut ratios: Vec<f64> = pipes
        .iter()
        .map(|p| {
            let base = p.base_resistance.max(1e-12);
            resistance_model.effective_resistance(p) / base
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ratios.len();
    let p = |q: f64| ratios[((n as f64 * q) as usize).min(n - 1)];
    let hot101 = ratios.iter().filter(|r| **r > 1.01).count();
    let hot12 = ratios.iter().filter(|r| **r > 1.2).count();
    let hot2 = ratios.iter().filter(|r| **r > 2.0).count();
    eprintln!(
        "[diag] iter={iter} pipes={n} p50={:.4} p90={:.4} p99={:.4} max={:.4} hot1.01={hot101} hot1.2={hot12} hot2={hot2}",
        p(0.50),
        p(0.90),
        p(0.99),
        ratios[n - 1],
    );
}

fn update_effective_conductance(
    network: &mut PipeNetwork,
    solve_pool: &rayon::ThreadPool,
    resistance_model: &ResistanceModel,
    graph_model: GraphModel,
) {
    use super::network::DIST_SCALE;
    if matches!(graph_model, GraphModel::TwoLevelSpan | GraphModel::TwoLevelPip) {
        if graph_model == GraphModel::TwoLevelPip {
            let stats = super::tile_cache::rebuild_span_cost_table_pip(
                network,
                resistance_model,
            );
            let avg_us = if stats.dijkstra_calls > 0 {
                stats.dijkstra_total_us as f64 / stats.dijkstra_calls as f64
            } else {
                0.0
            };
            eprintln!(
                "    SwitchMatrix: type_usage_pairs={} lookups={} hits={} dijkstra_calls={} avg_us={:.1} total_ms={:.1}",
                stats.unique_type_usage_pairs,
                stats.switch_lookups,
                stats.switch_cache_hits,
                stats.dijkstra_calls,
                avg_us,
                stats.dijkstra_total_us as f64 / 1000.0,
            );
        } else {
            super::tile_cache::rebuild_span_cost_table(network, resistance_model);
        }
        network.refresh_flat_edge_costs();
        return;
    }
    network.span_cost_table.reset_disabled(network.pipes.len());
    solve_pool.install(|| {
        let pipes = &mut network.pipes;
        let pipe_costs = &mut network.pipe_costs;
        let pipe_costs_int = &mut network.pipe_costs_int;
        pipes
            .par_iter_mut()
            .zip(pipe_costs.par_iter_mut())
            .zip(pipe_costs_int.par_iter_mut())
            .for_each(|((pipe, cost), cost_int)| {
                let r_eff = resistance_model.effective_resistance(pipe);
                let r = r_eff.max(1e-12);
                pipe.eff_conductance = 1.0 / r;
                *cost = r;
                let scaled = (r * DIST_SCALE).round();
                *cost_int = if scaled >= u32::MAX as f64 {
                    u32::MAX - 1
                } else {
                    (scaled as u32).max(1)
                };
            });
    });
    network.refresh_flat_edge_costs();
}

fn update_net_count_from_usage(
    network: &mut PipeNetwork,
    usage: &[f64],
    blend_alpha: f64,
    solve_pool: &rayon::ThreadPool,
) {
    solve_pool.install(|| {
        network
            .pipes
            .par_iter_mut()
            .zip(usage.par_iter())
            .for_each(|(pipe, &u)| {
                pipe.net_count = blend_alpha * pipe.net_count + (1.0 - blend_alpha) * u;
            });
    });
}

fn solve_distance_cache(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    skip_mask: Option<&[bool]>,
    displacement: Option<&super::displacement::DisplacementTable>,
) -> SolveAccum {
    solve_all_nets_with_displacement(
        network,
        net_infos,
        cfg,
        solve_pool,
        dist_cache,
        false,
        skip_mask,
        displacement,
    )
}

fn solve_usage_and_energy(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    displacement: Option<&super::displacement::DisplacementTable>,
) -> SolveAccum {
    solve_all_nets_with_displacement(
        network,
        net_infos,
        cfg,
        solve_pool,
        dist_cache,
        true,
        None,
        displacement,
    )
}

fn apply_usage_for_next_iter(
    network: &mut PipeNetwork,
    usage: &[f64],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
) {
    update_net_count_from_usage(network, usage, cfg.blend_alpha, solve_pool);
}

fn evaluate_cell_at(
    cell_idx: usize,
    gx: i32,
    gy: i32,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
) -> f64 {
    // Refuse tiles that can't host this cell type. This propagates
    // through every caller (BB, fullscan, bisection) via their
    // existing "pick min finite cost" logic — no per-caller edit needed.
    if !validity.is_valid(cell_idx, gx, gy) {
        return f64::INFINITY;
    }
    if cell_nets.is_empty() {
        return 0.0;
    }

    let node = coord_to_node(network, gx, gy);
    let steiner_w = cfg.steiner_weight;
    let mst_w = cfg.mst_edge_weight;
    let width = network.width as usize;
    // Candidate position in network grid coords. `gx`/`gy` are already in
    // that system (see `coord_to_node` call above), so no conversion.
    let cand_nx = gx as f64;
    let cand_ny = gy as f64;
    let mut cost = 0.0;
    for &(net_idx, pin_idx) in cell_nets {
        let info = &net_infos[net_idx];
        let pin = &info.pins[pin_idx];
        debug_assert_eq!(pin.cell_idx, Some(cell_idx));

        let weight = net_path_weight(info, cfg);

        if !pin.is_driver {
            let d = dist_cache.get(net_idx, node);
            if !d.is_finite() {
                return f64::INFINITY;
            }
            cost += weight * d;
        }

        // 1-Steiner pseudo-node term: pull every pin (driver + sinks) toward
        // the shared per-net centroid. This is the pairwise sink-sink
        // coupling that the driver-anchored star model is missing.
        if steiner_w > 0.0 {
            let cx = dist_cache.steiner_cx[net_idx];
            let cy = dist_cache.steiner_cy[net_idx];
            let dx = (cand_nx - cx).abs();
            let dy = (cand_ny - cy).abs();
            cost += steiner_w * weight * (dx + dy);
        }

        // MST edge pull: net-level cost is `Σ |p_i - p_j|` over MST edges;
        // per-pin contribution is the sum of distances from candidate
        // position to each MST neighbour. No central attractor → no
        // crowding pathology of the centroid term.
        if mst_w > 0.0 && net_idx < dist_cache.mst_neighbors.len() {
            let nbrs = &dist_cache.mst_neighbors[net_idx];
            if pin_idx < nbrs.len() {
                for &nbr_pin in &nbrs[pin_idx] {
                    let nbr_node = info.pins[nbr_pin as usize].node;
                    let nbr_x = (nbr_node % width) as f64;
                    let nbr_y = (nbr_node / width) as f64;
                    let dx = (cand_nx - nbr_x).abs();
                    let dy = (cand_ny - nbr_y).abs();
                    cost += mst_w * weight * (dx + dy);
                }
            }
        }
    }

    cost
}

fn current_cell_node(
    cell_idx: usize,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
) -> usize {
    for &(net_idx, pin_idx) in cell_nets {
        let pin = &net_infos[net_idx].pins[pin_idx];
        if pin.cell_idx == Some(cell_idx) {
            return pin.node;
        }
    }
    0
}

fn fullscan_find_best_position(
    cell_idx: usize,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
) -> (i32, i32) {
    let (pos, _) = fullscan_find_best_position_with_stats(
        cell_idx, cell_nets, net_infos, dist_cache, network, cfg, validity, false,
    );
    pos
}

fn fullscan_find_best_position_with_stats(
    cell_idx: usize,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    collect_stats: bool,
) -> ((i32, i32), Option<PlateauStat>) {
    let cur_node = current_cell_node(cell_idx, cell_nets, net_infos);
    let mut best_x = network.nodes[cur_node].tile_x;
    let mut best_y = network.nodes[cur_node].tile_y;
    let mut best_cost = evaluate_cell_at(
        cell_idx, best_x, best_y, cell_nets, net_infos, dist_cache, network, cfg, validity,
    );

    // Buffer of all scanned finite costs (only when collecting stats).
    let mut cost_buf: Vec<f64> = if collect_stats {
        Vec::with_capacity((network.width as usize) * (network.height as usize))
    } else {
        Vec::new()
    };
    let mut stat = PlateauStat {
        cost_min: f64::INFINITY,
        cost_max: f64::NEG_INFINITY,
        ..PlateauStat::default()
    };

    for gy in 0..network.height {
        for gx in 0..network.width {
            let cost = evaluate_cell_at(
                cell_idx, gx, gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
            );
            if collect_stats {
                diag::update_plateau(&mut stat, cost);
                if cost.is_finite() {
                    cost_buf.push(cost);
                }
            }
            if cost < best_cost {
                best_cost = cost;
                best_x = gx;
                best_y = gy;
            }
        }
    }

    let stats = if collect_stats {
        if best_cost.is_finite() {
            diag::classify_plateau(&mut stat, &cost_buf, best_cost);
        }
        Some(stat)
    } else {
        None
    };

    ((best_x, best_y), stats)
}

fn compute_bounding_boxes(
    cell_x: &[f64],
    cell_y: &[f64],
    net_infos: &[NetInfo],
    network: &PipeNetwork,
) -> BoundingBoxes {
    // Movable BB — from cell_x / cell_y.
    let n = cell_x.len();
    let mut mnx = f64::INFINITY;
    let mut mxx = f64::NEG_INFINITY;
    let mut mny = f64::INFINITY;
    let mut mxy = f64::NEG_INFINITY;
    for i in 0..n {
        if cell_x[i] < mnx {
            mnx = cell_x[i];
        }
        if cell_x[i] > mxx {
            mxx = cell_x[i];
        }
        if cell_y[i] < mny {
            mny = cell_y[i];
        }
        if cell_y[i] > mxy {
            mxy = cell_y[i];
        }
    }
    let movable_w = (mxx - mnx).max(0.0).round() as i32;
    let movable_h = (mxy - mny).max(0.0).round() as i32;
    let area = (movable_w as f64) * (movable_h as f64);
    let fill = if area > 0.0 { n as f64 / area } else { 0.0 };

    // All-pin BB — includes fixed pins too (sampled via net_infos pin nodes).
    let mut amnx = i32::MAX;
    let mut amxx = i32::MIN;
    let mut amny = i32::MAX;
    let mut amxy = i32::MIN;
    for info in net_infos {
        for pin in &info.pins {
            let node = &network.nodes[pin.node];
            if node.tile_x < amnx {
                amnx = node.tile_x;
            }
            if node.tile_x > amxx {
                amxx = node.tile_x;
            }
            if node.tile_y < amny {
                amny = node.tile_y;
            }
            if node.tile_y > amxy {
                amxy = node.tile_y;
            }
        }
    }
    let all_w = (amxx - amnx).max(0);
    let all_h = (amxy - amny).max(0);

    BoundingBoxes {
        movable_w,
        movable_h,
        movable_fill: fill,
        all_w,
        all_h,
    }
}

/// Compute design-level summary for synthesis/mapping sanity check.
/// Returns (summary_lines, fixed_cell_rows).
fn collect_design_summary(
    ctx: &Context,
    network: &PipeNetwork,
    idx_to_cell: &[CellId],
    cell_buckets: &[IdString],
    type_aware: &TypeAwarePlacement,
) -> (Vec<String>, Vec<(String, String, i32, i32, u32)>) {
    use std::collections::BTreeMap;

    let mut summary = Vec::new();
    summary.push(format!("=== Design summary ==="));
    summary.push(format!(
        "chipdb: w={} h={} (tile grid)",
        ctx.chipdb().width(),
        ctx.chipdb().height()
    ));
    summary.push(format!(
        "network: nodes={} pipes={} width={} height={} origin=({},{})",
        network.num_nodes(),
        network.num_pipes(),
        network.width,
        network.height,
        network.x0,
        network.y0,
    ));

    // Count cells in design by bucket.
    let mut design_by_bucket: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    // (total, with_bel, locked)
    let mut total_cells = 0usize;
    let mut cells_with_bel = 0usize;
    let mut locked_cells = 0usize;
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        total_cells += 1;
        let bucket = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
        let has_bel = cell.bel.is_some();
        let locked = cell.bel_strength.is_locked();
        if has_bel {
            cells_with_bel += 1;
        }
        if locked {
            locked_cells += 1;
        }
        let entry = design_by_bucket.entry(bucket).or_insert((0, 0, 0));
        entry.0 += 1;
        if has_bel {
            entry.1 += 1;
        }
        if locked {
            entry.2 += 1;
        }
        let _ = cell_id;
    }

    summary.push(format!(""));
    summary.push(format!("=== Design cell counts ==="));
    summary.push(format!("total alive cells: {}", total_cells));
    summary.push(format!("cells with BEL assigned: {}", cells_with_bel));
    summary.push(format!("cells with LOCKED bel: {}", locked_cells));
    summary.push(format!(
        "{:<12} {:>6} {:>6} {:>6}",
        "bucket", "total", "w_bel", "locked"
    ));
    for (b, (tot, wb, lk)) in &design_by_bucket {
        summary.push(format!("{:<12} {:>6} {:>6} {:>6}", b, tot, wb, lk));
    }

    // Count movable cells by bucket (the subset the placer touches).
    let mut movable_by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    for &b in cell_buckets {
        *movable_by_bucket
            .entry(ctx.name_of(b).to_string())
            .or_insert(0) += 1;
    }
    summary.push(format!(""));
    summary.push(format!("=== Movable cells (placer sees these) ==="));
    summary.push(format!("n_movable: {}", idx_to_cell.len()));
    for (b, c) in &movable_by_bucket {
        summary.push(format!("{:<12} {:>6}", b, c));
    }

    // Chipdb BEL capacity per bucket (from type_aware).
    summary.push(format!(""));
    summary.push(format!(
        "=== Chipdb BEL capacity (from TypeAwarePlacement) ==="
    ));
    for (bucket, cap_map) in &type_aware.tile_capacity {
        let bucket_name = ctx.name_of(*bucket).to_string();
        let n_tiles = cap_map.len();
        let total_bels: u32 = cap_map.values().sum();
        let need = movable_by_bucket.get(&bucket_name).copied().unwrap_or(0);
        summary.push(format!(
            "{:<12} tiles={:>5}  total_bels={:>8}  movable_need={:>5}  ratio={:.6}",
            bucket_name,
            n_tiles,
            total_bels,
            need,
            need as f64 / total_bels.max(1) as f64,
        ));
    }

    // IO / fixed cells positions.
    let mut fixed_rows: Vec<(String, String, i32, i32, u32)> = Vec::new();
    let mut io_xs = Vec::<i32>::new();
    let mut io_ys = Vec::<i32>::new();
    for (cell_id, cell_raw) in ctx.design.iter_alive_cells() {
        let locked = cell_raw.bel_strength.is_locked();
        if !locked {
            continue;
        }
        let Some(bel) = cell_raw.bel else { continue };
        let loc = ctx.bel(bel).loc();
        let bucket = ctx
            .name_of(ctx.resolve_bucket(cell_raw.cell_type))
            .to_string();
        let name = ctx.name_of(cell_raw.name).to_string();
        // Fanout: max of ports that resolve to a net with users.
        let mut fanout = 0u32;
        let cell_view = ctx.cell(cell_id);
        for pin in cell_view.ports() {
            if let Some(net_id) = cell_view.port_net(pin.port) {
                let net = ctx.design.net(net_id);
                let f = net.num_users() as u32;
                if f > fanout {
                    fanout = f;
                }
            }
        }
        fixed_rows.push((name, bucket, loc.x, loc.y, fanout));
        io_xs.push(loc.x);
        io_ys.push(loc.y);
    }

    if !io_xs.is_empty() {
        let mnx = *io_xs.iter().min().unwrap();
        let mxx = *io_xs.iter().max().unwrap();
        let mny = *io_ys.iter().min().unwrap();
        let mxy = *io_ys.iter().max().unwrap();
        summary.push(format!(""));
        summary.push(format!(
            "=== Fixed/locked cell positions (IOs and locked cells) ==="
        ));
        summary.push(format!("n_fixed: {}", io_xs.len()));
        summary.push(format!(
            "BB: x=[{},{}] w={}  y=[{},{}] h={}",
            mnx,
            mxx,
            mxx - mnx,
            mny,
            mxy,
            mxy - mny
        ));
        summary.push(format!(
            "chip frac: w={:.2}%  h={:.2}%",
            100.0 * (mxx - mnx) as f64 / ctx.chipdb().width().max(1) as f64,
            100.0 * (mxy - mny) as f64 / ctx.chipdb().height().max(1) as f64,
        ));

        // Histogram of IO positions by chip quadrant.
        let cx = ctx.chipdb().width() / 2;
        let cy = ctx.chipdb().height() / 2;
        let (mut q00, mut q10, mut q01, mut q11) = (0usize, 0usize, 0usize, 0usize);
        for (x, y) in io_xs.iter().zip(&io_ys) {
            match (*x < cx, *y < cy) {
                (true, true) => q00 += 1,
                (false, true) => q10 += 1,
                (true, false) => q01 += 1,
                (false, false) => q11 += 1,
            }
        }
        summary.push(format!(
            "quadrant counts (chip-center split): SW={} SE={} NW={} NE={}",
            q00, q10, q01, q11
        ));

        // Edge count: IOs within 5% of any edge.
        let ew = (ctx.chipdb().width() as f64 * 0.05).round() as i32;
        let eh = (ctx.chipdb().height() as f64 * 0.05).round() as i32;
        let edge = io_xs
            .iter()
            .zip(&io_ys)
            .filter(|(x, y)| {
                **x <= ew
                    || **x >= ctx.chipdb().width() - ew
                    || **y <= eh
                    || **y >= ctx.chipdb().height() - eh
            })
            .count();
        summary.push(format!("near chip edge (<5%): {}/{}", edge, io_xs.len()));
    }

    // Per-net: fixed-pin presence.
    let mut n_nets_total = 0usize;
    let mut n_nets_with_fixed = 0usize;
    let mut fanout_hist: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, net) in ctx.design.iter_alive_nets() {
        n_nets_total += 1;
        let fanout = net.num_users() as u32;
        let bucket_key = match fanout {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=5 => 5,
            6..=20 => 20,
            21..=50 => 50,
            51..=200 => 200,
            _ => 10_000,
        };
        *fanout_hist.entry(bucket_key).or_insert(0) += 1;
        // Fixed pin iff driver is locked/IO or any user is locked/IO.
        let mut has_fixed = false;
        if let Some(drv) = net.driver() {
            if ctx.design.cell(drv.cell).bel_strength.is_locked() {
                has_fixed = true;
            }
        }
        if !has_fixed {
            for user in net.users() {
                if ctx.design.cell(user.cell).bel_strength.is_locked() {
                    has_fixed = true;
                    break;
                }
            }
        }
        if has_fixed {
            n_nets_with_fixed += 1;
        }
    }
    summary.push(format!(""));
    summary.push(format!("=== Net summary ==="));
    summary.push(format!("total alive nets: {}", n_nets_total));
    summary.push(format!(
        "nets with fixed pin: {} ({:.1}%)",
        n_nets_with_fixed,
        100.0 * n_nets_with_fixed as f64 / n_nets_total.max(1) as f64
    ));
    summary.push(format!("fanout histogram (upper bucket bound):"));
    for (b, c) in &fanout_hist {
        summary.push(format!("  <={:<5}  {:>5}", b, c));
    }

    (summary, fixed_rows)
}

fn collect_cell_metadata(
    ctx: &Context,
    cell_net_map: &CellNetMap,
    net_infos: &[NetInfo],
    idx_to_cell: &[CellId],
) -> Vec<(usize, String, String, u32, u32, u32, bool)> {
    let mut out = Vec::with_capacity(idx_to_cell.len());
    for (ci, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell = ctx.design.cell(cell_id);
        let cell_name = ctx.name_of(cell.name).to_string();
        let bucket = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
        let cell_nets = &cell_net_map.map[ci];
        let n_nets = cell_nets.len() as u32;
        let mut max_fan = 0u32;
        let mut tot_fan = 0u32;
        let mut has_fixed = false;
        for &(net_idx, _) in cell_nets {
            let info = &net_infos[net_idx];
            let fan = info.pins.len().saturating_sub(1) as u32;
            if fan > max_fan {
                max_fan = fan;
            }
            tot_fan += fan;
            if info.has_fixed_pin {
                has_fixed = true;
            }
        }
        out.push((ci, cell_name, bucket, n_nets, max_fan, tot_fan, has_fixed));
    }
    out
}

fn per_net_hpwl(net_infos: &[NetInfo], network: &PipeNetwork) -> (Vec<String>, Vec<u32>, Vec<f64>) {
    let n = net_infos.len();
    let mut names = Vec::with_capacity(n);
    let mut fanout = Vec::with_capacity(n);
    let mut hpwl = Vec::with_capacity(n);
    for info in net_infos {
        names.push(info.debug_name.clone());
        fanout.push(info.pins.len().saturating_sub(1) as u32);
        if info.pins.is_empty() {
            hpwl.push(0.0);
            continue;
        }
        let first = &network.nodes[info.pins[0].node];
        let (mut mnx, mut mxx) = (first.tile_x, first.tile_x);
        let (mut mny, mut mxy) = (first.tile_y, first.tile_y);
        for pin in info.pins.iter().skip(1) {
            let node = &network.nodes[pin.node];
            if node.tile_x < mnx {
                mnx = node.tile_x;
            }
            if node.tile_x > mxx {
                mxx = node.tile_x;
            }
            if node.tile_y < mny {
                mny = node.tile_y;
            }
            if node.tile_y > mxy {
                mxy = node.tile_y;
            }
        }
        hpwl.push((mxx - mnx) as f64 + (mxy - mny) as f64);
    }
    (names, fanout, hpwl)
}

/// Jacobi-style simultaneous DCD.
///
/// Phase 1: for every cell, independently find its best position against the
///   frozen dist_cache and net_infos (no commits).
/// Phase 2: apply damped update to all cells at once:
///     cell_pos ← alpha * best_pos + (1 - alpha) * cell_pos
///
/// The damping (alpha ∈ (0, 1]) prevents overshoot when multiple cells
/// compete for the same region — each cell moves partway toward its ideal,
/// and shared-net conflicts resolve by compromise rather than greedy race.
/// Online log-sum-exp softmin accumulator.
///
/// Tracks the softmin-weighted expected position `(Σ w_i·gx_i, Σ w_i·gy_i) / Σ w_i`
/// where `w_i = exp(-theta · (c_i − c_min))`. Uses a running `anchor` at the
/// lowest cost seen so far so `exp()` stays bounded in `[0, 1]`; when a new
/// lower cost arrives the accumulated sums are rescaled down to keep the
/// anchor fresh. Also tracks the hard argmin for fallback.
/// Number of ranked fallback candidates retained per cell probe. Used when
/// the primary argmin/softmin tile is rejected by the mux-slot tracker:
/// the caller walks these in order and commits the first one that still
/// fits.
const PROBE_TOPK: usize = 8;

#[derive(Clone, Copy, Debug)]
struct SoftminAccumulator {
    anchor: f64,
    s_w: f64,
    s_wx: f64,
    s_wy: f64,
    best_x: i32,
    best_y: i32,
    best_cost: f64,
    theta: f64,
    /// Ranked fallback candidates sorted ascending by cost.
    /// `topk[0]` is the argmin; `topk[topk_len-1]` is the highest-cost
    /// candidate we're still retaining.
    topk: [(f64, i32, i32); PROBE_TOPK],
    topk_len: usize,
}

impl SoftminAccumulator {
    fn new(theta: f64) -> Self {
        Self {
            anchor: f64::INFINITY,
            s_w: 0.0,
            s_wx: 0.0,
            s_wy: 0.0,
            best_x: 0,
            best_y: 0,
            best_cost: f64::INFINITY,
            theta: theta.max(0.0),
            topk: [(f64::INFINITY, 0, 0); PROBE_TOPK],
            topk_len: 0,
        }
    }

    /// Ranked fallback candidates (argmin first, then 2nd-best, ...). Used
    /// by the commit path when the primary pick is rejected by the tracker.
    fn topk(&self) -> &[(f64, i32, i32)] {
        &self.topk[..self.topk_len]
    }

    #[inline]
    fn observe(&mut self, gx: i32, gy: i32, cost: f64) {
        if !cost.is_finite() {
            return;
        }
        if cost < self.best_cost {
            self.best_cost = cost;
            self.best_x = gx;
            self.best_y = gy;
        }
        // Maintain the top-K sorted ascending by cost. Fast path when the
        // buffer has a worse-than-`cost` last entry; otherwise we're not
        // in the K best and can skip.
        if self.topk_len < PROBE_TOPK {
            let mut i = self.topk_len;
            while i > 0 && self.topk[i - 1].0 > cost {
                self.topk[i] = self.topk[i - 1];
                i -= 1;
            }
            self.topk[i] = (cost, gx, gy);
            self.topk_len += 1;
        } else if cost < self.topk[PROBE_TOPK - 1].0 {
            let mut i = PROBE_TOPK - 1;
            self.topk[i] = (cost, gx, gy);
            while i > 0 && self.topk[i - 1].0 > self.topk[i].0 {
                self.topk.swap(i - 1, i);
                i -= 1;
            }
        }
        // If this probe lowers the anchor, shift accumulated sums down so
        // they remain expressed relative to the new anchor.
        if cost < self.anchor {
            if self.anchor.is_finite() && self.theta > 0.0 && self.s_w > 0.0 {
                let shift = self.anchor - cost;
                let scale = (-self.theta * shift).exp();
                if scale.is_finite() {
                    self.s_w *= scale;
                    self.s_wx *= scale;
                    self.s_wy *= scale;
                } else {
                    self.s_w = 0.0;
                    self.s_wx = 0.0;
                    self.s_wy = 0.0;
                }
            }
            self.anchor = cost;
        }
        let w = (-self.theta * (cost - self.anchor)).exp();
        if !w.is_finite() || w <= 0.0 {
            return;
        }
        self.s_w += w;
        self.s_wx += w * gx as f64;
        self.s_wy += w * gy as f64;
    }

    fn argmin(&self) -> (i32, i32) {
        (self.best_x, self.best_y)
    }

    /// Softmin expected position as continuous `(fx, fy)`. Falls back to the
    /// argmin if we saw no finite probe or the weights underflowed.
    fn softmin_continuous(&self) -> (f64, f64) {
        if self.s_w > 0.0 && self.s_w.is_finite() {
            (self.s_wx / self.s_w, self.s_wy / self.s_w)
        } else {
            (self.best_x as f64, self.best_y as f64)
        }
    }

    /// Rounded + grid-clamped softmin tile. Returns argmin when softmin is
    /// disabled (`theta <= 0`) or when the rounded tile is invalid for this
    /// cell (validity mask rejects it).
    fn softmin_tile(
        &self,
        cell_idx: usize,
        validity: &CellValidityMask,
        width: i32,
        height: i32,
    ) -> (i32, i32) {
        if self.theta <= 0.0 {
            return self.argmin();
        }
        let (fx, fy) = self.softmin_continuous();
        let gx = fx.round() as i32;
        let gy = fy.round() as i32;
        let gx = gx.clamp(0, width.saturating_sub(1).max(0));
        let gy = gy.clamp(0, height.saturating_sub(1).max(0));
        if validity.is_valid(cell_idx, gx, gy) {
            (gx, gy)
        } else {
            self.argmin()
        }
    }
}

/// Unified fullscan probe: one pass over the W×H grid computing both the
/// hard argmin and the softmin accumulator under a given `theta`. When
/// `theta <= 0` the softmin result falls back to argmin so callers can use
/// a single code path.
fn fullscan_sweep_probe(
    cell_idx: usize,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    theta: f64,
    collect_stats: bool,
) -> (SoftminAccumulator, Option<PlateauStat>) {
    let cur_node = current_cell_node(cell_idx, cell_nets, net_infos);
    let cur_x = network.nodes[cur_node].tile_x;
    let cur_y = network.nodes[cur_node].tile_y;

    let mut acc = SoftminAccumulator::new(theta);
    // Prime argmin with current position so a fully-INF scan still returns
    // something sensible.
    acc.best_x = cur_x;
    acc.best_y = cur_y;

    let mut cost_buf: Vec<f64> = if collect_stats {
        Vec::with_capacity((network.width as usize) * (network.height as usize))
    } else {
        Vec::new()
    };
    let mut stat = PlateauStat {
        cost_min: f64::INFINITY,
        cost_max: f64::NEG_INFINITY,
        ..PlateauStat::default()
    };

    for gy in 0..network.height {
        for gx in 0..network.width {
            // Mux-slot filter: skip candidates that would exceed destination
            // tile's output-mux capacity. This lets argmin/softmin fall
            // through to the next-best *legal* position automatically, with
            // no INF sentinels.
            if !mux_tracker.would_fit(cell_idx, cur_x, cur_y, gx, gy) {
                continue;
            }
            let cost = evaluate_cell_at(
                cell_idx, gx, gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
            );
            if collect_stats {
                diag::update_plateau(&mut stat, cost);
                if cost.is_finite() {
                    cost_buf.push(cost);
                }
            }
            acc.observe(gx, gy, cost);
        }
    }

    let stats = if collect_stats {
        if acc.best_cost.is_finite() {
            diag::classify_plateau(&mut stat, &cost_buf, acc.best_cost);
        }
        Some(stat)
    } else {
        None
    };
    (acc, stats)
}

fn place_dcd_sweep(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
    theta: f64,
) -> usize {
    let n = cell_x.len();
    let alpha = cfg.jacobi_alpha.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return 0;
    }

    let collect_stats = diag.enabled;
    let use_softmin = cfg.softmin_enabled && theta > 0.0;

    // Single-phase Jacobi: probe against the live (concurrently-updated)
    // tracker, apply damping, and atomically commit the move — all inside
    // the parallel map. The cost landscape (`net_infos`) stays frozen for
    // the sweep; only the mux-slot occupancy is live, so two threads can
    // not both claim the last slot in a tile (CAS in `try_commit`).
    // Each entry returns `(final_gx, final_gy, committed, stats)`.
    let probes: Vec<(i32, i32, bool, Option<PlateauStat>)> = (0..n)
        .into_par_iter()
        .map(|ci| {
            let cell_nets = &cell_net_map.map[ci];
            let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
            if cell_nets.is_empty() {
                return (old_gx, old_gy, false, None);
            }
            let (acc, stats) = fullscan_sweep_probe(
                ci,
                cell_nets,
                net_infos,
                dist_cache,
                network,
                cfg,
                validity,
                mux_tracker,
                if use_softmin { theta } else { 0.0 },
                collect_stats,
            );
            let (primary_gx, primary_gy) = if use_softmin {
                acc.softmin_tile(ci, validity, network.width, network.height)
            } else {
                acc.argmin()
            };
            // Walk the ranked candidate list: first the primary (softmin
            // tile or argmin), then the top-K fallbacks in order of cost.
            // Each candidate is damped against `old` and tried via an
            // atomic `try_commit`; first success wins. Cells stay put only
            // if every candidate fails.
            let try_candidate =
                |cand_gx: i32, cand_gy: i32| -> Option<(i32, i32)> {
                    if cand_gx == old_gx && cand_gy == old_gy {
                        return None;
                    }
                    let new_gx_f =
                        alpha * cand_gx as f64 + (1.0 - alpha) * old_gx as f64;
                    let new_gy_f =
                        alpha * cand_gy as f64 + (1.0 - alpha) * old_gy as f64;
                    let new_gx = new_gx_f.round() as i32;
                    let new_gy = new_gy_f.round() as i32;
                    if new_gx == old_gx && new_gy == old_gy {
                        return None;
                    }
                    if mux_tracker.try_commit(ci, old_gx, old_gy, new_gx, new_gy) {
                        Some((new_gx, new_gy))
                    } else {
                        None
                    }
                };
            if let Some((new_gx, new_gy)) = try_candidate(primary_gx, primary_gy) {
                return (new_gx, new_gy, true, stats);
            }
            for &(_, cand_gx, cand_gy) in acc.topk() {
                if cand_gx == primary_gx && cand_gy == primary_gy {
                    continue;
                }
                if let Some((new_gx, new_gy)) = try_candidate(cand_gx, cand_gy) {
                    return (new_gx, new_gy, true, stats);
                }
            }
            (old_gx, old_gy, false, stats)
        })
        .collect();

    // Record plateau stats serially after the parallel phase.
    if collect_stats {
        for (ci, (_, _, _, stat)) in probes.iter().enumerate() {
            if let Some(s) = stat {
                diag.record_plateau(ci, *s);
            }
        }
    }

    // Apply the committed positions: update cell_x/cell_y and the pin nodes
    // in net_infos. These writes are sequential but cheap compared to the
    // parallel probe+commit pass.
    let mut moved = 0usize;
    let mut final_positions: Vec<(i32, i32)> = Vec::with_capacity(n);
    for (ci, &(new_gx, new_gy, committed, _)) in probes.iter().enumerate() {
        let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
        let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
        if !committed {
            if collect_stats {
                final_positions.push((old_gx, old_gy));
            }
            continue;
        }
        cell_x[ci] = network.net_to_tile(new_gx as f64);
        cell_y[ci] = network.net_to_tile(new_gy as f64);
        let new_node = coord_to_node(network, new_gx, new_gy);
        for &(net_idx, pin_idx) in &cell_net_map.map[ci] {
            net_infos[net_idx].pins[pin_idx].node = new_node;
        }
        if collect_stats {
            diag.record_move(MoveRecord {
                cell_idx: ci,
                old_gx,
                old_gy,
                new_gx,
                new_gy,
            });
            final_positions.push((new_gx, new_gy));
        }
        moved += 1;
    }

    if collect_stats {
        diag.finalize_positions(&final_positions);
    }

    moved
}

/// 2D coupled bisection with a `K x K` coarse seed and a quadtree refinement.
///
/// The seed samples a grid of `K x K` points across the whole chip and picks
/// the cheapest one as the initial `best`. This gives us the correct basin
/// even when the cost surface is multi-modal. The quadtree then refines in a
/// `2q x 2q` window around the seed winner (where `q` is the seed step),
/// recursively probing the four quadrant centers and descending into the best.
///
/// Properties:
///   - Evaluations per cell ≈ K² + 4·log₂(q)  (K=8 gives ~92 probes).
///   - Energy-monotone: only commits a probe when its cost beats the running
///     best, so the returned cost is ≤ `start_cost` by construction.
///   - Couples x and y: the four quadrant probes sample both axes at once, so
///     non-separable minima are visible to the descent.
fn bisect_2d(
    cell_idx: usize,
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    start_x: i32,
    start_y: i32,
    start_cost: f64,
) -> (i32, i32, f64, usize, usize, usize) {
    let w = network.width;
    let h = network.height;
    if w <= 0 || h <= 0 {
        return (start_x, start_y, start_cost, 0, 0, 0);
    }
    let k = cfg.bisect_seed_k.max(2) as i32;
    let max_x = w - 1;
    let max_y = h - 1;
    let step_x = (max_x as f64 / (k - 1).max(1) as f64).max(1.0);
    let step_y = (max_y as f64 / (k - 1).max(1) as f64).max(1.0);

    let eval = |gx: i32, gy: i32| -> f64 {
        evaluate_cell_at(
            cell_idx, gx, gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
        )
    };

    let mut best_x = start_x;
    let mut best_y = start_y;
    let mut best_cost = start_cost;
    let mut n_probes = 0usize;
    let mut n_inf = 0usize;
    let mut n_improve = 0usize;

    // Phase 1: K × K coarse seed.
    for j in 0..k {
        for i in 0..k {
            let gx = ((i as f64) * step_x).round() as i32;
            let gy = ((j as f64) * step_y).round() as i32;
            let gx = gx.clamp(0, max_x);
            let gy = gy.clamp(0, max_y);
            let c = eval(gx, gy);
            n_probes += 1;
            if !c.is_finite() {
                n_inf += 1;
            }
            if c < best_cost {
                best_x = gx;
                best_y = gy;
                best_cost = c;
                n_improve += 1;
            }
        }
    }

    // Phase 2: 2D quadtree within ±q around best.
    let q_seed_x = step_x.round() as i32;
    let q_seed_y = step_y.round() as i32;
    let mut lo_x = (best_x - q_seed_x).max(0);
    let mut hi_x = (best_x + q_seed_x).min(max_x);
    let mut lo_y = (best_y - q_seed_y).max(0);
    let mut hi_y = (best_y + q_seed_y).min(max_y);

    let max_quadtree_iters = cfg.dcd_iters_per_cell.max(1) + 4;
    for _ in 0..max_quadtree_iters {
        if hi_x - lo_x <= 1 && hi_y - lo_y <= 1 {
            break;
        }
        let cx = (lo_x + hi_x) / 2;
        let cy = (lo_y + hi_y) / 2;
        let qx = ((hi_x - lo_x) / 4).max(1);
        let qy = ((hi_y - lo_y) / 4).max(1);

        // Four quadrant centers.
        let probes = [
            ((lo_x + qx).min(max_x), (lo_y + qy).min(max_y)), // SW
            ((hi_x - qx).max(0), (lo_y + qy).min(max_y)),     // SE
            ((lo_x + qx).min(max_x), (hi_y - qy).max(0)),     // NW
            ((hi_x - qx).max(0), (hi_y - qy).max(0)),         // NE
        ];

        let mut winner: Option<(i32, i32, f64, usize)> = None;
        for (idx, &(px, py)) in probes.iter().enumerate() {
            let c = eval(px, py);
            n_probes += 1;
            if !c.is_finite() {
                n_inf += 1;
            }
            if c < best_cost {
                if let Some((_, _, wc, _)) = winner {
                    if c < wc {
                        winner = Some((px, py, c, idx));
                    }
                } else {
                    winner = Some((px, py, c, idx));
                }
            }
        }

        if let Some((wx, wy, wc, idx)) = winner {
            best_x = wx;
            best_y = wy;
            best_cost = wc;
            n_improve += 1;
            // Shrink window to the winning quadrant.
            match idx {
                0 => {
                    hi_x = cx;
                    hi_y = cy;
                } // SW
                1 => {
                    lo_x = cx;
                    hi_y = cy;
                } // SE
                2 => {
                    hi_x = cx;
                    lo_y = cy;
                } // NW
                _ => {
                    lo_x = cx;
                    lo_y = cy;
                } // NE
            }
        } else {
            // No improvement — shrink window around the current best.
            lo_x = (best_x - qx).max(lo_x);
            hi_x = (best_x + qx).min(hi_x);
            lo_y = (best_y - qy).max(lo_y);
            hi_y = (best_y + qy).min(hi_y);
        }
    }

    (best_x, best_y, best_cost, n_probes, n_inf, n_improve)
}

/// 1D bisection on one axis with energy-monotone acceptance.
///
/// `fixed_coord` is the coordinate held constant on the other axis.
/// Re-run Dial-logit loading for one net and overwrite `dist_cache[net_idx]`.
/// Used after a driver cell moves so subsequent cells see the live field.
fn refresh_net_dist_cache(
    net_idx: usize,
    info: &NetInfo,
    dist_cache: &mut DistCache,
    network: &PipeNetwork,
    _cfg: &OptTransPlacerCfg,
    ws: &mut PathSolverWorkspace,
) {
    let Some(source) = source_node(info) else {
        return;
    };
    let n_sinks = info.pins.iter().filter(|p| !p.is_driver).count();
    if n_sinks == 0 {
        return;
    }
    let per_sink = 1.0 / n_sinks as f64;
    let sink_demands: Vec<(usize, f64)> = info
        .pins
        .iter()
        .filter(|pin| !pin.is_driver)
        .map(|pin| (pin.node, per_sink))
        .collect();

    ws.begin_net();
    path_solver::dial_logit_load(network, source, &sink_demands, ws, None);

    let dist_out = dist_cache.row_mut(net_idx);
    dist_out.clear();
    dist_out.reserve(ws.settle_order.len());
    for &node in &ws.settle_order {
        dist_out.insert(node as u32, ws.dist[node] as f32);
    }
    if dist_out.capacity() > 4 * dist_out.len().max(8) {
        dist_out.shrink_to_fit();
    }
}

/// Jacobi DCD sweep using parallel 2D bisection.
///
/// Mirrors `place_dcd_sweep` (Jacobi + fullscan) but swaps the per-cell
/// search to `bisect_2d`, which costs ~112 evals/cell vs fullscan's 16k.
/// The whole sweep evaluates against a frozen `dist_cache` snapshot — no
/// in-sweep refresh. The outer loop updates the field between sweeps.
fn place_dcd_sweep_jacobi_bisection(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
) -> usize {
    let n = cell_x.len();

    // Phase 1: parallel evaluation. Each cell finds its best position against
    // the frozen dist_cache. No commits, no shared mutation.
    let best: Vec<(i32, i32)> = (0..n)
        .into_par_iter()
        .map(|ci| {
            let cell_nets = &cell_net_map.map[ci];
            if cell_nets.is_empty() {
                let gx = network.tile_to_net(cell_x[ci]).round() as i32;
                let gy = network.tile_to_net(cell_y[ci]).round() as i32;
                return (gx, gy);
            }
            let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
            let cur_cost = evaluate_cell_at(
                ci, old_gx, old_gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
            );
            let (bx, by, _bc, _np, _ni, _im) = bisect_2d(
                ci, cell_nets, net_infos, dist_cache, network, cfg, validity, old_gx, old_gy,
                cur_cost,
            );
            (bx, by)
        })
        .collect();

    // Phase 2: serial commit (positions only — no dist_cache refresh).
    let mut moved = 0usize;
    for (ci, &(new_gx, new_gy)) in best.iter().enumerate() {
        let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
        let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
        if new_gx == old_gx && new_gy == old_gy {
            continue;
        }
        if !mux_tracker.try_commit(ci, old_gx, old_gy, new_gx, new_gy) {
            continue;
        }
        cell_x[ci] = network.net_to_tile(new_gx as f64);
        cell_y[ci] = network.net_to_tile(new_gy as f64);
        let new_node = coord_to_node(network, new_gx, new_gy);
        for &(net_idx, pin_idx) in &cell_net_map.map[ci] {
            net_infos[net_idx].pins[pin_idx].node = new_node;
        }
        if diag.enabled {
            diag.record_move(MoveRecord {
                cell_idx: ci,
                old_gx,
                old_gy,
                new_gx,
                new_gy,
            });
        }
        moved += 1;
    }
    moved
}

/// Best-first branch-and-bound search over a per-net region-min pyramid.
///
/// Guaranteed to return the same `(x, y, cost)` that a full grid scan would
/// return under the frozen `dist_cache`. Typically visits orders of
/// magnitude fewer grid points than fullscan because any quadtree cell whose
/// admissible lower bound `L = Σ w_net · min_{node∈cell} dist[net][node]`
/// already exceeds the current best is pruned.
///
/// Entry `(bound, level, cx, cy)` in the min-heap represents: "this
/// pyramid cell covers a rectangle whose true minimum cost is ≥ `bound`".
/// Since bounds are admissible and the heap pops monotonically, the first
/// time a leaf (level 0) is popped its evaluated cost equals the true
/// minimum; we continue popping in case later leaves have smaller cost.
/// When the heap top's `bound ≥ best_cost`, no remaining region can
/// improve — safe to terminate.
/// Min-heap entry for the branch-and-bound pyramid search. Ordered by `bound`
/// ascending; ties by `level` descending so coarser regions expand first
/// (wider exploration).
#[derive(Copy, Clone)]
struct BbNode {
    bound: f64,
    level: i32,
    cx: i32,
    cy: i32,
}
impl PartialEq for BbNode {
    fn eq(&self, o: &Self) -> bool {
        self.bound == o.bound && self.level == o.level
    }
}
impl Eq for BbNode {}
impl Ord for BbNode {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        let a = o
            .bound
            .partial_cmp(&self.bound)
            .unwrap_or(std::cmp::Ordering::Equal);
        if a != std::cmp::Ordering::Equal {
            return a;
        }
        o.level.cmp(&self.level)
    }
}
impl PartialOrd for BbNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

fn bb_active_nets(
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
) -> Vec<(usize, f64)> {
    let mut v = Vec::with_capacity(cell_nets.len());
    for &(net_idx, pin_idx) in cell_nets {
        let info = &net_infos[net_idx];
        if !info.pins[pin_idx].is_driver {
            v.push((net_idx, net_path_weight(info, cfg)));
        }
    }
    v
}

/// Compute the admissible lower bound at pyramid cell `(level, cx, cy)`.
/// `level == 0` reads `dist_cache` directly (true cost); `level >= 1` reads
/// the per-net region-min pyramid.
#[inline]
fn bb_bound(
    active_nets: &[(usize, f64)],
    dist_cache: &DistCache,
    pyramids: &[RegionMinPyramid],
    w: i32,
    level: i32,
    cx: i32,
    cy: i32,
) -> f64 {
    let mut b = 0.0;
    for &(ni, wt) in active_nets {
        let m = if level == 0 {
            let node = (cy as usize) * (w as usize) + (cx as usize);
            dist_cache.get(ni, node)
        } else {
            pyramids[ni].at(level, cx, cy) as f64
        };
        if !m.is_finite() {
            return f64::INFINITY;
        }
        b += wt * m;
    }
    b
}

#[inline]
fn bb_child_dims(pyramids: &[RegionMinPyramid], w: i32, h: i32, child_level: i32) -> (i32, i32) {
    if child_level == 0 {
        (w, h)
    } else {
        let lvl = &pyramids[0].levels[(child_level - 1) as usize];
        (lvl.w, lvl.h)
    }
}

fn bb_2d(
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    pyramids: &[RegionMinPyramid],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    start_x: i32,
    start_y: i32,
    start_cost: f64,
) -> (i32, i32, f64, usize) {
    use std::collections::BinaryHeap;

    let w = network.width;
    let h = network.height;
    if w <= 0 || h <= 0 || cell_nets.is_empty() {
        return (start_x, start_y, start_cost, 0);
    }

    let active_nets = bb_active_nets(cell_nets, net_infos, cfg);
    let max_level = pyramids.first().map(|p| p.max_level()).unwrap_or(0);

    let mut best_x = start_x;
    let mut best_y = start_y;
    let mut best_cost = start_cost;
    let mut n_evals = 0usize;
    let mut pq: BinaryHeap<BbNode> = BinaryHeap::new();

    if max_level >= 1 {
        let b = bb_bound(&active_nets, dist_cache, pyramids, w, max_level, 0, 0);
        if b.is_finite() && b < best_cost {
            pq.push(BbNode {
                bound: b,
                level: max_level,
                cx: 0,
                cy: 0,
            });
        }
    } else {
        for gy in 0..h {
            for gx in 0..w {
                let c = bb_bound(&active_nets, dist_cache, pyramids, w, 0, gx, gy);
                if c.is_finite() && c < best_cost {
                    pq.push(BbNode {
                        bound: c,
                        level: 0,
                        cx: gx,
                        cy: gy,
                    });
                }
            }
        }
    }

    while let Some(node) = pq.pop() {
        if node.bound >= best_cost {
            break;
        }
        if node.level == 0 {
            n_evals += 1;
            if node.bound < best_cost {
                best_cost = node.bound;
                best_x = node.cx;
                best_y = node.cy;
            }
            continue;
        }
        let child_level = node.level - 1;
        let (child_w, child_h) = bb_child_dims(pyramids, w, h, child_level);
        for (dx, dy) in [(0i32, 0i32), (1, 0), (0, 1), (1, 1)] {
            let ccx = 2 * node.cx + dx;
            let ccy = 2 * node.cy + dy;
            if ccx >= child_w || ccy >= child_h {
                continue;
            }
            let b = bb_bound(&active_nets, dist_cache, pyramids, w, child_level, ccx, ccy);
            if b.is_finite() && b < best_cost {
                pq.push(BbNode {
                    bound: b,
                    level: child_level,
                    cx: ccx,
                    cy: ccy,
                });
            }
        }
    }

    (best_x, best_y, best_cost, n_evals)
}

/// Variant of `bb_2d` that returns ALL positions tied at the minimum cost.
fn bb_2d_multi(
    cell_nets: &[(usize, usize)],
    net_infos: &[NetInfo],
    dist_cache: &DistCache,
    pyramids: &[RegionMinPyramid],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    start_x: i32,
    start_y: i32,
    start_cost: f64,
) -> (Vec<(i32, i32)>, f64, usize) {
    use std::collections::BinaryHeap;

    let w = network.width;
    let h = network.height;
    if w <= 0 || h <= 0 || cell_nets.is_empty() {
        return (vec![(start_x, start_y)], start_cost, 0);
    }

    let active_nets = bb_active_nets(cell_nets, net_infos, cfg);
    let max_level = pyramids.first().map(|p| p.max_level()).unwrap_or(0);
    let mut best_cost = start_cost;
    let mut tied: Vec<(i32, i32)> = vec![(start_x, start_y)];
    let mut n_evals = 0usize;
    let mut pq: BinaryHeap<BbNode> = BinaryHeap::new();

    if max_level >= 1 {
        let b = bb_bound(&active_nets, dist_cache, pyramids, w, max_level, 0, 0);
        if b.is_finite() && b <= best_cost {
            pq.push(BbNode {
                bound: b,
                level: max_level,
                cx: 0,
                cy: 0,
            });
        }
    } else {
        for gy in 0..h {
            for gx in 0..w {
                let c = bb_bound(&active_nets, dist_cache, pyramids, w, 0, gx, gy);
                if c.is_finite() && c <= best_cost {
                    pq.push(BbNode {
                        bound: c,
                        level: 0,
                        cx: gx,
                        cy: gy,
                    });
                }
            }
        }
    }

    while let Some(node) = pq.pop() {
        if node.bound > best_cost {
            break;
        }
        if node.level == 0 {
            n_evals += 1;
            if node.bound < best_cost {
                best_cost = node.bound;
                tied.clear();
                tied.push((node.cx, node.cy));
            } else if node.bound == best_cost
                && !tied.iter().any(|&(x, y)| x == node.cx && y == node.cy)
            {
                tied.push((node.cx, node.cy));
            }
            continue;
        }
        let child_level = node.level - 1;
        let (child_w, child_h) = bb_child_dims(pyramids, w, h, child_level);
        for (dx, dy) in [(0i32, 0i32), (1, 0), (0, 1), (1, 1)] {
            let ccx = 2 * node.cx + dx;
            let ccy = 2 * node.cy + dy;
            if ccx >= child_w || ccy >= child_h {
                continue;
            }
            let b = bb_bound(&active_nets, dist_cache, pyramids, w, child_level, ccx, ccy);
            if b.is_finite() && b <= best_cost {
                pq.push(BbNode {
                    bound: b,
                    level: child_level,
                    cx: ccx,
                    cy: ccy,
                });
            }
        }
    }

    if start_cost > best_cost {
        tied.retain(|&(x, y)| (x, y) != (start_x, start_y));
    }

    (tied, best_cost, n_evals)
}

/// Branch-and-bound DCD sweep.
///
/// Process cells in `iter_order` (topo forward or reversed). For each cell:
/// 1. Find ALL cost-tied optimum candidates via `bb_2d_multi`.
/// 2. Prefer the current position when it is tied; otherwise use lexicographic order.
/// 3. Commit cell to the winner.
fn place_dcd_sweep_jacobi_bb(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    pyramids: &[RegionMinPyramid],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
    iter_order: &[usize],
) -> usize {
    let want_perf = std::env::var("NPNR_OT_BB_PERF").ok().as_deref() == Some("1");

    // Phase A (parallel): compute each cell's tied-cost candidate list against
    // the frozen dist_cache. This is the expensive step and is embarrassingly
    // parallel — all cells read the same frozen caches and write only into
    // their own Vec slot.
    let tied_per_cell: Vec<(Vec<(i32, i32)>, usize)> = iter_order
        .par_iter()
        .map(|&ci| {
            let cell_nets = &cell_net_map.map[ci];
            if cell_nets.is_empty() {
                let gx = network.tile_to_net(cell_x[ci]).round() as i32;
                let gy = network.tile_to_net(cell_y[ci]).round() as i32;
                return (vec![(gx, gy)], 0);
            }
            let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
            let cur_cost = evaluate_cell_at(
                ci, old_gx, old_gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
            );
            let (tied, _best_cost, n_evals) = bb_2d_multi(
                cell_nets, net_infos, dist_cache, pyramids, network, cfg, old_gx, old_gy, cur_cost,
            );
            (tied, n_evals)
        })
        .collect();

    // Phase B (sequential): commit in topo order.
    let mut moved = 0usize;
    let mut total_ties = 0usize;
    let mut total_evals = 0usize;

    for (order_idx, &ci) in iter_order.iter().enumerate() {
        let cell_nets = &cell_net_map.map[ci];
        if cell_nets.is_empty() {
            continue;
        }

        let (tied, n_evals) = &tied_per_cell[order_idx];
        total_evals += n_evals;
        if tied.len() > 1 {
            total_ties += 1;
        }
        if tied.is_empty() {
            // No valid candidate position under the current dist_cache and
            // validity mask — leave cell where it is.
            continue;
        }

        let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
        let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
        let (new_gx, new_gy) = if tied.iter().any(|&(x, y)| x == old_gx && y == old_gy) {
            (old_gx, old_gy)
        } else {
            tied.iter().copied().min().unwrap()
        };
        if new_gx != old_gx || new_gy != old_gy {
            if !mux_tracker.try_commit(ci, old_gx, old_gy, new_gx, new_gy) {
                // Destination's mux slots are full — pick the next tied
                // candidate that fits, preferring candidates that are not
                // the current position.
                let mut accepted = false;
                for &(cx, cy) in tied.iter() {
                    if cx == old_gx && cy == old_gy {
                        accepted = true;
                        break;
                    }
                    if mux_tracker.try_commit(ci, old_gx, old_gy, cx, cy) {
                        cell_x[ci] = network.net_to_tile(cx as f64);
                        cell_y[ci] = network.net_to_tile(cy as f64);
                        let new_node = coord_to_node(network, cx, cy);
                        for &(net_idx, pin_idx) in cell_nets {
                            net_infos[net_idx].pins[pin_idx].node = new_node;
                        }
                        if diag.enabled {
                            diag.record_move(MoveRecord {
                                cell_idx: ci,
                                old_gx,
                                old_gy,
                                new_gx: cx,
                                new_gy: cy,
                            });
                        }
                        moved += 1;
                        accepted = true;
                        break;
                    }
                }
                if !accepted {
                    // All tied candidates full — leave cell at old position.
                }
                continue;
            }
            cell_x[ci] = network.net_to_tile(new_gx as f64);
            cell_y[ci] = network.net_to_tile(new_gy as f64);
            let new_node = coord_to_node(network, new_gx, new_gy);
            for &(net_idx, pin_idx) in cell_nets {
                net_infos[net_idx].pins[pin_idx].node = new_node;
            }
            if diag.enabled {
                diag.record_move(MoveRecord {
                    cell_idx: ci,
                    old_gx,
                    old_gy,
                    new_gx,
                    new_gy,
                });
            }
            moved += 1;
        }
    }

    if want_perf {
        eprintln!(
            "    bb: moved={} total_evals={} cells_with_ties={}",
            moved, total_evals, total_ties,
        );
    }

    moved
}

/// Sequential DCD sweep with bisection search and live dist_cache refresh.
///
/// Cells are processed in topological order (drivers before sinks). For each
/// cell we bisect on x, then on y, accepting only strict energy decreases.
/// After committing, nets where this cell is the driver have their Dial soft
/// costs recomputed so the next cell evaluates against the live field. Pipe
/// usage / R_eff stay frozen within the sweep; the outer loop refreshes them
/// between sweeps.
fn place_dcd_sweep_sequential_bisection(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &mut DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
) -> usize {
    let n_nodes = network.num_nodes();
    let n_pipes = network.num_pipes();
    let mut ws = PathSolverWorkspace::new(n_nodes, n_pipes);
    let mut moved = 0usize;
    let max_iters = cfg.dcd_iters_per_cell.max(1);
    let want_perf = std::env::var("NPNR_OT_SEQ_PERF").ok().as_deref() == Some("1");
    let mut t_search_ns: u128 = 0;
    let mut t_refresh_ns: u128 = 0;
    let mut n_refreshes: usize = 0;
    let mut n_driver_moves: usize = 0;

    for &ci in &cell_net_map.topo_order {
        let cell_nets = &cell_net_map.map[ci];
        if cell_nets.is_empty() {
            continue;
        }

        let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
        let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
        let cur_cost = evaluate_cell_at(
            ci, old_gx, old_gy, cell_nets, net_infos, dist_cache, network, cfg, validity,
        );
        let _ = max_iters;

        let t0 = std::time::Instant::now();
        let (new_gx, new_gy, _best_cost, _np, _ni, _im) = bisect_2d(
            ci, cell_nets, net_infos, dist_cache, network, cfg, validity, old_gx, old_gy, cur_cost,
        );
        if want_perf {
            t_search_ns += t0.elapsed().as_nanos();
        }

        if new_gx == old_gx && new_gy == old_gy {
            continue;
        }
        if !mux_tracker.try_commit(ci, old_gx, old_gy, new_gx, new_gy) {
            continue;
        }

        // Commit: update position, pin nodes.
        cell_x[ci] = network.net_to_tile(new_gx as f64);
        cell_y[ci] = network.net_to_tile(new_gy as f64);
        let new_node = coord_to_node(network, new_gx, new_gy);
        for &(net_idx, pin_idx) in cell_nets {
            net_infos[net_idx].pins[pin_idx].node = new_node;
        }

        // Region-aware refresh: only rebuild dist_cache when the cell crosses
        // a coarse region boundary. Within-region moves leave the resistance
        // field's apparent cost approximately unchanged, so we can defer the
        // path refresh to the between-sweep solve.
        let r = cfg.bisect_refresh_region.max(1);
        let crossed_region = (old_gx / r) != (new_gx / r) || (old_gy / r) != (new_gy / r);
        let t1 = std::time::Instant::now();
        let mut driver_count = 0usize;
        if crossed_region {
            for &(net_idx, pin_idx) in cell_nets {
                if net_infos[net_idx].pins[pin_idx].is_driver {
                    refresh_net_dist_cache(
                        net_idx,
                        &net_infos[net_idx],
                        dist_cache,
                        network,
                        cfg,
                        &mut ws,
                    );
                    driver_count += 1;
                }
            }
        }
        if want_perf {
            t_refresh_ns += t1.elapsed().as_nanos();
            n_refreshes += driver_count;
            if driver_count > 0 {
                n_driver_moves += 1;
            }
        }

        if diag.enabled {
            diag.record_move(MoveRecord {
                cell_idx: ci,
                old_gx,
                old_gy,
                new_gx,
                new_gy,
            });
        }
        moved += 1;
    }

    if want_perf {
        eprintln!(
            "    seq_perf: search={:.1}ms refresh={:.1}ms n_refresh={} driver_moves={} total_moves={}",
            t_search_ns as f64 / 1e6,
            t_refresh_ns as f64 / 1e6,
            n_refreshes,
            n_driver_moves,
            moved,
        );
    }

    moved
}

#[derive(Clone, Copy, Debug)]
struct ObjState {
    energy: f64,
    line: f64,
    friction: f64,
    max_overflow: f64,
    n_overflow: usize,
    overflow_excess: f64,
}

pub fn run_inner_outer(
    ctx: &mut Context,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    _idx_to_cell: &[CellId],
    alive_net_ids: &[NetId],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    resistance_model: &ResistanceModel,
    type_aware: &TypeAwarePlacement,
    cell_buckets: &[IdString],
    cell_pin_weights: &[f64],
    phys_max_x: f64,
    phys_max_y: f64,
    phys_grid_w: usize,
    phys_grid_h: usize,
) -> f64 {
    let n = cell_x.len();
    let n_nodes = network.num_nodes();
    let mut dist_cache = DistCache::new(0, n_nodes);
    let max_iter = cfg.max_outer_iters.max(1);
    let mut diag_ctx = DiagCtx::from_env();

    // Per-cell validity mask. Rejects candidate positions where the cell's
    // bucket has no compatible BEL, propagating through every helper via
    // `evaluate_cell_at` returning f64::INFINITY.
    let validity =
        CellValidityMask::build(ctx, _idx_to_cell, type_aware, network.width, network.height);
    let validity = &validity;

    // Runtime mux-slot tracker: hard commit-time gate that rejects moves into
    // tiles whose output-mux slots are full. Capacity is derived from the
    // chipdb (count of distinct output-node roots per tile). Seeded from the
    // current grid positions of every external-fanout cell.
    let mut mux_tracker = MuxSlotTracker::build(ctx, _idx_to_cell, network.x0, network.y0);
    {
        let init_gx: Vec<i32> = cell_x
            .iter()
            .map(|&x| network.tile_to_net(x).round() as i32)
            .collect();
        let init_gy: Vec<i32> = cell_y
            .iter()
            .map(|&y| network.tile_to_net(y).round() as i32)
            .collect();
        mux_tracker.seed_positions(&init_gx, &init_gy);
    }
    mux_tracker.report("post-seed");
    if diag_ctx.enabled {
        eprintln!(
            "DCD diag: instrumentation ON, output dir /tmp/claude/ot_diag (or $NPNR_OT_DIAG_DIR)"
        );
    }

    eprintln!(
        "DCD placer: {} cells, {} nodes, outer_iters={}, dcd_iters_per_cell={}, graph_model={:?}",
        n, n_nodes, max_iter, cfg.dcd_iters_per_cell, cfg.graph_model,
    );
    if cfg.softmin_enabled {
        eprintln!(
            "Softmin position update: enabled, theta {:.3} -> {:.3} (anneal over {} iters)",
            cfg.softmin_theta_start, cfg.softmin_theta_end, max_iter,
        );
    } else {
        eprintln!("Softmin position update: disabled (hard argmin commit)");
    }

    // DCD is a continuation method: per-iter BPR field strengthens as usage
    // feeds back. We trust the schedule and take the converged state — no
    // rollback to "best" iterate. Convergence is measured by per-cell
    // displacement between consecutive iters (a fixed-point signal that does
    // not depend on the BPR ramp). NPNR_OT_DISP_TOL sets the average-tile
    // threshold (default 0.5); NPNR_OT_DISP_STALL sets how many consecutive
    // sub-threshold iters trigger stop (default 3).
    let disp_tol = std::env::var("NPNR_OT_DISP_TOL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.max(0.0))
        .unwrap_or(0.5);
    let max_disp_stalls = std::env::var("NPNR_OT_DISP_STALL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let mut disp_stalls = 0usize;
    let mut prev_cell_x: Option<Vec<f64>> = None;
    let mut prev_cell_y: Option<Vec<f64>> = None;
    let mut prev_state: Option<ObjState> = None;
    let mut last_state: Option<ObjState> = None;
    // Per-net pin-node signature captured at the end of the previous outer
    // iter's post-solve. Compared against the current iter's pre-solve pin
    // layout to decide which nets' `dist_cache` rows can be reused.
    let mut prev_solve_pin_sigs: Vec<Vec<usize>> = Vec::new();
    for outer in 0..max_iter {
        let t_outer = std::time::Instant::now();
        // Exponential θ anneal: `theta_start → theta_end` over `max_iter`
        // outer iterations. `theta_iter = theta_start · (end/start)^(i/(N-1))`.
        // At iter 0 → theta_start (broad, exploratory); at last iter →
        // theta_end (sharp, argmin-like).
        let theta_iter = if !cfg.softmin_enabled {
            0.0
        } else if max_iter <= 1 {
            cfg.softmin_theta_end
        } else {
            let t = outer as f64 / (max_iter as f64 - 1.0);
            let s = cfg.softmin_theta_start.max(1e-9);
            let e = cfg.softmin_theta_end.max(1e-9);
            s * (e / s).powf(t)
        };
        let mut net_infos = collect_net_infos_simple(
            ctx,
            alive_net_ids,
            cell_to_idx,
            cell_x,
            cell_y,
            network,
            cfg,
        );
        let cell_net_map = CellNetMap::build(&net_infos, n);
        let prev_dist_cache_shape = (dist_cache.n_nets, dist_cache.n_nodes);
        dist_cache.ensure_shape(net_infos.len(), n_nodes);
        let dist_cache_was_realloced = prev_dist_cache_shape
            != (dist_cache.n_nets, dist_cache.n_nodes)
            || outer == 0;

        // Refresh per-net 1-Steiner centroids before the Jacobi sweep so
        // every `evaluate_cell_at` call in this iter uses a hub consistent
        // with the current pin layout. Skip when the term is disabled.
        if cfg.steiner_weight > 0.0 {
            compute_steiner_centroids(&mut dist_cache, &net_infos, network);
        }
        if cfg.mst_edge_weight > 0.0 {
            compute_mst_neighbors(&mut dist_cache, &net_infos, network);
        }

        // Skip-refresh: reuse last iter's dist_cache row for any net whose
        // pin node signature is identical to the pin layout from the
        // previous outer iter's post-solve. BPR drift on pipe costs makes
        // the cached labels mildly stale, but within the trust region of
        // the DCD sweep's argmin this is empirically negligible.
        let skip_mask: Option<Vec<bool>> =
            if dist_cache_was_realloced || prev_solve_pin_sigs.len() != net_infos.len() {
                None
            } else {
                Some(
                    net_infos
                        .iter()
                        .enumerate()
                        .map(|(i, info)| {
                            let sig = &prev_solve_pin_sigs[i];
                            if sig.len() != info.pins.len() {
                                return false;
                            }
                            info.pins
                                .iter()
                                .zip(sig.iter())
                                .all(|(pin, &node)| pin.node == node)
                        })
                        .collect(),
                )
            };
        let n_skipped = skip_mask
            .as_ref()
            .map(|m| m.iter().filter(|b| **b).count())
            .unwrap_or(0);

        let t_refresh = std::time::Instant::now();
        update_effective_conductance(network, solve_pool, resistance_model, cfg.graph_model);
        report_reff_distribution(network, resistance_model, outer);
        let displacement_table = if matches!(
            cfg.graph_model,
            GraphModel::DisplacementPure
                | GraphModel::DisplacementSparse
                | GraphModel::CorridorWarmstart
        ) {
            super::displacement::DisplacementTable::build(network)
        } else {
            None
        };
        let rss_before_pre_solve = process_rss_kb();
        let pre_solve = solve_distance_cache(
            network,
            &net_infos,
            cfg,
            solve_pool,
            &mut dist_cache,
            skip_mask.as_deref(),
            displacement_table.as_ref(),
        );
        let pre_refresh_ms = t_refresh.elapsed().as_millis();
        let (entries_pre, capacity_pre, est_pre) = dist_cache.memory_stats();
        eprintln!(
            "    rss_probe[outer={}]: pre_refresh_rss={:.0}MB post_pre_solve_rss={:.0}MB cache_entries={} cache_cap={} cache_est_mb={:.0}",
            outer,
            rss_before_pre_solve as f64 / 1024.0,
            process_rss_kb() as f64 / 1024.0,
            entries_pre,
            capacity_pre,
            est_pre as f64 / (1024.0 * 1024.0),
        );
        {
            let solves = pre_solve.diag_solves.max(1);
            eprintln!(
                "    pre_solve_diag[outer={}]: solves={} fallback={} ({:.1}%) cap_armed={} ({:.1}%) settle_avg={:.0} settle_max={} init_count={}",
                outer,
                pre_solve.diag_solves,
                pre_solve.diag_corridor_fallback,
                100.0 * pre_solve.diag_corridor_fallback as f64 / solves as f64,
                pre_solve.diag_cap_armed,
                100.0 * pre_solve.diag_cap_armed as f64 / solves as f64,
                pre_solve.diag_settle_sum as f64 / solves as f64,
                pre_solve.diag_settle_max,
                pre_solve.diag_init_count,
            );
        }

        // Cell-placement-by-tile-type histogram (iter 0 only). Confirms that
        // LUT/FF cells land on CLB tiles, not BRAM/DSP/IO columns.
        if outer == 0 {
            use rustc_hash::FxHashMap as _Map;
            let mut by_tt: _Map<String, usize> = _Map::default();
            let coarsen_i = network.coarsen as i32;
            for i in 0..n {
                let gx = network.tile_to_net(cell_x[i]).round() as i32;
                let gy = network.tile_to_net(cell_y[i]).round() as i32;
                // Coarse -> fine center tile
                let fx = gx * coarsen_i;
                let fy = gy * coarsen_i;
                let fx = fx.clamp(0, ctx.chipdb().width() - 1);
                let fy = fy.clamp(0, ctx.chipdb().height() - 1);
                let tile = ctx.chipdb().tile_by_xy(fx, fy);
                let name = ctx.chipdb().tile_type_name(tile).to_string();
                *by_tt.entry(name).or_insert(0) += 1;
            }
            let mut items: Vec<_> = by_tt.into_iter().collect();
            items.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("  Cell placement by tile type (iter 0, {} cells):", n);
            for (name, cnt) in items.iter().take(15) {
                eprintln!("    {}: {}", name, cnt);
            }
        }

        // HPWL vs routed-star sanity check. Runs RIGHT after the pre-sweep
        // solve so `dist_cache.dist[net][node]` is consistent
        // with the current pin.node values in net_infos. The sweep below will
        // update pin.node, so running this after the sweep compares stale
        // dist_cache entries against post-sweep pin positions — meaningless.
        if std::env::var("NPNR_OT_HPWL_CHECK").ok().as_deref() == Some("1") {
            let mut n_valid = 0usize;
            let mut sum_hpwl = 0.0f64;
            let mut sum_star = 0.0f64;
            let mut n_ratio_gt2 = 0usize;
            let mut n_ratio_gt5 = 0usize;
            let mut n_ratio_lt05 = 0usize;
            let mut n_ratio_lt02 = 0usize;
            let mut ratios: Vec<(f64, usize, f64, f64, u32)> = Vec::new();
            for (net_idx, info) in net_infos.iter().enumerate() {
                if info.pins.is_empty() {
                    continue;
                }
                let Some(first) = info.pins.first() else {
                    continue;
                };
                let first_node = &network.nodes[first.node];
                let (mut mnx, mut mxx) = (first_node.tile_x, first_node.tile_x);
                let (mut mny, mut mxy) = (first_node.tile_y, first_node.tile_y);
                for pin in info.pins.iter().skip(1) {
                    let node = &network.nodes[pin.node];
                    if node.tile_x < mnx {
                        mnx = node.tile_x;
                    }
                    if node.tile_x > mxx {
                        mxx = node.tile_x;
                    }
                    if node.tile_y < mny {
                        mny = node.tile_y;
                    }
                    if node.tile_y > mxy {
                        mxy = node.tile_y;
                    }
                }
                let hpwl = (mxx - mnx) as f64 + (mxy - mny) as f64;
                let mut star = 0.0f64;
                let mut ok = true;
                let mut fanout = 0u32;
                for pin in &info.pins {
                    if pin.is_driver {
                        continue;
                    }
                    fanout += 1;
                    let d = dist_cache.get(net_idx, pin.node);
                    if !d.is_finite() {
                        ok = false;
                        break;
                    }
                    star += d;
                }
                if !ok || fanout == 0 {
                    continue;
                }
                n_valid += 1;
                sum_hpwl += hpwl;
                sum_star += star;
                if star > 0.5 {
                    let r = hpwl / star;
                    if r > 2.0 {
                        n_ratio_gt2 += 1;
                    }
                    if r > 5.0 {
                        n_ratio_gt5 += 1;
                    }
                    if r < 0.5 {
                        n_ratio_lt05 += 1;
                    }
                    if r < 0.2 {
                        n_ratio_lt02 += 1;
                    }
                    ratios.push((r, net_idx, hpwl, star, fanout));
                }
            }
            let mean_hpwl = if n_valid > 0 {
                sum_hpwl / n_valid as f64
            } else {
                0.0
            };
            let mean_star = if n_valid > 0 {
                sum_star / n_valid as f64
            } else {
                0.0
            };
            let global_ratio = if sum_star > 0.0 {
                sum_hpwl / sum_star
            } else {
                0.0
            };
            eprintln!(
                "    HPWL_check (pre-sweep): n_valid={} sum_hpwl={:.0} sum_star={:.0} global_ratio={:.3} mean_hpwl={:.1} mean_star={:.1}  hpwl/star buckets: >5x={} >2x={} <0.5x={} <0.2x={}",
                n_valid, sum_hpwl, sum_star, global_ratio, mean_hpwl, mean_star,
                n_ratio_gt5, n_ratio_gt2, n_ratio_lt05, n_ratio_lt02,
            );
            ratios.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let n_show = ratios.len().min(10);
            if n_show > 0 {
                eprintln!(
                    "    Top {} nets by hpwl/star (HPWL overstates routing):",
                    n_show
                );
                for (r, ni, h, s, f) in ratios.iter().take(n_show) {
                    eprintln!(
                        "      net={} fanout={} hpwl={:.0} star={:.1} ratio={:.2}  name={}",
                        ni, f, h, s, r, net_infos[*ni].debug_name,
                    );
                }
            }
            ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if !ratios.is_empty() {
                let n_show_lo = ratios.len().min(10);
                eprintln!(
                    "    Top {} nets by star/hpwl (HPWL understates routing):",
                    n_show_lo
                );
                for (r, ni, h, s, f) in ratios.iter().take(n_show_lo) {
                    eprintln!(
                        "      net={} fanout={} hpwl={:.0} star={:.1} ratio={:.2}  name={}",
                        ni, f, h, s, r, net_infos[*ni].debug_name,
                    );
                }
            }
        }

        // Debug: compare DCD bisection vs full scan on iter 0. When
        // NPNR_OT_BIS_FS_DIAG is set, dump a per-cell CSV.
        if outer == 0 {
            let want_csv = std::env::var("NPNR_OT_BIS_FS_DIAG").ok().as_deref() == Some("1");
            let mut rows: Vec<String> = Vec::new();
            if want_csv {
                rows.push("ci,cur_x,cur_y,cur_cost,fs_x,fs_y,fs_cost,bis_x,bis_y,bis_cost,d_bis_fs,d_cur_fs,d_cur_bis,cost_gap_bis_fs,cost_gap_cur_fs,n_nets,n_finite_positions,pct_finite_positions,n_finite_per_net_min,n_finite_per_net_max,cur_finite,bis_finite,fs_finite,n_probes,n_probes_inf,n_improve_steps".to_string());
            }
            // Precompute, per cell, the count of (x,y) positions where ALL of its
            // nets have finite dist (i.e. where evaluate_cell_at would return
            // finite cost). This gives the "searchable region" size.
            let grid_area = (network.width as usize) * (network.height as usize);
            let mut mismatch = 0usize;
            let mut sum_d = 0i64;
            let mut max_d = 0i32;
            let mut sum_gap = 0.0f64;
            let mut max_gap = 0.0f64;
            let mut n_examined = 0usize;
            for ci in 0..n {
                let cell_nets = &cell_net_map.map[ci];
                if cell_nets.is_empty() {
                    continue;
                }
                let cur_gx_tmp = network.tile_to_net(cell_x[ci]).round() as i32;
                let cur_gy_tmp = network.tile_to_net(cell_y[ci]).round() as i32;
                let cur_cost_tmp = evaluate_cell_at(
                    ci,
                    cur_gx_tmp,
                    cur_gy_tmp,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                );
                let (bis_x, bis_y, _bis_c, n_probes, n_probes_inf, n_improve_steps) = bisect_2d(
                    ci,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                    cur_gx_tmp,
                    cur_gy_tmp,
                    cur_cost_tmp,
                );
                let (fs_x, fs_y) = fullscan_find_best_position(
                    ci,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                );
                let bis_cost = evaluate_cell_at(
                    ci,
                    bis_x,
                    bis_y,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                );
                let fs_cost = evaluate_cell_at(
                    ci,
                    fs_x,
                    fs_y,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                );
                let cur_gx = network.tile_to_net(cell_x[ci]).round() as i32;
                let cur_gy = network.tile_to_net(cell_y[ci]).round() as i32;
                let cur_cost = evaluate_cell_at(
                    ci,
                    cur_gx,
                    cur_gy,
                    cell_nets,
                    &net_infos,
                    &dist_cache,
                    network,
                    cfg,
                    validity,
                );
                let d_bis_fs = (bis_x - fs_x).abs() + (bis_y - fs_y).abs();
                let d_cur_fs = (cur_gx - fs_x).abs() + (cur_gy - fs_y).abs();
                let d_cur_bis = (cur_gx - bis_x).abs() + (cur_gy - bis_y).abs();
                let gap_bis_fs = (bis_cost - fs_cost).max(0.0);
                let gap_cur_fs = (cur_cost - fs_cost).max(0.0);
                if bis_x != fs_x || bis_y != fs_y {
                    mismatch += 1;
                }
                sum_d += d_bis_fs as i64;
                if d_bis_fs > max_d {
                    max_d = d_bis_fs;
                }
                if gap_bis_fs.is_finite() {
                    sum_gap += gap_bis_fs;
                    if gap_bis_fs > max_gap {
                        max_gap = gap_bis_fs;
                    }
                }
                n_examined += 1;
                if want_csv {
                    // Count positions where every net touching this cell has
                    // finite dist. Expensive (O(grid * n_nets)) but one-off.
                    let mut n_finite_positions = 0usize;
                    let mut per_net_counts: Vec<usize> = Vec::with_capacity(cell_nets.len());
                    for &(net_idx, pin_idx) in cell_nets.iter() {
                        let pin = &net_infos[net_idx].pins[pin_idx];
                        if pin.is_driver {
                            per_net_counts.push(grid_area);
                            continue;
                        }
                        // Sparse rows only hold finite settled entries, so
                        // every key counts.
                        let c = dist_cache.row(net_idx).len();
                        per_net_counts.push(c);
                    }
                    for gy in 0..network.height {
                        for gx in 0..network.width {
                            let node = coord_to_node(network, gx, gy);
                            let mut all_finite = true;
                            for &(net_idx, pin_idx) in cell_nets.iter() {
                                let pin = &net_infos[net_idx].pins[pin_idx];
                                if pin.is_driver {
                                    continue;
                                }
                                if !dist_cache.get(net_idx, node).is_finite() {
                                    all_finite = false;
                                    break;
                                }
                            }
                            if all_finite {
                                n_finite_positions += 1;
                            }
                        }
                    }
                    let pct_finite = 100.0 * n_finite_positions as f64 / grid_area as f64;
                    let min_per_net = per_net_counts.iter().copied().min().unwrap_or(0);
                    let max_per_net = per_net_counts.iter().copied().max().unwrap_or(0);
                    let cur_finite = cur_cost.is_finite() as u8;
                    let bis_finite = bis_cost.is_finite() as u8;
                    let fs_finite = fs_cost.is_finite() as u8;
                    rows.push(format!(
                        "{},{},{},{:.3},{},{},{:.3},{},{},{:.3},{},{},{},{:.3},{:.3},{},{},{:.2},{},{},{},{},{},{},{},{}",
                        ci, cur_gx, cur_gy, cur_cost,
                        fs_x, fs_y, fs_cost,
                        bis_x, bis_y, bis_cost,
                        d_bis_fs, d_cur_fs, d_cur_bis,
                        gap_bis_fs, gap_cur_fs,
                        cell_nets.len(),
                        n_finite_positions, pct_finite,
                        min_per_net, max_per_net,
                        cur_finite, bis_finite, fs_finite,
                        n_probes, n_probes_inf, n_improve_steps,
                    ));
                }
            }
            let avg_d = if n_examined > 0 {
                sum_d as f64 / n_examined as f64
            } else {
                0.0
            };
            let avg_gap = if n_examined > 0 {
                sum_gap / n_examined as f64
            } else {
                0.0
            };
            eprintln!(
                "    Bisection vs fullscan (iter 0, {} cells): mismatch={}/{}  avg_manhattan={:.2}  max_manhattan={}  avg_cost_gap={:.3}  max_cost_gap={:.3}",
                n_examined, mismatch, n_examined, avg_d, max_d, avg_gap, max_gap,
            );
            if want_csv {
                let path = std::env::var("NPNR_OT_BIS_FS_DIAG_PATH")
                    .unwrap_or_else(|_| "/tmp/bis_fs_diag.csv".to_string());
                if let Err(e) = std::fs::write(&path, rows.join("\n")) {
                    eprintln!("    could not write {}: {}", path, e);
                } else {
                    eprintln!("    wrote per-cell diag to {}", path);
                }
            }
        }

        // Congestion diagnostic: how many pipes are actually congested?
        {
            let total_pipes = network.pipes.len();
            let mut n_used = 0usize;
            let mut n_congested = 0usize; // usage > 50% capacity
            let mut n_saturated = 0usize; // usage > 90% capacity
            let mut max_util = 0.0f64;
            let mut sum_util = 0.0f64;
            let mut n_with_cap = 0usize;
            // Overflow breakdown: absolute excess (net_count - capacity) for
            // pipes where net_count > capacity. Distinguishes "just 1 net too
            // many" (mild) from "15 nets on a cap-1 pipe" (severe / barrier
            // rejecting outright).
            let mut n_over = 0usize; // net_count > capacity
            let mut max_excess = 0.0f64; // max(net_count - capacity) in wires
            let mut sum_excess = 0.0f64;
            let mut excess_1 = 0usize; // excess == 1
            let mut excess_2_5 = 0usize; // 2..=5
            let mut excess_6_20 = 0usize; // 6..=20
            let mut excess_gt20 = 0usize; // >20
                                          // Capacity bucketing of OVERLOADED pipes to see if saturation hits
                                          // skinny (cap=1..2) wires vs fat pipes.
            let mut over_cap1 = 0usize;
            let mut over_cap2 = 0usize;
            let mut over_cap3_5 = 0usize;
            let mut over_cap_gt5 = 0usize;
            // Span breakdown of OVERLOADED pipes. Shows whether saturation
            // concentrates on span-1 (local wires) or long-range shortcuts.
            let mut over_span1 = 0usize;
            let mut over_span2_3 = 0usize;
            let mut over_span4_6 = 0usize;
            let mut over_span7_12 = 0usize;
            let mut over_span_gt12 = 0usize;
            // Cross-tab: excess=1 by cap tier (answers "is excess-1 on cap=1 or fat?")
            let mut exc1_cap1 = 0usize;
            let mut exc1_cap2_5 = 0usize;
            let mut exc1_cap_gt5 = 0usize;
            // Worst offender
            let mut worst_nc = 0.0f64;
            let mut worst_cap = 0.0f64;
            let mut worst_span = 0i32;
            for pipe in &network.pipes {
                if pipe.capacity <= 0.0 {
                    continue;
                }
                n_with_cap += 1;
                let util = pipe.net_count / pipe.capacity;
                if pipe.net_count > 0.0 {
                    n_used += 1;
                }
                if util > 0.5 {
                    n_congested += 1;
                }
                if util > 0.9 {
                    n_saturated += 1;
                }
                if util > max_util {
                    max_util = util;
                }
                sum_util += util;
                let nc = pipe.net_count;
                let cap = pipe.capacity;
                if nc > cap {
                    n_over += 1;
                    let excess = nc - cap;
                    sum_excess += excess;
                    // Span: 1 for InterTile; |dx|+|dy| for LongRange.
                    let span = match pipe.pipe_type {
                        super::network::PipeType::InterTile(_) => 1i32,
                        super::network::PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                    };
                    if excess > max_excess {
                        max_excess = excess;
                        worst_nc = pipe.net_count;
                        worst_cap = pipe.capacity;
                        worst_span = span;
                    }
                    if excess <= 1.0 {
                        excess_1 += 1;
                        let cap_i = cap.round() as i64;
                        match cap_i {
                            1 => exc1_cap1 += 1,
                            2..=5 => exc1_cap2_5 += 1,
                            _ => exc1_cap_gt5 += 1,
                        }
                    } else if excess <= 5.0 {
                        excess_2_5 += 1;
                    } else if excess <= 20.0 {
                        excess_6_20 += 1;
                    } else {
                        excess_gt20 += 1;
                    }
                    match cap.round() as i64 {
                        1 => over_cap1 += 1,
                        2 => over_cap2 += 1,
                        3..=5 => over_cap3_5 += 1,
                        _ => over_cap_gt5 += 1,
                    }
                    match span {
                        1 => over_span1 += 1,
                        2..=3 => over_span2_3 += 1,
                        4..=6 => over_span4_6 += 1,
                        7..=12 => over_span7_12 += 1,
                        _ => over_span_gt12 += 1,
                    }
                }
            }
            let avg_util = if n_with_cap > 0 {
                sum_util / n_with_cap as f64
            } else {
                0.0
            };
            let avg_excess = if n_over > 0 {
                sum_excess / n_over as f64
            } else {
                0.0
            };
            eprintln!(
                "    Overflow: over={} ({:.1}% of used) avg_excess={:.2} max_excess={:.2} (nc={:.2} cap={:.0} span={})  excess_buckets: <=1={} 1-5={} 5-20={} >20={}  by_cap: cap1={} cap2={} cap3-5={} cap>5={}  by_span: s1={} s2-3={} s4-6={} s7-12={} s>12={}  le1_cross: cap1={} cap2-5={} cap>5={}",
                n_over,
                100.0 * n_over as f64 / n_used.max(1) as f64,
                avg_excess, max_excess, worst_nc, worst_cap, worst_span,
                excess_1, excess_2_5, excess_6_20, excess_gt20,
                over_cap1, over_cap2, over_cap3_5, over_cap_gt5,
                over_span1, over_span2_3, over_span4_6, over_span7_12, over_span_gt12,
                exc1_cap1, exc1_cap2_5, exc1_cap_gt5,
            );
            eprintln!(
                "    Congestion: pipes={} used={} congested(>50%)={} saturated(>90%)={} max_util={:.2} avg_util={:.4}",
                total_pipes, n_used, n_congested, n_saturated, max_util, avg_util,
            );

            if std::env::var("NPNR_OT_HOT_NETS").ok().as_deref() == Some("1") {
                let mut ranked: Vec<(f64, usize, f64, f64, i32)> = Vec::new();
                for (pi, pipe) in network.pipes.iter().enumerate() {
                    if pipe.capacity <= 0.0 || pipe.net_count <= pipe.capacity {
                        continue;
                    }
                    let span = match pipe.pipe_type {
                        super::network::PipeType::InterTile(_) => 1i32,
                        super::network::PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                    };
                    ranked.push((
                        pipe.net_count - pipe.capacity,
                        pi,
                        pipe.net_count,
                        pipe.capacity,
                        span,
                    ));
                }
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                for (excess, pi, nc, cap, span) in ranked.iter().take(5) {
                    let node_from = &network.nodes[network.pipes[*pi].from];
                    let node_to = &network.nodes[network.pipes[*pi].to];
                    let tt_from = ctx
                        .chipdb()
                        .tile_type_name(ctx.chipdb().tile_by_xy(node_from.tile_x, node_from.tile_y))
                        .to_string();
                    let tt_to = ctx
                        .chipdb()
                        .tile_type_name(ctx.chipdb().tile_by_xy(node_to.tile_x, node_to.tile_y))
                        .to_string();
                    eprintln!(
                        "    HotPipe[{}]: excess={:.2} usage={:.2} cap={:.0} span={} endpoints=({},{}:{})<->({},{}:{})",
                        pi, excess, nc, cap, span,
                        node_from.tile_x, node_from.tile_y, tt_from,
                        node_to.tile_x, node_to.tile_y, tt_to,
                    );
                }
            }
        }

        if outer == 0 && diag_ctx.enabled {
            let rows = collect_cell_metadata(ctx, &cell_net_map, &net_infos, _idx_to_cell);
            diag_ctx.dump_cell_metadata(&rows);
            let (summary_lines, fixed_rows) =
                collect_design_summary(ctx, network, _idx_to_cell, cell_buckets, type_aware);
            diag_ctx.dump_design_summary(&summary_lines);
            diag_ctx.dump_fixed_cells(&fixed_rows);
        }

        diag_ctx.sweep_begin(n, outer);
        let t_dcd = std::time::Instant::now();
        let moved = match cfg.sweep_mode {
            super::config::SweepMode::JacobiFullscan => place_dcd_sweep(
                &mut net_infos,
                &cell_net_map,
                &dist_cache,
                cell_x,
                cell_y,
                network,
                cfg,
                validity,
                &mux_tracker,
                &mut diag_ctx,
                theta_iter,
            ),
            super::config::SweepMode::SequentialBisection => place_dcd_sweep_sequential_bisection(
                &mut net_infos,
                &cell_net_map,
                &mut dist_cache,
                cell_x,
                cell_y,
                network,
                cfg,
                validity,
                &mux_tracker,
                &mut diag_ctx,
            ),
            super::config::SweepMode::JacobiBisection => place_dcd_sweep_jacobi_bisection(
                &mut net_infos,
                &cell_net_map,
                &dist_cache,
                cell_x,
                cell_y,
                network,
                cfg,
                validity,
                &mux_tracker,
                &mut diag_ctx,
            ),
            super::config::SweepMode::JacobiBB => {
                let want_perf = std::env::var("NPNR_OT_BB_PERF").ok().as_deref() == Some("1");

                // Forward pass: build pyramids from current dist_cache, then
                // optimize every cell in topological order (drivers before
                // sinks).
                let t_pyr = std::time::Instant::now();
                let pyramids = region_min::build_all(
                    &dist_cache.rows,
                    dist_cache.n_nodes,
                    network.width,
                    network.height,
                );
                let pyr_ms = t_pyr.elapsed().as_millis();
                if want_perf {
                    let bytes: usize = pyramids.iter().map(|p| p.bytes()).sum();
                    eprintln!(
                        "    bb_pyramid_build: {}ms, {} nets, {:.1} MB",
                        pyr_ms,
                        pyramids.len(),
                        bytes as f64 / (1024.0 * 1024.0),
                    );
                }
                // First-iter verification: BB must match fullscan argmin
                // (same cost). Different (x,y) is allowed when the cost surface
                // has ties.
                if outer == 0 && std::env::var("NPNR_OT_BB_VS_FS_DIAG").ok().as_deref() == Some("1")
                {
                    let mut cost_mismatch = 0usize;
                    let mut pos_mismatch = 0usize;
                    let mut max_cost_gap = 0.0f64;
                    let mut sum_cost_gap = 0.0f64;
                    for ci in 0..n {
                        let cell_nets = &cell_net_map.map[ci];
                        if cell_nets.is_empty() {
                            continue;
                        }
                        let ogx = network.tile_to_net(cell_x[ci]).round() as i32;
                        let ogy = network.tile_to_net(cell_y[ci]).round() as i32;
                        let cur = evaluate_cell_at(
                            ci,
                            ogx,
                            ogy,
                            cell_nets,
                            &net_infos,
                            &dist_cache,
                            network,
                            cfg,
                            validity,
                        );
                        let (bx, by, bc, _ne) = bb_2d(
                            cell_nets,
                            &net_infos,
                            &dist_cache,
                            &pyramids,
                            network,
                            cfg,
                            ogx,
                            ogy,
                            cur,
                        );
                        let (fx, fy) = fullscan_find_best_position(
                            ci,
                            cell_nets,
                            &net_infos,
                            &dist_cache,
                            network,
                            cfg,
                            validity,
                        );
                        let fc = evaluate_cell_at(
                            ci,
                            fx,
                            fy,
                            cell_nets,
                            &net_infos,
                            &dist_cache,
                            network,
                            cfg,
                            validity,
                        );
                        if bx != fx || by != fy {
                            pos_mismatch += 1;
                        }
                        let gap = (bc - fc).abs();
                        if gap > 1e-9 {
                            cost_mismatch += 1;
                        }
                        if gap.is_finite() {
                            if gap > max_cost_gap {
                                max_cost_gap = gap;
                            }
                            sum_cost_gap += gap;
                        }
                    }
                    let avg = sum_cost_gap / n.max(1) as f64;
                    eprintln!(
                        "    BB vs fullscan (iter 0, {} cells): cost_mismatch={}  pos_mismatch={}  avg_cost_gap={:.6}  max_cost_gap={:.6}",
                        n, cost_mismatch, pos_mismatch, avg, max_cost_gap,
                    );
                }
                place_dcd_sweep_jacobi_bb(
                    &mut net_infos,
                    &cell_net_map,
                    &dist_cache,
                    &pyramids,
                    cell_x,
                    cell_y,
                    network,
                    cfg,
                    validity,
                    &mux_tracker,
                    &mut diag_ctx,
                    &cell_net_map.topo_order,
                )
            }
        };
        let dcd_ms = t_dcd.elapsed().as_millis();

        common::clamp_positions(cell_x, cell_y, phys_max_x, phys_max_y);

        let t_post_refresh = std::time::Instant::now();
        let post_net_infos = collect_net_infos_simple(
            ctx,
            alive_net_ids,
            cell_to_idx,
            cell_x,
            cell_y,
            network,
            cfg,
        );
        let post_cell_net_map = CellNetMap::build(&post_net_infos, n);
        let (state_energy, refresh_ms, solve_stats) = if cfg.path_model == PathModel::BresenhamLogit
        {
            let post_solve = solve_bresenham_usage_only(network, &post_net_infos, cfg, solve_pool);
            apply_usage_for_next_iter(network, &post_solve.edge_usage, cfg, solve_pool);
            let refresh_ms = pre_refresh_ms + t_post_refresh.elapsed().as_millis();

            let mut solve_stats = pre_solve.stats;
            solve_stats.total_solves += post_solve.stats.total_solves;
            solve_stats.failures += post_solve.stats.failures;
            (post_solve.energy, refresh_ms, solve_stats)
        } else {
            dist_cache.ensure_shape(post_net_infos.len(), n_nodes);
            let rss_before_post = process_rss_kb();
            let post_solve = solve_usage_and_energy(
                network,
                &post_net_infos,
                cfg,
                solve_pool,
                &mut dist_cache,
                displacement_table.as_ref(),
            );
            let rss_after_post = process_rss_kb();
            apply_usage_for_next_iter(network, &post_solve.edge_usage, cfg, solve_pool);
            let refresh_ms = pre_refresh_ms + t_post_refresh.elapsed().as_millis();
            eprintln!(
                "    rss_probe[outer={}]: pre_post_solve={:.0}MB post_post_solve={:.0}MB after_apply={:.0}MB",
                outer,
                rss_before_post as f64 / 1024.0,
                rss_after_post as f64 / 1024.0,
                process_rss_kb() as f64 / 1024.0,
            );

            let mut solve_stats = pre_solve.stats;
            solve_stats.total_solves += post_solve.stats.total_solves;
            solve_stats.total_heap_pops += post_solve.stats.total_heap_pops;
            solve_stats.max_heap_pops = solve_stats
                .max_heap_pops
                .max(post_solve.stats.max_heap_pops);
            solve_stats.failures += post_solve.stats.failures;
            // Capture post-solve pin layout: this is what the next iter's
            // pre-solve will see as "current" pin state, so comparing against
            // this tells us which nets can skip re-solving.
            prev_solve_pin_sigs.clear();
            prev_solve_pin_sigs.extend(
                post_net_infos
                    .iter()
                    .map(|info| info.pins.iter().map(|p| p.node).collect::<Vec<usize>>()),
            );
            (post_solve.energy, refresh_ms, solve_stats)
        };

        let line = demand::continuous_line_estimate(ctx, cell_to_idx, cell_x, cell_y, network);
        let friction = super::congestion::compute_friction_energy(network);
        let (max_overflow, n_overflow, overflow_excess) = type_aware.compute_overflow(
            cell_buckets,
            cell_pin_weights,
            cell_x,
            cell_y,
            phys_grid_w,
            phys_grid_h,
        );

        let state = ObjState {
            energy: state_energy,
            line,
            friction,
            max_overflow,
            n_overflow,
            overflow_excess,
        };

        // Continuation step: trust the schedule, take this iter's state. The
        // diag `accepted` flag is true when this iter's line dropped versus
        // the previous iter (placement quality improved); `d_friction` is the
        // step delta. These are display-only — they do not gate progress.
        let accepted = prev_state.map_or(true, |prev| state.line < prev.line);
        let d_energy = prev_state.map_or(0.0, |prev| state.energy - prev.energy);
        let d_friction = prev_state.map_or(0.0, |prev| state.friction - prev.friction);
        if !mux_tracker.is_legal() {
            mux_tracker.report(&format!("sweep {} LEAK", outer));
        }
        // Average per-cell Manhattan displacement vs the previous iter's
        // committed positions. At a fixed point cells stop moving (or only
        // jitter sub-tile via softmin). Threshold-stalling on this is the
        // proper convergence signal for a continuation method.
        let disp_avg = match (&prev_cell_x, &prev_cell_y) {
            (Some(px), Some(py)) => {
                let mut sum = 0.0f64;
                for i in 0..n {
                    sum += (cell_x[i] - px[i]).abs();
                    sum += (cell_y[i] - py[i]).abs();
                }
                if n == 0 {
                    0.0
                } else {
                    sum / n as f64
                }
            }
            _ => f64::INFINITY,
        };
        if disp_avg < disp_tol {
            disp_stalls += 1;
        } else {
            disp_stalls = 0;
        }
        prev_cell_x = Some(cell_x.to_vec());
        prev_cell_y = Some(cell_y.to_vec());
        prev_state = Some(state);
        last_state = Some(state);

        // Diagnostic: per-net HPWL (tile coords via network nodes) and sweep summary.
        if diag_ctx.enabled {
            let (net_names, net_fanout, net_hpwl) = per_net_hpwl(&post_net_infos, network);
            let bb = compute_bounding_boxes(cell_x, cell_y, &post_net_infos, network);
            diag_ctx.sweep_end(
                n,
                line,
                friction,
                max_overflow,
                n_overflow,
                overflow_excess,
                moved,
                accepted,
                d_friction,
                &net_names,
                &net_fanout,
                &net_hpwl,
                &bb,
            );
        }

        // Energy decomposition: how much of the current energy is the
        // underlying wire-length cost vs. the congestion penalty. For each
        // pipe, `base_resistance * net_count` is the part the pipe would
        // cost if R_eff were at its floor (no congestion), while
        // `R_eff * net_count` is what it actually costs now. The difference
        // is the congestion contribution. This is weighted by `net_count`
        // rather than real tree traversals, so it is an approximation of
        // the exact tree-cost decomposition — but it tracks R_eff and
        // usage directly and is cheap per iter.
        let mut sum_base = 0.0f64;
        let mut sum_eff = 0.0f64;
        let mut n_sat = 0usize;
        let mut max_util = 0.0f64;
        for (pipe_idx, pipe) in network.pipes.iter().enumerate() {
            let nc = pipe.net_count;
            if nc <= 0.0 {
                continue;
            }
            let r_eff = network.pipe_cost(pipe_idx);
            sum_base += pipe.base_resistance * nc;
            sum_eff += r_eff * nc;
            if pipe.capacity > 0.0 {
                let u = nc / pipe.capacity;
                if u > max_util {
                    max_util = u;
                }
                if u > 0.9 {
                    n_sat += 1;
                }
            }
        }
        let cong_share = if sum_eff > 0.0 {
            (sum_eff - sum_base) / sum_eff
        } else {
            0.0
        };
        // R_eff / base ratio distribution on USED pipes. Buckets quantify
        // which congestion regime each pipe is in; pool-flipping across
        // iters is what drives energy oscillation.
        let mut r_lo = 0usize; // <=2x base (healthy)
        let mut r_med = 0usize; // 2x..10x
        let mut r_hi = 0usize; // 10x..100x
        let mut r_sat = 0usize; // >=100x (near / at eps clamp)
        for (pipe_idx, pipe) in network.pipes.iter().enumerate() {
            if pipe.net_count == 0.0 || pipe.base_resistance <= 0.0 {
                continue;
            }
            let r_eff = network.pipe_cost(pipe_idx);
            let ratio = r_eff / pipe.base_resistance;
            if ratio < 2.0 {
                r_lo += 1;
            } else if ratio < 10.0 {
                r_med += 1;
            } else if ratio < 100.0 {
                r_hi += 1;
            } else {
                r_sat += 1;
            }
        }
        eprintln!(
            "    E_decomp: base={:.1} eff={:.1} cong={:.1} cong_share={:.1}% sat_pipes={} max_util={:.2}  R_buckets: 1-2x={} 2-10x={} 10-100x={} >=100x={}",
            sum_base, sum_eff, sum_eff - sum_base, cong_share * 100.0, n_sat, max_util,
            r_lo, r_med, r_hi, r_sat,
        );
        if network.span_cost_table.enabled {
            let s = &network.span_cost_table.stats;
            let hit_rate = if s.lookups > 0 {
                100.0 * s.hits as f64 / s.lookups as f64
            } else {
                0.0
            };
            eprintln!(
                "    SpanCache: epoch={} entries={} lookups={} hits={} misses={} hit_rate={:.1}%",
                s.epoch, s.entries, s.lookups, s.hits, s.misses, hit_rate,
            );
        }

        // Random pair-swap probe: tests whether the current placement is a
        // coordinate-descent local minimum (exact single-cell argmin) but NOT
        // a coupled-move minimum. For N random pairs (i, j), compute whether
        // swapping P_i <-> P_j would reduce sink-side cost under the frozen
        // dist_cache. Uses `evaluate_cell_at` which sums Σ_sinks dist_cache[net][node]
        // and skips driver-side cost. If many pairs show improvement, the
        // plateau is a CD trap; if few, the plateau is driven by driver-
        // blindness or missing cluster moves.
        if let Some(n_probe) = std::env::var("NPNR_OT_SWAP_PROBE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            if n_probe > 0 && n >= 2 {
                // xorshift64 (no external rand dep needed for a diagnostic)
                let mut rng_state = cfg
                    .seed
                    .wrapping_add(outer as u64)
                    .wrapping_add(0x9E37_79B9_7F4A_7C15);
                if rng_state == 0 {
                    rng_state = 1;
                }
                let mut xs = || -> u64 {
                    let mut x = rng_state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    rng_state = x;
                    x
                };
                let mut n_tested = 0usize;
                let mut n_improve = 0usize;
                let mut sum_improve = 0.0f64;
                let mut max_improve = 0.0f64;
                let mut best_pair: Option<(usize, usize, f64)> = None;
                let mut sum_delta = 0.0f64;
                for _ in 0..n_probe {
                    let i = (xs() as usize) % n;
                    let mut j = (xs() as usize) % n;
                    while j == i {
                        j = (xs() as usize) % n;
                    }
                    let nets_i = &post_cell_net_map.map[i];
                    let nets_j = &post_cell_net_map.map[j];
                    if nets_i.is_empty() || nets_j.is_empty() {
                        continue;
                    }
                    let xi = network.tile_to_net(cell_x[i]).round() as i32;
                    let yi = network.tile_to_net(cell_y[i]).round() as i32;
                    let xj = network.tile_to_net(cell_x[j]).round() as i32;
                    let yj = network.tile_to_net(cell_y[j]).round() as i32;
                    let pre_i = evaluate_cell_at(
                        i,
                        xi,
                        yi,
                        nets_i,
                        &post_net_infos,
                        &dist_cache,
                        network,
                        cfg,
                        validity,
                    );
                    let pre_j = evaluate_cell_at(
                        j,
                        xj,
                        yj,
                        nets_j,
                        &post_net_infos,
                        &dist_cache,
                        network,
                        cfg,
                        validity,
                    );
                    let post_i = evaluate_cell_at(
                        i,
                        xj,
                        yj,
                        nets_i,
                        &post_net_infos,
                        &dist_cache,
                        network,
                        cfg,
                        validity,
                    );
                    let post_j = evaluate_cell_at(
                        j,
                        xi,
                        yi,
                        nets_j,
                        &post_net_infos,
                        &dist_cache,
                        network,
                        cfg,
                        validity,
                    );
                    if !(pre_i.is_finite()
                        && pre_j.is_finite()
                        && post_i.is_finite()
                        && post_j.is_finite())
                    {
                        continue;
                    }
                    let delta = (post_i + post_j) - (pre_i + pre_j);
                    n_tested += 1;
                    sum_delta += delta;
                    if delta < -1e-9 {
                        n_improve += 1;
                        let imp = -delta;
                        sum_improve += imp;
                        if imp > max_improve {
                            max_improve = imp;
                            best_pair = Some((i, j, imp));
                        }
                    }
                }
                let pct = if n_tested > 0 {
                    100.0 * n_improve as f64 / n_tested as f64
                } else {
                    0.0
                };
                let avg_imp = if n_improve > 0 {
                    sum_improve / n_improve as f64
                } else {
                    0.0
                };
                let avg_delta = if n_tested > 0 {
                    sum_delta / n_tested as f64
                } else {
                    0.0
                };
                let bp = best_pair
                    .map(|(i, j, d)| format!("best=(ci={},cj={},d=-{:.2})", i, j, d))
                    .unwrap_or_else(|| "best=none".to_string());
                eprintln!(
                    "    SwapProbe: tested={} improving={} ({:.1}%) avg_delta={:+.3} avg_improve={:.2} max_improve={:.2} {}",
                    n_tested, n_improve, pct, avg_delta, avg_imp, max_improve, bp,
                );
            }
        }

        eprintln!(
            "  DCD {:3}: nets={} line={:.0} energy={:.3} dE={:+.3} friction={:.1} bins={:.1}x({}) excess={:.1} moved={} disp={:.3} solves={} skip={} pops={} theta={:.3} refresh={}ms dcd={}ms total={}ms",
            outer,
            post_net_infos.len(),
            line,
            state.energy,
            d_energy,
            friction,
            max_overflow,
            n_overflow,
            overflow_excess,
            moved,
            disp_avg,
            solve_stats.total_solves,
            n_skipped,
            solve_stats.total_heap_pops,
            theta_iter,
            refresh_ms,
            dcd_ms,
            t_outer.elapsed().as_millis(),
        );

        let (entries, capacity, est_bytes) = dist_cache.memory_stats();
        eprintln!(
            "    cache_stats: entries={} capacity={} est_mb={:.0} rss_mb={:.0}",
            entries,
            capacity,
            est_bytes as f64 / (1024.0 * 1024.0),
            process_rss_kb() as f64 / 1024.0,
        );

        if moved == 0 {
            eprintln!("  DCD converged: no cells moved in iteration {}", outer);
            break;
        }
        if disp_stalls >= max_disp_stalls {
            eprintln!(
                "  DCD converged: avg per-cell displacement < {:.3} tile for {} iters",
                disp_tol, disp_stalls,
            );
            break;
        }
    }

    diag_ctx.finalize(network);

    let final_state = last_state.unwrap_or(ObjState {
        energy: 0.0,
        line: 0.0,
        friction: 0.0,
        max_overflow: 0.0,
        n_overflow: 0,
        overflow_excess: 0.0,
    });
    eprintln!(
        "DCD done: energy={:.3} line={:.0} friction={:.1} bins={:.1}x({}) excess={:.1}",
        final_state.energy,
        final_state.line,
        final_state.friction,
        final_state.max_overflow,
        final_state.n_overflow,
        final_state.overflow_excess,
    );
    eprintln!(
        "MuxSlotTracker: {} commit rejections over entire DCD",
        mux_tracker.rejected(),
    );
    mux_tracker.report("post-DCD");
    final_state.energy
}

#[cfg(test)]
mod softmin_tests {
    use super::SoftminAccumulator;

    #[test]
    fn collapses_to_argmin_at_large_theta() {
        // Three probes, distinct costs. theta=1000 forces softmin ≈ argmin.
        let mut acc = SoftminAccumulator::new(1000.0);
        acc.observe(5, 5, 10.0);
        acc.observe(10, 10, 1.0); // unique min
        acc.observe(0, 0, 5.0);
        let (fx, fy) = acc.softmin_continuous();
        assert!((fx - 10.0).abs() < 1e-6, "fx={}", fx);
        assert!((fy - 10.0).abs() < 1e-6, "fy={}", fy);
        assert_eq!(acc.argmin(), (10, 10));
    }

    #[test]
    fn averages_symmetric_bimodal_at_small_theta() {
        // Two equal-cost probes 10 tiles apart. Softmin should land at midpoint.
        let mut acc = SoftminAccumulator::new(0.1);
        acc.observe(0, 0, 3.0);
        acc.observe(10, 10, 3.0);
        let (fx, fy) = acc.softmin_continuous();
        assert!((fx - 5.0).abs() < 1e-9, "fx={}", fx);
        assert!((fy - 5.0).abs() < 1e-9, "fy={}", fy);
    }

    #[test]
    fn ignores_infinite_cost_probes() {
        // INF probes represent invalid tiles — must not pollute softmin.
        let mut acc = SoftminAccumulator::new(0.5);
        acc.observe(0, 0, f64::INFINITY);
        acc.observe(7, 3, 2.0);
        acc.observe(9, 9, f64::INFINITY);
        let (fx, fy) = acc.softmin_continuous();
        assert!((fx - 7.0).abs() < 1e-9);
        assert!((fy - 3.0).abs() < 1e-9);
        assert_eq!(acc.argmin(), (7, 3));
    }

    #[test]
    fn empty_observation_returns_argmin_seed() {
        // If nothing finite is observed, softmin_continuous falls back to best.
        let mut acc = SoftminAccumulator::new(0.5);
        acc.best_x = 4;
        acc.best_y = 6;
        acc.observe(1, 1, f64::INFINITY);
        let (fx, fy) = acc.softmin_continuous();
        assert!((fx - 4.0).abs() < 1e-9);
        assert!((fy - 6.0).abs() < 1e-9);
    }

    #[test]
    fn running_anchor_stays_stable_under_descending_costs() {
        // Feed probes in descending cost order; the anchor rescale must keep
        // the weighted mean exact.
        let probes: Vec<(i32, i32, f64)> = vec![
            (0, 0, 20.0),
            (10, 0, 5.0),
            (10, 10, 2.0),
            (0, 10, 2.0),
        ];
        // Reference: recompute softmin with the standard two-pass LSE.
        let theta = 0.3;
        let c_min = probes
            .iter()
            .map(|&(_, _, c)| c)
            .fold(f64::INFINITY, f64::min);
        let mut ref_sw = 0.0;
        let mut ref_sx = 0.0;
        let mut ref_sy = 0.0;
        for &(gx, gy, c) in &probes {
            let w = (-theta * (c - c_min)).exp();
            ref_sw += w;
            ref_sx += w * gx as f64;
            ref_sy += w * gy as f64;
        }
        let ref_fx = ref_sx / ref_sw;
        let ref_fy = ref_sy / ref_sw;

        let mut acc = SoftminAccumulator::new(theta);
        for &(gx, gy, c) in &probes {
            acc.observe(gx, gy, c);
        }
        let (fx, fy) = acc.softmin_continuous();
        assert!((fx - ref_fx).abs() < 1e-9, "fx={} ref={}", fx, ref_fx);
        assert!((fy - ref_fy).abs() < 1e-9, "fy={} ref={}", fy, ref_fy);
    }
}

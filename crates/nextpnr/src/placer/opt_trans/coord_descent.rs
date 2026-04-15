//! Discrete coordinate descent placer over all movable cell coordinates.
//!
//! The DCD loop treats placement as a `2 * n_cells` dimensional discrete
//! optimization problem. Within a sweep the resistance field is frozen, trial
//! moves are evaluated with scratch Dijkstra workspaces, and only the winning
//! moves update the cached distance fields. Pipe usage and effective
//! conductance are refreshed between sweeps.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellId, NetId};
use crate::placer::common;
use crate::placer::common::{CellValidityMask, TypeAwarePlacement};

use super::config::OptTransPlacerCfg;
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

/// Flat `n_nets × n_nodes` dist grid stored as a strided `Vec<f64>`. Row `ni`
/// starts at `ni * n_nodes`. Per-net slices are contiguous so cache-line reuse
/// across adjacent `node` indices is optimal.
struct DistCache {
    dist: Vec<f64>,
    n_nets: usize,
    n_nodes: usize,
}

impl DistCache {
    fn new(n_nets: usize, n_nodes: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n_nets * n_nodes],
            n_nets,
            n_nodes,
        }
    }

    fn ensure_shape(&mut self, n_nets: usize, n_nodes: usize) {
        if self.n_nets != n_nets || self.n_nodes != n_nodes {
            *self = Self::new(n_nets, n_nodes);
        } else {
            self.reset();
        }
    }

    fn reset(&mut self) {
        self.dist.fill(f64::INFINITY);
    }

    #[inline]
    fn row(&self, net_idx: usize) -> &[f64] {
        let start = net_idx * self.n_nodes;
        &self.dist[start..start + self.n_nodes]
    }

    #[inline]
    fn row_mut(&mut self, net_idx: usize) -> &mut [f64] {
        let start = net_idx * self.n_nodes;
        &mut self.dist[start..start + self.n_nodes]
    }

    #[inline]
    fn get(&self, net_idx: usize, node: usize) -> f64 {
        self.dist[net_idx * self.n_nodes + node]
    }
}

/// Per-net stored routing paths: `paths[net_idx]` lists every pipe index used on
/// the driver→sinks tree from the most recent Dijkstra. Used to update
/// `pipe.net_count` between sweeps without rerunning Dijkstra.
struct NetPaths {
    paths: Vec<Vec<u32>>,
}

impl NetPaths {
    fn new(n_nets: usize) -> Self {
        Self {
            paths: vec![Vec::new(); n_nets],
        }
    }

    fn ensure_shape(&mut self, n_nets: usize) {
        if self.paths.len() != n_nets {
            self.paths = vec![Vec::new(); n_nets];
        }
        // Note: do NOT clear when length matches — we deliberately preserve
        // stored paths between outer iterations so refresh_resistance can use
        // the previous sweep's tree to update pipe.net_count.
    }

    fn is_empty_for_all(&self) -> bool {
        self.paths.iter().all(|p| p.is_empty())
    }
}

/// Unsafe wrapper for parallel mutable access to disjoint NetPaths entries.
struct PathSlicePtr(*mut Vec<u32>, usize);
unsafe impl Send for PathSlicePtr {}
unsafe impl Sync for PathSlicePtr {}
impl PathSlicePtr {
    fn get_mut(&self, idx: usize) -> &mut Vec<u32> {
        assert!(idx < self.1);
        unsafe { &mut *self.0.add(idx) }
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
            let Some(drv) = info.pins.iter().find(|p| p.is_driver) else { continue };
            let Some(drv_ci) = drv.cell_idx else { continue };
            if drv.is_fixed || drv_ci >= n_cells { continue; }

            for sink in &info.pins {
                if sink.is_driver { continue; }
                let Some(sink_ci) = sink.cell_idx else { continue };
                if sink.is_fixed || sink_ci >= n_cells { continue; }
                if sink_ci == drv_ci { continue; }

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
                if placed[ci] { continue; }
                if deg[ci] < best_deg {
                    best_deg = deg[ci];
                    best = Some(ci);
                    if best_deg == 0 { break; }
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
}

struct PipeStamp {
    stamp: Vec<u32>,
    epoch: u32,
}

impl PipeStamp {
    fn new(n_pipes: usize) -> Self {
        Self {
            stamp: vec![0; n_pipes],
            epoch: 1,
        }
    }

    fn begin_net(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }
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
    if let Some(bel) = cell.bel {
        let loc = ctx.bel(bel).loc();
        (
            (loc.x - network.x0) as f64,
            (loc.y - network.y0) as f64,
            None,
        )
    } else {
        (
            network.net_to_tile((network.width - 1).max(0) as f64 * 0.5),
            network.net_to_tile((network.height - 1).max(0) as f64 * 0.5),
            None,
        )
    }
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
    let io_factor = if info.has_fixed_pin { cfg.io_boost } else { 1.0 };
    let crit = cfg
        .timing_criticality
        .get(&info.net_id)
        .copied()
        .unwrap_or(0.0) as f64;
    let timing = 1.0 + cfg.timing_weight.max(0.0) * crit.clamp(0.0, 1.0);
    io_factor * timing
}

fn source_node(info: &NetInfo) -> Option<usize> {
    info.pins
        .iter()
        .find(|pin| pin.is_driver)
        .map(|pin| pin.node)
}

fn sink_nodes(info: &NetInfo) -> Vec<usize> {
    let mut sinks: Vec<_> = info
        .pins
        .iter()
        .filter(|pin| !pin.is_driver)
        .map(|pin| pin.node)
        .collect();
    sinks.sort_unstable();
    sinks.dedup();
    sinks
}

fn net_cost(info: &NetInfo, dist: &[f64], cfg: &OptTransPlacerCfg) -> f64 {
    let mut cost = 0.0;
    for pin in &info.pins {
        if pin.is_driver {
            continue;
        }
        let d = dist[pin.node];
        if !d.is_finite() {
            return f64::INFINITY;
        }
        cost += d;
    }
    net_path_weight(info, cfg) * cost
}

fn mark_path_edge_usage(
    node: usize,
    source_node: usize,
    network: &PipeNetwork,
    ws: &PathSolverWorkspace,
    pipe_stamp: &mut PipeStamp,
    edge_usage: &mut [f64],
) {
    let epoch = pipe_stamp.epoch;
    path_solver::walk_path_back(
        node,
        source_node,
        network,
        &ws.prev_pipe,
        &mut pipe_stamp.stamp,
        epoch,
        |pipe_idx, _pipe, is_new| {
            if is_new {
                edge_usage[pipe_idx] += 1.0;
            }
        },
    );
}

fn collect_path_pipes(
    node: usize,
    source_node: usize,
    network: &PipeNetwork,
    ws: &PathSolverWorkspace,
    pipe_stamp: &mut PipeStamp,
    path_pipes: &mut Vec<u32>,
) {
    let epoch = pipe_stamp.epoch;
    path_solver::walk_path_back(
        node,
        source_node,
        network,
        &ws.prev_pipe,
        &mut pipe_stamp.stamp,
        epoch,
        |pipe_idx, _pipe, is_new| {
            if is_new {
                path_pipes.push(pipe_idx as u32);
            }
        },
    );
}

/// Unsafe wrapper for parallel mutable access to disjoint rows of the flat
/// `DistCache` buffer. Each row is a contiguous slice of length `stride`.
struct DistSlicePtr {
    base: *mut f64,
    stride: usize,
    n_nets: usize,
}
unsafe impl Send for DistSlicePtr {}
unsafe impl Sync for DistSlicePtr {}
impl DistSlicePtr {
    fn row_mut(&self, idx: usize) -> &mut [f64] {
        assert!(idx < self.n_nets);
        unsafe {
            std::slice::from_raw_parts_mut(self.base.add(idx * self.stride), self.stride)
        }
    }
}

fn solve_all_nets(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    dist_cache: &mut DistCache,
    collect_usage: bool,
    paths_out: Option<&mut NetPaths>,
) -> SolveAccum {
    let n_nodes = network.num_nodes();
    let n_pipes = network.num_pipes();
    let batch_size = cfg.net_parallel_batch_size.max(1);

    let dist_access = DistSlicePtr {
        base: dist_cache.dist.as_mut_ptr(),
        stride: dist_cache.n_nodes,
        n_nets: dist_cache.n_nets,
    };
    let path_access = paths_out.map(|p| {
        for entry in &mut p.paths {
            entry.clear();
        }
        let len = p.paths.len();
        PathSlicePtr(p.paths.as_mut_ptr(), len)
    });
    let path_access = &path_access;

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
                        },
                        PathSolverWorkspace::new(n_nodes, n_pipes),
                        PipeStamp::new(n_pipes),
                    )
                },
                |(mut accum, mut ws, mut pipe_stamp), (chunk_idx, chunk)| {
                    let base_net_idx = chunk_idx * batch_size;
                    for (local_idx, info) in chunk.iter().enumerate() {
                        let net_idx = base_net_idx + local_idx;
                        let Some(source) = source_node(info) else {
                            accum.stats.failures += 1;
                            continue;
                        };
                        let sinks = sink_nodes(info);
                        if sinks.is_empty() {
                            accum.stats.failures += 1;
                            continue;
                        }

                        ws.begin_net();
                        ws.mark_sinks(&sinks);
                        let heap_pops = if cfg.use_eikonal {
                            super::eikonal::fsm_solve(
                                network,
                                source,
                                sinks.len(),
                                &mut ws,
                            )
                        } else {
                            path_solver::dijkstra_early_terminate(
                                network,
                                source,
                                sinks.len(),
                                &mut ws,
                            )
                        };
                        ws.clear_sinks(&sinks);

                        accum.stats.total_solves += 1;
                        accum.stats.total_heap_pops += heap_pops;
                        accum.stats.max_heap_pops = accum.stats.max_heap_pops.max(heap_pops);

                        let dist_out = dist_access.row_mut(net_idx);
                        dist_out.fill(f64::INFINITY);
                        for &node in &ws.touched {
                            dist_out[node] = ws.phys_dist[node];
                        }

                        accum.energy += net_cost(info, &ws.dist, cfg);

                        if collect_usage {
                            pipe_stamp.begin_net();
                            for pin in &info.pins {
                                if pin.is_driver || !ws.dist[pin.node].is_finite() {
                                    continue;
                                }
                                mark_path_edge_usage(
                                    pin.node,
                                    source,
                                    network,
                                    &ws,
                                    &mut pipe_stamp,
                                    &mut accum.edge_usage,
                                );
                            }
                        }

                        if let Some(pa) = path_access.as_ref() {
                            let path_out = pa.get_mut(net_idx);
                            path_out.clear();
                            pipe_stamp.begin_net();
                            for pin in &info.pins {
                                if pin.is_driver || !ws.dist[pin.node].is_finite() {
                                    continue;
                                }
                                collect_path_pipes(
                                    pin.node, source, network, &ws,
                                    &mut pipe_stamp, path_out,
                                );
                            }
                        }
                    }
                    (accum, ws, pipe_stamp)
                },
            )
            .map(|(accum, _, _)| accum)
            .reduce(
                || SolveAccum {
                    edge_usage: vec![0.0; n_pipes],
                    energy: 0.0,
                    stats: PathStats::default(),
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
                    a
                },
            )
    })
}

fn update_effective_conductance(
    network: &mut PipeNetwork,
    solve_pool: &rayon::ThreadPool,
    resistance_model: &ResistanceModel,
) {
    solve_pool.install(|| {
        network.pipes.par_iter_mut().for_each(|pipe| {
            let r_eff = resistance_model.effective_resistance(pipe);
            pipe.eff_conductance = 1.0 / r_eff.max(1e-12);
        });
    });
}


/// Compute `pipe.net_count` from the stored per-net paths, applying the
/// blend-with-previous damping that the old pass-1 used.
fn update_net_count_from_paths(
    network: &mut PipeNetwork,
    paths: &NetPaths,
    blend_alpha: f64,
    solve_pool: &rayon::ThreadPool,
) {
    let n_pipes = network.pipes.len();
    let mut usage = vec![0u32; n_pipes];
    for net_path in &paths.paths {
        for &pipe_idx in net_path {
            let i = pipe_idx as usize;
            if i < n_pipes {
                usage[i] = usage[i].saturating_add(1);
            }
        }
    }
    solve_pool.install(|| {
        network
            .pipes
            .par_iter_mut()
            .zip(usage.par_iter())
            .for_each(|(pipe, &u)| {
                let new_count = u as f64;
                let blended = blend_alpha * pipe.net_count as f64
                    + (1.0 - blend_alpha) * new_count;
                pipe.net_count = blended.round() as u32;
            });
    });
}

fn refresh_resistance(
    network: &mut PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    resistance_model: &ResistanceModel,
    dist_cache: &mut DistCache,
    paths: &mut NetPaths,
) -> (f64, PathStats) {
    let blend_alpha = cfg.blend_alpha;

    if paths.is_empty_for_all() {
        // Bootstrap: first refresh has no stored paths. Run one solve under
        // current R_eff to populate dist_cache AND record paths. The R_eff
        // update happens on the next refresh from these paths.
        dist_cache.reset();
        let solve = solve_all_nets(
            network, net_infos, cfg, solve_pool, dist_cache, false, Some(paths),
        );
        return (solve.energy, solve.stats);
    }

    // Use the last stored paths to update pipe.net_count (no Dijkstra).
    update_net_count_from_paths(network, paths, blend_alpha, solve_pool);
    update_effective_conductance(network, solve_pool, resistance_model);

    // Single Dijkstra pass under the new R_eff: refills dist_cache AND records
    // fresh paths for the next refresh.
    dist_cache.reset();
    let solve = solve_all_nets(
        network, net_infos, cfg, solve_pool, dist_cache, false, Some(paths),
    );
    (solve.energy, solve.stats)
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
    // through every caller (BB, fullscan, bisection, tally) via their
    // existing "pick min finite cost" logic — no per-caller edit needed.
    if !validity.is_valid(cell_idx, gx, gy) {
        return f64::INFINITY;
    }
    if cell_nets.is_empty() {
        return 0.0;
    }

    let node = coord_to_node(network, gx, gy);
    let mut cost = 0.0;
    for &(net_idx, pin_idx) in cell_nets {
        let info = &net_infos[net_idx];
        let pin = &info.pins[pin_idx];
        debug_assert_eq!(pin.cell_idx, Some(cell_idx));

        if pin.is_driver {
            // Skip -- driver position emerges from sink forces on other nets.
        } else {
            let d = dist_cache.get(net_idx, node);
            if !d.is_finite() {
                return f64::INFINITY;
            }
            let weight = net_path_weight(info, cfg);
            cost += weight * d;
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
        if cell_x[i] < mnx { mnx = cell_x[i]; }
        if cell_x[i] > mxx { mxx = cell_x[i]; }
        if cell_y[i] < mny { mny = cell_y[i]; }
        if cell_y[i] > mxy { mxy = cell_y[i]; }
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
            if node.tile_x < amnx { amnx = node.tile_x; }
            if node.tile_x > amxx { amxx = node.tile_x; }
            if node.tile_y < amny { amny = node.tile_y; }
            if node.tile_y > amxy { amxy = node.tile_y; }
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
    summary.push(format!("chipdb: w={} h={} (tile grid)", ctx.chipdb().width(), ctx.chipdb().height()));
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
        if has_bel { cells_with_bel += 1; }
        if locked { locked_cells += 1; }
        let entry = design_by_bucket.entry(bucket).or_insert((0, 0, 0));
        entry.0 += 1;
        if has_bel { entry.1 += 1; }
        if locked { entry.2 += 1; }
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
        *movable_by_bucket.entry(ctx.name_of(b).to_string()).or_insert(0) += 1;
    }
    summary.push(format!(""));
    summary.push(format!("=== Movable cells (placer sees these) ==="));
    summary.push(format!("n_movable: {}", idx_to_cell.len()));
    for (b, c) in &movable_by_bucket {
        summary.push(format!("{:<12} {:>6}", b, c));
    }

    // Chipdb BEL capacity per bucket (from type_aware).
    summary.push(format!(""));
    summary.push(format!("=== Chipdb BEL capacity (from TypeAwarePlacement) ==="));
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
        if !locked { continue; }
        let Some(bel) = cell_raw.bel else { continue };
        let loc = ctx.bel(bel).loc();
        let bucket = ctx.name_of(ctx.resolve_bucket(cell_raw.cell_type)).to_string();
        let name = ctx.name_of(cell_raw.name).to_string();
        // Fanout: max of ports that resolve to a net with users.
        let mut fanout = 0u32;
        let cell_view = ctx.cell(cell_id);
        for pin in cell_view.ports() {
            if let Some(net_id) = cell_view.port_net(pin.port) {
                let net = ctx.design.net(net_id);
                let f = net.num_users() as u32;
                if f > fanout { fanout = f; }
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
        summary.push(format!("=== Fixed/locked cell positions (IOs and locked cells) ==="));
        summary.push(format!("n_fixed: {}", io_xs.len()));
        summary.push(format!(
            "BB: x=[{},{}] w={}  y=[{},{}] h={}",
            mnx, mxx, mxx - mnx, mny, mxy, mxy - mny
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
        if has_fixed { n_nets_with_fixed += 1; }
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
            if fan > max_fan { max_fan = fan; }
            tot_fan += fan;
            if info.has_fixed_pin { has_fixed = true; }
        }
        out.push((ci, cell_name, bucket, n_nets, max_fan, tot_fan, has_fixed));
    }
    out
}

fn per_net_hpwl(
    net_infos: &[NetInfo],
    network: &PipeNetwork,
) -> (Vec<String>, Vec<u32>, Vec<f64>) {
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
            if node.tile_x < mnx { mnx = node.tile_x; }
            if node.tile_x > mxx { mxx = node.tile_x; }
            if node.tile_y < mny { mny = node.tile_y; }
            if node.tile_y > mxy { mxy = node.tile_y; }
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
fn place_dcd_sweep(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    diag: &mut DiagCtx,
) -> usize {
    let n = cell_x.len();
    let alpha = cfg.jacobi_alpha.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return 0;
    }

    let collect_stats = diag.enabled;

    // Phase 1: compute each cell's best position against the frozen landscape.
    // This is embarrassingly parallel — each cell reads shared read-only state.
    let best: Vec<((i32, i32), Option<PlateauStat>)> = (0..n)
        .into_par_iter()
        .map(|ci| {
            let cell_nets = &cell_net_map.map[ci];
            if cell_nets.is_empty() {
                let gx = network.tile_to_net(cell_x[ci]).round() as i32;
                let gy = network.tile_to_net(cell_y[ci]).round() as i32;
                return ((gx, gy), None);
            }
            fullscan_find_best_position_with_stats(
                ci,
                cell_nets,
                net_infos,
                dist_cache,
                network,
                cfg,
                validity,
                collect_stats,
            )
        })
        .collect();

    // Record plateau stats serially after Phase 1.
    if collect_stats {
        for (ci, (_, stat)) in best.iter().enumerate() {
            if let Some(s) = stat {
                diag.record_plateau(ci, *s);
            }
        }
    }

    // Phase 2: apply damped simultaneous update.
    let mut moved = 0usize;
    let mut final_positions: Vec<(i32, i32)> = Vec::with_capacity(n);
    for ci in 0..n {
        let (best_gx, best_gy) = best[ci].0;
        let old_tile_x = cell_x[ci];
        let old_tile_y = cell_y[ci];
        let old_gx = network.tile_to_net(old_tile_x).round() as i32;
        let old_gy = network.tile_to_net(old_tile_y).round() as i32;

        if best_gx == old_gx && best_gy == old_gy {
            if collect_stats {
                final_positions.push((old_gx, old_gy));
            }
            continue;
        }

        // Damped blend in grid coordinates.
        let new_gx_f = alpha * best_gx as f64 + (1.0 - alpha) * old_gx as f64;
        let new_gy_f = alpha * best_gy as f64 + (1.0 - alpha) * old_gy as f64;
        let new_gx = new_gx_f.round() as i32;
        let new_gy = new_gy_f.round() as i32;

        if new_gx == old_gx && new_gy == old_gy {
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
                0 => { hi_x = cx; hi_y = cy; }            // SW
                1 => { lo_x = cx; hi_y = cy; }            // SE
                2 => { hi_x = cx; lo_y = cy; }            // NW
                _ => { lo_x = cx; lo_y = cy; }            // NE
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
/// Re-run Dijkstra for one net and overwrite `dist_cache[net_idx]` in place.
/// Used after a driver cell moves so subsequent cells see the live field.
fn refresh_net_dist_cache(
    net_idx: usize,
    info: &NetInfo,
    dist_cache: &mut DistCache,
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    ws: &mut PathSolverWorkspace,
) {
    let Some(source) = source_node(info) else {
        return;
    };
    let sinks = sink_nodes(info);
    if sinks.is_empty() {
        return;
    }

    ws.begin_net();
    ws.mark_sinks(&sinks);
    if cfg.use_eikonal {
        super::eikonal::fsm_solve(network, source, sinks.len(), ws);
    } else {
        path_solver::dijkstra_early_terminate(network, source, sinks.len(), ws);
    }
    ws.clear_sinks(&sinks);

    let dist_out = dist_cache.row_mut(net_idx);
    dist_out.fill(f64::INFINITY);
    for &node in &ws.touched {
        dist_out[node] = ws.phys_dist[node];
    }
}

/// Jacobi DCD sweep using parallel 2D bisection.
///
/// Mirrors `place_dcd_sweep` (Jacobi + fullscan) but swaps the per-cell
/// search to `bisect_2d`, which costs ~112 evals/cell vs fullscan's 16k.
/// The whole sweep evaluates against a frozen `dist_cache` snapshot — no
/// in-sweep refresh, no per-move Dijkstra. Between-sweep `refresh_resistance`
/// updates the field as before.
fn place_dcd_sweep_jacobi_bisection(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
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
                ci, cell_nets, net_infos, dist_cache, network, cfg,
                validity,
                old_gx, old_gy, cur_cost,
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
struct BbNode { bound: f64, level: i32, cx: i32, cy: i32 }
impl PartialEq for BbNode {
    fn eq(&self, o: &Self) -> bool { self.bound == o.bound && self.level == o.level }
}
impl Eq for BbNode {}
impl Ord for BbNode {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        let a = o.bound.partial_cmp(&self.bound).unwrap_or(std::cmp::Ordering::Equal);
        if a != std::cmp::Ordering::Equal { return a; }
        o.level.cmp(&self.level)
    }
}
impl PartialOrd for BbNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
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
        if !m.is_finite() { return f64::INFINITY; }
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
            pq.push(BbNode { bound: b, level: max_level, cx: 0, cy: 0 });
        }
    } else {
        for gy in 0..h {
            for gx in 0..w {
                let c = bb_bound(&active_nets, dist_cache, pyramids, w, 0, gx, gy);
                if c.is_finite() && c < best_cost {
                    pq.push(BbNode { bound: c, level: 0, cx: gx, cy: gy });
                }
            }
        }
    }

    while let Some(node) = pq.pop() {
        if node.bound >= best_cost { break; }
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
            if ccx >= child_w || ccy >= child_h { continue; }
            let b = bb_bound(&active_nets, dist_cache, pyramids, w, child_level, ccx, ccy);
            if b.is_finite() && b < best_cost {
                pq.push(BbNode { bound: b, level: child_level, cx: ccx, cy: ccy });
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
            pq.push(BbNode { bound: b, level: max_level, cx: 0, cy: 0 });
        }
    } else {
        for gy in 0..h {
            for gx in 0..w {
                let c = bb_bound(&active_nets, dist_cache, pyramids, w, 0, gx, gy);
                if c.is_finite() && c <= best_cost {
                    pq.push(BbNode { bound: c, level: 0, cx: gx, cy: gy });
                }
            }
        }
    }

    while let Some(node) = pq.pop() {
        if node.bound > best_cost { break; }
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
            if ccx >= child_w || ccy >= child_h { continue; }
            let b = bb_bound(&active_nets, dist_cache, pyramids, w, child_level, ccx, ccy);
            if b.is_finite() && b <= best_cost {
                pq.push(BbNode { bound: b, level: child_level, cx: ccx, cy: ccy });
            }
        }
    }

    if start_cost > best_cost {
        tied.retain(|&(x, y)| (x, y) != (start_x, start_y));
    }

    (tied, best_cost, n_evals)
}

/// Sequential tally-aware BB sweep.
///
/// Process cells in `iter_order` (topo forward or reversed). For each cell:
/// 1. Find ALL cost-tied optimum candidates via `bb_2d_multi`.
/// 2. Break ties by LEX ordering on (current_tally, self_overlap):
///    - `current_tally(P)` = Σ over pipes adjacent to node(P) of `pipe_tally[pipe]`
///    - `self_overlap(P)`  = count of pipes adjacent to node(P) that are ALSO
///       in cell c's net_paths union — proxy for "how much this candidate's
///       post-commit tally would grow from c's own nets being committed".
/// 3. Commit cell to the winner.
/// 4. Increment `pipe_tally[pipe]` for every pipe in c's nets' `net_paths`.
///
/// `pipe_tally` is expected to be zeroed at the start of each pass.
fn place_dcd_sweep_jacobi_bb_tally(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &DistCache,
    pyramids: &[RegionMinPyramid],
    net_paths: &NetPaths,
    pipe_tally: &mut [u32],
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    diag: &mut DiagCtx,
    iter_order: &[usize],
) -> usize {
    let want_perf = std::env::var("NPNR_OT_BB_PERF").ok().as_deref() == Some("1");

    // Precompute per-cell union of net_paths pipes, as a sorted Vec<u32>. Used
    // for self_overlap lookups via binary search.
    let n = cell_x.len();
    let mut cell_pipes: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ci, nets) in cell_net_map.map.iter().enumerate() {
        let mut u: Vec<u32> = Vec::new();
        for &(net_idx, _pin_idx) in nets {
            u.extend_from_slice(&net_paths.paths[net_idx]);
        }
        u.sort_unstable();
        u.dedup();
        cell_pipes[ci] = u;
    }

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
                cell_nets, net_infos, dist_cache, pyramids, network, cfg,
                old_gx, old_gy, cur_cost,
            );
            (tied, n_evals)
        })
        .collect();

    // Phase B (sequential): commit in topo order with live `pipe_tally` so
    // later cells see the updated tally from earlier commits. Cheap per-cell.
    let mut moved = 0usize;
    let mut total_ties = 0usize;
    let mut total_evals = 0usize;

    for (order_idx, &ci) in iter_order.iter().enumerate() {
        let cell_nets = &cell_net_map.map[ci];
        if cell_nets.is_empty() { continue; }

        let (tied, n_evals) = &tied_per_cell[order_idx];
        total_evals += n_evals;
        if tied.len() > 1 { total_ties += 1; }
        if tied.is_empty() {
            // No valid candidate position under the current dist_cache and
            // validity mask — leave cell where it is.
            continue;
        }

        // Lex tiebreak: pick candidate minimizing (current_tally, self_overlap).
        let self_pipes = &cell_pipes[ci];
        let mut best_idx = 0usize;
        let mut best_key = (u64::MAX, u32::MAX);
        for (i, &(px, py)) in tied.iter().enumerate() {
            let node = coord_to_node(network, px, py);
            let adj = &network.node_pipes[node];
            let mut ct: u64 = 0;
            let mut ov: u32 = 0;
            for &pipe_idx in adj {
                ct += pipe_tally[pipe_idx] as u64;
                if self_pipes.binary_search(&(pipe_idx as u32)).is_ok() {
                    ov += 1;
                }
            }
            let key = (ct, ov);
            if key < best_key {
                best_key = key;
                best_idx = i;
            }
        }
        let (new_gx, new_gy) = tied[best_idx];

        let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
        let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
        if new_gx != old_gx || new_gy != old_gy {
            cell_x[ci] = network.net_to_tile(new_gx as f64);
            cell_y[ci] = network.net_to_tile(new_gy as f64);
            let new_node = coord_to_node(network, new_gx, new_gy);
            for &(net_idx, pin_idx) in cell_nets {
                net_infos[net_idx].pins[pin_idx].node = new_node;
            }
            if diag.enabled {
                diag.record_move(MoveRecord {
                    cell_idx: ci,
                    old_gx, old_gy, new_gx, new_gy,
                });
            }
            moved += 1;
        }

        // Update tally: mark c's net_paths pipes as "used this sweep".
        for &pipe_idx in self_pipes {
            pipe_tally[pipe_idx as usize] += 1;
        }
    }

    if want_perf {
        eprintln!(
            "    bb_tally: moved={} total_evals={} cells_with_ties={}",
            moved, total_evals, total_ties,
        );
    }

    moved
}

/// Sequential DCD sweep with bisection search and live dist_cache refresh.
///
/// Cells are processed in topological order (drivers before sinks). For each
/// cell we bisect on x, then on y, accepting only strict energy decreases.
/// After committing, nets where this cell is the driver have their Dijkstra
/// distances recomputed so the next cell evaluates against the live field.
/// Pipe usage / R_eff stay frozen within the sweep; the outer loop refreshes
/// them between sweeps.
fn place_dcd_sweep_sequential_bisection(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &mut DistCache,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    diag: &mut DiagCtx,
) -> usize {
    let n = cell_x.len();
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
            ci, cell_nets, net_infos, dist_cache, network, cfg,
            validity,
            old_gx, old_gy, cur_cost,
        );
        if want_perf {
            t_search_ns += t0.elapsed().as_nanos();
        }

        if new_gx == old_gx && new_gy == old_gy {
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
        // Dijkstra to the between-sweep refresh.
        let r = cfg.bisect_refresh_region.max(1);
        let crossed_region =
            (old_gx / r) != (new_gx / r) || (old_gy / r) != (new_gy / r);
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
            if driver_count > 0 { n_driver_moves += 1; }
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
    chpwl: f64,
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
) {
    let n = cell_x.len();
    let n_nodes = network.num_nodes();
    let mut dist_cache = DistCache::new(0, n_nodes);
    let mut net_paths = NetPaths::new(0);
    let max_iter = cfg.max_outer_iters.max(1);
    let mut diag_ctx = DiagCtx::from_env();

    // Per-cell validity mask. Rejects candidate positions where the cell's
    // bucket has no compatible BEL, propagating through every helper via
    // `evaluate_cell_at` returning f64::INFINITY.
    let validity =
        CellValidityMask::build(ctx, _idx_to_cell, type_aware, network.width, network.height);
    let validity = &validity;
    if diag_ctx.enabled {
        eprintln!("DCD diag: instrumentation ON, output dir /tmp/claude/ot_diag (or $NPNR_OT_DIAG_DIR)");
    }

    eprintln!(
        "DCD placer: {} cells, {} nodes, outer_iters={}, dcd_iters_per_cell={}",
        n, n_nodes, max_iter, cfg.dcd_iters_per_cell,
    );

    let mut best_x = cell_x.to_vec();
    let mut best_y = cell_y.to_vec();
    let initial_chpwl = demand::continuous_hpwl(ctx, cell_to_idx, cell_x, cell_y, network);
    let mut best_state = ObjState {
        energy: f64::INFINITY,
        chpwl: initial_chpwl,
        line: f64::INFINITY,
        friction: f64::INFINITY,
        max_overflow: f64::INFINITY,
        n_overflow: usize::MAX,
        overflow_excess: f64::INFINITY,
    };

    eprintln!("  Initial chpwl={:.0}", initial_chpwl);

    let mut stalls = 0usize;
    let max_stalls = 5;
    for outer in 0..max_iter {
        let t_outer = std::time::Instant::now();
        let mut net_infos =
            collect_net_infos_simple(ctx, alive_net_ids, cell_to_idx, cell_x, cell_y, network, cfg);
        let cell_net_map = CellNetMap::build(&net_infos, n);
        dist_cache.ensure_shape(net_infos.len(), n_nodes);
        net_paths.ensure_shape(net_infos.len());

        let t_refresh = std::time::Instant::now();
        let refresh = refresh_resistance(
            network,
            &net_infos,
            cfg,
            solve_pool,
            resistance_model,
            &mut dist_cache,
            &mut net_paths,
        );
        let refresh_ms = t_refresh.elapsed().as_millis();

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

        // HPWL vs routed-star sanity check. Runs RIGHT after refresh_resistance
        // so `dist_cache.dist[net][node]` (phys_dist from driver) is consistent
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
                if info.pins.is_empty() { continue; }
                let Some(first) = info.pins.first() else { continue };
                let first_node = &network.nodes[first.node];
                let (mut mnx, mut mxx) = (first_node.tile_x, first_node.tile_x);
                let (mut mny, mut mxy) = (first_node.tile_y, first_node.tile_y);
                for pin in info.pins.iter().skip(1) {
                    let node = &network.nodes[pin.node];
                    if node.tile_x < mnx { mnx = node.tile_x; }
                    if node.tile_x > mxx { mxx = node.tile_x; }
                    if node.tile_y < mny { mny = node.tile_y; }
                    if node.tile_y > mxy { mxy = node.tile_y; }
                }
                let hpwl = (mxx - mnx) as f64 + (mxy - mny) as f64;
                let mut star = 0.0f64;
                let mut ok = true;
                let mut fanout = 0u32;
                for pin in &info.pins {
                    if pin.is_driver { continue; }
                    fanout += 1;
                    let d = dist_cache.get(net_idx, pin.node);
                    if !d.is_finite() { ok = false; break; }
                    star += d;
                }
                if !ok || fanout == 0 { continue; }
                n_valid += 1;
                sum_hpwl += hpwl;
                sum_star += star;
                if star > 0.5 {
                    let r = hpwl / star;
                    if r > 2.0 { n_ratio_gt2 += 1; }
                    if r > 5.0 { n_ratio_gt5 += 1; }
                    if r < 0.5 { n_ratio_lt05 += 1; }
                    if r < 0.2 { n_ratio_lt02 += 1; }
                    ratios.push((r, net_idx, hpwl, star, fanout));
                }
            }
            let mean_hpwl = if n_valid > 0 { sum_hpwl / n_valid as f64 } else { 0.0 };
            let mean_star = if n_valid > 0 { sum_star / n_valid as f64 } else { 0.0 };
            let global_ratio = if sum_star > 0.0 { sum_hpwl / sum_star } else { 0.0 };
            eprintln!(
                "    HPWL_check (pre-sweep): n_valid={} sum_hpwl={:.0} sum_star={:.0} global_ratio={:.3} mean_hpwl={:.1} mean_star={:.1}  hpwl/star buckets: >5x={} >2x={} <0.5x={} <0.2x={}",
                n_valid, sum_hpwl, sum_star, global_ratio, mean_hpwl, mean_star,
                n_ratio_gt5, n_ratio_gt2, n_ratio_lt05, n_ratio_lt02,
            );
            ratios.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let n_show = ratios.len().min(10);
            if n_show > 0 {
                eprintln!("    Top {} nets by hpwl/star (HPWL overstates routing):", n_show);
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
                eprintln!("    Top {} nets by star/hpwl (HPWL understates routing):", n_show_lo);
                for (r, ni, h, s, f) in ratios.iter().take(n_show_lo) {
                    eprintln!(
                        "      net={} fanout={} hpwl={:.0} star={:.1} ratio={:.2}  name={}",
                        ni, f, h, s, r, net_infos[*ni].debug_name,
                    );
                }
            }
        }

        // Validate Eikonal vs Dijkstra distances on first iteration.
        if outer == 0 && cfg.use_eikonal {
            let mut ws_dij = PathSolverWorkspace::new(n_nodes, network.num_pipes());
            let mut ws_fsm = PathSolverWorkspace::new(n_nodes, network.num_pipes());
            for ni in 0..5.min(net_infos.len()) {
                let info = &net_infos[ni];
                let Some(src) = source_node(info) else { continue };
                let sinks = sink_nodes(info);
                if sinks.is_empty() { continue; }

                ws_dij.begin_net();
                ws_dij.mark_sinks(&sinks);
                path_solver::dijkstra_early_terminate(network, src, sinks.len(), &mut ws_dij);
                ws_dij.clear_sinks(&sinks);

                ws_fsm.begin_net();
                ws_fsm.mark_sinks(&sinks);
                super::eikonal::fsm_solve(network, src, sinks.len(), &mut ws_fsm);
                ws_fsm.clear_sinks(&sinks);

                let mut max_phys_err = 0.0f64;
                let mut max_dist_err = 0.0f64;
                let mut n_cmp = 0usize;
                for &s in &sinks {
                    if ws_dij.phys_dist[s].is_finite() && ws_fsm.phys_dist[s].is_finite() {
                        max_phys_err = max_phys_err.max((ws_fsm.phys_dist[s] - ws_dij.phys_dist[s]).abs());
                        max_dist_err = max_dist_err.max((ws_fsm.dist[s] - ws_dij.dist[s]).abs());
                        n_cmp += 1;
                    }
                }
                eprintln!(
                    "    FSM vs Dijkstra net={} sinks={}: max_phys_err={:.6} max_dist_err={:.6}",
                    ni, sinks.len(), max_phys_err, max_dist_err,
                );
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
                if cell_nets.is_empty() { continue; }
                let cur_gx_tmp = network.tile_to_net(cell_x[ci]).round() as i32;
                let cur_gy_tmp = network.tile_to_net(cell_y[ci]).round() as i32;
                let cur_cost_tmp = evaluate_cell_at(ci, cur_gx_tmp, cur_gy_tmp, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                let (bis_x, bis_y, _bis_c, n_probes, n_probes_inf, n_improve_steps) =
                    bisect_2d(ci, cell_nets, &net_infos, &dist_cache, network, cfg, validity, cur_gx_tmp, cur_gy_tmp, cur_cost_tmp);
                let (fs_x, fs_y) = fullscan_find_best_position(ci, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                let bis_cost = evaluate_cell_at(ci, bis_x, bis_y, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                let fs_cost = evaluate_cell_at(ci, fs_x, fs_y, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                let cur_gx = network.tile_to_net(cell_x[ci]).round() as i32;
                let cur_gy = network.tile_to_net(cell_y[ci]).round() as i32;
                let cur_cost = evaluate_cell_at(ci, cur_gx, cur_gy, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                let d_bis_fs = (bis_x - fs_x).abs() + (bis_y - fs_y).abs();
                let d_cur_fs = (cur_gx - fs_x).abs() + (cur_gy - fs_y).abs();
                let d_cur_bis = (cur_gx - bis_x).abs() + (cur_gy - bis_y).abs();
                let gap_bis_fs = (bis_cost - fs_cost).max(0.0);
                let gap_cur_fs = (cur_cost - fs_cost).max(0.0);
                if bis_x != fs_x || bis_y != fs_y { mismatch += 1; }
                sum_d += d_bis_fs as i64;
                if d_bis_fs > max_d { max_d = d_bis_fs; }
                if gap_bis_fs.is_finite() { sum_gap += gap_bis_fs; if gap_bis_fs > max_gap { max_gap = gap_bis_fs; } }
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
                        let row = dist_cache.row(net_idx);
                        let c = row.iter().filter(|d| d.is_finite()).count();
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
            let avg_d = if n_examined > 0 { sum_d as f64 / n_examined as f64 } else { 0.0 };
            let avg_gap = if n_examined > 0 { sum_gap / n_examined as f64 } else { 0.0 };
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
            let mut n_over = 0usize;       // net_count > capacity
            let mut max_excess = 0i64;     // max(net_count - capacity) in wires
            let mut sum_excess = 0i64;
            let mut excess_1 = 0usize;     // excess == 1
            let mut excess_2_5 = 0usize;   // 2..=5
            let mut excess_6_20 = 0usize;  // 6..=20
            let mut excess_gt20 = 0usize;  // >20
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
            let mut worst_nc = 0u32;
            let mut worst_cap = 0.0f64;
            let mut worst_span = 0i32;
            for pipe in &network.pipes {
                if pipe.capacity <= 0.0 { continue; }
                n_with_cap += 1;
                let util = pipe.net_count as f64 / pipe.capacity;
                if pipe.net_count > 0 { n_used += 1; }
                if util > 0.5 { n_congested += 1; }
                if util > 0.9 { n_saturated += 1; }
                if util > max_util { max_util = util; }
                sum_util += util;
                let nc = pipe.net_count as i64;
                let cap_i = pipe.capacity as i64;
                if nc > cap_i {
                    n_over += 1;
                    let excess = nc - cap_i;
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
                    match excess {
                        1 => {
                            excess_1 += 1;
                            match cap_i {
                                1 => exc1_cap1 += 1,
                                2..=5 => exc1_cap2_5 += 1,
                                _ => exc1_cap_gt5 += 1,
                            }
                        }
                        2..=5 => excess_2_5 += 1,
                        6..=20 => excess_6_20 += 1,
                        _ => excess_gt20 += 1,
                    }
                    match cap_i {
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
            let avg_util = if n_with_cap > 0 { sum_util / n_with_cap as f64 } else { 0.0 };
            let avg_excess = if n_over > 0 { sum_excess as f64 / n_over as f64 } else { 0.0 };
            eprintln!(
                "    Overflow: over={} ({:.1}% of used) avg_excess={:.2} max_excess={} (nc={} cap={:.0} span={})  excess_buckets: 1={} 2-5={} 6-20={} >20={}  by_cap: cap1={} cap2={} cap3-5={} cap>5={}  by_span: s1={} s2-3={} s4-6={} s7-12={} s>12={}  exc1_cross: cap1={} cap2-5={} cap>5={}",
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

            // Which nets are piling onto the top-N saturated pipes? Walk
            // net_paths to find every net that listed the pipe in its tree.
            // Useful to answer "who's the ringleader of this hot spot?"
            if std::env::var("NPNR_OT_HOT_NETS").ok().as_deref() == Some("1") {
                let mut ranked: Vec<(i64, usize, u32, f64, i32)> = Vec::new(); // (excess, pipe_idx, nc, cap, span)
                for (pi, pipe) in network.pipes.iter().enumerate() {
                    let cap_i = pipe.capacity as i64;
                    if cap_i <= 0 { continue; }
                    let nc = pipe.net_count as i64;
                    if nc <= cap_i { continue; }
                    let span = match pipe.pipe_type {
                        super::network::PipeType::InterTile(_) => 1i32,
                        super::network::PipeType::LongRange { dx, dy } => dx.abs() + dy.abs(),
                    };
                    ranked.push((nc - cap_i, pi, pipe.net_count, pipe.capacity, span));
                }
                ranked.sort_by(|a, b| b.0.cmp(&a.0));
                let top_n = ranked.len().min(5);
                for (excess, pi, nc, cap, span) in ranked.iter().take(top_n) {
                    let mut users: Vec<(usize, u32, String)> = Vec::new();
                    for (net_idx, path) in net_paths.paths.iter().enumerate() {
                        if path.iter().any(|&p| p as usize == *pi) {
                            let info = &net_infos[net_idx];
                            let fanout = info.pins.len().saturating_sub(1) as u32;
                            users.push((net_idx, fanout, info.debug_name.clone()));
                        }
                    }
                    users.sort_by(|a, b| b.1.cmp(&a.1));
                    let node_from = &network.nodes[network.pipes[*pi].from];
                    let node_to = &network.nodes[network.pipes[*pi].to];
                    let tt_from = ctx.chipdb().tile_type_name(
                        ctx.chipdb().tile_by_xy(node_from.tile_x, node_from.tile_y)
                    ).to_string();
                    let tt_to = ctx.chipdb().tile_type_name(
                        ctx.chipdb().tile_by_xy(node_to.tile_x, node_to.tile_y)
                    ).to_string();
                    eprintln!(
                        "    HotPipe[{}]: excess={} nc={} cap={:.0} span={} endpoints=({},{}:{})<->({},{}:{})  {} nets on pipe (top fanout shown):",
                        pi, excess, nc, cap, span,
                        node_from.tile_x, node_from.tile_y, tt_from,
                        node_to.tile_x, node_to.tile_y, tt_to,
                        users.len(),
                    );
                    for (ni, fo, name) in users.iter().take(10) {
                        eprintln!("      net={} fanout={} name={}", ni, fo, name);
                    }
                    if users.len() > 10 {
                        eprintln!("      ... and {} more", users.len() - 10);
                    }
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

        let pre_sweep_x: Vec<f64> = cell_x.to_vec();
        let pre_sweep_y: Vec<f64> = cell_y.to_vec();

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
                &mut diag_ctx,
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
                &mut diag_ctx,
            ),
            super::config::SweepMode::JacobiBB => {
                let want_perf = std::env::var("NPNR_OT_BB_PERF").ok().as_deref() == Some("1");

                // Forward pass: build pyramids from current dist_cache, then
                // optimize every cell in topological order (drivers before
                // sinks).
                let t_pyr = std::time::Instant::now();
                let pyramids = region_min::build_all(
                    &dist_cache.dist,
                    dist_cache.n_nets,
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
                if outer == 0
                    && std::env::var("NPNR_OT_BB_VS_FS_DIAG").ok().as_deref() == Some("1")
                {
                    let mut cost_mismatch = 0usize;
                    let mut pos_mismatch = 0usize;
                    let mut max_cost_gap = 0.0f64;
                    let mut sum_cost_gap = 0.0f64;
                    for ci in 0..n {
                        let cell_nets = &cell_net_map.map[ci];
                        if cell_nets.is_empty() { continue; }
                        let ogx = network.tile_to_net(cell_x[ci]).round() as i32;
                        let ogy = network.tile_to_net(cell_y[ci]).round() as i32;
                        let cur = evaluate_cell_at(ci, ogx, ogy, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                        let (bx, by, bc, _ne) = bb_2d(
                            cell_nets, &net_infos, &dist_cache, &pyramids, network, cfg,
                            ogx, ogy, cur,
                        );
                        let (fx, fy) = fullscan_find_best_position(
                            ci, cell_nets, &net_infos, &dist_cache, network, cfg, validity,
                        );
                        let fc = evaluate_cell_at(ci, fx, fy, cell_nets, &net_infos, &dist_cache, network, cfg, validity);
                        if bx != fx || by != fy { pos_mismatch += 1; }
                        let gap = (bc - fc).abs();
                        if gap > 1e-9 { cost_mismatch += 1; }
                        if gap.is_finite() {
                            if gap > max_cost_gap { max_cost_gap = gap; }
                            sum_cost_gap += gap;
                        }
                    }
                    let avg = sum_cost_gap / n.max(1) as f64;
                    eprintln!(
                        "    BB vs fullscan (iter 0, {} cells): cost_mismatch={}  pos_mismatch={}  avg_cost_gap={:.6}  max_cost_gap={:.6}",
                        n, cost_mismatch, pos_mismatch, avg, max_cost_gap,
                    );
                }
                // Forward pass: tally-aware sequential commit in topo order.
                let topo_fwd = &cell_net_map.topo_order;
                let mut pipe_tally = vec![0u32; network.pipes.len()];
                let moved_fwd = place_dcd_sweep_jacobi_bb_tally(
                    &mut net_infos,
                    &cell_net_map,
                    &dist_cache,
                    &pyramids,
                    &net_paths,
                    &mut pipe_tally,
                    cell_x,
                    cell_y,
                    network,
                    cfg,
                    validity,
                    &mut diag_ctx,
                    topo_fwd,
                );

                // Backward pass runs against the SAME dist_cache / pyramids
                // / net_paths as the forward pass — no mid-iter refresh. The
                // full refresh (and implicit R_eff update via net_count blend)
                // happens once per outer iter in `refresh_resistance` at the
                // top of the next iter. Within an outer iter, only cell
                // positions and the per-sweep `pipe_tally` mutate; the
                // distance field is held constant across both passes.
                //
                // Reset tally for the backward pass; iterate in reversed
                // topological order.
                for v in pipe_tally.iter_mut() { *v = 0; }
                let topo_rev: Vec<usize> =
                    cell_net_map.topo_order.iter().rev().copied().collect();
                let moved_bwd = place_dcd_sweep_jacobi_bb_tally(
                    &mut net_infos,
                    &cell_net_map,
                    &dist_cache,
                    &pyramids,
                    &net_paths,
                    &mut pipe_tally,
                    cell_x,
                    cell_y,
                    network,
                    cfg,
                    validity,
                    &mut diag_ctx,
                    &topo_rev,
                );

                moved_fwd + moved_bwd
            }
        };
        let dcd_ms = t_dcd.elapsed().as_millis();

        common::clamp_positions(cell_x, cell_y, phys_max_x, phys_max_y);

        let chpwl = demand::continuous_hpwl(ctx, cell_to_idx, cell_x, cell_y, network);
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
            energy: refresh.0,
            chpwl,
            line,
            friction,
            max_overflow,
            n_overflow,
            overflow_excess,
        };

        // DCD objective: energy under current R_eff. Energy is a function of
        // (positions, R_eff); R_eff is intentionally updated between iters to
        // reflect congestion, so energy naturally drifts iter-to-iter even as
        // positions improve. We therefore do NOT revert on energy increase and
        // do NOT early-stop on "no new energy minimum". The lowest-energy
        // iterate is still tracked so the placer returns the best snapshot.
        // Convergence signals that ARE meaningful: `moved == 0` (positions
        // stable) and a small relative-energy delta between successive iters
        // (physical equilibrium of positions + R_eff).
        let improved = state.energy < best_state.energy;
        let d_energy = state.energy - best_state.energy;
        let d_hpwl = state.chpwl - best_state.chpwl;
        let d_friction = state.friction - best_state.friction;
        if improved {
            best_state = state;
            best_x.copy_from_slice(&pre_sweep_x);
            best_y.copy_from_slice(&pre_sweep_y);
        }
        // Relative-energy stability stall: iters where |dE| / E < 0.5% in a row
        // indicate the physical system has equilibrated.
        let rel_de = if state.energy.abs() > 1e-12 {
            (d_energy / state.energy.abs()).abs()
        } else {
            0.0
        };
        if rel_de < 0.005 {
            stalls += 1;
        } else {
            stalls = 0;
        }

        // Diagnostic: per-net HPWL (tile coords via network nodes) and sweep summary.
        if diag_ctx.enabled {
            let (net_names, net_fanout, net_hpwl) = per_net_hpwl(&net_infos, network);
            let bb = compute_bounding_boxes(cell_x, cell_y, &net_infos, network);
            diag_ctx.sweep_end(
                n,
                chpwl,
                line,
                friction,
                max_overflow,
                n_overflow,
                overflow_excess,
                moved,
                improved,
                d_hpwl,
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
        for pipe in &network.pipes {
            let nc = pipe.net_count as f64;
            if nc <= 0.0 { continue; }
            let r_eff = if pipe.eff_conductance > 0.0 { 1.0 / pipe.eff_conductance } else { pipe.base_resistance };
            sum_base += pipe.base_resistance * nc;
            sum_eff += r_eff * nc;
            if pipe.capacity > 0.0 {
                let u = nc / pipe.capacity;
                if u > max_util { max_util = u; }
                if u > 0.9 { n_sat += 1; }
            }
        }
        let cong_share = if sum_eff > 0.0 { (sum_eff - sum_base) / sum_eff } else { 0.0 };
        // R_eff / base ratio distribution on USED pipes. Buckets quantify
        // which congestion regime each pipe is in; pool-flipping across
        // iters is what drives energy oscillation.
        let mut r_lo = 0usize;  // <=2x base (healthy)
        let mut r_med = 0usize; // 2x..10x
        let mut r_hi = 0usize;  // 10x..100x
        let mut r_sat = 0usize; // >=100x (near / at eps clamp)
        for pipe in &network.pipes {
            if pipe.net_count == 0 || pipe.base_resistance <= 0.0 { continue; }
            let r_eff = if pipe.eff_conductance > 0.0 { 1.0 / pipe.eff_conductance } else { pipe.base_resistance };
            let ratio = r_eff / pipe.base_resistance;
            if ratio < 2.0 { r_lo += 1; }
            else if ratio < 10.0 { r_med += 1; }
            else if ratio < 100.0 { r_hi += 1; }
            else { r_sat += 1; }
        }
        eprintln!(
            "    E_decomp: base={:.1} eff={:.1} cong={:.1} cong_share={:.1}% sat_pipes={} max_util={:.2}  R_buckets: 1-2x={} 2-10x={} 10-100x={} >=100x={}",
            sum_base, sum_eff, sum_eff - sum_base, cong_share * 100.0, n_sat, max_util,
            r_lo, r_med, r_hi, r_sat,
        );

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
                let mut rng_state = cfg.seed.wrapping_add(outer as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                if rng_state == 0 { rng_state = 1; }
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
                    let nets_i = &cell_net_map.map[i];
                    let nets_j = &cell_net_map.map[j];
                    if nets_i.is_empty() || nets_j.is_empty() {
                        continue;
                    }
                    let xi = network.tile_to_net(cell_x[i]).round() as i32;
                    let yi = network.tile_to_net(cell_y[i]).round() as i32;
                    let xj = network.tile_to_net(cell_x[j]).round() as i32;
                    let yj = network.tile_to_net(cell_y[j]).round() as i32;
                    let pre_i = evaluate_cell_at(i, xi, yi, nets_i, &net_infos, &dist_cache, network, cfg, validity);
                    let pre_j = evaluate_cell_at(j, xj, yj, nets_j, &net_infos, &dist_cache, network, cfg, validity);
                    let post_i = evaluate_cell_at(i, xj, yj, nets_i, &net_infos, &dist_cache, network, cfg, validity);
                    let post_j = evaluate_cell_at(j, xi, yi, nets_j, &net_infos, &dist_cache, network, cfg, validity);
                    if !(pre_i.is_finite() && pre_j.is_finite() && post_i.is_finite() && post_j.is_finite()) {
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
                let pct = if n_tested > 0 { 100.0 * n_improve as f64 / n_tested as f64 } else { 0.0 };
                let avg_imp = if n_improve > 0 { sum_improve / n_improve as f64 } else { 0.0 };
                let avg_delta = if n_tested > 0 { sum_delta / n_tested as f64 } else { 0.0 };
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
            "  DCD {:3}: nets={} chpwl={:.0} line={:.0} energy={:.3} dE={:+.3} friction={:.1} bins={:.1}x({}) excess={:.1} moved={} stalls={} solves={} pops={} refresh={}ms dcd={}ms total={}ms",
            outer,
            net_infos.len(),
            chpwl,
            line,
            refresh.0,
            d_energy,
            friction,
            max_overflow,
            n_overflow,
            overflow_excess,
            moved,
            stalls,
            refresh.1.total_solves,
            refresh.1.total_heap_pops,
            refresh_ms,
            dcd_ms,
            t_outer.elapsed().as_millis(),
        );

        if moved == 0 {
            eprintln!("  DCD converged: no cells moved in iteration {}", outer);
            break;
        }
        if stalls >= max_stalls {
            eprintln!("  DCD stopped: {} consecutive non-improving iterations", max_stalls);
            break;
        }
    }

    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);

    diag_ctx.finalize(network);

    eprintln!(
        "DCD done: energy={:.3} chpwl={:.0} line={:.0} friction={:.1} bins={:.1}x({}) excess={:.1}",
        best_state.energy,
        best_state.chpwl,
        best_state.line,
        best_state.friction,
        best_state.max_overflow,
        best_state.n_overflow,
        best_state.overflow_excess,
    );
}

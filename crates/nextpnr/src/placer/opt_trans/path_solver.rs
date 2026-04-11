use std::cmp::Ordering;
use std::collections::BinaryHeap;

use log::warn;
use rayon::prelude::*;

use super::config::OptTransPlacerCfg;
use super::demand::{self, NetPinData, NetSolveInfo};
use super::network::{Pipe, PipeNetwork};

// ---------------------------------------------------------------------------
// Heap entry for Dijkstra
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeapEntry {
    pub dist: f64,
    pub node: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Workspace — only per-solve bookkeeping, graph is read-only & shared
// ---------------------------------------------------------------------------

pub struct PathSolverWorkspace {
    /// Per-node solve output. Initialised to INFINITY; only nodes actually
    /// reached by Dijkstra are overwritten. Reset is sparse via `touched`.
    pub dist: Vec<f64>,
    prev_pipe: Vec<usize>,

    /// Nodes whose dist/prev were written during the current solve.
    /// Used for O(touched) reset instead of O(V) memset.
    touched: Vec<usize>,

    /// Dense boolean: is this node a sink for the current net?
    is_sink: Vec<bool>,

    /// Per-net edge dedup stamp (same pattern as before).
    pipe_stamp: Vec<u32>,
    stamp_epoch: u32,

    /// Priority queue — reused, never reallocated.
    pub heap: BinaryHeap<HeapEntry>,
}

impl PathSolverWorkspace {
    fn new(n_nodes: usize, n_pipes: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n_nodes],
            prev_pipe: vec![usize::MAX; n_nodes],
            touched: Vec::with_capacity(1024),
            is_sink: vec![false; n_nodes],
            pipe_stamp: vec![0; n_pipes],
            stamp_epoch: 1,
            heap: BinaryHeap::with_capacity(4096),
        }
    }

    /// Sparse reset: only clears nodes written in the previous solve.
    fn begin_net(&mut self) {
        for &node in &self.touched {
            self.dist[node] = f64::INFINITY;
            self.prev_pipe[node] = usize::MAX;
        }
        self.touched.clear();
        self.heap.clear();

        self.stamp_epoch = self.stamp_epoch.wrapping_add(1);
        if self.stamp_epoch == 0 {
            self.pipe_stamp.fill(0);
            self.stamp_epoch = 1;
        }
    }

    fn mark_sinks(&mut self, sink_nodes: &[usize]) {
        for &node in sink_nodes {
            self.is_sink[node] = true;
        }
    }

    fn clear_sinks(&mut self, sink_nodes: &[usize]) {
        for &node in sink_nodes {
            self.is_sink[node] = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Path statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct PathStats {
    pub total_solves: usize,
    pub total_heap_pops: usize,
    pub max_heap_pops: usize,
    pub failures: usize,
}

#[derive(Debug, Clone)]
pub struct PathForwardStats {
    pub global_pressure: Vec<f64>,
    pub edge_usage: Vec<f64>,
    pub attraction_energy: f64,
}

#[derive(Debug)]
struct LocalAccum {
    grad: Vec<f64>,
    edge_usage: Vec<f64>,
    attraction_energy: f64,
    stats: PathStats,
}

// ---------------------------------------------------------------------------
// Public API (standalone Dijkstra for external callers)
// ---------------------------------------------------------------------------

pub fn dijkstra_single_source(
    network: &PipeNetwork,
    source_node: usize,
    edge_cost: impl Fn(&Pipe) -> f64,
    dist_out: &mut [f64],
) -> usize {
    dist_out.fill(f64::INFINITY);
    if source_node >= dist_out.len() {
        return 0;
    }
    dist_out[source_node] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node: source_node,
    });

    let mut heap_pops = 0usize;
    while let Some(HeapEntry {
        dist: cur_dist,
        node,
    }) = heap.pop()
    {
        heap_pops += 1;
        if cur_dist > dist_out[node] {
            continue;
        }
        for &pipe_idx in &network.node_pipes[node] {
            let pipe = &network.pipes[pipe_idx];
            let next = if pipe.from == node {
                pipe.to
            } else if pipe.to == node {
                pipe.from
            } else {
                continue;
            };
            let cost = edge_cost(pipe);
            if !cost.is_finite() || cost <= 0.0 {
                continue;
            }
            let candidate = cur_dist + cost;
            if candidate < dist_out[next] {
                dist_out[next] = candidate;
                heap.push(HeapEntry {
                    dist: candidate,
                    node: next,
                });
            }
        }
    }
    heap_pops
}

// ---------------------------------------------------------------------------
// Batch path computation (hot path)
// ---------------------------------------------------------------------------

pub fn compute_path_forward_stats(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    objective_scale: f64,
) -> (PathForwardStats, PathStats) {
    let (_, forward, stats) = compute_path_forward_and_gradient_impl(
        network,
        net_infos,
        cfg,
        solve_pool,
        0,
        objective_scale,
        false,
    );
    (forward, stats)
}

pub fn compute_path_gradient_from_usage(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    objective_scale: f64,
) -> (Vec<f64>, PathStats) {
    let (grad, _, stats) = compute_path_forward_and_gradient_impl(
        network,
        net_infos,
        cfg,
        solve_pool,
        n,
        objective_scale,
        true,
    );
    (grad, stats)
}

pub fn compute_path_forward_and_gradient(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    objective_scale: f64,
) -> (Vec<f64>, PathForwardStats, PathStats) {
    compute_path_forward_and_gradient_impl(
        network,
        net_infos,
        cfg,
        solve_pool,
        n,
        objective_scale,
        true,
    )
}

fn compute_path_forward_and_gradient_impl(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    objective_scale: f64,
    include_gradient: bool,
) -> (Vec<f64>, PathForwardStats, PathStats) {
    let n_nodes = network.num_nodes();
    let n_pipes = network.num_pipes();

    // Log graph size once.
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrd};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, AtomicOrd::Relaxed) {
        eprintln!(
            "    path solver: graph {} nodes, {} pipes, {} nets",
            n_nodes,
            n_pipes,
            net_infos.len(),
        );
    }

    let batch_size = cfg.net_parallel_batch_size.max(1);
    let accum = solve_pool.install(|| {
        net_infos
            .par_chunks(batch_size)
            .fold(
                || {
                    (
                        LocalAccum {
                            grad: vec![0.0; 2 * n],
                            edge_usage: vec![0.0; n_pipes],
                            attraction_energy: 0.0,
                            stats: PathStats::default(),
                        },
                        PathSolverWorkspace::new(n_nodes, n_pipes),
                    )
                },
                |(mut local, mut ws), chunk| {
                    for info in chunk {
                        evaluate_net_path(
                            network,
                            info,
                            cfg,
                            objective_scale,
                            include_gradient,
                            &mut local,
                            &mut ws,
                        );
                    }
                    (local, ws)
                },
            )
            .map(|(local, _)| local)
            .reduce(
                || LocalAccum {
                    grad: vec![0.0; 2 * n],
                    edge_usage: vec![0.0; n_pipes],
                    attraction_energy: 0.0,
                    stats: PathStats::default(),
                },
                |mut a, b| {
                    for (dst, &src) in a.grad.iter_mut().zip(b.grad.iter()) {
                        *dst += src;
                    }
                    for (dst, &src) in a.edge_usage.iter_mut().zip(b.edge_usage.iter()) {
                        *dst += src;
                    }
                    a.attraction_energy += b.attraction_energy;
                    a.stats.total_solves += b.stats.total_solves;
                    a.stats.total_heap_pops += b.stats.total_heap_pops;
                    a.stats.max_heap_pops = a.stats.max_heap_pops.max(b.stats.max_heap_pops);
                    a.stats.failures += b.stats.failures;
                    a
                },
            )
    });

    let forward = PathForwardStats {
        global_pressure: vec![0.0; n_nodes],
        edge_usage: accum.edge_usage,
        attraction_energy: accum.attraction_energy,
    };
    (accum.grad, forward, accum.stats)
}

// ---------------------------------------------------------------------------
// Per-net evaluation with early termination
// ---------------------------------------------------------------------------

fn evaluate_net_path(
    network: &PipeNetwork,
    info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    objective_scale: f64,
    include_gradient: bool,
    local: &mut LocalAccum,
    ws: &mut PathSolverWorkspace,
) {
    let Some(source_pin) = info.pin_data.iter().find(|pin| pin.is_driver) else {
        local.stats.failures += 1;
        warn!("path solver: net {:?} has no driver pin", info.net_id);
        return;
    };
    let Some(source_node) = nearest_source_node(source_pin) else {
        local.stats.failures += 1;
        warn!(
            "path solver: net {:?} has no valid source node",
            info.net_id
        );
        return;
    };

    // Collect all unique sink nodes for early termination.
    let mut sink_nodes: Vec<usize> = Vec::with_capacity(info.pin_data.len() * 4);
    for pin in &info.pin_data {
        if pin.is_driver {
            continue;
        }
        for j in 0..4 {
            if pin.weights[j] != 0.0 {
                sink_nodes.push(pin.nodes[j]);
            }
        }
    }
    sink_nodes.sort_unstable();
    sink_nodes.dedup();

    ws.begin_net();
    ws.mark_sinks(&sink_nodes);
    let heap_pops = dijkstra_early_terminate(network, source_node, sink_nodes.len(), ws);
    ws.clear_sinks(&sink_nodes);

    local.stats.total_solves += 1;
    local.stats.total_heap_pops += heap_pops;
    local.stats.max_heap_pops = local.stats.max_heap_pops.max(heap_pops);

    let timing_scale = net_path_weight(info, cfg) * objective_scale;
    for pin in &info.pin_data {
        if pin.is_driver {
            continue;
        }

        let Some(cost) = sink_cost(pin, &ws.dist) else {
            if !pin.is_fixed && pin.cell_idx.is_some() {
                local.stats.failures += 1;
                warn!(
                    "path solver: disconnected movable sink on net {:?} ({})",
                    info.net_id, info.debug_name
                );
            }
            continue;
        };

        local.attraction_energy += timing_scale * cost;
        mark_sink_paths(pin, source_node, network, ws, &mut local.edge_usage);

        if include_gradient {
            accumulate_sink_gradient(pin, &ws.dist, timing_scale, &mut local.grad);
        }
    }
}

fn nearest_source_node(pin: &NetPinData) -> Option<usize> {
    let mut best_weight = f64::NEG_INFINITY;
    let mut best_node = None;
    for j in 0..4 {
        let weight = pin.weights[j];
        let node = pin.nodes[j];
        if weight > best_weight
            || ((weight - best_weight).abs() <= f64::EPSILON && Some(node) < best_node)
        {
            best_weight = weight;
            best_node = Some(node);
        }
    }
    best_node
}

fn net_path_weight(info: &NetSolveInfo, cfg: &OptTransPlacerCfg) -> f64 {
    let base_io_factor = if info.has_fixed_pin {
        cfg.io_boost
    } else {
        1.0
    };
    let fanout = (info.pins.len() - 1) as f64;
    let fanout_scale = if cfg.fanout_weight_sqrt {
        fanout.sqrt().max(1.0)
    } else {
        1.0
    };
    let fanout_norm = fanout.powf(cfg.fanout_norm_exp.clamp(0.0, 1.0)).max(1.0);
    base_io_factor * fanout_scale / fanout_norm * demand::net_timing_weight(info, cfg)
}

// ---------------------------------------------------------------------------
// Dijkstra with early termination + sparse workspace
//
// The graph (network) is shared & read-only. Only dist[] and prev_pipe[]
// are per-thread, and we track which entries we wrote so the reset is
// O(touched) instead of O(V). This keeps the working set small for local
// nets and avoids cache-thrashing on the 144 k-node graph.
// ---------------------------------------------------------------------------

#[inline(never)]
fn dijkstra_early_terminate(
    network: &PipeNetwork,
    source_node: usize,
    n_sinks: usize,
    ws: &mut PathSolverWorkspace,
) -> usize {
    let dist = &mut ws.dist;
    let prev_pipe = &mut ws.prev_pipe;
    let is_sink = &ws.is_sink;
    let heap = &mut ws.heap;
    let touched = &mut ws.touched;

    // Seed source — record the touch.
    dist[source_node] = 0.0;
    touched.push(source_node);
    heap.push(HeapEntry {
        dist: 0.0,
        node: source_node,
    });

    let mut remaining = n_sinks;
    let pipes = &network.pipes;
    let node_pipes = &network.node_pipes;

    let mut heap_pops = 0usize;
    while let Some(HeapEntry {
        dist: cur_dist,
        node,
    }) = heap.pop()
    {
        heap_pops += 1;
        if cur_dist > dist[node] {
            continue;
        }

        if remaining > 0 && is_sink[node] {
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }

        for &pipe_idx in &node_pipes[node] {
            let pipe = &pipes[pipe_idx];
            let next = if pipe.from == node {
                pipe.to
            } else if pipe.to == node {
                pipe.from
            } else {
                continue;
            };

            let g = pipe.eff_conductance;
            if !g.is_finite() || g <= 0.0 {
                continue;
            }

            let candidate = cur_dist + 1.0 / g;
            if candidate < dist[next] {
                // First time reaching this node? Record for sparse reset.
                if dist[next] == f64::INFINITY {
                    touched.push(next);
                }
                dist[next] = candidate;
                prev_pipe[next] = pipe_idx;
                heap.push(HeapEntry {
                    dist: candidate,
                    node: next,
                });
            }
        }
    }

    heap_pops
}

// ---------------------------------------------------------------------------
// Cost / gradient / path-marking helpers
// ---------------------------------------------------------------------------

fn sink_cost(pin: &NetPinData, dist: &[f64]) -> Option<f64> {
    let mut cost = 0.0;
    for j in 0..4 {
        let weight = pin.weights[j];
        if weight == 0.0 {
            continue;
        }
        let d = dist[pin.nodes[j]];
        if !d.is_finite() {
            return None;
        }
        cost += weight * d;
    }
    Some(cost)
}

fn accumulate_sink_gradient(pin: &NetPinData, dist: &[f64], timing_scale: f64, grad: &mut [f64]) {
    let Some(ci) = pin.cell_idx else {
        return;
    };
    if pin.is_fixed || pin.is_driver {
        return;
    }

    let n = grad.len() / 2;
    if ci >= n {
        return;
    }

    let mut grad_x = 0.0;
    let mut grad_y = 0.0;
    for j in 0..4 {
        let d = dist[pin.nodes[j]];
        if !d.is_finite() {
            return;
        }
        grad_x += pin.dw_dx[j] * d;
        grad_y += pin.dw_dy[j] * d;
    }

    grad[ci] += timing_scale * grad_x;
    grad[n + ci] += timing_scale * grad_y;
}

fn mark_sink_paths(
    pin: &NetPinData,
    source_node: usize,
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    edge_usage: &mut [f64],
) {
    for j in 0..4 {
        if pin.weights[j] == 0.0 {
            continue;
        }
        let node = pin.nodes[j];
        if !ws.dist[node].is_finite() {
            continue;
        }
        mark_path_to_source(node, source_node, network, ws, edge_usage);
    }
}

fn mark_path_to_source(
    mut node: usize,
    source_node: usize,
    network: &PipeNetwork,
    ws: &mut PathSolverWorkspace,
    edge_usage: &mut [f64],
) {
    let mut steps = 0usize;
    let max_steps = ws.prev_pipe.len();
    while node != source_node && steps < max_steps {
        let pipe_idx = ws.prev_pipe[node];
        if pipe_idx == usize::MAX {
            break;
        }
        if ws.pipe_stamp[pipe_idx] != ws.stamp_epoch {
            ws.pipe_stamp[pipe_idx] = ws.stamp_epoch;
            edge_usage[pipe_idx] += 1.0;
        }
        let pipe = &network.pipes[pipe_idx];
        node = if pipe.from == node {
            pipe.to
        } else if pipe.to == node {
            pipe.from
        } else {
            break;
        };
        steps += 1;
    }
}

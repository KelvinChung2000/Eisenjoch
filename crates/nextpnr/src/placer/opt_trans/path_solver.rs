use std::cmp::Ordering;
use std::collections::BinaryHeap;

use log::warn;
use rayon::prelude::*;

use super::config::OptTransPlacerCfg;
use super::demand::{self, NetPinData, NetSolveInfo};
use super::network::{Pipe, PipeNetwork};

pub(crate) const LOGIT_THETA: f64 = 0.25;
const REASONABLE_EPS: f64 = 1e-10;
const CORRIDOR_STRETCH: f64 = 1.35;
const CORRIDOR_MIN_HALO: i32 = 12;
const CORRIDOR_REL_HALO_NUM: i32 = 1;
const CORRIDOR_REL_HALO_DEN: i32 = 4;

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
    /// Per-node routing cost. Used as Dijkstra priority.
    /// Initialised to INFINITY; only reached nodes are overwritten.
    pub dist: Vec<f64>,
    /// Dial forward weight at each reached node.
    path_weight: Vec<f64>,
    /// Dial backward demand loaded through each reached node.
    node_load: Vec<f64>,
    /// Per-edge load written by the current tentative solve. This lets a
    /// corridor attempt roll back partial usage before falling back to the
    /// full graph.
    edge_load: Vec<f64>,
    edge_touched: Vec<usize>,

    /// Nodes whose dist was written during the current solve.
    /// Used for O(touched) reset instead of O(V) memset.
    pub(crate) touched: Vec<usize>,

    /// Dijkstra settle order (nodes pushed as they are popped with a fresh
    /// min). Already in dist order — replaces a post-hoc sort of `touched`.
    pub(crate) settle_order: Vec<usize>,

    /// Per-node bitmap: true if the node is inside the current corridor.
    /// Populated once per net by `mark_corridor`; reset via `corridor_marked`.
    pub(crate) in_corridor: Vec<bool>,
    /// Nodes whose in_corridor bit was set this net (sparse reset).
    corridor_marked: Vec<usize>,

    /// Priority queue — reused, never reallocated.
    pub heap: BinaryHeap<HeapEntry>,
}

impl PathSolverWorkspace {
    pub(crate) fn new(n_nodes: usize, n_pipes: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n_nodes],
            path_weight: vec![0.0; n_nodes],
            node_load: vec![0.0; n_nodes],
            edge_load: vec![0.0; n_pipes],
            edge_touched: Vec::with_capacity(1024),
            touched: Vec::with_capacity(1024),
            settle_order: Vec::with_capacity(1024),
            in_corridor: vec![false; n_nodes],
            corridor_marked: Vec::with_capacity(1024),
            heap: BinaryHeap::with_capacity(4096),
        }
    }

    /// Sparse reset: only clears nodes written in the previous solve.
    pub(crate) fn begin_net(&mut self) {
        for &node in &self.touched {
            self.dist[node] = f64::INFINITY;
            self.path_weight[node] = 0.0;
            self.node_load[node] = 0.0;
        }
        for &pipe in &self.edge_touched {
            self.edge_load[pipe] = 0.0;
        }
        for &node in &self.corridor_marked {
            self.in_corridor[node] = false;
        }
        self.touched.clear();
        self.settle_order.clear();
        self.edge_touched.clear();
        self.corridor_marked.clear();
        self.heap.clear();
    }

    /// Populate `in_corridor` for every node in the bbox that passes the
    /// Steiner check. Called once per net; hot loops read the bitmap.
    fn mark_corridor(&mut self, network: &PipeNetwork, corridor: &Corridor) {
        for ty in corridor.min_y..=corridor.max_y {
            for tx in corridor.min_x..=corridor.max_x {
                let idx = network.node_index(tx, ty);
                if idx >= self.in_corridor.len() {
                    continue;
                }
                if corridor.contains_xy(tx, ty) {
                    self.in_corridor[idx] = true;
                    self.corridor_marked.push(idx);
                }
            }
        }
    }

    fn record_edge_usage(&mut self, pipe_idx: usize, flow: f64, edge_usage: &mut [f64]) {
        if self.edge_load[pipe_idx] == 0.0 {
            self.edge_touched.push(pipe_idx);
        }
        self.edge_load[pipe_idx] += flow;
        edge_usage[pipe_idx] += flow;
    }

    fn rollback_edge_usage(&mut self, edge_usage: &mut [f64]) {
        for &pipe in &self.edge_touched {
            edge_usage[pipe] -= self.edge_load[pipe];
            self.edge_load[pipe] = 0.0;
        }
        self.edge_touched.clear();
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

pub(crate) struct DialLogitResult {
    pub heap_pops: usize,
    pub energy: f64,
    pub missing_demand: f64,
}

#[derive(Debug)]
struct Corridor {
    source_x: i32,
    source_y: i32,
    halo: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    sinks: Vec<CorridorSink>,
}

#[derive(Debug)]
struct CorridorSink {
    x: i32,
    y: i32,
    direct: i32,
}

impl Corridor {
    fn from_demands(
        network: &PipeNetwork,
        source_node: usize,
        sink_demands: &[(usize, f64)],
    ) -> Option<Self> {
        let source = network.nodes.get(source_node)?;
        let mut min_x = source.tile_x;
        let mut max_x = source.tile_x;
        let mut min_y = source.tile_y;
        let mut max_y = source.tile_y;
        let mut sinks = Vec::new();
        let mut max_direct = 0i32;

        for &(node, demand) in sink_demands {
            if demand == 0.0 {
                continue;
            }
            let sink = network.nodes.get(node)?;
            min_x = min_x.min(sink.tile_x);
            max_x = max_x.max(sink.tile_x);
            min_y = min_y.min(sink.tile_y);
            max_y = max_y.max(sink.tile_y);
            let direct =
                manhattan_xy(source.tile_x, source.tile_y, sink.tile_x, sink.tile_y).max(1);
            max_direct = max_direct.max(direct);
            sinks.push(CorridorSink {
                x: sink.tile_x,
                y: sink.tile_y,
                direct,
            });
        }

        if sinks.is_empty() {
            return None;
        }

        let rel_halo = (max_direct * CORRIDOR_REL_HALO_NUM + CORRIDOR_REL_HALO_DEN - 1)
            / CORRIDOR_REL_HALO_DEN;
        let halo = CORRIDOR_MIN_HALO.max(rel_halo);
        Some(Self {
            source_x: source.tile_x,
            source_y: source.tile_y,
            halo,
            min_x: (min_x - halo).max(0),
            max_x: (max_x + halo).min(network.width - 1),
            min_y: (min_y - halo).max(0),
            max_y: (max_y + halo).min(network.height - 1),
            sinks,
        })
    }

    #[inline]
    fn contains_xy(&self, tx: i32, ty: i32) -> bool {
        if tx < self.min_x || tx > self.max_x || ty < self.min_y || ty > self.max_y {
            return false;
        }
        self.sinks.iter().any(|sink| {
            let via = manhattan_xy(self.source_x, self.source_y, tx, ty)
                + manhattan_xy(tx, ty, sink.x, sink.y);
            (via as f64) <= CORRIDOR_STRETCH * (sink.direct as f64) + (self.halo as f64)
        })
    }
}

#[inline]
fn manhattan_xy(ax: i32, ay: i32, bx: i32, by: i32) -> i32 {
    (ax - bx).abs() + (ay - by).abs()
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

    let mut sink_demands: Vec<(usize, f64)> = Vec::with_capacity(info.pin_data.len() * 4);
    for pin in &info.pin_data {
        if pin.is_driver {
            continue;
        }
        for j in 0..4 {
            let weight = pin.weights[j];
            if weight != 0.0 {
                sink_demands.push((pin.nodes[j], weight));
            }
        }
    }

    ws.begin_net();
    let result = dial_logit_load(
        network,
        source_node,
        &sink_demands,
        ws,
        Some(&mut local.edge_usage),
    );

    local.stats.total_solves += 1;
    local.stats.total_heap_pops += result.heap_pops;
    local.stats.max_heap_pops = local.stats.max_heap_pops.max(result.heap_pops);
    if result.missing_demand > 0.0 {
        local.stats.failures += 1;
        warn!(
            "path solver: unreachable sink demand on net {:?} ({})",
            info.net_id, info.debug_name
        );
    }

    let timing_scale = net_path_weight(info, cfg) * objective_scale;
    local.attraction_energy += timing_scale * result.energy;
    for pin in &info.pin_data {
        if pin.is_driver {
            continue;
        }

        let Some(_cost) = sink_cost(pin, &ws.dist) else {
            if !pin.is_fixed && pin.cell_idx.is_some() {
                local.stats.failures += 1;
                warn!(
                    "path solver: disconnected movable sink on net {:?} ({})",
                    info.net_id, info.debug_name
                );
            }
            continue;
        };

        if include_gradient {
            accumulate_sink_gradient(pin, &ws.dist, timing_scale, &mut local.grad);
        }
    }
}

pub(crate) fn nearest_source_node(pin: &NetPinData) -> Option<usize> {
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

pub(crate) fn net_path_weight(info: &NetSolveInfo, cfg: &OptTransPlacerCfg) -> f64 {
    let io_factor = if info.has_fixed_pin {
        cfg.io_boost
    } else {
        1.0
    };
    io_factor * demand::net_timing_weight(info, cfg)
}

#[inline(never)]
fn dijkstra_all_labels(
    network: &PipeNetwork,
    source_node: usize,
    ws: &mut PathSolverWorkspace,
    use_corridor: bool,
) -> usize {
    let dist = &mut ws.dist;
    let heap = &mut ws.heap;
    let touched = &mut ws.touched;
    let settle_order = &mut ws.settle_order;
    let in_corridor = &ws.in_corridor;
    let pipes = &network.pipes;
    let pipe_costs = &network.pipe_costs;
    let node_pipes = &network.node_pipes;

    dist[source_node] = 0.0;
    touched.push(source_node);
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
        if cur_dist > dist[node] {
            continue;
        }
        settle_order.push(node);

        for &pipe_idx in &node_pipes[node] {
            let pipe = &pipes[pipe_idx];
            // XOR trick: since this pipe is in `node_pipes[node]`, either
            // pipe.from == node or pipe.to == node. The other endpoint is
            // (pipe.from ^ pipe.to ^ node).
            let next = pipe.from ^ pipe.to ^ node;
            if use_corridor && !in_corridor[next] {
                continue;
            }

            let cost = pipe_costs[pipe_idx];
            let candidate = cur_dist + cost;
            if candidate < dist[next] {
                if dist[next] == f64::INFINITY {
                    touched.push(next);
                }
                dist[next] = candidate;
                heap.push(HeapEntry {
                    dist: candidate,
                    node: next,
                });
            }
        }
    }

    heap_pops
}

#[inline]
fn link_likelihood(label_from: f64, label_to: f64, cost: f64) -> f64 {
    let exponent = (LOGIT_THETA * (label_to - label_from - cost)).min(0.0);
    exponent.exp()
}

pub(crate) fn dial_logit_load(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    mut edge_usage: Option<&mut [f64]>,
) -> DialLogitResult {
    if let Some(corridor) = Corridor::from_demands(network, source_node, sink_demands) {
        ws.mark_corridor(network, &corridor);
        let result = dial_logit_load_inner(
            network,
            source_node,
            sink_demands,
            ws,
            edge_usage.as_deref_mut(),
            true,
        );
        if result.missing_demand == 0.0 {
            return result;
        }
        if let Some(usage) = edge_usage.as_deref_mut() {
            ws.rollback_edge_usage(usage);
        }
        ws.begin_net();
    }

    dial_logit_load_inner(network, source_node, sink_demands, ws, edge_usage, false)
}

fn dial_logit_load_inner(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    mut edge_usage: Option<&mut [f64]>,
    use_corridor: bool,
) -> DialLogitResult {
    let heap_pops = dijkstra_all_labels(network, source_node, ws, use_corridor);

    // Dijkstra settle order is already sorted by dist ascending — no post-hoc
    // sort needed. We read `settle_order` via a borrow-split into a scratch
    // slice so the forward/backward edge loops can still mutably borrow
    // `ws.path_weight` and `ws.node_load`.
    let pipes = &network.pipes;
    let pipe_costs = &network.pipe_costs;
    let node_pipes = &network.node_pipes;

    ws.path_weight[source_node] = 1.0;
    // Forward pass: spread Dial weight along strictly-downhill edges.
    // Iterate settle_order indices (avoids cloning/borrow-splitting the vec).
    let n_settled = ws.settle_order.len();
    for i in 0..n_settled {
        let node = ws.settle_order[i];
        let node_weight = ws.path_weight[node];
        if node_weight <= 0.0 || !node_weight.is_finite() {
            continue;
        }
        let label = ws.dist[node];
        for &pipe_idx in &node_pipes[node] {
            let pipe = &pipes[pipe_idx];
            let next = pipe.from ^ pipe.to ^ node;
            if use_corridor && !ws.in_corridor[next] {
                continue;
            }
            let next_label = ws.dist[next];
            if !next_label.is_finite() || next_label <= label + REASONABLE_EPS {
                continue;
            }
            let cost = pipe_costs[pipe_idx];
            let likelihood = link_likelihood(label, next_label, cost);
            if likelihood > 0.0 && likelihood.is_finite() {
                ws.path_weight[next] += node_weight * likelihood;
            }
        }
    }

    // Canonical per-net energy: `demand * label` summed across sinks.
    // `label` is the Dijkstra shortest-path distance under R_eff edges, so this
    // scalar blends wirelength (R_eff ≈ base when uncongested) and congestion
    // (BPR grows R_eff with usage) in a single strictly-positive quantity.
    let mut energy = 0.0;
    let mut missing_demand = 0.0;
    for &(node, demand) in sink_demands {
        if demand == 0.0 {
            continue;
        }
        let label = ws.dist[node];
        let weight = ws.path_weight[node];
        if !label.is_finite() || weight <= 0.0 || !weight.is_finite() {
            missing_demand += demand.abs();
            continue;
        }
        ws.node_load[node] += demand;
        energy += demand * label;
    }

    // Backward pass: distribute loaded demand back along predecessor edges.
    for i in (0..n_settled).rev() {
        let node = ws.settle_order[i];
        let load = ws.node_load[node];
        let node_weight = ws.path_weight[node];
        if load == 0.0 || node_weight <= 0.0 || !node_weight.is_finite() {
            continue;
        }
        let node_label = ws.dist[node];
        if !node_label.is_finite() {
            continue;
        }
        for &pipe_idx in &node_pipes[node] {
            let pipe = &pipes[pipe_idx];
            let pred = pipe.from ^ pipe.to ^ node;
            if use_corridor && !ws.in_corridor[pred] {
                continue;
            }
            let pred_label = ws.dist[pred];
            if !pred_label.is_finite() || pred_label >= node_label - REASONABLE_EPS {
                continue;
            }
            let cost = pipe_costs[pipe_idx];
            let likelihood = link_likelihood(pred_label, node_label, cost);
            if likelihood <= 0.0 || !likelihood.is_finite() {
                continue;
            }
            let pred_weight = ws.path_weight[pred];
            let share = pred_weight * likelihood / node_weight;
            if share <= 0.0 || !share.is_finite() {
                continue;
            }
            let flow = load * share;
            ws.node_load[pred] += flow;
            if let Some(usage) = edge_usage.as_deref_mut() {
                ws.record_edge_usage(pipe_idx, flow, usage);
            }
        }
    }

    DialLogitResult {
        heap_pops,
        energy,
        missing_demand,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, Node, PipeType};
    use rustc_hash::FxHashMap;

    fn test_network(n_nodes: usize, edges: &[(usize, usize, f64)]) -> PipeNetwork {
        let nodes = (0..n_nodes)
            .map(|i| Node {
                tile_x: i as i32,
                tile_y: 0,
                pressure: 0.0,
            })
            .collect();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); n_nodes];
        for &(from, to, cost) in edges {
            let idx = pipes.len();
            pipes.push(Pipe {
                from,
                to,
                base_resistance: cost,
                capacity: 10.0,
                flow: 0.0,
                net_count: 0.0,
                raw_cell_density: 0.0,
                cell_density: 0.0,
                eff_conductance: 1.0 / cost,
                pipe_type: PipeType::InterTile(Direction::East),
            });
            node_pipes[from].push(idx);
            node_pipes[to].push(idx);
        }
        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|p| 1.0 / p.eff_conductance.max(1e-12))
            .collect();
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_lookup: FxHashMap::default(),
            width: n_nodes as i32,
            height: 1,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: n_nodes,
            coarsen: 1,
        }
    }

    fn test_network_with_coords(
        coords: &[(i32, i32)],
        edges: &[(usize, usize, f64)],
    ) -> PipeNetwork {
        let nodes = coords
            .iter()
            .map(|&(tile_x, tile_y)| Node {
                tile_x,
                tile_y,
                pressure: 0.0,
            })
            .collect();
        let mut pipes = Vec::new();
        let mut node_pipes = vec![Vec::new(); coords.len()];
        for &(from, to, cost) in edges {
            let idx = pipes.len();
            pipes.push(Pipe {
                from,
                to,
                base_resistance: cost,
                capacity: 10.0,
                flow: 0.0,
                net_count: 0.0,
                raw_cell_density: 0.0,
                cell_density: 0.0,
                eff_conductance: 1.0 / cost,
                pipe_type: PipeType::InterTile(Direction::East),
            });
            node_pipes[from].push(idx);
            node_pipes[to].push(idx);
        }
        let width = coords.iter().map(|&(x, _)| x).max().unwrap_or(0) + 1;
        let height = coords.iter().map(|&(_, y)| y).max().unwrap_or(0) + 1;
        let pipe_costs: Vec<f64> = pipes
            .iter()
            .map(|p| 1.0 / p.eff_conductance.max(1e-12))
            .collect();
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_lookup: FxHashMap::default(),
            width,
            height,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: coords.len(),
            coarsen: 1,
        }
    }

    #[test]
    fn dial_single_path_loads_all_demand() {
        let network = test_network(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let mut usage = vec![0.0; network.num_pipes()];
        let result = dial_logit_load(&network, 0, &[(2, 1.0)], &mut ws, Some(&mut usage));

        assert_eq!(result.missing_demand, 0.0);
        assert!((usage[0] - 1.0).abs() < 1e-12);
        assert!((usage[1] - 1.0).abs() < 1e-12);
        assert!((ws.dist[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dial_equal_cost_diamond_splits_evenly() {
        let network = test_network(4, &[(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 1.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let mut usage = vec![0.0; network.num_pipes()];
        let result = dial_logit_load(&network, 0, &[(3, 1.0)], &mut ws, Some(&mut usage));

        assert_eq!(result.missing_demand, 0.0);
        for u in usage {
            assert!((u - 0.5).abs() < 1e-12);
        }
        // `ws.dist[3]` is the pure Dijkstra shortest-path distance; both arms
        // of the diamond cost 2, so label = 2 regardless of demand split.
        assert!((ws.dist[3] - 2.0).abs() < 1e-12);
        assert!((result.energy - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dial_unequal_cost_diamond_prefers_cheaper_branch() {
        let network = test_network(4, &[(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 2.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let mut usage = vec![0.0; network.num_pipes()];
        let result = dial_logit_load(&network, 0, &[(3, 1.0)], &mut ws, Some(&mut usage));

        assert_eq!(result.missing_demand, 0.0);
        assert!(usage[0] > usage[2]);
        assert!(usage[1] > usage[3]);
        let total_into_sink = usage[1] + usage[3];
        assert!((total_into_sink - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dial_reports_unreachable_sink_demand() {
        let network = test_network(3, &[(0, 1, 1.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let mut usage = vec![0.0; network.num_pipes()];
        let result = dial_logit_load(&network, 0, &[(2, 1.5)], &mut ws, Some(&mut usage));

        assert_eq!(result.missing_demand, 1.5);
        assert_eq!(usage[0], 0.0);
    }

    #[test]
    fn corridor_fallback_does_not_double_count_usage() {
        let network =
            test_network_with_coords(&[(0, 0), (0, 100), (10, 0)], &[(0, 1, 1.0), (1, 2, 1.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let mut usage = vec![0.0; network.num_pipes()];
        let result = dial_logit_load(&network, 0, &[(2, 1.0)], &mut ws, Some(&mut usage));

        assert_eq!(result.missing_demand, 0.0);
        assert!((usage[0] - 1.0).abs() < 1e-12);
        assert!((usage[1] - 1.0).abs() < 1e-12);
        assert!((ws.dist[2] - 2.0).abs() < 1e-12);
    }
}

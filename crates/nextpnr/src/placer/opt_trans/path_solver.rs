use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{LazyLock, Mutex, OnceLock};

use log::warn;
use rayon::prelude::*;

use super::config::OptTransPlacerCfg;
use super::demand::{self, NetPinData, NetSolveInfo};
use super::network::{Pipe, PipeNetwork, DIST_SCALE};

pub(crate) const LOGIT_THETA: f64 = 0.25;

/// Pick a chunk size that gives Rayon ~8 chunks per worker thread for
/// work-stealing without spawning thousands of fold-init allocations.
/// Each fold init pays a `vec![0.0; n_pipes]` (~20 MB on FPGA01), so on
/// large designs the default `cfg.net_parallel_batch_size = 4` produced
/// hundreds of inits per pre_solve. Floor at the configured minimum so
/// small designs keep their tuning.
pub(crate) fn auto_batch_size(cfg_min: usize, n_items: usize, pool: &rayon::ThreadPool) -> usize {
    let cfg_min = cfg_min.max(1);
    let threads = pool.current_num_threads().max(1);
    (n_items / (threads * 8)).max(cfg_min)
}

/// Precomputed `exp(-LOGIT_THETA · slack / DIST_SCALE)` as f32, indexed by
/// integer slack in `DIST_SCALE` units. `LOGIT_THETA = 0.25` and `DIST_SCALE =
/// 100.0` are both compile-time constants, so this table is static for the
/// life of the process.
///
/// Size: 8192 × 4 bytes = 32 KB → fits in L1d. Any slack ≥ table length
/// produces `exp(-20.48) ≈ 1.3e-9`, which is below the downstream f64
/// arithmetic's noise floor once multiplied by `path_weight` and clamped by
/// the `likelihood > 0.0 && likelihood.is_finite()` guards, so we short-
/// circuit to 0.0 (same observable behaviour as the scalar exp).
const LIKELIHOOD_LUT_SIZE: usize = 8192;

static LIKELIHOOD_LUT: LazyLock<[f32; LIKELIHOOD_LUT_SIZE]> = LazyLock::new(|| {
    let mut table = [0.0f32; LIKELIHOOD_LUT_SIZE];
    for (i, slot) in table.iter_mut().enumerate() {
        let exponent = -LOGIT_THETA * (i as f64) / DIST_SCALE;
        *slot = exponent.exp() as f32;
    }
    table
});

/// Cached once per process from `NPNR_OT_USE_FSM=1`. When set, `dial_logit_load`
/// routes through `fsm::fsm_dist_and_forward` instead of the fused Dijkstra.
fn use_fsm() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("NPNR_OT_USE_FSM").ok().as_deref() == Some("1"))
}
const REASONABLE_EPS: f64 = 1e-10;
const CORRIDOR_STRETCH_DEFAULT: f64 = 1.35;
const CORRIDOR_MIN_HALO_DEFAULT: i32 = 12;
const CORRIDOR_REL_HALO_NUM_DEFAULT: i32 = 1;
const CORRIDOR_REL_HALO_DEN_DEFAULT: i32 = 4;
const CORRIDOR_STRETCH_TIGHT: f64 = 1.10;
const CORRIDOR_MIN_HALO_TIGHT: i32 = 4;
const CORRIDOR_REL_HALO_NUM_TIGHT: i32 = 1;
const CORRIDOR_REL_HALO_DEN_TIGHT: i32 = 8;

/// Hard cap on per-net corridor halo. Without it, halo grows linearly with
/// pin span (rel_halo = span/rel_den), so a span-200 net gets halo=25 even in
/// tight mode and the resulting bbox covers a quarter of a large fabric. The
/// cap bounds DistCache memory: with cap=12, settle counts stay in the low
/// hundreds even for chip-spanning nets. Override via `NPNR_OT_CORRIDOR_HALO_MAX`.
const CORRIDOR_MAX_HALO_DEFAULT: i32 = 12;

/// Distance slack added to `max_sink_dist` when terminating Dijkstra. Once
/// every sink has been settled, the sweep continues out to
/// `max_sink_dist + slack` so DCD's per-cell candidate evaluation can still
/// query distances within `slack / DIST_SCALE` tiles of any pin.
/// Override via `NPNR_OT_CACHE_SLACK_TILES`. Default = 8 tiles.
const CACHE_SLACK_TILES_DEFAULT: u32 = 8;

fn cache_slack_int() -> u32 {
    use super::network::DIST_SCALE;
    static SLACK: OnceLock<u32> = OnceLock::new();
    *SLACK.get_or_init(|| {
        let tiles = std::env::var("NPNR_OT_CACHE_SLACK_TILES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(CACHE_SLACK_TILES_DEFAULT);
        tiles.saturating_mul(DIST_SCALE as u32)
    })
}

#[derive(Clone, Copy)]
struct CorridorParams {
    stretch: f64,
    min_halo: i32,
    rel_num: i32,
    rel_den: i32,
    max_halo: i32,
}

fn corridor_params() -> CorridorParams {
    static PARAMS: OnceLock<CorridorParams> = OnceLock::new();
    *PARAMS.get_or_init(|| {
        let tight = std::env::var("NPNR_OT_CORRIDOR_TIGHT").ok().as_deref() == Some("1");
        let max_halo = std::env::var("NPNR_OT_CORRIDOR_HALO_MAX")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .map(|v| v.max(1))
            .unwrap_or(CORRIDOR_MAX_HALO_DEFAULT);
        if tight {
            CorridorParams {
                stretch: CORRIDOR_STRETCH_TIGHT,
                min_halo: CORRIDOR_MIN_HALO_TIGHT,
                rel_num: CORRIDOR_REL_HALO_NUM_TIGHT,
                rel_den: CORRIDOR_REL_HALO_DEN_TIGHT,
                max_halo,
            }
        } else {
            CorridorParams {
                stretch: CORRIDOR_STRETCH_DEFAULT,
                min_halo: CORRIDOR_MIN_HALO_DEFAULT,
                rel_num: CORRIDOR_REL_HALO_NUM_DEFAULT,
                rel_den: CORRIDOR_REL_HALO_DEN_DEFAULT,
                max_halo,
            }
        }
    })
}

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
    /// Per-node f64 routing cost, populated from `dist_int` after Dijkstra
    /// completes so the Dial forward/backward passes and downstream consumers
    /// see real-valued labels.
    pub dist: Vec<f64>,
    /// Per-node u32-quantized routing cost used directly by the bucket-Dial
    /// Dijkstra. `u32::MAX` means unreached; every settled node is written
    /// with its min-cost label.
    pub(crate) dist_int: Vec<u32>,
    /// Dial forward weight at each reached node.
    pub(crate) path_weight: Vec<f64>,
    /// Dial backward demand loaded through each reached node.
    pub(crate) node_load: Vec<f64>,
    /// Per-edge load written by the current tentative solve. This lets a
    /// corridor attempt roll back partial usage before falling back to the
    /// full graph.
    pub(crate) edge_load: Vec<f64>,
    pub(crate) edge_touched: Vec<usize>,

    /// Dijkstra settle order (nodes pushed as they are popped with a fresh
    /// min). Already in dist order.
    pub(crate) settle_order: Vec<usize>,

    /// Every node whose `dist_int` this net wrote, settled or not.
    ///
    /// `settle_order` is NOT sufficient as a reset list. Sink-bounded
    /// termination breaks out of the pop loop while relaxed nodes are still
    /// sitting in the heap, and those nodes keep the `dist_int` this net gave
    /// them. Since a workspace is reused across nets, resetting only settled
    /// nodes leaves stale distances that the next net reads as if they were
    /// its own -- which also made results depend on how rayon happened to
    /// group nets into chunks.
    pub(crate) touched_nodes: Vec<usize>,

    /// Per-node bitmap: true if the node is inside the current corridor.
    /// Populated once per net by `mark_corridor`; reset via `corridor_marked`.
    pub(crate) in_corridor: Vec<bool>,
    /// Nodes whose in_corridor bit was set this net (sparse reset).
    pub(crate) corridor_marked: Vec<usize>,
    /// Tile-space bbox of the current corridor: `(min_x, min_y, max_x, max_y)`.
    /// `None` when `mark_corridor` wasn't called (full-graph fallback). FSM
    /// uses this to restrict its raster sweeps to the corridor rectangle.
    pub(crate) corridor_bbox: Option<(i32, i32, i32, i32)>,
    /// Subset of `network.tile_grid.long_pipes` whose endpoints are both in
    /// the current corridor. Materialised once by `mark_corridor` so FSM's
    /// per-round long-pipe relax step doesn't re-scan the full chip's long
    /// pipes (a profile-confirmed bottleneck).
    pub(crate) corridor_long_pipes: Vec<u32>,

    /// Pool of CorridorSink entries reused across nets.
    /// `build_corridor` clears and refills this.
    pub(crate) corridor_sinks: Vec<CorridorSink>,

    /// Per-net sink-node scratch buffer used by `dial_logit_load_inner` to
    /// hand the sink list to `dijkstra_and_forward` for sink-bounded
    /// termination. Reused across nets so the hot per-net path doesn't
    /// allocate.
    pub(crate) sink_nodes_scratch: Vec<usize>,

    /// Per-net pin (tile_x, tile_y) scratch used by the DistCache filter
    /// in `coord_descent::solve_all_nets_with_displacement` to drop cache
    /// entries that lie outside `R` Manhattan tiles of any pin. Reused.
    pub(crate) pin_xy_scratch: Vec<(i32, i32)>,

    /// Flat bucket array for Dial's O(V+E+D_max) Dijkstra. `buckets[d]`
    /// holds every node whose tentative dist == d. Grown on demand.
    /// After each Dijkstra all buckets are empty (reset is implicit).
    pub(crate) buckets: Vec<Vec<u32>>,
    /// Max bucket index touched in the current net; used to bound the reset
    /// scan if we ever break out of Dijkstra early.
    pub(crate) max_bucket: u32,

    /// Binary heap for the public `dijkstra_single_source` fallback path;
    /// never used by the hot `dijkstra_all_labels` (which uses buckets).
    pub heap: BinaryHeap<HeapEntry>,
}

/// Pool of reusable `PathSolverWorkspace`s.
///
/// A workspace owns an `edge_load` array with one slot per pipe -- 20.5 MB on
/// xc7_large -- so building a fresh one per rayon task per outer iteration was
/// the placer's largest remaining source of memset traffic (init_count=70 per
/// iteration on stereovision3). Only as many workspaces as there are
/// concurrently running tasks are ever live, so pooling turns ~70 allocations
/// per iteration into a handful for the whole run.
pub(crate) struct WorkspacePool {
    free: Mutex<Vec<PathSolverWorkspace>>,
    n_nodes: usize,
    n_pipes: usize,
}

impl WorkspacePool {
    pub(crate) fn new(n_nodes: usize, n_pipes: usize) -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            n_nodes,
            n_pipes,
        }
    }

    /// Take a workspace out of the pool, growing it if every one is in use.
    /// The guard puts it back when dropped.
    pub(crate) fn checkout(&self) -> WorkspaceGuard<'_> {
        let pooled = self
            .free
            .lock()
            .expect("workspace pool mutex poisoned")
            .pop();
        let ws = match pooled {
            Some(ws) => ws,
            None => PathSolverWorkspace::new(self.n_nodes, self.n_pipes),
        };
        WorkspaceGuard {
            pool: self,
            ws: Some(ws),
        }
    }
}

pub(crate) struct WorkspaceGuard<'a> {
    pool: &'a WorkspacePool,
    ws: Option<PathSolverWorkspace>,
}

impl std::ops::Deref for WorkspaceGuard<'_> {
    type Target = PathSolverWorkspace;
    fn deref(&self) -> &PathSolverWorkspace {
        self.ws.as_ref().expect("workspace taken")
    }
}

impl std::ops::DerefMut for WorkspaceGuard<'_> {
    fn deref_mut(&mut self) -> &mut PathSolverWorkspace {
        self.ws.as_mut().expect("workspace taken")
    }
}

impl Drop for WorkspaceGuard<'_> {
    fn drop(&mut self) {
        if let Some(ws) = self.ws.take() {
            self.pool
                .free
                .lock()
                .expect("workspace pool mutex poisoned")
                .push(ws);
        }
    }
}

impl PathSolverWorkspace {
    pub(crate) fn new(n_nodes: usize, n_pipes: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n_nodes],
            dist_int: vec![u32::MAX; n_nodes],
            path_weight: vec![0.0; n_nodes],
            node_load: vec![0.0; n_nodes],
            edge_load: vec![0.0; n_pipes],
            edge_touched: Vec::with_capacity(1024),
            settle_order: Vec::with_capacity(1024),
            touched_nodes: Vec::with_capacity(1024),
            in_corridor: vec![false; n_nodes],
            corridor_marked: Vec::with_capacity(1024),
            corridor_bbox: None,
            corridor_long_pipes: Vec::with_capacity(64),
            corridor_sinks: Vec::with_capacity(16),
            sink_nodes_scratch: Vec::with_capacity(16),
            pin_xy_scratch: Vec::with_capacity(16),
            buckets: Vec::new(),
            max_bucket: 0,
            heap: BinaryHeap::with_capacity(4096),
        }
    }

    /// Sparse reset: clears state written in the previous solve.
    /// `settle_order` is the authoritative list of updated nodes.
    pub(crate) fn begin_net(&mut self) {
        // `touched_nodes` covers the Dijkstra path; `settle_order` covers the
        // displacement fast path, which settles without relaxing. Resetting a
        // node twice is harmless.
        for &node in self.touched_nodes.iter().chain(self.settle_order.iter()) {
            self.dist[node] = f64::INFINITY;
            self.dist_int[node] = u32::MAX;
            self.path_weight[node] = 0.0;
            self.node_load[node] = 0.0;
        }
        for &pipe in &self.edge_touched {
            self.edge_load[pipe] = 0.0;
        }
        for &node in &self.corridor_marked {
            self.in_corridor[node] = false;
        }
        // Buckets are drained by the dial loop; nothing to clear unless we
        // broke early. Trim back to something reasonable to cap memory.
        let keep = (self.max_bucket as usize)
            .saturating_add(1)
            .min(self.buckets.len());
        for bucket in &mut self.buckets[..keep] {
            bucket.clear();
        }
        self.max_bucket = 0;
        self.settle_order.clear();
        self.touched_nodes.clear();
        self.edge_touched.clear();
        self.corridor_marked.clear();
        self.corridor_bbox = None;
        self.corridor_long_pipes.clear();
        self.corridor_sinks.clear();
        self.heap.clear();
    }

    /// Populate `in_corridor` for every node in the bbox that passes the
    /// Steiner check. Called once per net; hot loops read the bitmap.
    fn mark_corridor(&mut self, network: &PipeNetwork, corridor: &Corridor) {
        let sinks = &self.corridor_sinks[..];
        let width = network.width;
        let height = network.height;
        let bitmap_len = self.in_corridor.len();
        // Stash the bbox so FSM can scope its raster sweeps; the Steiner
        // inclusion still gates per-tile membership via `in_corridor`.
        self.corridor_bbox = Some((
            corridor.min_x.max(0),
            corridor.min_y.max(0),
            corridor.max_x.min(width - 1),
            corridor.max_y.min(height - 1),
        ));
        for ty in corridor.min_y..=corridor.max_y {
            if ty < 0 || ty >= height {
                continue;
            }
            for tx in corridor.min_x..=corridor.max_x {
                if tx < 0 || tx >= width {
                    continue;
                }
                if corridor.contains_xy(tx, ty, sinks) {
                    let idx = (ty * width + tx) as usize;
                    if idx < bitmap_len {
                        self.in_corridor[idx] = true;
                        self.corridor_marked.push(idx);
                    }
                }
            }
        }
        // Materialise the corridor-local long-pipe subset so FSM's
        // `long_pipe_relax` step iterates O(|in-corridor-longs|) instead of
        // scanning every long pipe on the chip each round.
        //
        // `corridor_long_pipes` has exactly one reader, `fsm::long_pipe_relax`,
        // so building it when the FSM solver is off is pure waste: the scan is
        // O(all long pipes on the chip) and runs once per net per outer
        // iteration, independent of design size.
        if use_fsm() {
            for &pipe_idx in &network.tile_grid.long_pipes {
                let pipe = &network.pipes[pipe_idx as usize];
                if self.in_corridor[pipe.from] && self.in_corridor[pipe.to] {
                    self.corridor_long_pipes.push(pipe_idx);
                }
            }
        }
    }

    /// Accumulate this net's flow on `pipe_idx`. Usage lands only in the
    /// workspace's own sparse pair (`edge_load` + `edge_touched`); the caller
    /// drains it once the net is solved. Writing straight into a chip-sized
    /// dense array here is what forced every rayon task to carry its own
    /// `vec![0.0; n_pipes]`.
    fn record_edge_usage(&mut self, pipe_idx: usize, flow: f64) {
        if self.edge_load[pipe_idx] == 0.0 {
            self.edge_touched.push(pipe_idx);
        }
        self.edge_load[pipe_idx] += flow;
    }

    /// Drain the current net's edge usage as `(pipe, flow)` pairs.
    ///
    /// Only the pipes this net actually touched are emitted, so the cost is
    /// proportional to the net's corridor rather than to the chip. Abandoned
    /// attempts need no rollback: nothing has been published yet, and
    /// `begin_net` clears `edge_load` / `edge_touched` before the retry.
    pub(crate) fn drain_edge_usage(&self, out: &mut Vec<(u32, f64)>) {
        out.reserve(self.edge_touched.len());
        for &pipe in &self.edge_touched {
            out.push((pipe as u32, self.edge_load[pipe]));
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

pub(crate) struct DialLogitResult {
    pub heap_pops: usize,
    pub energy: f64,
    pub missing_demand: f64,
    /// Retained for caller compatibility. Always false now that the
    /// unbounded fallback is gone.
    pub corridor_fallback: bool,
    /// Diagnostic: true when the sink-bounded distance cap actually armed
    /// during Dijkstra (i.e. every unique sink was settled). False when
    /// Dijkstra exhausted the corridor or heap before all sinks settled —
    /// those are the chip-spanning / disconnected nets that pay the
    /// `settle_order ≈ corridor_size` cost. Lets the solver fold count
    /// how often the bound saves work.
    pub cap_armed: bool,
}

#[derive(Debug)]
struct Corridor {
    source_x: i32,
    source_y: i32,
    halo: i32,
    stretch: f64,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CorridorSink {
    x: i32,
    y: i32,
    direct: i32,
}

/// Build a corridor from a source + sink list, writing sinks into the
/// caller-owned scratch vec (pooled on `PathSolverWorkspace` in hot paths).
fn build_corridor(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    sinks_out: &mut Vec<CorridorSink>,
) -> Option<Corridor> {
    let source = network.nodes.get(source_node)?;
    let mut min_x = source.tile_x;
    let mut max_x = source.tile_x;
    let mut min_y = source.tile_y;
    let mut max_y = source.tile_y;
    let mut max_direct = 0i32;

    sinks_out.clear();
    for &(node, demand) in sink_demands {
        if demand == 0.0 {
            continue;
        }
        let sink = network.nodes.get(node)?;
        min_x = min_x.min(sink.tile_x);
        max_x = max_x.max(sink.tile_x);
        min_y = min_y.min(sink.tile_y);
        max_y = max_y.max(sink.tile_y);
        let direct = manhattan_xy(source.tile_x, source.tile_y, sink.tile_x, sink.tile_y).max(1);
        max_direct = max_direct.max(direct);
        sinks_out.push(CorridorSink {
            x: sink.tile_x,
            y: sink.tile_y,
            direct,
        });
    }

    if sinks_out.is_empty() {
        return None;
    }

    let params = corridor_params();
    let rel_halo = (max_direct * params.rel_num + params.rel_den - 1) / params.rel_den;
    let halo = params.min_halo.max(rel_halo).min(params.max_halo);
    Some(Corridor {
        source_x: source.tile_x,
        source_y: source.tile_y,
        halo,
        stretch: params.stretch,
        min_x: (min_x - halo).max(0),
        max_x: (max_x + halo).min(network.width - 1),
        min_y: (min_y - halo).max(0),
        max_y: (max_y + halo).min(network.height - 1),
    })
}

impl Corridor {
    #[inline]
    fn contains_xy(&self, tx: i32, ty: i32, sinks: &[CorridorSink]) -> bool {
        if tx < self.min_x || tx > self.max_x || ty < self.min_y || ty > self.max_y {
            return false;
        }
        let via_src = manhattan_xy(self.source_x, self.source_y, tx, ty);
        let bound = self.halo as f64;
        for sink in sinks {
            let via = via_src + manhattan_xy(tx, ty, sink.x, sink.y);
            if (via as f64) <= self.stretch * (sink.direct as f64) + bound {
                return true;
            }
        }
        false
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

    let batch_size = auto_batch_size(cfg.net_parallel_batch_size, net_infos.len(), solve_pool);
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
    let result = dial_logit_load(network, source_node, &sink_demands, ws, true);
    for &pipe in &ws.edge_touched {
        local.edge_usage[pipe] += ws.edge_load[pipe];
    }

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
    demand::net_timing_weight(info, cfg)
}

#[inline(never)]
/// Fused binary-heap Dijkstra **and** Dial forward pass.
///
/// Was a bucket-Dial Dijkstra (`buckets[d]` indexed by absolute integer
/// distance), but BPR scaling under congestion blew `cost_int` past 10⁷ per
/// edge in iter 1+, sending `buckets.resize(idx+1, ...)` into multi-GB
/// allocations. The binary heap is `O((V+E)·log V)` instead of
/// `O(V+E+D_max)` but its memory footprint is proportional to live entries
/// only, independent of the absolute distance values.
///
/// When a node is popped it is settled at `dist_int[node] == cur`. In the
/// same edge iteration we already need for relaxation, we also compute
/// `path_weight[node]` by pulling contributions from every upstream-settled
/// predecessor (edges whose neighbour has `dist_int < cur`). Because the
/// graph is undirected, `node_pipes[node]` exposes both incoming (upstream)
/// and outgoing (not-yet-settled) edges in the same adjacency list — each
/// edge falls into exactly one branch of the dispatch below, so this fuses
/// Dijkstra + Dial forward into a single pass over the edge set.
fn dijkstra_and_forward(
    network: &PipeNetwork,
    source_node: usize,
    sink_nodes: &[usize],
    ws: &mut PathSolverWorkspace,
    use_corridor: bool,
) -> (usize, bool) {
    use super::network::DIST_SCALE;
    let inv_scale = 1.0 / DIST_SCALE;

    let dist_int = &mut ws.dist_int;
    let settle_order = &mut ws.settle_order;
    let path_weight = &mut ws.path_weight;
    let in_corridor = &ws.in_corridor;
    let adj = &network.flat_adjacency;
    let adj_neighbors = adj.neighbors.as_slice();
    let adj_cost_int = adj.cost_int.as_slice();
    let adj_offsets = adj.offsets.as_slice();
    let heap = &mut ws.heap;
    let touched = &mut ws.touched_nodes;
    heap.clear();

    if dist_int[source_node] == u32::MAX {
        touched.push(source_node);
    }
    dist_int[source_node] = 0;
    path_weight[source_node] = 1.0;
    heap.push(HeapEntry {
        dist: 0.0,
        node: source_node,
    });

    // Sink-bounded termination: stop popping once every sink has been
    // settled and the slack budget around the farthest sink is exhausted.
    // `max_dist_cap == u32::MAX` until that flips. Using `dist_int[sink]`
    // as the "settled?" check (set to label, was u32::MAX) avoids a parallel
    // bookkeeping vector at the cost of an O(n_sinks) scan when n_sinks is
    // tiny in practice (median = 2).
    let n_sinks = sink_nodes.len();
    let mut sinks_settled = 0usize;
    let mut max_sink_dist: u32 = 0;
    let cache_slack = cache_slack_int();
    let mut max_dist_cap: u32 = u32::MAX;

    let mut pops = 0usize;

    while let Some(HeapEntry { dist: cur_f, node }) = heap.pop() {
        // `dist` round-trips an integer through f64; `as u32` recovers the
        // exact bucket value because `cost_int < 2^31` keeps the running
        // sum well within f64's integer-exact range.
        let cur = cur_f as u32;
        if cur > max_dist_cap {
            // All sinks settled and we've exhausted the slack budget around
            // the farthest one. Anything beyond this label is uninteresting
            // for both DCD distance lookups and the Dial flow back-pass
            // (which only walks settled predecessors of sinks).
            break;
        }
        // Stale entry: a later push shortened this node's dist.
        if dist_int[node] < cur {
            continue;
        }
        pops += 1;
        settle_order.push(node);

        // Track sink settlement. Cheap because sink_nodes is small (median
        // 2, p99 ~30 on FPGA01); check fires once per node, exits early.
        if sinks_settled < n_sinks {
            for &sink in sink_nodes {
                if sink == node {
                    sinks_settled += 1;
                    if cur > max_sink_dist {
                        max_sink_dist = cur;
                    }
                    break;
                }
            }
            if sinks_settled == n_sinks && max_dist_cap == u32::MAX {
                max_dist_cap = max_sink_dist.saturating_add(cache_slack);
            }
        }

        // Start from whatever `path_weight[node]` already holds.
        // - Source was pre-set to 1.0 above.
        // - Every other settled node was initialised to 0.0 by `begin_net`;
        //   it accumulates from its upstream-settled predecessors below.
        let mut acc_weight = path_weight[node];

        let edge_start = adj_offsets[node] as usize;
        let edge_end = adj_offsets[node + 1] as usize;
        for e in edge_start..edge_end {
            let neighbour = adj_neighbors[e] as usize;
            if use_corridor && !in_corridor[neighbour] {
                continue;
            }
            let cost_int = adj_cost_int[e];

            let neigh_dist_int = dist_int[neighbour];
            if neigh_dist_int < cur {
                // Upstream settled predecessor: forward-pass contribution.
                let neigh_weight = path_weight[neighbour];
                if neigh_weight > 0.0 && neigh_weight.is_finite() {
                    let likelihood = link_likelihood_from_ints(neigh_dist_int, cur, cost_int);
                    if likelihood > 0.0 {
                        acc_weight += neigh_weight * likelihood;
                    }
                }
            } else {
                // Not yet settled (or equal dist tie): standard Dijkstra relax.
                let candidate = cur.saturating_add(cost_int);
                if candidate < neigh_dist_int {
                    if neigh_dist_int == u32::MAX {
                        // First write for this net: remember it so `begin_net`
                        // resets it even if we break before it settles.
                        touched.push(neighbour);
                    }
                    dist_int[neighbour] = candidate;
                    heap.push(HeapEntry {
                        dist: candidate as f64,
                        node: neighbour,
                    });
                }
            }
        }

        path_weight[node] = acc_weight;
    }

    ws.max_bucket = 0;

    // Materialise f64 labels for settled nodes; unvisited ones keep INF.
    for &node in settle_order.iter() {
        ws.dist[node] = dist_int[node] as f64 * inv_scale;
    }

    let cap_armed = max_dist_cap != u32::MAX;
    (pops, cap_armed)
}

/// Likelihood weighting `exp(-LOGIT_THETA · slack / DIST_SCALE)` where
/// `slack = dist_from_int + cost_int - dist_to_int`.
/// Because Dijkstra-optimality guarantees `dist_to_int <= dist_from_int +
/// cost_int`, the slack is a non-negative u32 and the exponent is <= 0.
///
/// Served from `LIKELIHOOD_LUT` (L1-resident 32 KB table). Slack values beyond
/// the table truncate to zero, matching the underflow behaviour of the scalar
/// exp within the callers' `> 0.0` guards.
#[inline]
pub(crate) fn link_likelihood_from_ints(
    dist_from_int: u32,
    dist_to_int: u32,
    cost_int: u32,
) -> f64 {
    let slack = dist_from_int
        .saturating_add(cost_int)
        .saturating_sub(dist_to_int);
    let idx = slack as usize;
    if idx < LIKELIHOOD_LUT_SIZE {
        LIKELIHOOD_LUT[idx] as f64
    } else {
        0.0
    }
}

/// Test-only reference Dijkstra: runs a full single-source shortest-path
/// computation over the network and returns `dist_int`. Used by `fsm.rs` tests
/// to validate equivalence with the hot path.
#[cfg(test)]
pub(crate) fn dijkstra_single_source_for_test(network: &PipeNetwork, source: usize) -> Vec<u32> {
    let n = network.num_nodes();
    let mut dist = vec![u32::MAX; n];
    dist[source] = 0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node: source,
    });
    while let Some(HeapEntry { dist: d_f, node }) = heap.pop() {
        let cur = dist[node];
        let cur_f = cur as f64;
        if d_f > cur_f + 0.5 {
            continue;
        }
        for &pipe_idx in &network.node_pipes[node] {
            let pipe = &network.pipes[pipe_idx];
            let other = pipe.from ^ pipe.to ^ node;
            let cost = network.pipe_cost_int(pipe_idx);
            let cand = cur.saturating_add(cost);
            if cand < dist[other] {
                dist[other] = cand;
                heap.push(HeapEntry {
                    dist: cand as f64,
                    node: other,
                });
            }
        }
    }
    dist
}

pub(crate) fn dial_logit_load(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    collect_usage: bool,
) -> DialLogitResult {
    // Bounded-corridor solve only. The historical "fall back to full-graph
    // Dijkstra when the corridor fails" branch was removed: per CLAUDE.md
    // ("Never add any fallback for any implementation"), and on FPGA01 it was
    // pinning DistCache memory at >20 GB by letting 5 % of nets settle the
    // entire 69 k-node graph. When the corridor cannot reach a sink, we
    // surface that as `missing_demand`/`failures` so the caller (and
    // pre_solve_diag) sees the gap; affected nodes simply have no entry in
    // `dist_cache` and downstream cost lookups return INFINITY.
    let Some(corridor) = build_corridor(network, source_node, sink_demands, &mut ws.corridor_sinks)
    else {
        return DialLogitResult {
            heap_pops: 0,
            energy: 0.0,
            missing_demand: sink_demands.iter().map(|(_, d)| d.abs()).sum(),
            corridor_fallback: false,
            cap_armed: false,
        };
    };
    ws.mark_corridor(network, &corridor);
    dial_logit_load_inner(network, source_node, sink_demands, ws, collect_usage, true)
}

fn dial_logit_load_inner(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    collect_usage: bool,
    use_corridor: bool,
) -> DialLogitResult {
    // Dijkstra + Dial forward pass are fused: `path_weight` and `dist` are
    // already populated for every settled node when this returns. Under the
    // `NPNR_OT_USE_FSM=1` gate, substitute the discrete Fast Sweeping solver;
    // it produces identical `dist_int` labels and matching `path_weight` from
    // a post-sweep forward pass.
    //
    // Sink-bounded termination needs the sink node list separated from the
    // demand values. Reuse the workspace buffer (`sink_nodes_scratch`) so the
    // hot per-net path doesn't allocate. Dedup is mandatory: high-fanout nets
    // have multiple pins on the same coarse-cell node, and the cap-fire
    // condition `sinks_settled == n_sinks` is satisfied per UNIQUE node — if
    // we leave duplicates in, the counter under-reports and the cap never
    // arms, leaving Dijkstra unbounded.
    ws.sink_nodes_scratch.clear();
    ws.sink_nodes_scratch.reserve(sink_demands.len());
    for &(node, _) in sink_demands {
        ws.sink_nodes_scratch.push(node);
    }
    ws.sink_nodes_scratch.sort_unstable();
    ws.sink_nodes_scratch.dedup();
    let (heap_pops, cap_armed) = if use_fsm() {
        // FSM path is unbounded today; sink list stays untyped here so we
        // don't churn that signature until FSM is back on the hot path.
        let pops = super::fsm::fsm_dist_and_forward(network, source_node, ws, use_corridor);
        (pops, false)
    } else {
        // Borrow split: sink_nodes_scratch lives on `ws`. Take a raw pointer
        // to the slice so the rest of `ws` stays mutable for the inner loop.
        let sink_ptr = ws.sink_nodes_scratch.as_ptr();
        let sink_len = ws.sink_nodes_scratch.len();
        let sinks: &[usize] = unsafe { std::slice::from_raw_parts(sink_ptr, sink_len) };
        dijkstra_and_forward(network, source_node, sinks, ws, use_corridor)
    };

    let adj = &network.flat_adjacency;
    let adj_neighbors = adj.neighbors.as_slice();
    let adj_pipe_idx = adj.pipe_idx.as_slice();
    let adj_cost_int = adj.cost_int.as_slice();
    let adj_offsets = adj.offsets.as_slice();
    let n_settled = ws.settle_order.len();

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
    // Operates in integer-slack domain so the likelihood is a single LUT hit
    // per edge — no libm exp on this critical path either.
    for i in (0..n_settled).rev() {
        let node = ws.settle_order[i];
        let load = ws.node_load[node];
        let node_weight = ws.path_weight[node];
        if load == 0.0 || node_weight <= 0.0 || !node_weight.is_finite() {
            continue;
        }
        let node_dist = ws.dist_int[node];
        if node_dist == u32::MAX {
            continue;
        }
        let edge_start = adj_offsets[node] as usize;
        let edge_end = adj_offsets[node + 1] as usize;
        for e in edge_start..edge_end {
            let pred = adj_neighbors[e] as usize;
            if use_corridor && !ws.in_corridor[pred] {
                continue;
            }
            let pred_dist = ws.dist_int[pred];
            // Strictly-upstream predecessor only. Unsettled preds carry
            // `u32::MAX` and fail the `<` test naturally.
            if pred_dist >= node_dist {
                continue;
            }
            let cost_int = adj_cost_int[e];
            let likelihood = link_likelihood_from_ints(pred_dist, node_dist, cost_int);
            if likelihood <= 0.0 {
                continue;
            }
            let pred_weight = ws.path_weight[pred];
            let share = pred_weight * likelihood / node_weight;
            if share <= 0.0 || !share.is_finite() {
                continue;
            }
            let flow = load * share;
            ws.node_load[pred] += flow;
            if collect_usage {
                let pipe_idx = adj_pipe_idx[e] as usize;
                ws.record_edge_usage(pipe_idx, flow);
            }
        }
    }

    DialLogitResult {
        heap_pops,
        energy,
        missing_demand,
        corridor_fallback: false,
        cap_armed,
    }
}

// ---------------------------------------------------------------------------
// Displacement-table fast path (DisplacementPure / DisplacementSparse)
// ---------------------------------------------------------------------------

/// Dispatch wrapper used by `coord_descent` when a `DisplacementTable` is
/// available. Uniform (source, sinks) pairs skip Dijkstra entirely and pull
/// their distances from the precomputed table; all other pairs fall through
/// to the full `dial_logit_load`. `DisplacementSparse` adds a hot-pipe
/// pre-check so nets whose Manhattan corridor touches a congested pipe take
/// the slow path instead.
pub(crate) fn dial_logit_load_dispatch(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    collect_usage: bool,
    displacement: Option<&super::displacement::DisplacementTable>,
    graph_model: super::config::GraphModel,
) -> DialLogitResult {
    use super::config::GraphModel;
    let fast_enabled = matches!(
        graph_model,
        GraphModel::DisplacementPure | GraphModel::DisplacementSparse
    );
    if fast_enabled {
        if let Some(disp) = displacement {
            if disp.is_uniform_pair(network, source_node, sink_demands) {
                let sparse = graph_model == GraphModel::DisplacementSparse;
                if let Some(result) = dial_logit_displacement(
                    network,
                    source_node,
                    sink_demands,
                    ws,
                    collect_usage,
                    disp,
                    sparse,
                ) {
                    return result;
                }
                // The bailed attempt published nothing: its usage lives only
                // in the workspace, which `begin_net` clears before the retry.
                ws.begin_net();
            }
        }
    }
    dial_logit_load(network, source_node, sink_demands, ws, collect_usage)
}

/// Fast path: populate `ws.dist / dist_int / path_weight / settle_order` for
/// every node within the displacement radius of `source_node` from the
/// precomputed D_ref lookup, then compute per-sink energy and splat sink
/// demand onto a deterministic Manhattan L-path (east-first, then south).
///
/// Returns `None` if the fast path bails (non-uniform coarsen, hot-pipe on
/// a Manhattan segment when `sparse_correction` is set, etc.), at which
/// point the caller falls through to `dial_logit_load`.
fn dial_logit_displacement(
    network: &PipeNetwork,
    source_node: usize,
    sink_demands: &[(usize, f64)],
    ws: &mut PathSolverWorkspace,
    collect_usage: bool,
    disp: &super::displacement::DisplacementTable,
    sparse_correction: bool,
) -> Option<DialLogitResult> {
    let width = network.width;
    let height = network.height;
    let n_nodes = network.nodes.len();
    if (width as usize).checked_mul(height as usize) != Some(n_nodes) {
        return None;
    }
    let src_node = network.nodes.get(source_node)?;
    let src_x = src_node.tile_x;
    let src_y = src_node.tile_y;

    if sparse_correction
        && any_hot_pipe_on_l_paths(network, src_x, src_y, source_node, sink_demands)
    {
        return None;
    }

    let r = disp.radius;
    let inv_scale = 1.0 / DIST_SCALE;
    for dy in -r..=r {
        let ny = src_y + dy;
        if ny < 0 || ny >= height {
            continue;
        }
        for dx in -r..=r {
            let nx = src_x + dx;
            if nx < 0 || nx >= width {
                continue;
            }
            let Some(d_int) = disp.lookup(dx, dy) else {
                continue;
            };
            let node_idx = (ny * width + nx) as usize;
            if node_idx >= n_nodes {
                continue;
            }
            if ws.dist_int[node_idx] != u32::MAX {
                continue;
            }
            ws.dist_int[node_idx] = d_int;
            ws.dist[node_idx] = (d_int as f64) * inv_scale;
            ws.path_weight[node_idx] = 1.0;
            ws.settle_order.push(node_idx);
        }
    }

    let mut energy = 0.0;
    let mut missing_demand = 0.0;
    for &(node, demand) in sink_demands {
        if demand == 0.0 {
            continue;
        }
        let label = ws.dist[node];
        if !label.is_finite() {
            missing_demand += demand.abs();
            continue;
        }
        energy += demand * label;
    }

    if collect_usage {
        for &(sink_node, demand) in sink_demands {
            if demand == 0.0 {
                continue;
            }
            let Some(sink) = network.nodes.get(sink_node) else {
                continue;
            };
            let dx = sink.tile_x - src_x;
            let dy = sink.tile_y - src_y;
            splat_l_path(network, src_x, src_y, dx, dy, demand, ws);
        }
    }

    Some(DialLogitResult {
        heap_pops: 0,
        energy,
        missing_demand,
        corridor_fallback: false,
        cap_armed: true,
    })
}

/// Deterministic east-then-south L-path splat from (src_x, src_y) to
/// (src_x + dx, src_y + dy). Pipes are looked up via `tile_grid.pipe_e /
/// pipe_s` (O(1)) and `record_edge_usage` accumulates them on the workspace.
fn splat_l_path(
    network: &PipeNetwork,
    src_x: i32,
    src_y: i32,
    dx: i32,
    dy: i32,
    demand: f64,
    ws: &mut PathSolverWorkspace,
) {
    let width = network.width;
    let tile_grid = &network.tile_grid;
    let step_x = dx.signum();
    let step_y = dy.signum();

    let mut cx = src_x;
    for _ in 0..dx.abs() {
        let tile_x = if step_x > 0 { cx } else { cx - 1 };
        if tile_x >= 0 && tile_x < width - 1 {
            let slot = (src_y * width + tile_x) as usize;
            if slot < tile_grid.pipe_e.len() {
                let pipe_idx = tile_grid.pipe_e[slot];
                if pipe_idx != u32::MAX {
                    ws.record_edge_usage(pipe_idx as usize, demand);
                }
            }
        }
        cx += step_x;
    }
    let mut cy = src_y;
    let fx = src_x + dx;
    for _ in 0..dy.abs() {
        let tile_y = if step_y > 0 { cy } else { cy - 1 };
        if tile_y >= 0 && tile_y < network.height - 1 {
            let slot = (tile_y * width + fx) as usize;
            if slot < tile_grid.pipe_s.len() {
                let pipe_idx = tile_grid.pipe_s[slot];
                if pipe_idx != u32::MAX {
                    ws.record_edge_usage(pipe_idx as usize, demand);
                }
            }
        }
        cy += step_y;
    }
}

/// Walk the same L-path and return `true` if any pipe along it has
/// `pipe_costs_int > 1.2 × base_int`. Used by `DisplacementSparse` to bail
/// to full Dijkstra when the cheap table can't represent the cost.
fn any_hot_pipe_on_l_paths(
    network: &PipeNetwork,
    src_x: i32,
    src_y: i32,
    _source_node: usize,
    sink_demands: &[(usize, f64)],
) -> bool {
    let width = network.width;
    let tile_grid = &network.tile_grid;
    let pipes = &network.pipes;
    let cost_int = &network.pipe_costs_int;
    let hot = |pipe_idx: u32| -> bool {
        if pipe_idx == u32::MAX {
            return false;
        }
        let idx = pipe_idx as usize;
        let pipe = &pipes[idx];
        let base_int = (pipe.base_resistance * DIST_SCALE).max(1.0) as u32;
        let threshold = base_int.saturating_mul(6) / 5; // 1.2×
        cost_int.get(idx).copied().unwrap_or(0) > threshold
    };
    for &(sink_node, demand) in sink_demands {
        if demand == 0.0 {
            continue;
        }
        let Some(sink) = network.nodes.get(sink_node) else {
            continue;
        };
        let dx = sink.tile_x - src_x;
        let dy = sink.tile_y - src_y;
        let step_x = dx.signum();
        let step_y = dy.signum();
        let mut cx = src_x;
        for _ in 0..dx.abs() {
            let tx = if step_x > 0 { cx } else { cx - 1 };
            if tx >= 0 && tx < width - 1 {
                let slot = (src_y * width + tx) as usize;
                if slot < tile_grid.pipe_e.len() && hot(tile_grid.pipe_e[slot]) {
                    return true;
                }
            }
            cx += step_x;
        }
        let mut cy = src_y;
        let fx = src_x + dx;
        for _ in 0..dy.abs() {
            let ty = if step_y > 0 { cy } else { cy - 1 };
            if ty >= 0 && ty < network.height - 1 {
                let slot = (ty * width + fx) as usize;
                if slot < tile_grid.pipe_s.len() && hot(tile_grid.pipe_s[slot]) {
                    return true;
                }
            }
            cy += step_y;
        }
    }
    false
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
        let nodes: Vec<Node> = (0..n_nodes)
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
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * crate::placer::opt_trans::network::DIST_SCALE).round() as u32).max(1))
            .collect();
        let tile_grid =
            crate::placer::opt_trans::network::TileGrid::build(&pipes, &nodes, n_nodes as i32, 1);
        let flat_adjacency =
            crate::placer::opt_trans::network::FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = std::sync::Arc::new(Vec::new());
        let n_pipe_entries = pipes.len();
        let n_node_entries = nodes.len();
        let mut net = PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table: crate::placer::opt_trans::tile_cache::SpanCostTable::disabled(
                n_pipe_entries,
            ),
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n_node_entries],
            width: n_nodes as i32,
            height: 1,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: n_nodes,
            coarsen: 1,
        };
        net.refresh_flat_edge_costs();
        net
    }

    fn test_network_with_coords(
        coords: &[(i32, i32)],
        edges: &[(usize, usize, f64)],
    ) -> PipeNetwork {
        let nodes: Vec<Node> = coords
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
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * crate::placer::opt_trans::network::DIST_SCALE).round() as u32).max(1))
            .collect();
        let tile_grid =
            crate::placer::opt_trans::network::TileGrid::build(&pipes, &nodes, width, height);
        let flat_adjacency =
            crate::placer::opt_trans::network::FlatAdjacency::build(&node_pipes, &pipes);
        let tile_templates = std::sync::Arc::new(Vec::new());
        let n_pipe_entries = pipes.len();
        let n_node_entries = nodes.len();
        let mut net = PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_costs,
            pipe_history: vec![0.0; pipe_costs_int.len()],
            pipe_costs_int,
            span_cost_table: crate::placer::opt_trans::tile_cache::SpanCostTable::disabled(
                n_pipe_entries,
            ),
            flat_adjacency,
            tile_templates,
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n_node_entries],
            width,
            height,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: coords.len(),
            coarsen: 1,
        };
        net.refresh_flat_edge_costs();
        net
    }

    /// Materialise the workspace's sparse edge usage as a dense per-pipe
    /// array so these tests can keep asserting on pipe indices.
    fn dense_usage(ws: &PathSolverWorkspace, n_pipes: usize) -> Vec<f64> {
        let mut pairs = Vec::new();
        ws.drain_edge_usage(&mut pairs);
        let mut usage = vec![0.0f64; n_pipes];
        for (pipe, flow) in pairs {
            usage[pipe as usize] += flow;
        }
        usage
    }

    #[test]
    fn dial_single_path_loads_all_demand() {
        let network = test_network(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let mut ws = PathSolverWorkspace::new(network.num_nodes(), network.num_pipes());
        ws.begin_net();
        let result = dial_logit_load(&network, 0, &[(2, 1.0)], &mut ws, true);
        let usage = dense_usage(&ws, network.num_pipes());

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
        let result = dial_logit_load(&network, 0, &[(3, 1.0)], &mut ws, true);
        let usage = dense_usage(&ws, network.num_pipes());

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
        let result = dial_logit_load(&network, 0, &[(3, 1.0)], &mut ws, true);
        let usage = dense_usage(&ws, network.num_pipes());

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
        let result = dial_logit_load(&network, 0, &[(2, 1.5)], &mut ws, true);
        let usage = dense_usage(&ws, network.num_pipes());

        assert_eq!(result.missing_demand, 1.5);
        assert_eq!(usage[0], 0.0);
    }

    // The historical `corridor_fallback_does_not_double_count_usage` test
    // exercised the rollback path that fired when the corridor missed sinks
    // and the solver re-ran on the full graph. That fallback was removed
    // (see CLAUDE.md "no fallback" rule and the FPGA01 memory work in
    // 2026-05-06 that capped DistCache). The "unreachable sink → missing
    // demand" behaviour is already covered by
    // `dial_reports_unreachable_sink_demand`.
}

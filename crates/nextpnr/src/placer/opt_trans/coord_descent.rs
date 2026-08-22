//! Discrete coordinate descent placer over all movable cell coordinates.
//!
//! The DCD loop treats placement as a `2 * n_cells` dimensional discrete
//! optimization problem. Within a sweep the resistance field is frozen, trial
//! moves are evaluated with cached Dial-logit soft costs, and only the winning
//! moves update the cached distance fields. Pipe usage and effective
//! conductance are refreshed between sweeps.

use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Arc;

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
use super::path_solver::{self, PathSolverWorkspace, PathStats, WorkspacePool};
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
    /// Shared rather than owned: the name breaks ties in the deterministic net
    /// sort, and re-resolving and re-allocating it for 105 000 nets twice an
    /// iteration cost more than the sort it feeds.
    debug_name: Arc<str>,
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
    /// Per-node Lagrangian multiplier for the per-tile capacity constraint.
    /// Updated between outer iters by `MuxSlotTracker::drain_rejection_pressure`
    /// — each rejected commit pushes pressure on the destination tile up by
    /// `cfg.tile_pressure_step`, decaying multiplicatively by
    /// `cfg.tile_pressure_decay` each iter. Read in `evaluate_cell_at` as an
    /// additive cost term scaled by `cfg.tile_pressure_weight`. Zero everywhere
    /// when the term is disabled (`tile_pressure_weight == 0.0`).
    tile_pressure: Vec<f64>,
    /// Network grid width, so a node index converts to tile coords without the
    /// network. Zero until `calibrate_analytic` runs.
    width: usize,
    /// Per-net cost per tile of Manhattan distance from the driver, fitted to
    /// the labels the last refresh produced. `NaN` where the net has no driver,
    /// no sinks, or no finite label. Read only under `NPNR_OT_ANALYTIC_DIST=1`.
    analytic_k: Vec<f32>,
    /// Per-net driver node, `u32::MAX` where the net has none.
    analytic_driver: Vec<u32>,
}

/// Serve `DistCache::get` from `k * manhattan(driver, node)` rather than the
/// stored labels, via `NPNR_OT_ANALYTIC_DIST=1`.
///
/// Measured on FPGA01 at `NPNR_OT_BPR_CAP=1.2`: `star/manh` sits at 0.325-0.343
/// across eight iterations and only 38-67 nets of 105 117 deviate more than
/// 25 % from the fit. Uncapped the same count is 57 837, so this is a knob that
/// only means anything under a bounded congestion price.
fn analytic_dist() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    let forced = *ON.get_or_init(|| match std::env::var("NPNR_OT_ANALYTIC_DIST") {
        Ok(v) => {
            let n: u32 = v
                .parse()
                .unwrap_or_else(|_| panic!("NPNR_OT_ANALYTIC_DIST must be 0 or 1, got {v:?}"));
            assert!(n <= 1, "NPNR_OT_ANALYTIC_DIST must be 0 or 1, got {n}");
            n == 1
        }
        Err(_) => false,
    });
    forced || ANALYTIC_PHASE.load(std::sync::atomic::Ordering::Relaxed)
}

/// True while the outer loop is inside the leading analytic phase, where no
/// Dijkstra runs at all.
static ANALYTIC_PHASE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Outer iterations to run on the analytic field with the per-net solve skipped
/// entirely, via `NPNR_OT_ANALYTIC_UNTIL`. Zero, the default, keeps every
/// iteration exact.
///
/// Measured at `NPNR_OT_BPR_CAP=1.2`: the analytic field's `line` leads the
/// exact one through outer=9 (-14.7 % at outer=0, -3.3 % at outer=9) and only
/// falls behind from outer=10, while outer=0 alone costs 118 s of refresh
/// against 55 s later. So the iterations it wins are also the dear ones.
fn analytic_until() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| match std::env::var("NPNR_OT_ANALYTIC_UNTIL") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("NPNR_OT_ANALYTIC_UNTIL must be an integer, got {v:?}")),
        Err(_) => 0,
    })
}

/// Manhattan radius beyond which the analytic field reports INFINITY, via
/// `NPNR_OT_ANALYTIC_RADIUS`. Unset means unbounded.
///
/// The stored labels only exist within `cache_radius_tiles` of a pin, so the
/// sweep used to prune distant candidates on an INFINITY it got for free. The
/// analytic field is finite everywhere, which is why it descends faster and
/// also why the sweep grew from 131 s to 504 s. This bounds the search without
/// going back to the radius-12 field that pruning implied.
fn analytic_radius() -> i32 {
    use std::sync::OnceLock;
    static R: OnceLock<i32> = OnceLock::new();
    *R.get_or_init(|| match std::env::var("NPNR_OT_ANALYTIC_RADIUS") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("NPNR_OT_ANALYTIC_RADIUS must be an integer, got {v:?}")),
        Err(_) => i32::MAX,
    })
}

/// Cost per tile used during the analytic phase, before any solve has produced
/// labels to fit. Default 0.33, the measured `star/manh` at cap 1.2.
fn analytic_k_seed() -> f32 {
    use std::sync::OnceLock;
    static K: OnceLock<f32> = OnceLock::new();
    *K.get_or_init(|| match std::env::var("NPNR_OT_ANALYTIC_K") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("NPNR_OT_ANALYTIC_K must be a float, got {v:?}")),
        Err(_) => 0.33,
    })
}

/// Enter or leave the leading analytic phase for `outer`. Returns true while
/// the phase is active, which is the caller's signal to skip the solve.
fn set_analytic_phase(outer: usize) -> bool {
    let active = outer < analytic_until();
    ANALYTIC_PHASE.store(active, std::sync::atomic::Ordering::Relaxed);
    active
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
            tile_pressure: vec![0.0; n_nodes],
            width: 0,
            analytic_k: vec![f32::NAN; n_nets],
            analytic_driver: vec![u32::MAX; n_nets],
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
        if analytic_dist() {
            return self.analytic_get(net_idx, node);
        }
        match self.rows[net_idx].get(&(node as u32)) {
            Some(&v) => v as f64,
            None => f64::INFINITY,
        }
    }

    /// `k * manhattan(driver, node)`. Uncalibrated nets return INFINITY, which
    /// is what the stored row returns for an unsolved net, so callers need no
    /// new case.
    #[inline]
    fn analytic_get(&self, net_idx: usize, node: usize) -> f64 {
        let k = self.analytic_k[net_idx];
        let driver = self.analytic_driver[net_idx];
        if !k.is_finite() || driver == u32::MAX {
            return f64::INFINITY;
        }
        let w = self.width;
        let (dx, dy) = ((driver as usize % w) as i32, (driver as usize / w) as i32);
        let (nx, ny) = ((node % w) as i32, (node / w) as i32);
        let manh = (nx - dx).abs() + (ny - dy).abs();
        if manh > analytic_radius() {
            return f64::INFINITY;
        }
        k as f64 * manh as f64
    }

    /// Fit one cost-per-tile rate per net to the labels the refresh just wrote,
    /// summing over sinks so a multi-sink net is fitted to all of them rather
    /// than to whichever one is read first. Costs one hash lookup per sink,
    /// against a Dijkstra per net, so it is free at this scale.
    fn seed_analytic(&mut self, net_infos: &[NetInfo], width: usize, k: f32) {
        self.width = width;
        if self.analytic_k.len() != self.n_nets {
            self.analytic_k = vec![f32::NAN; self.n_nets];
            self.analytic_driver = vec![u32::MAX; self.n_nets];
        }
        for (net_idx, info) in net_infos.iter().enumerate() {
            match info.pins.iter().find(|p| p.is_driver).map(|p| p.node) {
                Some(driver) => {
                    self.analytic_driver[net_idx] = driver as u32;
                    self.analytic_k[net_idx] = k;
                }
                None => {
                    self.analytic_driver[net_idx] = u32::MAX;
                    self.analytic_k[net_idx] = f32::NAN;
                }
            }
        }
    }

    fn calibrate_analytic(&mut self, net_infos: &[NetInfo], width: usize) {
        self.width = width;
        if self.analytic_k.len() != self.n_nets {
            self.analytic_k = vec![f32::NAN; self.n_nets];
            self.analytic_driver = vec![u32::MAX; self.n_nets];
        }
        for (net_idx, info) in net_infos.iter().enumerate() {
            let Some(driver) = info.pins.iter().find(|p| p.is_driver).map(|p| p.node) else {
                self.analytic_k[net_idx] = f32::NAN;
                self.analytic_driver[net_idx] = u32::MAX;
                continue;
            };
            let (dx, dy) = ((driver % width) as i32, (driver / width) as i32);
            let mut sum_dist = 0.0f64;
            let mut sum_manh = 0.0f64;
            for pin in info.pins.iter().filter(|p| !p.is_driver) {
                let Some(&d) = self.rows[net_idx].get(&(pin.node as u32)) else {
                    continue;
                };
                if !d.is_finite() {
                    continue;
                }
                let (px, py) = ((pin.node % width) as i32, (pin.node / width) as i32);
                sum_manh += ((px - dx).abs() + (py - dy).abs()) as f64;
                sum_dist += d as f64;
            }
            self.analytic_driver[net_idx] = driver as u32;
            self.analytic_k[net_idx] = if sum_manh > 0.0 {
                (sum_dist / sum_manh) as f32
            } else {
                f32::NAN
            };
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
fn compute_mst_neighbors(dist_cache: &mut DistCache, net_infos: &[NetInfo], network: &PipeNetwork) {
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

fn cache_netmap() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NPNR_OT_CACHE_NETMAP").ok().as_deref() == Some("1"))
}

/// Whether the configured sweep walks cells in driver-to-sink order.
///
/// Only the sequential sweeps do. Building the order for the parallel ones is
/// a topological sort over 150 000 cells that nothing then reads.
fn want_topo_order(cfg: &OptTransPlacerCfg) -> bool {
    matches!(
        cfg.sweep_mode,
        super::config::SweepMode::SequentialBisection | super::config::SweepMode::JacobiBB
    )
}

#[derive(Clone)]
struct CellNetMap {
    /// For each cell, the list of (net_idx, pin_idx) pairs it participates in.
    map: Vec<Vec<(usize, usize)>>,
    /// Topological cell order following driver → sink edges, with back edges
    /// broken by choosing the cell with smallest remaining in-degree.
    /// Driver-to-sink order, present only for the sweep modes that walk cells
    /// sequentially. `JacobiBisection` and the other parallel sweeps never read
    /// it, and building it cost 17 % of the profile, so it is absent rather
    /// than empty for them: a consumer that needs it and finds `None` is a
    /// dispatch bug and says so.
    topo_order: Option<Vec<usize>>,
}

impl CellNetMap {
    /// Build once per placement rather than twice per outer iteration, via
    /// `NPNR_OT_CACHE_NETMAP=1`.
    ///
    /// The map is cell-to-(net, pin) incidence plus a driver-to-sink topological
    /// order. Both are fixed by the netlist, and the outer loop only moves cells,
    /// so rebuilding them each iteration re-allocates ~150 000 small vectors for
    /// an identical answer. Measured at 24 % of the profile.
    ///
    /// Cached on the assumption that pin `cell_idx` and `is_fixed` do not change
    /// inside the loop. Deterministic runs make that testable exactly: a cached
    /// run must reproduce the uncached HPWL bit for bit.
    ///
    /// Shared through an `Arc` rather than cloned: the map holds one vector per
    /// cell, so handing out copies reintroduced most of the allocation the
    /// cache exists to avoid.
    fn build_cached(
        net_infos: &[NetInfo],
        n_cells: usize,
        want_topo: bool,
        cache: &mut Option<Arc<Self>>,
    ) -> Arc<Self> {
        if !cache_netmap() {
            return Arc::new(Self::build(net_infos, n_cells, want_topo));
        }
        if cache.is_none() {
            *cache = Some(Arc::new(Self::build(net_infos, n_cells, want_topo)));
        }
        Arc::clone(cache.as_ref().expect("just filled"))
    }

    fn build(net_infos: &[NetInfo], n_cells: usize, want_topo: bool) -> Self {
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
        let topo_order = want_topo.then(|| Self::build_topo_order(net_infos, n_cells));
        Self { map, topo_order }
    }

    /// Kahn's-style topological sort over driver → sink edges from `net_infos`.
    /// For cycles (back edges), relaxation picks the cell with the smallest
    /// remaining in-degree, breaking the cycle arbitrarily but deterministically.
    fn build_topo_order(net_infos: &[NetInfo], n_cells: usize) -> Vec<usize> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

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

        // Lowest-index-first among the cells whose in-degree has reached zero,
        // which is what a scan from index 0 with an early break on degree zero
        // selected. Kept in a heap so the scan does not walk the placed prefix
        // again on every step; that walk was 16.4 % of the placement profile.
        // A cell can be pushed more than once, so entries are validated on pop.
        let mut ready: BinaryHeap<Reverse<usize>> = (0..n_cells)
            .filter(|&ci| deg[ci] == 0)
            .map(Reverse)
            .collect();
        let mut remaining = n_cells;

        while remaining > 0 {
            let ci = loop {
                match ready.pop() {
                    Some(Reverse(ci)) if !placed[ci] && deg[ci] == 0 => break ci,
                    Some(_) => continue,
                    // Every remaining cell sits on a cycle. Break it the way the
                    // scan did: lowest index among the smallest in-degrees.
                    None => {
                        let mut best: Option<usize> = None;
                        let mut best_deg = u32::MAX;
                        for ci in 0..n_cells {
                            if !placed[ci] && deg[ci] < best_deg {
                                best_deg = deg[ci];
                                best = Some(ci);
                            }
                        }
                        break best.expect("remaining > 0 implies an unplaced cell");
                    }
                }
            };

            order.push(ci);
            placed[ci] = true;
            remaining -= 1;

            for &next in &out_edges[ci] {
                if !placed[next] && deg[next] > 0 {
                    deg[next] -= 1;
                    if deg[next] == 0 {
                        ready.push(Reverse(next));
                    }
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
    /// Settled nodes above `max_sink_dist`: zero-load tail kept only for
    /// `dist_cache`. See `DialLogitResult::settle_above_sink`.
    diag_above_sum: u64,
    diag_load_nz_sum: u64,
    diag_heur_prune_sum: u64,
    diag_load_mat_sum: u64,
    /// Settle-count histogram over nets, bucketed by settled-node count:
    /// [<100, <1k, <5k, <20k, <60k, >=60k]. `_nets` counts nets in the
    /// bucket, `_settles` the settles they account for — the mean alone
    /// hides whether a few chip-spanning nets dominate the pass.
    diag_hist_nets: [u64; 6],
    diag_hist_settles: [u64; 6],
    /// Same work split, keyed on FANOUT instead of settle count:
    /// [1, 2-4, 5-16, 17-64, 65-256, >256] sinks. Distinguishes "a few
    /// high-fanout nets dominate" from "many long 2-pin nets dominate" —
    /// the two have completely different fixes.
    diag_fan_nets: [u64; 6],
    diag_fan_settles: [u64; 6],
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

/// Nets that survive the constant and clock name tests of [`OptTransPlacerCfg`].
///
/// The tests read net names, and a name does not change while cells move, so
/// running them inside the per-iteration collect meant interning and
/// lowercasing 105 220 strings twice an iteration. The type exists so a caller
/// cannot pass the unfiltered list by accident.
struct NameFilteredNets {
    nets: Vec<(NetId, Arc<str>)>,
}

impl NameFilteredNets {
    fn new(ctx: &Context, alive_net_ids: &[NetId], cfg: &OptTransPlacerCfg) -> Self {
        let skip_constants = cfg.skip_constants;
        let skip_clocks = cfg.skip_clocks || cfg.exclude_globals;
        let nets = alive_net_ids
            .iter()
            .copied()
            .filter_map(|net_id| {
                let net_name = ctx.name_of(ctx.design.net(net_id).name);
                if skip_constants
                    && (net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET")
                {
                    return None;
                }
                if skip_clocks {
                    let lower = net_name.to_ascii_lowercase();
                    if lower.contains("clk") || lower.contains("clock") {
                        return None;
                    }
                }
                Some((net_id, Arc::from(net_name)))
            })
            .collect();
        Self { nets }
    }
}

fn collect_net_infos_simple(
    ctx: &Context,
    net_ids: &NameFilteredNets,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
) -> Vec<NetInfo> {
    let mut nets: Vec<_> = net_ids
        .nets
        .par_iter()
        .filter_map(|(net_id, net_name)| {
            let net_id = *net_id;
            let net = ctx.design.net(net_id);
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
                    debug_name: Arc::clone(net_name),
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

fn solve_all_nets_with_displacement(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    ws_pool: &WorkspacePool,
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

    // Inside the leading analytic phase nothing reads the labels, so skipping
    // every net is the whole point: this is where the refresh time goes.
    let analytic_skip: Option<Vec<bool>> =
        if ANALYTIC_PHASE.load(std::sync::atomic::Ordering::Relaxed) {
            Some(vec![true; net_infos.len()])
        } else {
            None
        };
    let skip_mask = match analytic_skip.as_deref() {
        Some(m) => Some(m),
        None => skip_mask,
    };

    let n_pipes = network.num_pipes();
    let batch_size =
        path_solver::auto_batch_size(cfg.net_parallel_batch_size, net_infos.len(), solve_pool);

    assert_eq!(
        dist_cache.rows.len(),
        net_infos.len(),
        "DistCache must hold exactly one row per net being solved"
    );
    let dist_rows = &mut dist_cache.rows;

    // Each chunk reports only the pipes its own nets touched. Results come
    // back in chunk order and are summed serially, so the floating-point
    // association order no longer depends on how rayon happened to schedule
    // the work. The previous shape -- a pairwise `reduce` over per-task
    // `vec![0.0; n_pipes]` arrays -- was both O(chip) per merge and a source
    // of run-to-run nondeterminism.
    let init_count = AtomicU32::new(0);
    let chunk_outs: Vec<ChunkUsage> = solve_pool.install(|| {
        net_infos
            .par_chunks(batch_size)
            .zip(dist_rows.par_chunks_mut(batch_size))
            .enumerate()
            .map_init(
                || {
                    init_count.fetch_add(1, AtomicOrdering::Relaxed);
                    ws_pool.checkout()
                },
                |ws, (chunk_idx, (chunk, dist_chunk))| {
                    let mut out = ChunkUsage::default();
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
                            out.stats.failures += 1;
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
                            out.stats.failures += 1;
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
                        let result = path_solver::dial_logit_load_dispatch(
                            network,
                            source,
                            &sink_demands,
                            ws,
                            collect_usage,
                            displacement,
                            cfg.graph_model,
                        );

                        out.stats.total_solves += 1;
                        out.stats.total_heap_pops += result.heap_pops;
                        out.stats.max_heap_pops = out.stats.max_heap_pops.max(result.heap_pops);
                        if result.missing_demand > 0.0 {
                            out.stats.failures += 1;
                        }

                        let settle_len = ws.settle_order.len() as u64;
                        out.diag_solves += 1;
                        out.diag_settle_sum += settle_len;
                        out.diag_above_sum += result.settle_above_sink as u64;
                        out.diag_load_nz_sum += result.load_nonzero as u64;
                        out.diag_heur_prune_sum += result.heuristic_prunable as u64;
                        out.diag_load_mat_sum += result.load_material as u64;
                        let bucket = match settle_len {
                            0..=99 => 0,
                            100..=999 => 1,
                            1_000..=4_999 => 2,
                            5_000..=19_999 => 3,
                            20_000..=59_999 => 4,
                            _ => 5,
                        };
                        out.diag_hist_nets[bucket] += 1;
                        out.diag_hist_settles[bucket] += settle_len;
                        let fan_bucket = match n_sinks {
                            0..=1 => 0,
                            2..=4 => 1,
                            5..=16 => 2,
                            17..=64 => 3,
                            65..=256 => 4,
                            _ => 5,
                        };
                        out.diag_fan_nets[fan_bucket] += 1;
                        out.diag_fan_settles[fan_bucket] += settle_len;
                        if settle_len as u32 > out.diag_settle_max {
                            out.diag_settle_max = settle_len as u32;
                        }
                        if result.corridor_fallback {
                            out.diag_corridor_fallback += 1;
                        }
                        if result.cap_armed {
                            out.diag_cap_armed += 1;
                        }
                        if collect_usage {
                            ws.accumulate_edge_usage();
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
                            ws.pin_xy_scratch
                                .push(((pin.node % width_n) as i32, (pin.node / width_n) as i32));
                        }

                        let dist_out = &mut dist_chunk[local_idx];
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
                        out.energy += net_path_weight(info, cfg) * result.energy;
                    }
                    out
                },
            )
            .collect()
    });

    let mut accum = merge_chunk_usage(
        n_pipes,
        chunk_outs,
        init_count.load(AtomicOrdering::Relaxed),
    );
    // Every guard has been dropped by now (the parallel iterator finished and
    // `chunk_outs` was collected), so all workspaces are back in the pool and
    // their running totals are complete.
    if collect_usage {
        ws_pool.drain_usage_into(&mut accum.edge_usage);
        {
            let (n_ws, bytes) = ws_pool.pool_stats();
            eprintln!(
                "MEM_POOL: workspaces={} dense_usage_mb={:.0} n_pipes={}",
                n_ws,
                bytes as f64 / (1024.0 * 1024.0),
                n_pipes,
            );
        }
    }
    accum
}

/// One chunk's contribution to a solve pass.
///
/// Edge usage is NOT here: it accumulates directly into the solving
/// workspace's dense `usage_accum` and is collected from the pool afterwards,
/// which keeps peak memory proportional to concurrency instead of to the total
/// number of pipe-touches across all nets.
#[derive(Default)]
struct ChunkUsage {
    energy: f64,
    stats: PathStats,
    diag_corridor_fallback: u64,
    diag_cap_armed: u64,
    diag_solves: u64,
    diag_settle_sum: u64,
    diag_settle_max: u32,
    /// Settled nodes above `max_sink_dist`: zero-load tail kept only for
    /// `dist_cache`. See `DialLogitResult::settle_above_sink`.
    diag_above_sum: u64,
    diag_load_nz_sum: u64,
    diag_heur_prune_sum: u64,
    diag_load_mat_sum: u64,
    /// Settle-count histogram over nets, bucketed by settled-node count:
    /// [<100, <1k, <5k, <20k, <60k, >=60k]. `_nets` counts nets in the
    /// bucket, `_settles` the settles they account for — the mean alone
    /// hides whether a few chip-spanning nets dominate the pass.
    diag_hist_nets: [u64; 6],
    diag_hist_settles: [u64; 6],
    /// Same work split, keyed on FANOUT instead of settle count:
    /// [1, 2-4, 5-16, 17-64, 65-256, >256] sinks. Distinguishes "a few
    /// high-fanout nets dominate" from "many long 2-pin nets dominate" —
    /// the two have completely different fixes.
    diag_fan_nets: [u64; 6],
    diag_fan_settles: [u64; 6],
}

/// Sum per-chunk scalar contributions in chunk order. Usage is merged
/// separately, from the workspace pool.
fn merge_chunk_usage(n_pipes: usize, chunks: Vec<ChunkUsage>, init_count: u32) -> SolveAccum {
    let mut accum = SolveAccum {
        edge_usage: vec![0.0; n_pipes],
        energy: 0.0,
        stats: PathStats::default(),
        diag_corridor_fallback: 0,
        diag_cap_armed: 0,
        diag_solves: 0,
        diag_settle_sum: 0,
        diag_settle_max: 0,
        diag_above_sum: 0,
        diag_load_nz_sum: 0,
        diag_load_mat_sum: 0,
        diag_heur_prune_sum: 0,
        diag_hist_nets: [0; 6],
        diag_hist_settles: [0; 6],
        diag_fan_nets: [0; 6],
        diag_fan_settles: [0; 6],
        diag_init_count: init_count,
    };
    for chunk in chunks {
        accum.energy += chunk.energy;
        accum.stats.total_solves += chunk.stats.total_solves;
        accum.stats.total_heap_pops += chunk.stats.total_heap_pops;
        accum.stats.max_heap_pops = accum.stats.max_heap_pops.max(chunk.stats.max_heap_pops);
        accum.stats.failures += chunk.stats.failures;
        accum.diag_corridor_fallback += chunk.diag_corridor_fallback;
        accum.diag_cap_armed += chunk.diag_cap_armed;
        accum.diag_solves += chunk.diag_solves;
        accum.diag_settle_sum += chunk.diag_settle_sum;
        accum.diag_above_sum += chunk.diag_above_sum;
        accum.diag_load_nz_sum += chunk.diag_load_nz_sum;
        accum.diag_heur_prune_sum += chunk.diag_heur_prune_sum;
        accum.diag_load_mat_sum += chunk.diag_load_mat_sum;
        for b in 0..6 {
            accum.diag_hist_nets[b] += chunk.diag_hist_nets[b];
            accum.diag_hist_settles[b] += chunk.diag_hist_settles[b];
            accum.diag_fan_nets[b] += chunk.diag_fan_nets[b];
            accum.diag_fan_settles[b] += chunk.diag_fan_settles[b];
        }
        accum.diag_settle_max = accum.diag_settle_max.max(chunk.diag_settle_max);
    }
    accum
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
    let batch_size =
        path_solver::auto_batch_size(cfg.net_parallel_batch_size, net_infos.len(), solve_pool);
    assert_eq!(
        dist_cache.rows.len(),
        net_infos.len(),
        "DistCache must hold exactly one row per net being solved"
    );
    let dist_rows = &mut dist_cache.rows;

    solve_pool
        .install(|| {
            net_infos
                .par_chunks(batch_size)
                .zip(dist_rows.par_chunks_mut(batch_size))
                .fold(
                    || TemplateAccum {
                        edge_usage: vec![0.0; n_pipes],
                        energy: 0.0,
                        failures: 0,
                    },
                    |mut accum, (chunk, dist_chunk)| {
                        for (info, dist_out) in chunk.iter().zip(dist_chunk.iter_mut()) {
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
    let batch_size =
        path_solver::auto_batch_size(cfg.net_parallel_batch_size, net_infos.len(), solve_pool);

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
            diag_above_sum: 0,
            diag_load_nz_sum: 0,
            diag_load_mat_sum: 0,
            diag_heur_prune_sum: 0,
            diag_hist_nets: [0; 6],
            diag_hist_settles: [0; 6],
            diag_fan_nets: [0; 6],
            diag_fan_settles: [0; 6],
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
fn report_reff_distribution(
    network: &PipeNetwork,
    resistance_model: &ResistanceModel,
    iter: usize,
) {
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
    if matches!(
        graph_model,
        GraphModel::TwoLevelSpan | GraphModel::TwoLevelPip
    ) {
        if graph_model == GraphModel::TwoLevelPip {
            let stats = super::tile_cache::rebuild_span_cost_table_pip(network, resistance_model);
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
    ws_pool: &WorkspacePool,
    dist_cache: &mut DistCache,
    skip_mask: Option<&[bool]>,
    displacement: Option<&super::displacement::DisplacementTable>,
) -> SolveAccum {
    let accum = solve_all_nets_with_displacement(
        network,
        net_infos,
        cfg,
        solve_pool,
        ws_pool,
        dist_cache,
        false,
        skip_mask,
        displacement,
    );
    if ANALYTIC_PHASE.load(std::sync::atomic::Ordering::Relaxed) {
        dist_cache.seed_analytic(net_infos, network.width as usize, analytic_k_seed());
    } else if analytic_dist() {
        dist_cache.calibrate_analytic(net_infos, network.width as usize);
    }
    accum
}

fn solve_usage_and_energy(
    network: &PipeNetwork,
    net_infos: &[NetInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    ws_pool: &WorkspacePool,
    dist_cache: &mut DistCache,
    displacement: Option<&super::displacement::DisplacementTable>,
) -> SolveAccum {
    let accum = solve_all_nets_with_displacement(
        network,
        net_infos,
        cfg,
        solve_pool,
        ws_pool,
        dist_cache,
        true,
        None,
        displacement,
    );
    if ANALYTIC_PHASE.load(std::sync::atomic::Ordering::Relaxed) {
        dist_cache.seed_analytic(net_infos, network.width as usize, analytic_k_seed());
    } else if analytic_dist() {
        dist_cache.calibrate_analytic(net_infos, network.width as usize);
    }
    accum
}

/// Dual ascent on the per-pipe capacity constraint (PathFinder's `h`).
///
/// Driven by the **raw** per-iter usage rather than the EMA-blended
/// `net_count`: the EMA deliberately forgets, and an overflow that is smoothed
/// away before it is priced never enters the dual. Monotone by construction —
/// there is no decay term, and there must not be one.
fn update_pipe_history(network: &mut PipeNetwork, usage: &[f64], step: f64) {
    if step <= 0.0 {
        return;
    }
    let hist = &mut network.pipe_history;
    for (pipe_idx, pipe) in network.pipes.iter().enumerate() {
        let eff_cap = super::resistance::effective_capacity(pipe);
        if eff_cap <= 0.0 {
            continue;
        }
        let overflow = usage[pipe_idx] - eff_cap;
        if overflow > 0.0 {
            hist[pipe_idx] += step * pipe.base_resistance * (overflow / eff_cap);
        }
    }
}

fn apply_usage_for_next_iter(
    network: &mut PipeNetwork,
    usage: &[f64],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
) {
    // History first: it reads the raw usage, before the EMA folds it away.
    update_pipe_history(network, usage, cfg.hardening_step);
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

    // Lagrangian capacity term: pile on a per-tile pressure that grew
    // each iter from the count of rejected commits at this node. This is
    // a cell-density signal (not routing demand), so it bites tiles that
    // are full even when no net actually passes through them — the failure
    // mode that pure BPR misses.
    let tp_w = cfg.tile_pressure_weight;
    if tp_w > 0.0 && node < dist_cache.tile_pressure.len() {
        cost += tp_w * dist_cache.tile_pressure[node];
    }

    // Proximal term: `mu * |x - x_k|`, the displacement from where this cell
    // already sits. Two things at once.
    //
    // As optimisation, it turns each sweep into a proximal-point step —
    // minimise `f(x) + mu*|x - x_k|` rather than `f(x)` — which is the
    // standard way to make an iteration a descent on a surface you only
    // trust locally. The frozen dist_cache is exactly such a surface: it was
    // computed for the *previous* configuration.
    //
    // As mechanics, it damps the Jacobi swap. Every cell picks its argmin
    // against the same frozen field and all moves commit at once, so two
    // cells can trade places and re-create the configuration they left —
    // the move-level twin of the usage-level limit cycle. Charging for
    // distance travelled makes a swap cost something.
    //
    // Zero at the current tile by construction, so mu only ever breaks ties
    // in favour of staying put.
    let prox_w = cfg.proximal_weight;
    if prox_w > 0.0 {
        let cur_node = current_cell_node(cell_idx, cell_nets, net_infos);
        let cur_x = (cur_node % width) as f64;
        let cur_y = (cur_node / width) as f64;
        cost += prox_w * ((cand_nx - cur_x).abs() + (cand_ny - cur_y).abs());
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
        names.push(info.debug_name.to_string());
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

/// One cell's move plan: current position plus ranked candidate destinations,
/// primary first. Produced by the sweep's probe phase and consumed by its
/// commit phase; splitting the two is what lets commits run in a fixed order.
struct CellPlan {
    old_gx: i32,
    old_gy: i32,
    cands: [(i32, i32); PROBE_TOPK + 1],
    n_cands: usize,
    stats: Option<PlateauStat>,
}

/// Cells probed per deterministic chunk, via `NPNR_OT_DET_CHUNK`.
///
/// `None` (unset) means one chunk covering the sweep -- pure Jacobi, every cell
/// probing the sweep-start occupancy. Smaller values interleave probe and
/// commit, so later chunks see earlier commits. Only read when
/// `deterministic_sweep()` is on. A non-numeric or zero value is a
/// configuration error and panics rather than silently falling back.
fn det_sweep_chunk() -> Option<usize> {
    use std::sync::OnceLock;
    static CHUNK: OnceLock<Option<usize>> = OnceLock::new();
    *CHUNK.get_or_init(|| match std::env::var("NPNR_OT_DET_CHUNK") {
        Ok(v) => {
            let n: usize = v.parse().unwrap_or_else(|_| {
                panic!("NPNR_OT_DET_CHUNK must be a positive integer, got {v:?}")
            });
            assert!(n > 0, "NPNR_OT_DET_CHUNK must be > 0, got {n}");
            Some(n)
        }
        Err(_) => None,
    })
}

/// Commit moves in cell order instead of racing, via `NPNR_OT_DET_SWEEP=1`.
///
/// Off by default: it changes which cell wins a contested tile slot, so it is
/// a placement-behaviour change, not just a reproducibility one. A non-numeric
/// value is a configuration error and panics rather than silently falling back.
fn deterministic_sweep() -> bool {
    use std::sync::OnceLock;
    static DET: OnceLock<bool> = OnceLock::new();
    *DET.get_or_init(|| match std::env::var("NPNR_OT_DET_SWEEP") {
        Ok(v) => {
            let n: u32 = v.parse().unwrap_or_else(|_| {
                panic!("NPNR_OT_DET_SWEEP must be 0 or 1, got {v:?}")
            });
            assert!(n <= 1, "NPNR_OT_DET_SWEEP must be 0 or 1, got {n}");
            n == 1
        }
        Err(_) => false,
    })
}

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

    // Two phases, selected by `deterministic_sweep()`:
    //
    //   racy (default): probe and commit inside one parallel map. A cell sees
    //     occupancy other threads are still changing, so a contested tile slot
    //     goes to whichever thread reaches the CAS first -- the run-to-run and
    //     thread-count dependence measured in
    //     `docs/opt_trans_refresh_speedup.md`.
    //   deterministic: probe every cell against frozen occupancy, then commit
    //     in ascending cell index. Same candidates in the same rank order; a
    //     contested slot now goes to the lower cell index instead of the
    //     faster thread.
    //
    // Both share `plan_cell` and `commit_cell`, so the only difference is when
    // commits become visible. `net_infos` is frozen across the sweep either
    // way; only mux-slot occupancy is live.
    let probes: Vec<(i32, i32, bool, Option<PlateauStat>)> = {
        let plan_cell = |ci: usize| -> CellPlan {
            let cell_nets = &cell_net_map.map[ci];
            let mut plan = CellPlan {
                old_gx: network.tile_to_net(cell_x[ci]).round() as i32,
                old_gy: network.tile_to_net(cell_y[ci]).round() as i32,
                cands: [(0, 0); PROBE_TOPK + 1],
                n_cands: 0,
                stats: None,
            };
            if cell_nets.is_empty() {
                return plan;
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
            plan.stats = stats;
            let (primary_gx, primary_gy) = if use_softmin {
                acc.softmin_tile(ci, validity, network.width, network.height)
            } else {
                acc.argmin()
            };
            plan.cands[0] = (primary_gx, primary_gy);
            plan.n_cands = 1;
            for &(_, cand_gx, cand_gy) in acc.topk() {
                if cand_gx == primary_gx && cand_gy == primary_gy {
                    continue;
                }
                plan.cands[plan.n_cands] = (cand_gx, cand_gy);
                plan.n_cands += 1;
            }
            plan
        };
        // Each candidate is damped against `old` and tried via `try_commit`;
        // first success wins. Cells stay put only if every candidate fails.
        let commit_cell = |ci: usize, plan: &CellPlan| -> (i32, i32, bool) {
            let (old_gx, old_gy) = (plan.old_gx, plan.old_gy);
            for &(cand_gx, cand_gy) in &plan.cands[..plan.n_cands] {
                if cand_gx == old_gx && cand_gy == old_gy {
                    continue;
                }
                let new_gx =
                    (alpha * cand_gx as f64 + (1.0 - alpha) * old_gx as f64).round() as i32;
                let new_gy =
                    (alpha * cand_gy as f64 + (1.0 - alpha) * old_gy as f64).round() as i32;
                if new_gx == old_gx && new_gy == old_gy {
                    continue;
                }
                if mux_tracker.try_commit(ci, old_gx, old_gy, new_gx, new_gy) {
                    return (new_gx, new_gy, true);
                }
            }
            (old_gx, old_gy, false)
        };

        if deterministic_sweep() {
            // Chunked schedule: probe a chunk in parallel against occupancy
            // frozen at chunk start, commit it in index order, then let the
            // next chunk see the update. Chunk bounds come from the cell index,
            // never the worker count, so the result is thread-independent.
            //
            // `chunk = n` is pure Jacobi -- every cell probes the sweep-start
            // state. Smaller chunks recover the Gauss-Seidel information that
            // the racy path was getting by accident, at the cost of more
            // barriers. This is the knob the race was setting implicitly.
            let chunk = det_sweep_chunk().unwrap_or(n).min(n).max(1);
            let mut out: Vec<(i32, i32, bool, Option<PlateauStat>)> = Vec::with_capacity(n);
            let mut start = 0usize;
            while start < n {
                let end = (start + chunk).min(n);
                let plans: Vec<CellPlan> =
                    (start..end).into_par_iter().map(|ci| plan_cell(ci)).collect();
                for (offset, plan) in plans.into_iter().enumerate() {
                    let ci = start + offset;
                    let (gx, gy, committed) = commit_cell(ci, &plan);
                    out.push((gx, gy, committed, plan.stats));
                }
                start = end;
            }
            out
        } else {
            (0..n)
                .into_par_iter()
                .map(|ci| {
                    let plan = plan_cell(ci);
                    let (gx, gy, committed) = commit_cell(ci, &plan);
                    (gx, gy, committed, plan.stats)
                })
                .collect()
        }
    };

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
    refresh_dist_row(dist_cache.row_mut(net_idx), info, network, ws);
}

/// Recompute one net's dist_cache row (driver-anchored corridor Dijkstra labels)
/// in place. It touches only the passed-in `row` (plus the per-thread `ws`), so
/// it is safe to call in parallel across DISJOINT rows via `rows.par_iter_mut()`
/// — this is what the colored-GS per-color refresh relies on (no unsafe needed,
/// unlike `solve_all_nets` which must also write shared edge_usage). A
/// driver-less or sink-less net leaves its row unchanged (matches the prior
/// single-net refresh semantics).
fn refresh_dist_row(
    row: &mut FxHashMap<u32, f32>,
    info: &NetInfo,
    network: &PipeNetwork,
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
    path_solver::dial_logit_load(network, source, &sink_demands, ws, false);

    row.clear();
    row.reserve(ws.settle_order.len());
    for &node in &ws.settle_order {
        row.insert(node as u32, ws.dist[node] as f32);
    }
    if row.capacity() > 4 * row.len().max(8) {
        row.shrink_to_fit();
    }
}

/// DIAGNOSTIC (NPNR_OT_COLOR_DIAG=1): measure the cell net-adjacency coloring
/// that colored Gauss-Seidel would use, then exit. High-fanout nets form huge
/// cliques that blow up the chromatic number, so nets with more than
/// `NPNR_OT_COLOR_FANOUT` (default 16) movable cells are excluded from the
/// conflict graph. If the resulting color count C is small (tens), colored-GS is
/// viable (C incremental refreshes per sweep); if large, a spatial decomposition
/// is needed instead.
/// Greedy net-adjacency coloring of cells for colored Gauss-Seidel. Two cells
/// receive distinct colors iff they share a net with at most `fanout_threshold`
/// movable cells. High-fanout nets (clocks) are EXCLUDED from the conflict graph
/// — they form huge cliques that would explode the color count; same-color cells
/// may share a high-fanout net and update with mild staleness, which is
/// acceptable since their per-pin pull is small and averaged.
struct CellColoring {
    /// Per-cell color id in `0..num_colors`. Cells with no low-fanout net still
    /// get a valid color (0) — they are trivially independent.
    colors: Vec<u32>,
    num_colors: u32,
    n_low_fanout_nets: usize,
    n_high_fanout_excluded: usize,
    max_degree: u32,
}

fn build_cell_coloring(
    net_infos: &[NetInfo],
    cell_net_map: &CellNetMap,
    n: usize,
    fanout_threshold: usize,
) -> CellColoring {
    let n_nets = net_infos.len();
    let mut net_cells: Vec<Vec<usize>> = vec![Vec::new(); n_nets];
    for (ni, info) in net_infos.iter().enumerate() {
        for pin in &info.pins {
            if let Some(ci) = pin.cell_idx {
                if !pin.is_fixed && ci < n {
                    net_cells[ni].push(ci);
                }
            }
        }
    }
    let mut n_high = 0usize;
    let mut n_low = 0usize;
    for cells in net_cells.iter_mut() {
        if cells.len() > fanout_threshold {
            cells.clear(); // exclude high-fanout net from the conflict graph
            n_high += 1;
        } else if cells.len() >= 2 {
            n_low += 1;
        }
    }

    // Approx degree = sum over low-fanout nets of (net_size - 1). Used only to
    // order cells largest-first (Welsh-Powell) for a tighter color count.
    let mut degree: Vec<u32> = vec![0; n];
    for cells in net_cells.iter() {
        let k = cells.len();
        if k >= 2 {
            for &ci in cells {
                degree[ci] += (k - 1) as u32;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| degree[b].cmp(&degree[a]));

    // Greedy coloring with a stamp-based forbidden set (no per-cell allocation).
    let mut colors: Vec<u32> = vec![u32::MAX; n];
    let mut stamp: Vec<u32> = vec![0u32; 1];
    let mut cur_stamp = 0u32;
    let mut num_colors = 0u32;
    for &c in &order {
        cur_stamp += 1;
        for &(ni, _) in &cell_net_map.map[c] {
            for &other in &net_cells[ni] {
                if other == c {
                    continue;
                }
                let col = colors[other];
                if col != u32::MAX {
                    let cu = col as usize;
                    if cu >= stamp.len() {
                        stamp.resize(cu + 1, 0);
                    }
                    stamp[cu] = cur_stamp;
                }
            }
        }
        let mut chosen = 0u32;
        while (chosen as usize) < stamp.len() && stamp[chosen as usize] == cur_stamp {
            chosen += 1;
        }
        colors[c] = chosen;
        if chosen + 1 > num_colors {
            num_colors = chosen + 1;
        }
    }
    // Cells touched by no low-fanout net never entered the loop body's forbidden
    // path but still got a color (the loop colors every cell). num_colors >= 1
    // whenever n > 0.
    if num_colors == 0 {
        num_colors = 1;
    }

    CellColoring {
        colors,
        num_colors,
        n_low_fanout_nets: n_low,
        n_high_fanout_excluded: n_high,
        max_degree: degree.iter().copied().max().unwrap_or(0),
    }
}

/// DIAGNOSTIC (NPNR_OT_COLOR_DIAG=1): print the coloring stats and exit.
fn measure_cell_coloring(net_infos: &[NetInfo], cell_net_map: &CellNetMap, n: usize) {
    let threshold: usize = std::env::var("NPNR_OT_COLOR_FANOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let col = build_cell_coloring(net_infos, cell_net_map, n, threshold);
    let mut sizes: Vec<usize> = vec![0; col.num_colors as usize];
    for &c in &col.colors {
        if c != u32::MAX {
            sizes[c as usize] += 1;
        }
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let top: Vec<usize> = sizes.iter().take(8).copied().collect();
    eprintln!(
        "  COLOR_DIAG: cells={} fanout_threshold={} nets_total={} low_fanout_nets={} high_fanout_excluded={} => C={} max_degree={} top_color_sizes={:?}",
        n, threshold, net_infos.len(), col.n_low_fanout_nets, col.n_high_fanout_excluded, col.num_colors, col.max_degree, top,
    );
    eprintln!(
        "  COLOR_DIAG: colored-GS would need ~C={} incremental refreshes/sweep; viable if C is small (tens). Exiting.",
        col.num_colors,
    );
    std::process::exit(0);
}

/// DIAGNOSTIC sweep: move each cell to its true-HPWL optimum (the median of its
/// connected nets' other-pin bbox bounds) against frozen positions, then commit
/// with the capacity gate. Bypasses the dist_cache/Dijkstra star objective
/// entirely. Compare its HPWL trajectory against the star sweeps: if median wins
/// big, objective fidelity is the cap; if it ties, the convergence/local-CD
/// paradigm is the cap.
fn place_dcd_sweep_median(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
) -> usize {
    let n = cell_x.len();
    let width = network.width as usize;

    // Phase 1: parallel — each cell computes its median target against frozen
    // positions. The cell's HPWL contribution is piecewise-linear convex in x
    // (and y); the minimizer is the median of the per-net {lo, hi} breakpoints.
    let best: Vec<(i32, i32)> = (0..n)
        .into_par_iter()
        .map(|ci| {
            let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
            let cell_nets = &cell_net_map.map[ci];
            if cell_nets.is_empty() {
                return (old_gx, old_gy);
            }
            let mut xs: Vec<f32> = Vec::with_capacity(cell_nets.len() * 2);
            let mut ys: Vec<f32> = Vec::with_capacity(cell_nets.len() * 2);
            for &(net_idx, pin_idx) in cell_nets {
                let pins = &net_infos[net_idx].pins;
                let mut lo_x = f32::INFINITY;
                let mut hi_x = f32::NEG_INFINITY;
                let mut lo_y = f32::INFINITY;
                let mut hi_y = f32::NEG_INFINITY;
                let mut any = false;
                for (pi, pin) in pins.iter().enumerate() {
                    if pi == pin_idx {
                        continue;
                    }
                    let nx = (pin.node % width) as f32;
                    let ny = (pin.node / width) as f32;
                    lo_x = lo_x.min(nx);
                    hi_x = hi_x.max(nx);
                    lo_y = lo_y.min(ny);
                    hi_y = hi_y.max(ny);
                    any = true;
                }
                if any {
                    xs.push(lo_x);
                    xs.push(hi_x);
                    ys.push(lo_y);
                    ys.push(hi_y);
                }
            }
            if xs.is_empty() {
                return (old_gx, old_gy);
            }
            xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let tx = xs[xs.len() / 2].round() as i32;
            let ty = ys[ys.len() / 2].round() as i32;
            if validity.is_valid(ci, tx, ty) {
                (tx, ty)
            } else {
                (old_gx, old_gy)
            }
        })
        .collect();

    // Phase 2: serial commit with capacity gate (no dist_cache refresh — the
    // median objective never reads it).
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

/// Colored Gauss-Seidel sweep (v1, sequential incremental refresh).
///
/// Cells are greedy-colored by net-adjacency (high-fanout nets excluded). Colors
/// are processed in sequence; within a color all cells argmin in PARALLEL via
/// bisection against the CURRENT dist_cache, which already reflects earlier
/// colors' moves this sweep (the Gauss-Seidel freshness that damped Jacobi
/// lacks). After committing a color, only the touched nets' dist_cache rows are
/// refreshed before the next color. The objective and spreading are unchanged
/// (existing star/corridor + BPR; no density). The v1 per-color refresh is
/// SEQUENTIAL (safe Rust) — fine for small designs (sv3) to validate the
/// fresh-field hypothesis; parallelizing it (for FPGA01 scale) is a follow-up.
#[allow(clippy::too_many_arguments)]
fn place_dcd_sweep_colored_gs(
    net_infos: &mut [NetInfo],
    cell_net_map: &CellNetMap,
    dist_cache: &mut DistCache,
    coloring: &CellColoring,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    network: &PipeNetwork,
    cfg: &OptTransPlacerCfg,
    validity: &CellValidityMask,
    mux_tracker: &MuxSlotTracker,
    diag: &mut DiagCtx,
) -> usize {
    let n = cell_x.len();
    // Per-sweep displacement cap (damping): limit each cell's move to within
    // `move_cap` tiles of its current position. Counteracts the over-compaction
    // cascade caused by the half-fresh field (fresh distances, stale congestion):
    // small per-sweep steps let the outer congestion refresh keep pace. Default
    // = unlimited (i32::MAX) so behaviour is unchanged unless NPNR_OT_COLOR_MOVE_CAP
    // is set.
    let move_cap: i32 = std::env::var("NPNR_OT_COLOR_MOVE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(i32::MAX);

    let mut by_color: Vec<Vec<usize>> = vec![Vec::new(); coloring.num_colors as usize];
    for ci in 0..n {
        let c = coloring.colors[ci] as usize;
        if c < by_color.len() {
            by_color[c].push(ci);
        }
    }

    let mut moved = 0usize;

    for color_cells in &by_color {
        if color_cells.is_empty() {
            continue;
        }
        // Phase 1: parallel argmin against the CURRENT (fresh) dist_cache.
        let best: Vec<(usize, i32, i32)> = {
            let ni: &[NetInfo] = net_infos;
            let dc: &DistCache = dist_cache;
            let cx: &[f64] = cell_x;
            let cy: &[f64] = cell_y;
            color_cells
                .par_iter()
                .map(|&ci| {
                    let cell_nets = &cell_net_map.map[ci];
                    let ogx = network.tile_to_net(cx[ci]).round() as i32;
                    let ogy = network.tile_to_net(cy[ci]).round() as i32;
                    if cell_nets.is_empty() {
                        return (ci, ogx, ogy);
                    }
                    let cur =
                        evaluate_cell_at(ci, ogx, ogy, cell_nets, ni, dc, network, cfg, validity);
                    let (bx, by, _bc, _a, _b, _c) =
                        bisect_2d(ci, cell_nets, ni, dc, network, cfg, validity, ogx, ogy, cur);
                    if move_cap != i32::MAX {
                        let cbx = bx.clamp(ogx - move_cap, ogx + move_cap);
                        let cby = by.clamp(ogy - move_cap, ogy + move_cap);
                        if (cbx, cby) != (bx, by) && !validity.is_valid(ci, cbx, cby) {
                            return (ci, ogx, ogy); // capped target invalid -> hold
                        }
                        return (ci, cbx, cby);
                    }
                    (ci, bx, by)
                })
                .collect()
        };

        // Phase 2: serial commit (capacity gate) + collect touched nets.
        let mut touched: Vec<usize> = Vec::new();
        for (ci, ngx, ngy) in best {
            let ogx = network.tile_to_net(cell_x[ci]).round() as i32;
            let ogy = network.tile_to_net(cell_y[ci]).round() as i32;
            if ngx == ogx && ngy == ogy {
                continue;
            }
            if !mux_tracker.try_commit(ci, ogx, ogy, ngx, ngy) {
                continue;
            }
            cell_x[ci] = network.net_to_tile(ngx as f64);
            cell_y[ci] = network.net_to_tile(ngy as f64);
            let new_node = coord_to_node(network, ngx, ngy);
            for &(net_idx, pin_idx) in &cell_net_map.map[ci] {
                net_infos[net_idx].pins[pin_idx].node = new_node;
                touched.push(net_idx);
            }
            if diag.enabled {
                diag.record_move(MoveRecord {
                    cell_idx: ci,
                    old_gx: ogx,
                    old_gy: ogy,
                    new_gx: ngx,
                    new_gy: ngy,
                });
            }
            moved += 1;
        }

        // Phase 3: incremental refresh of only the touched nets so the next color
        // sees the fresh field. Parallel over DISJOINT dist_cache rows — safe (no
        // unsafe): each row is independent and gets a per-thread workspace via
        // for_each_init. `solve_all_nets` reaches its rows the same way, by
        // zipping `par_chunks_mut` over the cache alongside the net list; the
        // shared `edge_usage` write it also performs is handled by per-thread
        // fold accumulators rather than shared mutable state.
        touched.sort_unstable();
        touched.dedup();
        let mut touched_mask = vec![false; net_infos.len()];
        for &net_idx in &touched {
            touched_mask[net_idx] = true;
        }
        let ni: &[NetInfo] = net_infos;
        let n_nodes = network.num_nodes();
        let n_pipes = network.num_pipes();
        dist_cache.rows.par_iter_mut().enumerate().for_each_init(
            || PathSolverWorkspace::new(n_nodes, n_pipes),
            |tls, (net_idx, row)| {
                if touched_mask[net_idx] {
                    refresh_dist_row(row, &ni[net_idx], network, tls);
                }
            },
        );
    }
    moved
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

    // Swap-move pass: escape coupled-move local minima. When a cell's
    // frozen-field argmin tile is occupied (so the cell could not move there in
    // Phase B), try swapping it with an occupant of that tile. A swap is
    // capacity-neutral (each tile loses one cell and gains one), so the
    // MuxSlotTracker counts stay correct without any update — provided both
    // cells have the same external-fanout status (only such cells are counted).
    // The only other requirements are mutual validity and a strict cost
    // decrease under the frozen dist_cache. Gated behind NPNR_OT_SWAP_MOVES=1.
    if std::env::var("NPNR_OT_SWAP_MOVES").ok().as_deref() == Some("1") {
        // Tile -> occupant cell indices, from post-Phase-B positions.
        let mut occ: FxHashMap<(i32, i32), Vec<usize>> = FxHashMap::default();
        for ci in 0..cell_x.len() {
            if cell_net_map.map[ci].is_empty() {
                continue;
            }
            let gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let gy = network.tile_to_net(cell_y[ci]).round() as i32;
            occ.entry((gx, gy)).or_default().push(ci);
        }

        let mut swapped = vec![false; cell_x.len()];
        let mut swaps_applied = 0usize;
        let mut swaps_attempted = 0usize;

        for (order_idx, &ci) in iter_order.iter().enumerate() {
            if swapped[ci] {
                continue;
            }
            let nets_i = &cell_net_map.map[ci];
            if nets_i.is_empty() {
                continue;
            }
            let (tied, _) = &tied_per_cell[order_idx];
            if tied.is_empty() {
                continue;
            }
            let old_gx = network.tile_to_net(cell_x[ci]).round() as i32;
            let old_gy = network.tile_to_net(cell_y[ci]).round() as i32;
            // If the cell already sits on one of its argmin tiles it is not
            // blocked — there is nothing to swap for.
            if tied.iter().any(|&(x, y)| x == old_gx && y == old_gy) {
                continue;
            }
            let (tx, ty) = tied.iter().copied().min().unwrap();
            if !validity.is_valid(ci, tx, ty) {
                continue;
            }
            let cur_i = evaluate_cell_at(
                ci, old_gx, old_gy, nets_i, net_infos, dist_cache, network, cfg, validity,
            );
            let want_i = evaluate_cell_at(
                ci, tx, ty, nets_i, net_infos, dist_cache, network, cfg, validity,
            );
            if !cur_i.is_finite() || !want_i.is_finite() {
                continue;
            }
            // Candidate occupants of the desired tile (clone to drop the borrow
            // on `occ` before any mutation below).
            let cand: Vec<usize> = match occ.get(&(tx, ty)) {
                Some(v) => v
                    .iter()
                    .copied()
                    .filter(|&j| j != ci && !swapped[j])
                    .collect(),
                None => continue,
            };
            for j in cand {
                let nets_j = &cell_net_map.map[j];
                if nets_j.is_empty() {
                    continue;
                }
                // Capacity-neutrality holds only when both cells are counted
                // the same way by the tracker.
                if mux_tracker.drives_external(ci) != mux_tracker.drives_external(j) {
                    continue;
                }
                // Skip pairs sharing a net so the independent delta is exact.
                let shares = nets_i
                    .iter()
                    .any(|&(ni, _)| nets_j.iter().any(|&(nj, _)| nj == ni));
                if shares {
                    continue;
                }
                if !validity.is_valid(j, old_gx, old_gy) {
                    continue;
                }
                let cur_j = evaluate_cell_at(
                    j, tx, ty, nets_j, net_infos, dist_cache, network, cfg, validity,
                );
                let want_j = evaluate_cell_at(
                    j, old_gx, old_gy, nets_j, net_infos, dist_cache, network, cfg, validity,
                );
                if !cur_j.is_finite() || !want_j.is_finite() {
                    continue;
                }
                swaps_attempted += 1;
                let delta = (want_i + want_j) - (cur_i + cur_j);
                if delta < -1e-9 {
                    cell_x[ci] = network.net_to_tile(tx as f64);
                    cell_y[ci] = network.net_to_tile(ty as f64);
                    cell_x[j] = network.net_to_tile(old_gx as f64);
                    cell_y[j] = network.net_to_tile(old_gy as f64);
                    let node_i = coord_to_node(network, tx, ty);
                    let node_j = coord_to_node(network, old_gx, old_gy);
                    for &(net_idx, pin_idx) in nets_i {
                        net_infos[net_idx].pins[pin_idx].node = node_i;
                    }
                    for &(net_idx, pin_idx) in nets_j {
                        net_infos[net_idx].pins[pin_idx].node = node_j;
                    }
                    swapped[ci] = true;
                    swapped[j] = true;
                    // Maintain the occupancy index: ci now at (tx,ty), j at old.
                    if let Some(v) = occ.get_mut(&(tx, ty)) {
                        if let Some(pos) = v.iter().position(|&c| c == j) {
                            v[pos] = ci;
                        }
                    }
                    if let Some(v) = occ.get_mut(&(old_gx, old_gy)) {
                        if let Some(pos) = v.iter().position(|&c| c == ci) {
                            v[pos] = j;
                        }
                    }
                    swaps_applied += 1;
                    break;
                }
            }
        }

        moved += swaps_applied;
        eprintln!(
            "    swap_moves: applied={} attempted={}",
            swaps_applied, swaps_attempted,
        );
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

    let topo_order = cell_net_map
        .topo_order
        .as_ref()
        .expect("sequential bisection needs the topological order");
    for &ci in topo_order {
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
    let t_dcd_entry = std::time::Instant::now();
    let n = cell_x.len();
    let n_nodes = network.num_nodes();
    let mut dist_cache = DistCache::new(0, n_nodes);
    // One pool for the whole placement: solver workspaces are chip-sized and
    // were previously rebuilt for every rayon task on every outer iteration.
    let ws_pool = WorkspacePool::new(n_nodes, network.num_pipes());
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
    {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        let adj = &network.flat_adjacency;
        let pipes_b = network.pipes.capacity() * std::mem::size_of::<super::network::Pipe>();
        let nodes_b = network.nodes.capacity() * std::mem::size_of::<super::network::Node>();
        let adj_b = (adj.offsets.capacity() + adj.neighbors.capacity() + adj.pipe_idx.capacity()
            + adj.cost_int.capacity()) * 4
            + adj.cost_f.capacity() * 4;
        let costvecs_b = network.pipe_costs.capacity() * 8
            + network.pipe_costs_int.capacity() * 4
            + network.pipe_history.capacity() * 8
            + network.span_cost_table.pipe_entry.capacity() * 4
            + network.span_cost_table.costs.capacity() * 8
            + network.span_cost_table.costs_int.capacity() * 4;
        eprintln!(
            "MEM_STATIC: pipes={:.0}MB nodes={:.0}MB adjacency={:.0}MB pipe_cost_vecs={:.0}MB validity_mask={:.0}MB n_pipes={} n_edges={} rss={:.0}MB",
            mb(pipes_b), mb(nodes_b), mb(adj_b), mb(costvecs_b),
            mb(validity.heap_bytes()),
            network.pipes.len(), adj.neighbors.len(),
            process_rss_kb() as f64 / 1024.0,
        );
    }
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
    let mut netmap_cache: Option<Arc<CellNetMap>> = None;
    let named_nets = NameFilteredNets::new(ctx, alive_net_ids, cfg);
    eprintln!("  ALG_T dcd_setup_done: {:.2}s (pre-loop)", t_dcd_entry.elapsed().as_secs_f64());
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
            &named_nets,
            cell_to_idx,
            cell_x,
            cell_y,
            network,
            cfg,
        );
        let cell_net_map = CellNetMap::build_cached(&net_infos, n, want_topo_order(cfg), &mut netmap_cache);
        if outer == 0 && std::env::var("NPNR_OT_COLOR_DIAG").ok().as_deref() == Some("1") {
            measure_cell_coloring(&net_infos, &cell_net_map, n);
        }
        let prev_dist_cache_shape = (dist_cache.n_nets, dist_cache.n_nodes);
        dist_cache.ensure_shape(net_infos.len(), n_nodes);
        let dist_cache_was_realloced =
            prev_dist_cache_shape != (dist_cache.n_nets, dist_cache.n_nodes) || outer == 0;

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
        if set_analytic_phase(outer) {
            eprintln!("    analytic_phase[outer={}]: solve skipped, k={}", outer, analytic_k_seed());
        }
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
        eprintln!("PHASE_MARK outer={} enter=pre_solve rss={:.0}MB", outer, process_rss_kb() as f64 / 1024.0);
        let rss_before_pre_solve = process_rss_kb();
        let pre_solve = solve_distance_cache(
            network,
            &net_infos,
            cfg,
            solve_pool,
            &ws_pool,
            &mut dist_cache,
            skip_mask.as_deref(),
            displacement_table.as_ref(),
        );
        if std::env::var("NPNR_OT_DETERMINISM").ok().as_deref() == Some("1") {
            let mut h_dist: u64 = 0;
            for row in dist_cache.rows.iter() {
                // Order-independent per row so hashmap traversal order cannot
                // colour the result.
                let mut r: u64 = 0;
                for (k, v) in row.iter() {
                    r = r.wrapping_add(
                        (*k as u64)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (v.to_bits() as u64),
                    );
                }
                h_dist = h_dist.rotate_left(5).wrapping_add(r);
            }
            let mut h_pos0: u64 = 0;
            for (x, y) in cell_x.iter().zip(cell_y.iter()) {
                h_pos0 = h_pos0
                    .rotate_left(5)
                    .wrapping_add(x.to_bits())
                    .rotate_left(7)
                    .wrapping_add(y.to_bits());
            }
            eprintln!(
                "    determinism_pre[outer={}]: pos_in={:016x} dist={:016x} energy={:.17e}",
                outer, h_pos0, h_dist, pre_solve.energy,
            );
        }
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
            /// Hops a greedy longest-first decomposition needs to cover `d`
            /// tiles. Assumes the UltraScale span set {12,4,2,1}; this is a
            /// diagnostic only, so it is not read from the chipdb.
            fn greedy_span_hops(d: i32) -> u32 {
                let mut left = d;
                let mut hops = 0u32;
                for span in [12, 4, 2, 1] {
                    while left >= span {
                        left -= span;
                        hops += 1;
                    }
                }
                hops
            }

            /// Cost of a MONOTONE straight walk from `src` to `dst`, priced at
            /// the current `R_eff`. At each node take the neighbour with the
            /// best cost-per-tile-of-progress that never moves away on either
            /// axis -- O(degree) per hop, no search, no heap. This is the
            /// closed-form competitor: comparing it to the Dijkstra label
            /// isolates what the SEARCH buys from what BPR pricing buys, which
            /// a tiles-vs-cost ratio cannot separate once pipes saturate.
            fn straight_cost(network: &PipeNetwork, src: usize, dst: usize) -> Option<f64> {
                let adj = &network.flat_adjacency;
                let (tx, ty) = (network.nodes[dst].tile_x, network.nodes[dst].tile_y);
                let mut cur = src;
                let mut total = 0.0f64;
                for _ in 0..1024 {
                    let (cx, cy) = (network.nodes[cur].tile_x, network.nodes[cur].tile_y);
                    let rem = (tx - cx).abs() + (ty - cy).abs();
                    if rem == 0 {
                        return Some(total);
                    }
                    let (s, e) = (
                        adj.offsets[cur] as usize,
                        adj.offsets[cur + 1] as usize,
                    );
                    let mut best: Option<(f64, f64, usize)> = None;
                    for k in s..e {
                        let nb = adj.neighbors[k] as usize;
                        let (nx, ny) = (network.nodes[nb].tile_x, network.nodes[nb].tile_y);
                        if (tx - nx).abs() > (tx - cx).abs()
                            || (ty - ny).abs() > (ty - cy).abs()
                        {
                            continue;
                        }
                        let prog = rem - ((tx - nx).abs() + (ty - ny).abs());
                        if prog <= 0 {
                            continue;
                        }
                        let cost = adj.cost_f[k] as f64;
                        let rate = cost / prog as f64;
                        if best.map_or(true, |(br, _, _)| rate < br) {
                            best = Some((rate, cost, nb));
                        }
                    }
                    let (_, cost, nb) = best?;
                    total += cost;
                    cur = nb;
                }
                None
            }

            let mut n_valid = 0usize;
            let mut sum_hpwl = 0.0f64;
            let mut sum_star = 0.0f64;
            let mut n_ratio_gt2 = 0usize;
            let mut n_ratio_gt5 = 0usize;
            let mut n_ratio_lt05 = 0usize;
            let mut n_ratio_lt02 = 0usize;
            let mut sum_manh = 0.0f64;
            let mut sum_hops = 0.0f64;
            let mut sum_straight = 0.0f64;
            let mut n_straight_fail = 0usize;
            let mut n_search_wins = 0usize;
            let mut n_detour_gt5pct = 0usize;
            let mut n_detour_gt25pct = 0usize;
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
                // Closed-form control for the Dijkstra: the same star sum priced
                // as pure Manhattan, plus the hop count a greedy span
                // decomposition would need. At exp=1.0 wire cost is
                // span-invariant, so `star` can only exceed `manh` via the
                // per-hop switch-matrix term or via a genuine detour around
                // congestion -- which is exactly what we need to size before
                // replacing the path solve with a closed form.
                let driver_node = info.pins.iter().find(|p| p.is_driver).map(|p| p.node);
                let driver_xy = driver_node.map(|nd| {
                    let n = &network.nodes[nd];
                    (n.tile_x, n.tile_y)
                });
                let mut star = 0.0f64;
                let mut straight = 0.0f64;
                let mut manh = 0.0f64;
                let mut hops = 0u32;
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
                    if let Some((sx, sy)) = driver_xy {
                        let n = &network.nodes[pin.node];
                        let (dx, dy) = ((n.tile_x - sx).abs(), (n.tile_y - sy).abs());
                        manh += (dx + dy) as f64;
                        hops += greedy_span_hops(dx) + greedy_span_hops(dy);
                    }
                    if let Some(dn) = driver_node {
                        match straight_cost(network, dn, pin.node) {
                            Some(c) => straight += c,
                            None => n_straight_fail += 1,
                        }
                    }
                }
                if !ok || fanout == 0 {
                    continue;
                }
                n_valid += 1;
                sum_hpwl += hpwl;
                sum_star += star;
                sum_manh += manh;
                sum_hops += hops as f64;
                sum_straight += straight;
                if straight > star * 1.05 {
                    n_search_wins += 1;
                }
                if manh > 0.5 {
                    let dr = star / manh;
                    if dr > 1.05 {
                        n_detour_gt5pct += 1;
                    }
                    if dr > 1.25 {
                        n_detour_gt25pct += 1;
                    }
                }
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
            // Closed-form control. `star/manh` is the entire value the Dijkstra
            // adds over an O(1) Manhattan lookup: 1.000 means the search is
            // recomputing a number we already know.
            eprintln!(
                "    ClosedForm_check: sum_manh={:.0} star/manh={:.4} sum_hops={:.0} hops/sink={:.2} detour>5%={} detour>25%={} (of {} nets)",
                sum_manh,
                if sum_manh > 0.0 { sum_star / sum_manh } else { 0.0 },
                sum_hops,
                if n_valid > 0 { sum_hops / n_valid as f64 } else { 0.0 },
                n_detour_gt5pct,
                n_detour_gt25pct,
                n_valid,
            );
            // The decisive ratio. `straight` is priced at the SAME R_eff as
            // `star`, so straight/star == 1.000 means the search found nothing
            // a monotone greedy walk missed, at any congestion level.
            eprintln!(
                "    SearchValue_check: sum_straight={:.0} sum_star={:.0} straight/star={:.4} search_wins>5%={} of {} nets (walk_fail={})",
                sum_straight,
                sum_star,
                if sum_star > 0.0 { sum_straight / sum_star } else { 0.0 },
                n_search_wins,
                n_valid,
                n_straight_fail,
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

        // Debug: compare DCD bisection vs full scan on iter 0. Gated on
        // NPNR_OT_BIS_FS_DIAG=1 — the fullscan_find_best_position call inside
        // this loop runs O(grid_area * n_nets_per_cell) per cell and dominates
        // iter 0 wall time on FPGA-scale designs.
        if outer == 0 && std::env::var("NPNR_OT_BIS_FS_DIAG").ok().as_deref() == Some("1") {
            let want_csv = true;
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

        if std::env::var("NPNR_OT_DETERMINISM").ok().as_deref() == Some("1") {
            let mut h: u64 = 0;
            for (x, y) in cell_x.iter().zip(cell_y.iter()) {
                h = h.rotate_left(5)
                    .wrapping_add(x.to_bits())
                    .rotate_left(7)
                    .wrapping_add(y.to_bits());
            }
            eprintln!("    det_presweep[outer={}]: pos={:016x}", outer, h);
        }
        diag_ctx.sweep_begin(n, outer);
        eprintln!("PHASE_MARK outer={} enter=dcd_sweep rss={:.0}MB", outer, process_rss_kb() as f64 / 1024.0);
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
                    cell_net_map
                        .topo_order
                        .as_ref()
                        .expect("JacobiBB needs the topological order"),
                )
            }
            super::config::SweepMode::MedianDiag => place_dcd_sweep_median(
                &mut net_infos,
                &cell_net_map,
                cell_x,
                cell_y,
                network,
                validity,
                &mux_tracker,
                &mut diag_ctx,
            ),
            super::config::SweepMode::ColoredGs => {
                let threshold: usize = std::env::var("NPNR_OT_COLOR_FANOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(16);
                let coloring = build_cell_coloring(&net_infos, &cell_net_map, n, threshold);
                if outer == 0 {
                    eprintln!(
                        "    ColoredGS: C={} colors (fanout_threshold={}, {} high-fanout nets excluded)",
                        coloring.num_colors, threshold, coloring.n_high_fanout_excluded,
                    );
                }
                place_dcd_sweep_colored_gs(
                    &mut net_infos,
                    &cell_net_map,
                    &mut dist_cache,
                    &coloring,
                    cell_x,
                    cell_y,
                    network,
                    cfg,
                    validity,
                    &mux_tracker,
                    &mut diag_ctx,
                )
            }
        };
        if std::env::var("NPNR_OT_DETERMINISM").ok().as_deref() == Some("1") {
            let mut h: u64 = 0;
            for (x, y) in cell_x.iter().zip(cell_y.iter()) {
                h = h.rotate_left(5)
                    .wrapping_add(x.to_bits())
                    .rotate_left(7)
                    .wrapping_add(y.to_bits());
            }
            eprintln!("    det_postsweep[outer={}]: pos={:016x} moved={}", outer, h, moved);
        }
        eprintln!("PHASE_MARK outer={} leave=dcd_sweep rss={:.0}MB", outer, process_rss_kb() as f64 / 1024.0);
        let dcd_ms = t_dcd.elapsed().as_millis();

        common::clamp_positions(cell_x, cell_y, phys_max_x, phys_max_y);

        eprintln!("PHASE_MARK outer={} enter=post_refresh rss={:.0}MB", outer, process_rss_kb() as f64 / 1024.0);
        let t_post_refresh = std::time::Instant::now();
        let post_net_infos = collect_net_infos_simple(
            ctx,
            &named_nets,
            cell_to_idx,
            cell_x,
            cell_y,
            network,
            cfg,
        );
        let post_cell_net_map = CellNetMap::build_cached(&post_net_infos, n, want_topo_order(cfg), &mut netmap_cache);
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
                &ws_pool,
                &mut dist_cache,
                displacement_table.as_ref(),
            );
            let rss_after_post = process_rss_kb();
            {
                // The post-solve is the only full solve pass after outer=0
                // (the pre-solve skips every net), so this is where the
                // refresh cost actually is.
                let solves = post_solve.diag_solves.max(1);
                let settle_avg = post_solve.diag_settle_sum as f64 / solves as f64;
                let above_avg = post_solve.diag_above_sum as f64 / solves as f64;
                eprintln!(
                    "    post_solve_diag[outer={}]: solves={} cap_armed={} ({:.1}%) settle_avg={:.0} above_sink_avg={:.0} ({:.1}%) settle_max={}",
                    outer,
                    post_solve.diag_solves,
                    post_solve.diag_cap_armed,
                    100.0 * post_solve.diag_cap_armed as f64 / solves as f64,
                    settle_avg,
                    above_avg,
                    if settle_avg > 0.0 { 100.0 * above_avg / settle_avg } else { 0.0 },
                    post_solve.diag_settle_max,
                );
                let nz_avg = post_solve.diag_load_nz_sum as f64 / solves as f64;
                let mat_avg = post_solve.diag_load_mat_sum as f64 / solves as f64;
                eprintln!(
                    "    load_support[outer={}]: settle_avg={:.0} load_nz_avg={:.0} ({:.1}%) load_material_avg={:.0} ({:.1}%)",
                    outer,
                    settle_avg,
                    nz_avg,
                    if settle_avg > 0.0 { 100.0 * nz_avg / settle_avg } else { 0.0 },
                    mat_avg,
                    if settle_avg > 0.0 { 100.0 * mat_avg / settle_avg } else { 0.0 },
                );
                let heur_avg = post_solve.diag_heur_prune_sum as f64 / solves as f64;
                eprintln!(
                    "    prune_ceiling[outer={}]: settle_avg={:.0} heuristic_prunable_avg={:.0} ({:.1}%) vs oracle_zero_load={:.1}%",
                    outer,
                    settle_avg,
                    heur_avg,
                    if settle_avg > 0.0 { 100.0 * heur_avg / settle_avg } else { 0.0 },
                    if settle_avg > 0.0 { 100.0 * (settle_avg - nz_avg) / settle_avg } else { 0.0 },
                );
                let total_settles: u64 = post_solve.diag_hist_settles.iter().sum();
                let labels = ["<100", "<1k", "<5k", "<20k", "<60k", ">=60k"];
                let hist: Vec<String> = (0..6)
                    .map(|b| {
                        format!(
                            "{}: {}nets/{:.1}%work",
                            labels[b],
                            post_solve.diag_hist_nets[b],
                            if total_settles > 0 {
                                100.0 * post_solve.diag_hist_settles[b] as f64
                                    / total_settles as f64
                            } else {
                                0.0
                            }
                        )
                    })
                    .collect();
                eprintln!("    settle_hist[outer={}]: {}", outer, hist.join("  "));
                let fan_labels = ["fan1", "fan2-4", "fan5-16", "fan17-64", "fan65-256", "fan>256"];
                let fan: Vec<String> = (0..6)
                    .map(|b| {
                        format!(
                            "{}: {}nets/{:.1}%work",
                            fan_labels[b],
                            post_solve.diag_fan_nets[b],
                            if total_settles > 0 {
                                100.0 * post_solve.diag_fan_settles[b] as f64
                                    / total_settles as f64
                            } else {
                                0.0
                            }
                        )
                    })
                    .collect();
                eprintln!("    fanout_hist[outer={}]: {}", outer, fan.join("  "));
            }
            apply_usage_for_next_iter(network, &post_solve.edge_usage, cfg, solve_pool);
            let refresh_ms = pre_refresh_ms + t_post_refresh.elapsed().as_millis();
            eprintln!(
                "    rss_probe[outer={}]: pre_post_solve={:.0}MB post_post_solve={:.0}MB after_apply={:.0}MB",
                outer,
                rss_before_post as f64 / 1024.0,
                rss_after_post as f64 / 1024.0,
                process_rss_kb() as f64 / 1024.0,
            );

            if std::env::var("NPNR_OT_DETERMINISM").ok().as_deref() == Some("1") {
                let mut h_pos: u64 = 0;
                for (x, y) in cell_x.iter().zip(cell_y.iter()) {
                    h_pos = h_pos
                        .rotate_left(5)
                        .wrapping_add(x.to_bits())
                        .rotate_left(7)
                        .wrapping_add(y.to_bits());
                }
                let mut h_usage: u64 = 0;
                for u in &post_solve.edge_usage {
                    h_usage = h_usage.rotate_left(5).wrapping_add(u.to_bits());
                }
                let mut h_cost: u64 = 0;
                for pipe_idx in 0..network.num_pipes() {
                    h_cost = h_cost
                        .rotate_left(5)
                        .wrapping_add(network.pipe_cost(pipe_idx).to_bits());
                }
                eprintln!(
                    "    determinism[outer={}]: pos={:016x} usage={:016x} cost={:016x} energy={:.17e}",
                    outer, h_pos, h_usage, h_cost, post_solve.energy,
                );
            }
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

        // Drain this iter's per-tile reject counts into the Lagrangian
        // pressure field. Next iter's `evaluate_cell_at` will see the
        // updated `dist_cache.tile_pressure`. We resize defensively in
        // case `n_nodes` changed under `ensure_shape` (shouldn't, but
        // guards against future net_count refreshes that may shrink/grow
        // the grid).
        if cfg.tile_pressure_weight > 0.0 {
            if dist_cache.tile_pressure.len() != n_nodes {
                dist_cache.tile_pressure.resize(n_nodes, 0.0);
            }
            let width = network.width;
            let drained = mux_tracker.drain_rejection_pressure(
                cfg.tile_pressure_decay,
                cfg.tile_pressure_step,
                &mut dist_cache.tile_pressure,
                |gx, gy| (gy as usize) * (width as usize) + (gx as usize),
            );
            let (mut sum, mut maxp, mut nonzero) = (0.0f64, 0.0f64, 0usize);
            for &p in &dist_cache.tile_pressure {
                if p > 0.0 {
                    sum += p;
                    nonzero += 1;
                    if p > maxp {
                        maxp = p;
                    }
                }
            }
            eprintln!(
                "    tile_pressure[outer={}]: drained_rejects={} nonzero_tiles={} max={:.3} mean_nz={:.3} weight={:.3} decay={:.3} step={:.3}",
                outer, drained, nonzero, maxp,
                if nonzero > 0 { sum / nonzero as f64 } else { 0.0 },
                cfg.tile_pressure_weight, cfg.tile_pressure_decay, cfg.tile_pressure_step,
            );
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
        // `beckmann` is the potential the iteration ought to be descending;
        // `base`/`max_util` are the congestion-free wirelength cost and the
        // real occupancy peak. None of the three contains the hardening term,
        // so they stay comparable across congestion arms — unlike
        // `cong_share`, which rises by construction once history is on.
        let beckmann = super::congestion::compute_beckmann_potential(network);
        let hist_total: f64 = network.pipe_history.iter().sum();
        eprintln!(
            "    E_decomp: base={:.1} eff={:.1} cong={:.1} cong_share={:.1}% sat_pipes={} max_util={:.2}  R_buckets: 1-2x={} 2-10x={} 10-100x={} >=100x={}",
            sum_base, sum_eff, sum_eff - sum_base, cong_share * 100.0, n_sat, max_util,
            r_lo, r_med, r_hi, r_sat,
        );
        eprintln!(
            "    E_potential: beckmann={:.1} history_total={:.1}",
            beckmann, hist_total,
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
        let probes: Vec<(i32, i32, f64)> =
            vec![(0, 0, 20.0), (10, 0, 5.0), (10, 10, 2.0), (0, 10, 2.0)];
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

#[cfg(test)]
mod hardening_tests {
    use super::*;
    use crate::placer::opt_trans::network::{
        hist_to_int, Direction, FlatAdjacency, Node, Pipe, PipeType, TileGrid, DIST_SCALE,
    };
    use crate::placer::opt_trans::tile_cache::{rebuild_span_cost_table, SpanCostTable};

    /// `n`-node horizontal chain with one pipe per adjacent pair. Every pipe
    /// shares tile type, span, capacity and base cost, so they all collapse
    /// into a single `SpanCostTable` bucket — which is exactly the condition
    /// under which a per-pipe term can silently vanish.
    fn chain(n: usize, capacity: f64, net_count: f64) -> PipeNetwork {
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node {
                tile_x: i as i32,
                tile_y: 0,
                pressure: 0.0,
            })
            .collect();
        let pipes: Vec<Pipe> = (0..n - 1)
            .map(|i| Pipe {
                from: i,
                to: i + 1,
                base_resistance: 1.0,
                capacity,
                flow: 0.0,
                net_count,
                raw_cell_density: 0.0,
                cell_density: 0.0,
                eff_conductance: 1.0,
                pipe_type: PipeType::InterTile(Direction::East),
            })
            .collect();
        let mut node_pipes = vec![Vec::new(); n];
        for (i, p) in pipes.iter().enumerate() {
            node_pipes[p.from].push(i);
            node_pipes[p.to].push(i);
        }
        let pipe_costs: Vec<f64> = pipes.iter().map(|p| p.base_resistance).collect();
        let pipe_costs_int: Vec<u32> = pipe_costs
            .iter()
            .map(|&c| ((c * DIST_SCALE).round() as u32).max(1))
            .collect();
        let n_pipes = pipes.len();
        let tile_grid = TileGrid::build(&pipes, &nodes, n as i32, 1);
        let flat_adjacency = FlatAdjacency::build(&node_pipes, &pipes);
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_history: vec![0.0; n_pipes],
            pipe_costs,
            pipe_costs_int,
            span_cost_table: SpanCostTable::disabled(n_pipes),
            flat_adjacency,
            tile_templates: std::sync::Arc::new(Vec::new()),
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n],
            width: n as i32,
            height: 1,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: 0,
            coarsen: 1,
        }
    }

    /// `eff_capacity` for a span-1 pipe: nominal × borrow_slack(1) = ×1.25.
    const SPAN1_SLACK: f64 = 1.25;

    #[test]
    fn history_grows_only_above_effective_capacity() {
        // eff_cap = 10 × 1.25 = 12.5. Raw usage below that must not register.
        let cases: &[(f64, bool)] = &[
            (0.0, false),
            (10.0, false),
            (12.5, false),
            (12.6, true),
            (25.0, true),
        ];
        for &(usage, expect_growth) in cases {
            let mut net = chain(2, 10.0, usage);
            update_pipe_history(&mut net, &[usage], 0.1);
            let h = net.pipe_history[0];
            assert_eq!(
                h > 0.0,
                expect_growth,
                "usage={} eff_cap={} gave h={}",
                usage,
                10.0 * SPAN1_SLACK,
                h
            );
        }
    }

    #[test]
    fn history_is_monotone_and_never_decays() {
        // Alternating overflow / no-overflow iters. History must be
        // non-decreasing throughout and strictly greater at the end than
        // after the first overflow — a decaying term would fail both.
        let mut net = chain(2, 10.0, 0.0);
        let schedule = [25.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0];
        let mut prev = 0.0;
        let mut after_first_overflow = None;
        for (i, &u) in schedule.iter().enumerate() {
            update_pipe_history(&mut net, &[u], 0.1);
            let h = net.pipe_history[0];
            assert!(h >= prev, "iter {}: history fell {} -> {}", i, prev, h);
            if i == 0 {
                after_first_overflow = Some(h);
            }
            prev = h;
        }
        let first = after_first_overflow.unwrap();
        assert!(first > 0.0);
        // Six quiet-or-overflow iters later it must have grown, not decayed.
        assert!(prev > first, "final {} did not exceed first {}", prev, first);
    }

    #[test]
    fn zero_step_disables_the_term_entirely() {
        let mut net = chain(2, 1.0, 100.0);
        update_pipe_history(&mut net, &[100.0], 0.0);
        assert_eq!(net.pipe_history[0], 0.0);
    }

    /// The load-bearing test: two pipes that are IDENTICAL in every
    /// `TileSpanCostKey` field share one `SpanCostTable` entry. If history
    /// were folded into the bucketed cost it would collide; it must instead
    /// survive as a per-pipe addend on the packed edge arrays that Dijkstra
    /// actually reads.
    #[test]
    fn history_survives_span_cost_table_bucketing() {
        for enable_table in [false, true] {
            let mut net = chain(3, 10.0, 4.0);
            net.pipe_history[0] = 0.7;
            net.pipe_history[1] = 0.0;
            if enable_table {
                rebuild_span_cost_table(&mut net, &ResistanceModel);
                // Precondition: both pipes really did land in one bucket.
                assert_eq!(
                    net.span_cost_table.pipe_entry[0], net.span_cost_table.pipe_entry[1],
                    "fixture no longer collides; the test has stopped testing anything"
                );
            }
            net.refresh_flat_edge_costs();

            let c0 = net.pipe_cost(0);
            let c1 = net.pipe_cost(1);
            assert!(
                (c0 - c1 - 0.7).abs() < 1e-9,
                "table={}: pipe_cost gap {} != 0.7",
                enable_table,
                c0 - c1
            );

            // And the same gap must appear in the packed per-edge arrays.
            let edge_of = |pidx: u32| {
                net.flat_adjacency
                    .pipe_idx
                    .iter()
                    .position(|&p| p == pidx)
                    .expect("pipe missing from flat adjacency")
            };
            let e0 = edge_of(0);
            let e1 = edge_of(1);
            let gap_int =
                net.flat_adjacency.cost_int[e0] as i64 - net.flat_adjacency.cost_int[e1] as i64;
            assert_eq!(
                gap_int,
                hist_to_int(0.7) as i64,
                "table={}: edge cost_int gap {} != {}",
                enable_table,
                gap_int,
                hist_to_int(0.7)
            );
        }
    }

    #[test]
    fn zero_history_adds_exactly_zero_cost() {
        // No `max(1)` floor on the history quantization — otherwise every
        // edge in the graph would silently gain a unit of cost.
        assert_eq!(hist_to_int(0.0), 0);
        assert_eq!(hist_to_int(-1.0), 0);
        assert_eq!(hist_to_int(1.0), DIST_SCALE as u32);
    }

    #[test]
    fn network_reset_clears_history() {
        let mut net = chain(2, 10.0, 0.0);
        update_pipe_history(&mut net, &[25.0], 0.1);
        assert!(net.pipe_history[0] > 0.0);
        net.reset();
        assert_eq!(net.pipe_history[0], 0.0);
    }
}

#[cfg(test)]
mod driver_hole_tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, FlatAdjacency, Node, Pipe, PipeType, TileGrid};
    use crate::placer::opt_trans::tile_cache::SpanCostTable;

    pub(super) const GRID_W: i32 = 8;
    const H: i32 = 1;

    pub(super) fn line_network() -> PipeNetwork {
        let n = (GRID_W * H) as usize;
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node { tile_x: i as i32, tile_y: 0, pressure: 0.0 })
            .collect();
        let pipes: Vec<Pipe> = (0..n - 1)
            .map(|i| Pipe {
                from: i,
                to: i + 1,
                base_resistance: 1.0,
                capacity: 10.0,
                flow: 0.0,
                net_count: 0.0,
                raw_cell_density: 0.0,
                cell_density: 0.0,
                eff_conductance: 1.0,
                pipe_type: PipeType::InterTile(Direction::East),
            })
            .collect();
        let mut node_pipes = vec![Vec::new(); n];
        for (i, p) in pipes.iter().enumerate() {
            node_pipes[p.from].push(i);
            node_pipes[p.to].push(i);
        }
        let n_pipes = pipes.len();
        let pipe_costs = vec![1.0; n_pipes];
        let pipe_costs_int = vec![100u32; n_pipes];
        let tile_grid = TileGrid::build(&pipes, &nodes, GRID_W, H);
        let flat_adjacency = FlatAdjacency::build(&node_pipes, &pipes);
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_history: vec![0.0; n_pipes],
            pipe_costs,
            pipe_costs_int,
            span_cost_table: SpanCostTable::disabled(n_pipes),
            flat_adjacency,
            tile_templates: std::sync::Arc::new(Vec::new()),
            tile_grid,
            pipe_lookup: FxHashMap::default(),
            tile_type_by_node: vec![0; n],
            width: GRID_W,
            height: H,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: n,
            coarsen: 1,
        }
    }

    /// One net: cell 0 drives, two fixed sinks parked at x=6 and x=7.
    pub(super) fn driver_net() -> Vec<NetInfo> {
        vec![NetInfo {
            net_id: NetId::from_raw(0),
            debug_name: Arc::from("n"),
            pins: vec![
                PinNode { node: 0, cell_idx: Some(0), is_driver: true, is_fixed: false },
                PinNode { node: 6, cell_idx: None, is_driver: false, is_fixed: true },
                PinNode { node: 7, cell_idx: None, is_driver: false, is_fixed: true },
            ],
            has_fixed_pin: true,
        }]
    }

    fn driver_cost_profile(mst_w: f64) -> Vec<f64> {
        let network = line_network();
        let net_infos = driver_net();
        let mut dist_cache = DistCache::new(1, network.num_nodes());
        // Driver-rooted labels exist but the driver's own cost never reads
        // them -- that is precisely the hole under test.
        for node in 0..network.num_nodes() {
            dist_cache.row_mut(0).insert(node as u32, node as f32);
        }
        let mut cfg = OptTransPlacerCfg::default();
        cfg.mst_edge_weight = mst_w;
        cfg.locked_pin_weight = 1.0;
        if mst_w > 0.0 {
            compute_mst_neighbors(&mut dist_cache, &net_infos, &network);
        }
        let validity = CellValidityMask::all_valid(1, GRID_W, H);
        let cell_nets = [(0usize, 0usize)];
        (0..GRID_W)
            .map(|gx| {
                evaluate_cell_at(0, gx, 0, &cell_nets, &net_infos, &dist_cache, &network, &cfg, &validity)
            })
            .collect()
    }

    /// THE DRIVER HOLE. `evaluate_cell_at` charges `weight * d` only for
    /// non-driver pins, so a cell's own fanout exerts zero pull on it: the
    /// driver's cost is identical at every legal position, and the argmin is
    /// whatever the tie-break happens to pick. This is the mechanism behind
    /// LUT->FF pairs only ever being tightened from the FF side.
    #[test]
    fn driver_cost_is_flat_without_a_tension_term() {
        let profile = driver_cost_profile(0.0);
        let first = profile[0];
        for (gx, &c) in profile.iter().enumerate() {
            assert!(
                (c - first).abs() < 1e-12,
                "driver cost at x={} is {}, expected flat {} -- hole is closed unexpectedly",
                gx,
                c,
                first
            );
        }
    }

    /// With the tension term on, the driver is pulled onto its MST
    /// neighbour. The MST over pins at x=0/6/7 is 0-6-7, so the driver's
    /// only tension edge is to the sink at x=6: cost falls monotonically up
    /// to x=6 and rises past it, a V with the minimum on the neighbour
    /// rather than a flat line with an arbitrary tie-break.
    #[test]
    fn tension_term_pulls_the_driver_onto_its_mst_neighbour() {
        let profile = driver_cost_profile(1.0);
        let argmin = profile
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmin, 6, "driver settled at x={}, profile {:?}", argmin, profile);
        for w in profile[..=argmin].windows(2) {
            assert!(w[1] < w[0], "not decreasing up to the neighbour: {:?}", profile);
        }
        for w in profile[argmin..].windows(2) {
            assert!(w[1] > w[0], "not increasing past the neighbour: {:?}", profile);
        }
    }

    /// The pull scales with the weight -- the term is a real lever, not a
    /// tie-break nudge.
    #[test]
    fn tension_gradient_scales_with_weight() {
        let spread = |w: f64| {
            let p = driver_cost_profile(w);
            p[0] - p[(GRID_W - 1) as usize]
        };
        let s1 = spread(1.0);
        let s4 = spread(4.0);
        assert!(s1 > 0.0, "no gradient at w=1");
        assert!(
            (s4 / s1 - 4.0).abs() < 1e-9,
            "spread {} at w=4 vs {} at w=1 is not 4x",
            s4,
            s1
        );
    }
}

#[cfg(test)]
mod proximal_tests {
    use super::driver_hole_tests::{driver_net, line_network, GRID_W};
    use super::*;

    /// Cost of placing the (single) movable cell at each x, with the net's
    /// other pins fixed far away, for a given proximal weight.
    fn profile(prox_w: f64) -> Vec<f64> {
        let network = line_network();
        let net_infos = driver_net();
        let mut dist_cache = DistCache::new(1, network.num_nodes());
        for node in 0..network.num_nodes() {
            dist_cache.row_mut(0).insert(node as u32, node as f32);
        }
        let mut cfg = OptTransPlacerCfg::default();
        cfg.locked_pin_weight = 1.0;
        cfg.proximal_weight = prox_w;
        let validity = CellValidityMask::all_valid(1, GRID_W, 1);
        let cell_nets = [(0usize, 0usize)];
        (0..GRID_W)
            .map(|gx| {
                evaluate_cell_at(
                    0, gx, 0, &cell_nets, &net_infos, &dist_cache, &network, &cfg, &validity,
                )
            })
            .collect()
    }

    /// The cell sits at node 0. Staying put must cost exactly what it cost
    /// before the term existed — otherwise mu shifts the whole surface
    /// instead of only pricing movement.
    #[test]
    fn proximal_is_zero_at_the_current_position() {
        let off = profile(0.0);
        let on = profile(2.5);
        assert!(
            (on[0] - off[0]).abs() < 1e-12,
            "staying put changed cost: {} -> {}",
            off[0],
            on[0]
        );
    }

    /// Away from the current tile the charge is exactly `mu * manhattan`.
    #[test]
    fn proximal_charges_exactly_mu_times_displacement() {
        for mu in [0.5, 2.0, 10.0] {
            let off = profile(0.0);
            let on = profile(mu);
            for gx in 0..GRID_W as usize {
                let expected = mu * gx as f64;
                let charged = on[gx] - off[gx];
                assert!(
                    (charged - expected).abs() < 1e-9,
                    "mu={} at x={}: charged {} expected {}",
                    mu,
                    gx,
                    charged,
                    expected
                );
            }
        }
    }

    /// A large enough mu pins the cell in place — the trust region has shrunk
    /// to a single tile. This is what makes it a damping knob.
    #[test]
    fn large_mu_pins_the_cell_where_it_is() {
        let p = profile(1000.0);
        let argmin = p
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmin, 0, "cell moved despite a huge proximal weight: {:?}", p);
    }
}

//! Per-net routability legality check.
//!
//! After legalization, every placed cell's BEL pin wires are claimed by its
//! net (via `DriverNodeRegistry`). This module asks the harder question:
//! given those claims, can the wire graph still deliver each net's signal
//! from driver to all sinks via *some* path that avoids other nets' pin
//! wires? If not, the placement is infeasible — no router can route it,
//! regardless of budget or heuristic.
//!
//! Two complementary passes share `PinOwnership` (every placed cell's BEL
//! pin wires + node peers, mapped to the cell's net):
//!
//! - [`check_routability_global`]: one multi-target A* per net inside its
//!   union bbox, finding every sink in a single traversal and terminating
//!   as soon as the last sink is reached. Fast filter for typical-fanout
//!   nets (low-hundreds of milliseconds for sv3).
//! - [`check_routability`]: per-(driver, sink) manhattan-guided A* with a
//!   tight per-pair bbox. Slower but more precise — confirms specific
//!   sink/driver pairs that the global pass either flagged or could not
//!   conclude on.
//!
//! Both skip the packer constant nets `$PACKER_GND_NET`/`$PACKER_VCC_NET`
//! (general fabric is the wrong test for them; every slice has local
//! tieoffs), and both report sinks as `unreached` only when a search
//! definitively drained — visit-budget exits go to `n_inconclusive` so
//! the placer can't react to noise. Clock nets are *not* filtered: there
//! is no structural way to identify them from the netlist alone, so they
//! either drain through fabric or land in the inconclusive bucket.
//!
//! The global pass is *not* a Dijkstra over the kernel A* (`router::astar`):
//! the kernel's bbox filter only constrains pip destinations, so a single
//! mega-node (e.g. a 65 K-peer clock distribution wire) can pull tens of
//! thousands of out-of-bbox peers into the search and burn the visit
//! budget. The local multi-target search in this module bbox-filters peer
//! expansion itself, keeping the working set bounded by the bbox area.

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::netlist::{CellPin, NetId};
use crate::router::astar::{astar_search, AStarExit, AStarOptions, PathCostModel};
use crate::timing::DelayT;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct GlobalEntry {
    wire: WireId,
    cost: i32,
    estimate: i32,
}
impl PartialEq for GlobalEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}
impl Eq for GlobalEntry {}
impl PartialOrd for GlobalEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GlobalEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on estimate, tiebreak on lower cost.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
    }
}

/// Halo (in tiles) around the (driver, sink) bbox for the per-pair search.
/// Grows with manhattan distance so long nets get proportionally more
/// detour room without ballooning the search space for short ones.
const PERPAIR_HALO_FLOOR: i32 = 12;
/// Halo around the union bbox for the global per-net pass. Smaller because
/// the union bbox is already wider than any per-pair bbox.
const GLOBAL_HALO: i32 = 8;
/// Per-search visit floor. Small nets are dominated by graph branching, not
/// bbox area, so they get a flat budget regardless of bbox.
const PERPAIR_VISIT_FLOOR: usize = 50_000;
const GLOBAL_VISIT_FLOOR: usize = 30_000;
/// Visit budget per unit of bbox area for the per-pair search. A* with a
/// manhattan heuristic focuses toward the goal so a small multiplier is
/// enough; `bbox_area * 4` covers typical 30×30 bboxes without ballooning.
const VISIT_PER_TILE_PERPAIR: usize = 4;
/// Visit budget per unit of bbox area for the global multi-target search.
/// Multi-target A* with bbox-filtered peer expansion keeps the working set
/// bounded, but each tile holds dozens of wires, so a healthy multiplier
/// is needed to actually drain the bbox.
const VISIT_PER_TILE_GLOBAL: usize = 64;
/// Largest union-bbox area (in tiles) the global pass will attempt.
/// Past this size, draining is too expensive per net even with the
/// bbox-filtered peer expansion — fall back to inconclusive and let the
/// per-pair pass handle the net (its tight per-sink bbox stays small).
const MAX_GLOBAL_BBOX_TILES: usize = 25_000;
/// Max peers walked per node-equivalence expansion when registering a pin
/// wire's ownership. Mirrors `legalize::common::NODE_PEER_CAP`.
const NODE_PEER_CAP: usize = 4096;

#[derive(Debug, Clone)]
pub struct UnreachedSink {
    pub user_index: usize,
    pub sink_wire: WireId,
    pub exit: AStarExit,
    pub visited: usize,
}

#[derive(Debug, Clone)]
pub struct InfeasibleNet {
    pub net_id: NetId,
    pub driver_wire: WireId,
    pub unreached: Vec<UnreachedSink>,
}

#[derive(Debug, Default)]
pub struct RoutabilityReport {
    pub infeasible: Vec<InfeasibleNet>,
    pub elapsed_ms: f64,
    pub n_checked: usize,
    /// Number of (driver, sink) searches performed (per-pair mode only).
    pub n_pairs: usize,
    /// Nets filtered out by [`should_skip_net`] (GND/VCC/clocks).
    pub n_skipped: usize,
    /// Nets where the search hit its visit budget before terminating.
    /// Their sinks are not counted as infeasible — the result is unknown.
    pub n_inconclusive: usize,
}

impl RoutabilityReport {
    pub fn is_feasible(&self) -> bool {
        self.infeasible.is_empty()
    }
}

/// Skip the packer-emitted constant nets — they don't route through
/// general fabric (every slice has local switch-matrix tieoffs), so a
/// fabric reachability test would only produce false positives. The names
/// are reserved by the packer at `frontend/parser.rs`, so this match is
/// reliable. Clock nets cannot be identified from the netlist alone, so
/// they are *not* filtered here; the inconclusive bucket absorbs any
/// global-clock-routed net the search can't drain.
fn should_skip_net(ctx: &Context, net_id: NetId) -> bool {
    let net_name = ctx.name_of(ctx.design.net(net_id).name);
    net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET"
}

/// Pin-wire ownership: every BEL pin wire (and its node peers) of a placed
/// cell mapped to the wire's owning net.
pub struct PinOwnership {
    map: FxHashMap<WireId, NetId>,
}

impl PinOwnership {
    pub fn build(ctx: &Context) -> Self {
        let mut map: FxHashMap<WireId, NetId> = FxHashMap::default();
        let bound: Vec<(crate::netlist::CellId, crate::chipdb::BelId)> = ctx
            .design
            .iter_alive_cells()
            .filter_map(|(cid, cell)| cell.bel.map(|b| (cid, b)))
            .collect();
        for (cid, bel) in bound {
            for (w, net) in crate::placer::legalize::common::cell_pin_wires_pub(ctx, cid, bel) {
                Self::insert_with_peers(ctx, w, net, &mut map);
            }
        }
        Self { map }
    }

    fn insert_with_peers(ctx: &Context, w: WireId, net: NetId, map: &mut FxHashMap<WireId, NetId>) {
        map.insert(w, net);
        let mut peers = 0usize;
        ctx.chipdb().node_wires_cb(w, |nw| {
            if peers >= NODE_PEER_CAP {
                peers += 1;
                return;
            }
            peers += 1;
            map.insert(nw, net);
        });
    }

    #[inline]
    pub fn owner(&self, w: WireId) -> Option<NetId> {
        self.map.get(&w).copied()
    }
}

/// Cost model used by both passes. With `use_heuristic = true` it drives a
/// manhattan-guided A* (per-pair); with `false` the heuristic is zero so the
/// kernel runs in Dijkstra mode (global).
struct ReachabilityModel<'a> {
    ownership: &'a PinOwnership,
    own_net: NetId,
    bbox: [BoundingBox; 1],
    use_heuristic: bool,
}

impl<'a> PathCostModel for ReachabilityModel<'a> {
    #[inline]
    fn pip_cost(&self, _ctx: &Context, _pip: crate::chipdb::PipId) -> DelayT {
        1
    }

    #[inline]
    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        if !self.use_heuristic {
            return 0;
        }
        let chipdb = ctx.chipdb();
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        let (dx, dy) = chipdb.tile_xy(dst.tile());
        ((wx - dx).abs() + (wy - dy).abs()) as DelayT
    }

    #[inline]
    fn is_blocked(&self, _ctx: &Context, wire: WireId) -> bool {
        match self.ownership.owner(wire) {
            Some(owner) => owner != self.own_net,
            None => false,
        }
    }

    #[inline]
    fn bboxes(&self) -> &[BoundingBox] {
        &self.bbox
    }
}

#[inline]
fn bbox_visit_limit(
    xmin: i32,
    xmax: i32,
    ymin: i32,
    ymax: i32,
    per_tile: usize,
    floor: usize,
) -> usize {
    let bbox_w = (xmax - xmin + 1).max(1) as usize;
    let bbox_h = (ymax - ymin + 1).max(1) as usize;
    bbox_w
        .saturating_mul(bbox_h)
        .saturating_mul(per_tile)
        .max(floor)
}

fn collect_checkable_nets(ctx: &Context) -> (Vec<NetId>, usize) {
    let mut net_ids: Vec<NetId> = Vec::new();
    let mut n_skipped = 0usize;
    for (id, net) in ctx.design.iter_alive_nets() {
        if !net.has_driver() || net.num_users() == 0 {
            continue;
        }
        if should_skip_net(ctx, id) {
            n_skipped += 1;
            continue;
        }
        net_ids.push(id);
    }
    (net_ids, n_skipped)
}

/// Per-(driver, sink) A* feasibility check. Slower but more precise: each
/// search uses a tight (driver, sink) bbox and a manhattan heuristic.
///
/// A sink is reported as `unreached` only when the search drained the heap
/// without finding it — `VisitLimit` exits are inconclusive (budget too
/// small) and contribute to `n_inconclusive` instead. This avoids false
/// positives that would otherwise be amplified by every iteration of the
/// surrounding placer loop.
pub fn check_routability(ctx: &Context) -> RoutabilityReport {
    let t = std::time::Instant::now();
    let ownership = PinOwnership::build(ctx);
    let (net_ids, n_skipped) = collect_checkable_nets(ctx);

    let n_checked = net_ids.len();
    let pair_counter = std::sync::atomic::AtomicUsize::new(0);
    let inconclusive_counter = std::sync::atomic::AtomicUsize::new(0);

    let infeasible: Vec<InfeasibleNet> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            check_net_perpair(
                ctx,
                net_id,
                &ownership,
                &pair_counter,
                &inconclusive_counter,
            )
        })
        .collect();

    RoutabilityReport {
        infeasible,
        elapsed_ms: t.elapsed().as_secs_f64() * 1000.0,
        n_checked,
        n_pairs: pair_counter.load(std::sync::atomic::Ordering::Relaxed),
        n_skipped,
        n_inconclusive: inconclusive_counter.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Per-net Dijkstra feasibility check. One exhaustive search per net within
/// the union bbox of (driver, all sinks); every reached wire lands in the
/// visited map and each sink is checked against it.
///
/// Only nets whose Dijkstra drained the heap contribute to `infeasible` —
/// nets that hit the visit budget are recorded in `n_inconclusive` instead.
/// This keeps the global pass usable as a coarse filter without firing on
/// any large-bbox high-fanout net the budget can't fully explore.
pub fn check_routability_global(ctx: &Context) -> RoutabilityReport {
    let t = std::time::Instant::now();
    let ownership = PinOwnership::build(ctx);
    let (net_ids, n_skipped) = collect_checkable_nets(ctx);

    let n_checked = net_ids.len();
    let inconclusive_counter = std::sync::atomic::AtomicUsize::new(0);

    let infeasible: Vec<InfeasibleNet> = net_ids
        .par_iter()
        .filter_map(|&net_id| check_net_global(ctx, net_id, &ownership, &inconclusive_counter))
        .collect();

    RoutabilityReport {
        infeasible,
        elapsed_ms: t.elapsed().as_secs_f64() * 1000.0,
        n_checked,
        n_pairs: 0,
        n_skipped,
        n_inconclusive: inconclusive_counter.load(std::sync::atomic::Ordering::Relaxed),
    }
}

fn cell_pin_wire(ctx: &Context, pin: CellPin) -> Option<WireId> {
    let cell = ctx.cell(pin.cell);
    let bel = cell.bel()?;
    bel.pin_wire(pin.port).map(|w| w.id())
}

fn collect_sinks(ctx: &Context, net_id: NetId) -> (Option<WireId>, Vec<(WireId, usize)>) {
    let net = ctx.net(net_id);
    let driver = match net.driver() {
        Some(d) => d,
        None => return (None, Vec::new()),
    };
    let src_wire = match cell_pin_wire(ctx, driver) {
        Some(w) => w,
        None => return (None, Vec::new()),
    };

    let mut sinks: Vec<(WireId, usize)> = Vec::new();
    for (i, user) in net.users().iter().enumerate() {
        if !user.is_valid() {
            continue;
        }
        if let Some(w) = cell_pin_wire(ctx, *user) {
            sinks.push((w, i));
        }
    }
    (Some(src_wire), sinks)
}

fn check_net_perpair(
    ctx: &Context,
    net_id: NetId,
    ownership: &PinOwnership,
    pair_counter: &std::sync::atomic::AtomicUsize,
    inconclusive_counter: &std::sync::atomic::AtomicUsize,
) -> Option<InfeasibleNet> {
    let (src_wire, sinks) = collect_sinks(ctx, net_id);
    let src_wire = src_wire?;
    if sinks.is_empty() {
        return None;
    }

    let chipdb = ctx.chipdb();
    let (sx, sy) = chipdb.tile_xy(src_wire.tile());
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src_wire);

    let mut unreached: Vec<UnreachedSink> = Vec::new();
    let mut had_inconclusive = false;
    for &(sink_wire, user_idx) in &sinks {
        pair_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let (tx, ty) = chipdb.tile_xy(sink_wire.tile());
        let manhattan = ((sx - tx).abs() + (sy - ty).abs()) as i32;
        // Halo grows with manhattan so long routes get detour room without
        // bloating the search for short ones.
        let halo = PERPAIR_HALO_FLOOR.max(manhattan / 16);
        let xmin = sx.min(tx) - halo;
        let xmax = sx.max(tx) + halo;
        let ymin = sy.min(ty) - halo;
        let ymax = sy.max(ty) + halo;
        let visit_limit = bbox_visit_limit(
            xmin,
            xmax,
            ymin,
            ymax,
            VISIT_PER_TILE_PERPAIR,
            PERPAIR_VISIT_FLOOR,
        );

        let model = ReachabilityModel {
            ownership,
            own_net: net_id,
            bbox: [BoundingBox {
                x0: xmin,
                y0: ymin,
                x1: xmax,
                y1: ymax,
            }],
            use_heuristic: true,
        };

        let opts = AStarOptions {
            visit_limit: Some(visit_limit),
            exhaustive: false,
            retain_trace: false,
            stop_on_first_touch: true,
        };

        let res = astar_search(ctx, &model, &src_set, sink_wire, &opts);
        match res.trace.exit {
            AStarExit::Reached => {}
            AStarExit::HeapDrained => {
                // Definitive: the search exhausted reachable wires within
                // the bbox without finding the sink.
                unreached.push(UnreachedSink {
                    user_index: user_idx,
                    sink_wire,
                    exit: res.trace.exit,
                    visited: res.trace.visit_count,
                });
            }
            AStarExit::VisitLimit => {
                // Inconclusive: budget hit before the search drained.
                had_inconclusive = true;
            }
        }
    }

    if had_inconclusive {
        inconclusive_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    if unreached.is_empty() {
        None
    } else {
        Some(InfeasibleNet {
            net_id,
            driver_wire: src_wire,
            unreached,
        })
    }
}

fn check_net_global(
    ctx: &Context,
    net_id: NetId,
    ownership: &PinOwnership,
    inconclusive_counter: &std::sync::atomic::AtomicUsize,
) -> Option<InfeasibleNet> {
    let (src_wire, sinks) = collect_sinks(ctx, net_id);
    let src_wire = src_wire?;
    if sinks.is_empty() {
        return None;
    }

    let chipdb = ctx.chipdb();
    let (sx, sy) = chipdb.tile_xy(src_wire.tile());
    let mut xmin = sx;
    let mut xmax = sx;
    let mut ymin = sy;
    let mut ymax = sy;
    for &(sink_wire, _) in &sinks {
        let (tx, ty) = chipdb.tile_xy(sink_wire.tile());
        xmin = xmin.min(tx);
        xmax = xmax.max(tx);
        ymin = ymin.min(ty);
        ymax = ymax.max(ty);
    }
    xmin -= GLOBAL_HALO;
    xmax += GLOBAL_HALO;
    ymin -= GLOBAL_HALO;
    ymax += GLOBAL_HALO;

    let bbox_w = (xmax - xmin + 1).max(1) as usize;
    let bbox_h = (ymax - ymin + 1).max(1) as usize;
    let bbox_area = bbox_w.saturating_mul(bbox_h);
    if bbox_area > MAX_GLOBAL_BBOX_TILES {
        inconclusive_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }

    let visit_limit = bbox_visit_limit(
        xmin,
        xmax,
        ymin,
        ymax,
        VISIT_PER_TILE_GLOBAL,
        GLOBAL_VISIT_FLOOR,
    );
    let bbox = BoundingBox {
        x0: xmin,
        y0: ymin,
        x1: xmax,
        y1: ymax,
    };

    let result = multi_target_search(ctx, src_wire, &sinks, &bbox, ownership, net_id, visit_limit);

    if std::env::var("NPNR_OT_GLOBAL_BBOX_HIST").ok().as_deref() == Some("1") {
        eprintln!(
            "global-result: net={:?} area={} reached={}/{} visited={} drained={}",
            net_id,
            bbox_area,
            result.reached_count,
            sinks.len(),
            result.visit_count,
            result.drained,
        );
    }

    // A search that hit the visit limit is inconclusive — we don't know
    // whether the missing sinks are unreachable or just beyond the budget.
    if !result.drained {
        inconclusive_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }

    let mut unreached: Vec<UnreachedSink> = Vec::new();
    for &(sink_wire, user_idx) in &sinks {
        if !result.reached.contains(&sink_wire) {
            unreached.push(UnreachedSink {
                user_index: user_idx,
                sink_wire,
                exit: AStarExit::HeapDrained,
                visited: result.visit_count,
            });
        }
    }

    if unreached.is_empty() {
        None
    } else {
        Some(InfeasibleNet {
            net_id,
            driver_wire: src_wire,
            unreached,
        })
    }
}

struct MultiSearchResult {
    reached: FxHashSet<WireId>,
    reached_count: usize,
    visit_count: usize,
    drained: bool,
}

/// Multi-target A* with early termination. Searches from `src` for every
/// sink in `sinks`, terminating as soon as all sinks are reached, the heap
/// drains, or `visit_limit` pops are made.
///
/// The bbox filter is applied to *both* pip destinations and node-equivalent
/// peers. The kernel A* only filters pip destinations, which lets a single
/// chip-spanning node (e.g. a clock distribution wire) pull tens of
/// thousands of out-of-bbox peers into the search space and burn the
/// budget on them. Filtering peers here keeps the working set bounded by
/// the bbox area.
///
/// Heuristic: min manhattan tile-distance from the current wire to any
/// remaining sink. Admissible (each pip costs >= 1; manhattan is a lower
/// bound on the number of pip hops to any sink).
fn multi_target_search(
    ctx: &Context,
    src: WireId,
    sinks: &[(WireId, usize)],
    bbox: &BoundingBox,
    ownership: &PinOwnership,
    own_net: NetId,
    visit_limit: usize,
) -> MultiSearchResult {
    let chipdb = ctx.chipdb();

    let mut remaining: FxHashSet<WireId> = sinks.iter().map(|&(w, _)| w).collect();
    let mut reached: FxHashSet<WireId> = FxHashSet::default();
    let mut best_cost: FxHashMap<WireId, i32> = FxHashMap::default();
    let mut nodes_expanded: FxHashSet<u64> = FxHashSet::default();
    let mut heap: BinaryHeap<GlobalEntry> = BinaryHeap::new();

    // Cache sink (x, y) for quick heuristic recomputation.
    let sink_xy: Vec<(WireId, i32, i32)> = sinks
        .iter()
        .map(|&(w, _)| {
            let (x, y) = chipdb.tile_xy(w.tile());
            (w, x, y)
        })
        .collect();

    let heuristic = |wire: WireId, remaining: &FxHashSet<WireId>| -> i32 {
        if remaining.is_empty() {
            return 0;
        }
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        sink_xy
            .iter()
            .filter(|(s, _, _)| remaining.contains(s))
            .map(|(_, sx, sy)| (wx - sx).abs() + (wy - sy).abs())
            .min()
            .unwrap_or(0)
    };

    // Mark `src` as reached if it's also a sink (some BEL pin wires can be
    // shared between driver and sink positions on the same node).
    let mut mark_reached =
        |wire: WireId, remaining: &mut FxHashSet<WireId>, reached: &mut FxHashSet<WireId>| {
            if remaining.remove(&wire) {
                reached.insert(wire);
            }
        };

    best_cost.insert(src, 0);
    mark_reached(src, &mut remaining, &mut reached);
    let h0 = heuristic(src, &remaining);
    heap.push(GlobalEntry {
        wire: src,
        cost: 0,
        estimate: h0,
    });

    let mut visit_count = 0usize;
    let mut drained = true;

    while let Some(GlobalEntry { wire, cost, .. }) = heap.pop() {
        visit_count += 1;
        if visit_count > visit_limit {
            drained = false;
            break;
        }
        if remaining.is_empty() {
            break;
        }

        // Stale skip: a cheaper path already explored this wire.
        if let Some(&prev) = best_cost.get(&wire) {
            if cost > prev {
                continue;
            }
        }

        // Node-as-vertex dedup: each node is expanded once.
        let nid = chipdb.node_id(wire);
        let should_expand = match nid {
            Some(id) => nodes_expanded.insert(id),
            None => true,
        };
        if !should_expand {
            continue;
        }

        // Collect node members (popped wire + its peers, bbox-filtered).
        // Peers outside the bbox are ignored entirely — neither marked as
        // reached nor used as a launching point for further pips.
        let mut members: Vec<WireId> = Vec::with_capacity(8);
        members.push(wire);
        if nid.is_some() {
            chipdb.node_wires_cb(wire, |peer| {
                let (px, py) = chipdb.tile_xy(peer.tile());
                if bbox.contains(px, py) {
                    members.push(peer);
                }
            });
        }

        for &member in &members {
            // Peers inherit the popped wire's cost (free node hop).
            if member != wire {
                let prev = best_cost.get(&member).copied().unwrap_or(i32::MAX);
                if cost < prev {
                    best_cost.insert(member, cost);
                }
            }
            // Mark sinks reached — applies to the popped wire too.
            mark_reached(member, &mut remaining, &mut reached);
            if remaining.is_empty() {
                break;
            }

            // Expand pips downhill from this member.
            let info = chipdb.wire_info(member);
            for &pip_idx in info.pips_downhill.get() {
                let pip = PipId::new(member.tile(), pip_idx);
                let next = chipdb.pip_dst_wire(pip);
                let (nx, ny) = chipdb.tile_xy(next.tile());
                if !bbox.contains(nx, ny) {
                    continue;
                }
                if let Some(owner) = ownership.owner(next) {
                    if owner != own_net {
                        continue;
                    }
                }
                let new_cost = cost + 1;
                let prev = best_cost.get(&next).copied().unwrap_or(i32::MAX);
                if new_cost >= prev {
                    continue;
                }
                best_cost.insert(next, new_cost);
                let h = heuristic(next, &remaining);
                heap.push(GlobalEntry {
                    wire: next,
                    cost: new_cost,
                    estimate: new_cost + h,
                });
            }
        }
    }

    let reached_count = reached.len();
    MultiSearchResult {
        reached,
        reached_count,
        visit_count,
        drained,
    }
}

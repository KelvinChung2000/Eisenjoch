//! Router3: Gupta-Sproull raster router with beam-search PIP walking.
//!
//! Uses the Gupta-Sproull anti-aliased line drawing algorithm to establish a
//! tile corridor, then walks the actual PIP graph with a greedy beam search
//! scored by timing (progress toward sink / PIP delay). Long wires naturally
//! win because they cover many tiles for one PIP traversal.
//!
//! Iterates with congestion feedback. Intended for fast wirelength estimation.

use std::sync::Arc;

use crate::chipdb::{PipId, WireId};
use crate::context::Context;
use crate::metrics::BoundingBox;
use crate::netlist::NetId;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use super::common::{
    apply_route_plan, collect_routable_nets, collect_sink_wires, find_local_const_pip,
    is_global_clock_wire, resolve_source_wire, source_wire_const_value,
    unroute_net, validate_all_routed, RoutePlan, SinkRoute,
};
use super::maze::{astar_route, astar_route_multihop};
use super::RouterError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RasterRouterCfg {
    pub max_iterations: usize,
    pub initial_cong_weight: f32,
    pub cong_growth: f32,
    pub filter_radius: f32,
    pub beam_width: usize,
    pub max_beam_steps: usize,
    /// If true, use pure greedy line-walk instead of beam search.
    pub use_greedy: bool,
    pub verbose: bool,
}

impl Default for RasterRouterCfg {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            initial_cong_weight: 0.05,
            cong_growth: 1.1,
            filter_radius: 1.0,
            beam_width: 128,
            max_beam_steps: 1000,
            use_greedy: false,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CongestionMap
// ---------------------------------------------------------------------------

struct CongestionMap {
    usage: Vec<f32>,
    history: Vec<f32>,
    #[allow(dead_code)]
    width: i32,
    #[allow(dead_code)]
    height: i32,
}

impl CongestionMap {
    fn new(width: i32, height: i32) -> Self {
        let n = (width * height) as usize;
        Self {
            usage: vec![0.0; n],
            history: vec![0.0; n],
            width,
            height,
        }
    }

    fn reset_usage(&mut self) {
        self.usage.fill(0.0);
    }

    fn add_usage(&mut self, tile: i32, amount: f32) {
        self.usage[tile as usize] += amount;
    }

    fn update_history(&mut self) {
        for i in 0..self.usage.len() {
            let excess = self.usage[i] - 1.0;
            if excess > 0.0 {
                // Blend: keep 50% of old history + add new excess.
                self.history[i] = self.history[i] * 0.5 + excess;
            } else {
                // Decay history on non-congested tiles.
                self.history[i] *= 0.5;
            }
        }
    }

    fn congestion_at(&self, tile: i32) -> f32 {
        // History only - usage is reset each pass.
        self.history[tile as usize]
    }

    fn max_usage(&self) -> f32 {
        self.usage.iter().cloned().fold(0.0f32, f32::max)
    }

    fn num_congested_tiles(&self) -> usize {
        self.usage.iter().filter(|&&u| u > 1.0).count()
    }
}

// ---------------------------------------------------------------------------
// Gupta-Sproull rasterizer
// ---------------------------------------------------------------------------

/// Build a set of tiles that form the routing corridor via Gupta-Sproull
/// anti-aliased line drawing. Returns a HashSet for O(1) corridor membership
/// testing, plus the ordered path for direction guidance.
fn raster_corridor(
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    cong: &CongestionMap,
    cong_weight: f32,
    filter_radius: f32,
    chip_width: i32,
    chip_height: i32,
) -> (Vec<(i32, i32)>, FxHashSet<i32>) {
    let mut path = Vec::new();
    let mut corridor = FxHashSet::default();

    if x0 == x1 && y0 == y1 {
        path.push((x0, y0));
        corridor.insert(y0 * chip_width + x0);
        return (path, corridor);
    }

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let steep = dy > dx;

    let (mut px, mut py, end_x, _end_y, adx, ady) = if steep {
        (y0, x0, y1, x1, dy, dx)
    } else {
        (x0, y0, x1, y1, dx, dy)
    };

    let step_x: i32 = if end_x > px { 1 } else { -1 };
    let step_y: i32 = if _end_y > py { 1 } else { -1 };

    let mut d = 2 * ady - adx;
    let mut two_v_dx: i32;
    let inv_denom = 1.0 / (2.0 * ((adx * adx + ady * ady) as f32).sqrt());
    let two_dx_inv_denom = 2.0 * adx as f32 * inv_denom;

    let to_tile = |major: i32, minor: i32| -> (i32, i32) {
        if steep {
            (minor, major)
        } else {
            (major, minor)
        }
    };

    let tile_idx = |x: i32, y: i32| -> i32 { y * chip_width + x };

    let eval = |major: i32, minor: i32, dist: f32| -> f32 {
        let (cx, cy) = to_tile(major, minor);
        if cx < 0 || cx >= chip_width || cy < 0 || cy >= chip_height {
            return f32::NEG_INFINITY;
        }
        let coverage = (1.0 - dist / filter_radius).max(0.0);
        coverage - cong_weight * cong.congestion_at(tile_idx(cx, cy))
    };

    // Add all three candidate pixels to corridor (wider corridor for beam search).
    let mut add_pixel = |major: i32, minor: i32| {
        let (cx, cy) = to_tile(major, minor);
        if cx >= 0 && cx < chip_width && cy >= 0 && cy < chip_height {
            corridor.insert(tile_idx(cx, cy));
        }
    };

    // First pixel.
    let (fx, fy) = to_tile(px, py);
    path.push((fx, fy));
    add_pixel(px, py);
    add_pixel(px, py + 1);
    add_pixel(px, py - 1);

    for _ in 0..adx {
        px += step_x;
        if d >= 0 {
            py += step_y;
            d -= 2 * adx;
            two_v_dx = d + adx;
        } else {
            two_v_dx = d + adx;
        }
        d += 2 * ady;

        let dist_main = (two_v_dx as f32 * inv_denom).abs();
        let dist_plus = two_dx_inv_denom - dist_main;
        let dist_minus = two_dx_inv_denom + dist_main;

        let p_main = eval(px, py, dist_main);
        let p_plus = eval(px, py + step_y, dist_plus);
        let p_minus = eval(px, py - step_y, dist_minus);

        let chosen_minor = if p_main >= p_plus && p_main >= p_minus {
            py
        } else if p_plus > p_minus {
            py + step_y
        } else {
            py - step_y
        };

        let (cx, cy) = to_tile(px, chosen_minor);
        if let Some(&last) = path.last() {
            if last != (cx, cy) {
                path.push((cx, cy));
            }
        } else {
            path.push((cx, cy));
        }

        // Add all three candidates to the corridor for beam width.
        add_pixel(px, py);
        add_pixel(px, py + step_y);
        add_pixel(px, py - step_y);

        py = chosen_minor;
    }

    // Ensure endpoint.
    let (ex, ey) = to_tile(end_x, _end_y);
    if path.last() != Some(&(ex, ey)) {
        path.push((ex, ey));
    }
    corridor.insert(tile_idx(ex, ey));

    (path, corridor)
}

// ---------------------------------------------------------------------------
// Beam search PIP walker
// ---------------------------------------------------------------------------

/// A candidate in the beam search. `best_dist` caches the node-min
/// manhattan distance at the time the candidate was created so the
/// stall-detection loop doesn't have to recompute it.
#[derive(Clone)]
struct BeamCandidate {
    wire: WireId,
    path_idx: u32, // index into path_arena, u32::MAX = root (no pips)
    cost: f32,
    best_dist: i32,
}

/// Reconstruct the PIP path by walking the arena chain from leaf to root.
fn reconstruct_path(arena: &[(PipId, u32)], mut idx: u32) -> Vec<PipId> {
    let mut pips = Vec::new();
    while idx != u32::MAX {
        let (pip, parent) = arena[idx as usize];
        pips.push(pip);
        idx = parent;
    }
    pips.reverse();
    pips
}

/// Greedy beam search from source wires to a destination wire.
///
/// At each step, expands `pips_downhill` from all beam candidates. Each PIP
/// is scored by remaining manhattan distance to the sink (lower = better).
/// Node-aware: considers the closest tile reachable via the wire's routing
/// node. Long wires naturally win because their node members reach tiles
/// close to the sink for a single PIP traversal.
fn beam_search_route(
    ctx: &Context,
    net: NetId,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    corridor: &FxHashSet<i32>,
    cong: &CongestionMap,
    cong_weight: f32,
    beam_width: usize,
    max_steps: usize,
) -> Option<Vec<PipId>> {
    let chipdb = ctx.chipdb();
    let (dst_x, dst_y) = chipdb.tile_xy(dst_wire.tile());

    // Build destination node set for hit detection.
    let mut dst_nodes: FxHashSet<WireId> = FxHashSet::default();
    dst_nodes.insert(dst_wire);
    chipdb.node_wires_cb(dst_wire, |nw| {
        dst_nodes.insert(nw);
    });

    // Check trivial case.
    for src in src_wires {
        if dst_nodes.contains(src) {
            return Some(Vec::new());
        }
    }

    let manhattan = |wire: WireId| -> i32 {
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        (wx - dst_x).abs() + (wy - dst_y).abs()
    };

    // Per-node memoization. Every quantity computed from `node_wires_cb`
    // in the beam loop is a property of the *node*, not of the individual
    // wire — the minimum manhattan over peers, whether any peer is
    // owned by a different net, whether any peer hits `dst_nodes`, and
    // which peers have outgoing PIPs. Walking a 65k-peer node once per
    // wire would turn every per-wire operation into O(peers²); caching
    // by node id collapses repeat lookups to O(1) after the first walk.
    let mut node_taken_cache: FxHashMap<u64, bool> = FxHashMap::default();
    let mut node_hit_cache: FxHashMap<u64, bool> = FxHashMap::default();
    let mut node_best_dist_cache: FxHashMap<u64, i32> = FxHashMap::default();
    let mut node_expansion_cache: FxHashMap<u64, Arc<Vec<WireId>>> = FxHashMap::default();
    let mut node_min_manhattan_cache: FxHashMap<u64, i32> = FxHashMap::default();

    // Seed beam with all source wires and their node members.
    // visited tracks PIP destination wires to avoid re-expansion.
    // Node members are NOT added to visited - they remain reachable
    // as PIP destinations from other paths.
    let mut visited_pips: FxHashSet<u64> = FxHashSet::default(); // packed pip ids
    let mut beam: Vec<BeamCandidate> = Vec::new();

    // Path arena: each entry is (pip, parent_index). u32::MAX means root.
    let mut path_arena: Vec<(PipId, u32)> = Vec::new();

    // Node-min manhattan: fetch (or compute once) the minimum manhattan
    // distance across all peers of `wire`'s node. A per-wire computation
    // walks the whole node every call; caching by node id makes the
    // *second* query on the same node O(1).
    let compute_node_min_manhattan = |wire: WireId,
                                      cache: &mut FxHashMap<u64, i32>|
     -> i32 {
        let nid = chipdb.node_id(wire);
        if let Some(id) = nid {
            if let Some(&v) = cache.get(&id) {
                return v;
            }
        }
        let mut best = manhattan(wire);
        chipdb.node_wires_cb(wire, |nw| {
            let d = manhattan(nw);
            if d < best {
                best = d;
            }
        });
        if let Some(id) = nid {
            cache.insert(id, best);
        }
        best
    };

    for &wire in src_wires {
        let nm = compute_node_min_manhattan(wire, &mut node_min_manhattan_cache);
        beam.push(BeamCandidate {
            wire,
            path_idx: u32::MAX,
            cost: nm as f32,
            best_dist: nm,
        });
        // Every peer on the same node shares `nm` (node-min manhattan is
        // a node property), so we avoid another node walk per peer.
        chipdb.node_wires_cb(wire, |nw| {
            beam.push(BeamCandidate {
                wire: nw,
                path_idx: u32::MAX,
                cost: nm as f32,
                best_dist: nm,
            });
        });
    }
    // Dedup seeds by wire.
    beam.sort_by_key(|c| c.wire.raw());
    beam.dedup_by_key(|c| c.wire.raw());

    // Bound the initial beam by cost so that mega-nodes (tens of thousands
    // of peers seeded at once) don't turn step 0 into an O(N) inner loop
    // before the usual per-step truncation kicks in.
    if beam.len() > beam_width {
        beam.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        beam.truncate(beam_width);
    }

    // Wall-clock timeout per beam search call: 500ms prevents hanging
    // while allowing enough time for complex routes.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
    let mut step_count: usize = 0;

    // Stall detection: break if best manhattan doesn't improve for 50 steps.
    let mut best_manhattan = i32::MAX;
    let mut stall_count: usize = 0;

    let effective_beam_width = beam_width;

    for _step in 0..max_steps {
        if beam.is_empty() {
            return None;
        }

        let mut next_candidates: Vec<BeamCandidate> = Vec::new();

        for candidate in &beam {
            let wire_info = chipdb.wire_info(candidate.wire);
            let downhill = wire_info.pips_downhill.get();

            for &pip_idx in downhill {
                // Check timeout every 1000 PIPs to avoid syscall overhead.
                step_count += 1;
                if step_count % 1000 == 0 && std::time::Instant::now() > deadline {
                    return None;
                }
                let pip = PipId::new(candidate.wire.tile(), pip_idx);

                // Skip PIPs we've already used.
                if !visited_pips.insert(pip.raw()) {
                    continue;
                }

                let next_wire = chipdb.pip_dst_wire(pip);

                // Wire-binding check: skip PIPs whose destination (or any
                // node-equivalent) is bound to a different net. Without this,
                // beam search is blind to actual claims and produces plans
                // that apply_route_plan rejects via try_bind_wire_node.
                let mut wire_taken = match ctx.wire_binding(next_wire) {
                    Some((owner, _)) if owner != net => true,
                    _ => false,
                };
                if !wire_taken {
                    let nid = chipdb.node_id(next_wire);
                    let cached = nid.and_then(|id| node_taken_cache.get(&id).copied());
                    wire_taken = match cached {
                        Some(v) => v,
                        None => {
                            let mut taken = false;
                            chipdb.node_wires_cb(next_wire, |nw| {
                                if let Some((owner, _)) = ctx.wire_binding(nw) {
                                    if owner != net {
                                        taken = true;
                                    }
                                }
                            });
                            if let Some(id) = nid {
                                node_taken_cache.insert(id, taken);
                            }
                            taken
                        }
                    };
                }
                if wire_taken {
                    continue;
                }

                // Push PIP into the arena and get its index.
                let new_path_idx = path_arena.len() as u32;
                path_arena.push((pip, candidate.path_idx));

                // Check hit on direct wire.
                if dst_nodes.contains(&next_wire) {
                    return Some(reconstruct_path(&path_arena, new_path_idx));
                }

                // Check hit via node expansion and find best distance.
                // Both are pure functions of the routing node, so cache them
                // by node id to avoid re-walking mega-nodes.
                let nid = chipdb.node_id(next_wire);
                let (hit, best_dist) = match nid {
                    Some(id)
                        if node_hit_cache.contains_key(&id)
                            && node_best_dist_cache.contains_key(&id) =>
                    {
                        (node_hit_cache[&id], node_best_dist_cache[&id])
                    }
                    _ => {
                        let mut hit = false;
                        let mut best = manhattan(next_wire);
                        chipdb.node_wires_cb(next_wire, |nw| {
                            if dst_nodes.contains(&nw) {
                                hit = true;
                            }
                            let d = manhattan(nw);
                            if d < best {
                                best = d;
                            }
                        });
                        if let Some(id) = nid {
                            node_hit_cache.insert(id, hit);
                            node_best_dist_cache.insert(id, best);
                        }
                        (hit, best)
                    }
                };

                if hit {
                    return Some(reconstruct_path(&path_arena, new_path_idx));
                }

                // Corridor penalty: wires outside the raster corridor are penalized
                // proportionally to manhattan distance so it scales with net length.
                let corridor_bonus = if corridor.contains(&next_wire.tile()) {
                    0.0f32
                } else {
                    (best_dist as f32 * 0.5).max(2.0)
                };

                // Congestion penalty.
                let cong_pen = cong_weight * cong.congestion_at(next_wire.tile());

                // Cost = remaining distance + penalties. Lower is better.
                let cost = best_dist as f32 + corridor_bonus + cong_pen;

                next_candidates.push(BeamCandidate {
                    wire: next_wire,
                    path_idx: new_path_idx,
                    cost,
                    best_dist,
                });
            }
        }

        if next_candidates.is_empty() {
            return None;
        }

        // Keep top candidates (lowest cost = closest to destination).
        next_candidates.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        next_candidates.truncate(effective_beam_width);

        // Expand node members of surviving candidates into the beam.
        // Only add node members that have PIPs (useful for exploration).
        // The "peers with downhill pips" list is a property of the node
        // and is Arc-shared via `node_expansion_cache` so mega-nodes pay
        // one walk per search, not one per surviving candidate.
        let mut expanded = Vec::new();
        for c in &next_candidates {
            let nid = chipdb.node_id(c.wire);
            let peers: Arc<Vec<WireId>> = match nid {
                Some(id) => {
                    if let Some(cached) = node_expansion_cache.get(&id) {
                        Arc::clone(cached)
                    } else {
                        let mut v = Vec::new();
                        chipdb.node_wires_cb(c.wire, |nw| {
                            if !chipdb.wire_info(nw).pips_downhill.get().is_empty() {
                                v.push(nw);
                            }
                        });
                        let arc = Arc::new(v);
                        node_expansion_cache.insert(id, Arc::clone(&arc));
                        arc
                    }
                }
                None => Arc::new(Vec::new()),
            };
            for &nw in peers.iter() {
                expanded.push(BeamCandidate {
                    wire: nw,
                    path_idx: c.path_idx,
                    cost: c.cost,
                    best_dist: c.best_dist,
                });
            }
        }
        next_candidates.extend(expanded);

        // Re-bound the beam after peer expansion: on a mega-node each
        // survivor can pull in thousands of peers, and without a second
        // truncation the beam width compounds across iterations.
        if next_candidates.len() > effective_beam_width * 2 {
            next_candidates.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
            next_candidates.truncate(effective_beam_width * 2);
        }

        // Stall detection uses the per-candidate `best_dist` cached at
        // creation time, avoiding a manhattan recompute per survivor.
        let step_best = next_candidates
            .iter()
            .map(|c| c.best_dist)
            .min()
            .unwrap_or(i32::MAX);
        if step_best < best_manhattan {
            best_manhattan = step_best;
            stall_count = 0;
        } else {
            stall_count += 1;
            if stall_count >= 50 {
                break;
            }
        }

        beam = next_candidates;
    }

    None
}

// ---------------------------------------------------------------------------
// Segment decomposition + bounded local search
// ---------------------------------------------------------------------------

/// Decompose a rasterized tile path into segments at direction changes.
/// Each segment is a run of consecutive same-direction tiles.
/// Returns vec of (start_idx, end_idx) into the path array.
fn decompose_segments(path: &[(i32, i32)]) -> Vec<(usize, usize)> {
    if path.len() < 2 {
        return vec![(0, path.len().saturating_sub(1))];
    }

    let mut segments = Vec::new();
    let mut seg_start = 0;
    let mut prev_dx = path[1].0 - path[0].0;
    let mut prev_dy = path[1].1 - path[0].1;

    for i in 2..path.len() {
        let dx = path[i].0 - path[i - 1].0;
        let dy = path[i].1 - path[i - 1].1;
        if (dx.signum(), dy.signum()) != (prev_dx.signum(), prev_dy.signum()) {
            segments.push((seg_start, i - 1));
            seg_start = i - 1; // overlap: end of prev = start of next
            prev_dx = dx;
            prev_dy = dy;
        }
    }
    segments.push((seg_start, path.len() - 1));
    segments
}

/// Cost model for `segment_astar`. Corridor pruning is expressed as a union
/// of small per-path-point bboxes (see `bboxes`); congestion enters the cost
/// as a per-wire penalty proportional to the arrival tile's usage.
struct SegmentCostModel<'a> {
    bboxes: &'a [BoundingBox],
    cong: &'a CongestionMap,
    cong_weight: f32,
    target_x: i32,
    target_y: i32,
}

impl<'a> PathCostModel for SegmentCostModel<'a> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn wire_penalty(&self, _ctx: &Context, wire: WireId) -> DelayT {
        (self.cong_weight * self.cong.congestion_at(wire.tile())) as DelayT
    }
    fn heuristic(&self, ctx: &Context, wire: WireId, _dst: WireId) -> DelayT {
        let (wx, wy) = ctx.chipdb().tile_xy(wire.tile());
        (wx - self.target_x).abs() + (wy - self.target_y).abs()
    }
    fn bboxes(&self) -> &[BoundingBox] {
        self.bboxes
    }
}

/// Build a union of small bboxes covering a corridor: one
/// `(2*margin+1) × (2*margin+1)` box around each tile on the path, clamped
/// to the grid. Used by `segment_astar` to express corridor-shaped tile
/// pruning via `PathCostModel::bboxes`.
fn corridor_bboxes(
    chipdb: &crate::chipdb::ChipDb,
    path: &[(i32, i32)],
    start: usize,
    end: usize,
    margin: i32,
) -> Vec<BoundingBox> {
    let w = chipdb.width();
    let h = chipdb.height();
    let mut out: Vec<BoundingBox> = Vec::with_capacity(end - start + 1);
    for i in start..=end {
        let (x, y) = path[i];
        let x0 = (x - margin).max(0);
        let y0 = (y - margin).max(0);
        let x1 = (x + margin).min(w - 1);
        let y1 = (y + margin).min(h - 1);
        out.push(BoundingBox { x0, y0, x1, y1 });
    }
    out
}

/// Pick a representative wire in `target_tile`: the wire with the most
/// `pips_uphill`, so it's reachable from many directions. Used as the
/// destination anchor when the actual goal is "any wire in `target_tile`".
fn representative_wire_in_tile(chipdb: &crate::chipdb::ChipDb, target_tile: i32) -> Option<WireId> {
    let tt = chipdb.tile_type(target_tile);
    let n_wires = tt.wires.get().len();
    let mut best: Option<(WireId, usize)> = None;
    for wi in 0..n_wires {
        let w = WireId::new(target_tile, wi as i32);
        let up = chipdb.wire_info(w).pips_uphill.get().len();
        match best {
            Some((_, bu)) if up <= bu => {}
            _ => best = Some((w, up)),
        }
    }
    best.map(|(w, _)| w)
}

/// Confined A* search within a segment's tile region.
///
/// Thin wrapper over [`astar_search`] that expresses the corridor as a union
/// of small bboxes (`PathCostModel::bboxes`). The destination is the final
/// sink when this is the last segment; otherwise a representative wire in
/// `target_tile` (the one with the most `pips_uphill`), which the search
/// will reach after any wire in `target_tile` enters the frontier via
/// intra-tile switch-matrix PIPs.
///
/// Returns `(pips, end_wire)` on success. `end_wire` is the `dst` wire
/// (reached by the path); callers seed the next segment from it plus its
/// node peers.
fn segment_astar(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    target_tile: i32,
    final_dst: Option<WireId>,
    bboxes: &[BoundingBox],
    cong: &CongestionMap,
    cong_weight: f32,
    max_expansions: usize,
) -> Option<(Vec<PipId>, WireId)> {
    let chipdb = ctx.chipdb();
    let (target_x, target_y) = chipdb.tile_xy(target_tile);

    let dst_wire = match final_dst {
        Some(w) => w,
        None => representative_wire_in_tile(chipdb, target_tile)?,
    };

    // Trivial hit: dst (or a peer) already in `src_wires`.
    if src_wires.contains(&dst_wire) {
        return Some((Vec::new(), dst_wire));
    }

    let model = SegmentCostModel {
        bboxes,
        cong,
        cong_weight,
        target_x,
        target_y,
    };
    let opts = AStarOptions {
        visit_limit: Some(max_expansions),
        exhaustive: false,
    };
    let result = astar_search(ctx, &model, src_wires, dst_wire, &opts);
    result.path.map(|pips| (pips, dst_wire))
}

/// Unrestricted A* for same-tile (or short-distance) routing: no bbox
/// pruning, manhattan heuristic, 20k-pop budget.
///
/// Wrapper over [`astar_search`]; kept separate from the full router
/// `astar_route` because this path never uses the lookahead and never sees
/// wire-penalty state.
fn local_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
) -> Option<Vec<PipId>> {
    struct LocalModel;
    impl PathCostModel for LocalModel {
        fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
            default_pip_cost(ctx, pip)
        }
        fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
            let chipdb = ctx.chipdb();
            let (wx, wy) = chipdb.tile_xy(wire.tile());
            let (dx, dy) = chipdb.tile_xy(dst.tile());
            (wx - dx).abs() + (wy - dy).abs()
        }
    }
    let opts = AStarOptions {
        visit_limit: Some(20_000),
        exhaustive: false,
    };
    astar_search(ctx, &LocalModel, src_wires, dst_wire, &opts).path
}

/// Route using segment decomposition: break the rasterized line into
/// direction-change segments, do confined A* within each segment's tiles.
fn segment_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    path: &[(i32, i32)],
    cong: &CongestionMap,
    cong_weight: f32,
) -> Option<Vec<PipId>> {
    let chipdb = ctx.chipdb();

    let mut dst_nodes: FxHashSet<WireId> = FxHashSet::default();
    dst_nodes.insert(dst_wire);
    chipdb.node_wires_cb(dst_wire, |nw| {
        dst_nodes.insert(nw);
    });

    for src in src_wires {
        if dst_nodes.contains(src) {
            return Some(Vec::new());
        }
    }

    // Same-tile: use local route (no tile confinement, generous budget).
    if path.len() < 2 {
        return local_route(ctx, src_wires, dst_wire);
    }

    let segments = decompose_segments(path);
    let mut all_pips: Vec<PipId> = Vec::new();
    let mut current_wires: FxHashSet<WireId> = src_wires.clone();

    for (seg_idx, &(start, end)) in segments.iter().enumerate() {
        let is_last = seg_idx == segments.len() - 1;
        let seg_len = end - start + 1;

        let margin: i32 = 1;

        // Corridor pruning: one small bbox per path tile, union = corridor.
        let bboxes = corridor_bboxes(chipdb, path, start, end, margin);

        let (tx, ty) = path[end];
        let target_tile = chipdb.tile_by_xy(tx, ty);
        // For the final segment the real sink wire is the goal. For
        // intermediate segments we only need to *reach* target_tile; the
        // wrapper picks a representative wire with the most pips_uphill.
        let final_dst = if is_last { Some(dst_wire) } else { None };

        // Confined A* within this segment's tiles.
        let budget = (seg_len * 3000).max(5000) as usize;
        match segment_astar(
            ctx,
            &current_wires,
            target_tile,
            final_dst,
            &bboxes,
            cong,
            cong_weight,
            budget,
        ) {
            Some((pips, end_wire)) => {
                all_pips.extend_from_slice(&pips);

                // Update current wires for next segment.
                current_wires.clear();
                current_wires.insert(end_wire);
                chipdb.node_wires_cb(end_wire, |nw| {
                    current_wires.insert(nw);
                });
                // Also add PIP destination wires.
                for &pip in &pips {
                    let dw = chipdb.pip_dst_wire(pip);
                    current_wires.insert(dw);
                    chipdb.node_wires_cb(dw, |nw| {
                        current_wires.insert(nw);
                    });
                }
            }
            None => {
                // Diagnostic: which segment failed?
                let (sx, sy) = path[start];
                let (ex, ey) = path[end];
                let seg_len = end - start + 1;
                eprintln!(
                    "  seg_fail: seg {}/{} ({},{})→({},{}) len={} tiles={} is_last={}",
                    seg_idx,
                    segments.len(),
                    sx,
                    sy,
                    ex,
                    ey,
                    seg_len,
                    bboxes.len(),
                    is_last,
                );
                return None;
            }
        }
    }

    Some(all_pips)
}

// ---------------------------------------------------------------------------
// Net routing
// ---------------------------------------------------------------------------

/// Route a net using pure greedy line-walk (no beam search).
fn route_net_greedy(
    ctx: &Context,
    net: NetId,
    cong: &CongestionMap,
    cong_weight: f32,
    cfg: &RasterRouterCfg,
) -> Result<RoutePlan, RouterError> {
    let source_wire = match resolve_source_wire(ctx, net)? {
        Some(w) => w,
        None => {
            return Ok(RoutePlan {
                net,
                source_wire: WireId::INVALID,
                sink_routes: Vec::new(),
            });
        }
    };

    let chipdb = ctx.chipdb();
    let chip_w = chipdb.width();
    let chip_h = chipdb.height();

    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());

    let mut tree_tiles: Vec<(i32, i32)> = vec![(src_x, src_y)];
    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    chipdb.node_wires_cb(source_wire, |nw| {
        tree_wires.insert(nw);
    });

    let mut remaining_sinks: Vec<(WireId, i32, i32)> = sink_wires
        .iter()
        .map(|&w| {
            let (sx, sy) = chipdb.tile_xy(w.tile());
            (w, sx, sy)
        })
        .collect();

    let mut sink_routes = Vec::new();

    while !remaining_sinks.is_empty() {
        let mut best_sink_idx = 0;
        let mut best_tree_tile = tree_tiles[0];
        let mut best_dist = i32::MAX;
        for (si, &(_, sx, sy)) in remaining_sinks.iter().enumerate() {
            for &(tx, ty) in &tree_tiles {
                let d = (sx - tx).abs() + (sy - ty).abs();
                if d < best_dist {
                    best_dist = d;
                    best_sink_idx = si;
                    best_tree_tile = (tx, ty);
                }
            }
        }

        let (sink_wire, sink_x, sink_y) = remaining_sinks.remove(best_sink_idx);

        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }

        // For same-tile routing, use unconstrained local_route.
        // For everything else, use segment-based confined A*.
        let same_tile =
            best_dist == 0 || (sink_x == best_tree_tile.0 && sink_y == best_tree_tile.1);
        let use_local = same_tile;

        let route_result = if use_local {
            local_route(ctx, &tree_wires, sink_wire)
        } else {
            // Rasterize the line path.
            let (path, _corridor) = raster_corridor(
                best_tree_tile.0,
                best_tree_tile.1,
                sink_x,
                sink_y,
                cong,
                cong_weight,
                cfg.filter_radius,
                chip_w,
                chip_h,
            );
            segment_route(ctx, &tree_wires, sink_wire, &path, cong, cong_weight)
        };

        match route_result {
            Some(pips) => {
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    let (tx, ty) = chipdb.tile_xy(dst.tile());
                    tree_tiles.push((tx, ty));
                    tree_wires.insert(dst);
                    chipdb.node_wires_cb(dst, |nw| {
                        tree_wires.insert(nw);
                    });
                }
                tree_wires.insert(sink_wire);
                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                let net_name = ctx.net(net).name_id();
                let (sx, sy) = chipdb.tile_xy(source_wire.tile());
                let est_delay = ctx.estimate_delay_for_net(net);
                let constraint = ctx.net(net).clock_constraint();
                let n_users = ctx.net(net).num_users();
                eprintln!(
                    "  FAIL net={} src=({},{}) sink=({},{}) dist={} delay={} clk={} users={} local={}",
                    ctx.name_of(net_name), sx, sy, sink_x, sink_y,
                    best_dist, est_delay, constraint, n_users, use_local,
                );
                return Err(RouterError::NoPath(ctx.name_of(net_name).to_owned()));
            }
        }
    }

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

fn route_net_raster(
    ctx: &Context,
    net: NetId,
    cong: &CongestionMap,
    cong_weight: f32,
    cfg: &RasterRouterCfg,
) -> Result<RoutePlan, RouterError> {
    let source_wire = match resolve_source_wire(ctx, net)? {
        Some(w) => w,
        None => {
            return Ok(RoutePlan {
                net,
                source_wire: WireId::INVALID,
                sink_routes: Vec::new(),
            });
        }
    };

    let chipdb = ctx.chipdb();
    let chip_w = chipdb.width();
    let chip_h = chipdb.height();

    // Skip constant nets.
    let const_val = source_wire_const_value(ctx, source_wire);
    if const_val != 0 {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let sink_wires = collect_sink_wires(ctx, net);
    if sink_wires.is_empty() {
        return Ok(RoutePlan {
            net,
            source_wire,
            sink_routes: Vec::new(),
        });
    }

    let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());

    // Nearest-unrouted-sink Steiner heuristic.
    let mut tree_tiles: Vec<(i32, i32)> = vec![(src_x, src_y)];
    let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
    tree_wires.insert(source_wire);
    chipdb.node_wires_cb(source_wire, |nw| {
        tree_wires.insert(nw);
    });

    let mut remaining_sinks: Vec<(WireId, i32, i32)> = sink_wires
        .iter()
        .map(|&w| {
            let (sx, sy) = chipdb.tile_xy(w.tile());
            (w, sx, sy)
        })
        .collect();

    // Per-net time budget: 2s base + 10ms per sink. Prevents high-fanout nets
    // from monopolizing routing time across rip-up-reroute passes.
    let net_deadline = std::time::Instant::now()
        + std::time::Duration::from_millis(2000 + remaining_sinks.len() as u64 * 10);

    let mut sink_routes = Vec::new();

    while !remaining_sinks.is_empty() {
        // Check per-net time budget.
        if std::time::Instant::now() > net_deadline {
            break;
        }

        // Find nearest unrouted sink to any tile in the tree.
        let mut best_sink_idx = 0;
        let mut best_tree_tile = tree_tiles[0];
        let mut best_dist = i32::MAX;

        for (si, &(_, sx, sy)) in remaining_sinks.iter().enumerate() {
            for &(tx, ty) in &tree_tiles {
                let d = (sx - tx).abs() + (sy - ty).abs();
                if d < best_dist {
                    best_dist = d;
                    best_sink_idx = si;
                    best_tree_tile = (tx, ty);
                }
            }
        }

        let (sink_wire, sink_x, sink_y) = remaining_sinks.remove(best_sink_idx);

        if tree_wires.contains(&sink_wire) {
            sink_routes.push(SinkRoute {
                sink_wire,
                pips: vec![],
            });
            continue;
        }

        // Rasterize corridor from nearest tree tile to sink.
        let (_path, corridor) = raster_corridor(
            best_tree_tile.0,
            best_tree_tile.1,
            sink_x,
            sink_y,
            cong,
            cong_weight,
            cfg.filter_radius,
            chip_w,
            chip_h,
        );

        // Beam search through the actual PIP graph.
        match beam_search_route(
            ctx,
            net,
            &tree_wires,
            sink_wire,
            &corridor,
            cong,
            cong_weight,
            cfg.beam_width,
            cfg.max_beam_steps,
        ) {
            Some(pips) => {
                // Add path tiles and wires to tree.
                for &pip in &pips {
                    let dst = chipdb.pip_dst_wire(pip);
                    let (tx, ty) = chipdb.tile_xy(dst.tile());
                    tree_tiles.push((tx, ty));
                    tree_wires.insert(dst);
                    chipdb.node_wires_cb(dst, |nw| {
                        tree_wires.insert(nw);
                    });
                }
                tree_wires.insert(sink_wire);

                sink_routes.push(SinkRoute { sink_wire, pips });
            }
            None => {
                // Preserve partial progress: keep sinks routed so far, skip
                // this sink, and let the A* cleanup pass attempt it later.
                // Returning Err would discard the whole plan and leave the
                // net empty, which makes rip-up passes destructive.
                continue;
            }
        }
    }

    Ok(RoutePlan {
        net,
        source_wire,
        sink_routes,
    })
}

// ---------------------------------------------------------------------------
// Router implementation
// ---------------------------------------------------------------------------

pub struct RasterRouter;

impl super::Router for RasterRouter {
    type Config = RasterRouterCfg;

    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), RouterError> {
        let nets = collect_routable_nets(ctx);
        self.route_nets(ctx, cfg, &nets)
    }

    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), RouterError> {
        let chip_w = ctx.chipdb().width();
        let chip_h = ctx.chipdb().height();

        eprintln!(
            "RasterRouter: {} nets on {}x{}, beam_width={}, max_steps={}",
            nets.len(),
            chip_w,
            chip_h,
            cfg.beam_width,
            cfg.max_beam_steps,
        );

        let mut cong = CongestionMap::new(chip_w, chip_h);
        let mut cong_weight = cfg.initial_cong_weight;
        let mut best_routed = 0usize;
        let mut best_wl = 0u64;

        // Pre-reserve BEL-pin wires. Every alive net's driver output wire and
        // every sink (user) input wire is bound to its owning net before any
        // routing begins. This prevents one net's route from claiming a
        // routing node that includes another net's driver output — a common
        // failure mode on dense designs where sibling bus bits ('vert[2]' vs
        // 'vert[3]') had adjacent BEL outputs sharing a node. Beam search
        // already skips PIPs whose dst is owned by a different net, so
        // reserved pins naturally steer routes around them.
        let mut pin_wires: Vec<(crate::chipdb::WireId, NetId)> = Vec::new();
        for &net in nets {
            let n = ctx.net(net);
            if !n.is_alive() || !n.has_driver() {
                continue;
            }
            if let Some(driver) = n.driver_cell_port() {
                if let Some(bel) = ctx.cell(driver.cell).bel() {
                    if let Some(w) = bel.pin_wire(driver.port) {
                        pin_wires.push((w.id(), net));
                    }
                }
            }
            for user in n.users() {
                if !user.is_valid() {
                    continue;
                }
                if let Some(ubel) = ctx.cell(user.cell).bel() {
                    if let Some(uw) = ubel.pin_wire(user.port) {
                        pin_wires.push((uw.id(), net));
                    }
                }
            }
        }
        let mut reservations_applied = 0usize;
        let mut reservation_conflicts = 0usize;
        for (wire, net) in pin_wires {
            match ctx.try_bind_wire_node(wire, net, crate::common::PlaceStrength::Strong) {
                Ok(()) => reservations_applied += 1,
                Err(_) => reservation_conflicts += 1,
            }
        }
        eprintln!(
            "RasterRouter: pre-reserved {} BEL-pin wires ({} conflicts)",
            reservations_applied, reservation_conflicts
        );

        // Wire-level negotiation state (PathFinder-style).
        let neg_cfg = super::common::NegotiationCfg::default();
        let mut neg_state = super::common::NegotiationState::new(neg_cfg);

        // Pre-build lookahead for A* cleanup (reused across iterations).
        let lookahead = super::lookahead::Lookahead::build(ctx, 40);

        // Iteration 0: route ALL nets (beam search).
        // Iterations 1+: selective rip-up of congested nets only (PathFinder-style).
        for iter in 0..cfg.max_iterations {
            let nets_to_route: Vec<NetId>;

            if iter == 0 {
                // First pass: route everything.
                nets_to_route = nets.to_vec();
                cong.reset_usage();
            } else {
                // Subsequent passes: rip up congested nets AND retry any net
                // whose routing tree is still empty (iter-0 bind-conflict or
                // beam-search failure). Empty-tree nets contribute no wire
                // usage so find_congested_nets can't surface them.
                let congested = neg_state.find_congested_nets(&ctx.design);
                for &net in &congested {
                    neg_state.remove_net_usage(&ctx.design, net);
                    if ctx.net(net).wires().len() > 0 {
                        unroute_net(ctx, net);
                    }
                }
                let mut to_retry = congested;
                for idx in ctx.design.iter_net_indices() {
                    let n = ctx.net(idx);
                    if n.is_alive() && n.has_driver() && n.num_users() > 0 && n.wires().is_empty() {
                        to_retry.push(idx);
                    }
                }
                to_retry.sort_unstable();
                to_retry.dedup();
                if to_retry.is_empty() {
                    eprintln!("RasterRouter: no wire congestion at pass {}, validating", iter);
                    return validate_all_routed(ctx);
                }
                cong.update_history();
                cong_weight *= cfg.cong_growth;
                // Rebuild tile congestion from remaining (non-ripped-up) routes.
                cong.reset_usage();
                for &net in nets {
                    for &wire in ctx.net(net).wires().keys() {
                        cong.add_usage(wire.tile(), 1.0);
                    }
                }
                nets_to_route = to_retry;
            }

            // Sort nets: high-fanout first.
            let mut sorted_nets: Vec<(NetId, f64)> = nets_to_route
                .iter()
                .map(|&net| {
                    let n_users = ctx.net(net).num_users() as f64;
                    let score = n_users * 100.0;
                    (net, score)
                })
                .collect();
            sorted_nets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if iter == 0 {
                let top5: Vec<(f64, usize)> = sorted_nets
                    .iter()
                    .take(5)
                    .map(|(net, s)| (*s, ctx.net(*net).num_users()))
                    .collect();
                eprintln!("  priority top-5 (score, fanout): {:?}", top5);
            }

            let ordered_nets: Vec<NetId> = sorted_nets.iter().map(|x| x.0).collect();

            // Split: high-fanout sequential, rest parallel.
            let hf_cutoff = 20usize;
            let mut hf_nets: Vec<NetId> = Vec::new();
            let mut lf_nets: Vec<NetId> = Vec::new();
            for &net in &ordered_nets {
                if ctx.net(net).num_users() >= hf_cutoff {
                    hf_nets.push(net);
                } else {
                    lf_nets.push(net);
                }
            }

            let mut hf_routed = 0usize;
            for &net in &hf_nets {
                let result = if cfg.use_greedy {
                    route_net_greedy(ctx, net, &cong, cong_weight, cfg)
                } else {
                    route_net_raster(ctx, net, &cong, cong_weight, cfg)
                };
                if let Ok(plan) = result {
                    if plan.source_wire.is_valid() && !plan.sink_routes.is_empty() {
                        match apply_route_plan(ctx, &plan) {
                            Ok(()) => {
                                for sr in &plan.sink_routes {
                                    for &pip in &sr.pips {
                                        let tile = ctx.chipdb().pip_dst_wire(pip).tile();
                                        cong.add_usage(tile, 1.0);
                                    }
                                }
                                hf_routed += 1;
                            }
                            Err(_conflict) => {}
                        }
                    }
                }
            }

            if iter == 0 && !hf_nets.is_empty() {
                eprintln!(
                    "  high-fanout: {}/{} routed sequentially",
                    hf_routed,
                    hf_nets.len()
                );
            }

            // Serialize low-fanout planning so each plan sees prior applies.
            // Parallel planning with par_iter() was racing: workers planned
            // independently through the same node, and the bind-conflict check
            // in apply_route_plan rejected >60% of plans. Sequential planning
            // lets each net observe current bindings via cong/ctx state.
            let mut routed = hf_routed;
            let mut failed = hf_nets.len() - hf_routed;

            for &net in &lf_nets {
                let result = if cfg.use_greedy {
                    route_net_greedy(ctx, net, &cong, cong_weight, cfg)
                } else {
                    route_net_raster(ctx, net, &cong, cong_weight, cfg)
                };
                match result {
                    Ok(plan) if plan.source_wire.is_valid() && !plan.sink_routes.is_empty() => {
                        match apply_route_plan(ctx, &plan) {
                            Ok(()) => {
                                for sr in &plan.sink_routes {
                                    for &pip in &sr.pips {
                                        let tile = ctx.chipdb().pip_dst_wire(pip).tile();
                                        cong.add_usage(tile, 1.0);
                                    }
                                }
                                routed += 1;
                            }
                            Err(_conflict) => {
                                failed += 1;
                            }
                        }
                    }
                    _ => {
                        failed += 1;
                    }
                }
            }

            // Update wire-level negotiation state.
            neg_state.update_usage(&ctx.design);
            neg_state.update_history();
            neg_state.present_cost *= neg_state.cfg.present_cost_growth;
            neg_state.present_cost = neg_state.present_cost.min(8.0);

            let congested = cong.num_congested_tiles();
            let wire_congested = neg_state.count_congested_wires();
            let wl = routed as u64; // use routed count as proxy for quality

            eprintln!(
                "RasterRouter pass {}: routed={}, failed={}, tile_cong={}, wire_cong={}, max_usage={:.0}, wl={}, cong_w={:.2}",
                iter, routed, failed, congested, wire_congested, cong.max_usage(), wl, cong_weight,
            );

            if routed > best_routed || (routed == best_routed && wl < best_wl) {
                best_routed = routed;
                best_wl = wl;
            }

            if congested == 0 && wire_congested == 0 && failed == 0 {
                eprintln!("RasterRouter: converged at pass {}", iter);
                return Ok(());
            }

            // Cleanup pass: route remaining failed nets with A*.
            // Runs every iteration (not just iter==0) with per-net step budgets.
            if failed > 0 {
                let mut failed_nets: Vec<NetId> = ordered_nets
                    .iter()
                    .filter(|&&net| {
                        let n = ctx.net(net);
                        n.is_alive() && n.has_driver() && n.num_users() > 0 && n.wires().is_empty()
                    })
                    .copied()
                    .collect();

                // Route const nets first: their per-sink lookup is O(1) and
                // deterministic, so we never lose a const net to the wall-clock
                // budget getting burned by a slow non-const A*.
                failed_nets.sort_by_key(|&n| {
                    let cv = resolve_source_wire(ctx, n)
                        .ok()
                        .flatten()
                        .map(|w| source_wire_const_value(ctx, w))
                        .unwrap_or(0);
                    if cv != 0 {
                        0
                    } else if resolve_source_wire(ctx, n)
                        .ok()
                        .flatten()
                        .map(|w| is_global_clock_wire(ctx, w))
                        .unwrap_or(false)
                    {
                        1
                    } else {
                        2
                    }
                });

                if !failed_nets.is_empty() {
                    // Build wire penalty from negotiation state for A* cost.
                    let wire_penalty: FxHashMap<WireId, crate::timing::DelayT> = neg_state
                        .wire_history
                        .iter()
                        .map(|(&w, &h)| {
                            (
                                w,
                                (h * neg_state.cfg.history_cost_multiplier)
                                    as crate::timing::DelayT,
                            )
                        })
                        .collect();

                    // Hoist clock-backbone enumeration out of the per-net
                    // loop. Iterating every wire in the chipdb for each failed
                    // net costs ~2.5M wires × ~16 nets per cleanup pass —
                    // slow enough to eat the entire 30 s wall budget before
                    // the first A* runs. Compute once per iteration.
                    let mut clock_backbone_wires: Vec<WireId> = Vec::new();
                    let needs_clock_backbone = failed_nets.iter().any(|&n| {
                        resolve_source_wire(ctx, n)
                            .ok()
                            .flatten()
                            .map(|w| is_global_clock_wire(ctx, w))
                            .unwrap_or(false)
                    });
                    if needs_clock_backbone {
                        for w in ctx.chipdb().wires() {
                            if is_global_clock_wire(ctx, w) {
                                clock_backbone_wires.push(w);
                            }
                        }
                    }

                    eprintln!(
                        "RasterRouter: A* cleanup for {} failed nets (iter {})",
                        failed_nets.len(),
                        iter,
                    );

                    let mut cleanup_routed = 0usize;
                    let cleanup_start = std::time::Instant::now();
                    let cleanup_budget = std::time::Duration::from_secs(30);

                    for &net in &failed_nets {
                        if cleanup_start.elapsed() > cleanup_budget {
                            eprintln!(
                                "RasterRouter: A* cleanup timeout after {}s, {}/{} recovered",
                                cleanup_budget.as_secs(),
                                cleanup_routed,
                                failed_nets.len(),
                            );
                            break;
                        }
                        let source_wire = match resolve_source_wire(ctx, net) {
                            Ok(Some(w)) => w,
                            _ => continue,
                        };

                        let mut sink_wires = collect_sink_wires(ctx, net);

                        // Sort sinks nearest-first so the routing tree grows
                        // toward the destination, making subsequent A* calls cheaper.
                        let (sx, sy) = ctx.chipdb().tile_xy(source_wire.tile());
                        sink_wires.sort_by_key(|&sw| {
                            let (wx, wy) = ctx.chipdb().tile_xy(sw.tile());
                            (wx - sx).abs() + (wy - sy).abs()
                        });

                        // Visit limit per sink. xc7_large needs hundreds of
                        // node hops through ~10k-PIP switch matrices, so a
                        // few-thousand budget cannot reach anything further
                        // than ~2 composite columns with the lookahead.
                        let per_sink_limit = if sink_wires.len() > 50 {
                            Some(200_000)
                        } else {
                            Some(500_000)
                        };

                        let src_const = source_wire_const_value(ctx, source_wire);

                        let mut tree_wires: FxHashSet<WireId> = FxHashSet::default();
                        tree_wires.insert(source_wire);
                        ctx.chipdb().node_wires_cb(source_wire, |nw| {
                            tree_wires.insert(nw);
                        });

                        // Global-clock nets are driven by a single BUFG output
                        // whose `pips_downhill` is near-empty. Seeding A* with
                        // the whole clock backbone lets the search start from
                        // anywhere nearby — without this the heap drains after
                        // ≤4 expansions and no sink is reachable.
                        if is_global_clock_wire(ctx, source_wire) {
                            for &w in &clock_backbone_wires {
                                tree_wires.insert(w);
                            }
                        }

                        let mut sink_routes = Vec::new();
                        // For constant nets, remember each sink's switch-matrix
                        // anchor (the `pip.src` whose `const_value` matches).
                        // These are registered as leaves after apply_route_plan
                        // so the validator's chain walk can terminate at them.
                        let mut const_anchors: FxHashSet<WireId> = FxHashSet::default();

                        for &sink_wire in &sink_wires {
                            // Check cleanup budget per-sink to avoid hanging
                            // inside a single high-fanout net.
                            if cleanup_start.elapsed() > cleanup_budget {
                                break;
                            }

                            // Constant-net fast path: each tile's switch matrix
                            // ties GND/VCC to a local wire reachable from every
                            // const-consuming sink via one PIP. Use that path
                            // directly — it's deterministic, O(|pips_uphill|),
                            // and avoids the multi-source A* heap blow-up that
                            // a chip-wide const pool would cause.
                            if src_const != 0 {
                                if let Some(pip) =
                                    find_local_const_pip(ctx, sink_wire, src_const)
                                {
                                    let anchor = ctx.chipdb().pip_src_wire(pip);
                                    const_anchors.insert(anchor);
                                    // Let subsequent sinks see this anchor as
                                    // part of the routing tree so A* fallback
                                    // (if triggered) skips re-routing to it.
                                    tree_wires.insert(anchor);
                                    tree_wires.insert(sink_wire);
                                    ctx.chipdb().node_wires_cb(sink_wire, |nw| {
                                        tree_wires.insert(nw);
                                    });
                                    sink_routes.push(SinkRoute {
                                        sink_wire,
                                        pips: vec![pip],
                                    });
                                    continue;
                                }
                            }

                            if tree_wires.contains(&sink_wire) {
                                sink_routes.push(SinkRoute {
                                    sink_wire,
                                    pips: vec![],
                                });
                                continue;
                            }

                            // Use multi-pip-hop expansion for long-distance
                            // sinks. Single-hop A* on IOB→interior routes
                            // wastes its visit budget wandering in the IOB
                            // column because the lookahead has no sample for
                            // IO wire types and every short-range hop looks
                            // equally promising. Two-hop expansion lets the
                            // heuristic see past the IOB wall.
                            let (sink_tx_early, sink_ty_early) =
                                ctx.chipdb().tile_xy(sink_wire.tile());
                            let manhattan =
                                (sink_tx_early - sx).abs() + (sink_ty_early - sy).abs();
                            let route_fn = if manhattan > 50 {
                                astar_route_multihop
                            } else {
                                astar_route
                            };

                            match route_fn(
                                ctx,
                                &tree_wires,
                                sink_wire,
                                &wire_penalty,
                                None,
                                50,
                                Some(&lookahead),
                                per_sink_limit,
                            ) {
                                Some(pips) => {
                                    for &pip in &pips {
                                        let dw = ctx.chipdb().pip_dst_wire(pip);
                                        tree_wires.insert(dw);
                                        ctx.chipdb().node_wires_cb(dw, |nw| {
                                            tree_wires.insert(nw);
                                        });
                                    }
                                    sink_routes.push(SinkRoute { sink_wire, pips });
                                }
                                None => {
                                    // Partial routing: bump history for wires in the
                                    // rasterized corridor tiles so that subsequent
                                    // iterations route away from this congested region.
                                    let (sink_tx, sink_ty) = ctx.chipdb().tile_xy(sink_wire.tile());
                                    let (_path, corridor_tiles) = raster_corridor(
                                        sx,
                                        sy,
                                        sink_tx,
                                        sink_ty,
                                        &cong,
                                        cong_weight,
                                        cfg.filter_radius,
                                        chip_w,
                                        chip_h,
                                    );
                                    for &tile_idx in &corridor_tiles {
                                        let tile = tile_idx as i32;
                                        let num_wires =
                                            ctx.chipdb().tile_type(tile).wires.get().len();
                                        for wi in 0..num_wires {
                                            let w = WireId::new(tile, wi as i32);
                                            *neg_state
                                                .wire_history
                                                .entry(w)
                                                .or_insert(0.0) += 0.5;
                                        }
                                    }
                                    continue;
                                }
                            }
                        }

                        // Apply the route if at least some sinks succeeded.
                        if !sink_routes.is_empty() {
                            // For A*-routed const sinks the first PIP's `src`
                            // is a switch-matrix const wire — gather those too
                            // so the validator accepts mixed local + A* const
                            // routes.
                            if src_const != 0 {
                                for sink in &sink_routes {
                                    if let Some(&first_pip) = sink.pips.first() {
                                        let anchor = ctx.chipdb().pip_src_wire(first_pip);
                                        if ctx.chipdb().wire_info(anchor).const_value
                                            == src_const
                                        {
                                            const_anchors.insert(anchor);
                                        }
                                    }
                                }
                            }

                            let plan = RoutePlan {
                                net,
                                source_wire,
                                sink_routes,
                            };
                            if apply_route_plan(ctx, &plan).is_ok() {
                                // Register const anchors as leaves so the
                                // validator's chain walk terminates there via
                                // the const_value match rule. Binding is
                                // idempotent per net; anchor wires are tile-
                                // local so cross-net conflicts only arise if
                                // two different nets carry the same const
                                // value (normally only one GND and one VCC
                                // net exist).
                                let mut anchor_conflict = false;
                                for &anchor in &const_anchors {
                                    if ctx
                                        .try_bind_wire_node(
                                            anchor,
                                            net,
                                            crate::common::PlaceStrength::Strong,
                                        )
                                        .is_err()
                                    {
                                        anchor_conflict = true;
                                        break;
                                    }
                                    ctx.design.net_edit(net).add_wire(
                                        anchor,
                                        None,
                                        crate::common::PlaceStrength::Strong,
                                    );
                                }
                                if anchor_conflict {
                                    unroute_net(ctx, net);
                                } else {
                                    cleanup_routed += 1;
                                }
                            }
                        }
                    }

                    failed -= cleanup_routed;
                    eprintln!(
                        "  A* cleanup: +{} routed, {} still failed",
                        cleanup_routed, failed,
                    );
                }
            }

            if failed == 0 {
                eprintln!("RasterRouter: all nets routed at pass {}, validating", iter);
                return validate_all_routed(ctx);
            }
        }

        eprintln!(
            "RasterRouter: {} passes, best_routed={}, congested={}",
            cfg.max_iterations,
            best_routed,
            cong.num_congested_tiles(),
        );
        Err(RouterError::Congestion(
            cfg.max_iterations,
            cong.num_congested_tiles(),
        ))
    }
}

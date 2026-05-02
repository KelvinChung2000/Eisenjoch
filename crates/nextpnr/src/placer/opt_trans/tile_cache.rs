//! Span-template cost cache for the opt_trans pipe graph.
//!
//! This is the first step toward the two-level model: it keeps the existing
//! pipe graph and per-pipe usage accounting, but shares BPR cost computation
//! across pipes with the same source tile type, span delta, usage, capacity,
//! and base-cost bucket.
//!
//! v2 (TwoLevelPip) additionally holds per-tile-type internal PIP templates
//! and a switch-matrix cost cache; see `TileTypeTemplate` and
//! `SwitchMatrixCache` below.

use rustc_hash::FxHashMap;

use crate::chipdb::{port_key, span_bucket_of, Side, TileLocalWs, TileTypeTemplate};

use super::network::{PipeNetwork, DIST_SCALE};
use super::resistance::{bpr_alpha, bpr_beta, ResistanceModel};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TileSpanCostKey {
    pub tile_type: u16,
    pub dx: i16,
    pub dy: i16,
    pub usage: u16,
    pub capacity: u16,
    pub base_q: u16,
}

#[derive(Clone, Debug, Default)]
pub struct SpanCostStats {
    pub epoch: u64,
    pub lookups: usize,
    pub entries: usize,
    pub hits: usize,
    pub misses: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SpanCostTable {
    pub enabled: bool,
    pub epoch: u64,
    pub pipe_entry: Vec<u32>,
    pub costs: Vec<f64>,
    pub costs_int: Vec<u32>,
    pub stats: SpanCostStats,
}

impl SpanCostTable {
    pub fn disabled(n_pipes: usize) -> Self {
        Self {
            enabled: false,
            epoch: 0,
            pipe_entry: vec![u32::MAX; n_pipes],
            costs: Vec::new(),
            costs_int: Vec::new(),
            stats: SpanCostStats::default(),
        }
    }

    pub fn reset_disabled(&mut self, n_pipes: usize) {
        self.enabled = false;
        self.pipe_entry.resize(n_pipes, u32::MAX);
        self.costs.clear();
        self.costs_int.clear();
        self.stats = SpanCostStats::default();
    }
}

#[inline]
fn quantize_usage(usage: f64) -> u16 {
    usage.max(0.0).round().min(u16::MAX as f64) as u16
}

#[inline]
fn quantize_capacity(capacity: f64) -> u16 {
    capacity.max(0.0).round().min(u16::MAX as f64) as u16
}

#[inline]
fn quantize_base(base: f64) -> u16 {
    (base.max(0.0) * DIST_SCALE)
        .round()
        .min(u16::MAX as f64) as u16
}

#[inline]
fn quantize_cost_int(cost: f64) -> u32 {
    let scaled = (cost.max(1e-12) * DIST_SCALE).round();
    if scaled >= u32::MAX as f64 {
        u32::MAX - 1
    } else {
        (scaled as u32).max(1)
    }
}

/// Quantise per-tile total usage (sum of `pipe.net_count` over outgoing
/// pipes). Caps at `u16::MAX` to fit the cache key.
#[inline]
fn quantize_tile_usage(total: f64) -> u16 {
    if total <= 0.0 {
        return 0;
    }
    total.round().min(u16::MAX as f64) as u16
}

/// Compute per-tile quantised throughput: sum of `net_count` over outgoing
/// pipes at each endpoint. Used as the BPR load for tile-internal PIPs.
fn compute_tile_usage(network: &PipeNetwork) -> Vec<u16> {
    let n_nodes = network.nodes.len();
    let mut accum = vec![0.0f64; n_nodes];
    for pipe in &network.pipes {
        let nc = pipe.net_count.max(0.0);
        accum[pipe.from] += nc;
        accum[pipe.to] += nc;
    }
    let mut out = Vec::with_capacity(n_nodes);
    for v in accum {
        out.push(quantize_tile_usage(v));
    }
    out
}

/// Per-iter stats for the switch-matrix cache inside
/// `rebuild_span_cost_table_pip`.
#[derive(Clone, Debug, Default)]
pub struct SwitchMatrixStats {
    pub unique_type_usage_pairs: usize,
    pub switch_lookups: usize,
    pub switch_cache_hits: usize,
    pub dijkstra_calls: usize,
    pub dijkstra_total_us: u128,
}

/// v2 TwoLevelPip rebuild: same span-cost table layout as v1, plus an
/// additive switch-matrix cost from chipdb PIPs.
pub fn rebuild_span_cost_table_pip(
    network: &mut PipeNetwork,
    resistance_model: &ResistanceModel,
) -> SwitchMatrixStats {
    let n_pipes = network.pipes.len();
    let mut map: FxHashMap<TileSpanCostKey, u32> = FxHashMap::default();
    let mut costs = Vec::new();
    let mut costs_int = Vec::new();
    let mut pipe_entry = std::mem::take(&mut network.span_cost_table.pipe_entry);
    pipe_entry.resize(n_pipes, u32::MAX);

    let tile_usage = compute_tile_usage(network);
    let templates = network.tile_templates.clone();

    let mut sm_cache: FxHashMap<(u16, u16), FxHashMap<u16, u32>> = FxHashMap::default();
    let mut ws = TileLocalWs::new();
    let mut stats = SwitchMatrixStats::default();

    for (pipe_idx, pipe) in network.pipes.iter().enumerate() {
        let (dx, dy) = network.pipe_delta(pipe_idx);
        let tile_type = network.tile_type_by_node[pipe.from];
        let key = TileSpanCostKey {
            tile_type,
            dx: dx.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            dy: dy.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            usage: quantize_usage(pipe.net_count),
            capacity: quantize_capacity(pipe.capacity),
            base_q: quantize_base(pipe.base_resistance),
        };
        let entry = if let Some(&entry) = map.get(&key) {
            entry
        } else {
            let wire_cost = resistance_model.effective_resistance(pipe).max(1e-12);

            // Switch-matrix lookup via per-(tile_type, tile_usage_q) cache.
            let tile_usage_q = tile_usage
                .get(pipe.from)
                .copied()
                .unwrap_or(0);
            let cache_key = (tile_type, tile_usage_q);
            stats.switch_lookups += 1;
            let port_map = if let Some(existing) = sm_cache.get(&cache_key) {
                stats.switch_cache_hits += 1;
                existing
            } else {
                let template = templates
                    .get(tile_type as usize)
                    .cloned()
                    .unwrap_or_else(TileTypeTemplate::empty);
                let t0 = std::time::Instant::now();
                let port_map = compute_switch_matrix_costs(&template, tile_usage_q, &mut ws);
                stats.dijkstra_total_us += t0.elapsed().as_micros();
                stats.dijkstra_calls += 1;
                sm_cache.insert(cache_key, port_map);
                sm_cache.get(&cache_key).unwrap()
            };
            stats.unique_type_usage_pairs = sm_cache.len();

            let sm_cost_int = if let Some(side) = Side::from_delta(dx, dy) {
                let span_bucket = span_bucket_of(dx.abs() + dy.abs());
                port_map
                    .get(&port_key(side, span_bucket))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            let sm_cost_f = sm_cost_int as f64 / DIST_SCALE;

            let total_cost = wire_cost + sm_cost_f;
            let entry = costs.len() as u32;
            costs.push(total_cost);
            costs_int.push(quantize_cost_int(total_cost));
            map.insert(key, entry);
            entry
        };
        pipe_entry[pipe_idx] = entry;
    }

    let misses = costs.len();
    let hits = n_pipes.saturating_sub(misses);
    let epoch = network.span_cost_table.epoch.wrapping_add(1);
    network.span_cost_table = SpanCostTable {
        enabled: true,
        epoch,
        pipe_entry,
        costs,
        costs_int,
        stats: SpanCostStats {
            epoch,
            lookups: n_pipes,
            entries: misses,
            hits,
            misses,
        },
    };
    stats
}

pub fn rebuild_span_cost_table(network: &mut PipeNetwork, resistance_model: &ResistanceModel) {
    let n_pipes = network.pipes.len();
    let mut map: FxHashMap<TileSpanCostKey, u32> = FxHashMap::default();
    let mut costs = Vec::new();
    let mut costs_int = Vec::new();
    let mut pipe_entry = std::mem::take(&mut network.span_cost_table.pipe_entry);
    pipe_entry.resize(n_pipes, u32::MAX);

    for (pipe_idx, pipe) in network.pipes.iter().enumerate() {
        let (dx, dy) = network.pipe_delta(pipe_idx);
        let key = TileSpanCostKey {
            tile_type: network.tile_type_by_node[pipe.from],
            dx: dx.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            dy: dy.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            usage: quantize_usage(pipe.net_count),
            capacity: quantize_capacity(pipe.capacity),
            base_q: quantize_base(pipe.base_resistance),
        };
        let entry = if let Some(&entry) = map.get(&key) {
            entry
        } else {
            let cost = resistance_model.effective_resistance(pipe).max(1e-12);
            let entry = costs.len() as u32;
            costs.push(cost);
            costs_int.push(quantize_cost_int(cost));
            map.insert(key, entry);
            entry
        };
        pipe_entry[pipe_idx] = entry;
    }

    let misses = costs.len();
    let hits = n_pipes.saturating_sub(misses);
    let epoch = network.span_cost_table.epoch.wrapping_add(1);
    network.span_cost_table = SpanCostTable {
        enabled: true,
        epoch,
        pipe_entry,
        costs,
        costs_int,
        stats: SpanCostStats {
            epoch,
            lookups: n_pipes,
            entries: misses,
            hits,
            misses,
        },
    };
}

// =============================================================================
// v2 TwoLevelPip — switch-matrix cost cache built on chipdb TileTypeTemplate
// =============================================================================
//
// `Side`, `span_bucket_of`, `port_key`, `TileTypeTemplate`, `TileLocalWs`,
// and `PIP_BASE_COST_INT` now live in `crate::chipdb::tile_template`; they
// are imported above and also used by `router::lookahead`.

/// Compute switch-matrix cost for every (out_side, out_span_bucket) of a
/// single tile type at a given quantised usage.
///
/// Returns `(out_port_key → u32_cost)`. Runs **one Dijkstra per distinct
/// `out_side`** (up to 4 per call); within each run the super-source seeds
/// all boundary wires on sides **other than** `out_side`, so the measured
/// distance is the cost to *arrive* from some neighbouring-tile entry and
/// then reach the target out-side port. The previous version seeded all
/// boundary wires (including the target), which collapsed every SM cost
/// to 0.
pub fn compute_switch_matrix_costs(
    template: &TileTypeTemplate,
    tile_usage_q: u16,
    ws: &mut TileLocalWs,
) -> FxHashMap<u16, u32> {
    let mut out: FxHashMap<u16, u32> = FxHashMap::default();
    if template.n_wires == 0 || template.n_pips == 0 || template.boundary_wires_all.is_empty() {
        return out;
    }

    // BPR-scaled per-PIP cost. Synthetic capacity = 1 per PIP (each PIP carries
    // at most 1 net in an idealised switch), so the saturation ratio equals
    // tile_usage_q itself. Cap at 1024 to avoid integer overflow on the α·u^β
    // scale factor; in practice tile_usage on sv3 stays well below that.
    let usage = tile_usage_q.min(1024) as f64;
    let scale = 1.0 + bpr_alpha() * usage.powf(bpr_beta());
    let scaled_pip_cost =
        ((template.pip_base_cost as f64 * scale).round() as u32).max(template.pip_base_cost);

    // Collect the set of out_sides that actually have boundary wires in this
    // template. We run one Dijkstra per such side.
    let mut sides_present: Vec<Side> = Vec::new();
    for &port in template.boundary_wires_by_port.keys() {
        let side = match (port >> 8) as u8 {
            0 => Side::North,
            1 => Side::East,
            2 => Side::South,
            3 => Side::West,
            _ => continue,
        };
        if !sides_present.contains(&side) {
            sides_present.push(side);
        }
    }

    for target_side in sides_present {
        // Super-source: all boundary wires EXCEPT those whose port is on
        // `target_side`. If the excluded set empties the super-source (single
        // tile type with only one boundary side), skip — switch-matrix cost is
        // undefined here.
        ws.reset(template.n_wires);
        if ws.buckets.is_empty() {
            ws.buckets.push(Vec::new());
        } else {
            ws.buckets[0].clear();
        }
        let mut seeded = 0usize;
        for (&port, wires) in &template.boundary_wires_by_port {
            let port_side = (port >> 8) as u8;
            if port_side == target_side as u8 {
                continue;
            }
            for &w in wires {
                let wu = w as usize;
                if wu < ws.dist.len() && ws.dist[wu] != 0 {
                    ws.dist[wu] = 0;
                    ws.buckets[0].push(w);
                    seeded += 1;
                }
            }
        }
        if seeded == 0 {
            continue;
        }

        // Bucket-Dial Dijkstra. Uniform edge cost = scaled_pip_cost.
        let mut cur: u32 = 0;
        loop {
            if (cur as usize) >= ws.buckets.len() {
                break;
            }
            let node = match ws.buckets[cur as usize].pop() {
                Some(n) => n,
                None => {
                    cur += 1;
                    continue;
                }
            };
            let node_u = node as usize;
            if ws.dist[node_u] < cur {
                continue;
            }
            let start = template.pip_offsets[node_u] as usize;
            let end = template.pip_offsets[node_u + 1] as usize;
            for &dst in &template.pip_dst[start..end] {
                let du = dst as usize;
                if du >= ws.dist.len() {
                    continue;
                }
                let nd = cur.saturating_add(scaled_pip_cost);
                if nd < ws.dist[du] {
                    ws.dist[du] = nd;
                    let idx = nd as usize;
                    if idx >= ws.buckets.len() {
                        ws.buckets.resize(idx + 1, Vec::new());
                    }
                    ws.buckets[idx].push(dst);
                }
            }
        }

        // Harvest (target_side, span_bucket) entries.
        for (&port, wires) in &template.boundary_wires_by_port {
            let port_side = (port >> 8) as u8;
            if port_side != target_side as u8 {
                continue;
            }
            let mut best = u32::MAX;
            for &w in wires {
                let wu = w as usize;
                if wu < ws.dist.len() && ws.dist[wu] < best {
                    best = ws.dist[wu];
                }
            }
            if best != u32::MAX {
                out.insert(port, best);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chipdb::PIP_BASE_COST_INT;

    #[test]
    fn usage_quantization_is_non_negative() {
        assert_eq!(quantize_usage(-1.0), 0);
        assert_eq!(quantize_usage(1.49), 1);
        assert_eq!(quantize_usage(1.5), 2);
    }

    #[test]
    fn cost_quantization_is_positive() {
        assert_eq!(quantize_cost_int(0.0), 1);
        assert!(quantize_cost_int(2.5) > quantize_cost_int(1.0));
    }

    #[test]
    fn side_from_delta_classifies_cardinals() {
        assert_eq!(Side::from_delta(1, 0), Some(Side::East));
        assert_eq!(Side::from_delta(-1, 0), Some(Side::West));
        assert_eq!(Side::from_delta(0, 1), Some(Side::South));
        assert_eq!(Side::from_delta(0, -1), Some(Side::North));
        assert_eq!(Side::from_delta(0, 0), None);
        // Dominant-axis tiebreak: |dx| == |dy| goes to east/west.
        assert_eq!(Side::from_delta(1, 1), Some(Side::East));
        assert_eq!(Side::from_delta(-1, -1), Some(Side::West));
    }

    #[test]
    fn span_bucket_of_is_monotone() {
        assert!(span_bucket_of(1) <= span_bucket_of(3));
        assert!(span_bucket_of(3) <= span_bucket_of(6));
        assert!(span_bucket_of(6) <= span_bucket_of(12));
        assert!(span_bucket_of(12) <= span_bucket_of(50));
    }

    #[test]
    fn empty_template_has_no_pips() {
        let t = TileTypeTemplate::empty();
        assert_eq!(t.n_wires, 0);
        assert_eq!(t.n_pips, 0);
        assert_eq!(t.pip_offsets, vec![0]);
        assert!(t.pip_dst.is_empty());
        assert!(t.boundary_wires_by_port.is_empty());
        assert!(t.boundary_wires_all.is_empty());
    }

    /// Build a hand-crafted template: 4 wires in a line (0 -> 1 -> 2 -> 3),
    /// with wire 0 and wire 3 as boundary ports (0 on west, 3 on east), each
    /// in span-bucket 0.
    fn make_toy_template() -> TileTypeTemplate {
        let n_wires = 4;
        let pip_offsets = vec![0u32, 1, 2, 3, 3];
        let pip_dst = vec![1u32, 2, 3];
        let mut boundary_wires_by_port: FxHashMap<u16, Vec<u32>> = FxHashMap::default();
        boundary_wires_by_port.insert(port_key(Side::West, 0), vec![0]);
        boundary_wires_by_port.insert(port_key(Side::East, 0), vec![3]);
        TileTypeTemplate {
            n_wires,
            n_pips: 3,
            pip_offsets,
            pip_dst,
            pip_base_cost: PIP_BASE_COST_INT,
            boundary_wires_by_port,
            boundary_wires_all: vec![0, 3],
        }
    }

    #[test]
    fn switch_matrix_zero_usage_walks_pips_to_reach_opposite_side() {
        // Toy: 4-wire chain 0 -> 1 -> 2 -> 3, wire 0 is West port, wire 3 is
        // East port. Going from West entry-side super-source (seeds wire 0 at
        // dist 0) to East exit port (wire 3) takes 3 PIP hops at base cost
        // PIP_BASE_COST_INT, so 3 × 10 = 30.
        let t = make_toy_template();
        let mut ws = TileLocalWs::new();
        let costs = compute_switch_matrix_costs(&t, 0, &mut ws);
        let east = port_key(Side::East, 0);
        let west = port_key(Side::West, 0);
        // Cost to reach East exit via PIPs from West entry:
        assert_eq!(costs.get(&east).copied(), Some(30));
        // Cost to reach West exit via PIPs from East entry: the template is a
        // one-way chain, so this should be unreachable (no entry in the map).
        assert_eq!(costs.get(&west).copied(), None);
    }

    #[test]
    fn switch_matrix_cost_monotone_in_tile_usage() {
        let t = make_toy_template();
        let mut ws = TileLocalWs::new();
        let c_u0 = compute_switch_matrix_costs(&t, 0, &mut ws);
        let c_u10 = compute_switch_matrix_costs(&t, 10, &mut ws);
        let c_u50 = compute_switch_matrix_costs(&t, 50, &mut ws);
        let east = port_key(Side::East, 0);
        let c0 = c_u0.get(&east).copied().unwrap();
        let c10 = c_u10.get(&east).copied().unwrap();
        let c50 = c_u50.get(&east).copied().unwrap();
        assert!(c0 <= c10, "c0={} c10={}", c0, c10);
        assert!(c10 <= c50, "c10={} c50={}", c10, c50);
        // At u=50, BPR factor is 1 + 0.05·50⁴ ≈ 312,501 — massive, cost
        // should be orders of magnitude above the base 30.
        assert!(c50 > c0 * 100, "c50={} should dwarf c0={}", c50, c0);
    }

    #[test]
    fn switch_matrix_cost_returns_empty_for_empty_template() {
        let t = TileTypeTemplate::empty();
        let mut ws = TileLocalWs::new();
        let costs = compute_switch_matrix_costs(&t, 0, &mut ws);
        assert!(costs.is_empty());
    }

    #[test]
    fn switch_matrix_cost_handles_missing_boundary_wires() {
        // Template with PIPs but no boundary classifications (e.g. a routing
        // tile whose wires all live entirely inside the tile). Should return
        // an empty cost map without panicking.
        let t = TileTypeTemplate {
            n_wires: 3,
            n_pips: 2,
            pip_offsets: vec![0, 1, 2, 2],
            pip_dst: vec![1, 2],
            pip_base_cost: PIP_BASE_COST_INT,
            boundary_wires_by_port: FxHashMap::default(),
            boundary_wires_all: Vec::new(),
        };
        let mut ws = TileLocalWs::new();
        let costs = compute_switch_matrix_costs(&t, 5, &mut ws);
        assert!(costs.is_empty());
    }
}

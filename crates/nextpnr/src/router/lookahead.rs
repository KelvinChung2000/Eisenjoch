//! Routing-cost lookahead, per driver-BEL class.
//!
//! Each net's reachable subgraph depends on what drives it. A SLICE-
//! driven data net cannot reach the GCLK backbone; a BUFG-driven clock
//! net can. So the lookahead computes one rate per driver-BEL type,
//! based on a fabric-reachability BFS from that type's output pins.
//!
//! ## Build
//!
//! For each unique driver-BEL type seen in the chipdb (e.g. SLICEL,
//! SLICEM, BUFG, IOB, RAMB36E1, DSP48E1):
//!
//! 1. Seed: every output pin wire of every BEL of that type, in every
//!    tile type that has one.
//! 2. Closure: BFS forward through (intra-tile pips) and (inter-tile
//!    node-share peers) until fixpoint. Result: the per-tile-type set
//!    of wires fabric-reachable from this driver class.
//! 3. Rate: `min over pips P with src_wire(P) ∈ reachable[class]
//!    of pip_cost(P) / max_node_share_offset(dst_wire(P))`.
//!    Stored as a (num, den) rational so floor division stays
//!    admissible.
//!
//! ## Query
//!
//! `estimate_delay(src, dst, class) = manhattan(src, dst) × rate[class]`.
//! Floor-divided integer math. Admissible because every fabric-reachable
//! pip from this class achieves at most this rate per tile, so any
//! actual route that respects class reachability pays at least this
//! per tile of advance.

use crate::chipdb::{ChipDb, PipId, WireId};
use crate::context::Context;
use crate::read_packed;
use crate::timing::DelayT;
use rustc_hash::FxHashMap;

use super::astar::default_pip_cost;

/// Compact identifier for a driver-BEL class. Indexes `Lookahead::rates`.
pub type LookaheadClass = u32;

/// Sentinel for "no class info available" (e.g. constant-driven net or
/// driver BEL not yet placed). Treated as: no class-based gating in
/// [`Lookahead::is_reachable`], and the loosest-rate class for h.
pub const UNKNOWN_CLASS: LookaheadClass = LookaheadClass::MAX;

pub struct Lookahead {
    /// Per-class min per-tile rate, as (num, den).
    rates: Vec<(i64, i64)>,
    /// Driver-BEL type constid → class index.
    class_of_bel_type: FxHashMap<i32, LookaheadClass>,
    /// Class to use when the driver BEL type is unknown — the loosest
    /// (smallest rate) class, so unknown drivers stay admissible.
    default_class: LookaheadClass,
    /// Diagnostic names per class.
    class_names: Vec<String>,
    /// `reachable[class][tile_type][wire_idx]`: whether this wire is in
    /// the class's fabric-reachability closure. Used by the router to
    /// hard-gate A* expansion — a fabric net cannot traverse clock-
    /// only wires (and vice-versa) regardless of cost.
    reachable: Vec<Vec<Vec<bool>>>,
    /// Precomputed per-pip cost in the wire-RC + amortized-mux model:
    /// `cost = floor(base * sqrt(span)) + floor(orig_pip_delay / span)`,
    /// where `base` is the cheapest single-tile pip's original delay and
    /// `span` is the dst-wire's electrical-net diameter. Indexed
    /// `[tile_type][pip_idx]`. Queried by the router via [`Lookahead::pip_cost`]
    /// so that the heuristic and the actual A* path cost stay in the
    /// same units (admissibility).
    pip_costs: Vec<Vec<i64>>,
}

impl Lookahead {
    pub fn build(ctx: &Context) -> Self {
        let chipdb = ctx.chipdb();
        let w = chipdb.width();
        let h = chipdb.height();
        let num_tile_types = chipdb.num_tile_types();

        // Pick one interior-preferred rep per tile type so longline
        // pip dst-wire spans aren't truncated by the chip edge.
        const REP_MARGIN: i32 = 13;
        let is_interior = |tile: i32| -> bool {
            let (x, y) = chipdb.tile_xy(tile);
            x >= REP_MARGIN && x < w - REP_MARGIN && y >= REP_MARGIN && y < h - REP_MARGIN
        };
        let mut rep_of_type: Vec<Option<i32>> = vec![None; num_tile_types];
        for tile in 0..(w * h) {
            let tt = chipdb.tile_type_index(tile) as usize;
            match rep_of_type[tt] {
                None => rep_of_type[tt] = Some(tile),
                Some(curr) if !is_interior(curr) && is_interior(tile) => {
                    rep_of_type[tt] = Some(tile);
                }
                _ => {}
            }
        }

        // Per-tt forward pip adjacency and node-share adjacency between
        // rep tiles (tt, wire_idx) ↔ (tt, wire_idx). Used by every
        // class's BFS — built once.
        let mut adj_per_tt: Vec<Vec<Vec<i32>>> = Vec::with_capacity(num_tile_types);
        let mut wires_per_tt: Vec<usize> = Vec::with_capacity(num_tile_types);
        for tt in 0..num_tile_types {
            let tt_data = chipdb.tile_type_by_index(tt as i32);
            let n_wires = tt_data.wires.len();
            wires_per_tt.push(n_wires);
            let mut adj = vec![Vec::new(); n_wires];
            for pip in tt_data.pips.get() {
                let src: i32 = unsafe { read_packed!(*pip, src_wire) };
                let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
                if src >= 0 && (src as usize) < n_wires && dst >= 0 {
                    adj[src as usize].push(dst);
                }
            }
            adj_per_tt.push(adj);
        }

        let mut node_share: FxHashMap<(usize, i32), Vec<(usize, i32)>> = FxHashMap::default();
        for tt in 0..num_tile_types {
            let Some(rep) = rep_of_type[tt] else { continue };
            for wi in 0..wires_per_tt[tt] {
                let wire = WireId::new(rep, wi as i32);
                let mut peers: Vec<(usize, i32)> = Vec::new();
                chipdb.node_wires_cb(wire, |peer| {
                    let ptt = chipdb.tile_type_index(peer.tile()) as usize;
                    let pwi = peer.index();
                    if ptt == tt && pwi == wi as i32 {
                        return;
                    }
                    peers.push((ptt, pwi));
                });
                if !peers.is_empty() {
                    node_share.insert((tt, wi as i32), peers);
                }
            }
        }

        // Group output pin seeds by driver BEL type constid.
        let mut class_seeds: FxHashMap<i32, Vec<(usize, i32)>> = FxHashMap::default();
        let mut class_name_map: FxHashMap<i32, String> = FxHashMap::default();
        for tt in 0..num_tile_types {
            if rep_of_type[tt].is_none() {
                continue;
            }
            let bels = chipdb.tile_type_by_index(tt as i32).bels.get();
            for bel in bels {
                let bel_type_id: i32 = bel.bel_type();
                let mut had_output = false;
                for pin in bel.pins.get() {
                    if pin.dir() != 1 {
                        continue;
                    }
                    let wi = pin.wire();
                    if wi < 0 || (wi as usize) >= wires_per_tt[tt] {
                        continue;
                    }
                    class_seeds.entry(bel_type_id).or_default().push((tt, wi));
                    had_output = true;
                }
                if had_output {
                    class_name_map.entry(bel_type_id).or_insert_with(|| {
                        chipdb
                            .constid_str(bel_type_id)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("?{}", bel_type_id))
                    });
                }
            }
        }

        // For each class, run BFS over (pips, node-share) to closure;
        // then compute min(cost/span) over reachable pips.
        // Precompute once, used by every class's rate scan: per-(tt,
        // pip_idx) the (orig_cost, span, src_wire) needed to score a pip.
        // src_wire = -1 marks "skip" (invalid pip).
        let mut pip_raw: Vec<Vec<(i64, i64, i32)>> = Vec::with_capacity(num_tile_types);
        for tt in 0..num_tile_types {
            let Some(rep_tile) = rep_of_type[tt] else {
                pip_raw.push(Vec::new());
                continue;
            };
            let (rep_x, rep_y) = chipdb.tile_xy(rep_tile);
            let pips = chipdb.tile_type_by_index(tt as i32).pips.get();
            let mut data = Vec::with_capacity(pips.len());
            for (pip_idx, pip) in pips.iter().enumerate() {
                let src: i32 = unsafe { read_packed!(*pip, src_wire) };
                let dst_wi: i32 = unsafe { read_packed!(*pip, dst_wire) };
                if dst_wi < 0 {
                    data.push((0, 1, -1));
                    continue;
                }
                let pid = PipId::new(rep_tile, pip_idx as i32);
                let cost = default_pip_cost(ctx, pid).max(1) as i64;
                let dst_wire = WireId::new(rep_tile, dst_wi);
                let mut max_span: i64 = 0;
                chipdb.node_wires_cb(dst_wire, |peer| {
                    let (px, py) = chipdb.tile_xy(peer.tile());
                    let s = ((px - rep_x).abs() + (py - rep_y).abs()) as i64;
                    if s > max_span {
                        max_span = s;
                    }
                });
                data.push((cost, max_span.max(1), src));
            }
            pip_raw.push(data);
        }

        // Apply the wire-RC + amortized-mux cost model:
        //   new_cost = floor(base * sqrt(span)) + floor(orig_cost / span)
        // where `base` is the cheapest span=1 pip's original cost across
        // the whole chipdb. The wire-RC term inflates long-span pips so
        // they no longer look "almost free" to A*; the amortized-mux term
        // keeps the prjxray timing as a per-tile contribution. Both
        // router and lookahead read these costs (via `pip_cost`), so g
        // and h stay in the same units and h remains admissible.
        let mut base_1span_cost: i64 = i64::MAX;
        for tt in 0..num_tile_types {
            for &(cost, span, src) in &pip_raw[tt] {
                if src < 0 || span != 1 {
                    continue;
                }
                if cost < base_1span_cost {
                    base_1span_cost = cost;
                }
            }
        }
        if base_1span_cost == i64::MAX {
            base_1span_cost = 1;
        }
        eprintln!(
            "Lookahead pip-cost model: cost = {}*sqrt(span) + orig_delay/span",
            base_1span_cost,
        );

        let mut pip_costs: Vec<Vec<i64>> = Vec::with_capacity(num_tile_types);
        let mut pip_rate_data: Vec<Vec<(i64, i64, i32)>> = Vec::with_capacity(num_tile_types);
        for tt in 0..num_tile_types {
            let raw = &pip_raw[tt];
            let mut costs: Vec<i64> = Vec::with_capacity(raw.len());
            let mut rated: Vec<(i64, i64, i32)> = Vec::with_capacity(raw.len());
            for &(orig_cost, span, src) in raw {
                if src < 0 {
                    costs.push(1);
                    rated.push((0, 1, -1));
                    continue;
                }
                let wire_rc = (base_1span_cost as f64 * (span as f64).sqrt()).floor() as i64;
                let amort_mux = orig_cost / span;
                let new_cost = (wire_rc + amort_mux).max(1);
                costs.push(new_cost);
                rated.push((new_cost, span, src));
            }
            pip_costs.push(costs);
            pip_rate_data.push(rated);
        }

        let mut bel_types: Vec<i32> = class_seeds.keys().copied().collect();
        bel_types.sort();
        let mut rates: Vec<(i64, i64)> = Vec::with_capacity(bel_types.len());
        let mut class_of_bel_type: FxHashMap<i32, LookaheadClass> = FxHashMap::default();
        let mut class_names: Vec<String> = Vec::with_capacity(bel_types.len());
        let mut reachable_per_class: Vec<Vec<Vec<bool>>> = Vec::with_capacity(bel_types.len());

        for (cls_idx, &bel_type) in bel_types.iter().enumerate() {
            let cls = cls_idx as LookaheadClass;
            class_of_bel_type.insert(bel_type, cls);
            let name = class_name_map[&bel_type].clone();

            // Per-tile-type bitmap of reachable wires. Vec<bool> is fine
            // here — total size is num_classes × num_tile_types ×
            // wires_per_tt, on the order of a few hundred kB for xc7.
            let mut reach_bits: Vec<Vec<bool>> = (0..num_tile_types)
                .map(|tt| vec![false; wires_per_tt[tt]])
                .collect();
            let mut queue: Vec<(usize, i32)> = Vec::new();
            for &(tt, wi) in &class_seeds[&bel_type] {
                if (wi as usize) < reach_bits[tt].len() && !reach_bits[tt][wi as usize] {
                    reach_bits[tt][wi as usize] = true;
                    queue.push((tt, wi));
                }
            }
            while let Some((tt, wi)) = queue.pop() {
                let adj = &adj_per_tt[tt];
                if (wi as usize) < adj.len() {
                    for &dst in &adj[wi as usize] {
                        let du = dst as usize;
                        if du < reach_bits[tt].len() && !reach_bits[tt][du] {
                            reach_bits[tt][du] = true;
                            queue.push((tt, dst));
                        }
                    }
                }
                if let Some(peers) = node_share.get(&(tt, wi)) {
                    for &(ptt, pwi) in peers {
                        let pu = pwi as usize;
                        if pu < reach_bits[ptt].len() && !reach_bits[ptt][pu] {
                            reach_bits[ptt][pu] = true;
                            queue.push((ptt, pwi));
                        }
                    }
                }
            }

            let mut best_num: i64 = i64::MAX;
            let mut best_den: i64 = 1;
            let mut pips_in_class = 0usize;
            for tt in 0..num_tile_types {
                for &(cost, span, src) in &pip_rate_data[tt] {
                    if src < 0 {
                        continue;
                    }
                    let su = src as usize;
                    if su >= reach_bits[tt].len() || !reach_bits[tt][su] {
                        continue;
                    }
                    pips_in_class += 1;
                    if cost.saturating_mul(best_den) < best_num.saturating_mul(span) {
                        best_num = cost;
                        best_den = span;
                    }
                }
            }
            if best_num == i64::MAX {
                best_num = 1;
                best_den = 1;
            }
            eprintln!(
                "Lookahead class {} ({}): {} reachable pips, rate = {}/{} ≈ {:.3}/tile",
                cls,
                name,
                pips_in_class,
                best_num,
                best_den,
                best_num as f64 / best_den as f64,
            );
            rates.push((best_num, best_den));
            class_names.push(name);
            reachable_per_class.push(reach_bits);
        }

        // Default class = the one with the SMALLEST rate (loosest h),
        // so unknown drivers stay admissible by construction.
        let mut default_class: LookaheadClass = 0;
        if !rates.is_empty() {
            let mut best = rates[0];
            for (i, &r) in rates.iter().enumerate() {
                if r.0.saturating_mul(best.1) < best.0.saturating_mul(r.1) {
                    best = r;
                    default_class = i as LookaheadClass;
                }
            }
        }

        if rates.is_empty() {
            // No BELs at all in the chipdb — degenerate case. One class
            // with rate 1/1 and an empty reachable set.
            rates.push((1, 1));
            class_names.push("DEFAULT".into());
            reachable_per_class.push(
                (0..num_tile_types)
                    .map(|tt| vec![true; wires_per_tt[tt]])
                    .collect(),
            );
        }

        eprintln!(
            "Lookahead: {} driver classes, default={} ({})",
            rates.len(),
            default_class,
            class_names[default_class as usize],
        );

        Self {
            rates,
            class_of_bel_type,
            default_class,
            class_names,
            reachable: reachable_per_class,
            pip_costs,
        }
    }

    /// Per-pip cost in the wire-RC + amortized-mux model. Returns the
    /// canonical `default_pip_cost` value for any pip whose tile-type or
    /// pip index falls outside the precomputed table (degenerate /
    /// out-of-chip pips). Router callers use this so that g and h are
    /// in the same units.
    #[inline]
    pub fn pip_cost(&self, chipdb: &ChipDb, pip: PipId) -> DelayT {
        let tt = chipdb.tile_type_index(pip.tile()) as usize;
        let pi = pip.index() as usize;
        match self.pip_costs.get(tt).and_then(|v| v.get(pi)) {
            Some(&c) => c.clamp(1, DelayT::MAX as i64) as DelayT,
            None => 1,
        }
    }

    /// Map a driver BEL type constid to a class index. Returns `None`
    /// for unrecognized types; callers should treat that as
    /// [`UNKNOWN_CLASS`] (no gate, default-rate h) rather than picking
    /// `default_class`, since `default_class`'s reachable set is one
    /// specific driver type's, not a superset of all types'.
    pub fn class_for_bel_type(&self, bel_type: i32) -> Option<LookaheadClass> {
        self.class_of_bel_type.get(&bel_type).copied()
    }

    pub fn default_class(&self) -> LookaheadClass {
        self.default_class
    }

    pub fn class_name(&self, class: LookaheadClass) -> &str {
        self.class_names
            .get(class as usize)
            .map(|s| s.as_str())
            .unwrap_or("?")
    }

    /// Whether `wire` is in `class`'s fabric-reachability closure. The
    /// router uses this to hard-gate A* expansion: a fabric-driven net
    /// cannot legally land on clock-only wires regardless of cost, so
    /// they shouldn't be considered as candidates at all.
    #[inline]
    pub fn is_reachable(&self, chipdb: &ChipDb, wire: WireId, class: LookaheadClass) -> bool {
        let cls = class as usize;
        let Some(per_tt) = self.reachable.get(cls) else {
            return true;
        };
        let tt = chipdb.tile_type_index(wire.tile()) as usize;
        let Some(bits) = per_tt.get(tt) else {
            return true;
        };
        let wi = wire.index() as usize;
        bits.get(wi).copied().unwrap_or(false)
    }

    /// Admissible lower bound on routing cost from `src` to `dst` for a
    /// net whose driver belongs to `class`. `UNKNOWN_CLASS` falls back
    /// to the loosest-rate (default) class so unknown drivers stay
    /// admissible against any actual driver class.
    pub fn estimate_delay(
        &self,
        chipdb: &ChipDb,
        src: WireId,
        dst: WireId,
        class: LookaheadClass,
    ) -> DelayT {
        let (sx, sy) = chipdb.tile_xy(src.tile());
        let (dxt, dyt) = chipdb.tile_xy(dst.tile());
        let manhattan = ((dxt - sx).abs() + (dyt - sy).abs()) as i64;
        let idx = if (class as usize) < self.rates.len() {
            class as usize
        } else {
            self.default_class as usize
        };
        let (num, den) = self.rates[idx];
        let h = manhattan.saturating_mul(num) / den.max(1);
        h.clamp(0, DelayT::MAX as i64) as DelayT
    }
}

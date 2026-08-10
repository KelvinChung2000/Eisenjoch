//! Per-tile-type PIP graph + boundary-wire classification.
//!
//! Shared between `placer::opt_trans` (switch-matrix cost cache) and
//! `router::lookahead` (per-class span heuristic). Both consumers need:
//!   * A CSR of intra-tile PIP connectivity keyed by wire index.
//!   * Each boundary wire's dominant-axis `Side` and `span_bucket`
//!     (one-hop reach categorised by the chipdb's natural span tiers).
//!   * A pooled workspace for bucket-Dial Dijkstra over the CSR.
//!
//! Span buckets `{0-1, 2-3, 4-6, 7-12, >12}` match the per-wire node-span
//! histogram on xc7_large (single/double/quad-hex/longline/chip-wide).

use rustc_hash::FxHashMap;

use crate::chipdb::ChipDb;
use crate::read_packed;

/// Cardinal side of a tile boundary port. Packed with a span bucket into a
/// u16 for compact cache keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Side {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

impl Side {
    /// Classify a `(dx, dy)` boundary delta into a cardinal side.
    /// Dominant-axis rule: larger `|dx|` wins; ties go to east/west.
    pub fn from_delta(dx: i32, dy: i32) -> Option<Self> {
        if dx == 0 && dy == 0 {
            return None;
        }
        if dx.abs() >= dy.abs() {
            if dx > 0 {
                Some(Side::East)
            } else {
                Some(Side::West)
            }
        } else if dy > 0 {
            Some(Side::South)
        } else {
            Some(Side::North)
        }
    }
}

/// Bucket a span into one of five categories: 1 / 2-3 / 4-6 / 7-12 / >12.
#[inline]
pub fn span_bucket_of(span: i32) -> u8 {
    match span {
        0 => 0,
        1 => 0,
        2 | 3 => 1,
        4..=6 => 2,
        7..=12 => 3,
        _ => 4,
    }
}

/// Pack `(side, span_bucket)` into a cache-key nibble.
#[inline]
pub fn port_key(side: Side, span_bucket: u8) -> u16 {
    ((side as u16) << 8) | span_bucket as u16
}

/// Per-PIP base cost in `DIST_SCALE`-integer units for the opt_trans v2
/// switch-matrix cache. A single PIP hop ≈ 0.1 × one-tile wire cost.
pub const PIP_BASE_COST_INT: u32 = 10;

/// Per-tile-type internal PIP graph + boundary-wire classification.
///
/// Memory: `O(n_wires + n_pips)` per tile type. On xc7_large the worst type
/// (BRAM) has ~4.4k wires and ~16k PIPs — under 100 KB per type, and ≤ 1 MB
/// total across all tile types.
#[derive(Clone, Debug, Default)]
pub struct TileTypeTemplate {
    pub n_wires: usize,
    pub n_pips: usize,
    /// CSR offsets: for wire `w`, its outgoing PIPs are
    /// `pip_dst[pip_offsets[w]..pip_offsets[w+1]]`.
    pub pip_offsets: Vec<u32>,
    pub pip_dst: Vec<u32>,
    /// Base cost per PIP hop in `DIST_SCALE`-integer units (uniform for
    /// v2 phase 1).
    pub pip_base_cost: u32,
    /// Boundary wires grouped by `port_key(side, span_bucket)`.
    pub boundary_wires_by_port: FxHashMap<u16, Vec<u32>>,
    /// All boundary wires (any side) — used as the super-source for
    /// entry-side-averaged tile-local Dijkstra.
    pub boundary_wires_all: Vec<u32>,
}

impl TileTypeTemplate {
    pub fn empty() -> Self {
        Self {
            n_wires: 0,
            n_pips: 0,
            pip_offsets: vec![0],
            pip_dst: Vec::new(),
            pip_base_cost: PIP_BASE_COST_INT,
            boundary_wires_by_port: FxHashMap::default(),
            boundary_wires_all: Vec::new(),
        }
    }

    /// Build from chipdb for a single tile type. `representative_tile` is any
    /// tile instance of that type; used to resolve wire node shapes.
    pub fn from_chipdb(chipdb: &ChipDb, tt_idx: i32, representative_tile: i32) -> Self {
        let tt = chipdb.tile_type_by_index(tt_idx);
        let n_wires = tt.wires.len();
        let pips = tt.pips.get();
        let n_pips = pips.len();

        // Count outgoing PIPs per source wire, then CSR-assemble.
        let mut out_deg = vec![0u32; n_wires];
        for pip in pips {
            let src: i32 = unsafe { read_packed!(*pip, src_wire) };
            let s = src as usize;
            if s < n_wires {
                out_deg[s] += 1;
            }
        }
        let mut pip_offsets = Vec::with_capacity(n_wires + 1);
        pip_offsets.push(0u32);
        let mut running = 0u32;
        for &d in &out_deg {
            running += d;
            pip_offsets.push(running);
        }
        let mut cursors = pip_offsets.clone();
        let mut pip_dst = vec![0u32; running as usize];
        for pip in pips {
            let src: i32 = unsafe { read_packed!(*pip, src_wire) };
            let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
            let s = src as usize;
            if s < n_wires {
                let slot = cursors[s] as usize;
                cursors[s] = slot as u32 + 1;
                pip_dst[slot] = dst.max(0) as u32;
            }
        }

        // Boundary classification via wire_node_shape — each wire whose node
        // shape extends outside the representative tile (any non-zero
        // `(dx, dy)`) is a boundary wire. Dominant-axis delta sets the side;
        // span bucket follows from the max `|dx|+|dy|`.
        let mut boundary_wires_by_port: FxHashMap<u16, Vec<u32>> = FxHashMap::default();
        let mut boundary_all = Vec::new();
        for wire_idx in 0..n_wires {
            let Some(shape) = chipdb.wire_node_shape(representative_tile, wire_idx) else {
                continue;
            };
            let mut is_boundary = false;
            let mut best_side: Option<Side> = None;
            let mut best_span = 0i32;
            for tw in shape.tile_wires.get() {
                let dx: i16 = unsafe { read_packed!(*tw, dx) };
                let dy: i16 = unsafe { read_packed!(*tw, dy) };
                let dx = dx as i32;
                let dy = dy as i32;
                let span = dx.abs() + dy.abs();
                if span == 0 {
                    continue;
                }
                is_boundary = true;
                if span > best_span {
                    if let Some(s) = Side::from_delta(dx, dy) {
                        best_side = Some(s);
                        best_span = span;
                    }
                }
            }
            if is_boundary {
                boundary_all.push(wire_idx as u32);
                if let Some(side) = best_side {
                    let key = port_key(side, span_bucket_of(best_span));
                    boundary_wires_by_port
                        .entry(key)
                        .or_default()
                        .push(wire_idx as u32);
                }
            }
        }

        Self {
            n_wires,
            n_pips,
            pip_offsets,
            pip_dst,
            pip_base_cost: PIP_BASE_COST_INT,
            boundary_wires_by_port,
            boundary_wires_all: boundary_all,
        }
    }
}

/// Per-thread workspace for tile-local Dijkstra. Pooled via `thread_local!`
/// at the call site to avoid per-miss allocation.
#[derive(Debug)]
pub struct TileLocalWs {
    /// Shortest-path distance per wire (`u32::MAX` = unreached).
    pub dist: Vec<u32>,
    /// Bucket heap: outer `Vec` indexed by integer distance, inner `Vec` is
    /// the list of wires at that distance.
    pub buckets: Vec<Vec<u32>>,
}

impl TileLocalWs {
    pub fn new() -> Self {
        Self {
            dist: Vec::new(),
            buckets: Vec::new(),
        }
    }

    pub fn reset(&mut self, n_wires: usize) {
        self.dist.clear();
        self.dist.resize(n_wires, u32::MAX);
        for b in &mut self.buckets {
            b.clear();
        }
    }
}

impl Default for TileLocalWs {
    fn default() -> Self {
        Self::new()
    }
}

//! Precomputed routing cost lookahead table.
//!
//! For each wire type and Manhattan offset (dx, dy), stores the minimum
//! routing cost to reach a wire at that offset. This provides an O(1)
//! admissible heuristic for A*, replacing the naive Manhattan distance
//! estimate that degenerates on architectures with many PIPs per tile.
//!
//! The table is computed once at router startup by running Dijkstra from
//! representative wires of each type at a sample tile.

use crate::chipdb::{ChipDb, WireId, PipId};
use crate::timing::DelayT;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Compact wire class: groups wires by their routing capability.
/// Wires of the same class at different tiles have equivalent routing reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WireClass(pub u16);

/// Precomputed lookahead table.
pub struct Lookahead {
    /// Map from (tile_type_index, wire_index_in_tile) → WireClass.
    wire_class_map: Vec<Vec<WireClass>>,
    /// Cost table: cost[wire_class][dx + max_dx][dy + max_dy].
    /// Indexed by WireClass, then by offset from source tile.
    cost_tables: Vec<Vec<Vec<DelayT>>>,
    /// Grid dimensions for offset clamping.
    max_dx: i32,
    max_dy: i32,
    /// Number of wire classes.
    num_classes: usize,
}

#[derive(Clone)]
struct DijkEntry {
    wire: WireId,
    cost: DelayT,
}

impl Eq for DijkEntry {}
impl PartialEq for DijkEntry {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Ord for DijkEntry {
    fn cmp(&self, other: &Self) -> Ordering { other.cost.cmp(&self.cost) }
}
impl PartialOrd for DijkEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Lookahead {
    /// Build the lookahead table from the chipdb.
    ///
    /// For each wire type in a representative tile, run Dijkstra to find
    /// the minimum cost to reach each (dx, dy) offset. The table covers
    /// offsets up to ±max_range tiles.
    pub fn build(chipdb: &ChipDb, speed_grade_idx: usize, max_range: i32) -> Self {
        let w = chipdb.width();
        let h = chipdb.height();

        // Find a representative tile near the center with many PIPs.
        let center_x = w / 2;
        let center_y = h / 2;
        let mut best_tile = 0i32;
        let mut best_pips = 0usize;
        for dy in -5..=5 {
            for dx in -5..=5 {
                let tx = (center_x + dx).clamp(0, w - 1);
                let ty = (center_y + dy).clamp(0, h - 1);
                let tile = ty * w + tx;
                let tt = chipdb.tile_type(tile);
                let n_pips = tt.pips.len();
                if n_pips > best_pips {
                    best_pips = n_pips;
                    best_tile = tile;
                }
            }
        }

        let (rep_x, rep_y) = chipdb.tile_xy(best_tile);
        let tt = chipdb.tile_type(best_tile);
        let n_wires = tt.wires.len();

        eprintln!(
            "Lookahead: representative tile ({},{}) with {} wires, {} PIPs",
            rep_x, rep_y, n_wires, best_pips,
        );

        // Each wire in the representative tile is its own class.
        // Wire i in the tile → WireClass(i).
        let num_classes = n_wires;

        // Build wire_class_map: for each tile type, map wire index → WireClass.
        // For the representative tile type, it's identity. For other tile types,
        // we map by wire name matching (same name = same class).
        let num_tile_types = chipdb.num_tile_types();

        // Collect wire names for the representative tile type.
        let mut rep_wire_names: Vec<i32> = Vec::with_capacity(n_wires);
        for wi in 0..n_wires {
            let wire = WireId::new(best_tile, wi as i32);
            let info = chipdb.wire_info(wire);
            let name: i32 = unsafe { crate::read_packed!(*info, name) };
            rep_wire_names.push(name);
        }

        // Build name→class map.
        let mut name_to_class: FxHashMap<i32, WireClass> = FxHashMap::default();
        for (i, &name) in rep_wire_names.iter().enumerate() {
            name_to_class.insert(name, WireClass(i as u16));
        }

        // For each tile type, map wire indices to classes.
        let mut wire_class_map: Vec<Vec<WireClass>> = Vec::with_capacity(num_tile_types);
        let unknown_class = WireClass(u16::MAX);
        for tt_idx in 0..num_tile_types {
            let tt = chipdb.tile_type_by_index(tt_idx as i32);
            let nw = tt.wires.len();
            let mut classes = vec![unknown_class; nw];
            // We need a tile of this type to read wire names.
            // Find one by scanning (cache for performance).
            let mut found_tile = None;
            for tile in 0..(w * h) {
                if chipdb.tile_type_index(tile) as usize == tt_idx {
                    found_tile = Some(tile);
                    break;
                }
            }
            if let Some(tile) = found_tile {
                for wi in 0..nw {
                    let wire = WireId::new(tile, wi as i32);
                    let info = chipdb.wire_info(wire);
                    let name: i32 = unsafe { crate::read_packed!(*info, name) };
                    if let Some(&cls) = name_to_class.get(&name) {
                        classes[wi] = cls;
                    }
                }
            }
            wire_class_map.push(classes);
        }

        // Run Dijkstra from each wire in the representative tile.
        let max_dx = max_range;
        let max_dy = max_range;
        let table_w = (2 * max_dx + 1) as usize;
        let table_h = (2 * max_dy + 1) as usize;

        // Initialize cost tables with infinity.
        let mut cost_tables: Vec<Vec<Vec<DelayT>>> = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            cost_tables.push(vec![vec![DelayT::MAX; table_h]; table_w]);
        }

        // Get speed grade for pip delay computation.
        let sg = chipdb.speed_grade(speed_grade_idx);

        // Run Dijkstra from each wire class (each wire in the rep tile).
        for src_wi in 0..n_wires {
            let src_wire = WireId::new(best_tile, src_wi as i32);

            let mut heap: BinaryHeap<DijkEntry> = BinaryHeap::new();
            let mut visited: FxHashMap<WireId, DelayT> = FxHashMap::default();

            heap.push(DijkEntry { wire: src_wire, cost: 0 });
            visited.insert(src_wire, 0);

            // Also seed with node wires (multi-tile nodes).
            chipdb.node_wires_cb(src_wire, |nw| {
                let (nx, ny) = chipdb.tile_xy(nw.tile());
                let hop_dx = (nx - rep_x).abs();
                let hop_dy = (ny - rep_y).abs();
                let wire_cost = (hop_dx + hop_dy) as DelayT; // small cost for node hops
                if !visited.contains_key(&nw) || wire_cost < visited[&nw] {
                    visited.insert(nw, wire_cost);
                    heap.push(DijkEntry { wire: nw, cost: wire_cost });
                }
            });

            // Dijkstra with limited expansion (don't explore entire chip).
            let mut expanded = 0usize;
            const MAX_EXPAND: usize = 50_000;

            while let Some(entry) = heap.pop() {
                if expanded >= MAX_EXPAND { break; }

                if let Some(&prev) = visited.get(&entry.wire) {
                    if entry.cost > prev { continue; }
                }

                // Record cost at this tile offset.
                let (wx, wy) = chipdb.tile_xy(entry.wire.tile());
                let dx = wx - rep_x;
                let dy = wy - rep_y;
                if dx.abs() <= max_dx && dy.abs() <= max_dy {
                    let ix = (dx + max_dx) as usize;
                    let iy = (dy + max_dy) as usize;
                    let table = &mut cost_tables[src_wi];
                    if entry.cost < table[ix][iy] {
                        table[ix][iy] = entry.cost;
                    }
                }

                // Expand node wires.
                chipdb.node_wires_cb(entry.wire, |nw| {
                    let (nx, ny) = chipdb.tile_xy(nw.tile());
                    let hop_cost = ((nx - wx).abs() + (ny - wy).abs()) as DelayT;
                    let new_cost = entry.cost + hop_cost;
                    if !visited.contains_key(&nw) || new_cost < visited[&nw] {
                        visited.insert(nw, new_cost);
                        heap.push(DijkEntry { wire: nw, cost: new_cost });
                    }
                });

                // Expand PIP destinations.
                let wire_info = chipdb.wire_info(entry.wire);
                let downhill = wire_info.pips_downhill.get();
                for &pip_idx in downhill {
                    let pip = PipId::new(entry.wire.tile(), pip_idx);
                    let pip_delay = match sg {
                        Some(sg) => chipdb.compute_pip_delay(sg, pip),
                        None => 100,
                    };
                    let next_wire = chipdb.pip_dst_wire(pip);
                    let new_cost = entry.cost + pip_delay;

                    if !visited.contains_key(&next_wire) || new_cost < *visited.get(&next_wire).unwrap() {
                        visited.insert(next_wire, new_cost);
                        heap.push(DijkEntry { wire: next_wire, cost: new_cost });
                    }
                }

                expanded += 1;
            }
        }

        eprintln!(
            "Lookahead: {} wire classes, table {}x{}, range ±{}",
            num_classes, table_w, table_h, max_range,
        );

        Self {
            wire_class_map,
            cost_tables,
            max_dx,
            max_dy,
            num_classes,
        }
    }

    /// Estimate delay from a source wire to a destination wire using the
    /// precomputed lookahead table. Returns the minimum cost from any wire
    /// of the source's class to reach the destination tile's offset.
    pub fn estimate_delay(
        &self,
        chipdb: &ChipDb,
        src: WireId,
        dst: WireId,
    ) -> DelayT {
        let tt_idx = chipdb.tile_type_index(src.tile()) as usize;
        let wi = src.index() as usize;

        // Look up wire class.
        let class = if tt_idx < self.wire_class_map.len() {
            let classes = &self.wire_class_map[tt_idx];
            if wi < classes.len() {
                classes[wi]
            } else {
                WireClass(u16::MAX)
            }
        } else {
            WireClass(u16::MAX)
        };

        // If class is unknown, fall back to Manhattan distance.
        if class.0 as usize >= self.num_classes {
            let (sx, sy) = chipdb.tile_xy(src.tile());
            let (dx, dy) = chipdb.tile_xy(dst.tile());
            return ((sx - dx).abs() + (sy - dy).abs()) * 10;
        }

        let (sx, sy) = chipdb.tile_xy(src.tile());
        let (dx, dy) = chipdb.tile_xy(dst.tile());
        let offset_x = dx - sx;
        let offset_y = dy - sy;

        // Clamp to table range.
        let ox = offset_x.clamp(-self.max_dx, self.max_dx);
        let oy = offset_y.clamp(-self.max_dy, self.max_dy);
        let ix = (ox + self.max_dx) as usize;
        let iy = (oy + self.max_dy) as usize;

        let table_cost = self.cost_tables[class.0 as usize][ix][iy];

        // If outside precomputed range, extrapolate with Manhattan.
        if table_cost == DelayT::MAX {
            let extra_dx = (offset_x.abs() - self.max_dx).max(0);
            let extra_dy = (offset_y.abs() - self.max_dy).max(0);
            return (extra_dx + extra_dy) * 10;
        }

        // Add extrapolation for offset beyond table.
        let extra_dx = (offset_x.abs() - ox.abs()).max(0);
        let extra_dy = (offset_y.abs() - oy.abs()).max(0);
        table_cost + (extra_dx + extra_dy) * 10
    }
}

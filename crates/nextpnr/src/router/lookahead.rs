//! Precomputed routing cost lookahead table.
//!
//! For each wire type and Manhattan offset (dx, dy), stores the minimum
//! routing cost to reach a wire at that offset. This provides an O(1)
//! admissible heuristic for A*, replacing the naive Manhattan distance
//! estimate that degenerates on architectures with many PIPs per tile.
//!
//! The table is computed once at router startup by running Dijkstra from
//! representative wires of each type at a sample tile.

use crate::chipdb::{ChipDb, PipId, WireId};
use crate::context::Context;
use crate::timing::DelayT;
use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};

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

/// Dijkstra cost model for the lookahead builder: same `pip_delay + 1` per
/// PIP as `router::maze`, zero heuristic (Dijkstra is A* with `h = 0`).
struct DijkstraCostModel;

impl PathCostModel for DijkstraCostModel {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, _ctx: &Context, _wire: WireId, _dst: WireId) -> DelayT {
        0
    }
}

impl Lookahead {
    /// Build the lookahead table.
    ///
    /// For each wire type in a representative tile, run `astar_search` in
    /// pure-Dijkstra mode (`heuristic = 0`, `exhaustive = true`) until the
    /// per-class visit budget (50k pops) is exhausted, then harvest the
    /// `visited` map to populate the cost-tables indexed by (class, dx, dy).
    pub fn build(ctx: &Context, max_range: i32) -> Self {
        let chipdb = ctx.chipdb();
        let _speed_grade_idx = ctx.speed_grade_idx();
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

        // Dijkstra == A* with zero heuristic. Delegate the actual search
        // to the shared `astar_search` kernel and harvest the visited map
        // to populate the per-offset cost table. Staying on the same
        // kernel as `router::maze` guarantees the admissible-heuristic
        // invariant: `h(w) ≤ true_cost(w, dst)` always holds because both
        // sides compute the true cost via the same edge-weight function.
        const MAX_EXPAND: usize = 50_000;
        let model = DijkstraCostModel;
        let mut src_set: FxHashSet<WireId> = FxHashSet::default();
        for src_wi in 0..n_wires {
            let src_wire = WireId::new(best_tile, src_wi as i32);
            src_set.clear();
            src_set.insert(src_wire);

            // Dst is ignored in exhaustive mode; any WireId satisfies the
            // signature. The search runs until the visit budget is
            // exhausted (or the heap drains).
            let result = astar_search(
                ctx,
                &model,
                &src_set,
                src_wire,
                &AStarOptions {
                    visit_limit: Some(MAX_EXPAND),
                    exhaustive: true,
                },
            );

            // Harvest reachable costs into the per-offset table. Non-zero
            // offsets floor at 1 so distant-peer-via-node-hop doesn't
            // record cost 0 (which would give A* h=0 at that offset and
            // destroy the gradient).
            let table = &mut cost_tables[src_wi];
            for (&wire, &(cost, _pen, _pip, _from)) in &result.trace.visited {
                let (wx, wy) = chipdb.tile_xy(wire.tile());
                let dx = wx - rep_x;
                let dy = wy - rep_y;
                if dx.abs() > max_dx || dy.abs() > max_dy {
                    continue;
                }
                let ix = (dx + max_dx) as usize;
                let iy = (dy + max_dy) as usize;
                let floor: DelayT = if dx == 0 && dy == 0 { 0 } else { 1 };
                let recorded = cost.max(floor);
                if recorded < table[ix][iy] {
                    table[ix][iy] = recorded;
                }
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
    pub fn estimate_delay(&self, chipdb: &ChipDb, src: WireId, dst: WireId) -> DelayT {
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

        // Fallback when Dijkstra never reached this offset (either because it's
        // outside the precomputed range or the MAX_EXPAND cap fired before the
        // offset was written). Use manhattan × 10 as a simple admissible lower
        // bound, same as the unknown-class fallback above. Without this,
        // in-range-but-unreached cells returned 0, leaving A* with no gradient.
        if table_cost == DelayT::MAX {
            return (offset_x.abs() + offset_y.abs()) * 10;
        }

        // Add extrapolation for offset beyond table.
        let extra_dx = (offset_x.abs() - ox.abs()).max(0);
        let extra_dy = (offset_y.abs() - oy.abs()).max(0);
        table_cost + (extra_dx + extra_dy) * 10
    }
}

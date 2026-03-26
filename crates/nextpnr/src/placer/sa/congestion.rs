//! Congestion cache for incremental demand tracking during SA.

use crate::context::Context;
use crate::metrics::{bresenham_line, compute_tile_capacities};
use crate::netlist::NetId;
use rustc_hash::FxHashMap;

/// Cached congestion state for incremental updates during SA.
///
/// Maintains per-edge demand and capacity grids that can be incrementally
/// updated as nets are moved. Only over-capacity edges contribute to the
/// congestion cost penalty.
///
/// Uses a per-net Bresenham point cache to avoid recomputing line segments
/// when removing demand (the common revert path). When adding demand the
/// Bresenham lines are computed fresh (positions may have changed) and
/// cached for future removal.
///
/// The congestion cost is tracked fully incrementally: each edge update
/// adjusts `cached_cost` by the delta in that edge's over-capacity penalty,
/// making `total_congestion_cost()` O(1).
pub struct CongestionCache {
    /// East-edge demand grid [y][x].
    h_demand: Vec<Vec<f64>>,
    /// South-edge demand grid [y][x].
    v_demand: Vec<Vec<f64>>,
    /// East-edge capacity grid [y][x].
    h_capacity: Vec<Vec<f64>>,
    /// South-edge capacity grid [y][x].
    v_capacity: Vec<Vec<f64>>,
    /// Grid width.
    grid_w: i32,
    /// Grid height.
    grid_h: i32,
    /// Cached Bresenham point lists per net (one Vec<(i32,i32)> per driver->sink pair).
    net_points: FxHashMap<NetId, Vec<Vec<(i32, i32)>>>,
    /// Running congestion cost (sum of max(0, ratio-1) for over-capacity edges).
    cached_cost: f64,
}

impl CongestionCache {
    /// Build capacity grids from chipdb and initialize demand by tracing all placed nets.
    pub fn new(ctx: &Context) -> Self {
        let grid_w = ctx.chipdb().width();
        let grid_h = ctx.chipdb().height();
        let wu = grid_w as usize;
        let hu = grid_h as usize;

        let (h_capacity, v_capacity) = compute_tile_capacities(ctx);

        let mut cache = Self {
            h_demand: vec![vec![0.0; wu]; hu],
            v_demand: vec![vec![0.0; wu]; hu],
            h_capacity,
            v_capacity,
            grid_w,
            grid_h,
            net_points: FxHashMap::default(),
            cached_cost: 0.0,
        };

        // Initialize demand from all current nets.
        for (net_idx, _) in ctx.design.iter_alive_nets() {
            cache.add_net_demand(ctx, net_idx, 1.0);
        }

        cache
    }

    /// O(1) congestion cost lookup.
    pub fn total_congestion_cost(&self) -> f64 {
        self.cached_cost
    }

    /// Update a single edge's demand and adjust cached_cost incrementally.
    #[inline]
    fn update_edge(demand: &mut f64, capacity: f64, sign: f64, cached_cost: &mut f64) {
        let old_ratio = *demand / capacity;
        let old_penalty = if old_ratio > 1.0 { old_ratio - 1.0 } else { 0.0 };

        *demand = (*demand + sign).max(0.0);

        let new_ratio = *demand / capacity;
        let new_penalty = if new_ratio > 1.0 { new_ratio - 1.0 } else { 0.0 };

        *cached_cost += new_penalty - old_penalty;
    }

    /// Apply edge crossings from a Bresenham point list, updating demand and cost.
    fn apply_crossings(&mut self, points: &[(i32, i32)], sign: f64) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        for window in points.windows(2) {
            let (x1, y1) = window[0];
            let (x2, y2) = window[1];
            let dx = x2 - x1;
            let dy = y2 - y1;

            if dx != 0 {
                let ex = if dx > 0 { x1 } else { x2 };
                let ey = y1;
                if ex >= 0 && ex < gw - 1 && ey >= 0 && ey < gh {
                    Self::update_edge(
                        &mut self.h_demand[ey as usize][ex as usize],
                        self.h_capacity[ey as usize][ex as usize],
                        sign,
                        &mut self.cached_cost,
                    );
                }
            }
            if dy != 0 {
                let ex = x1;
                let ey = if dy > 0 { y1 } else { y2 };
                if ex >= 0 && ex < gw && ey >= 0 && ey < gh - 1 {
                    Self::update_edge(
                        &mut self.v_demand[ey as usize][ex as usize],
                        self.v_capacity[ey as usize][ex as usize],
                        sign,
                        &mut self.cached_cost,
                    );
                }
            }
        }
    }

    /// Add or remove demand for a net.
    ///
    /// `sign` should be +1.0 to add demand or -1.0 to remove it.
    ///
    /// When removing (`sign < 0`): uses cached Bresenham points so no line
    /// recomputation is needed. When adding (`sign > 0`): computes fresh
    /// Bresenham lines from current cell positions and caches them.
    pub fn add_net_demand(&mut self, ctx: &Context, net_idx: NetId, sign: f64) {
        let net = ctx.design.net(net_idx);
        if !net.alive {
            return;
        }

        if sign < 0.0 {
            // Remove demand using cached points (no Bresenham recomputation).
            if let Some(point_lists) = self.net_points.remove(&net_idx) {
                for points in &point_lists {
                    self.apply_crossings(points, sign);
                }
            }
        } else {
            // Add demand: compute fresh Bresenham lines and cache them.
            let driver = match net.driver() {
                Some(pin) => pin,
                None => return,
            };
            let driver_bel = match ctx.design.cell(driver.cell).bel {
                Some(bel) => bel,
                None => return,
            };
            let driver_loc = ctx.chipdb().bel_loc(driver_bel);

            let mut point_lists = Vec::with_capacity(net.users().len());

            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let sink_bel = match ctx.design.cell(user.cell).bel {
                    Some(bel) => bel,
                    None => continue,
                };
                let sink_loc = ctx.chipdb().bel_loc(sink_bel);

                let points =
                    bresenham_line(driver_loc.x, driver_loc.y, sink_loc.x, sink_loc.y);
                self.apply_crossings(&points, sign);
                point_lists.push(points);
            }

            self.net_points.insert(net_idx, point_lists);
        }
    }
}

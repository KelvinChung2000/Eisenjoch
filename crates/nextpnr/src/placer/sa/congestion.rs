//! Congestion cache for incremental demand tracking during SA.

use crate::context::Context;
use crate::metrics::compute_tile_capacities;
use crate::metrics::congestion::bresenham_line_into;
use crate::netlist::NetId;
use rustc_hash::FxHashSet;

/// Cached congestion state for incremental updates during SA.
///
/// Maintains per-edge demand and capacity grids that can be incrementally
/// updated as nets are moved. Only over-capacity edges contribute to the
/// congestion cost penalty.
///
/// Removing a net re-traces it from its cells' current positions. Callers
/// remove a net's demand before moving any of its cells and add it back after,
/// so those are the positions it was added at, and nothing per arc needs to be
/// kept: storing the traced points scaled with total wirelength.
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
    /// Nets whose demand is currently applied.
    applied: FxHashSet<NetId>,
    /// Reused line buffer.
    points: Vec<(i32, i32)>,
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
            applied: FxHashSet::default(),
            points: Vec::new(),
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
        let old_penalty = if old_ratio > 1.0 {
            old_ratio - 1.0
        } else {
            0.0
        };

        *demand = (*demand + sign).max(0.0);

        let new_ratio = *demand / capacity;
        let new_penalty = if new_ratio > 1.0 {
            new_ratio - 1.0
        } else {
            0.0
        };

        *cached_cost += new_penalty - old_penalty;
    }

    /// Apply the edge crossings of the line in `self.points`, updating demand
    /// and cost.
    fn apply_crossings(&mut self, sign: f64) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        for i in 1..self.points.len() {
            let window = [self.points[i - 1], self.points[i]];
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
    /// Removing a net that is not applied does nothing, and removal must
    /// happen before any of the net's cells move.
    pub fn add_net_demand(&mut self, ctx: &Context, net_idx: NetId, sign: f64) {
        let net = ctx.design.net(net_idx);
        if !net.alive {
            return;
        }
        if sign < 0.0 && !self.applied.remove(&net_idx) {
            return;
        }

        let driver = match net.driver() {
            Some(pin) => pin,
            None => return,
        };
        let driver_bel = match ctx.design.cell(driver.cell).bel {
            Some(bel) => bel,
            None => return,
        };
        let driver_loc = ctx.chipdb().bel_loc(driver_bel);

        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let sink_bel = match ctx.design.cell(user.cell).bel {
                Some(bel) => bel,
                None => continue,
            };
            let sink_loc = ctx.chipdb().bel_loc(sink_bel);

            bresenham_line_into(
                &mut self.points,
                driver_loc.x,
                driver_loc.y,
                sink_loc.x,
                sink_loc.y,
            );
            self.apply_crossings(sign);
        }

        if sign >= 0.0 {
            self.applied.insert(net_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chipdb::Loc;
    use crate::common::PlaceStrength;
    use crate::context::example_context;
    use crate::netlist::PortType;

    /// What the cache keeps per net is one flag, not the traced arcs, and
    /// removing a net re-traces it to exactly zero demand.
    #[test]
    fn per_net_state_does_not_scale_with_wirelength() {
        let mut ctx = example_context();
        let lut = ctx.id("LUT6");
        let (drv, sink) = (
            ctx.design.add_cell(ctx.id("drv"), lut),
            ctx.design.add_cell(ctx.id("sink"), lut),
        );
        let (out, inp) = (ctx.id("F"), ctx.id("I[0]"));
        let net = ctx.design.add_net(ctx.id("n"));
        ctx.design.cell_edit(drv).add_port(out, PortType::Out);
        ctx.design.cell_edit(drv).set_port_net(out, Some(net), None);
        ctx.design.net_edit(net).set_driver(drv, out);
        ctx.design.cell_edit(sink).add_port(inp, PortType::In);
        let user = ctx.design.net_edit(net).add_user(sink, inp);
        ctx.design
            .cell_edit(sink)
            .set_port_net(inp, Some(net), Some(user));
        let far = ctx.chipdb().width() - 2;
        for (cell, xy) in [(drv, 1), (sink, far)] {
            let bel = ctx
                .bucket_bel_at(lut, Loc::new(xy, xy, 0))
                .expect("LUT slot");
            assert!(ctx.bind_bel(bel, cell, PlaceStrength::Placer));
        }

        let mut cache = CongestionCache::new(&ctx);
        assert_eq!(cache.applied.len(), 1);
        let total =
            |c: &CongestionCache| -> f64 { c.h_demand.iter().chain(&c.v_demand).flatten().sum() };
        assert!(total(&cache) > 0.0);

        cache.add_net_demand(&ctx, net, -1.0);
        assert!(cache.applied.is_empty());
        assert_eq!(total(&cache), 0.0);
        // A second removal finds nothing applied and changes nothing.
        cache.add_net_demand(&ctx, net, -1.0);
        assert_eq!(total(&cache), 0.0);
    }
}

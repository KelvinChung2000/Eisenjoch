//! The placer1 cost model: bounding-box wirelength plus criticality-weighted
//! arc delay.
//!
//! Port of the cost half of `SAPlacer` (`common/place/placer1.cc`). The
//! bounding boxes here carry no extremity counts, unlike upstream's
//! `BoundingBox`: those exist only to make `CELL_MOVED_INWARDS` updates O(1),
//! and we recompute each touched net's box outright. Same numbers, less
//! bookkeeping -- see [`CostModel::stage`].

use rustc_hash::FxHashMap;

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use crate::timing::TimingAnalyser;

use super::config::RefineCfg;

/// Half-perimeter bounding box of one net.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct BoundingBox {
    pub x0: i32,
    pub x1: i32,
    pub y0: i32,
    pub y1: i32,
}

impl BoundingBox {
    /// `BoundingBox::hpwl` -- the scaled half-perimeter.
    #[inline]
    fn hpwl(&self, cfg: &RefineCfg) -> i64 {
        (cfg.hpwl_scale_x * (self.x1 - self.x0) + cfg.hpwl_scale_y * (self.y1 - self.y0)) as i64
    }
}

/// Per-net cost state, indexed densely; the equivalent of upstream's
/// `net_bounds` / `net_arc_tcost` keyed by `NetInfo::udata`.
pub(super) struct CostModel {
    /// Dense index -> net, upstream's `net_by_udata`.
    nets: Vec<NetId>,
    /// Net -> dense index, upstream's `NetInfo::udata`.
    index: FxHashMap<NetId, usize>,
    /// `ignore_net`: no driver, unplaced driver, or a global-buffer driver.
    ignored: Vec<bool>,
    bounds: Vec<BoundingBox>,
    /// Timing cost per (net, user index).
    arc_cost: Vec<Vec<f64>>,
    /// Nets touched by each cell, so a move only rescores what it can change.
    nets_of_cell: FxHashMap<CellId, Vec<usize>>,
    pub wirelen: i64,
    pub timing: f64,
}

/// A move's effect on cost, held until it is committed or rolled back.
pub(super) struct Staged {
    /// The nets rescored, with their replacement state.
    touched: Vec<(usize, BoundingBox, Vec<f64>)>,
    pub wirelen_delta: i64,
    pub timing_delta: f64,
}

impl CostModel {
    pub(super) fn build(ctx: &Context) -> Self {
        let nets: Vec<NetId> = ctx.design.iter_alive_nets().map(|(id, _)| id).collect();
        let index: FxHashMap<NetId, usize> =
            nets.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // A cell's move can only change nets it is attached to.
        let mut nets_of_cell: FxHashMap<CellId, Vec<usize>> = FxHashMap::default();
        for (cell_id, _) in ctx.design.iter_alive_cells() {
            let cell = ctx.cell(cell_id);
            let mut owned: Vec<usize> = cell
                .ports()
                .filter_map(|pin| cell.port_net(pin.port))
                .filter_map(|net| index.get(&net).copied())
                .collect();
            owned.sort_unstable();
            owned.dedup();
            nets_of_cell.insert(cell_id, owned);
        }

        let n = nets.len();
        Self {
            nets,
            index,
            ignored: vec![false; n],
            bounds: vec![BoundingBox::default(); n],
            arc_cost: vec![Vec::new(); n],
            nets_of_cell,
            wirelen: 0,
            timing: 0.0,
        }
    }

    /// `ignore_net` -- a net with no driver, an unplaced driver, or a driver on
    /// a global buffer is left out of the cost entirely.
    fn is_ignored(ctx: &Context, net: NetId) -> bool {
        let info = ctx.design.net(net);
        if !info.driver.is_valid() {
            return true;
        }
        match ctx.design.cell(info.driver.cell).bel {
            None => true,
            Some(bel) => ctx.bel_global_buf(bel),
        }
    }

    /// `get_net_bounds`.
    fn net_bounds(ctx: &Context, net: NetId) -> BoundingBox {
        let info = ctx.design.net(net);
        let driver_bel = ctx
            .design
            .cell(info.driver.cell)
            .bel
            .expect("ignored nets are filtered before bounds are taken");
        let dloc = ctx.chipdb().bel_loc(driver_bel);
        let mut bb = BoundingBox {
            x0: dloc.x,
            x1: dloc.x,
            y0: dloc.y,
            y1: dloc.y,
        };
        for &user in info.users.iter() {
            if !user.is_valid() {
                continue;
            }
            let Some(bel) = ctx.design.cell(user.cell).bel else {
                continue;
            };
            let loc = ctx.chipdb().bel_loc(bel);
            bb.x0 = bb.x0.min(loc.x);
            bb.x1 = bb.x1.max(loc.x);
            bb.y0 = bb.y0.min(loc.y);
            bb.y1 = bb.y1.max(loc.y);
        }
        bb
    }

    /// `get_timing_cost` -- arc delay weighted by criticality raised to
    /// `crit_exp`, so the critical few dominate.
    fn arc_costs(
        ctx: &Context,
        tmg: &TimingAnalyser,
        net: NetId,
        cfg: &RefineCfg,
    ) -> Vec<f64> {
        let info = ctx.design.net(net);
        if !cfg.timing_driven || info.users.len() >= cfg.timing_fanout_thresh {
            return Vec::new();
        }
        // TMG_IGNORE on the driver port means the net carries no timing.
        if ctx.port_timing_class(info.driver.cell, info.driver.port)
            == crate::timing::TimingPortClass::Ignore
        {
            return vec![0.0; info.users.len()];
        }
        info.users
            .iter()
            .map(|&user| {
                if !user.is_valid() {
                    return 0.0;
                }
                let crit = tmg.port_criticality(user.cell, user.port) as f64;
                let delay = ctx.delay_ns(ctx.predict_arc_delay(net, user));
                delay * crit.powf(cfg.crit_exp)
            })
            .collect()
    }

    /// `setup_costs` + `total_wirelen_cost` + `total_timing_cost` -- rebuild
    /// everything and recompute the totals from scratch.
    ///
    /// Upstream does this once per outer iteration specifically so rounding
    /// error from incremental updates cannot accumulate; keep that property.
    pub(super) fn refresh(&mut self, ctx: &Context, tmg: &TimingAnalyser, cfg: &RefineCfg) {
        self.wirelen = 0;
        self.timing = 0.0;
        for i in 0..self.nets.len() {
            let net = self.nets[i];
            self.ignored[i] = Self::is_ignored(ctx, net);
            if self.ignored[i] {
                self.bounds[i] = BoundingBox::default();
                self.arc_cost[i].clear();
                continue;
            }
            self.bounds[i] = Self::net_bounds(ctx, net);
            self.arc_cost[i] = Self::arc_costs(ctx, tmg, net, cfg);
            self.wirelen += self.bounds[i].hpwl(cfg);
            self.timing += self.arc_cost[i].iter().sum::<f64>();
        }
    }

    /// Rescore every net touched by `cells` against the *current* bindings.
    ///
    /// Call after the move is applied: the caller then either
    /// [`commit`](Self::commit)s the result or unwinds the bindings and drops
    /// it. This replaces upstream's `add_move_cell` / `compute_cost_changes`
    /// pair; recomputing a touched net's box is equivalent to its
    /// inward/outward update and cannot drift from it.
    pub(super) fn stage(
        &self,
        ctx: &Context,
        tmg: &TimingAnalyser,
        cfg: &RefineCfg,
        cells: &[CellId],
    ) -> Staged {
        let mut seen: Vec<usize> = cells
            .iter()
            .filter_map(|c| self.nets_of_cell.get(c))
            .flatten()
            .copied()
            .collect();
        seen.sort_unstable();
        seen.dedup();

        let mut staged = Staged {
            touched: Vec::with_capacity(seen.len()),
            wirelen_delta: 0,
            timing_delta: 0.0,
        };
        for i in seen {
            if self.ignored[i] {
                continue;
            }
            let net = self.nets[i];
            let bb = Self::net_bounds(ctx, net);
            let arcs = Self::arc_costs(ctx, tmg, net, cfg);
            staged.wirelen_delta += bb.hpwl(cfg) - self.bounds[i].hpwl(cfg);
            staged.timing_delta +=
                arcs.iter().sum::<f64>() - self.arc_cost[i].iter().sum::<f64>();
            staged.touched.push((i, bb, arcs));
        }
        staged
    }

    /// Arcs carrying a nonzero timing cost. A zero here means criticality came
    /// back flat and the timing half of the objective is doing nothing.
    pub(super) fn critical_arcs(&self) -> usize {
        self.arc_cost
            .iter()
            .flatten()
            .filter(|&&c| c > 0.0)
            .count()
    }

    /// `commit_cost_changes`.
    pub(super) fn commit(&mut self, staged: Staged) {
        for (i, bb, arcs) in staged.touched {
            self.bounds[i] = bb;
            self.arc_cost[i] = arcs;
        }
        self.wirelen += staged.wirelen_delta;
        self.timing += staged.timing_delta;
    }
}

//! Faithful port of nextpnr's `place_common`.
//!
//! Source: upstream YosysHQ nextpnr `main` @ `4d235150`,
//! `common/place/place_common.{h,cc}`.
//!
//! ```text
//!  nextpnr -- Next Generation Place and Route
//!
//!  Copyright (C) 2018  gatecat <gatecat@ds0.me>
//!
//!  Permission to use, copy, modify, and/or distribute this software for any
//!  purpose with or without fee is hereby granted, provided that the above
//!  copyright notice and this permission notice appear in all copies.
//!
//!  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//!  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//!  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//!  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//!  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//!  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//!  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//! ```
//!
//! The wirelength model every nextpnr placer scores against, plus the relative
//! constraint legaliser.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::chipdb::{BelId, Loc};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellId, NetId, Property};
use crate::placer::fast_bels::FastBels;
use crate::placer::PlacerError;
use crate::timing::TimingPortClass;

/// `wirelen_t`.
pub type WirelenT = i64;

/// `MetricType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricType {
    /// Timing-weighted cost, used to drive placement decisions.
    Cost,
    /// Raw half-perimeter wirelength, used for reporting.
    Wirelength,
}

/// Every live cell, in a stable order.
///
/// The legaliser both iterates and mutates the design, so the ids are
/// collected up front rather than held as a borrow.
fn cell_ids(ctx: &Context) -> Vec<CellId> {
    ctx.cells().map(|c| c.id()).collect()
}

/// Distance reported for a cell that is unplaced or whose cluster cannot be
/// placed at all. Large enough to dominate any real offset sum.
const UNPLACED_DISTANCE: i32 = 100000;

/// `get_net_metric` -- the half-perimeter wirelength of one net, optionally
/// weighted by timing, accumulating total negative slack into `tns`.
///
/// Returns 0 for nets with no driver, an unplaced driver, or a global-buffer
/// driver: global nets are routed on dedicated resources, so costing them as
/// ordinary routing would drag every placement toward the clock source.
pub fn get_net_metric(
    ctx: &Context,
    net: NetId,
    metric_type: MetricType,
    tns: &mut f32,
) -> WirelenT {
    let net_info = ctx.design.net(net);
    if !net_info.driver.is_valid() {
        return 0;
    }
    let driver_cell = net_info.driver.cell;
    let Some(driver_bel) = ctx.design.cell(driver_cell).bel else {
        return 0;
    };
    if ctx.bel_global_buf(driver_bel) {
        return 0;
    }

    let timing_driven = ctx.timing_driven()
        && metric_type == MetricType::Cost
        && ctx.port_timing_class(driver_cell, net_info.driver.port) != TimingPortClass::Ignore;

    let mut negative_slack = 0;
    let mut worst_slack = i32::MAX;

    let driver_loc = ctx.chipdb().bel_loc(driver_bel);
    let (mut xmin, mut xmax) = (driver_loc.x, driver_loc.x);
    let (mut ymin, mut ymax) = (driver_loc.y, driver_loc.y);

    for &load in net_info.users.iter() {
        if !load.is_valid() {
            continue;
        }
        let Some(load_bel) = ctx.design.cell(load.cell).bel else {
            continue;
        };

        // Slack is accumulated for every placed load, including ones on global
        // buffers -- the bounding box skips those, the timing sum does not.
        if timing_driven {
            let net_delay = ctx.predict_arc_delay(net, load);
            let slack = -net_delay;
            if slack < 0 {
                negative_slack += slack;
            }
            worst_slack = worst_slack.min(slack);
        }

        if ctx.bel_global_buf(load_bel) {
            continue;
        }
        let load_loc = ctx.chipdb().bel_loc(load_bel);
        xmin = xmin.min(load_loc.x);
        ymin = ymin.min(load_loc.y);
        xmax = xmax.max(load_loc.x);
        ymax = ymax.max(load_loc.y);
    }

    let hpwl = ((ymax - ymin) + (xmax - xmin)) as f64;
    let wirelength = if timing_driven {
        // Nets with worse slack are inflated, up to 5x, so the placer pulls
        // them tight first. Truncated to an integer exactly as the C++ cast
        // does -- rounding here would shift costs.
        (hpwl * 5.0_f64.min(1.0 + (-ctx.delay_ns(worst_slack) / 5.0).exp())) as WirelenT
    } else {
        hpwl as WirelenT
    };

    *tns += ctx.delay_ns(negative_slack) as f32;
    wirelength
}

/// `get_cell_metric` -- the summed metric of every net touching a cell.
///
/// Nets are gathered into a set first, so a net connected to the cell on more
/// than one port is counted once.
pub fn get_cell_metric(ctx: &Context, cell: CellId, metric_type: MetricType) -> WirelenT {
    let info = ctx.design.cell(cell);
    let mut nets: Vec<NetId> = Vec::new();
    let mut seen: FxHashSet<NetId> = FxHashSet::default();
    for port in info.ports.values() {
        if let Some(net) = port.net() {
            if seen.insert(net) {
                nets.push(net);
            }
        }
    }
    // nextpnr iterates a std::set<IdString>, i.e. in net-name order. Sorting by
    // id keeps the summation order stable run to run; the sum itself is
    // integer, so order does not change the result.
    nets.sort_unstable();

    let mut wirelength = 0;
    let mut tns = 0.0f32;
    for net in nets {
        wirelength += get_net_metric(ctx, net, metric_type, &mut tns);
    }
    wirelength
}

/// `get_cell_metric_at_bel` -- the cell's metric as if it sat on `bel`.
///
/// nextpnr mutates `cell->bel` in place and restores it. This does the same
/// through the design, which is why it needs `&mut Context` where the C++ took
/// a const pointer and cast the constness away.
pub fn get_cell_metric_at_bel(
    ctx: &mut Context,
    cell: CellId,
    bel: BelId,
    metric_type: MetricType,
) -> WirelenT {
    let old_bel = ctx.design.cell(cell).bel;
    ctx.design.cell_mut(cell).bel = Some(bel);
    let wirelen = get_cell_metric(ctx, cell, metric_type);
    ctx.design.cell_mut(cell).bel = old_bel;
    wirelen
}

/// `get_constraints_distance` -- how far a cell is from satisfying its cluster
/// constraints, in summed Manhattan offset error.
///
/// Zero means satisfied. An unplaced cell, or a cluster root whose placement
/// cannot be resolved, returns [`UNPLACED_DISTANCE`].
pub fn get_constraints_distance(ctx: &Context, cell: CellId) -> i32 {
    let mut dist = 0;
    let info = ctx.design.cell(cell);

    let Some(bel) = info.bel else {
        return UNPLACED_DISTANCE;
    };
    let loc = ctx.chipdb().bel_loc(bel);

    let Some(cluster) = info.cluster else {
        return dist;
    };

    let root = ctx.cluster_root_cell(cluster);
    if root == cell {
        // Root: every member must sit exactly where the cluster says.
        let Some(placement) = ctx.cluster_placement(cluster, bel) else {
            return UNPLACED_DISTANCE;
        };
        for (member, want_bel) in placement {
            let Some(member_bel) = ctx.design.cell(member).bel else {
                return UNPLACED_DISTANCE;
            };
            let c_loc = ctx.chipdb().bel_loc(member_bel);
            let p_loc = ctx.chipdb().bel_loc(want_bel);
            dist += (c_loc.x - p_loc.x).abs();
            dist += (c_loc.y - p_loc.y).abs();
            dist += (c_loc.z - p_loc.z).abs();
        }
    } else {
        // Child: only x and y are checked against the root, matching nextpnr.
        // z is left to the root's own placement check above.
        let Some(root_bel) = ctx.design.cell(root).bel else {
            return UNPLACED_DISTANCE;
        };
        let root_loc = ctx.chipdb().bel_loc(root_bel);
        let offset = ctx.cluster_offset(cell);
        dist += ((root_loc.x + offset.x) - loc.x).abs();
        dist += ((root_loc.y + offset.y) - loc.y).abs();
    }

    dist
}

/// `ConstraintLegaliseWorker::IncreasingDiameterSearch`.
///
/// Walks outwards from `start`, alternating sign, clamped to `[min, max]`:
/// start, start+1, start-1, start+2, ... When one direction runs off the end it
/// keeps going in the other, so an off-centre start still covers the range.
#[derive(Clone, Copy, Debug)]
pub struct IncreasingDiameterSearch {
    start: i32,
    min: i32,
    max: i32,
    diameter: i32,
    sign: i32,
}

impl Default for IncreasingDiameterSearch {
    /// The default is already exhausted: `max < min`, so `done()` is true.
    fn default() -> Self {
        Self {
            start: 0,
            min: 0,
            max: -1,
            diameter: 0,
            sign: 0,
        }
    }
}

impl IncreasingDiameterSearch {
    /// A search over the single value `x`.
    pub fn at(x: i32) -> Self {
        Self {
            start: x,
            min: x,
            max: x,
            diameter: 0,
            sign: 0,
        }
    }

    /// A search from `start`, bounded by `min` and `max` inclusive.
    pub fn new(start: i32, min: i32, max: i32) -> Self {
        Self {
            start,
            min,
            max,
            diameter: 0,
            sign: 0,
        }
    }

    /// Whether the search has covered the whole range.
    #[inline]
    pub fn done(&self) -> bool {
        self.diameter > (self.max - self.min)
    }

    /// The current value, clamped into range.
    #[inline]
    pub fn get(&self) -> i32 {
        (self.start + self.sign * self.diameter)
            .max(self.min)
            .min(self.max)
    }

    /// Step to the next value.
    pub fn next(&mut self) {
        if self.sign == 0 {
            self.sign = 1;
            self.diameter = 1;
        } else if self.sign == -1 {
            self.sign = 1;
            if (self.start + self.sign * self.diameter) > self.max {
                self.sign = -1;
            }
            self.diameter += 1;
        } else {
            self.sign = -1;
            if (self.start + self.sign * self.diameter) < self.min {
                self.sign = 1;
                self.diameter += 1;
            }
        }
    }

    /// Restart from `start`.
    pub fn reset(&mut self) {
        self.sign = 0;
        self.diameter = 0;
    }
}

/// `ConstraintLegaliseWorker`.
pub struct ConstraintLegaliseWorker {
    ripped_cells: FxHashSet<CellId>,
    old_locations: FxHashMap<CellId, Loc>,
    cluster2cells: FxHashMap<CellId, Vec<CellId>>,
    fast_bels: FastBels,
}

/// A candidate placement for a cell and its constrained children.
type CellLocations = FxHashMap<CellId, Loc>;

impl ConstraintLegaliseWorker {
    /// `ConstraintLegaliseWorker(ctx)`.
    pub fn new(ctx: &Context) -> Self {
        let mut cluster2cells: FxHashMap<CellId, Vec<CellId>> = FxHashMap::default();
        for cell in cell_ids(ctx) {
            if let Some(cluster) = ctx.design.cell(cell).cluster {
                cluster2cells.entry(cluster).or_default().push(cell);
            }
        }
        Self {
            ripped_cells: FxHashSet::default(),
            old_locations: FxHashMap::default(),
            cluster2cells,
            // check_bel_available = false: the legaliser deliberately considers
            // occupied BELs, because it is allowed to rip up weakly bound cells.
            fast_bels: FastBels::new(false, 0),
        }
    }

    /// `valid_loc_for` -- whether `loc` can host `cell` and its cluster, and if
    /// so what the resulting placement is.
    ///
    /// Rejects any tile containing a strongly bound cell: the legaliser may
    /// need to rip up whatever is in the way, and strong bindings are not
    /// available to rip.
    fn valid_loc_for(
        &self,
        ctx: &Context,
        cell: CellId,
        loc: Loc,
        solution: &mut CellLocations,
        used_locations: &mut FxHashSet<(i32, i32, i32)>,
    ) -> bool {
        let Some(loc_bel) = ctx.bel_by_location(loc) else {
            return false;
        };
        let info = ctx.design.cell(cell);

        match info.cluster {
            None => {
                if !ctx.bel(loc_bel).is_valid_for_cell_type(info.cell_type) {
                    return false;
                }
                if !ctx.bel(loc_bel).is_available() {
                    let confl = ctx.conflicting_bel_cell(loc_bel);
                    if confl
                        .map(|c| ctx.design.cell(c).bel_strength >= PlaceStrength::Strong)
                        .unwrap_or(false)
                    {
                        return false;
                    }
                }
                if Self::tile_has_strong_cell(ctx, loc.x, loc.y) {
                    return false;
                }
                used_locations.insert((loc.x, loc.y, loc.z));
                solution.insert(cell, loc);
            }
            Some(cluster) => {
                let Some(placement) = ctx.cluster_placement(cluster, loc_bel) else {
                    return false;
                };
                for (member, member_bel) in placement {
                    let p_loc = ctx.chipdb().bel_loc(member_bel);
                    if !ctx.bel(member_bel).is_available() {
                        let confl = ctx.conflicting_bel_cell(member_bel);
                        if confl
                            .map(|c| ctx.design.cell(c).bel_strength >= PlaceStrength::Strong)
                            .unwrap_or(false)
                        {
                            return false;
                        }
                    }
                    if Self::tile_has_strong_cell(ctx, p_loc.x, p_loc.y) {
                        return false;
                    }
                    used_locations.insert((p_loc.x, p_loc.y, p_loc.z));
                    solution.insert(member, p_loc);
                }
            }
        }

        true
    }

    /// Whether any BEL in the tile holds a strongly bound cell.
    fn tile_has_strong_cell(ctx: &Context, x: i32, y: i32) -> bool {
        ctx.bels_by_tile(x, y).into_iter().any(|tilebel| {
            ctx.bel(tilebel)
                .bound_cell()
                .map(|c| ctx.design.cell(c.id()).bel_strength >= PlaceStrength::Strong)
                .unwrap_or(false)
        })
    }

    /// `lockdown_chain` -- pin a cluster in place once it is legal.
    fn lockdown_chain(&self, ctx: &mut Context, root: CellId) {
        ctx.design.cell_mut(root).bel_strength = PlaceStrength::Strong;
        if let Some(cluster) = ctx.design.cell(root).cluster {
            if let Some(children) = self.cluster2cells.get(&cluster) {
                for &child in &children.clone() {
                    ctx.design.cell_mut(child).bel_strength = PlaceStrength::Strong;
                }
            }
        }
    }

    /// `legalise_cell` -- make one cluster root satisfy its constraints.
    ///
    /// Non-roots return immediately: a cluster is legalised as a unit through
    /// its root.
    fn legalise_cell(&mut self, ctx: &mut Context, cell: CellId) -> bool {
        let info = ctx.design.cell(cell);
        if let Some(cluster) = info.cluster {
            if ctx.cluster_root_cell(cluster) != cell {
                return true;
            }
        }

        if get_constraints_distance(ctx, cell) == 0 {
            if ctx.design.cell(cell).cluster.is_some() {
                self.lockdown_chain(ctx, cell);
            }
            return true;
        }

        let current_loc = match ctx.design.cell(cell).bel {
            Some(bel) => ctx.chipdb().bel_loc(bel),
            None => *self
                .old_locations
                .get(&cell)
                .unwrap_or(&Loc::new(0, 0, 0)),
        };

        let mut x_search = IncreasingDiameterSearch::new(current_loc.x, 0, ctx.grid_dim_x() - 1);
        let mut y_search = IncreasingDiameterSearch::new(current_loc.y, 0, ctx.grid_dim_y() - 1);
        let mut z_search = IncreasingDiameterSearch::new(
            current_loc.z,
            0,
            ctx.tile_bel_dim_z(current_loc.x, current_loc.y),
        );

        while !x_search.done() {
            let root_loc = Loc::new(x_search.get(), y_search.get(), z_search.get());

            // Odometer: z rolls into y, y rolls into x.
            z_search.next();
            if z_search.done() {
                z_search.reset();
                y_search.next();
                if y_search.done() {
                    y_search.reset();
                    x_search.next();
                }
            }

            let mut solution = CellLocations::default();
            let mut used = FxHashSet::default();
            if !self.valid_loc_for(ctx, cell, root_loc, &mut solution, &mut used) {
                continue;
            }

            // Unbind everything in the solution before binding any of it:
            // members frequently swap slots with each other.
            for &member in solution.keys() {
                if let Some(bel) = ctx.design.cell(member).bel {
                    ctx.unbind_bel(bel);
                }
            }

            let entries: Vec<(CellId, Loc)> = solution.iter().map(|(&c, &l)| (c, l)).collect();
            for (member, member_loc) in &entries {
                let target = ctx
                    .bel_by_location(*member_loc)
                    .expect("valid_loc_for resolved this BEL");
                if !ctx.bel(target).is_available() {
                    if let Some(confl) = ctx.conflicting_bel_cell(target) {
                        assert!(
                            ctx.design.cell(confl).bel_strength < PlaceStrength::Strong,
                            "valid_loc_for admitted a location holding a strongly bound cell"
                        );
                        ctx.unbind_bel(target);
                        self.ripped_cells.insert(confl);
                    }
                }
                assert!(
                    ctx.bind_bel(target, *member, PlaceStrength::Strong),
                    "target BEL was just freed but the bind failed"
                );
                self.ripped_cells.remove(member);
            }

            // Binding the cluster can invalidate neighbours sharing the tile;
            // rip those so they get replaced below.
            for (_, member_loc) in &entries {
                for bel in ctx.bels_by_tile(member_loc.x, member_loc.y) {
                    let Some(bel_cell) = ctx.bel(bel).bound_cell().map(|c| c.id()) else {
                        continue;
                    };
                    if solution.contains_key(&bel_cell) {
                        continue;
                    }
                    if !ctx.is_bel_location_valid(bel) {
                        assert!(
                            ctx.design.cell(bel_cell).bel_strength < PlaceStrength::Strong,
                            "a strongly bound neighbour became invalid"
                        );
                        ctx.unbind_bel(bel);
                        self.ripped_cells.insert(bel_cell);
                    }
                }
            }

            assert_eq!(
                get_constraints_distance(ctx, cell),
                0,
                "cell is still unconstrained after a placement that valid_loc_for accepted"
            );
            return true;
        }

        false
    }

    /// `place_single_cell` -- find a home for a ripped-up cell, following the
    /// displacement chain when it in turn rips someone else out.
    ///
    /// Samples random BELs in a window that grows every `5 * diameter` misses,
    /// keeping the cheapest legal candidate seen. Occupied candidates are
    /// costed 5x, so a free BEL wins unless a taken one is much better.
    fn place_single_cell(&mut self, ctx: &mut Context, start: CellId) -> bool {
        let mut current = Some(start);

        // Declared outside the chain loop, exactly as in the C++: the search
        // window persists as the displacement chain ripples, so each displaced
        // cell starts from the width its predecessor had to widen to. Resetting
        // it per link would change both the RNG draw count and where ripped
        // cells land.
        let mut diameter = 1;

        while let Some(cell) = current {
            if let Some(bel) = ctx.design.cell(cell).bel {
                ctx.unbind_bel(bel);
            }

            let cell_type = ctx.design.cell(cell).cell_type;
            let region = ctx.design.cell(cell).region;
            let old_loc = *self
                .old_locations
                .get(&cell)
                .unwrap_or(&Loc::new(0, 0, 0));

            let mut iter = 0;
            let mut best_bel: Option<BelId> = None;
            let mut best_metric = WirelenT::MAX;
            let mut ripup_target: Option<CellId>;

            loop {
                iter += 1;
                if iter >= (5 * diameter) {
                    iter = 0;
                    if diameter < ctx.grid_dim_x().max(ctx.grid_dim_y()) {
                        diameter += 1;
                    }
                    if best_bel.is_some() {
                        break;
                    }
                }

                let (nx, ny) = {
                    let dx = ctx.rng_mut().rng_n(diameter);
                    let dy = ctx.rng_mut().rng_n(diameter);
                    (
                        old_loc.x - (diameter / 2) + dx,
                        old_loc.y - (diameter / 2) + dy,
                    )
                };

                let bel = {
                    let (bel_data, _) = self.fast_bels.bels_for_cell_type(ctx, cell_type);
                    if nx < 0 || nx >= bel_data.len() as i32 {
                        continue;
                    }
                    let column = &bel_data[nx as usize];
                    if ny < 0 || ny >= column.len() as i32 {
                        continue;
                    }
                    let fb = &column[ny as usize];
                    if fb.is_empty() {
                        continue;
                    }
                    let pick = fb.len() as i32;
                    let idx = ctx.rng_mut().rng_n(pick) as usize;
                    let (bel_data, _) = self.fast_bels.bels_for_cell_type(ctx, cell_type);
                    bel_data[nx as usize][ny as usize][idx]
                };

                // Region constraint: eisenjoch expresses a region as tile
                // rectangles rather than nextpnr's explicit BEL set, so
                // membership is checked positionally.
                if let Some(region_idx) = region {
                    if !ctx.is_bel_in_region(bel, region_idx) {
                        continue;
                    }
                }
                if !ctx.bel(bel).is_valid_for_cell_type(cell_type) {
                    continue;
                }

                ripup_target = ctx.bel(bel).bound_cell().map(|c| c.id());
                if let Some(target) = ripup_target {
                    let target_info = ctx.design.cell(target);
                    if target_info.bel_strength > PlaceStrength::Strong
                        || target_info.cluster.is_some()
                    {
                        continue;
                    }
                    ctx.unbind_bel(bel);
                } else if !ctx.bel(bel).is_available() {
                    continue;
                }

                assert!(
                    ctx.bind_bel(bel, cell, PlaceStrength::Weak),
                    "BEL was free or just freed but the bind failed"
                );
                if !ctx.is_bel_location_valid(bel) {
                    ctx.unbind_bel(bel);
                    if let Some(target) = ripup_target {
                        assert!(
                            ctx.bind_bel(bel, target, PlaceStrength::Weak),
                            "failed to restore the previous occupant"
                        );
                    }
                    continue;
                }

                let mut new_metric = get_cell_metric(ctx, cell, MetricType::Cost);
                if ripup_target.is_some() {
                    new_metric *= 5;
                }
                if new_metric < best_metric {
                    best_bel = Some(bel);
                    best_metric = new_metric;
                }

                ctx.unbind_bel(bel);
                if let Some(target) = ripup_target {
                    assert!(
                        ctx.bind_bel(bel, target, PlaceStrength::Weak),
                        "failed to restore the previous occupant"
                    );
                }
            }

            let Some(best) = best_bel else {
                return false;
            };

            let displaced = ctx.bel(best).bound_cell().map(|c| c.id());
            if displaced.is_some() {
                ctx.unbind_bel(best);
            }
            assert!(
                ctx.bind_bel(best, cell, PlaceStrength::Weak),
                "best BEL was just freed but the bind failed"
            );

            // nextpnr back-annotates the chosen BEL onto the cell so a later
            // re-read of the design reproduces this placement.
            let bel_name = ctx.chipdb().bel_name(best).to_owned();
            let bel_attr = ctx.id("BEL");
            ctx.design
                .cell_mut(cell)
                .attrs
                .insert(bel_attr, Property::string(bel_name));

            // Follow the chain: whoever we displaced is placed next.
            current = displaced;
        }

        true
    }

    /// `print_stats` -- report displacement, returning the number of cells that
    /// moved or are still unplaced.
    fn print_stats(&self, ctx: &Context, point: &str) -> usize {
        let mut distance_sum = 0.0f32;
        let mut max_distance = 0.0f32;
        let mut moved_cells = 0usize;
        let mut unplaced_cells = 0usize;

        for (&cell, &orig) in &self.old_locations {
            let Some(bel) = ctx.design.cell(cell).bel else {
                unplaced_cells += 1;
                continue;
            };
            let new_loc = ctx.chipdb().bel_loc(bel);
            if new_loc != orig {
                let dx = (new_loc.x - orig.x) as f32;
                let dy = (new_loc.y - orig.y) as f32;
                let distance = (dx * dx + dy * dy).sqrt();
                moved_cells += 1;
                distance_sum += distance;
                max_distance = max_distance.max(distance);
            }
        }

        log::info!("    moved {moved_cells} cells, {unplaced_cells} unplaced (after {point})");
        if moved_cells > 0 {
            log::info!("       average distance {}", distance_sum / moved_cells as f32);
            log::info!("       maximum distance {max_distance}");
        }

        moved_cells + unplaced_cells
    }

    /// `legalise_constraints` -- the whole pass. Returns the displacement score.
    pub fn legalise_constraints(&mut self, ctx: &mut Context) -> Result<usize, PlacerError> {
        log::info!("Legalising relative constraints...");

        for cell in cell_ids(ctx) {
            let loc = ctx
                .design
                .cell(cell)
                .bel
                .map(|b| ctx.chipdb().bel_loc(b))
                .unwrap_or(Loc::new(0, 0, 0));
            self.old_locations.insert(cell, loc);
        }

        for cell in cell_ids(ctx) {
            if !self.legalise_cell(ctx, cell) {
                let name = ctx.name_of(ctx.design.cell(cell).name).to_owned();
                return Err(PlacerError::PlacementFailed(format!(
                    "failed to place chain starting at cell '{name}'"
                )));
            }
        }

        if self.print_stats(ctx, "legalising chains") == 0 {
            return Ok(0);
        }

        // Sorted so the replacement order -- and therefore the RNG draws in
        // place_single_cell -- does not depend on hash iteration order.
        let mut ripped: Vec<CellId> = self.ripped_cells.iter().copied().collect();
        ripped.sort_unstable();
        for cell in ripped {
            if !self.place_single_cell(ctx, cell) {
                let name = ctx.name_of(ctx.design.cell(cell).name).to_owned();
                return Err(PlacerError::PlacementFailed(format!(
                    "failed to place cell '{name}' after relative constraint legalisation"
                )));
            }
        }

        let score = self.print_stats(ctx, "replacing ripped up cells");

        for cell in cell_ids(ctx) {
            if get_constraints_distance(ctx, cell) != 0 {
                let name = ctx.name_of(ctx.design.cell(cell).name).to_owned();
                return Err(PlacerError::PlacementFailed(format!(
                    "constraint satisfaction check failed for cell '{name}'"
                )));
            }
        }

        Ok(score)
    }
}

/// `legalise_relative_constraints` -- true when the pass had to move something.
pub fn legalise_relative_constraints(ctx: &mut Context) -> Result<bool, PlacerError> {
    let mut worker = ConstraintLegaliseWorker::new(ctx);
    Ok(worker.legalise_constraints(ctx)? > 0)
}

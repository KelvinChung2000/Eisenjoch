//! Half-Perimeter Wire Length (HPWL) metric computation.

use crate::context::Context;
use crate::netlist::NetId;

/// Compute HPWL for a single net.
///
/// HPWL = (max_x - min_x) + (max_y - min_y) across all connected cell locations.
/// Returns 0.0 for nets with no driver, no users, or dead nets.
pub fn net_hpwl(ctx: &Context, net_idx: NetId) -> f64 {
    let net = ctx.net(net_idx);
    if !net.is_alive() || net.users().is_empty() {
        return 0.0;
    }

    let Some(driver_pin) = net.driver_cell_port() else {
        return 0.0;
    };

    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;

    let cell_indices = std::iter::once(driver_pin.cell)
        .chain(net.users().iter().filter(|u| u.is_valid()).map(|u| u.cell));

    for cell_idx in cell_indices {
        if let Some(bel) = ctx.cell(cell_idx).bel() {
            let loc = bel.loc();
            min_x = min_x.min(loc.x);
            max_x = max_x.max(loc.x);
            min_y = min_y.min(loc.y);
            max_y = max_y.max(loc.y);
        }
    }

    if min_x > max_x {
        return 0.0;
    }

    ((max_x - min_x) + (max_y - min_y)) as f64
}

/// Total HPWL across all alive nets (parallel).
pub fn total_hpwl(ctx: &Context) -> f64 {
    use rayon::prelude::*;

    let net_indices: Vec<NetId> = ctx.design.iter_alive_nets().map(|(idx, _)| idx).collect();
    net_indices.par_iter().map(|&idx| net_hpwl(ctx, idx)).sum()
}

/// Bresenham line estimate wirelength for a single net.
///
/// Counts unique tile-to-tile edge crossings across all driver-to-sink lines,
/// approximating a Steiner tree. This is cheaper than routing while still
/// reflecting how many fabric boundaries the straight-line connections cross.
pub fn net_line_estimate(ctx: &Context, net_idx: NetId) -> f64 {
    use rustc_hash::FxHashSet;

    let net = ctx.net(net_idx);
    if !net.is_alive() || net.users().is_empty() {
        return 0.0;
    }

    let Some(driver_pin) = net.driver_cell_port() else {
        return 0.0;
    };
    let Some(driver_bel) = ctx.cell(driver_pin.cell).bel() else {
        return 0.0;
    };
    let driver_loc = driver_bel.loc();
    let width = ctx.chipdb().width();
    let estimated_edge_count = net.users().iter().filter(|u| u.is_valid()).count().max(1) * 16;

    let mut edges: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(estimated_edge_count, Default::default());

    for user in net.users().iter() {
        if !user.is_valid() {
            continue;
        }
        let Some(sink_bel) = ctx.cell(user.cell).bel() else {
            continue;
        };
        let sink_loc = sink_bel.loc();
        let points =
            super::congestion::bresenham_line(driver_loc.x, driver_loc.y, sink_loc.x, sink_loc.y);
        for pair in points.windows(2) {
            let (x1, y1) = pair[0];
            let (x2, y2) = pair[1];
            let edge = {
                let a = (y1 * width + x1) as u32;
                let b = (y2 * width + x2) as u32;
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                ((lo as u64) << 32) | (hi as u64)
            };
            edges.insert(edge);
        }
    }

    edges.len() as f64
}

/// Total Bresenham line estimate across all alive nets (parallel).
pub fn total_line_estimate(ctx: &Context) -> f64 {
    use rayon::prelude::*;

    let net_indices: Vec<NetId> = ctx.design.iter_alive_nets().map(|(idx, _)| idx).collect();
    net_indices
        .par_iter()
        .map(|&idx| net_line_estimate(ctx, idx))
        .sum()
}

/// Total routed wirelength (wire count across all nets). Only meaningful after routing.
pub fn total_routed_wirelength(ctx: &Context) -> usize {
    ctx.design
        .iter_alive_nets()
        .map(|(_, net)| net.wires.len())
        .sum()
}

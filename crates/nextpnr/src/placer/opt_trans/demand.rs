//! Continuous line-length estimate for the Beckmann placer.

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashMap;
use std::env;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::metrics::congestion::bresenham_line_into;

use super::network::PipeNetwork;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pin position in network coords plus the index of the cell whose motion
/// drives this pin. Cluster children move rigidly with their root, so their
/// position is `root_pos + (constr_x, constr_y)` and the driving index is
/// the root's. Returns `None` for the third value only when the pin belongs
/// to a non-movable already-placed cell.
fn pin_pos(
    ctx: &Context,
    cell_id: CellId,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
    coord_scale: f64,
) -> (f64, f64, Option<usize>) {
    if let Some(&idx) = cell_to_idx.get(&cell_id) {
        return (cell_x[idx], cell_y[idx], Some(idx));
    }

    let cell = ctx.design.cell(cell_id);

    if let Some(root_id) = cell.cluster {
        if root_id != cell_id {
            if let Some(&root_idx) = cell_to_idx.get(&root_id) {
                return (
                    cell_x[root_idx] + cell.constr_x as f64 / coord_scale,
                    cell_y[root_idx] + cell.constr_y as f64 / coord_scale,
                    Some(root_idx),
                );
            }
            let root_cell = ctx.design.cell(root_id);
            if let Some(root_bel) = root_cell.bel {
                let loc = ctx.bel(root_bel).loc();
                return (
                    (loc.x + cell.constr_x - network.x0) as f64 / coord_scale,
                    (loc.y + cell.constr_y - network.y0) as f64 / coord_scale,
                    None,
                );
            }
        }
    }

    if let Some(bel) = cell.bel {
        let loc = ctx.bel(bel).loc();
        return (
            (loc.x - network.x0) as f64 / coord_scale,
            (loc.y - network.y0) as f64 / coord_scale,
            None,
        );
    }

    panic!(
        "demand::pin_pos: cell {} is neither movable, a placed cluster child, nor bound to a BEL",
        ctx.name_of(cell.name),
    );
}

/// Compute a continuous analogue of the line estimate from current floating-point positions.
pub fn continuous_line_estimate(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> f64 {
    let coord_scale = 1.0;
    let width = ctx.chipdb().width();
    let skip_constants = env::var("NPNR_OT_INCLUDE_CONSTANTS").ok().as_deref() != Some("1");
    let skip_clocks = env::var("NPNR_OT_EXCLUDE_CLOCKS").ok().as_deref() == Some("1")
        || env::var("NPNR_OT_EXCLUDE_GLOBALS").ok().as_deref() == Some("1");
    let net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();

    net_ids
        .par_iter()
        .map(|&net_id| {
            let net = ctx.design.net(net_id);
            let net_name = ctx.name_of(net.name);
            let is_const = net_name == "$PACKER_GND_NET" || net_name == "$PACKER_VCC_NET";
            if skip_constants && is_const {
                return 0.0;
            }
            if skip_clocks {
                let lower = net_name.to_ascii_lowercase();
                if lower.contains("clk") || lower.contains("clock") {
                    return 0.0;
                }
            }
            let Some(dp) = net.driver() else {
                return 0.0;
            };
            if net.num_users() == 0 {
                return 0.0;
            }

            let (dx, dy, _) = pin_pos(
                ctx,
                dp.cell,
                cell_to_idx,
                cell_x,
                cell_y,
                network,
                coord_scale,
            );
            let driver = (dx.round() as i32, dy.round() as i32);
            // Only the number of distinct tile edges matters, and a net's are
            // few, so they are collected and sorted rather than hashed. The
            // hash set here was 20 % of the placement profile.
            let mut edges: Vec<u64> = Vec::with_capacity(8 * net.num_users());
            let mut points: Vec<(i32, i32)> = Vec::new();

            for user in net.users() {
                if !user.is_valid() {
                    continue;
                }
                let (sx, sy, _) = pin_pos(
                    ctx,
                    user.cell,
                    cell_to_idx,
                    cell_x,
                    cell_y,
                    network,
                    coord_scale,
                );
                let sink = (sx.round() as i32, sy.round() as i32);
                bresenham_line_into(&mut points, driver.0, driver.1, sink.0, sink.1);
                for pair in points.windows(2) {
                    let (x1, y1) = pair[0];
                    let (x2, y2) = pair[1];
                    let a = (y1 * width + x1) as u32;
                    let b = (y2 * width + x2) as u32;
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    edges.push(((lo as u64) << 32) | (hi as u64));
                }
            }

            edges.sort_unstable();
            edges.dedup();
            edges.len() as f64
        })
        .reduce(|| 0.0, |a, b| a + b)
}


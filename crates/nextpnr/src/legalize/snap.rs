//! Snap continuous positions to the nearest CLB column and row.
//!
//! The continuous placer works on the full tile grid, but BELs only exist
//! on CLB tiles. Snapping reduces legalization displacement significantly.

use std::collections::HashSet;

use crate::common::IdString;
use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::opt_trans::network::PipeNetwork;

use rustc_hash::FxHashMap;

/// Snap continuous positions to the nearest CLB column and row.
///
/// Also performs per-type capacity spreading to avoid overcrowding at
/// individual CLB tiles.
pub fn snap_to_clb_grid(
    ctx: &Context,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    idx_to_cell: &[CellId],
    network: &PipeNetwork,
) {
    let chipdb = ctx.chipdb();
    let net_w = network.width;
    let net_h = network.height;
    let x0 = network.x0;
    let y0 = network.y0;

    // Build sorted list of CLB x-columns and y-rows.
    let mut clb_xs: Vec<f64> = Vec::new();
    let mut clb_ys: Vec<f64> = Vec::new();
    let mut seen_x = HashSet::new();
    let mut seen_y = HashSet::new();
    for vy in 0..net_h {
        for vx in 0..net_w {
            let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
            let bel_count = chipdb.tile_type(tile).bels.len();
            if bel_count >= 4 {
                if seen_x.insert(vx) {
                    clb_xs.push(vx as f64);
                }
                if seen_y.insert(vy) {
                    clb_ys.push(vy as f64);
                }
            }
        }
    }
    clb_xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    clb_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if clb_xs.is_empty() || clb_ys.is_empty() {
        return;
    }

    let n = cell_x.len();
    let mut snap_max_disp = 0.0f64;
    let mut snap_total_disp = 0.0f64;

    for i in 0..n {
        let cx = cell_x[i];
        let best_x = clb_xs
            .iter()
            .copied()
            .min_by(|a, b| (a - cx).abs().partial_cmp(&(b - cx).abs()).unwrap())
            .unwrap();
        cell_x[i] = best_x;

        let cy = cell_y[i];
        let best_y = clb_ys
            .iter()
            .copied()
            .min_by(|a, b| (a - cy).abs().partial_cmp(&(b - cy).abs()).unwrap())
            .unwrap();
        cell_y[i] = best_y;

        let d = ((best_x - cx).powi(2) + (best_y - cy).powi(2)).sqrt();
        snap_total_disp += d;
        snap_max_disp = snap_max_disp.max(d);
    }

    let snap_avg = snap_total_disp / n.max(1) as f64;
    eprintln!(
        "Snapped {} cells to {} CLB columns x {} CLB rows (avg_disp={:.1}, max_disp={:.1})",
        n,
        clb_xs.len(),
        clb_ys.len(),
        snap_avg,
        snap_max_disp,
    );

    // Per-type capacity spreading: ensure no tile has more cells than BELs.
    spread_overcrowded_cells(ctx, cell_x, cell_y, idx_to_cell, network, &clb_xs, &clb_ys);
}

/// Move excess cells from overcrowded tiles to the nearest tile with capacity.
fn spread_overcrowded_cells(
    ctx: &Context,
    cell_x: &mut [f64],
    cell_y: &mut [f64],
    idx_to_cell: &[CellId],
    network: &PipeNetwork,
    clb_xs: &[f64],
    clb_ys: &[f64],
) {
    use rustc_hash::FxHashSet;

    let chipdb = ctx.chipdb();
    let x0 = network.x0;
    let y0 = network.y0;
    let n = cell_x.len();

    // Count distinct cell types.
    let mut cell_types_present: FxHashSet<IdString> = FxHashSet::default();
    for i in 0..n {
        let ct = ctx.design.cell(idx_to_cell[i]).cell_type;
        cell_types_present.insert(ct);
    }
    let n_types = cell_types_present.len().max(1);

    // Per-tile capacity = total BELs / number of cell types (fair share).
    let mut tile_cap: FxHashMap<(i32, i32), usize> = FxHashMap::default();
    for &cx in clb_xs {
        for &cy in clb_ys {
            let vx = cx as i32;
            let vy = cy as i32;
            let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
            let total_bels = chipdb.tile_type(tile).bels.len();
            tile_cap.insert((vx, vy), total_bels / n_types);
        }
    }

    // Group cells by (type, snap position).
    let mut type_tile_cells: FxHashMap<(IdString, i32, i32), Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        let ct = ctx.design.cell(idx_to_cell[i]).cell_type;
        let tx = cell_x[i].round() as i32;
        let ty = cell_y[i].round() as i32;
        type_tile_cells.entry((ct, tx, ty)).or_default().push(i);
    }

    // Per-type remaining capacity.
    let mut remaining_cap: FxHashMap<(IdString, i32, i32), usize> = FxHashMap::default();
    for &ct in &cell_types_present {
        for (&(vx, vy), &cap) in &tile_cap {
            remaining_cap.insert((ct, vx, vy), cap);
        }
    }

    let mut spread_count = 0usize;
    let mut spread_max_disp = 0.0f64;

    let mut groups: Vec<_> = type_tile_cells.into_iter().collect();
    groups.sort_by_key(|(_, cells)| std::cmp::Reverse(cells.len()));

    for ((ct, tx, ty), cells) in &groups {
        let cap = remaining_cap.get(&(*ct, *tx, *ty)).copied().unwrap_or(0);
        if cells.len() <= cap {
            if let Some(c) = remaining_cap.get_mut(&(*ct, *tx, *ty)) {
                *c = c.saturating_sub(cells.len());
            }
            continue;
        }

        let mut cell_dists: Vec<(usize, f64)> = cells
            .iter()
            .map(|&ci| {
                let dx = cell_x[ci] - *tx as f64;
                let dy = cell_y[ci] - *ty as f64;
                (ci, dx * dx + dy * dy)
            })
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        remaining_cap.insert((*ct, *tx, *ty), 0);

        for &(ci, _) in cell_dists.iter().skip(cap) {
            let mut best: Option<(i32, i32, f64)> = None;
            for (&(bt, bx, by), cap_left) in &remaining_cap {
                if bt != *ct || *cap_left == 0 {
                    continue;
                }
                let dx = (bx - tx) as f64;
                let dy = (by - ty) as f64;
                let d = dx * dx + dy * dy;
                if best.is_none() || d < best.unwrap().2 {
                    best = Some((bx, by, d));
                }
            }
            if let Some((bx, by, d)) = best {
                cell_x[ci] = bx as f64;
                cell_y[ci] = by as f64;
                *remaining_cap.get_mut(&(*ct, bx, by)).unwrap() -= 1;
                spread_count += 1;
                spread_max_disp = spread_max_disp.max(d.sqrt());
            }
        }
    }

    if spread_count > 0 {
        eprintln!(
            "Spread {} cells from overcrowded tiles (max_disp={:.1})",
            spread_count, spread_max_disp,
        );
    }
}

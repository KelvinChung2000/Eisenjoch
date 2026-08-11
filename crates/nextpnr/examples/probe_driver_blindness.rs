//! Measure how far the DCD's sink-only cost displaces each cell's optimum.
//!
//! `evaluate_cell_at` scores a cell against `dist_cache.get(net, node)` only
//! for nets where the cell is a SINK (`coord_descent.rs`, `if !pin.is_driver`).
//! The nets it DRIVES contribute nothing. That is not laziness: `dist_cache` is
//! anchored at the driver, so a driver move invalidates the whole row and the
//! table cannot price it.
//!
//! Every driver->sink term depends on both endpoints, so exactly one endpoint
//! of every term is unpriced. This probe asks what that costs, on the placement
//! the DCD actually produces:
//!
//!   visible_opt = the Manhattan optimum over the neighbours DCD can see
//!                 (drivers of the nets this cell sinks on)
//!   true_opt    = the Manhattan optimum over ALL pin neighbours
//!                 (those drivers, plus the sinks of the nets it drives)
//!
//! `gap` is the distance between them: how far the cell is pulled from where
//! the full objective wants it. `excess` is the wirelength left on the table.
//!
//! Manhattan is a surrogate for the congestion-aware Dijkstra metric the placer
//! uses -- it measures the asymmetry, not the placer's own cost. That is the
//! point: the asymmetry is structural and survives any choice of metric.
//!
//! Usage:
//!   PROBE_ITERS=5 PROBE_THREADS=4 probe_driver_blindness [chipdb] [design]

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use std::path::Path;
use std::time::Instant;

/// Manhattan optimum of a point set is the coordinate-wise median.
fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN positions"));
    v[v.len() / 2]
}

fn sum_manhattan(pts: &[(f64, f64)], x: f64, y: f64) -> f64 {
    pts.iter()
        .map(|(px, py)| (px - x).abs() + (py - y).abs())
        .sum()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = std::env::var("PROBE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    cfg.num_threads = std::env::var("PROBE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let t = Instant::now();
    place_opt_trans(&mut ctx, &cfg).expect("place_opt_trans");
    let place_secs = t.elapsed().as_secs_f64();

    // Dense index over placed cells: CellId's slot accessor is crate-private,
    // so build the mapping here rather than indexing by raw id.
    let mut slot_of: std::collections::HashMap<nextpnr::netlist::CellId, usize> =
        std::collections::HashMap::new();
    let mut pos: Vec<(f64, f64)> = Vec::new();
    for (cid, cell) in ctx.design.iter_alive_cells() {
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            slot_of.insert(cid, pos.len());
            pos.push((loc.x as f64, loc.y as f64));
        }
    }

    // Per-cell neighbour sets, split by whether the DCD can price them.
    let mut seen: Vec<(f64, f64)> = Vec::new();
    let mut in_nbrs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); pos.len()];
    let mut out_nbrs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); pos.len()];
    let mut n_pairs = 0usize;

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
            continue;
        }
        let Some(driver) = net.driver() else { continue };
        let Some(&dslot) = slot_of.get(&driver.cell) else {
            continue;
        };
        let dpos = pos[dslot];
        seen.clear();
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let Some(&uslot) = slot_of.get(&user.cell) else {
                continue;
            };
            n_pairs += 1;
            // The sink sees the driver: this is the term DCD prices.
            in_nbrs[uslot].push(dpos);
            seen.push(pos[uslot]);
        }
        // The driver sees its sinks: these are the terms DCD does NOT price.
        out_nbrs[dslot].extend_from_slice(&seen);
    }

    let mut gaps: Vec<f64> = Vec::new();
    let mut rel_excesses: Vec<f64> = Vec::new();
    let mut total_excess = 0.0f64;
    let mut total_true_cost = 0.0f64;
    let mut n_scored = 0usize;
    let mut n_blind_only = 0usize; // cells with NO priced neighbour at all
    let mut sum_in = 0usize;
    let mut sum_out = 0usize;

    for slot in 0..pos.len() {
        let ins = &in_nbrs[slot];
        let outs = &out_nbrs[slot];
        if ins.is_empty() && outs.is_empty() {
            continue;
        }
        let (cx, cy) = pos[slot];
        if ins.is_empty() {
            n_blind_only += 1;
        }

        let mut vx: Vec<f64> = ins.iter().map(|p| p.0).collect();
        let mut vy: Vec<f64> = ins.iter().map(|p| p.1).collect();
        let (vox, voy) = if ins.is_empty() {
            (cx, cy)
        } else {
            (median(&mut vx), median(&mut vy))
        };

        let all: Vec<(f64, f64)> = ins.iter().chain(outs.iter()).copied().collect();
        let mut ax: Vec<f64> = all.iter().map(|p| p.0).collect();
        let mut ay: Vec<f64> = all.iter().map(|p| p.1).collect();
        let (tox, toy) = (median(&mut ax), median(&mut ay));

        let gap = (vox - tox).abs() + (voy - toy).abs();
        // Wirelength the FULL neighbour set pays because the cell sits at the
        // sink-only optimum instead of the true one. Both medians are genuine
        // Manhattan minimisers of their own sets, so this is >= 0.
        let true_cost = sum_manhattan(&all, tox, toy);
        let excess = sum_manhattan(&all, vox, voy) - true_cost;

        gaps.push(gap);
        // Relative, because absolute tile counts just track how sparse the
        // design is on the die: sv3 puts 341 cells on a 311x223 grid.
        if true_cost > 0.0 {
            rel_excesses.push(excess / true_cost);
        }
        total_excess += excess;
        total_true_cost += true_cost;
        sum_in += ins.len();
        sum_out += outs.len();
        n_scored += 1;
        let _ = (cx, cy);
    }

    gaps.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    rel_excesses.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let nf = n_scored.max(1) as f64;
    let hpwl = nextpnr::metrics::total_hpwl(&ctx);

    println!(
        "DRIVERBLIND place_s={place_secs:.1} iters={} cells_scored={n_scored} pairs={n_pairs} \
         hpwl={hpwl:.0} avg_priced_nbrs={:.2} avg_unpriced_nbrs={:.2} unpriced_only_cells={n_blind_only}",
        cfg.max_outer_iters,
        sum_in as f64 / nf,
        sum_out as f64 / nf,
    );
    println!(
        "DRIVERBLIND gap_mean={:.2} gap_p50={:.2} gap_p90={:.2} gap_max={:.2} frac_gap_gt1={:.3}",
        gaps.iter().sum::<f64>() / nf,
        pct(&gaps, 0.50),
        pct(&gaps, 0.90),
        gaps.last().copied().unwrap_or(0.0),
        gaps.iter().filter(|g| **g > 1.0).count() as f64 / nf,
    );
    println!(
        "DRIVERBLIND rel_excess_p50={:.3} rel_excess_p90={:.3} frac_rel_gt05={:.3} \
         frac_rel_gt20={:.3} aggregate_rel_excess={:.3}",
        pct(&rel_excesses, 0.50),
        pct(&rel_excesses, 0.90),
        rel_excesses.iter().filter(|r| **r > 0.05).count() as f64 / nf,
        rel_excesses.iter().filter(|r| **r > 0.20).count() as f64 / nf,
        total_excess / total_true_cost.max(1.0),
    );
}

//! Main Beckmann OT placement algorithm.
//!
//! Uses inner/outer decomposition: expensive Dial-logit solves in the outer
//! loop cache soft-cost fields, cheap discrete coordinate descent in the inner
//! loop moves cells to minimum-cost grid nodes.

use crate::common::IdString;
use crate::context::Context;
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::report;
use crate::placer::PlacerError;
use rayon::{prelude::*, ThreadPoolBuilder};

use crate::placer::common::TypeAwarePlacement;

use super::config::OptTransPlacerCfg;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Main Beckmann OT placement using inner/outer coordinate descent.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    let mut cfg = cfg.clone();
    cfg.apply_env_overrides();
    let t_alg = std::time::Instant::now();
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;
    eprintln!("  ALG_T prepare_discrete: {:.1}s", t_alg.elapsed().as_secs_f64());
    crate::solver::set_solver_threads(cfg.num_threads);

    let target_scale = 1.0;
    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    eprintln!("  ALG_T collect_movable: {:.1}s", t_alg.elapsed().as_secs_f64());
    let alive_net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();
    let n = idx_to_cell.len();
    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    let phys_max_x = (ctx.chipdb().width() - 1) as f64;
    let phys_max_y = (ctx.chipdb().height() - 1) as f64;
    let phys_grid_w = ctx.chipdb().width() as usize;
    let phys_grid_h = ctx.chipdb().height() as usize;
    let solve_pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads.max(1))
        .build()
        .map_err(|e| PlacerError::PlacementFailed(format!("thread pool: {e}")))?;
    eprintln!("  ALG_T pre_network: {:.1}s", t_alg.elapsed().as_secs_f64());
    let mut network = PipeNetwork::from_context(ctx, target_scale);
    eprintln!("  ALG_T network: {:.2}s", t_alg.elapsed().as_secs_f64());

    match cfg.init_strategy {
        super::config::InitStrategy::Centroid => {
            common::init_positions_center_drop(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        super::config::InitStrategy::Topological => {
            common::init_positions_topological(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        _ => {
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
    }

    if std::env::var("NPNR_OT_DUMP_DIST").ok().as_deref() == Some("1") {
        report_distribution("init", &cell_x, &cell_y, phys_max_x, phys_max_y);
    }

    solve_pool.install(|| {
        cell_x.par_iter_mut().for_each(|v| *v -= network.x0 as f64);
        cell_y.par_iter_mut().for_each(|v| *v -= network.y0 as f64);
    });

    let type_aware = TypeAwarePlacement::build(ctx, network.x0, network.y0);
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();
    let cell_pin_weights: Vec<f64> = idx_to_cell
        .iter()
        .map(|&ci| {
            let cell = ctx.cell(ci);
            cell.ports()
                .filter(|pin| cell.port_net(pin.port).is_some())
                .count() as f64
        })
        .collect();

    let resistance_model = ResistanceModel;

    eprintln!(
        "Coordinate descent mode: {} cells, scale {:.2}",
        n, target_scale
    );

    // Multi-start warm-up: try several random initial placements with a short
    // DCD each, keep the lowest-energy one as the starting point for the full
    // run. Empirically the energy landscape has multiple basins that differ
    // by 30-40% in final energy; picking among warmups avoids the lottery
    // of single-seed initialization.
    let warmup_starts = std::env::var("NPNR_OT_WARMUP_STARTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let warmup_iters = std::env::var("NPNR_OT_WARMUP_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);

    if warmup_starts >= 2 {
        eprintln!(
            "Warmup multi-start: {} candidates × {} iters",
            warmup_starts, warmup_iters
        );
        let mut best_energy = f64::INFINITY;
        let mut best_x = cell_x.clone();
        let mut best_y = cell_y.clone();

        for k in 0..warmup_starts {
            let mut wx = cell_x.clone();
            let mut wy = cell_y.clone();
            // Re-shuffle the BEL bindings under a per-warmup seed to give
            // each candidate a different initial layout. Cells stay within
            // their valid bucket; only WHICH valid BEL is picked changes.
            let warmup_seed = cfg
                .seed
                .wrapping_add(k as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ctx.reseed_rng(warmup_seed);
            // Unbind movable cells, rebind randomly via prepare_discrete's
            // helper, then re-read positions.
            for &cid in &idx_to_cell {
                if let Some(bel) = ctx.design.cell(cid).bel {
                    if !ctx.design.cell(cid).bel_strength.is_locked() {
                        ctx.unbind_bel(bel);
                    }
                }
            }
            common::initial_placement(ctx)?;
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut wx, &mut wy);
            solve_pool.install(|| {
                wx.par_iter_mut().for_each(|v| *v -= network.x0 as f64);
                wy.par_iter_mut().for_each(|v| *v -= network.y0 as f64);
            });

            // Fresh dynamic state for the warmup attempt.
            network.reset();

            let mut warmup_cfg = cfg.clone();
            warmup_cfg.max_outer_iters = warmup_iters;
            let energy = super::coord_descent::run_inner_outer(
                ctx,
                &mut wx,
                &mut wy,
                &mut network,
                &cell_to_idx,
                &idx_to_cell,
                &alive_net_ids,
                &warmup_cfg,
                &solve_pool,
                &resistance_model,
                &type_aware,
                &cell_buckets,
                &cell_pin_weights,
                phys_max_x,
                phys_max_y,
                phys_grid_w,
                phys_grid_h,
            );
            eprintln!(
                "  Warmup {}/{}: seed=0x{:x} energy={:.1}",
                k + 1,
                warmup_starts,
                warmup_seed,
                energy
            );
            if energy < best_energy {
                best_energy = energy;
                best_x = wx;
                best_y = wy;
            }
        }

        eprintln!(
            "Warmup selected: energy={:.1} (continuing with full DCD)",
            best_energy
        );
        cell_x = best_x;
        cell_y = best_y;
        network.reset();
    }

    super::coord_descent::run_inner_outer(
        ctx,
        &mut cell_x,
        &mut cell_y,
        &mut network,
        &cell_to_idx,
        &idx_to_cell,
        &alive_net_ids,
        &cfg,
        &solve_pool,
        &resistance_model,
        &type_aware,
        &cell_buckets,
        &cell_pin_weights,
        phys_max_x,
        phys_max_y,
        phys_grid_w,
        phys_grid_h,
    );
    eprintln!("  ALG_T dcd_done: {:.2}s", t_alg.elapsed().as_secs_f64());
    super::coord_descent::report_collect_time();

    if std::env::var("NPNR_OT_DUMP_SPAN_USAGE").ok().as_deref() == Some("1") {
        network.report_span_utilization("post-DCD");
    }

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();

    // Two full wirelength passes that no placement decision reads. HeAP's
    // driver computes its equivalent after stopping the clock, so keeping
    // them inside `place()` charged us for measurement HeAP is not charged
    // for. `NPNR_OT_REPORT_HPWL=1` puts them back for anything that parses
    // the lines.
    let report_hpwl = std::env::var("NPNR_OT_REPORT_HPWL").ok().as_deref() == Some("1");
    if report_hpwl {
        let pre_legal_hpwl = continuous_hpwl(ctx, &cell_to_idx, &phys_x, &phys_y);
        eprintln!("Pre-legalization: HPWL={:.0}", pre_legal_hpwl);
    }

    if std::env::var("NPNR_OT_DUMP_DIST").ok().as_deref() == Some("1") {
        report_distribution("DCD-end", &phys_x, &phys_y, phys_max_x, phys_max_y);
    }

    crate::placer::legalize::legalize(ctx, &idx_to_cell, &phys_x, &phys_y, &cfg.legalization)?;
    eprintln!("  ALG_T legalize_done: {:.2}s", t_alg.elapsed().as_secs_f64());

    if std::env::var("NPNR_OT_CHECK_ROUTABILITY").ok().as_deref() == Some("1") {
        let report_g = crate::placer::routability::check_routability_global(ctx);
        eprintln!(
            "Routability (global): {} nets ({} skipped) in {:.1}ms, {} infeasible, {} inconclusive",
            report_g.n_checked,
            report_g.n_skipped,
            report_g.elapsed_ms,
            report_g.infeasible.len(),
            report_g.n_inconclusive
        );
        for inf in report_g.infeasible.iter().take(15) {
            let net = ctx.net(inf.net_id);
            let name = ctx.name_of(net.name_id());
            eprintln!(
                "  global-infeasible: net='{}' driver={:?} unreached={}",
                name,
                inf.driver_wire,
                inf.unreached.len()
            );
        }

        let report_p = crate::placer::routability::check_routability(ctx);
        eprintln!(
            "Routability (per-pair): {} nets ({} skipped), {} pairs in {:.1}ms, {} infeasible, {} inconclusive",
            report_p.n_checked,
            report_p.n_skipped,
            report_p.n_pairs,
            report_p.elapsed_ms,
            report_p.infeasible.len(),
            report_p.n_inconclusive
        );
        for inf in report_p.infeasible.iter().take(15) {
            let net = ctx.net(inf.net_id);
            let name = ctx.name_of(net.name_id());
            eprintln!(
                "  perpair-infeasible: net='{}' driver={:?} unreached={}",
                name,
                inf.driver_wire,
                inf.unreached.len()
            );
        }
    }

    if report_hpwl {
        report::report_post_legalization(ctx);
    }
    // Ranking 105 000 nets by wirelength is a diagnostic, and it cost 1.5s of
    // the 5.3s run -- more than legalisation. HeAP's driver computes its
    // equivalent after stopping the clock, so leaving this on by default made
    // every place-time comparison against it wrong by that much.
    if std::env::var("NPNR_OT_REPORT_TOP_NETS").ok().as_deref() == Some("1") {
        report::report_top_net_hpwl(ctx, 10);
    }
    report::dump_position_csv_from_env(ctx, "NPNR_OT_DUMP_CSV", &idx_to_cell, &phys_x, &phys_y)
        .expect("dump placement CSV");

    PlacerPipeline::validate(ctx)?;
    Ok(())
}

/// Dump distribution stats for continuous positions: centroid, std dev,
/// bounding box, and a coarse occupancy histogram. Useful for tracing
/// where cells actually end up at each pipeline stage.
fn report_distribution(stage: &str, x: &[f64], y: &[f64], max_x: f64, max_y: f64) {
    let n = x.len();
    if n == 0 {
        eprintln!("[dist {}] empty", stage);
        return;
    }
    let nf = n as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / nf;
    let mean_y: f64 = y.iter().sum::<f64>() / nf;
    let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / nf;
    let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / nf;
    let mut min_x = f64::INFINITY;
    let mut max_xv = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_yv = f64::NEG_INFINITY;
    for i in 0..n {
        if x[i] < min_x {
            min_x = x[i];
        }
        if x[i] > max_xv {
            max_xv = x[i];
        }
        if y[i] < min_y {
            min_y = y[i];
        }
        if y[i] > max_yv {
            max_yv = y[i];
        }
    }
    // 8x8 coarse occupancy
    let bw = (max_x + 1.0) / 8.0;
    let bh = (max_y + 1.0) / 8.0;
    let mut bins = [[0u32; 8]; 8];
    for i in 0..n {
        let bx = ((x[i] / bw).floor() as i32).clamp(0, 7) as usize;
        let by = ((y[i] / bh).floor() as i32).clamp(0, 7) as usize;
        bins[by][bx] += 1;
    }
    // Within-1-tile-of-centroid fraction
    let mut piled = 0u32;
    for i in 0..n {
        let dx = (x[i] - mean_x).abs();
        let dy = (y[i] - mean_y).abs();
        if dx <= 1.0 && dy <= 1.0 {
            piled += 1;
        }
    }
    eprintln!(
        "[dist {}] n={} centroid=({:.1},{:.1}) std=({:.2},{:.2}) bbox=({:.0},{:.0})-({:.0},{:.0}) piled<=1tile={:.1}%",
        stage, n, mean_x, mean_y, var_x.sqrt(), var_y.sqrt(),
        min_x, min_y, max_xv, max_yv,
        100.0 * piled as f64 / nf,
    );
    eprintln!("[dist {}] 8x8 occupancy (row top->bottom):", stage);
    for row in (0..8).rev() {
        let line: Vec<String> = bins[row].iter().map(|c| format!("{:>6}", c)).collect();
        eprintln!("  {}", line.join(" "));
    }
}

/// Compute HPWL from continuous (physical) cell positions, mirroring
/// `metrics::wirelength::total_hpwl` but reading from `cell_x`/`cell_y`
/// indexed by `cell_to_idx`. Cluster children move rigidly with their root,
/// so their pin position is `root_pos + (constr_x, constr_y)`.
fn continuous_hpwl(
    ctx: &Context,
    cell_to_idx: &rustc_hash::FxHashMap<crate::netlist::CellId, usize>,
    phys_x: &[f64],
    phys_y: &[f64],
) -> f64 {
    use rayon::prelude::*;

    let net_ids: Vec<crate::netlist::NetId> =
        ctx.design.iter_alive_nets().map(|(id, _)| id).collect();

    net_ids
        .par_iter()
        .map(|&net_id| {
            let net = ctx.net(net_id);
            if !net.is_alive() || net.users().is_empty() {
                return 0.0;
            }
            let Some(driver_pin) = net.driver_cell_port() else {
                return 0.0;
            };

            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            let mut visit = |cid: crate::netlist::CellId| {
                if let Some(&idx) = cell_to_idx.get(&cid) {
                    min_x = min_x.min(phys_x[idx]);
                    max_x = max_x.max(phys_x[idx]);
                    min_y = min_y.min(phys_y[idx]);
                    max_y = max_y.max(phys_y[idx]);
                    return;
                }
                let cell = ctx.design.cell(cid);
                if let Some(root_id) = cell.cluster {
                    if root_id != cid {
                        if let Some(&ridx) = cell_to_idx.get(&root_id) {
                            let x = phys_x[ridx] + cell.constr_x as f64;
                            let y = phys_y[ridx] + cell.constr_y as f64;
                            min_x = min_x.min(x);
                            max_x = max_x.max(x);
                            min_y = min_y.min(y);
                            max_y = max_y.max(y);
                            return;
                        }
                    }
                }
                if let Some(bel) = cell.bel {
                    let loc = ctx.bel(bel).loc();
                    min_x = min_x.min(loc.x as f64);
                    max_x = max_x.max(loc.x as f64);
                    min_y = min_y.min(loc.y as f64);
                    max_y = max_y.max(loc.y as f64);
                }
            };

            visit(driver_pin.cell);
            for u in net.users() {
                if u.is_valid() {
                    visit(u.cell);
                }
            }

            if min_x.is_finite() {
                (max_x - min_x) + (max_y - min_y)
            } else {
                0.0
            }
        })
        .sum()
}

//! Main Beckmann OT placement algorithm.
//!
//! Fixed-resolution Adam driven by the per-net path solver.

use crate::common::IdString;
use crate::context::Context;
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::report;
use crate::placer::PlacerError;
use crate::solver::optimizer::AdamOptimizer;
use rayon::{prelude::*, ThreadPoolBuilder};
use rustc_hash::FxHashMap;

use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;

use super::config::OptTransPlacerCfg;
use super::demand;
use super::demand::NetSolveInfo;
use super::network::PipeNetwork;
use super::path_solver;
use super::resistance::ResistanceModel;
use std::env;

#[derive(Clone, Copy, Debug)]
struct ObjectiveState {
    energy: f64,
    line: f64,
    chpwl: f64,
    max_overflow: f64,
    n_overflow: usize,
    overflow_excess: f64,
}

#[derive(Clone, Copy, Debug)]
struct IterSnapshot {
    energy_before: f64,
    energy_after: f64,
    chpwl_before: f64,
    chpwl_after: f64,
    line_before: f64,
    line_after: f64,
    max_overflow_before: f64,
    max_overflow_after: f64,
    overflow_excess_before: f64,
    overflow_excess_after: f64,
    rel_drop: f64,
}

fn update_effective_conductance(
    solve_pool: &rayon::ThreadPool,
    network: &mut PipeNetwork,
    resistance_model: &ResistanceModel,
) {
    solve_pool.install(|| {
        network.pipes.par_iter_mut().for_each(|pipe| {
            let r_eff = resistance_model.effective_resistance(pipe);
            pipe.eff_conductance = 1.0 / r_eff.max(1e-12);
        });
    });
}

/// Write the path solver's per-pipe usage back into `pipe.net_count` so that
/// the next `update_effective_conductance` sweep can apply the parameterless
/// log-barrier. `edge_usage` is the hard integer count of nets whose Dijkstra
/// path traversed the pipe in the previous iteration (stored as f64 by the
/// path solver; values are non-negative integers in practice).
fn refresh_net_counts_from_usage(
    solve_pool: &rayon::ThreadPool,
    network: &mut PipeNetwork,
    edge_usage: &[f64],
) {
    solve_pool.install(|| {
        network
            .pipes
            .par_iter_mut()
            .zip(edge_usage.par_iter())
            .for_each(|(pipe, &usage)| {
                // usage is non-negative by construction; saturate to u32 max.
                let clamped = usage.max(0.0).min(u32::MAX as f64);
                pipe.net_count = clamped as u32;
            });
    });
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

fn clone_network(network: &PipeNetwork) -> PipeNetwork {
    PipeNetwork {
        nodes: network.nodes.clone(),
        pipes: network.pipes.clone(),
        node_pipes: network.node_pipes.clone(),
        pipe_lookup: network.pipe_lookup.clone(),
        width: network.width,
        height: network.height,
        x0: network.x0,
        y0: network.y0,
        zero_bel_tiles: network.zero_bel_tiles,
        total_bels: network.total_bels,
        coarsen: network.coarsen,
    }
}

fn collect_net_infos(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    include_debug_names: bool,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    network: &PipeNetwork,
) -> Vec<NetSolveInfo> {
    let c = network.coarsen as f64;
    let mut coarse_x = Vec::with_capacity(cell_x.len());
    let mut coarse_y = Vec::with_capacity(cell_y.len());
    fill_coarse_coords(&mut coarse_x, &mut coarse_y, cell_x, cell_y, c);
    collect_net_infos_from_coarse(
        ctx,
        net_ids,
        include_debug_names,
        cell_to_idx,
        &coarse_x,
        &coarse_y,
        network,
    )
}

fn fill_coarse_coords(
    coarse_x: &mut Vec<f64>,
    coarse_y: &mut Vec<f64>,
    cell_x: &[f64],
    cell_y: &[f64],
    c: f64,
) {
    coarse_x.clear();
    coarse_y.clear();
    coarse_x.reserve(cell_x.len().saturating_sub(coarse_x.capacity()));
    coarse_y.reserve(cell_y.len().saturating_sub(coarse_y.capacity()));
    coarse_x.extend(cell_x.iter().map(|&x| x / c));
    coarse_y.extend(cell_y.iter().map(|&y| y / c));
}

fn collect_net_infos_from_coarse(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    include_debug_names: bool,
    cell_to_idx: &FxHashMap<CellId, usize>,
    coarse_x: &[f64],
    coarse_y: &[f64],
    network: &PipeNetwork,
) -> Vec<NetSolveInfo> {
    demand::collect_nets_for_solve(
        ctx,
        net_ids,
        include_debug_names,
        cell_to_idx,
        coarse_x,
        coarse_y,
        network,
    )
}

fn update_pressure_and_flow(
    network: &mut PipeNetwork,
    global_pressure: &[f64],
    solve_pool: &rayon::ThreadPool,
) {
    solve_pool.install(|| {
        network
            .nodes
            .par_iter_mut()
            .zip(global_pressure.par_iter())
            .for_each(|(node, &p)| {
                node.pressure = p;
            });
        network.pipes.par_iter_mut().for_each(|pipe| {
            pipe.flow =
                (global_pressure[pipe.from] - global_pressure[pipe.to]) * pipe.eff_conductance;
        });
    });
}

#[derive(Clone, Debug)]
struct FluxForwardStats {
    global_pressure: Vec<f64>,
    attraction_energy: f64,
}

fn compute_flux_forward_stats(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    objective_scale: f64,
) -> FluxForwardStats {
    let (forward, _stats) = path_solver::compute_path_forward_stats(
        network,
        net_infos,
        cfg,
        solve_pool,
        objective_scale,
    );
    FluxForwardStats {
        global_pressure: forward.global_pressure,
        attraction_energy: forward.attraction_energy,
    }
}

fn compute_kirchhoff_gradient_with_system(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    objective_scale: f64,
) -> (Vec<f64>, f64, path_solver::PathStats) {
    let (grad, forward, stats) = path_solver::compute_path_forward_and_gradient(
        network,
        net_infos,
        cfg,
        solve_pool,
        n,
        objective_scale,
    );
    update_pressure_and_flow(network, &forward.global_pressure, solve_pool);
    refresh_net_counts_from_usage(solve_pool, network, &forward.edge_usage);
    (grad, forward.attraction_energy, stats)
}

fn compute_objective_eval_with_system(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    update_flow: bool,
) -> ObjectiveEval {
    let forward = compute_flux_forward_stats(network, net_infos, cfg, solve_pool, 1.0);
    if update_flow {
        update_pressure_and_flow(network, &forward.global_pressure, solve_pool);
    }
    ObjectiveEval {
        energy: forward.attraction_energy,
    }
}

fn compute_kirchhoff_energy_terms_with_system(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    objective_scale: f64,
) -> Vec<NetEnergyTerm> {
    let mut terms = Vec::with_capacity(net_infos.len());
    for net_info in net_infos {
        let (forward, _stats) = path_solver::compute_path_forward_stats(
            network,
            std::slice::from_ref(net_info),
            cfg,
            solve_pool,
            objective_scale,
        );
        terms.push(NetEnergyTerm {
            name: if net_info.debug_name.is_empty() {
                format!("{:?}", net_info.net_id)
            } else {
                net_info.debug_name.clone()
            },
            energy: forward.attraction_energy,
            pins: net_info.pins.len(),
        });
    }
    terms.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    terms
}

#[derive(Clone, Debug)]
struct NetEnergyTerm {
    name: String,
    energy: f64,
    pins: usize,
}

#[derive(Clone, Copy, Debug)]
struct EnergyBreakdown {
    total: f64,
}

#[derive(Clone, Copy, Debug)]
struct ObjectiveEval {
    energy: f64,
}

#[derive(Clone, Copy, Debug)]
struct UtilStats {
    max_count: u32,
    mean_count: f64,
    active_pipes: usize,
    overused_bins: usize,
    overuse_excess: u32,
    max_overuse: u32,
    max_bin_demand: u32,
    mean_bin_demand: f64,
    max_bin_avail: u32,
    mean_bin_avail: f64,
    max_flow: f64,
    mean_flow: f64,
    max_capacity: f64,
    mean_capacity: f64,
}

fn compute_energy_breakdown(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    include_debug_names: bool,
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
) -> EnergyBreakdown {
    let mut eval_network = clone_network(network);
    let net_infos = collect_net_infos(
        ctx,
        net_ids,
        include_debug_names,
        cell_to_idx,
        cell_x,
        cell_y,
        &eval_network,
    );
    let stats = compute_flux_forward_stats(&mut eval_network, &net_infos, cfg, solve_pool, 1.0);
    EnergyBreakdown {
        total: stats.attraction_energy,
    }
}

fn compute_util_stats(network: &PipeNetwork) -> UtilStats {
    let mut max_count: u32 = 0;
    let mut sum_count: f64 = 0.0;
    let mut active_pipes = 0usize;
    let mut max_flow: f64 = 0.0;
    let mut sum_flow: f64 = 0.0;
    let mut max_capacity: f64 = 0.0;
    let mut sum_capacity: f64 = 0.0;
    let mut capacity_pipes = 0usize;
    for pipe in &network.pipes {
        if pipe.capacity <= 0.0 {
            continue;
        }
        capacity_pipes += 1;
        sum_capacity += pipe.capacity;
        max_capacity = max_capacity.max(pipe.capacity);
        let flow = pipe.flow.abs();
        sum_flow += flow;
        max_flow = max_flow.max(flow);
        if pipe.net_count > 0 {
            active_pipes += 1;
            sum_count += pipe.net_count as f64;
            max_count = max_count.max(pipe.net_count);
        }
    }

    let mut overused_bins = 0usize;
    let mut overuse_excess = 0u32;
    let mut max_overuse = 0u32;
    let mut max_bin_demand = 0u32;
    let mut sum_bin_demand = 0.0f64;
    let mut max_bin_avail = 0u32;
    let mut sum_bin_avail = 0.0f64;
    let mut active_bins = 0usize;
    for pipe_ids in &network.node_pipes {
        let mut demand = 0u32;
        let mut avail = 0u32;
        for &pid in pipe_ids {
            let pipe = &network.pipes[pid];
            if pipe.capacity <= 0.0 {
                continue;
            }
            avail = avail.saturating_add(1);
            demand = demand.saturating_add(pipe.net_count);
        }
        if avail == 0 {
            continue;
        }
        active_bins += 1;
        max_bin_demand = max_bin_demand.max(demand);
        max_bin_avail = max_bin_avail.max(avail);
        sum_bin_demand += demand as f64;
        sum_bin_avail += avail as f64;
        if demand > avail {
            overused_bins += 1;
            let excess = demand - avail;
            overuse_excess = overuse_excess.saturating_add(excess);
            max_overuse = max_overuse.max(excess);
        }
    }

    let mean_count = if active_pipes > 0 {
        sum_count / active_pipes as f64
    } else {
        0.0
    };
    let mean_bin_demand = if active_bins > 0 {
        sum_bin_demand / active_bins as f64
    } else {
        0.0
    };
    let mean_bin_avail = if active_bins > 0 {
        sum_bin_avail / active_bins as f64
    } else {
        0.0
    };
    let mean_flow = if capacity_pipes > 0 {
        sum_flow / capacity_pipes as f64
    } else {
        0.0
    };
    let mean_capacity = if capacity_pipes > 0 {
        sum_capacity / capacity_pipes as f64
    } else {
        0.0
    };
    UtilStats {
        max_count,
        mean_count,
        active_pipes,
        overused_bins,
        overuse_excess,
        max_overuse,
        max_bin_demand,
        mean_bin_demand,
        max_bin_avail,
        mean_bin_avail,
        max_flow,
        mean_flow,
        max_capacity,
        mean_capacity,
    }
}

/// Main Beckmann OT placement using fixed-resolution Dijkstra gradients and Adam.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    let mut cfg = cfg.clone();
    cfg.fanout_norm_exp = env::var("NPNR_OT_FANOUT_NORM_EXP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(cfg.fanout_norm_exp)
        .clamp(0.0, 1.0);
    cfg.fanout_weight_sqrt = env::var("NPNR_OT_FANOUT_WEIGHT_SQRT").ok().as_deref() == Some("1")
        || cfg.fanout_weight_sqrt;

    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;
    crate::solver::set_solver_threads(cfg.num_threads);

    let target_scale = 1.0;
    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
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
    let mut network = PipeNetwork::from_context(ctx, target_scale);

    match cfg.init_strategy {
        super::config::InitStrategy::Centroid => {
            common::init_positions_center_drop(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        _ => {
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
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

    let debug_net_energy = env::var("NPNR_OT_DEBUG_NET_ENERGY").ok().as_deref() == Some("1");

    eprintln!(
        "Fixed-resolution Adam: {} cells, scale {:.2}",
        n, target_scale
    );

    let resistance_model = ResistanceModel;
    let mut adam = AdamOptimizer::new(2 * n, 1.0);
    let mut coarse_x_buf = Vec::with_capacity(n);
    let mut coarse_y_buf = Vec::with_capacity(n);
    let mut step = vec![0.0; 2 * n];
    let mut prev_x = vec![0.0; n];
    let mut prev_y = vec![0.0; n];
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_state = ObjectiveState {
        energy: f64::INFINITY,
        line: f64::INFINITY,
        chpwl: f64::INFINITY,
        max_overflow: f64::INFINITY,
        n_overflow: usize::MAX,
        overflow_excess: f64::INFINITY,
    };
    let mut last_snapshot: Option<IterSnapshot> = None;
    let mut total_iters = 0usize;
    let mut finite_step_iters = 0usize;
    let mut rel_progress_ema: f64 = 0.05;
    let mut rel_progress_ref: f64 = 0.05;
    let ema_beta = cfg.energy_progress_ema_beta.clamp(0.0, 0.999);

    for inner_iter in 0..cfg.max_iters {
        let t_iter = std::time::Instant::now();
        let (prev_max_overflow, prev_n_overflow, prev_overflow_excess) = type_aware
            .compute_overflow(
                &cell_buckets,
                &cell_pin_weights,
                &cell_x,
                &cell_y,
                phys_grid_w,
                phys_grid_h,
            );
        if inner_iter % cfg.report_interval == 0 {
            eprintln!(
                "  bin overuse: max={:.1}x bins_over={} excess={:.1}",
                prev_max_overflow, prev_n_overflow, prev_overflow_excess,
            );
        }

        fill_coarse_coords(
            &mut coarse_x_buf,
            &mut coarse_y_buf,
            &cell_x,
            &cell_y,
            network.coarsen as f64,
        );
        let net_infos = collect_net_infos_from_coarse(
            ctx,
            &alive_net_ids,
            debug_net_energy,
            &cell_to_idx,
            &coarse_x_buf,
            &coarse_y_buf,
            &network,
        );
        let t_phase = std::time::Instant::now();
        update_effective_conductance(&solve_pool, &mut network, &resistance_model);
        let update_ms = t_phase.elapsed().as_millis();
        let t_phase = std::time::Instant::now();
        let (grad, energy, path_stats) = compute_kirchhoff_gradient_with_system(
            &mut network,
            &net_infos,
            &cfg,
            &solve_pool,
            n,
            1.0,
        );
        let grad_ms = t_phase.elapsed().as_millis();

        let prev_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let prev_line =
            demand::continuous_line_estimate(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let current_state = ObjectiveState {
            energy,
            line: prev_line,
            chpwl: prev_chpwl,
            max_overflow: prev_max_overflow,
            n_overflow: prev_n_overflow,
            overflow_excess: prev_overflow_excess,
        };
        if inner_iter == 0 {
            best_state = current_state;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
        }

        let prev_breakdown = if debug_net_energy && inner_iter % cfg.report_interval == 0 {
            Some(compute_energy_breakdown(
                ctx,
                &alive_net_ids,
                debug_net_energy,
                &mut network,
                &cell_to_idx,
                &cell_x,
                &cell_y,
                &cfg,
                &solve_pool,
            ))
        } else {
            None
        };
        let prev_terms = if debug_net_energy && inner_iter % cfg.report_interval == 0 {
            Some(compute_kirchhoff_energy_terms_with_system(
                &network,
                &net_infos,
                &cfg,
                &solve_pool,
                1.0,
            ))
        } else {
            None
        };

        prev_x.copy_from_slice(&cell_x);
        prev_y.copy_from_slice(&cell_y);

        let progress_ratio = if rel_progress_ref > 1e-12 {
            (rel_progress_ema / rel_progress_ref).sqrt()
        } else {
            1.0
        };
        let lr_base = (prev_chpwl / n as f64).sqrt().max(0.1);
        let adam_lr = cfg.adam_lr_gain * lr_base * progress_ratio.clamp(0.02, 1.0);
        adam.set_lr(adam_lr);
        adam.step(&grad, &mut step);
        let raw_step_max = max_abs(&step);
        let directional_derivative: f64 = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();

        for i in 0..n {
            cell_x[i] = prev_x[i] + step[i];
            cell_y[i] = prev_y[i] + step[n + i];
        }
        common::clamp_positions(&mut cell_x, &mut cell_y, phys_max_x, phys_max_y);

        fill_coarse_coords(
            &mut coarse_x_buf,
            &mut coarse_y_buf,
            &cell_x,
            &cell_y,
            network.coarsen as f64,
        );
        let trial_net_infos = collect_net_infos_from_coarse(
            ctx,
            &alive_net_ids,
            debug_net_energy,
            &cell_to_idx,
            &coarse_x_buf,
            &coarse_y_buf,
            &network,
        );
        let mut trial_energy = compute_objective_eval_with_system(
            &mut network,
            &trial_net_infos,
            &cfg,
            &solve_pool,
            false,
        )
        .energy;

        // Update relative energy progress EMA.
        let finite_step = trial_energy.is_finite();
        if !finite_step {
            cell_x.copy_from_slice(&prev_x);
            cell_y.copy_from_slice(&prev_y);
            trial_energy = energy;
        }
        let rel_progress = if finite_step && energy.abs() > 1e-12 {
            ((energy - trial_energy) / energy.abs()).max(0.0)
        } else {
            0.0
        };
        rel_progress_ema = ema_beta * rel_progress_ema + (1.0 - ema_beta) * rel_progress;
        if inner_iter == 3.min(cfg.max_iters.saturating_sub(1)) {
            rel_progress_ref = rel_progress_ema.max(1e-12);
        }

        let mut trial_state = current_state;
        let mut trial_breakdown = None;
        let mut trial_terms = None;
        if finite_step {
            trial_state.energy = trial_energy;
            trial_state.line =
                demand::continuous_line_estimate(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
            trial_state.chpwl =
                demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
            (
                trial_state.max_overflow,
                trial_state.n_overflow,
                trial_state.overflow_excess,
            ) = type_aware.compute_overflow(
                &cell_buckets,
                &cell_pin_weights,
                &cell_x,
                &cell_y,
                phys_grid_w,
                phys_grid_h,
            );
            if debug_net_energy && inner_iter % cfg.report_interval == 0 {
                trial_breakdown = Some(compute_energy_breakdown(
                    ctx,
                    &alive_net_ids,
                    debug_net_energy,
                    &mut network,
                    &cell_to_idx,
                    &cell_x,
                    &cell_y,
                    &cfg,
                    &solve_pool,
                ));
                fill_coarse_coords(
                    &mut coarse_x_buf,
                    &mut coarse_y_buf,
                    &cell_x,
                    &cell_y,
                    network.coarsen as f64,
                );
                let trial_net_infos = collect_net_infos_from_coarse(
                    ctx,
                    &alive_net_ids,
                    debug_net_energy,
                    &cell_to_idx,
                    &coarse_x_buf,
                    &coarse_y_buf,
                    &network,
                );
                trial_terms = Some(compute_kirchhoff_energy_terms_with_system(
                    &network,
                    &trial_net_infos,
                    &cfg,
                    &solve_pool,
                    1.0,
                ));
            }
        }

        total_iters = inner_iter + 1;
        if finite_step {
            finite_step_iters += 1;
        }

        let actual_drop = energy - trial_energy;
        let predicted_drop = -directional_derivative;
        let drop_ratio = if predicted_drop.abs() > 1e-12 {
            actual_drop / predicted_drop
        } else {
            0.0
        };
        let rel_drop_ppm = if energy.abs() > 1e-12 {
            actual_drop / energy * 1.0e6
        } else {
            0.0
        };
        let rel_drop = if energy.abs() > 1e-12 {
            actual_drop / energy.abs()
        } else {
            0.0
        };
        last_snapshot = Some(IterSnapshot {
            energy_before: energy,
            energy_after: trial_energy,
            chpwl_before: prev_chpwl,
            chpwl_after: trial_state.chpwl,
            line_before: prev_line,
            line_after: trial_state.line,
            max_overflow_before: prev_max_overflow,
            max_overflow_after: trial_state.max_overflow,
            overflow_excess_before: prev_overflow_excess,
            overflow_excess_after: trial_state.overflow_excess,
            rel_drop,
        });

        let improved = finite_step && trial_state.energy < best_state.energy;
        if improved {
            best_state = trial_state;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
        }

        let iter_ms = t_iter.elapsed().as_millis();
        if inner_iter % cfg.report_interval == 0 {
            let grad_norm = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "  I{:3}: step={} lr={:.3} ema={:.3e} energy={:.3}->{:.3} dE={:.3} pred={:.3} ratio={:.3} relppm={:.1} rawstep={:.3e} chpwl={:.0}->{:.0} line={:.0}->{:.0} bins={:.1}x({}) excess={:.1} metric={:.0} grad={:.3e} {}ms",
                inner_iter,
                if finite_step { "finite" } else { "nonfinite" },
                adam_lr,
                rel_progress_ema,
                energy,
                trial_energy,
                actual_drop,
                predicted_drop,
                drop_ratio,
                rel_drop_ppm,
                raw_step_max,
                prev_chpwl,
                trial_state.chpwl,
                prev_line,
                trial_state.line,
                trial_state.max_overflow,
                trial_state.n_overflow,
                trial_state.overflow_excess,
                trial_state.energy,
                grad_norm,
                iter_ms,
            );
            let avg_pops = if path_stats.total_solves > 0 {
                path_stats.total_heap_pops / path_stats.total_solves
            } else {
                0
            };
            eprintln!(
                "    phases: update={}ms grad={}ms | path: {} solves, {} total_pops, {} avg_pops, {} max_pops, {} fail",
                update_ms,
                grad_ms,
                path_stats.total_solves,
                path_stats.total_heap_pops,
                avg_pops,
                path_stats.max_heap_pops,
                path_stats.failures,
            );
            if let (Some(prev_breakdown), Some(trial_breakdown)) = (prev_breakdown, trial_breakdown)
            {
                let util = compute_util_stats(&network);
                eprintln!(
                    "    energy={:.6}->{:.6}",
                    prev_breakdown.total, trial_breakdown.total
                );
                eprintln!(
                    "    net_count max={} mean={:.3} active_pipes={} bin_demand max={} mean={:.3} bin_avail max={} mean={:.3} overused_bins={} overuse_excess={} max_overuse={} flow max={:.6} mean={:.6} cap max={:.3} mean={:.3}",
                    util.max_count,
                    util.mean_count,
                    util.active_pipes,
                    util.max_bin_demand,
                    util.mean_bin_demand,
                    util.max_bin_avail,
                    util.mean_bin_avail,
                    util.overused_bins,
                    util.overuse_excess,
                    util.max_overuse,
                    util.max_flow,
                    util.mean_flow,
                    util.max_capacity,
                    util.mean_capacity,
                );
            }
            if let (Some(prev_terms), Some(trial_terms)) = (&prev_terms, &trial_terms) {
                let prev_total: f64 = prev_terms.iter().map(|t| t.energy).sum();
                let trial_total: f64 = trial_terms.iter().map(|t| t.energy).sum();
                let top_n = prev_terms.len().min(5);
                let prev_top_sum: f64 = prev_terms.iter().take(top_n).map(|t| t.energy).sum();
                let trial_top_sum: f64 = trial_terms.iter().take(top_n).map(|t| t.energy).sum();
                let trial_by_name: FxHashMap<&str, &NetEnergyTerm> = trial_terms
                    .iter()
                    .map(|term| (term.name.as_str(), term))
                    .collect();
                eprintln!(
                    "    net_energy total={:.3}->{:.3} top{}={:.3}({:.1}%) -> {:.3}({:.1}%) count={}",
                    prev_total,
                    trial_total,
                    top_n,
                    prev_top_sum,
                    100.0 * prev_top_sum / prev_total.max(1e-12),
                    trial_top_sum,
                    100.0 * trial_top_sum / trial_total.max(1e-12),
                    prev_terms.len(),
                );
                for (rank, prev_term) in prev_terms.iter().take(top_n).enumerate() {
                    let trial_energy = trial_by_name
                        .get(prev_term.name.as_str())
                        .map(|term| term.energy)
                        .unwrap_or(prev_term.energy);
                    eprintln!(
                        "      top{} {} pins={} energy={:.3}->{:.3} dE={:.3}",
                        rank + 1,
                        prev_term.name,
                        prev_term.pins,
                        prev_term.energy,
                        trial_energy,
                        prev_term.energy - trial_energy,
                    );
                }
            }
        }
    }

    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);

    {
        let mut coarse_x_buf = Vec::with_capacity(n);
        let mut coarse_y_buf = Vec::with_capacity(n);
        // Diagnostic-only gradient: use whatever `pipe.eff_conductance` was
        // set to on the last accepted iteration. `compute_kirchhoff_gradient_with_system`
        // will refresh `pipe.net_count` from the path solver's fresh edge
        // usage at `best_x/best_y`, but that only matters for any subsequent
        // operation — the exit block runs once for logging.
        fill_coarse_coords(
            &mut coarse_x_buf,
            &mut coarse_y_buf,
            &cell_x,
            &cell_y,
            network.coarsen as f64,
        );
        let exit_net_infos = collect_net_infos_from_coarse(
            ctx,
            &alive_net_ids,
            debug_net_energy,
            &cell_to_idx,
            &coarse_x_buf,
            &coarse_y_buf,
            &network,
        );
        let (exit_grad, exit_energy, _) = compute_kirchhoff_gradient_with_system(
            &mut network,
            &exit_net_infos,
            &cfg,
            &solve_pool,
            n,
            1.0,
        );
        let grad_norm = exit_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grad_inf = exit_grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let rel_grad = if exit_energy.abs() > 1e-12 {
            grad_norm / exit_energy.abs()
        } else {
            0.0
        };
        let mut cell_grad_mags: Vec<f64> = (0..n)
            .map(|i| (exit_grad[i].powi(2) + exit_grad[n + i].powi(2)).sqrt())
            .collect();
        cell_grad_mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mean_cell_grad = cell_grad_mags.iter().sum::<f64>() / n as f64;
        let median_cell_grad = cell_grad_mags[n / 2];
        let top5_mean = cell_grad_mags.iter().take(5).sum::<f64>() / 5.0f64.min(n as f64);
        let descent_potential = grad_norm * grad_norm;
        eprintln!(
            "Optimality: energy={:.3} grad_norm={:.3e} grad_inf={:.3e} rel_grad={:.6} cell_grad: max={:.3e} mean={:.3e} median={:.3e} top5_mean={:.3e} descent_potential={:.3e} accepted={}/{} iters",
            exit_energy,
            grad_norm,
            grad_inf,
            rel_grad,
            cell_grad_mags[0],
            mean_cell_grad,
            median_cell_grad,
            top5_mean,
            descent_potential,
            finite_step_iters,
            total_iters,
        );
    }

    eprintln!(
        "Placement loop done: energy={:.3} chpwl={:.0} line={:.0} bins={:.1}x({}) excess={:.1} reason=max_iters",
        best_state.energy,
        best_state.chpwl,
        best_state.line,
        best_state.max_overflow,
        best_state.n_overflow,
        best_state.overflow_excess,
    );
    if let Some(snapshot) = last_snapshot {
        eprintln!(
            "Summary: energy={:.3}->{:.3} chpwl={:.0}->{:.0} line={:.0}->{:.0} bins={:.1}->{:.1} excess={:.1}->{:.1} rel_drop={:.4}",
            snapshot.energy_before,
            snapshot.energy_after,
            snapshot.chpwl_before,
            snapshot.chpwl_after,
            snapshot.line_before,
            snapshot.line_after,
            snapshot.max_overflow_before,
            snapshot.max_overflow_after,
            snapshot.overflow_excess_before,
            snapshot.overflow_excess_after,
            snapshot.rel_drop,
        );
    }

    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Final chpwl={:.0} ({} total iters)", pre_chpwl, total_iters);

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();
    crate::placer::legalize::legalize(ctx, &idx_to_cell, &phys_x, &phys_y, &cfg.legalization)?;

    report::report_post_legalization(ctx, pre_chpwl);
    report::report_top_net_hpwl(ctx, 10);

    // Dump cell positions + legalized positions to CSV if requested
    report::dump_position_csv_from_env(ctx, "NPNR_OT_DUMP_CSV", &idx_to_cell, &phys_x, &phys_y)
        .expect("dump placement CSV");

    PlacerPipeline::validate(ctx)?;
    Ok(())
}

//! Main Beckmann OT placement algorithm.
//!
//! Progressive refinement with L-BFGS: starts on a coarse grid where cells
//! bypass local minima, then progressively refines. At each resolution level,
//! L-BFGS runs until convergence before moving to the next finer grid.
//! Cell positions carry over as warm start; L-BFGS history resets per level.
//! CHPWL may worsen at finer levels — that's expected and desired for routability.

use crate::common::IdString;
use crate::context::Context;
use crate::metrics::{total_hpwl, total_line_estimate};
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::cg::{cg_scratch_size_nrhs, solve_cg_batched_reuse_with_par};
use crate::solver::optimizer::LbfgsOptimizer;
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOpRef};
use log::info;
use rayon::{prelude::*, ThreadPoolBuilder};
use rustc_hash::FxHashMap;

use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;

use super::config::OptTransPlacerCfg;
use super::demand::NetSolveInfo;
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;
use std::env;

#[derive(Clone, Copy, Debug)]
struct OverflowScore {
    chpwl: f64,
    max_overflow: f64,
    n_overflow: usize,
    overflow_excess: f64,
}

struct KirchhoffSystem {
    laplacian: SparseMatrix,
    pipe_to_offdiag: Vec<usize>,
    diag_shift_scale: f64,
    precond: crate::solver::preconditioner::JacobiPreconditioner,
}

#[derive(Default)]
struct ChunkWorkspace {
    rhs_batch: Vec<f64>,
    pressure_batch: Vec<f64>,
    rhs_touched: Vec<Vec<usize>>,
    scratch: Option<dyn_stack::MemBuffer>,
}

#[derive(Clone, Copy, Debug)]
enum LevelConvergenceReason {
    EnergyDrop { rel_drop: f64, iter: usize },
    Stagnation { stagnant_iters: usize, iter: usize },
    MaxIters,
}

impl LevelConvergenceReason {
    fn summary(self) -> String {
        match self {
            Self::EnergyDrop { rel_drop, iter } => {
                format!("energy_rel_drop={rel_drop:.4} at iter {iter}")
            }
            Self::Stagnation {
                stagnant_iters,
                iter,
            } => format!("stagnated {stagnant_iters} iters at iter {iter}"),
            Self::MaxIters => "max_iters".to_string(),
        }
    }
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

fn fine_level_metric(score: OverflowScore) -> f64 {
    score.max_overflow * 1000.0 + score.overflow_excess * 100.0 + score.n_overflow as f64
}

fn approx_leq(lhs: f64, rhs: f64, abs_eps: f64, rel_eps: f64) -> bool {
    lhs <= rhs + abs_eps.max(rhs.abs() * rel_eps)
}

fn fine_level_better(candidate: OverflowScore, best: OverflowScore) -> bool {
    const MAX_OVERFLOW_ABS_EPS: f64 = 0.25;
    const MAX_OVERFLOW_REL_EPS: f64 = 0.05;
    const EXCESS_ABS_EPS: f64 = 1.0;
    const EXCESS_REL_EPS: f64 = 0.10;
    const TILE_EPS: usize = 4;

    let max_improved = candidate.max_overflow + MAX_OVERFLOW_ABS_EPS
        < best.max_overflow * (1.0 - MAX_OVERFLOW_REL_EPS);
    if max_improved
        && approx_leq(
            candidate.overflow_excess,
            best.overflow_excess,
            EXCESS_ABS_EPS,
            EXCESS_REL_EPS,
        )
    {
        return true;
    }

    let excess_improved =
        candidate.overflow_excess + EXCESS_ABS_EPS < best.overflow_excess * (1.0 - EXCESS_REL_EPS);
    if excess_improved
        && approx_leq(
            candidate.max_overflow,
            best.max_overflow,
            MAX_OVERFLOW_ABS_EPS,
            MAX_OVERFLOW_REL_EPS,
        )
    {
        return true;
    }

    let overflow_close = approx_leq(
        candidate.max_overflow,
        best.max_overflow,
        MAX_OVERFLOW_ABS_EPS,
        MAX_OVERFLOW_REL_EPS,
    ) && approx_leq(
        candidate.overflow_excess,
        best.overflow_excess,
        EXCESS_ABS_EPS,
        EXCESS_REL_EPS,
    );

    if overflow_close {
        if candidate.n_overflow + TILE_EPS < best.n_overflow {
            return true;
        }
        if candidate.n_overflow.abs_diff(best.n_overflow) <= TILE_EPS
            && candidate.chpwl < best.chpwl
        {
            return true;
        }
    }

    false
}

fn update_kirchhoff_system_values(
    solve_pool: &rayon::ThreadPool,
    network: &mut PipeNetwork,
    resistance_model: &ResistanceModel,
    system: &mut KirchhoffSystem,
) {
    let laplacian = &mut system.laplacian;
    laplacian.diag_mut().fill(0.0);
    solve_pool.install(|| {
        network.pipes.par_iter_mut().for_each(|pipe| {
            let r_eff = resistance_model.effective_resistance(pipe, 0.0);
            pipe.eff_conductance = 1.0 / r_eff.max(1e-12);
        });
    });
    for (pipe, &off_idx) in network.pipes.iter().zip(system.pipe_to_offdiag.iter()) {
        let w = pipe.eff_conductance;
        laplacian.diag_mut()[pipe.from] += w;
        laplacian.diag_mut()[pipe.to] += w;
        laplacian.off_diag_mut()[off_idx].2 = -w;
    }
    let epsilon = system.diag_shift_scale * laplacian.diagonal_mean();
    laplacian.add_uniform_diagonal_shift(epsilon);
    system.precond.update(laplacian.diag());
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
        resolution: network.resolution,
        zero_bel_tiles: network.zero_bel_tiles,
        coarsen: network.coarsen,
    }
}

fn build_kirchhoff_system(
    solve_pool: &rayon::ThreadPool,
    network: &mut PipeNetwork,
    resistance_model: &ResistanceModel,
) -> KirchhoffSystem {
    let n_nodes = network.num_nodes();
    let mut laplacian = SparseMatrix::new(n_nodes);
    laplacian.reserve_off_diag(network.num_pipes());
    let mut pipe_to_offdiag = Vec::with_capacity(network.num_pipes());
    for pipe in &network.pipes {
        let (lo, hi) = if pipe.from < pipe.to {
            (pipe.from, pipe.to)
        } else {
            (pipe.to, pipe.from)
        };
        pipe_to_offdiag.push(laplacian.off_diag().len());
        laplacian.add_entry(lo, hi, 0.0);
    }
    let diag_shift_scale = env::var("NPNR_OT_DIAG_SHIFT_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-5);
    let mut system = KirchhoffSystem {
        laplacian,
        pipe_to_offdiag,
        diag_shift_scale,
        precond: crate::solver::preconditioner::JacobiPreconditioner::new(&vec![1.0; n_nodes]),
    };
    update_kirchhoff_system_values(solve_pool, network, resistance_model, &mut system);
    system
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
            pipe.flow = (global_pressure[pipe.from] - global_pressure[pipe.to]) * pipe.eff_conductance;
        });
    });
}

fn solve_net_chunks<State, Init, Process, Reduce>(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    system: &mut KirchhoffSystem,
    init: Init,
    process: Process,
    reduce: Reduce,
) -> State
where
    State: Send,
    Init: Fn() -> State + Sync + Send,
    Process: Fn(&mut State, &NetSolveInfo, &[f64], &[f64]) + Sync + Send,
    Reduce: Fn(State, State) -> State + Sync + Send,
{
    let n_nodes = network.num_nodes();
    let op = SparseMatrixOpRef::from_matrix(&system.laplacian);
    let precond = &system.precond;
    let batch_size = cfg.cg_batch_size.max(1);

    solve_pool.install(|| {
        net_infos
            .par_chunks(batch_size)
            .fold(
                || (init(), ChunkWorkspace::default()),
                |(mut state, mut ws), net_chunk| {
                    let nrhs = net_chunk.len();
                    let batch_len = n_nodes * nrhs;
                    if ws.rhs_batch.len() < batch_len {
                        ws.rhs_batch.resize(batch_len, 0.0);
                    }
                    if ws.pressure_batch.len() < batch_len {
                        ws.pressure_batch.resize(batch_len, 0.0);
                    }
                    if ws.rhs_touched.len() < nrhs {
                        ws.rhs_touched.resize_with(nrhs, Vec::new);
                    } else if ws.rhs_touched.len() > nrhs {
                        ws.rhs_touched.truncate(nrhs);
                    }
                    let rhs_batch = &mut ws.rhs_batch[..batch_len];
                    let pressure_batch = &mut ws.pressure_batch[..batch_len];
                    let scratch_req = cg_scratch_size_nrhs(&op, precond, nrhs);

                    for (col, net_info) in net_chunk.iter().enumerate() {
                        let start = col * n_nodes;
                        demand::build_net_rhs_reuse_into(
                            net_info,
                            cfg,
                            &mut rhs_batch[start..start + n_nodes],
                            &mut ws.rhs_touched[col],
                        );
                    }

                    let scratch = ws
                        .scratch
                        .get_or_insert_with(|| dyn_stack::MemBuffer::new(scratch_req));
                    if scratch.len() < scratch_req.size_bytes() {
                        *scratch = dyn_stack::MemBuffer::new(scratch_req);
                    }
                    let rhs_mat = faer::MatRef::from_column_major_slice(rhs_batch, n_nodes, nrhs);
                    let p_mat = faer::MatMut::from_column_major_slice_mut(
                        pressure_batch,
                        n_nodes,
                        nrhs,
                    );
                    solve_cg_batched_reuse_with_par(
                        &op,
                        precond,
                        rhs_mat,
                        p_mat,
                        cfg.cg_tol,
                        cfg.cg_max_iters,
                        faer::Par::Seq,
                        scratch,
                    );

                    for (col, net_info) in net_chunk.iter().enumerate() {
                        let start = col * n_nodes;
                        process(
                            &mut state,
                            net_info,
                            &rhs_batch[start..start + n_nodes],
                            &pressure_batch[start..start + n_nodes],
                        );
                    }

                    (state, ws)
                },
            )
            .map(|(state, _)| state)
            .reduce(|| init(), reduce)
    })
}

fn compute_kirchhoff_gradient_with_system(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    system: &mut KirchhoffSystem,
) -> (Vec<f64>, f64) {
    let c = network.coarsen as f64;

    struct Accum {
        pressure: Vec<f64>,
        grad: Vec<f64>,
        energy: f64,
    }

    let n_nodes = network.num_nodes();
    let accum = solve_net_chunks(
        network,
        net_infos,
        cfg,
        solve_pool,
        system,
        || Accum {
            pressure: vec![0.0; n_nodes],
            grad: vec![0.0; 2 * n],
            energy: 0.0,
        },
        |local, net_info, rhs, pressure| {
            for (dst, &src) in local.pressure.iter_mut().zip(pressure.iter()) {
                *dst += src;
            }
            let raw_energy = demand::physical_net_energy(net_info, rhs, pressure);
            let (net_energy, grad_weight) = demand::transformed_net_energy(raw_energy);
            let (gx, gy) = local.grad.split_at_mut(n);
            demand::accumulate_physical_energy_gradient(
                net_info,
                pressure,
                network,
                &cfg,
                grad_weight,
                gx,
                gy,
            );
            local.energy += net_energy;
        },
        |mut a, b| {
            for (dst, &src) in a.pressure.iter_mut().zip(b.pressure.iter()) {
                *dst += src;
            }
            for (dst, &src) in a.grad.iter_mut().zip(b.grad.iter()) {
                *dst += src;
            }
            a.energy += b.energy;
            a
        },
    );

    let mut grad = accum.grad;
    let energy = accum.energy;
    let global_pressure = accum.pressure;
    update_pressure_and_flow(network, &global_pressure, solve_pool);
    if c > 1.0 {
        grad.iter_mut().for_each(|g| *g /= c);
    }
    (grad, energy)
}

fn compute_kirchhoff_energy_with_system(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    system: &mut KirchhoffSystem,
    update_flow: bool,
) -> f64 {
    let n_nodes = network.num_nodes();

    struct Accum {
        pressure: Vec<f64>,
        energy: f64,
    }

    let accum = solve_net_chunks(
        network,
        net_infos,
        cfg,
        solve_pool,
        system,
        || Accum {
            pressure: vec![0.0; n_nodes],
            energy: 0.0,
        },
        |local, net_info, rhs, pressure| {
            for (dst, &src) in local.pressure.iter_mut().zip(pressure.iter()) {
                *dst += src;
            }
            let raw_energy = demand::physical_net_energy(net_info, rhs, pressure);
            let (net_energy, _) = demand::transformed_net_energy(raw_energy);
            local.energy += net_energy;
        },
        |mut a, b| {
            for (dst, &src) in a.pressure.iter_mut().zip(b.pressure.iter()) {
                *dst += src;
            }
            a.energy += b.energy;
            a
        },
    );

    if update_flow {
        let global_pressure = accum.pressure;
        update_pressure_and_flow(network, &global_pressure, solve_pool);
    }
    accum.energy
}

fn compute_kirchhoff_energy_terms_with_system(
    network: &PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    system: &mut KirchhoffSystem,
) -> Vec<NetEnergyTerm> {
    let mut terms = solve_net_chunks(
        network,
        net_infos,
        cfg,
        solve_pool,
        system,
        Vec::<NetEnergyTerm>::new,
        |local, net_info, rhs, pressure| {
            let raw_energy = demand::physical_net_energy(net_info, rhs, pressure);
            let (energy, _) = demand::transformed_net_energy(raw_energy);
            local.push(NetEnergyTerm {
                name: net_info.debug_name.clone(),
                energy,
                pins: net_info.pins.len(),
            });
        },
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    );
    terms.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    terms
}

/// Compute the Kirchhoff energy gradient for all nets on the 2D grid.
fn compute_kirchhoff_gradient(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    include_debug_names: bool,
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cfg: &OptTransPlacerCfg,
    resistance_model: &ResistanceModel,
    solve_pool: &rayon::ThreadPool,
    n: usize,
) -> (Vec<f64>, f64) {
    let mut system = build_kirchhoff_system(solve_pool, network, resistance_model);
    let net_infos = collect_net_infos(
        ctx,
        net_ids,
        include_debug_names,
        cell_to_idx,
        cell_x,
        cell_y,
        network,
    );
    compute_kirchhoff_gradient_with_system(network, &net_infos, cfg, solve_pool, n, &mut system)
}

fn compute_kirchhoff_energy(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    include_debug_names: bool,
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cfg: &OptTransPlacerCfg,
    resistance_model: &ResistanceModel,
    solve_pool: &rayon::ThreadPool,
) -> f64 {
    let mut system = build_kirchhoff_system(solve_pool, network, resistance_model);
    let net_infos = collect_net_infos(
        ctx,
        net_ids,
        include_debug_names,
        cell_to_idx,
        cell_x,
        cell_y,
        network,
    );
    compute_kirchhoff_energy_with_system(network, &net_infos, cfg, solve_pool, &mut system, true)
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
    base: f64,
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

fn compute_kirchhoff_energy_terms(
    ctx: &Context,
    net_ids: &[crate::netlist::NetId],
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cfg: &OptTransPlacerCfg,
    resistance_model: &ResistanceModel,
    solve_pool: &rayon::ThreadPool,
) -> Vec<NetEnergyTerm> {
    let mut system = build_kirchhoff_system(solve_pool, network, resistance_model);
    let net_infos = collect_net_infos(ctx, net_ids, true, cell_to_idx, cell_x, cell_y, network);
    compute_kirchhoff_energy_terms_with_system(network, &net_infos, cfg, solve_pool, &mut system)
}

fn base_resistance_model() -> ResistanceModel {
    ResistanceModel {
        congestion_scale: 0.0,
        congestion_power: 2.0,
        timing_weight: 0.0,
    }
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
    resistance_model: &ResistanceModel,
    solve_pool: &rayon::ThreadPool,
) -> EnergyBreakdown {
    let mut total_network = clone_network(network);
    let total = compute_kirchhoff_energy(
        ctx,
        net_ids,
        include_debug_names,
        &mut total_network,
        cell_to_idx,
        cell_x,
        cell_y,
        cfg,
        resistance_model,
        solve_pool,
    );
    let mut base_network = clone_network(network);
    let base = compute_kirchhoff_energy(
        ctx,
        net_ids,
        include_debug_names,
        &mut base_network,
        cell_to_idx,
        cell_x,
        cell_y,
        cfg,
        &base_resistance_model(),
        solve_pool,
    );
    EnergyBreakdown { total, base }
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

/// Main Beckmann OT placement: progressive refinement with L-BFGS.
///
/// Outer loop: coarse -> fine grid (doubling resolution).
/// Inner loop: L-BFGS at each level until convergence.
/// Positions carry over; L-BFGS history resets per level.
/// Always runs all levels -- finer levels improve routability even if CHPWL worsens.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    let mut cfg = cfg.clone();
    cfg.fanout_norm_exp = env::var("NPNR_OT_FANOUT_NORM_EXP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(cfg.fanout_norm_exp)
        .clamp(0.0, 1.0);
    cfg.fanout_weight_sqrt =
        env::var("NPNR_OT_FANOUT_WEIGHT_SQRT").ok().as_deref() == Some("1")
            || cfg.fanout_weight_sqrt;

    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;
    crate::solver::set_solver_threads(cfg.num_threads);

    let target_scale = cfg.subtile_resolution as f64;

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

    // Build level schedule: doubling scale from coarsest to target.
    let mut levels: Vec<f64> = Vec::new();
    {
        let mut s = if env::var("NPNR_OT_EXTRA_COARSE").ok().as_deref() == Some("1") {
            0.025
        } else {
            0.05
        };
        while s < target_scale - 1e-9 {
            levels.push(s);
            s *= 2.0;
        }
        levels.push(target_scale);
    }

    // Use coarsest level for init offset.
    let init_network = PipeNetwork::from_context(ctx, levels[0]);

    match cfg.init_strategy {
        super::config::InitStrategy::Centroid => {
            common::init_positions_center_drop(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        _ => {
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
    }

    solve_pool.install(|| {
        cell_x
            .par_iter_mut()
            .for_each(|v| *v -= init_network.x0 as f64);
        cell_y
            .par_iter_mut()
            .for_each(|v| *v -= init_network.y0 as f64);
    });

    let type_aware = TypeAwarePlacement::build(ctx, init_network.x0, init_network.y0);
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

    let mut resistance_model = ResistanceModel {
        congestion_scale: 0.01,
        congestion_power: 2.0,
        timing_weight: cfg.timing_weight,
    };

    let mut global_iter = 0usize;
    let freeze_resistance = env::var("NPNR_OT_FREEZE_RESISTANCE").ok().as_deref() == Some("1");
    let plain_gd = env::var("NPNR_OT_PLAIN_GD").ok().as_deref() == Some("1");
    let debug_net_energy = env::var("NPNR_OT_DEBUG_NET_ENERGY").ok().as_deref() == Some("1");
    let max_level_index = env::var("NPNR_OT_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let energy_delta_tol = env::var("NPNR_OT_ENERGY_DELTA_TOL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01)
        .max(0.0);
    let gd_step = env::var("NPNR_OT_GD_STEP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-2);

    eprintln!(
        "Progressive L-BFGS: {} cells, {} levels {:?}",
        n,
        levels.len(),
        levels
            .iter()
            .map(|s| format!("{:.2}", s))
            .collect::<Vec<_>>(),
    );

    // --- Outer loop: progressive refinement, always run all levels ---
    for (level_idx, &scale) in levels.iter().enumerate() {
        if let Some(max_level) = max_level_index {
            if level_idx > max_level {
                break;
            }
        }
        let t_level = std::time::Instant::now();
        let mut network = PipeNetwork::from_context(ctx, scale);

        // Trust region scales with coarsen factor.
        let max_disp = cfg.step_scale * (network.coarsen as f64) * 5.0;

        // Fresh L-BFGS for this level.
        let mut lbfgs = LbfgsOptimizer::new(7);
        let mut level_stagnation = 0usize;
        // More patience at fine levels: congestion reduction is slower than CHPWL.
        let level_patience = if level_idx == 0 {
            cfg.stagnation_patience
        } else {
            cfg.stagnation_patience * 3
        };

        // Track best at THIS level (not global).
        let mut level_best_metric = f64::INFINITY;
        let mut level_best_score = OverflowScore {
            chpwl: f64::INFINITY,
            max_overflow: f64::INFINITY,
            n_overflow: usize::MAX,
            overflow_excess: f64::INFINITY,
        };
        let mut level_best_x = cell_x.clone();
        let mut level_best_y = cell_y.clone();
        let mut flat_pos = vec![0.0; 2 * n];
        let mut prev_x = vec![0.0; n];
        let mut prev_y = vec![0.0; n];
        let mut coarse_x_buf = Vec::with_capacity(n);
        let mut coarse_y_buf = Vec::with_capacity(n);
        let mut accepted_system = build_kirchhoff_system(&solve_pool, &mut network, &resistance_model);

        eprintln!(
            "Level {}/{}: scale={:.2} grid={}x{} (c={}, {} nodes) max_disp={:.1}",
            level_idx,
            levels.len() - 1,
            scale,
            network.width,
            network.height,
            network.coarsen,
            network.num_nodes(),
            max_disp,
        );

        let mut convergence_reason = LevelConvergenceReason::MaxIters;
        let mut last_snapshot: Option<IterSnapshot> = None;

        // --- Inner loop: L-BFGS until convergence at this level ---
        for inner_iter in 0..cfg.max_iters {
            let t_iter = std::time::Instant::now();
            demand::update_net_counts(&alive_net_ids, &cell_to_idx, &cell_x, &cell_y, &mut network, ctx);

            let (prev_max_overflow, prev_n_overflow, prev_overflow_excess) =
                type_aware.compute_overflow(
                    &cell_buckets,
                    &cell_pin_weights,
                    &cell_x,
                    &cell_y,
                    phys_grid_w,
                    phys_grid_h,
                );
            if inner_iter % cfg.report_interval == 0 {
                eprintln!(
                    "  overflow: max={:.1}x tiles_over={} excess={:.1}",
                    prev_max_overflow,
                    prev_n_overflow,
                    prev_overflow_excess,
                );
            }

            fill_coarse_coords(
                &mut coarse_x_buf,
                &mut coarse_y_buf,
                &cell_x,
                &cell_y,
                network.coarsen as f64,
            );
            let accepted_net_infos = collect_net_infos_from_coarse(
                ctx,
                &alive_net_ids,
                debug_net_energy,
                &cell_to_idx,
                &coarse_x_buf,
                &coarse_y_buf,
                &network,
            );
            update_kirchhoff_system_values(&solve_pool, &mut network, &resistance_model, &mut accepted_system);
            let (grad, energy) = compute_kirchhoff_gradient_with_system(
                &mut network,
                &accepted_net_infos,
                &cfg,
                &solve_pool,
                n,
                &mut accepted_system,
            );

            // Step proposal: default L-BFGS, optionally plain gradient descent for diagnostics.
            flat_pos[..n].copy_from_slice(&cell_x);
            flat_pos[n..].copy_from_slice(&cell_y);

            let target_disp = if prev_overflow_excess <= 8.0 {
                max_disp * 0.25
            } else if prev_overflow_excess <= 32.0 {
                max_disp * 0.35
            } else {
                max_disp * 0.5
            };

            let step = if plain_gd {
                grad.iter().map(|g| -gd_step * g).collect::<Vec<_>>()
            } else {
                let raw_step = lbfgs.step(&flat_pos, &grad);
                let raw_max = max_abs(&raw_step);
                if raw_max > 0.0 {
                    let scale = (target_disp / raw_max).clamp(1.0, 1024.0);
                    raw_step.into_iter().map(|v| v * scale).collect::<Vec<_>>()
                } else {
                    raw_step
                }
            };
            prev_x.copy_from_slice(&cell_x);
            prev_y.copy_from_slice(&cell_y);
            let prev_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
            let prev_line = demand::continuous_line_estimate(
                ctx,
                &cell_to_idx,
                &cell_x,
                &cell_y,
                &network,
            );
            let prev_breakdown = if debug_net_energy && inner_iter % cfg.report_interval == 0 {
                Some(EnergyBreakdown {
                    total: energy,
                    base: {
                        let mut base_network = clone_network(&network);
                        let mut base_system =
                            build_kirchhoff_system(&solve_pool, &mut base_network, &base_resistance_model());
                        compute_kirchhoff_energy_with_system(
                            &mut base_network,
                            &accepted_net_infos,
                            &cfg,
                            &solve_pool,
                            &mut base_system,
                            false,
                        )
                    },
                })
            } else {
                None
            };
            let prev_terms = if debug_net_energy && inner_iter % cfg.report_interval == 0 {
                Some(compute_kirchhoff_energy_terms_with_system(
                    &network,
                    &accepted_net_infos,
                    &cfg,
                    &solve_pool,
                    &mut accepted_system,
                ))
            } else {
                None
            };
            let mut trial_energy = energy;
            let mut trial_chpwl = prev_chpwl;
            let mut trial_line = prev_line;
            let mut trial_max_overflow = prev_max_overflow;
            let mut trial_n_overflow = prev_n_overflow;
            let mut trial_overflow_excess = prev_overflow_excess;
            let mut trial_terms: Option<Vec<NetEnergyTerm>> = None;
            let mut trial_breakdown: Option<EnergyBreakdown> = None;
            let raw_step_max = max_abs(&step);
            let directional_derivative: f64 =
                grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
            for i in 0..n {
                cell_x[i] += step[i].clamp(-max_disp, max_disp);
                cell_y[i] += step[n + i].clamp(-max_disp, max_disp);
            }
            common::clamp_positions(&mut cell_x, &mut cell_y, phys_max_x, phys_max_y);
            demand::update_net_counts(&alive_net_ids, &cell_to_idx, &cell_x, &cell_y, &mut network, ctx);
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
            update_kirchhoff_system_values(&solve_pool, &mut network, &resistance_model, &mut accepted_system);
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
                    &resistance_model,
                    &solve_pool,
                ));
                trial_terms = Some(compute_kirchhoff_energy_terms_with_system(
                    &network,
                    &trial_net_infos,
                    &cfg,
                    &solve_pool,
                    &mut accepted_system,
                ));
            }
            trial_energy = compute_kirchhoff_energy_with_system(
                &mut network,
                &trial_net_infos,
                &cfg,
                &solve_pool,
                &mut accepted_system,
                false,
            );
            if !trial_energy.is_finite() {
                cell_x.copy_from_slice(&prev_x);
                cell_y.copy_from_slice(&prev_y);
                demand::update_net_counts(&alive_net_ids, &cell_to_idx, &cell_x, &cell_y, &mut network, ctx);
                update_kirchhoff_system_values(&solve_pool, &mut network, &resistance_model, &mut accepted_system);
                trial_energy = energy;
                trial_chpwl = prev_chpwl;
                trial_line = prev_line;
                trial_max_overflow = prev_max_overflow;
                trial_n_overflow = prev_n_overflow;
                trial_overflow_excess = prev_overflow_excess;
                trial_breakdown = prev_breakdown;
                trial_terms = prev_terms.clone();
            } else {
                trial_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
                trial_line = demand::continuous_line_estimate(
                    ctx,
                    &cell_to_idx,
                    &cell_x,
                    &cell_y,
                    &network,
                );
                (trial_max_overflow, trial_n_overflow, trial_overflow_excess) =
                    type_aware.compute_overflow(
                        &cell_buckets,
                        &cell_pin_weights,
                        &cell_x,
                        &cell_y,
                        phys_grid_w,
                        phys_grid_h,
                    );
            }

            let chpwl = trial_chpwl;
            let max_overflow = trial_max_overflow;
            let n_overflow = trial_n_overflow;
            let overflow_excess = trial_overflow_excess;
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
                chpwl_after: chpwl,
                line_before: prev_line,
                line_after: trial_line,
                max_overflow_before: prev_max_overflow,
                max_overflow_after: max_overflow,
                overflow_excess_before: prev_overflow_excess,
                overflow_excess_after: overflow_excess,
                rel_drop,
            });

            let score = OverflowScore {
                chpwl,
                max_overflow,
                n_overflow,
                overflow_excess,
            };

            // Coarse levels optimize HPWL directly. Fine levels use a guarded
            // Pareto comparison so small overflow gains do not justify large HPWL regressions.
            let metric = if level_idx == 0 {
                chpwl
            } else {
                fine_level_metric(score)
            };

            let improved = if level_idx == 0 {
                metric < level_best_metric
            } else {
                fine_level_better(score, level_best_score)
            };

            if improved {
                level_best_metric = metric;
                level_best_score = score;
                level_best_x.copy_from_slice(&cell_x);
                level_best_y.copy_from_slice(&cell_y);
                level_stagnation = 0;
            } else {
                level_stagnation += 1;
            }

            let iter_ms = t_iter.elapsed().as_millis();
            if inner_iter % cfg.report_interval == 0 {
                let grad_norm = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
                eprintln!(
                    "  L{}.{:3}: energy={:.3}->{:.3} dE={:.3} pred={:.3} ratio={:.3} relppm={:.1} rawstep={:.3e} ls={:.3} chpwl={:.0}->{:.0} line={:.0}->{:.0} ovfl={:.1}x({}) excess={:.1} metric={:.0} grad={:.3e} {}ms",
                    level_idx,
                    inner_iter,
                    energy,
                    trial_energy,
                    actual_drop,
                    predicted_drop,
                    drop_ratio,
                    rel_drop_ppm,
                    raw_step_max,
                    1.0,
                    prev_chpwl,
                    chpwl,
                    prev_line,
                    trial_line,
                    max_overflow,
                    n_overflow,
                    overflow_excess,
                    metric,
                    grad_norm,
                    iter_ms,
                );
                if let (Some(prev_breakdown), Some(trial_breakdown)) =
                    (prev_breakdown, trial_breakdown)
                {
                    let util = compute_util_stats(&network);
                    eprintln!(
                        "    energy_split total={:.6}->{:.6} base={:.6}->{:.6} extra={:.6}->{:.6}",
                        prev_breakdown.total,
                        trial_breakdown.total,
                        prev_breakdown.base,
                        trial_breakdown.base,
                        prev_breakdown.total - prev_breakdown.base,
                        trial_breakdown.total - trial_breakdown.base,
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
                    let trial_top_sum: f64 =
                        trial_terms.iter().take(top_n).map(|t| t.energy).sum();
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

            global_iter += 1;

            if actual_drop >= 0.0 && energy.abs() > 1e-12 {
                if rel_drop <= energy_delta_tol {
                    eprintln!(
                        "  Level {} converged at iter {} (energy_rel_drop={:.4} <= tol={:.4})",
                        level_idx, inner_iter, rel_drop, energy_delta_tol,
                    );
                    convergence_reason = LevelConvergenceReason::EnergyDrop { rel_drop, iter: inner_iter };
                    break;
                }
            }

            // Converged at this level: the guarded score stopped improving.
            if level_stagnation >= level_patience && inner_iter >= cfg.stagnation_warmup {
                eprintln!(
                    "  Level {} converged at iter {} (stagnated {} iters, best_metric={:.0})",
                    level_idx, inner_iter, level_stagnation, level_best_metric,
                );
                convergence_reason = LevelConvergenceReason::Stagnation {
                    stagnant_iters: level_stagnation,
                    iter: inner_iter,
                };
                break;
            }
        }

        // Use this level's best as starting point for the next level.
        cell_x.copy_from_slice(&level_best_x);
        cell_y.copy_from_slice(&level_best_y);

        let level_ms = t_level.elapsed().as_millis();
        eprintln!(
            "Level {} done: chpwl={:.0} ovfl={:.1}x({}) excess={:.1} ({:.1}s, reason={})",
            level_idx,
            level_best_score.chpwl,
            level_best_score.max_overflow,
            level_best_score.n_overflow,
            level_best_score.overflow_excess,
            level_ms as f64 / 1000.0,
            convergence_reason.summary(),
        );
        if level_idx == 0 {
            if let Some(snapshot) = last_snapshot {
                eprintln!(
                    "L0 summary: energy={:.3}->{:.3} chpwl={:.0}->{:.0} line={:.0}->{:.0} ovfl={:.1}->{:.1} excess={:.1}->{:.1} rel_drop={:.4} reason={}",
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
                    convergence_reason.summary(),
                );
            }
        }
    }

    // Final legalization at target scale.
    let network = PipeNetwork::from_context(ctx, target_scale);
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Final chpwl={:.0} ({} total iters)", pre_chpwl, global_iter,);

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();
    crate::placer::legalize::legalize(ctx, &idx_to_cell, &phys_x, &phys_y, &cfg.legalization)?;

    let post_hpwl = total_hpwl(ctx);
    let post_line = total_line_estimate(ctx);
    eprintln!(
        "Post-legalization: HPWL={:.0}, line={:.0}, delta={:+.0} ({:+.1}%)",
        post_hpwl,
        post_line,
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

    {
        use crate::metrics::wirelength::net_hpwl;
        let mut net_hpwls: Vec<(f64, usize, crate::netlist::NetId)> = Vec::new();
        for (net_id, net) in ctx.design.iter_alive_nets() {
            if net.driver().is_none() || net.num_users() == 0 {
                continue;
            }
            net_hpwls.push((net_hpwl(ctx, net_id), net.num_users(), net_id));
        }
        net_hpwls.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f64 = net_hpwls.iter().map(|(h, _, _)| h).sum();
        eprintln!("Per-net HPWL: {} nets, total={:.0}", net_hpwls.len(), total);
        for (i, &(h, f, _)) in net_hpwls.iter().take(10).enumerate() {
            eprintln!(
                "  #{}: hpwl={:.0} fanout={} ({:.1}%)",
                i,
                h,
                f,
                h / total * 100.0
            );
        }
        let mut cumul = 0.0;
        for (i, &(h, _, _)) in net_hpwls.iter().enumerate() {
            cumul += h;
            if cumul > total * 0.5 {
                eprintln!(
                    "  50% of HPWL from top {} nets (of {})",
                    i + 1,
                    net_hpwls.len()
                );
                break;
            }
        }
    }

    PlacerPipeline::validate(ctx)?;
    info!("OptTrans placement complete");
    Ok(())
}

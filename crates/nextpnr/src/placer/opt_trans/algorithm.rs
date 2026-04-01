//! Main Beckmann OT placement algorithm.
//!
//! Uses driver-only pressure for the transport cost gradient (no self-interaction).
//! Full Multigrid (FMG) V-cycle: solve at coarse levels first, then refine with
//! residual correction so fine levels only fix short-range errors.
//! Congestion resistance on shared pipes provides natural spreading.

use crate::common::IdString;
use crate::context::Context;
use crate::metrics::{total_hpwl, total_line_estimate};
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::cg::{cg_scratch_size, solve_cg_reuse};
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
use log::info;
use rayon::{prelude::*, ThreadPoolBuilder};
use rustc_hash::FxHashMap;

use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;

use super::config::OptTransPlacerCfg;
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

fn rebuild_laplacian(
    solve_pool: &rayon::ThreadPool,
    network: &mut PipeNetwork,
    resistance_model: &ResistanceModel,
    laplacian: &mut SparseMatrix,
) {
    laplacian.clear();

    solve_pool.install(|| {
        network.pipes.par_iter_mut().for_each(|pipe| {
            let r_eff = resistance_model.effective_resistance(pipe, 0.0);
            let conductance = 1.0 / r_eff.max(1e-12);
            pipe.eff_conductance = conductance;
        });
    });

    laplacian.add_connections_from_iter(
        network
            .pipes
            .iter()
            .map(|pipe| (pipe.from, pipe.to, pipe.eff_conductance)),
    );

    let epsilon = 1e-4 * laplacian.diagonal_mean();
    laplacian.add_uniform_diagonal_shift(epsilon);
}

/// Compute the Kirchhoff energy gradient for all nets at the current network resolution.
///
/// Returns a `Vec<f64>` of length `2*n` where `grad[0..n]` is dE/dx and `grad[n..2*n]` is dE/dy,
/// both in tile-unit coordinates.
fn compute_kirchhoff_gradient(
    ctx: &Context,
    network: &mut PipeNetwork,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    cfg: &OptTransPlacerCfg,
    resistance_model: &ResistanceModel,
    solve_pool: &rayon::ThreadPool,
    n: usize,
) -> Vec<f64> {
    let c = network.coarsen as f64;
    let coarse_x: Vec<f64> = cell_x.iter().map(|&x| x / c).collect();
    let coarse_y: Vec<f64> = cell_y.iter().map(|&y| y / c).collect();

    let n_nodes = network.num_nodes();

    // Build Kirchhoff Laplacian from current pipe state.
    let mut laplacian = SparseMatrix::new(n_nodes);
    laplacian.reserve_off_diag(network.num_pipes());
    rebuild_laplacian(solve_pool, network, resistance_model, &mut laplacian);

    let op = SparseMatrixOp::from_matrix(&mut laplacian);
    let jacobi = crate::solver::preconditioner::JacobiPreconditioner::new(laplacian.diag());
    let precond = &jacobi;
    let scratch_req = cg_scratch_size(&op, precond);

    let net_infos =
        demand::collect_nets_for_solve(ctx, cell_to_idx, &coarse_x, &coarse_y, network);

    struct Accum {
        pressure: Vec<f64>,
        grad: Vec<f64>,
    }

    let accum = solve_pool.install(|| {
        net_infos
            .par_iter()
            .fold(
                || Accum {
                    pressure: vec![0.0; n_nodes],
                    grad: vec![0.0; 2 * n],
                },
                |mut local, net_info| {
                    let driver_rhs = demand::build_driver_rhs(net_info, network, cfg);
                    let mut pressure = vec![0.0; n_nodes];
                    thread_local! {
                        static BUF: std::cell::RefCell<Option<dyn_stack::MemBuffer>> =
                            const { std::cell::RefCell::new(None) };
                    }
                    BUF.with(|cell| {
                        let mut borrow = cell.borrow_mut();
                        let buf = borrow
                            .get_or_insert_with(|| dyn_stack::MemBuffer::new(scratch_req));
                        if buf.len() < scratch_req.size_bytes() {
                            *buf = dyn_stack::MemBuffer::new(scratch_req);
                        }
                        let buf = borrow.as_mut().unwrap();
                        let rhs_mat =
                            faer::MatRef::from_column_major_slice(&driver_rhs, n_nodes, 1);
                        let p_mat = faer::MatMut::from_column_major_slice_mut(
                            &mut pressure,
                            n_nodes,
                            1,
                        );
                        solve_cg_reuse(
                            &op,
                            precond,
                            rhs_mat,
                            p_mat,
                            cfg.cg_tol,
                            cfg.cg_max_iters,
                            buf,
                        );
                    });

                    for (dst, &src) in local.pressure.iter_mut().zip(pressure.iter()) {
                        *dst += src;
                    }
                    let fanout_weight = 1.0;
                    let (gx, gy) = local.grad.split_at_mut(n);
                    demand::accumulate_energy_gradient(
                        net_info, &pressure, network, cfg, fanout_weight, gx, gy,
                    );
                    local
                },
            )
            .reduce(
                || Accum {
                    pressure: vec![0.0; n_nodes],
                    grad: vec![0.0; 2 * n],
                },
                |mut a, b| {
                    for (dst, &src) in a.pressure.iter_mut().zip(b.pressure.iter()) {
                        *dst += src;
                    }
                    for (dst, &src) in a.grad.iter_mut().zip(b.grad.iter()) {
                        *dst += src;
                    }
                    a
                },
            )
    });

    let mut grad = accum.grad;

    // Scale gradient from coarse-grid units to tile units: dE/d(tile) = dE/d(coarse) / coarsen.
    if c > 1.0 {
        grad.iter_mut().for_each(|g| *g /= c);
    }

    // Update node pressures and pipe flows for resistance model.
    let global_pressure = accum.pressure;
    solve_pool.install(|| {
        network
            .nodes
            .par_iter_mut()
            .zip(global_pressure.par_iter())
            .for_each(|(node, &p)| {
                node.pressure = p;
            });

        network.pipes.par_iter_mut().for_each(|pipe| {
            let dp = global_pressure[pipe.from] - global_pressure[pipe.to];
            pipe.flow = dp * pipe.eff_conductance;
        });
    });

    grad
}

/// Main Beckmann OT placement algorithm using Full Multigrid (FMG) V-cycle.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    // Initial network at target resolution (used for grid dimensions).
    let target_scale = cfg.subtile_resolution as f64;
    let network = PipeNetwork::from_context(ctx, target_scale);

    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    let n = idx_to_cell.len();
    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    let solve_pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads.max(1))
        .build()
        .map_err(|e| {
            PlacerError::PlacementFailed(format!(
                "failed to build opt_trans Rayon thread pool: {e}"
            ))
        })?;

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

    // Build per-cell-type valid tile positions and capacities.
    let type_aware = TypeAwarePlacement::build(ctx, network.x0, network.y0);

    // Build per-cell resolved bucket, parallel to cell_x/cell_y.
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();

    // No initial snapping — let cells start at their true init positions.
    // Type-aware snapping happens at the end, before legalization.

    info!(
        "OptTrans placer: {} cells, {}x{} grid ({}x{} subtile), {} pipes",
        n,
        network.width,
        network.height,
        network.subtile_width(),
        network.subtile_height(),
        network.num_pipes(),
    );

    let mut resistance_model = ResistanceModel {
        congestion_exponent: 0.0,
        interference_weight: cfg.interference_weight,
        timing_weight: cfg.timing_weight,
    };

    let mut best_chpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_level_iter = (0usize, 0usize); // (level_idx, smooth_iter)

    // -----------------------------------------------------------------------
    // FMG level schedule: doubling scale from coarsest to target resolution.
    // -----------------------------------------------------------------------
    let mut levels: Vec<f64> = Vec::new();
    {
        let mut s = 0.05;
        while s < target_scale - 1e-9 {
            levels.push(s);
            s *= 2.0;
        }
        levels.push(target_scale);
    }

    eprintln!(
        "FMG schedule: {} levels {:?}",
        levels.len(),
        levels.iter().map(|s| format!("{:.2}", s)).collect::<Vec<_>>()
    );

    // Coarse gradient from previous level for residual correction.
    let mut coarse_grad = vec![0.0f64; 2 * n];
    let mut global_iter = 0usize;

    for (level_idx, &scale) in levels.iter().enumerate() {
        let t_level = std::time::Instant::now();
        let mut network = PipeNetwork::from_context(ctx, scale);
        let c = network.coarsen as f64;
        let max_x = (network.width - 1) as f64;
        let max_y = (network.height - 1) as f64;

        // More smoothing at coarser levels, fewer at finer.
        let n_smooth = if level_idx == 0 {
            60
        } else {
            (40 / level_idx).max(10)
        };

        // Step size proportional to coarsen factor.
        let lr = cfg.step_scale * c;

        eprintln!(
            "Level {}/{}: scale={:.2} ({}x{}, c={}, {} nodes) n_smooth={} lr={:.2}",
            level_idx,
            levels.len() - 1,
            scale,
            network.width,
            network.height,
            network.coarsen,
            network.num_nodes(),
            n_smooth,
            lr,
        );

        for smooth_iter in 0..n_smooth {
            let t_iter = std::time::Instant::now();

            // Compute adaptive congestion exponent from per-type tile overflow.
            {
                let (max_overflow, n_overflow) = type_aware.compute_overflow(
                    &cell_buckets,
                    &cell_x,
                    &cell_y,
                    network.width as usize,
                    network.height as usize,
                );

                resistance_model.congestion_exponent = if max_overflow > 1.0 {
                    (2.0 * max_overflow.ln()).min(4.0)
                } else {
                    0.0
                };

                if smooth_iter % cfg.report_interval == 0 {
                    eprintln!(
                        "  overflow: max={:.1}x tiles_over={} alpha={:.2}",
                        max_overflow, n_overflow, resistance_model.congestion_exponent,
                    );
                }
            }

            // Compute gradient at this level.
            let mut grad = compute_kirchhoff_gradient(
                ctx,
                &mut network,
                &cell_to_idx,
                &cell_x,
                &cell_y,
                cfg,
                &resistance_model,
                &solve_pool,
                n,
            );

            // Residual correction: subtract coarse gradient so fine levels only
            // correct the short-range error not already handled by coarser grids.
            if level_idx > 0 {
                for (g, &cg) in grad.iter_mut().zip(coarse_grad.iter()) {
                    *g -= cg;
                }
            }

            // Gradient descent step (no Adam — multigrid structure controls scale).
            // grad is the energy gradient (positive = uphill), so step downhill.
            solve_pool.install(|| {
                let (gx, gy) = grad.split_at(n);
                cell_x
                    .par_iter_mut()
                    .zip(gx.par_iter())
                    .for_each(|(x, &g)| *x += lr * g);
                cell_y
                    .par_iter_mut()
                    .zip(gy.par_iter())
                    .for_each(|(y, &g)| *y += lr * g);
            });

            // Only clamp to grid bounds during optimization.
            // Type-aware snapping happens once at the end, before legalization.
            // Snapping during optimization fights the gradient (discretizes back).
            common::clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);
            demand::update_net_counts(&cell_to_idx, &cell_x, &cell_y, &mut network, ctx);

            // Track best CHPWL.
            let chpwl =
                demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
            if chpwl < best_chpwl {
                best_chpwl = chpwl;
                best_x.copy_from_slice(&cell_x);
                best_y.copy_from_slice(&cell_y);
                best_level_iter = (level_idx, smooth_iter);
            }

            let iter_ms = t_iter.elapsed().as_millis();
            if smooth_iter % cfg.report_interval == 0 || smooth_iter == n_smooth - 1 {
                let grad_norm = grad.par_iter().map(|v| v * v).sum::<f64>().sqrt();
                eprintln!(
                    "OT L{}.{:2}: chpwl={:.0} grid={}x{} c={} lr={:.2} grad={:.3e} {}ms",
                    level_idx,
                    smooth_iter,
                    chpwl,
                    network.width,
                    network.height,
                    network.coarsen,
                    lr,
                    grad_norm,
                    iter_ms,
                );
            }

            global_iter += 1;
        }

        // Save converged gradient at this level for next level's residual correction.
        coarse_grad = compute_kirchhoff_gradient(
            ctx,
            &mut network,
            &cell_to_idx,
            &cell_x,
            &cell_y,
            cfg,
            &resistance_model,
            &solve_pool,
            n,
        );

        let level_chpwl =
            demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let level_ms = t_level.elapsed().as_millis();
        eprintln!(
            "Level {} done: chpwl={:.0} best={:.0} ({:.1}s)",
            level_idx,
            level_chpwl,
            best_chpwl,
            level_ms as f64 / 1000.0,
        );
    }

    // Restore best positions found across all levels.
    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);

    // Rebuild network at target scale for final CHPWL and legalization.
    let network = PipeNetwork::from_context(ctx, target_scale);
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!(
        "Best at L{}.{}: chpwl={:.0} ({} total iters)",
        best_level_iter.0, best_level_iter.1, pre_chpwl, global_iter,
    );

    // Legalize: snap to valid tiles + assign BELs in one pass.
    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();

    crate::placer::legalize::legalize(
        ctx, &idx_to_cell, &phys_x, &phys_y, &cfg.legalization,
    )?;

    let post_hpwl = total_hpwl(ctx);
    let post_line = total_line_estimate(ctx);
    eprintln!(
        "Post-legalization: HPWL={:.0}, line={:.0}, delta={:+.0} ({:+.1}%)",
        post_hpwl,
        post_line,
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

    // Per-net HPWL breakdown (post-legalization BEL positions).
    {
        use crate::metrics::wirelength::net_hpwl;
        let mut net_hpwls: Vec<(f64, usize, crate::netlist::NetId)> = Vec::new();
        for (net_id, net) in ctx.design.iter_alive_nets() {
            if net.driver().is_none() || net.num_users() == 0 { continue; }
            let h = net_hpwl(ctx, net_id);
            let fanout = net.num_users();
            net_hpwls.push((h, fanout, net_id));
        }
        net_hpwls.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f64 = net_hpwls.iter().map(|(h, _, _)| h).sum();
        eprintln!("Per-net HPWL: {} nets, total={:.0}", net_hpwls.len(), total);
        for (i, &(h, f, _)) in net_hpwls.iter().take(10).enumerate() {
            eprintln!("  #{}: hpwl={:.0} fanout={} ({:.1}%)", i, h, f, h / total * 100.0);
        }
        let mut cumul = 0.0;
        for (i, &(h, _, _)) in net_hpwls.iter().enumerate() {
            cumul += h;
            if cumul > total * 0.5 {
                eprintln!("  50% of HPWL from top {} nets (of {})", i + 1, net_hpwls.len());
                break;
            }
        }
    }

    PlacerPipeline::validate(ctx)?;
    info!("OptTrans placement complete");
    Ok(())
}

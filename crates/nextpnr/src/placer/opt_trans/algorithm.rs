//! Main Beckmann OT placement algorithm.
//!
//! Uses driver-only pressure for the transport cost gradient (no self-interaction).
//! Adam optimizer provides per-cell adaptive learning rates — handles the 1/d
//! gradient acceleration near optima without overshooting.
//! Congestion resistance on shared pipes provides natural spreading.

use crate::common::IdString;
use crate::context::Context;
use crate::metrics::{total_hpwl, total_line_estimate};
use crate::placer::common;
use crate::placer::legalize::Legalizer;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::cg::{cg_scratch_size, solve_cg_reuse};
use crate::solver::optimizer::AdamOptimizer;
use crate::solver::preconditioner::amg::{AmgPreconditioner, AmgStructure};
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
use log::info;
use rayon::{prelude::*, ThreadPoolBuilder};

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

/// Main Beckmann OT placement algorithm.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    let mut network = PipeNetwork::from_context(ctx, cfg.subtile_resolution);

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

    let max_x = (network.width - 1) as f64;
    let max_y = (network.height - 1) as f64;

    // Build per-cell-type valid tile positions and capacities.
    let type_aware = TypeAwarePlacement::build(ctx, network.x0, network.y0);

    // Build per-cell resolved bucket, parallel to cell_x/cell_y.
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();

    // Snap initial positions to valid columns for each cell's type.
    for (x, &bucket) in cell_x.iter_mut().zip(cell_buckets.iter()) {
        *x = type_aware.snap_x(bucket, *x);
    }

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
        congestion_exponent: 0.0, // computed dynamically from overflow
        interference_weight: cfg.interference_weight,
        timing_weight: cfg.timing_weight,
    };

    let mut best_chpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_iter = 0usize;
    let mut laplacian = SparseMatrix::new(network.num_nodes());
    laplacian.reserve_off_diag(network.num_pipes());

    // Build AMG structure once from pipe network topology (sparsity pattern is constant).
    let pipe_pairs: Vec<(usize, usize)> = network
        .pipes
        .iter()
        .map(|p| {
            let (lo, hi) = if p.from < p.to {
                (p.from, p.to)
            } else {
                (p.to, p.from)
            };
            (lo, hi)
        })
        .collect();
    let amg_structure = AmgStructure::new(network.num_nodes(), &pipe_pairs);
    info!("AMG structure: {} levels", amg_structure.num_levels(),);
    let mut amg_precond: Option<AmgPreconditioner> = None;

    // Adam optimizer: lr = step_scale controls tiles/iter for median cell.
    let mut adam = AdamOptimizer::new(2 * n, cfg.step_scale);
    let mut grad = vec![0.0; 2 * n];
    let mut delta = vec![0.0; 2 * n];
    let mut global_pressure = vec![0.0f64; network.num_nodes()];
    let _cg_batch_size = cfg.cg_batch_size.max(1);
    let mut prev_n_nodes = 0usize;
    for iter in 0..cfg.max_iters {
        let n_nodes = network.num_nodes();
        let t_iter = std::time::Instant::now();

        // Warm-start: reuse previous iteration's global_pressure as CG initial guess.
        // If the grid changed size, reinitialize to zeros.
        if n_nodes != prev_n_nodes {
            global_pressure.resize(n_nodes, 0.0);
            global_pressure.fill(0.0);
        }
        prev_n_nodes = n_nodes;

        // a. Compute adaptive congestion exponent from per-type tile overflow.
        {
            let (max_overflow, n_overflow) = type_aware.compute_overflow(
                &cell_buckets,
                &cell_x,
                &cell_y,
                network.width as usize,
                network.height as usize,
            );

            // Exponent: 0 when legal, ramps with log of overflow, capped at 4.
            resistance_model.congestion_exponent = if max_overflow > 1.0 {
                (2.0 * max_overflow.ln()).min(4.0)
            } else {
                0.0
            };

            if iter % cfg.report_interval == 0 {
                eprintln!(
                    "  overflow: max={:.1}x tiles_over={} alpha={:.2}",
                    max_overflow, n_overflow, resistance_model.congestion_exponent,
                );
            }
        }

        // Build Kirchhoff Laplacian from current pipe state.
        rebuild_laplacian(&solve_pool, &mut network, &resistance_model, &mut laplacian);

        let op = SparseMatrixOp::from_matrix(&mut laplacian);

        // AMG preconditioner: reuse structure, update numeric values each iteration.
        match amg_precond.as_mut() {
            Some(amg) => {
                amg.update_values(laplacian.diag(), laplacian.off_diag());
            }
            None => {
                amg_precond = Some(AmgPreconditioner::new(
                    amg_structure.clone(),
                    laplacian.diag(),
                    laplacian.off_diag(),
                ));
            }
        }
        let jacobi = crate::solver::preconditioner::JacobiPreconditioner::new(laplacian.diag());
        let precond = &jacobi;
        let scratch_req = cg_scratch_size(&op, precond);

        let net_infos =
            demand::collect_nets_for_solve(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let n_nets = net_infos.len();
        grad.fill(0.0);

        {
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
                            let driver_rhs = demand::build_driver_rhs(net_info, &network, cfg);
                            let mut pressure = vec![0.0; n_nodes];
                            thread_local! {
                                static BUF: std::cell::RefCell<Option<dyn_stack::MemBuffer>> =
                                    const { std::cell::RefCell::new(None) };
                            }
                            BUF.with(|cell| {
                                let mut borrow = cell.borrow_mut();
                                let buf = borrow
                                    .get_or_insert_with(|| dyn_stack::MemBuffer::new(scratch_req));
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
                            let (gx, gy) = local.grad.split_at_mut(n);
                            demand::accumulate_energy_gradient(
                                net_info, &pressure, &network, cfg, 1.0, gx, gy,
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

            global_pressure.copy_from_slice(&accum.pressure);
            grad.copy_from_slice(&accum.grad);

            // Update node pressures and pipe flows.
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
        }

        // c. Adam step (raw gradient — Adam adapts per-cell from scale).
        adam.step(&grad, &mut delta);

        let (dx, dy) = delta.split_at(n);
        solve_pool.install(|| {
            cell_x
                .par_iter_mut()
                .zip(dx.par_iter())
                .for_each(|(x, &d)| *x += d);
            cell_y
                .par_iter_mut()
                .zip(dy.par_iter())
                .for_each(|(y, &d)| *y += d);
        });

        let disp_rms = ((delta.par_iter().map(|v| v * v).sum::<f64>()) / (2.0 * n as f64)).sqrt();

        // Snap positions to nearest valid column (and optionally row) for each cell's type.
        // Cells optimize within their type's valid tile set, so legalization
        // only needs to resolve within-tile overlaps (much smaller displacement).
        for ((x, y), &bucket) in cell_x
            .iter_mut()
            .zip(cell_y.iter_mut())
            .zip(cell_buckets.iter())
        {
            *x = type_aware.snap_x(bucket, *x);
            let snapped_x = x.round() as i32;
            *y = type_aware.snap_y(bucket, snapped_x, *y);
        }

        common::clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);
        demand::update_net_counts(&cell_to_idx, &cell_x, &cell_y, &mut network, ctx);

        // d. Track best.
        let chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        if chpwl < best_chpwl {
            best_chpwl = chpwl;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
            best_iter = iter;
        }

        let iter_ms = t_iter.elapsed().as_millis();
        if iter % cfg.report_interval == 0 || iter == cfg.max_iters - 1 {
            let grad_norm = grad.par_iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "OT {:3}: chpwl={:.0} rms={:.2} grad={:.3e} {}ms nets={}",
                iter, chpwl, disp_rms, grad_norm, iter_ms, n_nets,
            );
        }

        if disp_rms < 0.01 && iter >= 10 {
            eprintln!("Converged at iter {} (rms={:.3e})", iter, disp_rms);
            break;
        }
    }

    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Best iter {}: chpwl={:.0}", best_iter, pre_chpwl);

    // Legalize: snap to valid tiles + assign BELs in one pass.
    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();

    // Use SnapLegalizer: type-aware snap + spread + BEL assignment in one pass.
    let type_aware = crate::placer::common::TypeAwarePlacement::build(ctx, 0, 0);
    let legalizer = crate::placer::legalize::SnapLegalizer;
    legalizer.legalize(ctx, &idx_to_cell, &phys_x, &phys_y, &type_aware)?;

    let post_hpwl = total_hpwl(ctx);
    let post_line = total_line_estimate(ctx);
    eprintln!(
        "Post-legalization: HPWL={:.0}, line={:.0}, delta={:+.0} ({:+.1}%)",
        post_hpwl,
        post_line,
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

    PlacerPipeline::validate(ctx)?;
    info!("OptTrans placement complete");
    Ok(())
}

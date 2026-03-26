//! Main Beckmann OT placement algorithm.

use log::info;
use crate::context::Context;
use crate::metrics::total_hpwl;
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
use crate::solver::preconditioner::JacobiPreconditioner;
use crate::solver::cg::solve_cg;

use super::anderson::AndersonAccelerator;
use super::config::OptTransPlacerCfg;
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Main Beckmann OT placement algorithm.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    // 1. Build pipe network from chipdb.
    let mut network = PipeNetwork::from_context(ctx);

    // 2. Collect movable cells, init positions.
    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    let n = idx_to_cell.len();

    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);

    // Convert to virtual grid coords.
    for x in cell_x.iter_mut() {
        *x -= network.x0 as f64;
    }
    for y in cell_y.iter_mut() {
        *y -= network.y0 as f64;
    }

    info!(
        "OptTrans placer: {} movable cells, {}x{} grid, {} pipes",
        n,
        network.width,
        network.height,
        network.num_pipes(),
    );

    // 3. Build resistance model.
    let resistance_model = ResistanceModel {
        congestion_exponent: cfg.congestion_exponent,
        interference_weight: cfg.interference_weight,
        timing_weight: cfg.timing_weight,
    };

    // 4. Anderson accelerator.
    let mut anderson = AndersonAccelerator::new(cfg.anderson_depth, n * 2);

    let max_x = (network.width - 1) as f64;
    let max_y = (network.height - 1) as f64;


    // Track best positions.
    let mut best_chpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_iter = 0usize;

    // 5. Main loop.
    for iter in 0..cfg.max_iters {
        // a. Build demand from cell positions.
        let rhs = demand::build_demands(ctx, &cell_to_idx, &cell_x, &cell_y, &network, cfg);

        // b. Update pipe resistances from flows + interference.
        //    Also build the Kirchhoff Laplacian.
        let n_j = network.num_junctions();
        let mut laplacian = SparseMatrix::new(n_j);

        for pipe in &network.pipes {
            let r_eff = resistance_model.effective_resistance(pipe, 0.0);
            let conductance = 1.0 / r_eff.max(1e-12);
            laplacian.add_connection(pipe.from, pipe.to, conductance);
        }

        // Regularize: Kirchhoff Laplacian has a zero eigenvalue (constant
        // vector in nullspace). Add ε·I to all nodes to make it strictly SPD.
        // ε should be small relative to the smallest nonzero eigenvalue but
        // large enough for CG to converge. Use 1e-4 × average diagonal.
        let avg_diag = laplacian.diag().iter().sum::<f64>() / n_j as f64;
        let epsilon = 1e-4 * avg_diag;
        for i in 0..n_j {
            laplacian.add_diagonal(i, epsilon);
        }

        // c. Solve L*P = S via Jacobi-preconditioned CG.
        //    TODO: AMG preconditioning works on regular 2D grids (42 iters vs 375
        //    Jacobi iters on 30x30) but fails on the pipe network's 4-port-per-tile
        //    Schur condensation structure. Needs AMG coarsening adapted for this
        //    graph topology.
        let op = SparseMatrixOp::from_matrix(&mut laplacian);
        let precond = JacobiPreconditioner::new(laplacian.diag());
        let mut pressure = vec![0.0; n_j];
        let cg_result = solve_cg(&op, &precond, &rhs, &mut pressure, cfg.cg_tol, cfg.cg_max_iters);

        // d. Update junction pressures and pipe flows.
        for (i, j) in network.junctions.iter_mut().enumerate() {
            j.pressure = pressure[i];
        }
        for pipe in &mut network.pipes {
            let r_eff = resistance_model.effective_resistance(pipe, 0.0);
            pipe.flow = (pressure[pipe.from] - pressure[pipe.to]) / r_eff.max(1e-12);
        }

        // e. Compute displacement: -grad(P) at cell positions.
        let (dx, dy) = demand::compute_displacement(&cell_x, &cell_y, &network);

        // f. Anderson acceleration.
        //    Current position vector and target (position + displacement).
        let current_pos: Vec<f64> = cell_x.iter().chain(cell_y.iter()).copied().collect();
        let target: Vec<f64> = cell_x
            .iter()
            .zip(&dx)
            .map(|(x, d)| x + d)
            .chain(cell_y.iter().zip(&dy).map(|(y, d)| y + d))
            .collect();
        let residual: Vec<f64> = target
            .iter()
            .zip(&current_pos)
            .map(|(t, c)| t - c)
            .collect();

        let next = anderson.step(&current_pos, &residual);
        cell_x.copy_from_slice(&next[..n]);
        cell_y.copy_from_slice(&next[n..]);

        // g. Clamp to grid.
        common::clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

        // h. Update net counts for interference.
        demand::update_net_counts(&cell_to_idx, &cell_x, &cell_y, &mut network, ctx);

        // i. Track best positions.
        let chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        if chpwl < best_chpwl {
            best_chpwl = chpwl;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
            best_iter = iter;
        }

        // j. Report + convergence.
        let is_report = iter % cfg.report_interval == 0 || iter == cfg.max_iters - 1;
        if is_report {
            let resid = anderson.residual_norm();
            let max_util = network.max_utilization();
            eprintln!(
                "OptTrans iter {}: chpwl={:.0}, residual={:.3e}, max_util={:.2}, cg={}",
                iter, chpwl, resid, max_util, cg_result.iterations,
            );
        }

        // Convergence: residual norm below threshold.
        let rel_resid = anderson.residual_norm() / (n as f64).sqrt().max(1.0);
        if rel_resid < 1e-4 && iter >= 5 {
            eprintln!(
                "OptTrans converged at iteration {} (rel_residual={:.3e})",
                iter, rel_resid,
            );
            break;
        }
    }

    // Restore best positions.
    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);
    eprintln!(
        "Best iteration: {} (chpwl={:.0})",
        best_iter, best_chpwl,
    );

    // 6. Pre-legalization diagnostics.
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Pre-legalization: CHPWL={:.0}", pre_chpwl);

    // 7. CLB snap + legalize.
    crate::placer::legalize::snap_to_clb_grid(ctx, &mut cell_x, &mut cell_y, &idx_to_cell, &network);

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();
    crate::placer::legalize::legalize_ring(ctx, &idx_to_cell, &phys_x, &phys_y)?;

    // 8. Post-legalization HPWL.
    let post_hpwl = total_hpwl(ctx);
    eprintln!(
        "Post-legalization: HPWL={:.0}, delta={:+.0} ({:+.1}%)",
        post_hpwl,
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

    PlacerPipeline::validate(ctx)?;
    info!("OptTrans placement complete");
    Ok(())
}

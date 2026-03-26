//! Main placement loop for ElectroPlace.

use log::info;

use crate::context::Context;
use crate::placer::common::{
    add_wa_wirelength_gradient, clamp_positions, compute_pin_weights,
    NesterovLoopState,
};
use crate::solver::NesterovSolver;
use crate::placer::PlacerError;
use crate::placer::PlacerPipeline;

use super::config::ElectroPlaceCfg;
use super::density;
use crate::legalize::legalize_electro;

const DENSITY_NORM_EPSILON: f64 = 1e-30;
/// Initial density penalty ratio: eta * (wl_norm / den_norm).
const DENSITY_ETA: f64 = 0.1;
/// Multiplicative growth factor for density penalty.
const DENSITY_GROW_MULT: f64 = 1.025;
/// Threshold after which density penalty grows additively.
const DENSITY_GROW_ADDITIVE_THRESHOLD: f64 = 50.0;
/// Additive growth increment.
const DENSITY_GROW_ADDITIVE: f64 = 1.0;
/// Overlap convergence threshold.
const OVERLAP_CONVERGE: f64 = 0.1;
/// Minimum iterations before checking overlap convergence.
const CONVERGENCE_MIN_ITERS: usize = 20;

pub fn place_electro(ctx: &mut Context, cfg: &ElectroPlaceCfg) -> Result<(), PlacerError> {
    let setup = PlacerPipeline::prepare(ctx, cfg.seed)?;
    let cell_to_idx = setup.cell_to_idx;
    let idx_to_cell = setup.idx_to_cell;
    let mut cell_x = setup.cell_x;
    let mut cell_y = setup.cell_y;

    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    let max_x = (w - 1) as f64;
    let max_y = (h - 1) as f64;
    let grid_w = w as usize;
    let grid_h = h as usize;

    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(());
    }

    let mut nesterov_x = NesterovSolver::new(n, cfg.nesterov_step_size);
    let mut nesterov_y = NesterovSolver::new(n, cfg.nesterov_step_size);
    nesterov_x.set_positions(&cell_x);
    nesterov_y.set_positions(&cell_y);

    info!(
        "ElectroPlace: {} movable cells, {}x{} grid, wl_coeff={}",
        n, grid_w, grid_h, cfg.wl_coeff
    );

    let pin_weights = compute_pin_weights(ctx, &cell_to_idx, n);

    let mut density_penalty = 0.0;
    let mut density_initialized = false;
    let mut loop_state = NesterovLoopState::new(&cell_x, &cell_y);

    for iter in 0..cfg.max_iters {
        nesterov_x.look_ahead_into(&mut cell_x);
        nesterov_y.look_ahead_into(&mut cell_y);
        clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

        // WA wirelength gradient (no gamma parameter).
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];
        add_wa_wirelength_gradient(
            ctx, &cell_to_idx, &cell_x, &cell_y, cfg.wl_coeff,
            &mut grad_x, &mut grad_y, None,
        );

        // DCT-based density computation.
        let concrete_density = density::compute_concrete_density(&cell_x, &cell_y, grid_w, grid_h);
        let overlap = density::compute_overlap(&concrete_density);

        let (field_x, field_y) = density::compute_density_field(
            &concrete_density, grid_w, grid_h, cfg.target_density,
        );
        let mut density_grad_x = vec![0.0; n];
        let mut density_grad_y = vec![0.0; n];
        density::compute_density_gradient(
            &cell_x, &cell_y, &field_x, &field_y, grid_w, grid_h,
            &mut density_grad_x, &mut density_grad_y,
        );

        // Initialize or grow density penalty.
        if !density_initialized {
            let wl_norm = crate::placer::common::gradient_norm(&grad_x, &grad_y);
            let den_norm = crate::placer::common::gradient_norm(&density_grad_x, &density_grad_y);
            if den_norm > DENSITY_NORM_EPSILON {
                density_penalty = DENSITY_ETA * wl_norm / den_norm;
                density_initialized = true;
            }
        } else if density_penalty < DENSITY_GROW_ADDITIVE_THRESHOLD {
            density_penalty *= DENSITY_GROW_MULT;
        } else {
            density_penalty += DENSITY_GROW_ADDITIVE;
        }

        // Combine gradients.
        for i in 0..n {
            grad_x[i] += density_penalty * density_grad_x[i];
            grad_y[i] += density_penalty * density_grad_y[i];
        }

        // Simple preconditioner: precond[i] = max(1.0, pin_count[i] + density_penalty).
        for i in 0..n {
            let precond = (pin_weights[i] + density_penalty).max(1.0);
            grad_x[i] /= precond;
            grad_y[i] /= precond;
        }

        // Barzilai-Borwein step size (after first iteration).
        if iter > 0 {
            if let Some(bb_x) = nesterov_x.bb_step_size(&loop_state.prev_grad_x, &grad_x) {
                nesterov_x.set_step_size(bb_x.clamp(1e-4, 1.0));
            }
            if let Some(bb_y) = nesterov_y.bb_step_size(&loop_state.prev_grad_y, &grad_y) {
                nesterov_y.set_step_size(bb_y.clamp(1e-4, 1.0));
            }
        }
        loop_state.save_gradients(&grad_x, &grad_y);

        let step_x = nesterov_x.step(&grad_x);
        let step_y = nesterov_y.step(&grad_y);

        nesterov_x.clamp_positions_range(0.0, max_x);
        nesterov_y.clamp_positions_range(0.0, max_y);

        // Periodic legalization + convergence check.
        if iter % cfg.legalize_interval == 0 || iter == cfg.max_iters - 1 {
            cell_x.copy_from_slice(nesterov_x.positions());
            cell_y.copy_from_slice(nesterov_y.positions());
            clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

            let displacement = legalize_electro(ctx, &idx_to_cell, &cell_x, &cell_y)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
            loop_state.record_metric(hpwl, &cell_x, &cell_y, iter);

            eprintln!(
                "ElectroPlace iter {}: HPWL={:.0}, disp={:.1}, step=({:.4},{:.4}), density_p={:.3}, overlap={:.4}",
                iter, hpwl, displacement, step_x, step_y, density_penalty, overlap,
            );

            // Overlap-based convergence (only after minimum iterations to avoid
            // premature exit on sparse designs where initial overlap is already low).
            if iter >= CONVERGENCE_MIN_ITERS && overlap < OVERLAP_CONVERGE {
                eprintln!("ElectroPlace converged at iteration {} (overlap {:.4} < {})", iter, overlap, OVERLAP_CONVERGE);
                break;
            }
        }
    }

    let _ = legalize_electro(ctx, &idx_to_cell, &loop_state.best_positions_x, &loop_state.best_positions_y)?;

    PlacerPipeline::validate(ctx)?;
    info!("ElectroPlace complete");
    Ok(())
}

//! ElectroPlace: ePlace-style electric-field-based analytical placer.
//!
//! Uses Nesterov accelerated gradient descent with:
//! - Log-Sum-Exp smooth HPWL for wirelength
//! - FFT-based density penalty (electric field analogy)
//! - Reuses shared solver infrastructure from `placer::solver`

pub mod config;
pub mod density;

pub use config::ElectroPlaceCfg;

use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use log::info;
use rustc_hash::FxHashSet;

use super::common::{
    add_wirelength_gradient, apply_preconditioner, clamp_positions, collect_movable_cells,
    compute_pin_weights, gradient_norm, init_positions_from_bels, initial_placement,
    place_cluster_children, unbind_movable_cells, validate_all_placed, with_locked_others,
    NesterovLoopState,
};
use super::solver::NesterovSolver;
use super::PlacerError;

const DIVERGENCE_RATIO: f64 = 1.10;
const DIVERGENCE_MIN_ITERS: usize = 40;
const CONVERGENCE_MIN_ITERS: usize = 50;
const CONVERGENCE_THRESHOLD: f64 = 0.001;
const DENSITY_WEIGHT_SCALE: f64 = 0.1;
const PRECOND_OVERFLOW_THRESHOLD: f64 = 0.3;
const DENSITY_NORM_EPSILON: f64 = 1e-30;

pub struct PlacerElectro;

impl super::Placer for PlacerElectro {
    type Config = ElectroPlaceCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_electro(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_electro(ctx, cfg))
    }
}

pub fn place_electro(ctx: &mut Context, cfg: &ElectroPlaceCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    initial_placement(ctx)?;
    ctx.populate_bel_buckets();

    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    let max_x = (w - 1) as f64;
    let max_y = (h - 1) as f64;
    let grid_w = w as usize;
    let grid_h = h as usize;

    let (cell_to_idx, idx_to_cell) = collect_movable_cells(ctx);
    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);

    let mut nesterov_x = NesterovSolver::new(n, cfg.nesterov_step_size);
    let mut nesterov_y = NesterovSolver::new(n, cfg.nesterov_step_size);
    nesterov_x.set_positions(&cell_x);
    nesterov_y.set_positions(&cell_y);

    let mut gamma = if cfg.gamma_init <= 0.0 {
        (w as f64 / 10.0).max(1.0)
    } else {
        cfg.gamma_init
    };

    info!(
        "ElectroPlace: {} movable cells, {}x{} grid",
        n, grid_w, grid_h
    );

    let pin_weights = compute_pin_weights(ctx, &cell_to_idx, n);
    let mut loop_state = NesterovLoopState::new(&cell_x, &cell_y);
    let mut density_weight = cfg.density_weight;
    let auto_density = cfg.density_weight <= 0.0;

    for iter in 0..cfg.max_iters {
        nesterov_x.look_ahead_into(&mut cell_x);
        nesterov_y.look_ahead_into(&mut cell_y);
        clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];
        add_wirelength_gradient(
            ctx, &cell_to_idx, &cell_x, &cell_y, gamma, &mut grad_x, &mut grad_y,
        );

        let density_map = density::compute_density_map(
            &cell_x, &cell_y, grid_w, grid_h, cfg.target_density,
        );
        let mut density_grad_x = vec![0.0; n];
        let mut density_grad_y = vec![0.0; n];
        density::compute_density_gradient(
            &cell_x, &cell_y, &density_map, grid_w, grid_h,
            &mut density_grad_x, &mut density_grad_y,
        );

        let overflow: f64 = density_map.iter().map(|d| d.max(0.0)).sum();
        let overflow_ratio = (overflow / (n as f64).max(1.0)).clamp(0.0, 1.0);

        // Auto-compute density weight once, then hold. Re-computing each
        // iteration causes decay to 0 as the wirelength gradient shrinks.
        if auto_density && density_weight <= 0.0 {
            let wl_norm = gradient_norm(&grad_x, &grad_y);
            let den_norm = gradient_norm(&density_grad_x, &density_grad_y);
            if den_norm > DENSITY_NORM_EPSILON {
                density_weight = DENSITY_WEIGHT_SCALE * wl_norm / den_norm;
            }
        }

        for i in 0..n {
            grad_x[i] += density_weight * density_grad_x[i];
            grad_y[i] += density_weight * density_grad_y[i];
        }

        apply_preconditioner(
            &mut grad_x, &mut grad_y, &pin_weights,
            loop_state.precond_alpha, density_weight,
        );

        if iter > 0 {
            loop_state.update_step_sizes(&mut nesterov_x, &mut nesterov_y, &grad_x, &grad_y);
        }
        loop_state.save_gradients(&grad_x, &grad_y);

        let step_x = nesterov_x.step(&grad_x);
        let step_y = nesterov_y.step(&grad_y);

        nesterov_x.clamp_positions_range(0.0, max_x);
        nesterov_y.clamp_positions_range(0.0, max_y);

        nesterov_x.adaptive_restart(&grad_x);
        nesterov_y.adaptive_restart(&grad_y);

        loop_state.maybe_increase_precond_alpha(overflow_ratio, PRECOND_OVERFLOW_THRESHOLD, iter);

        if iter % cfg.legalize_interval == 0 || iter == cfg.max_iters - 1 {
            cell_x.copy_from_slice(nesterov_x.positions());
            cell_y.copy_from_slice(nesterov_y.positions());
            clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

            let displacement = legalize_electro(ctx, &idx_to_cell, &cell_x, &cell_y)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
            loop_state.record_hpwl(hpwl, &cell_x, &cell_y);

            eprintln!(
                "ElectroPlace iter {}: HPWL={:.0}, displacement={:.1}, step=({:.4},{:.4}), gamma={:.2}, density_w={:.3}, overflow={:.3}",
                iter, hpwl, displacement, step_x, step_y, gamma, density_weight, overflow_ratio,
            );

            if hpwl > loop_state.best_hpwl * DIVERGENCE_RATIO && iter > DIVERGENCE_MIN_ITERS {
                eprintln!(
                    "ElectroPlace: divergence detected at iter {}, reverting to best (HPWL {:.0} > {:.0})",
                    iter, hpwl, loop_state.best_hpwl,
                );
                nesterov_x.set_positions(&loop_state.best_positions_x);
                nesterov_y.set_positions(&loop_state.best_positions_y);
                break;
            }

            if iter > CONVERGENCE_MIN_ITERS {
                let rel_change = (hpwl - loop_state.best_hpwl).abs() / loop_state.best_hpwl.max(1.0);
                if rel_change < CONVERGENCE_THRESHOLD {
                    eprintln!("ElectroPlace converged at iteration {} (HPWL {:.0})", iter, hpwl);
                    break;
                }
            }
        }

        gamma = (gamma * cfg.gamma_decay).max(cfg.gamma_min);
    }

    let _ = legalize_electro(ctx, &idx_to_cell, &loop_state.best_positions_x, &loop_state.best_positions_y)?;

    validate_all_placed(ctx)?;
    info!("ElectroPlace complete");
    Ok(())
}

fn legalize_electro(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> Result<f64, PlacerError> {
    unbind_movable_cells(ctx, idx_to_cell);

    let mut total_displacement = 0.0;

    for (i, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        let target_x = cell_x[i];
        let target_y = cell_y[i];

        let mut best_bel = None;
        let mut best_cost = f64::INFINITY;

        for bel_view in ctx.bels_for_bucket(cell_type) {
            if !bel_view.is_available() {
                continue;
            }
            let loc = bel_view.loc();
            let dx = loc.x as f64 - target_x;
            let dy = loc.y as f64 - target_y;
            let cost = dx * dx + dy * dy;

            if cost < best_cost {
                best_cost = cost;
                best_bel = Some(bel_view.id());
            }
        }

        let bel = best_bel.ok_or_else(|| {
            PlacerError::NoBelsAvailable(ctx.name_of(cell_type).to_owned())
        })?;

        if !ctx.bind_bel(bel, cell_id, PlaceStrength::Placer) {
            let cell_name = ctx.design.cell(cell_id).name;
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to bind cell {} during ElectroPlace legalization",
                ctx.name_of(cell_name)
            )));
        }

        total_displacement += best_cost;
        place_cluster_children(ctx, cell_id, bel)?;
    }

    Ok(total_displacement)
}

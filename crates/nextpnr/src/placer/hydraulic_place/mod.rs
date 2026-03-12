//! Hydraulic placer: pure Kirchhoff resistive network model.
//!
//! Maps FPGA placement to a pipe network where:
//! - Pressure P = the unified objective (wirelength proxy via pressure drop)
//! - Pipe resistance R = routing difficulty (fewer wires = higher R)
//! - Turbulent resistance R_eff = R * (1 + beta * (Q/C)^2) = congestion penalty
//! - Net demand d = Kirchhoff current injection (+1 driver, -1/fanout sinks)
//! - Resistive energy E = d^T * P = sum(R * Q^2) = the ONE cost function
//!
//! No LSE wirelength. No density penalty. No artificial weights.
//! The pressure gradient IS the unified force.

pub mod config;
pub mod kirchhoff;
pub mod legalize;
pub mod network;
pub mod state;

pub use config::HydraulicPlacerCfg;

use crate::context::Context;
use crate::netlist::CellId;
use log::info;
use rustc_hash::FxHashSet;

use super::common::{initial_placement, validate_all_placed, with_locked_others, NesterovLoopState};
use super::PlacerError;

const DIVERGENCE_RATIO: f64 = 1.05;
const DIVERGENCE_MIN_ITERS: usize = 40;
const CONVERGENCE_MIN_ITERS: usize = 50;
const CONVERGENCE_THRESHOLD: f64 = 0.001;

pub struct PlacerHydraulic;

impl super::Placer for PlacerHydraulic {
    type Config = HydraulicPlacerCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_hydraulic(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_hydraulic(ctx, cfg))
    }
}

pub fn place_hydraulic(ctx: &mut Context, cfg: &HydraulicPlacerCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    initial_placement(ctx)?;
    ctx.populate_bel_buckets();
    let mut state = state::HydraulicState::new(ctx, cfg);

    if state.num_cells() == 0 {
        return Ok(());
    }

    info!(
        "Hydraulic placer: {} movable cells, {}x{} grid, {} pipes",
        state.num_cells(),
        state.network.width,
        state.network.height,
        state.network.num_pipes(),
    );

    let n = state.num_cells();
    let pin_weights = state.compute_kirchhoff_pin_weights(ctx);
    let mut loop_state = NesterovLoopState::new(&state.cell_x, &state.cell_y);

    let max_x = (state.network.width - 1) as f64;
    let max_y = (state.network.height - 1) as f64;

    for iter in 0..cfg.max_outer_iters {
        // Nesterov look-ahead + clamp.
        state.nesterov_x.look_ahead_into(&mut state.cell_x);
        state.nesterov_y.look_ahead_into(&mut state.cell_y);
        state.clamp_positions();

        // Build demand from net connectivity (Kirchhoff current injection).
        let demand = state.compute_net_demands(ctx);

        // Turbulence ramp: 0 -> turbulence_beta over the first half of iterations.
        let ramp = (2.0 * iter as f64 / cfg.max_outer_iters as f64).min(1.0);
        let beta = cfg.turbulence_beta * ramp;

        // Solve Kirchhoff system: L * P = d (pressure IS the objective).
        let result = kirchhoff::kirchhoff_solve(
            &mut state.network,
            &demand,
            beta,
            cfg.newton_iters,
            cfg.cg_max_iters,
            cfg.cg_tolerance,
        );

        // Unified gradient: pressure gradient (no separate wirelength term).
        let (mut grad_x, mut grad_y) = state.compute_pressure_gradient();

        // Simple pin-weight normalization (not WA preconditioner).
        for i in 0..n {
            let precond = pin_weights[i].max(1.0);
            grad_x[i] /= precond;
            grad_y[i] /= precond;
        }

        // Lipschitz step size + Nesterov step.
        if iter > 0 {
            loop_state.update_step_sizes(
                &mut state.nesterov_x,
                &mut state.nesterov_y,
                &grad_x,
                &grad_y,
            );
        }
        loop_state.save_gradients(&grad_x, &grad_y);

        let step_x = state.nesterov_x.step(&grad_x);
        let step_y = state.nesterov_y.step(&grad_y);

        state.nesterov_x.clamp_positions_range(0.0, max_x);
        state.nesterov_y.clamp_positions_range(0.0, max_y);

        state.nesterov_x.adaptive_restart(&grad_x);
        state.nesterov_y.adaptive_restart(&grad_y);

        state.sync_from_nesterov();
        state.clamp_positions();

        // Periodic legalization + convergence check.
        if iter % cfg.legalize_interval == 0 || iter == cfg.max_outer_iters - 1 {
            let displacement = legalize::legalize_hydraulic(ctx, &state)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
            loop_state.record_hpwl(hpwl, &state.cell_x, &state.cell_y);

            eprintln!(
                "Hydraulic iter {}: HPWL={:.0}, disp={:.1}, step=({:.4},{:.4}), energy={:.2e}, beta={:.2}",
                iter, hpwl, displacement, step_x, step_y, result.energy, beta,
            );

            if hpwl > loop_state.best_hpwl * DIVERGENCE_RATIO && iter > DIVERGENCE_MIN_ITERS {
                eprintln!(
                    "Hydraulic: divergence detected at iter {}, reverting to best",
                    iter
                );
                state.cell_x.copy_from_slice(&loop_state.best_positions_x);
                state.cell_y.copy_from_slice(&loop_state.best_positions_y);
                state.sync_to_nesterov();
                break;
            }

            if iter > CONVERGENCE_MIN_ITERS {
                let rel_change =
                    (hpwl - loop_state.best_hpwl).abs() / loop_state.best_hpwl.max(1.0);
                if rel_change < CONVERGENCE_THRESHOLD {
                    eprintln!("Hydraulic placer converged at iteration {}", iter);
                    break;
                }
            }
        }
    }

    state.cell_x.copy_from_slice(&loop_state.best_positions_x);
    state.cell_y.copy_from_slice(&loop_state.best_positions_y);
    legalize::legalize_hydraulic(ctx, &state)?;

    validate_all_placed(ctx)?;

    info!("Hydraulic placement complete");
    Ok(())
}

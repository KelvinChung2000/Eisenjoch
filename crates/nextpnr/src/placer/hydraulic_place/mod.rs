//! Gas hydraulic placer: compressible fluid on a pipe network.
//!
//! Models cells as compressible gas flowing through the FPGA wire network:
//! - Pressure P = κ × density (equation of state)
//! - Pipe resistance R = 1/n_wires (routing difficulty)
//! - Turbulence R_eff = R * (1 + beta * tanh((Q/C)^2)) (congestion)
//! - Net demand = gas injection at driver, extraction at sinks
//! - Pump energy E = Σ |P_driver - P_sinks| (minimise routing cost)
//!
//! IOs are the pumps. Gas equilibrates through pipes. Cells follow pressure gradients.

pub mod config;
pub mod kirchhoff;
pub mod legalize;
pub mod network;
pub mod state;
pub mod timing;

pub use config::HydraulicPlacerCfg;

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{initial_placement, validate_all_placed, with_locked_others};
use super::common::{NesterovLoopState, compute_pin_weights};
use super::solver::NesterovSolver;
use super::PlacerError;

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
    let mut state = state::HydraulicState::new(ctx, cfg.init_strategy);

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

    let pin_weights = compute_pin_weights(ctx, &state.cell_to_idx, n);
    let mut loop_state = NesterovLoopState::new(&state.cell_x, &state.cell_y);
    let mut nesterov_x = NesterovSolver::new(n, cfg.nesterov_step_size);
    let mut nesterov_y = NesterovSolver::new(n, cfg.nesterov_step_size);
    nesterov_x.set_positions(&state.cell_x);
    nesterov_y.set_positions(&state.cell_y);
    let max_x = (state.network.width - 1) as f64;
    let max_y = (state.network.height - 1) as f64;

    let mut criticality: FxHashMap<NetId, f64> = FxHashMap::default();

    // Extract target clock period from timing analyser.
    let target_period = if cfg.timing_weight > 0.0 {
        let mut ta = crate::timing::TimingAnalyser::new();
        ta.setup_and_run(ctx);
        let min_period_ps = ta.clock_constraints().values().copied().min().unwrap_or(10_000);
        min_period_ps as f64
    } else {
        0.0
    };

    for iter in 0..cfg.max_outer_iters {
        let progress = iter as f64 / cfg.max_outer_iters as f64;
        let beta = cfg.turbulence_beta * (1.0 - (-3.0 * progress).exp());

        // 1. Nesterov look-ahead + optional expanding box.
        nesterov_x.look_ahead_into(&mut state.cell_x);
        nesterov_y.look_ahead_into(&mut state.cell_y);
        if cfg.enable_expanding_box {
            state.clamp_to_box(progress);
        } else {
            state.clamp_positions();
        }

        // 2. Initialize combined gradient to zero.
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];

        // 3. Star model force (if enabled).
        if cfg.star_weight > 0.0 {
            let nw: Option<FxHashMap<NetId, f64>> = if cfg.timing_weight > 0.0 {
                Some(criticality.iter().map(|(&k, &v)| (k, 1.0 + cfg.timing_weight * v)).collect())
            } else {
                None
            };
            let (sx, sy) = state.compute_star_force(ctx, cfg.wl_coeff, nw.as_ref());
            for i in 0..n {
                grad_x[i] += cfg.star_weight * sx[i];
                grad_y[i] += cfg.star_weight * sy[i];
            }
        }

        // 4. Gas hydraulic pressure (if enabled).
        let pressure_weight = cfg.pressure_weight_start
            + (cfg.pressure_weight_end - cfg.pressure_weight_start) * progress;
        if cfg.gas_temperature > 0.0 && pressure_weight > 0.0 {
            let demand = state.compute_net_demands(ctx, &criticality, cfg.timing_weight, cfg.io_boost, cfg.pump_gain);
            kirchhoff::gas_hydraulic_solve(
                &mut state.network, &demand, cfg.gas_temperature, beta,
                cfg.newton_iters * 5,
            );
            let (px, py) = state.compute_pressure_gradient(0.0);
            for i in 0..n {
                grad_x[i] += pressure_weight * px[i];
                grad_y[i] += pressure_weight * py[i];
            }
        }

        // 5. Fluid timing → criticality.
        if cfg.timing_weight > 0.0 {
            let timing_result = timing::compute_fluid_timing(ctx, &state, target_period, beta);
            criticality = timing_result.net_criticality;
        }

        // 6. Preconditioner.
        for i in 0..n {
            let w = (pin_weights[i] + pressure_weight).max(1.0);
            grad_x[i] /= w;
            grad_y[i] /= w;
        }

        // 7. Step sizing + Nesterov step.
        if iter > 0 {
            loop_state.update_step_sizes(&mut nesterov_x, &mut nesterov_y, &grad_x, &grad_y);
        }
        loop_state.save_gradients(&grad_x, &grad_y);

        nesterov_x.step(&grad_x);
        nesterov_y.step(&grad_y);
        nesterov_x.clamp_positions_range(0.0, max_x);
        nesterov_y.clamp_positions_range(0.0, max_y);
        state.cell_x.copy_from_slice(nesterov_x.positions());
        state.cell_y.copy_from_slice(nesterov_y.positions());

        // 8. Periodic legalization + convergence.
        if iter % cfg.legalize_interval == 0 || iter == cfg.max_outer_iters - 1 {
            let displacement = legalize::legalize_greedy(ctx, &state)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
            let line_est = crate::metrics::wirelength::total_line_estimate(ctx);
            loop_state.record_metric(hpwl, &state.cell_x, &state.cell_y);
            eprintln!(
                "Hydraulic iter {}: hpwl={:.0}, line_est={:.0}, disp={:.1}, p_w={:.2}, beta={:.2}",
                iter, hpwl, line_est, displacement, pressure_weight, beta,
            );

            if hpwl > loop_state.best_metric * 1.05 && iter > 40 {
                eprintln!("Hydraulic: divergence at iter {}, reverting", iter);
                state.cell_x.copy_from_slice(&loop_state.best_positions_x);
                state.cell_y.copy_from_slice(&loop_state.best_positions_y);
                nesterov_x.set_positions(&state.cell_x);
                nesterov_y.set_positions(&state.cell_y);
                break;
            }
            if iter > 50 {
                let rel = (hpwl - loop_state.best_metric).abs() / loop_state.best_metric.max(1.0);
                if rel < 0.001 {
                    eprintln!("Hydraulic placer converged at iteration {}", iter);
                    break;
                }
            }
        }
    }

    // Restore best and final legalize.
    state.cell_x.copy_from_slice(&loop_state.best_positions_x);
    state.cell_y.copy_from_slice(&loop_state.best_positions_y);
    legalize::legalize_hydraulic(ctx, &state, cfg.lap_max_cells)?;

    validate_all_placed(ctx)?;

    info!("Hydraulic placement complete");
    Ok(())
}

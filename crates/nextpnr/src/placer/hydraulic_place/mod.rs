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

        // 1. Nesterov look-ahead (kinetic momentum).
        nesterov_x.look_ahead_into(&mut state.cell_x);
        nesterov_y.look_ahead_into(&mut state.cell_y);
        if cfg.enable_expanding_box {
            state.clamp_to_box(progress);
        } else {
            state.clamp_positions();
        }

        // 2. Build net demand vector (source at drivers, sink at users).
        //    IO pins act as pressure anchors via io_boost.
        //    Timing-critical nets pump harder via pump_gain.
        let demand = state.compute_net_demands(
            ctx, &criticality, cfg.timing_weight, cfg.io_boost, cfg.pump_gain,
        );

        // 3. Kirchhoff solve: L(R_eff) · P = demand.
        //    Global CG solve propagates pressure through the entire pipe network.
        //    Turbulence R_eff = R * (1 + β·tanh(Q/C)²) penalises congested pipes.
        kirchhoff::kirchhoff_solve(
            &mut state.network,
            &demand,
            beta,
            cfg.newton_iters,
            cfg.cg_max_iters,
            cfg.cg_tolerance,
        );

        // 4. Pressure gradient at cell positions (bilinear interpolation).
        let (px, py) = state.compute_pressure_gradient(0.0);

        // 5. Asymmetric force: driver cells move -∇P, sink cells move +∇P.
        //    Per-cell sign weight from net demand contribution (smooth tanh).
        let demand_sign = state.compute_cell_demand_sign(
            ctx, &criticality, cfg.timing_weight, cfg.io_boost,
        );
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];
        for i in 0..n {
            grad_x[i] = demand_sign[i] * px[i];
            grad_y[i] = demand_sign[i] * py[i];
        }

        // 6. Density spreading force (compressible gas repulsion).
        //    Always repulsive: ALL cells pushed away from overcrowded tiles.
        //    Temperature anneals: starts hot (strong spreading), cools down.
        let temperature = cfg.gas_temperature * (1.0 - 0.5 * progress);
        if temperature > 0.0 {
            let sigma = 2.0 * (1.0 - progress).max(0.5);
            let (dx, dy) = state.compute_gas_gradient(ctx, temperature, sigma);
            for i in 0..n {
                grad_x[i] += dx[i];
                grad_y[i] += dy[i];
            }
        }

        // 7. Fluid timing → criticality update.
        if cfg.timing_weight > 0.0 {
            let timing_result = timing::compute_fluid_timing(ctx, &state, target_period, beta);
            criticality = timing_result.net_criticality;
        }

        // 8. Viscosity preconditioner: critical cells have higher effective mass.
        let viscosity = state.compute_cell_viscosity(ctx, &criticality, cfg.timing_weight);
        for i in 0..n {
            let w = (pin_weights[i] + viscosity[i]).max(1.0);
            grad_x[i] /= w;
            grad_y[i] /= w;
        }

        // 9. Step sizing + Nesterov momentum step.
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

        // 10. Periodic legalization + convergence.
        if iter % cfg.legalize_interval == 0 || iter == cfg.max_outer_iters - 1 {
            let displacement = legalize::legalize_greedy(ctx, &state)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
            let line_est = crate::metrics::wirelength::total_line_estimate(ctx);
            loop_state.record_metric(hpwl, &state.cell_x, &state.cell_y);
            eprintln!(
                "Hydraulic iter {}: hpwl={:.0}, line_est={:.0}, disp={:.1}, beta={:.2}",
                iter, hpwl, line_est, displacement, beta,
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

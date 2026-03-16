//! Hydraulic placer: minimum transport energy placement.
//!
//! Minimizes the free energy F(x) = E_transport(x) + λ · S_density(x) where:
//!
//! - E_transport = ½ P^T S: electrical flow energy through the routing graph.
//!   Kirchhoff system LP = S gives equilibrium potentials P for demand S.
//!   This is a convex relaxation of routing — flow splits across parallel paths,
//!   automatically distributing demand and revealing congestion gradients.
//!
//! - S_density = Σ ρ·ln(ρ) + hard_wall(ρ): density entropy preventing cell overlap.
//!   Cells have physical size 1/n_bels. Hard wall at ρ=1 prevents overcrowding.
//!   Temperature T from cell kinetic energy provides natural annealing.
//!
//! - Congestion: R_eff = R·(1 + β·tanh(Q/C)²) increases resistance on congested
//!   edges, naturally steering demand away from overutilized channels.
//!
//! The gradient ∂F/∂x drives cell motion via Adam optimizer.
//! Kirchhoff gradient (asymmetric: drivers↓, sinks↑) minimizes transport energy.
//! Density gradient (symmetric: always repulsive) minimizes entropy.

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
use super::common::{NesterovLoopState, compute_pin_weights, gradient_norm};
use super::solver::AdamSolver;
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

    // === Auto-scale parameters from design characteristics ===
    // The design is a gas: utilization = compression, grid = container.

    // Count total BELs for true utilization.
    let mut total_bels = 0usize;
    for y in 0..state.network.height {
        for x in 0..state.network.width {
            let tile = ctx.chipdb().tile_by_xy(x, y);
            total_bels += ctx.chipdb().tile_type(tile).bels.len();
        }
    }
    let utilization = n as f64 / (total_bels as f64).max(1.0);
    let grid_diag = ((state.network.width as f64).powi(2)
        + (state.network.height as f64).powi(2)).sqrt();

    // IO boost: amplify boundary pressure for dilute gas.
    // Dilute designs (low util) need stronger IO pull to gather cells.
    let io_boost = cfg.io_boost * (1.0 + 2.0 * (1.0 - utilization).max(0.0));

    // Step size: scale with grid diagonal for proportional movement.
    let step_size = cfg.nesterov_step_size * grid_diag / 50.0;

    // Divergence patience: larger designs need more iterations.
    let diverge_patience = (60.0 + 2.0 * (n as f64).sqrt()) as usize;
    let converge_patience = diverge_patience + 20;

    eprintln!(
        "Auto-scaled: util={:.1}%, io_boost={:.1}, step={:.3}, patience={}",
        utilization * 100.0, io_boost, step_size, diverge_patience,
    );

    let pin_weights = compute_pin_weights(ctx, &state.cell_to_idx, n);
    let mut loop_state = NesterovLoopState::new(&state.cell_x, &state.cell_y);

    // Adam optimizer: per-cell adaptive step sizes, stable for non-smooth objectives.
    // beta1=0.9 (momentum), beta2=0.999 (adaptive scaling).
    let mut adam_x = AdamSolver::new(n, step_size);
    let mut adam_y = AdamSolver::new(n, step_size);
    adam_x.set_positions(&state.cell_x);
    adam_y.set_positions(&state.cell_y);
    let max_x = (state.network.width - 1) as f64;
    let max_y = (state.network.height - 1) as f64;

    let mut criticality: FxHashMap<NetId, f64> = FxHashMap::default();
    let mut density_lambda = 0.0; // Adaptive density penalty multiplier.

    // Cell velocity tracking for thermodynamic temperature.
    let mut prev_x = state.cell_x.clone();
    let mut prev_y = state.cell_y.clone();
    let mut vel_x = vec![0.0; n];
    let mut vel_y = vec![0.0; n];


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
        // Turbulence ramp: caps at 50% of configured beta.
        let beta = (cfg.turbulence_beta * 0.5) * (1.0 - (-3.0 * progress).exp());

        // 1. Optional expanding box constraint.
        if cfg.enable_expanding_box {
            state.clamp_to_box(progress);
        } else {
            state.clamp_positions();
        }

        // 2. Build net demand vector (IO-boosted, timing-amplified).
        let demand = state.compute_net_demands(
            ctx, &criticality, cfg.timing_weight, io_boost, cfg.pump_gain,
        );

        // 3. Kirchhoff solve: L(R_eff) · P = demand.
        kirchhoff::kirchhoff_solve(
            &mut state.network, &demand, beta,
            cfg.newton_iters, cfg.cg_max_iters, cfg.cg_tolerance,
        );

        // 4. Pressure gradient at cell positions.
        let (px, py) = state.compute_pressure_gradient(0.0);

        // 5. Asymmetric force from demand sign.
        //    Adam does x -= alpha * m/sqrt(v), so grad points AWAY from target.
        let demand_sign = state.compute_cell_demand_sign(
            ctx, &criticality, cfg.timing_weight, io_boost,
        );
        let mut grad_x: Vec<f64> = demand_sign.iter().zip(&px).map(|(s, p)| s * p).collect();
        let mut grad_y: Vec<f64> = demand_sign.iter().zip(&py).map(|(s, p)| s * p).collect();

        // 6. Density spreading with thermodynamic temperature.
        //    P = κ · ρ · T where T = base + average(|v|²) at each tile.
        //    Hot regions (cells moving) spread more; cold regions freeze naturally.
        let sigma = 2.0 * (1.0 - progress).max(0.5);
        let velocities = if iter > 0 {
            Some((vel_x.as_slice(), vel_y.as_slice()))
        } else {
            None
        };
        let (dx, dy) = state.compute_gas_gradient(ctx, 1.0, sigma, velocities);

        let wl_norm = gradient_norm(&grad_x, &grad_y);
        let dens_norm = gradient_norm(&dx, &dy);
        if dens_norm > 1e-20 && wl_norm > 1e-20 {
            let ratio = wl_norm / dens_norm;
            if iter == 0 {
                density_lambda = 0.5 * ratio;
            } else {
                // Overflow-driven boost: increase λ when tiles are overcrowded.
                // The more overflow, the stronger the density penalty.
                let (overflow_ratio, _, _) = state.overlap_metrics(ctx);
                let overflow_boost = 1.0 + 5.0 * overflow_ratio; // 1× at 0% overflow, 6× at 100%
                let target = ratio * overflow_boost;
                density_lambda = 0.9 * density_lambda + 0.1 * target;
            }
        }

        for ((gx, gy), (ddx, ddy)) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(dx.iter().zip(&dy)) {
            *gx += density_lambda * ddx;
            *gy += density_lambda * ddy;
        }

        // 7. Fluid timing → criticality.
        if cfg.timing_weight > 0.0 {
            let timing_result = timing::compute_fluid_timing(ctx, &state, target_period, beta);
            criticality = timing_result.net_criticality;
        }

        // 9. Viscosity preconditioner.
        let viscosity = state.compute_cell_viscosity(ctx, &criticality, cfg.timing_weight);
        for ((gx, gy), (&pw, &v)) in grad_x.iter_mut().zip(grad_y.iter_mut())
            .zip(pin_weights.iter().zip(&viscosity))
        {
            let w = (pw + v).max(1.0);
            *gx /= w;
            *gy /= w;
        }

        // 10. Adam step: per-cell adaptive step sizes, built-in momentum.
        adam_x.step(&grad_x);
        adam_y.step(&grad_y);
        adam_x.clamp_positions_range(0.0, max_x);
        adam_y.clamp_positions_range(0.0, max_y);
        state.cell_x.copy_from_slice(adam_x.positions());
        state.cell_y.copy_from_slice(adam_y.positions());

        // 11. Compute cell velocities for thermodynamic temperature.
        for i in 0..n {
            vel_x[i] = state.cell_x[i] - prev_x[i];
            vel_y[i] = state.cell_y[i] - prev_y[i];
        }
        prev_x.copy_from_slice(&state.cell_x);
        prev_y.copy_from_slice(&state.cell_y);

        // 10. Track continuous HPWL for best-position selection.
        let chpwl_now = state.continuous_hpwl(ctx);
        loop_state.record_metric(chpwl_now, &state.cell_x, &state.cell_y);

        // 11. Reporting + convergence.
        let is_report_iter = iter % cfg.report_interval == 0 || iter == cfg.max_outer_iters - 1;
        if is_report_iter {
            let (overflow_ratio, max_rho, _) = state.overlap_metrics(ctx);
            eprintln!(
                "Hydraulic iter {}: chpwl={:.0}, λ={:.2}, overflow={:.1}%, maxρ={:.1}, β={:.2}",
                iter, chpwl_now, density_lambda, overflow_ratio * 100.0, max_rho, beta,
            );

            if chpwl_now > loop_state.best_metric * 1.20 && iter > diverge_patience {
                eprintln!("Hydraulic: divergence at iter {}, reverting", iter);
                state.cell_x.copy_from_slice(&loop_state.best_positions_x);
                state.cell_y.copy_from_slice(&loop_state.best_positions_y);
                adam_x.set_positions(&state.cell_x);
                adam_y.set_positions(&state.cell_y);
                break;
            }
            if iter > converge_patience {
                let rel = (chpwl_now - loop_state.best_metric).abs() / loop_state.best_metric.max(1.0);
                if rel < 0.005 {
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

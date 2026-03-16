//! Optimal transport placer: minimum transport energy placement.
//!
//! Minimizes the free energy F(x) = E_transport(x) + λ · S_density(x) where:
//!
//! - E_transport = ½ P^T S: electrical flow energy through the routing graph.
//!   Kirchhoff system LP = S gives equilibrium potentials P for demand S.
//!   This is a convex relaxation of routing — flow splits across parallel paths,
//!   automatically distributing demand and revealing congestion gradients.
//!
//! - S_density = Σ ρ·ln(ρ): density entropy preventing cell overlap.
//!   Per-tile Augmented Lagrangian multipliers enforce capacity constraints:
//!   λ[tile] grows at overcrowded tiles, stays zero at tiles below capacity.
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

pub use config::OptTransPlacerCfg;

use crate::context::Context;
use crate::netlist::{CellId, NetId};
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{initial_placement, validate_all_placed, with_locked_others};
use super::common::{NesterovLoopState, compute_pin_weights};
use super::solver::AdamSolver;
use super::PlacerError;

pub struct PlacerOptTrans;

impl super::Placer for PlacerOptTrans {
    type Config = OptTransPlacerCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_opt_trans(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_opt_trans(ctx, cfg))
    }
}

pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    initial_placement(ctx)?;
    ctx.populate_bel_buckets();
    let mut state = state::OptTransState::new(ctx, cfg.init_strategy);

    if state.num_cells() == 0 {
        return Ok(());
    }

    info!(
        "OptTrans placer: {} movable cells, {}x{} grid, {} pipes",
        state.num_cells(),
        state.network.width,
        state.network.height,
        state.network.num_pipes(),
    );

    let n = state.num_cells();

    // === Auto-scale parameters from design characteristics ===

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

    // IO boost: amplify boundary pressure for low utilization designs.
    // Low utilization designs need stronger IO pull to gather cells.
    let io_boost = cfg.io_boost * (1.0 + 2.0 * (1.0 - utilization).max(0.0));

    // Step size: scale with grid diagonal for proportional movement.
    let step_size = cfg.nesterov_step_size * grid_diag / 50.0;

    // Convergence patience: larger designs need more iterations before checking.
    let converge_patience = (80.0 + 2.0 * (n as f64).sqrt()) as usize;

    eprintln!(
        "Auto-scaled: util={:.1}%, io_boost={:.1}, step={:.3}, patience={}",
        utilization * 100.0, io_boost, step_size, converge_patience,
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

    // Per-tile Augmented Lagrangian multipliers for density enforcement.
    let w = state.network.width as usize;
    let h = state.network.height as usize;
    let mut tile_lambda = vec![0.0; w * h];
    let al_alpha = 0.1; // AL dual update step size.

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
        // Nonlinear resistance ramp: caps at 50% of configured beta.
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

        // 6. Density spreading via per-tile Augmented Lagrangian multipliers.
        //    P[tile] = λ[tile] * ρ[tile] — overcrowded tiles get increasing λ.
        let sigma = 2.0 * (1.0 - progress).max(0.5);
        let (dx, dy) = state.compute_density_gradient(ctx, sigma, &tile_lambda);

        let overlap = if iter > 0 { Some(state.overlap_metrics(ctx)) } else { None };

        for ((gx, gy), (ddx, ddy)) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(dx.iter().zip(&dy)) {
            *gx += ddx;
            *gy += ddy;
        }

        // 7. Timing feedback → criticality.
        if cfg.timing_weight > 0.0 {
            let timing_result = timing::compute_fluid_timing(ctx, &state, target_period, beta);
            criticality = timing_result.net_criticality;
        }

        // 8. Viscosity preconditioner.
        let viscosity = state.compute_cell_viscosity(ctx, &criticality, cfg.timing_weight);
        for ((gx, gy), (&pw, &v)) in grad_x.iter_mut().zip(grad_y.iter_mut())
            .zip(pin_weights.iter().zip(&viscosity))
        {
            let w = (pw + v).max(1.0);
            *gx /= w;
            *gy /= w;
        }

        // 9. Adam step: per-cell adaptive step sizes, built-in momentum.
        adam_x.step(&grad_x);
        adam_y.step(&grad_y);
        adam_x.clamp_positions_range(0.0, max_x);
        adam_y.clamp_positions_range(0.0, max_y);
        state.cell_x.copy_from_slice(adam_x.positions());
        state.cell_y.copy_from_slice(adam_y.positions());

        // 10. Augmented Lagrangian dual update: increase λ at overcrowded tiles.
        //     Also periodically reset Adam moments so cells can respond to the
        //     growing density multipliers (Adam adapts to early gradient magnitudes
        //     and ignores later changes without a reset).
        {
            let density = state.build_density_field(ctx);
            let num_tiles = w * h;
            for i in 0..num_tiles {
                tile_lambda[i] = (tile_lambda[i] + al_alpha * (density[i] - 1.0)).max(0.0);
            }
            // Reset Adam every 50 iterations so cells re-adapt to new λ landscape.
            if iter > 0 && iter % 50 == 0 {
                adam_x.reset_moments();
                adam_y.reset_moments();
            }
        }

        // 11. Track best position using the same objective the optimizer minimizes:
        //    score = chpwl + λ_avg · E_density where E_density = Σ max(0,ρ-1)².
        //    This selects the position that best balances wirelength and legalizability.
        let chpwl_now = state.continuous_hpwl(ctx);
        let e_density = state.density_energy(ctx);
        let avg_lambda = tile_lambda.iter().sum::<f64>() / (w * h) as f64;
        let score = chpwl_now + avg_lambda.max(1.0) * e_density;
        loop_state.record_metric(score, &state.cell_x, &state.cell_y);

        // 12. Reporting + convergence.
        let is_report_iter = iter % cfg.report_interval == 0 || iter == cfg.max_outer_iters - 1;
        if is_report_iter {
            let (overflow_ratio, max_rho, _) = overlap
                .unwrap_or_else(|| state.overlap_metrics(ctx));
            let max_lambda = tile_lambda.iter().copied().fold(0.0_f64, f64::max);
            eprintln!(
                "OptTrans iter {}: chpwl={:.0}, maxλ={:.2}, overflow={:.1}%, maxρ={:.1}, β={:.2}",
                iter, chpwl_now, max_lambda, overflow_ratio * 100.0, max_rho, beta,
            );

            if iter > converge_patience {
                let rel = (chpwl_now - loop_state.best_metric).abs() / loop_state.best_metric.max(1.0);
                if rel < 0.005 {
                    eprintln!("OptTrans placer converged at iteration {}", iter);
                    break;
                }
            }
        }
    }

    // Restore best and final legalize.
    state.cell_x.copy_from_slice(&loop_state.best_positions_x);
    state.cell_y.copy_from_slice(&loop_state.best_positions_y);
    legalize::legalize_opt_trans(ctx, &state, cfg.lap_max_cells)?;

    validate_all_placed(ctx)?;

    info!("OptTrans placement complete");
    Ok(())
}

//! Optimal transport placer: minimum transport energy placement.
//!
//! Minimizes score = CHPWL(x) + λ_avg · E_density(x) where:
//!
//! - CHPWL: continuous half-perimeter wirelength from cell positions.
//!   Kirchhoff system LP = S gives equilibrium potentials P for demand S,
//!   providing pressure gradients that drive cells toward shorter routes.
//!   This is a convex relaxation of routing — flow splits across parallel paths,
//!   automatically distributing demand and revealing congestion gradients.
//!
//! - E_density = Σ (ρ-1)²: symmetric quadratic capacity deviation penalty.
//!   Per-tile Augmented Lagrangian multipliers enforce capacity constraints:
//!   λ[tile] grows at overcrowded tiles via fixed α=0.1 dual step.
//!
//! - Congestion: R_eff = R·(1 + β·(Q/C)²)·(1 + density_penalty) couples both
//!   flow-based turbulence and density overflow into pipe resistance, steering
//!   the Kirchhoff solver around congested regions (Benamou-Brenier coupling).
//!
//! The gradient ∂F/∂x drives cell motion via Adam optimizer.
//! Kirchhoff gradient (asymmetric: drivers↓, sinks↑) minimizes transport energy.
//! Density gradient (symmetric: always repulsive) spreads overcrowded cells.

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
use super::solver::VelocityFieldSolver;
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

    // Use cached BEL capacity for utilization.
    // Only count tiles with real BELs (cap < 1000, since empty tiles use 1e6 sentinel).
    let total_bels: f64 = state.tile_bel_cap.iter().filter(|&&c| c < 1000.0).sum();
    let utilization = n as f64 / total_bels.max(1.0);
    let grid_diag = ((state.network.width as f64).powi(2)
        + (state.network.height as f64).powi(2)).sqrt();

    // IO boost: amplify boundary pressure for low utilization designs.
    // Low utilization designs need stronger IO pull to gather cells.
    let io_boost = cfg.io_boost * (1.0 + 2.0 * (1.0 - utilization).max(0.0));

    // Step size: scale with grid diagonal for proportional movement.
    let step_size = cfg.nesterov_step_size * grid_diag / 50.0;

    // Convergence patience: larger designs need more iterations.
    // Scale with both cell count AND utilization — sparse grids converge fast.
    let base_patience = 20.0 + 2.0 * (n as f64).sqrt();
    let util_scale = (utilization * 100.0).max(1.0).min(100.0) / 10.0; // 0.1..10
    let converge_patience = (base_patience * util_scale.max(1.0)) as usize;

    eprintln!(
        "Auto-scaled: util={:.1}%, io_boost={:.1}, step={:.3}, patience={}",
        utilization * 100.0, io_boost, step_size, converge_patience,
    );

    let pin_weights = compute_pin_weights(ctx, &state.cell_to_idx, n);
    let mut loop_state = NesterovLoopState::new(&state.cell_x, &state.cell_y);

    // Velocity field solver: damped momentum without per-coordinate rescaling.
    // Preserves gradient geometry so the Helmholtz de-clustering force
    // can act as a truly decoupled operator-split correction.
    let mut vel_x = VelocityFieldSolver::new(n, step_size);
    let mut vel_y = VelocityFieldSolver::new(n, step_size);
    vel_x.set_alpha(cfg.velocity_alpha);
    vel_y.set_alpha(cfg.velocity_alpha);
    vel_x.set_eta_helmholtz(step_size * cfg.helmholtz_eta_ratio);
    vel_y.set_eta_helmholtz(step_size * cfg.helmholtz_eta_ratio);
    vel_x.set_positions(&state.cell_x);
    vel_y.set_positions(&state.cell_y);
    let max_x = (state.network.width - 1) as f64;
    let max_y = (state.network.height - 1) as f64;

    let mut criticality: FxHashMap<NetId, f64> = FxHashMap::default();

    // Per-tile Augmented Lagrangian multipliers for density enforcement.
    let w = state.network.width as usize;
    let h = state.network.height as usize;
    let mut tile_lambda = vec![0.0; w * h];
    let al_alpha = 0.1; // AL dual update step size.
    let mut helmholtz_cache = vec![0.0; w * h]; // warm-start for Helmholtz CG
    let mut density_overflow_buf = vec![0.0; w * h]; // reusable buffer

    // Pre-compute Helmholtz operator structure (geometry is constant).
    // Only the diagonal kappa_sq term changes per iteration.
    let helm_off_diag = helmholtz_off_diag(w, h);
    let helm_neighbor_count = helmholtz_neighbor_count(w, h);
    let mut helm_diag = vec![0.0_f64; w * h];
    let mut helm_rhs = vec![0.0; w * h]; // reusable RHS buffer

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

        // 2b. Build density field (reused for Kirchhoff coupling, gradient, and AL update).
        //     Helmholtz-smoothed overflow for resistance coupling: the Kirchhoff
        //     solver sees a non-local congestion field.
        let pre_density = state.build_density_field(ctx);
        let density_overflow: &[f64] = if iter > 0 {
            // Update Helmholtz diagonal for current kappa (off-diag is cached).
            let screen_len = grid_diag / 3.0 * (1.0 - progress) + 5.0 * progress;
            let kappa = 1.0 / screen_len;
            let kappa_sq = kappa * kappa;
            for i in 0..w * h {
                helm_diag[i] = helm_neighbor_count[i] + kappa_sq;
                helm_rhs[i] = kappa * (pre_density[i] - 1.0).max(0.0);
            }
            // Solve (-Δ + κ²)φ = κ·overflow → smoothed congestion field.
            // Relaxed tolerance (1e-2) — this is a smoothing operation.
            crate::placer::solver::cg::preconditioned_conjugate_gradient(
                &helm_diag, &helm_off_diag, &helm_rhs,
                &mut helmholtz_cache,
                1e-2, 20,
            );
            // Amplify 10x into reusable buffer.
            for i in 0..w * h {
                density_overflow_buf[i] = 10.0 * helmholtz_cache[i].max(0.0);
            }
            &density_overflow_buf
        } else {
            &[]
        };

        // 3. Kirchhoff solve: L(R_eff) · P = demand.
        //    Density overflow increases pipe resistance in congested regions.
        kirchhoff::kirchhoff_solve(
            &mut state.network, &demand, beta,
            cfg.newton_iters, cfg.cg_max_iters, cfg.cg_tolerance,
            &density_overflow,
        );

        // 4. Pressure gradient at cell positions.
        let (px, py) = state.compute_pressure_gradient(0.0);

        // 5. Asymmetric force from demand sign.
        //    Sources (positive demand) move down-gradient (toward sinks).
        //    Sinks (negative demand) move up-gradient (toward sources).
        //    This minimizes transport energy: E = Σ |P_driver - P_sink|.
        let demand_sign = state.compute_cell_demand_sign(
            ctx, &criticality, cfg.timing_weight, io_boost,
        );
        let mut grad_x: Vec<f64> = demand_sign.iter().zip(&px).map(|(s, p)| s * p).collect();
        let mut grad_y: Vec<f64> = demand_sign.iter().zip(&py).map(|(s, p)| s * p).collect();

        // 6. Density spreading via per-tile Augmented Lagrangian multipliers.
        //    P[tile] = λ[tile] * ρ[tile] — overcrowded tiles get increasing λ.
        let sigma = 2.0 * (1.0 - progress).max(0.5);
        let (dx, dy) = state.density_gradient_from_field(&pre_density, sigma, &tile_lambda);

        let overlap = if iter > 0 {
            Some(state::OptTransState::overlap_metrics_from_field(&pre_density))
        } else {
            None
        };

        // 7. Steiner anchor: pull pins toward net routing centers.
        //    Keeps nets spatially compact, approximating Steiner junction topology.
        let anchor_weight = 0.02;
        let (ax, ay) = state.compute_anchor_gradient(ctx, anchor_weight);
        for ((gx, gy), (aax, aay)) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(ax.iter().zip(&ay)) {
            *gx += aax;
            *gy += aay;
        }

        // 8. Timing feedback → criticality.
        if cfg.timing_weight > 0.0 {
            let timing_result = timing::compute_fluid_timing(ctx, &state, target_period, beta);
            criticality = timing_result.net_criticality;
        }

        // 8. Viscosity preconditioner: scale Kirchhoff+anchor by pin_weight,
        //    then add density gradient UNSCALED.
        //    Kirchhoff gradient is ∝ degree (more pins = more demand = stronger pressure),
        //    so dividing by pin_weight normalizes it across cells.
        //    Density gradient is ∝ mass (equal for all cells) — scaling it by pin_weight
        //    would attenuate spreading for high-degree cells, the exact ones forming clusters.
        let viscosity = state.compute_cell_viscosity(ctx, &criticality, cfg.timing_weight);
        for ((gx, gy), ((&ddx, &ddy), (&pw, &v))) in grad_x.iter_mut().zip(grad_y.iter_mut())
            .zip(dx.iter().zip(&dy).zip(pin_weights.iter().zip(&viscosity)))
        {
            let w = (pw + v).max(1.0);
            *gx /= w;
            *gy /= w;
            *gx += ddx;
            *gy += ddy;
        }

        // 9. Velocity field step: damped momentum, geometry-preserving.
        //    Soft gradient normalization: cap RMS to 1.0 to prevent overshooting
        //    without losing relative magnitude information between cells.
        {
            let rms = ((grad_x.iter().map(|g| g * g).sum::<f64>()
                + grad_y.iter().map(|g| g * g).sum::<f64>()) / (2 * n) as f64).sqrt();
            if rms > 1.0 {
                let scale = 1.0 / rms;
                for g in grad_x.iter_mut() { *g *= scale; }
                for g in grad_y.iter_mut() { *g *= scale; }
            }
        }
        vel_x.step(&grad_x);
        vel_y.step(&grad_y);

        // 9b. Decoupled Helmholtz de-clustering: separate step size, bypasses viscosity.
        //     Directly pushes cells away from smoothed congestion potential.
        if iter > 0 && !helmholtz_cache.is_empty() {
            let (hx, hy) = state.field_gradient(&helmholtz_cache, w);
            vel_x.step_helmholtz(&hx);
            vel_y.step_helmholtz(&hy);
        }

        vel_x.clamp_positions_range(0.0, max_x);
        vel_y.clamp_positions_range(0.0, max_y);
        state.cell_x.copy_from_slice(vel_x.positions());
        state.cell_y.copy_from_slice(vel_y.positions());

        // 10. Augmented Lagrangian dual update: increase λ at overcrowded tiles.
        //     Uses pre_density (before velocity step) as approximation to avoid
        //     a second expensive build_density_field call. The velocity step is
        //     small (~0.2 tiles) so the density field barely changes.
        {
            let num_tiles = w * h;
            for i in 0..num_tiles {
                tile_lambda[i] = (tile_lambda[i] + al_alpha * (pre_density[i] - 1.0)).max(0.0);
            }
            // Dampen velocity periodically so cells re-adapt to the changed λ landscape.
            // Halving (not zeroing) maintains trajectory continuity.
            if iter > 0 && iter % 50 == 0 {
                vel_x.dampen(0.5);
                vel_y.dampen(0.5);
            }
        }

        // 11. Track best position: CHPWL with soft density penalty.
        //     score = CHPWL × max(1, maxρ/target) — multiplicative so density
        //     inflates wirelength cost in congested regions without dominating.
        //     target_rho = 3.0: below this, pure CHPWL; above, proportional penalty.
        let chpwl_now = state.continuous_hpwl(ctx);
        let (_, max_rho, _) = overlap
            .unwrap_or_else(|| state::OptTransState::overlap_metrics_from_field(&pre_density));
        let target_rho = 3.0;
        let score = chpwl_now * (max_rho / target_rho).max(1.0);
        loop_state.record_metric(score, &state.cell_x, &state.cell_y, iter);

        // 12. Reporting + convergence.
        let is_report_iter = iter % cfg.report_interval == 0 || iter == cfg.max_outer_iters - 1;
        if is_report_iter {
            let (overflow_ratio, max_rho, _) = overlap
                .unwrap_or_else(|| state::OptTransState::overlap_metrics_from_field(&pre_density));
            let max_lambda = tile_lambda.iter().copied().fold(0.0_f64, f64::max);
            eprintln!(
                "OptTrans iter {}: chpwl={:.0}, maxλ={:.2}, overflow={:.1}%, maxρ={:.1}, β={:.2}",
                iter, chpwl_now, max_lambda, overflow_ratio * 100.0, max_rho, beta,
            );

            // Converge if best metric hasn't improved for `patience` iterations.
            let stale = iter as i64 - loop_state.best_iter as i64;
            if stale > converge_patience as i64 {
                eprintln!("OptTrans placer converged at iteration {} (no improvement for {} iters)", iter, stale);
                break;
            }

            // Early termination: if overflow is low enough for legalization
            // to succeed, stop early. At low utilization, even 5% overflow
            // legalizes fine since there's abundant capacity.
            if iter >= 20 {
                let (ov, _, _) = overlap
                    .unwrap_or_else(|| state::OptTransState::overlap_metrics_from_field(&pre_density));
                let ov_threshold = if utilization < 0.05 { 0.03 } else { 0.02 };
                if ov < ov_threshold {
                    eprintln!("OptTrans placer converged at iteration {} (overflow {:.1}% < {:.0}%)", iter, ov * 100.0, ov_threshold * 100.0);
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

/// Per-tile neighbor count for the Helmholtz Laplacian (Neumann BCs).
/// Constant for a given grid — compute once, reuse across iterations.
/// Stored as f64 to avoid per-iteration u8→f64 casts in the diagonal update.
fn helmholtz_neighbor_count(w: usize, h: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0u32;
            if x > 0 { n += 1; }
            if x + 1 < w { n += 1; }
            if y > 0 { n += 1; }
            if y + 1 < h { n += 1; }
            counts[y * w + x] = n as f64;
        }
    }
    counts
}

/// Helmholtz operator off-diagonal: -1 for each grid edge (upper triangle).
/// Constant for a given grid — compute once, reuse across iterations.
fn helmholtz_off_diag(w: usize, h: usize) -> Vec<(usize, usize, f64)> {
    let mut off = Vec::with_capacity(2 * w * h);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if x + 1 < w { off.push((idx, idx + 1, -1.0)); }
            if y + 1 < h { off.push((idx, idx + w, -1.0)); }
        }
    }
    off
}

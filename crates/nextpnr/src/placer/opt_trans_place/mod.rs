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
use super::solver::{DirectSolver, VelocityFieldSolver};
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

/// Lock cells whose BELs only exist on boundary/IO tiles.
///
/// These cells (IOB, clock buffers, etc.) cannot benefit from the continuous
/// solve because their valid positions are sparse and fixed at the chip edge.
/// Locking them makes them fixed anchors in the Kirchhoff system, naturally
/// pulling connected logic toward the boundary via pressure gradients.
fn lock_boundary_cells(ctx: &mut Context) {
    use crate::common::PlaceStrength;
    use rustc_hash::FxHashSet;

    // Collect all cell types present in the design.
    let mut cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for (_id, cell) in ctx.design.iter_alive_cells() {
        cell_types.insert(cell.cell_type);
    }

    // For each cell type, check if its BELs are on CLB tiles (>= 4 BELs per tile).
    // Cell types with NO BELs on CLB tiles are boundary/IO types.
    let mut clb_cell_types: FxHashSet<crate::common::IdString> = FxHashSet::default();
    for &ct in &cell_types {
        let on_clb = ctx.bels_for_bucket(ct).any(|b| {
            let loc = b.loc();
            let tile = ctx.chipdb().tile_by_xy(loc.x, loc.y);
            ctx.chipdb().tile_type(tile).bels.len() >= 4
        });
        if on_clb {
            clb_cell_types.insert(ct);
        }
    }

    let mut locked_count = 0usize;
    let cell_ids: Vec<_> = ctx
        .design
        .iter_alive_cells()
        .map(|(id, _)| id)
        .collect();

    for cell_id in cell_ids {
        let cell = ctx.design.cell(cell_id);
        if cell.bel_strength.is_locked() {
            continue;
        }
        if clb_cell_types.contains(&cell.cell_type) {
            continue;
        }
        if cell.bel.is_some() {
            ctx.design.cell_mut(cell_id).bel_strength = PlaceStrength::Locked;
            locked_count += 1;
        }
    }

    if locked_count > 0 {
        eprintln!("Locked {} boundary/IO cells as fixed anchors", locked_count);
    }
}

pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    initial_placement(ctx)?;
    ctx.populate_bel_buckets();

    // Lock IO cells at their initial placement. IO BELs are only at the chip
    // boundary and cannot participate in the continuous fluid solve. Locking
    // them makes them fixed anchors that attract connected logic cells toward
    // the boundary via the Kirchhoff pressure gradient.
    lock_boundary_cells(ctx);

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

    // IO boost: amplify demand on nets with a fixed (locked/IO) pin.
    // With IO cells locked as fixed anchors, a mild boost is enough —
    // the Kirchhoff pressure gradient already pulls logic toward IOs.
    let io_boost = cfg.io_boost;

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

    // Build solver infrastructure (one-time cost, gated on config).
    let pipe_endpoints: Vec<(usize, usize)> = state
        .network
        .pipes
        .iter()
        .map(|p| {
            let (a, b) = (p.from, p.to);
            if a < b { (a, b) } else { (b, a) }
        })
        .collect();
    let n_junctions = state.network.num_junctions();

    let mut direct_solver = if matches!(cfg.kirchhoff_solver, config::KirchhoffSolver::Direct) {
        Some(DirectSolver::new(n_junctions, &pipe_endpoints, w, h))
    } else {
        None
    };

    let mut amg_solver = if matches!(cfg.kirchhoff_solver, config::KirchhoffSolver::AmgCG) {
        let amg_structure = crate::placer::solver::amg::AmgStructure::new(n_junctions, &pipe_endpoints);
        let dummy_diag = vec![1.0; n_junctions];
        let dummy_offdiag: Vec<(usize, usize, f64)> = pipe_endpoints.iter().map(|&(a,b)| (a, b, -1.0)).collect();
        Some(crate::placer::solver::amg::AmgPreconditioner::new(
            amg_structure, &dummy_diag, &dummy_offdiag,
        ))
    } else {
        None
    };

    // Extract target clock period from timing analyser.
    let target_period = if cfg.timing_weight > 0.0 {
        let mut ta = crate::timing::TimingAnalyser::new();
        ta.setup_and_run(ctx);
        let min_period_ps = ta.clock_constraints().values().copied().min().unwrap_or(10_000);
        min_period_ps as f64
    } else {
        0.0
    };

    let mut t_demand = 0.0f64;
    let mut t_density = 0.0f64;
    let mut t_kirchhoff = 0.0f64;
    let mut t_gradient = 0.0f64;
    let mut t_velocity = 0.0f64;
    let mut t_other = 0.0f64;

    for iter in 0..cfg.max_outer_iters {
        let progress = iter as f64 / cfg.max_outer_iters as f64;
        // Beckmann coupling strength ramp: starts at 0, grows to full beta.
        let beta = cfg.turbulence_beta * (1.0 - (-3.0 * progress).exp());
        let t_iter_start = std::time::Instant::now();

        // 1. Optional expanding box constraint.
        if cfg.enable_expanding_box {
            state.clamp_to_box(progress);
        } else {
            state.clamp_positions();
        }

        let t0 = std::time::Instant::now();
        // 2. Build net demand vector (IO-boosted, timing-amplified).
        let demand = state.compute_net_demands(
            ctx, &criticality, cfg.timing_weight, io_boost, cfg.pump_gain,
        );

        let t1 = std::time::Instant::now();
        t_demand += (t1 - t0).as_secs_f64();
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

        let t2 = std::time::Instant::now();
        t_density += (t2 - t1).as_secs_f64();
        // 3. Kirchhoff solve: L(R_eff) · P = demand.
        //    Density overflow increases pipe resistance in congested regions.
        {
            let mut solver_ctx = kirchhoff::SolverCtx {
                solver_type: cfg.kirchhoff_solver,
                cg_max_iters: cfg.cg_max_iters,
                cg_tol: cfg.cg_tolerance,
                direct: direct_solver.as_mut(),
                amg: amg_solver.as_mut(),
            };
            kirchhoff::kirchhoff_solve(
                &mut state.network, &demand, beta,
                cfg.newton_iters, &density_overflow, &mut solver_ctx,
            );
        }

        // 3b. Beckmann aggregate flow update + self-normalization.
        {
            let ema_alpha = 0.3;
            for pipe in &mut state.network.pipes {
                let current = pipe.flow.abs();
                pipe.aggregate_flow = ema_alpha * pipe.aggregate_flow
                    + (1.0 - ema_alpha) * current;
            }

            // Self-normalize: compute P95 of (aggregate_flow / capacity)
            // over pipes with real routing capacity (capacity > 1, excludes
            // NULL-tile pipes that have capacity=1 and skew the distribution).
            // This keeps Beckmann utilization in [0, ~3] range, preserving
            // RELATIVE differences while preventing penalty saturation.
            let mut utils: Vec<f64> = state.network.pipes.iter()
                .filter(|p| p.capacity > 1.0)
                .map(|p| p.aggregate_flow / p.capacity)
                .collect();
            if !utils.is_empty() {
                utils.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let p95_idx = (utils.len() as f64 * 0.95) as usize;
                let p95 = utils[p95_idx.min(utils.len() - 1)].max(1e-6);
                state.network.agg_flow_scale = p95;
            }
        }

        let t3 = std::time::Instant::now();
        t_kirchhoff += (t3 - t2).as_secs_f64();
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

        let t4 = std::time::Instant::now();
        t_gradient += (t4 - t3).as_secs_f64();
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

        let t5 = std::time::Instant::now();
        t_velocity += (t5 - t4).as_secs_f64();
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
        // Lexicographic scoring: strongly prefer low overflow, then minimize HPWL.
        let target_rho = 1.5;
        let score = if max_rho > target_rho {
            let overflow_penalty = (max_rho / target_rho).powi(2);
            chpwl_now * overflow_penalty
        } else {
            chpwl_now
        };
        loop_state.record_metric(score, &state.cell_x, &state.cell_y, iter);

        // 12. Reporting + convergence.
        let is_report_iter = iter % cfg.report_interval == 0 || iter == cfg.max_outer_iters - 1;
        if is_report_iter {
            let (overflow_ratio, max_rho, _) = overlap
                .unwrap_or_else(|| state::OptTransState::overlap_metrics_from_field(&pre_density));
            let max_lambda = tile_lambda.iter().copied().fold(0.0_f64, f64::max);
            let max_agg_util = state.network.pipes.iter()
                .filter(|p| p.capacity > 0.0)
                .map(|p| p.aggregate_flow / p.capacity / state.network.agg_flow_scale)
                .fold(0.0_f64, f64::max);
            eprintln!(
                "OptTrans iter {}: chpwl={:.0}, maxλ={:.2}, overflow={:.1}%, maxρ={:.1}, β={:.2}, maxAggUtil={:.2}",
                iter, chpwl_now, max_lambda, overflow_ratio * 100.0, max_rho, beta, max_agg_util,
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

    // Restore best continuous positions.
    // Print timing breakdown
    let t_total = t_demand + t_density + t_kirchhoff + t_gradient + t_velocity;
    eprintln!(
        "Timing breakdown ({:.1}s total): demand={:.0}ms density={:.0}ms kirchhoff={:.0}ms gradient={:.0}ms velocity={:.0}ms",
        t_total, t_demand*1000.0, t_density*1000.0, t_kirchhoff*1000.0, t_gradient*1000.0, t_velocity*1000.0,
    );

    state.cell_x.copy_from_slice(&loop_state.best_positions_x);
    state.cell_y.copy_from_slice(&loop_state.best_positions_y);

    // Snap continuous positions to nearest CLB column/row before legalization.
    // The continuous placer works on the full tile grid, but BELs only exist
    // on CLB tiles (~29% of columns). Snapping reduces legalization displacement
    // from ~29 tiles to ~1 tile.
    {
        let chipdb = ctx.chipdb();
        let net_w = state.network.width;
        let net_h = state.network.height;
        let x0 = state.network.x0;
        let y0 = state.network.y0;

        // Build sorted list of CLB x-columns and y-rows within the cropped grid.
        let mut clb_xs: Vec<f64> = Vec::new();
        let mut clb_ys: Vec<f64> = Vec::new();
        let mut seen_x = std::collections::HashSet::new();
        let mut seen_y = std::collections::HashSet::new();
        for vy in 0..net_h {
            for vx in 0..net_w {
                let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
                let bel_count = chipdb.tile_type(tile).bels.len();
                // Only count tiles with LUT/DFF BELs (not GND/VCC/IO)
                if bel_count >= 4 {
                    if seen_x.insert(vx) {
                        clb_xs.push(vx as f64);
                    }
                    if seen_y.insert(vy) {
                        clb_ys.push(vy as f64);
                    }
                }
            }
        }
        clb_xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        clb_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !clb_xs.is_empty() && !clb_ys.is_empty() {
            let mut snap_max_disp = 0.0f64;
            let mut snap_total_disp = 0.0f64;
            for i in 0..state.num_cells() {
                // Snap x to nearest CLB column
                let cx = state.cell_x[i];
                let best_x = clb_xs.iter().copied()
                    .min_by(|a, b| (a - cx).abs().partial_cmp(&(b - cx).abs()).unwrap())
                    .unwrap();
                state.cell_x[i] = best_x;

                // Snap y to nearest CLB row
                let cy = state.cell_y[i];
                let best_y = clb_ys.iter().copied()
                    .min_by(|a, b| (a - cy).abs().partial_cmp(&(b - cy).abs()).unwrap())
                    .unwrap();
                state.cell_y[i] = best_y;
                let d = ((best_x - cx).powi(2) + (best_y - cy).powi(2)).sqrt();
                snap_total_disp += d;
                snap_max_disp = snap_max_disp.max(d);
            }
            let snap_avg = snap_total_disp / state.num_cells().max(1) as f64;
            eprintln!(
                "Snapped {} cells to {} CLB columns × {} CLB rows (avg_disp={:.1}, max_disp={:.1})",
                state.num_cells(), clb_xs.len(), clb_ys.len(), snap_avg, snap_max_disp,
            );

            // Post-snap per-type spreading: ensure no CLB tile has more cells
            // of each type than that type's fair share of BELs.
            use crate::common::IdString;

            // Count distinct cell types.
            let mut cell_types_present: FxHashSet<IdString> = FxHashSet::default();
            for i in 0..state.num_cells() {
                let ct = ctx.design.cell(state.idx_to_cell[i]).cell_type;
                cell_types_present.insert(ct);
            }
            let n_types = cell_types_present.len().max(1);

            // Per-tile capacity = total BELs / number of cell types (fair share).
            let mut tile_cap: FxHashMap<(i32, i32), usize> = FxHashMap::default();
            for &cx in &clb_xs {
                for &cy in &clb_ys {
                    let vx = cx as i32;
                    let vy = cy as i32;
                    let tile = chipdb.tile_by_xy(vx + x0, vy + y0);
                    let total_bels = chipdb.tile_type(tile).bels.len();
                    tile_cap.insert((vx, vy), total_bels / n_types);
                }
            }

            // Group cells by (type, snap position).
            let mut type_tile_cells: FxHashMap<(IdString, i32, i32), Vec<usize>> = FxHashMap::default();
            for i in 0..state.num_cells() {
                let ct = ctx.design.cell(state.idx_to_cell[i]).cell_type;
                let tx = state.cell_x[i].round() as i32;
                let ty = state.cell_y[i].round() as i32;
                type_tile_cells.entry((ct, tx, ty)).or_default().push(i);
            }

            // Per-type remaining capacity (each type gets its fair share).
            let mut remaining_cap: FxHashMap<(IdString, i32, i32), usize> = FxHashMap::default();
            for &ct in &cell_types_present {
                for (&(vx, vy), &cap) in &tile_cap {
                    remaining_cap.insert((ct, vx, vy), cap);
                }
            }

            let mut spread_count = 0usize;
            let mut spread_max_disp = 0.0f64;

            let mut groups: Vec<_> = type_tile_cells.into_iter().collect();
            groups.sort_by_key(|(_, cells)| std::cmp::Reverse(cells.len()));

            for ((ct, tx, ty), cells) in &groups {
                let cap = remaining_cap.get(&(*ct, *tx, *ty)).copied().unwrap_or(0);
                if cells.len() <= cap {
                    if let Some(c) = remaining_cap.get_mut(&(*ct, *tx, *ty)) {
                        *c = c.saturating_sub(cells.len());
                    }
                    continue;
                }

                let mut cell_dists: Vec<(usize, f64)> = cells
                    .iter()
                    .map(|&ci| {
                        let dx = state.cell_x[ci] - *tx as f64;
                        let dy = state.cell_y[ci] - *ty as f64;
                        (ci, dx * dx + dy * dy)
                    })
                    .collect();
                cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                remaining_cap.insert((*ct, *tx, *ty), 0);

                for &(ci, _) in cell_dists.iter().skip(cap) {
                    let mut best: Option<(i32, i32, f64)> = None;
                    for (&(bt, bx, by), cap_left) in &remaining_cap {
                        if bt != *ct || *cap_left == 0 {
                            continue;
                        }
                        let dx = (bx - tx) as f64;
                        let dy = (by - ty) as f64;
                        let d = dx * dx + dy * dy;
                        if best.is_none() || d < best.unwrap().2 {
                            best = Some((bx, by, d));
                        }
                    }
                    if let Some((bx, by, d)) = best {
                        state.cell_x[ci] = bx as f64;
                        state.cell_y[ci] = by as f64;
                        *remaining_cap.get_mut(&(*ct, bx, by)).unwrap() -= 1;
                        spread_count += 1;
                        spread_max_disp = spread_max_disp.max(d.sqrt());
                    }
                }
            }

            if spread_count > 0 {
                eprintln!(
                    "Spread {} cells from overcrowded tiles (max_disp={:.1})",
                    spread_count, spread_max_disp,
                );
            }
        }
    }

    // --- Pre-legalization diagnostics ---
    let pre_chpwl = state.continuous_hpwl(ctx);
    let pre_density = state.build_density_field(ctx);
    let (pre_overflow, pre_max_rho, _) =
        state::OptTransState::overlap_metrics_from_field(&pre_density);
    eprintln!(
        "Pre-legalization:  CHPWL={:.0}, overflow={:.1}%, maxρ={:.1}",
        pre_chpwl, pre_overflow * 100.0, pre_max_rho,
    );

    legalize::legalize_opt_trans(ctx, &state, cfg.lap_max_cells)?;

    // --- Post-legalization diagnostics ---
    // Compute displacement: for each cell, distance from continuous position
    // to legalized BEL position.
    let mut total_disp = 0.0f64;
    let mut total_disp_x = 0.0f64;
    let mut total_disp_y = 0.0f64;
    let mut max_disp = 0.0f64;
    let mut n_displaced = 0usize;
    let x0 = state.network.x0 as f64;
    let y0 = state.network.y0 as f64;
    for (cell_id, &ci) in &state.cell_to_idx {
        if let Some(bel_id) = ctx.cell(*cell_id).bel_id() {
            let loc = ctx.chipdb().bel_loc(bel_id);
            // cell_x/y are in virtual (cropped) coords; loc is physical.
            let dx = (state.cell_x[ci] + x0) - loc.x as f64;
            let dy = (state.cell_y[ci] + y0) - loc.y as f64;
            let d = (dx * dx + dy * dy).sqrt();
            total_disp += d;
            total_disp_x += dx.abs();
            total_disp_y += dy.abs();
            if d > max_disp {
                max_disp = d;
            }
            if d > 0.5 {
                n_displaced += 1;
            }
        }
    }
    let avg_disp = if state.num_cells() > 0 {
        total_disp / state.num_cells() as f64
    } else {
        0.0
    };
    // Post-legalization HPWL from actual placed positions.
    let post_hpwl = {
        let mut hpwl = 0.0f64;
        for net_idx in ctx.design.iter_net_indices() {
            let net = ctx.net(net_idx);
            if !net.is_alive() {
                continue;
            }
            let driver = match net.driver() {
                Some(d) => d,
                None => continue,
            };
            let mut min_x = i32::MAX;
            let mut max_x = i32::MIN;
            let mut min_y = i32::MAX;
            let mut max_y = i32::MIN;
            // Driver
            if let Some(bel_id) = ctx.cell(driver.cell).bel_id() {
                let loc = ctx.chipdb().bel_loc(bel_id);
                let (x, y) = (loc.x, loc.y);
                min_x = min_x.min(x); max_x = max_x.max(x);
                min_y = min_y.min(y); max_y = max_y.max(y);
            }
            // Users
            for user in net.users() {
                if let Some(bel_id) = ctx.cell(user.cell).bel_id() {
                    let loc = ctx.chipdb().bel_loc(bel_id);
                    let (x, y) = (loc.x, loc.y);
                    min_x = min_x.min(x); max_x = max_x.max(x);
                    min_y = min_y.min(y); max_y = max_y.max(y);
                }
            }
            if min_x <= max_x {
                hpwl += (max_x - min_x + max_y - min_y) as f64;
            }
        }
        hpwl
    };
    let avg_disp_x = if state.num_cells() > 0 { total_disp_x / state.num_cells() as f64 } else { 0.0 };
    let avg_disp_y = if state.num_cells() > 0 { total_disp_y / state.num_cells() as f64 } else { 0.0 };
    eprintln!(
        "Post-legalization: HPWL={:.0}, displaced={}/{} (avg={:.1}, max={:.1} tiles, avg_dx={:.1}, avg_dy={:.1})",
        post_hpwl, n_displaced, state.num_cells(), avg_disp, max_disp, avg_disp_x, avg_disp_y,
    );
    eprintln!(
        "Legalization cost: ΔHPWL={:+.0} ({:+.1}%)",
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

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

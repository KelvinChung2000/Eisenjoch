//! Main Beckmann OT placement algorithm.

use log::info;
use crate::context::Context;
use crate::metrics::total_hpwl;
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
use crate::solver::preconditioner::{JacobiPreconditioner, AmgPreconditioner};
use crate::solver::cg::{solve_cg, solve_cg_batched};

use super::config::{OptTransPlacerCfg, PreconditionerType};
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Main Beckmann OT placement algorithm.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    // 1. Build subtile pipe network from chipdb.
    let mut network = PipeNetwork::from_context(ctx, cfg.subtile_resolution);

    // 2. Collect movable cells, init positions.
    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    let n = idx_to_cell.len();

    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    match cfg.init_strategy {
        super::config::InitStrategy::Centroid => {
            common::init_positions_center_drop(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        _ => {
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
    }

    // Convert to virtual grid coords (offset by grid origin).
    for x in cell_x.iter_mut() {
        *x -= network.x0 as f64;
    }
    for y in cell_y.iter_mut() {
        *y -= network.y0 as f64;
    }

    let max_x = (network.width - 1) as f64;
    let max_y = (network.height - 1) as f64;

    info!(
        "OptTrans placer: {} movable cells, {}x{} grid ({}x{} subtile, N={}), {} pipes",
        n,
        network.width,
        network.height,
        network.subtile_width(),
        network.subtile_height(),
        network.resolution,
        network.num_pipes(),
    );

    // 3. Build resistance model.
    let resistance_model = ResistanceModel {
        congestion_exponent: cfg.congestion_exponent,
        interference_weight: cfg.interference_weight,
        timing_weight: cfg.timing_weight,
        density_weight: 0.0,
    };

    // 4. Tracking and amplification.
    let mut best_chpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_iter = 0usize;
    let grid_diag = ((max_x * max_x + max_y * max_y) as f64).sqrt();
    let mut cached_amg: Option<AmgPreconditioner> = None;

    // Force amplification: CHPWL / (n × utilization).
    // Compensates for diluted force field on large grids.
    // Fixed at initial scale to avoid positive feedback.
    // Force amplification: grid_diag compensates for diluted flow field.
    // The raw flow vector is O(1/sqrt(n_nodes)), so scaling by grid_diag
    // produces steps proportional to the grid size.
    let amplification = grid_diag;

    // 5. Main loop.
    for iter in 0..cfg.max_iters {
        let n_nodes = network.num_nodes();

        // b. Build the Kirchhoff Laplacian from pipe conductances.
        let mut laplacian = SparseMatrix::new(n_nodes);

        for pipe in &mut network.pipes {
            let r_eff = resistance_model.effective_resistance(pipe, 0.0);
            let conductance = 1.0 / r_eff.max(1e-12);
            pipe.eff_conductance = conductance;
            laplacian.add_connection(pipe.from, pipe.to, conductance);
        }

        // Matrix diagnostics (first iteration only).
        if iter == 0 {
            let diag = laplacian.diag();
            let diag_min = diag.iter().cloned().fold(f64::INFINITY, f64::min);
            let diag_max = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n_zero_diag = diag.iter().filter(|&&d| d == 0.0).count();

            // Off-diagonal: check for positive values (should all be negative for Kirchhoff)
            let off = laplacian.off_diag();
            let off_min = off.iter().map(|&(_, _, v)| v).fold(f64::INFINITY, f64::min);
            let off_max = off.iter().map(|&(_, _, v)| v).fold(f64::NEG_INFINITY, f64::max);
            let n_positive_offdiag = off.iter().filter(|&&(_, _, v)| v > 0.0).count();

            // Conductance range (from pipes)
            let cond_min = network.pipes.iter()
                .map(|p| 1.0 / resistance_model.effective_resistance(p, 0.0).max(1e-12))
                .fold(f64::INFINITY, f64::min);
            let cond_max = network.pipes.iter()
                .map(|p| 1.0 / resistance_model.effective_resistance(p, 0.0).max(1e-12))
                .fold(f64::NEG_INFINITY, f64::max);

            // Row sum check (should be ~0 for Kirchhoff before regularization)
            let mut row_sums = vec![0.0f64; n_nodes];
            for i in 0..n_nodes {
                row_sums[i] += diag[i];
            }
            for &(lo, hi, val) in off {
                row_sums[lo] += val;
                row_sums[hi] += val;
            }
            let row_sum_max = row_sums.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let row_sum_avg = row_sums.iter().map(|v| v.abs()).sum::<f64>() / n_nodes as f64;

            eprintln!("  MATRIX: n={}, nnz_offdiag={}", n_nodes, off.len());
            eprintln!("  diag: [{:.3e}, {:.3e}], zero_diag={}", diag_min, diag_max, n_zero_diag);
            eprintln!("  offdiag: [{:.3e}, {:.3e}], positive_offdiag={}", off_min, off_max, n_positive_offdiag);
            eprintln!("  conductance: [{:.3e}, {:.3e}], ratio={:.1e}", cond_min, cond_max, cond_max / cond_min.max(1e-30));
            eprintln!("  row_sum (pre-reg): max={:.3e}, avg={:.3e}", row_sum_max, row_sum_avg);
        }

        // Regularize: add ε·I to make strictly SPD.
        let avg_diag = laplacian.diag().iter().sum::<f64>() / n_nodes as f64;
        let epsilon = 1e-4 * avg_diag;
        for i in 0..n_nodes {
            laplacian.add_diagonal(i, epsilon);
        }

        // c. Build operator and preconditioner (shared across all solves).
        let op = SparseMatrixOp::from_matrix(&mut laplacian);
        let amg_ref = match cfg.preconditioner {
            PreconditionerType::Amg => {
                let amg = cached_amg.get_or_insert_with(|| {
                    AmgPreconditioner::setup(n_nodes, laplacian.diag(), laplacian.off_diag())
                });
                if iter > 0 {
                    amg.update_values(laplacian.diag(), laplacian.off_diag());
                }
                Some(&*amg)
            }
            _ => None,
        };
        let jacobi_precond = if amg_ref.is_none() {
            Some(JacobiPreconditioner::new(laplacian.diag()))
        } else {
            None
        };

        // d. Per-net IO-only Kirchhoff solves → displacement + congestion update.
        // Single formulation: same Laplacian, one CG per net. Per-net pressures
        // are summed into global node pressures and pipe flows for congestion.
        let mut dx = vec![0.0; n];
        let mut dy = vec![0.0; n];
        let mut global_pressure = vec![0.0f64; n_nodes];

        let net_infos = demand::collect_nets_for_solve(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let n_nets = net_infos.len();
        let mut total_per_net_iters = 0usize;
        let cg_result;

        {
            let t_batch = std::time::Instant::now();
            let batch_tol = (cfg.cg_tol * 100.0).min(1e-2);
            let max_per_net_iters = 30;
            let mut per_net_pressure = vec![0.0; n_nodes];
            let mut last_cg_iters = 0usize;
            let mut last_cg_resid = 0.0f64;

            for net_info in &net_infos {
                let net_rhs = demand::build_net_rhs(net_info, &network, cfg);
                per_net_pressure.fill(0.0);

                let nr = if let Some(amg) = amg_ref {
                    solve_cg(&op, amg, &net_rhs, &mut per_net_pressure, batch_tol, max_per_net_iters)
                } else {
                    solve_cg(&op, jacobi_precond.as_ref().unwrap(), &net_rhs, &mut per_net_pressure, batch_tol, max_per_net_iters)
                };
                total_per_net_iters += nr.iterations;
                last_cg_iters = nr.iterations;
                last_cg_resid = nr.residual;

                // Accumulate into global pressure for congestion tracking.
                for i in 0..n_nodes {
                    global_pressure[i] += per_net_pressure[i];
                }

                // dp/dist with role sign from per-net pressure field.
                demand::accumulate_dp_dist_force(net_info, &per_net_pressure, &network, &mut dx, &mut dy);
            }

            // Update node pressures and pipe flows from summed per-net fields.
            for (i, node) in network.nodes.iter_mut().enumerate() {
                node.pressure = global_pressure[i];
            }
            for pipe in &mut network.pipes {
                let r_eff = resistance_model.effective_resistance(pipe, 0.0);
                pipe.flow = (global_pressure[pipe.from] - global_pressure[pipe.to]) / r_eff.max(1e-12);
            }

            cg_result = crate::solver::cg::CgResult {
                iterations: total_per_net_iters,
                converged: true,
                residual: last_cg_resid,
            };

            // Second term of energy gradient: density spreading.
            // Gaussian-smoothed density gradient pushes cells from overcrowded
            // tiles toward open area, bridging across non-CLB column gaps.
            demand::project_velocity(&cell_x, &cell_y, &mut dx, &mut dy, &network);

            // Amplify raw force and clamp per-cell magnitude.
            // Adaptive clamp: large steps when CHPWL is high and utilization
            // is low (cells far from targets, room to move), shrinks as
            // placement improves (cells near targets, need precision).
            let chpwl_now = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network).max(1.0);
            // Adaptive step: chpwl/n gives average wirelength per cell (how far to go).
            // Decays naturally as placement improves.
            let avg_wl = chpwl_now / n as f64;
            let max_step = (avg_wl * cfg.step_scale * 0.1).min(grid_diag * 0.1);
            for i in 0..n {
                dx[i] *= amplification;
                dy[i] *= amplification;
                let mag = (dx[i] * dx[i] + dy[i] * dy[i]).sqrt();
                if mag > max_step {
                    let scale = max_step / mag;
                    dx[i] *= scale;
                    dy[i] *= scale;
                }
            }

            let batch_ms = t_batch.elapsed().as_millis();
            if iter == 0 {
                eprintln!("    per-net CG: {} nets, {} total iters, {:.1}ms, amp={:.1}",
                    n_nets, total_per_net_iters, batch_ms, amplification);
            }
        }

        // Diagnostics.
        {
            let p_min = global_pressure.iter().cloned().fold(f64::INFINITY, f64::min);
            let p_max = global_pressure.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let dx_max = dx.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let dy_max = dy.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let disp_rms = ((dx.iter().map(|v| v*v).sum::<f64>() + dy.iter().map(|v| v*v).sum::<f64>()) / (2.0 * n as f64)).sqrt();
            eprintln!(
                "  [{}] P=[{:.1e},{:.1e}] dx={:.4} dy={:.4} rms={:.4} nets={} batch_cg={}",
                iter, p_min, p_max, dx_max, dy_max, disp_rms, n_nets, total_per_net_iters,
            );

            // Congestion diagnostics every 10 iters.
            if iter % 10 == 0 {
                let mut flow_abs: Vec<f64> = network.pipes.iter().map(|p| p.flow.abs()).collect();
                flow_abs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let flow_max = flow_abs.first().copied().unwrap_or(0.0);
                let flow_p90 = flow_abs.get(flow_abs.len() / 10).copied().unwrap_or(0.0);
                let n_high_util = network.pipes.iter().filter(|p| p.flow.abs() / p.capacity.max(1.0) > 1.0).count();
                let n_high_nets = network.pipes.iter().filter(|p| p.net_count > 3).count();
                let max_nets = network.pipes.iter().map(|p| p.net_count).max().unwrap_or(0);

                // R_eff range
                let r_effs: Vec<f64> = network.pipes.iter()
                    .map(|p| resistance_model.effective_resistance(p, 0.0))
                    .collect();
                let r_min = r_effs.iter().cloned().fold(f64::INFINITY, f64::min);
                let r_max = r_effs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                eprintln!(
                    "    congestion: flow_max={:.2} flow_p90={:.2} pipes_util>1={} pipes_nets>3={} max_nets={} R=[{:.2},{:.2}]",
                    flow_max, flow_p90, n_high_util, n_high_nets, max_nets, r_min, r_max,
                );
            }

            // Dump field and cell data at iter 0 for visualization.
            if iter == 0 {
                let near_zero = (0..n).filter(|&i| dx[i].abs() < 0.01 && dy[i].abs() < 0.01).count();
                eprintln!("    near-zero displacement: {}/{}", near_zero, n);

                // Dump pressure field as tile-averaged grid.
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create("/tmp/pressure_field.csv") {
                    let res = network.resolution;
                    let tw = network.width as usize;
                    let th = network.height as usize;
                    writeln!(f, "tx,ty,pressure").ok();
                    for ty in 0..th {
                        for tx in 0..tw {
                            let mut p_sum = 0.0;
                            for sy in 0..res {
                                for sx in 0..res {
                                    let ni = network.node_index(tx as i32, ty as i32, sx, sy);
                                    p_sum += global_pressure[ni];
                                }
                            }
                            let n_sub = (res * res) as f64;
                            writeln!(f, "{},{},{:.6}", tx, ty, p_sum / n_sub).ok();
                        }
                    }
                }
                // Dump cell positions + raw displacement vectors.
                if let Ok(mut f) = std::fs::File::create("/tmp/cell_disp.csv") {
                    writeln!(f, "cell_idx,x,y,dx,dy").ok();
                    for ci in 0..n {
                        writeln!(f, "{},{:.2},{:.2},{:.6},{:.6}", ci, cell_x[ci], cell_y[ci], dx[ci], dy[ci]).ok();
                    }
                }
                // Dump fixed/IO cell positions.
                if let Ok(mut f) = std::fs::File::create("/tmp/fixed_cells.csv") {
                    writeln!(f, "x,y").ok();
                    for (_cell_id, cell) in ctx.design.iter_alive_cells() {
                        if !cell.bel_strength.is_locked() { continue; }
                        if let Some(bel) = cell.bel {
                            let loc = ctx.bel(bel).loc();
                            writeln!(f, "{},{}", loc.x - network.x0, loc.y - network.y0).ok();
                        }
                    }
                }
                // Build per-cell IO target centroids from net connectivity.
                // For each net, find fixed pins and associate them with movable cells on the same net.
                let mut cell_io_x: Vec<f64> = vec![0.0; n];
                let mut cell_io_y: Vec<f64> = vec![0.0; n];
                let mut cell_io_count: Vec<usize> = vec![0; n];
                let mut cell_net_count: Vec<usize> = vec![0; n];

                for (_net_id, net) in ctx.design.iter_alive_nets() {
                    let Some(dp) = net.driver() else { continue };

                    // Collect fixed pin positions on this net.
                    let mut fixed_pins: Vec<(f64, f64)> = Vec::new();
                    let mut movable_indices: Vec<usize> = Vec::new();

                    let dc = ctx.design.cell(dp.cell);
                    if dc.bel_strength.is_locked() {
                        if let Some(bel) = dc.bel {
                            let loc = ctx.bel(bel).loc();
                            fixed_pins.push(((loc.x - network.x0) as f64, (loc.y - network.y0) as f64));
                        }
                    } else if let Some(&ci) = cell_to_idx.get(&dp.cell) {
                        movable_indices.push(ci);
                    }

                    for user in net.users() {
                        if !user.is_valid() { continue; }
                        let uc = ctx.design.cell(user.cell);
                        if uc.bel_strength.is_locked() {
                            if let Some(bel) = uc.bel {
                                let loc = ctx.bel(bel).loc();
                                fixed_pins.push(((loc.x - network.x0) as f64, (loc.y - network.y0) as f64));
                            }
                        } else if let Some(&ci) = cell_to_idx.get(&user.cell) {
                            movable_indices.push(ci);
                        }
                    }

                    // Associate fixed pins with movable cells on this net.
                    for &ci in &movable_indices {
                        cell_net_count[ci] += 1;
                        for &(fx, fy) in &fixed_pins {
                            cell_io_x[ci] += fx;
                            cell_io_y[ci] += fy;
                            cell_io_count[ci] += 1;
                        }
                    }
                }

                if let Ok(mut f) = std::fs::File::create("/tmp/cell_connectivity.csv") {
                    writeln!(f, "cell_idx,x,y,dx,dy,io_cx,io_cy,n_io_pins,n_nets").ok();
                    for ci in 0..n {
                        let (io_cx, io_cy) = if cell_io_count[ci] > 0 {
                            (cell_io_x[ci] / cell_io_count[ci] as f64,
                             cell_io_y[ci] / cell_io_count[ci] as f64)
                        } else {
                            (f64::NAN, f64::NAN)
                        };
                        writeln!(f, "{},{:.2},{:.2},{:.6},{:.6},{:.2},{:.2},{},{}",
                            ci, cell_x[ci], cell_y[ci], dx[ci], dy[ci], io_cx, io_cy,
                            cell_io_count[ci], cell_net_count[ci]).ok();
                    }
                }
                eprintln!("    dumped /tmp/pressure_field.csv, /tmp/cell_disp.csv, /tmp/fixed_cells.csv, /tmp/cell_connectivity.csv");
            }
        }

        // f. Direct step.
        for i in 0..n {
            cell_x[i] += dx[i];
            cell_y[i] += dy[i];
        }

        // h. Clamp to grid.
        common::clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

        // i. Update net counts and cell density for next iteration's resistance.
        demand::update_net_counts(&cell_to_idx, &cell_x, &cell_y, &mut network, ctx);
        demand::update_cell_density(&cell_x, &cell_y, &mut network);

        // j. Track best positions.
        let chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        if chpwl < best_chpwl {
            best_chpwl = chpwl;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
            best_iter = iter;
        }

        // k. Report.
        let disp_rms = ((dx.iter().map(|v| v*v).sum::<f64>() + dy.iter().map(|v| v*v).sum::<f64>()) / (2.0 * n as f64)).sqrt();
        let is_report = iter % cfg.report_interval == 0 || iter == cfg.max_iters - 1;
        if is_report {
            let max_util = network.max_utilization();
            eprintln!(
                "OptTrans iter {}: chpwl={:.0}, disp_rms={:.2}, max_util={:.2}, cg={}",
                iter, chpwl, disp_rms, max_util, cg_result.iterations,
            );
        }

        // Convergence: displacement rms below threshold.
        if disp_rms < 0.01 && iter >= 5 {
            eprintln!("OptTrans converged at iteration {} (disp_rms={:.3e})", iter, disp_rms);
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

    // 6. Pre-legalization diagnostics: rebuild demand/flow at best position to inspect.
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    {
        let final_rhs = demand::build_demands(ctx, &cell_to_idx, &cell_x, &cell_y, &network, cfg);
        let rhs_l1: f64 = final_rhs.iter().map(|v| v.abs()).sum();

        // Demand per tile column (sum absolute demand in each x-column).
        let res = network.resolution;
        let tw = network.width as usize;
        let th = network.height as usize;
        let mut col_demand = vec![0.0f64; tw];
        let mut col_cells = vec![0u32; tw];
        for i in 0..n {
            let tx = (cell_x[i] as usize).min(tw - 1);
            col_cells[tx] += 1;
        }
        for ty in 0..th {
            for tx in 0..tw {
                for sy in 0..res {
                    for sx in 0..res {
                        let ni = network.node_index(tx as i32, ty as i32, sx, sy);
                        col_demand[tx] += final_rhs[ni].abs();
                    }
                }
            }
        }

        // Flow stats per x-column.
        let mut col_flow = vec![0.0f64; tw];
        let mut col_max_flow = vec![0.0f64; tw];
        for pipe in &network.pipes {
            let from_node = &network.nodes[pipe.from];
            let tx = from_node.tile_x as usize;
            let flow = pipe.flow.abs();
            col_flow[tx] += flow;
            col_max_flow[tx] = col_max_flow[tx].max(flow);
        }

        eprintln!("  final state: rhs_l1={:.1}, max_util={:.2}", rhs_l1, network.max_utilization());
        eprintln!("  x-column breakdown (cells | demand | total_flow | max_pipe_flow):");
        for tx in 0..tw {
            if col_cells[tx] > 0 || col_demand[tx] > 0.1 || col_flow[tx] > 0.5 {
                eprintln!("    x={:3}: cells={:3} demand={:6.1} flow={:8.1} max_flow={:.2}",
                    tx, col_cells[tx], col_demand[tx], col_flow[tx], col_max_flow[tx]);
            }
        }
    }
    eprintln!("Pre-legalization: CHPWL={:.0}", pre_chpwl);

    // 7. CLB snap + legalize.
    crate::placer::legalize::snap_to_clb_grid(ctx, &mut cell_x, &mut cell_y, &idx_to_cell, &network);

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();

    match cfg.legalization.as_str() {
        "sorted" => {
            crate::placer::legalize::sorted_legalize(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
        "bipartite" => {
            let cost = Box::new(crate::placer::legalize::DistanceCost);
            crate::placer::legalize::bipartite::legalize_bipartite(
                ctx, &idx_to_cell, &phys_x, &phys_y, &*cost, cfg.lap_max_cells,
            )?;
        }
        "greedy" => {
            crate::placer::legalize::legalize_electro(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
        _ => {
            // Default: ring legalizer
            crate::placer::legalize::legalize_ring(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
    }

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

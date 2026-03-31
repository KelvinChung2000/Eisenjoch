//! Main Beckmann OT placement algorithm.
//!
//! Uses driver-only pressure for the transport cost gradient (no self-interaction).
//! Adam optimizer provides per-cell adaptive learning rates — handles the 1/d
//! gradient acceleration near optima without overshooting.
//! Congestion resistance on shared pipes provides natural spreading.

use crate::common::IdString;
use crate::context::Context;
use crate::metrics::{total_hpwl, total_line_estimate};
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::PlacerError;
use crate::solver::cg::{cg_scratch_size, solve_cg_reuse};
use crate::solver::optimizer::AdamOptimizer;
use crate::solver::preconditioner::amg::{AmgPreconditioner, AmgStructure};
use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
use log::info;
use rayon::{prelude::*, ThreadPoolBuilder};
use rustc_hash::{FxHashMap, FxHashSet};

/// Pre-computed per-cell-type valid tile positions and capacities.
struct TypeAwarePlacement {
    /// For each cell type (resolved bucket): sorted list of valid x-coordinates (virtual grid).
    valid_xs: FxHashMap<IdString, Vec<f64>>,
    /// For each cell type: sorted list of valid y-coordinates per x-column.
    valid_ys: FxHashMap<IdString, FxHashMap<i32, Vec<f64>>>,
    /// Per-tile capacity for each cell type: (vx, vy) -> n_compatible_bels.
    tile_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>>,
}

impl TypeAwarePlacement {
    fn build(ctx: &Context, network: &super::network::PipeNetwork) -> Self {
        // Collect all cell types present in the design (resolved to buckets).
        let mut cell_types_present: FxHashSet<IdString> = FxHashSet::default();
        for (_cell_id, cell) in ctx.design.iter_alive_cells() {
            if cell.bel_strength.is_locked() {
                continue;
            }
            let bucket = ctx.resolve_bucket(cell.cell_type);
            cell_types_present.insert(bucket);
        }

        let mut valid_xs: FxHashMap<IdString, Vec<f64>> = FxHashMap::default();
        let mut valid_ys: FxHashMap<IdString, FxHashMap<i32, Vec<f64>>> = FxHashMap::default();
        let mut tile_capacity: FxHashMap<IdString, FxHashMap<(i32, i32), u32>> =
            FxHashMap::default();

        for &ct in &cell_types_present {
            let mut xs_set: FxHashSet<i32> = FxHashSet::default();
            let mut ys_per_x: FxHashMap<i32, FxHashSet<i32>> = FxHashMap::default();
            let cap_map = tile_capacity.entry(ct).or_default();

            for bel in ctx.bels_for_bucket(ct) {
                let loc = bel.loc();
                let vx = loc.x - network.x0;
                let vy = loc.y - network.y0;
                xs_set.insert(vx);
                ys_per_x.entry(vx).or_default().insert(vy);
                *cap_map.entry((vx, vy)).or_insert(0) += 1;
            }

            let mut xs: Vec<f64> = xs_set.into_iter().map(|x| x as f64).collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            valid_xs.insert(ct, xs);

            let mut ys_map: FxHashMap<i32, Vec<f64>> = FxHashMap::default();
            for (x, ys_set) in ys_per_x {
                let mut ys: Vec<f64> = ys_set.into_iter().map(|y| y as f64).collect();
                ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ys_map.insert(x, ys);
            }
            valid_ys.insert(ct, ys_map);
        }

        let n_types = cell_types_present.len();
        for ct in &cell_types_present {
            let n_pos = valid_xs.get(ct).map(|v| v.len()).unwrap_or(0);
            let n_bels: u32 = tile_capacity
                .get(ct)
                .map(|m| m.values().sum())
                .unwrap_or(0);
            eprintln!(
                "  type {}: {} valid columns, {} total BELs",
                ctx.name_of(*ct),
                n_pos,
                n_bels,
            );
        }
        eprintln!("TypeAwarePlacement: {} cell types", n_types);

        Self {
            valid_xs,
            valid_ys,
            tile_capacity,
        }
    }

    /// Snap x to nearest valid column for this cell type. Returns the snapped x.
    fn snap_x(&self, bucket: IdString, x: f64) -> f64 {
        let xs = match self.valid_xs.get(&bucket) {
            Some(xs) if !xs.is_empty() => xs,
            _ => return x,
        };
        let idx = xs.partition_point(|&c| c < x);
        if idx == 0 {
            xs[0]
        } else if idx >= xs.len() {
            *xs.last().unwrap()
        } else {
            let left = xs[idx - 1];
            let right = xs[idx];
            if (x - left) <= (right - x) {
                left
            } else {
                right
            }
        }
    }

    /// Snap y to nearest valid row for this cell type at the given (already-snapped) x column.
    fn snap_y(&self, bucket: IdString, snapped_x: i32, y: f64) -> f64 {
        let ys = match self.valid_ys.get(&bucket) {
            Some(ys_map) => match ys_map.get(&snapped_x) {
                Some(ys) if !ys.is_empty() => ys,
                _ => return y,
            },
            _ => return y,
        };
        let idx = ys.partition_point(|&c| c < y);
        if idx == 0 {
            ys[0]
        } else if idx >= ys.len() {
            *ys.last().unwrap()
        } else {
            let left = ys[idx - 1];
            let right = ys[idx];
            if (y - left) <= (right - y) {
                left
            } else {
                right
            }
        }
    }
}

use super::config::OptTransPlacerCfg;
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Main Beckmann OT placement algorithm.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    let mut network = PipeNetwork::from_context(ctx, cfg.subtile_resolution);

    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    let n = idx_to_cell.len();
    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    let solve_pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads.max(1))
        .build()
        .map_err(|e| {
            PlacerError::PlacementFailed(format!(
                "failed to build opt_trans Rayon thread pool: {e}"
            ))
        })?;

    match cfg.init_strategy {
        super::config::InitStrategy::Centroid => {
            common::init_positions_center_drop(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
        _ => {
            common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        }
    }

    solve_pool.install(|| {
        cell_x.par_iter_mut().for_each(|v| *v -= network.x0 as f64);
        cell_y.par_iter_mut().for_each(|v| *v -= network.y0 as f64);
    });

    let max_x = (network.width - 1) as f64;
    let max_y = (network.height - 1) as f64;

    // Build per-cell-type valid tile positions and capacities.
    let type_aware = TypeAwarePlacement::build(ctx, &network);

    // Build per-cell resolved bucket, parallel to cell_x/cell_y.
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();

    // Snap initial positions to valid columns for each cell's type.
    for i in 0..n {
        let bucket = cell_buckets[i];
        cell_x[i] = type_aware.snap_x(bucket, cell_x[i]);
    }

    info!(
        "OptTrans placer: {} cells, {}x{} grid ({}x{} subtile), {} pipes",
        n,
        network.width,
        network.height,
        network.subtile_width(),
        network.subtile_height(),
        network.num_pipes(),
    );

    let mut resistance_model = ResistanceModel {
        congestion_exponent: 0.0, // computed dynamically from overflow
        interference_weight: cfg.interference_weight,
        timing_weight: cfg.timing_weight,
    };

    let mut best_chpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();
    let mut best_iter = 0usize;
    let mut laplacian = SparseMatrix::new(network.num_nodes());
    laplacian.reserve_off_diag(network.num_pipes());

    // Build AMG structure once from pipe network topology (sparsity pattern is constant).
    let pipe_pairs: Vec<(usize, usize)> = network
        .pipes
        .iter()
        .map(|p| {
            let (lo, hi) = if p.from < p.to {
                (p.from, p.to)
            } else {
                (p.to, p.from)
            };
            (lo, hi)
        })
        .collect();
    let amg_structure = AmgStructure::new(network.num_nodes(), &pipe_pairs);
    info!(
        "AMG structure: {} levels",
        amg_structure.num_levels(),
    );
    let mut amg_precond: Option<AmgPreconditioner> = None;

    // Adam optimizer: lr = step_scale controls tiles/iter for median cell.
    let mut adam = AdamOptimizer::new(2 * n, cfg.step_scale);
    let mut grad = vec![0.0; 2 * n];
    let mut delta = vec![0.0; 2 * n];
    let mut global_pressure = vec![0.0f64; network.num_nodes()];
    let _cg_batch_size = cfg.cg_batch_size.max(1);
    let mut prev_n_nodes = 0usize;
    for iter in 0..cfg.max_iters {
        let n_nodes = network.num_nodes();
        let t_iter = std::time::Instant::now();

        // Warm-start: reuse previous iteration's global_pressure as CG initial guess.
        // If the grid changed size, reinitialize to zeros.
        if n_nodes != prev_n_nodes {
            global_pressure.resize(n_nodes, 0.0);
            global_pressure.fill(0.0);
        }
        prev_n_nodes = n_nodes;

        // a. Compute adaptive congestion exponent from per-type tile overflow.
        //    For each cell type, count how many cells of that type occupy each tile,
        //    and divide by the number of compatible BELs at that tile.
        //    The max overflow across all types drives the congestion exponent.
        {
            // Count cells per type per tile.
            let mut type_tile_count: FxHashMap<IdString, FxHashMap<(i32, i32), u32>> =
                FxHashMap::default();
            for i in 0..n {
                let bucket = cell_buckets[i];
                let tx = cell_x[i].round() as i32;
                let ty = cell_y[i].round() as i32;
                *type_tile_count
                    .entry(bucket)
                    .or_default()
                    .entry((tx, ty))
                    .or_insert(0) += 1;
            }

            let mut max_overflow: f64 = 0.0;
            let mut n_overflow = 0usize;

            for (bucket, tile_counts) in &type_tile_count {
                let cap_map = match type_aware.tile_capacity.get(bucket) {
                    Some(m) => m,
                    None => continue,
                };
                for (&(tx, ty), &cnt) in tile_counts {
                    let cap = cap_map.get(&(tx, ty)).copied().unwrap_or(0);
                    if cap > 0 {
                        let ratio = cnt as f64 / cap as f64;
                        if ratio > max_overflow {
                            max_overflow = ratio;
                        }
                        if cnt > cap {
                            n_overflow += 1;
                        }
                    } else if cnt > 0 {
                        // Cells at a tile with zero capacity for their type: max overflow.
                        max_overflow = max_overflow.max(cnt as f64);
                        n_overflow += 1;
                    }
                }
            }

            // Exponent: 0 when legal, ramps with log of overflow, capped at 4.
            resistance_model.congestion_exponent = if max_overflow > 1.0 {
                (2.0 * max_overflow.ln()).min(4.0)
            } else {
                0.0
            };

            if iter % cfg.report_interval == 0 {
                eprintln!(
                    "  overflow: max={:.1}x tiles_over={} alpha={:.2}",
                    max_overflow, n_overflow, resistance_model.congestion_exponent,
                );
            }
        }

        // Build Kirchhoff Laplacian.
        // Parallel phase: compute conductances for all pipes
        laplacian.clear();
        solve_pool.install(|| {
            network.pipes.par_iter_mut().for_each(|pipe| {
                let r_eff = resistance_model.effective_resistance(pipe, 0.0);
                let conductance = 1.0 / r_eff.max(1e-12);
                pipe.eff_conductance = conductance;
            });
        });

        // Sequential assembly: add all connections to sparse matrix
        for pipe in &network.pipes {
            laplacian.add_connection(pipe.from, pipe.to, pipe.eff_conductance);
        }

        let avg_diag = laplacian.diag().iter().sum::<f64>() / n_nodes as f64;
        let epsilon = 1e-4 * avg_diag;
        for i in 0..n_nodes {
            laplacian.add_diagonal(i, epsilon);
        }

        let op = SparseMatrixOp::from_matrix(&mut laplacian);

        // AMG preconditioner: reuse structure, update numeric values each iteration.
        match amg_precond.as_mut() {
            Some(amg) => {
                amg.update_values(laplacian.diag(), laplacian.off_diag());
            }
            None => {
                amg_precond = Some(AmgPreconditioner::new(
                    amg_structure.clone(),
                    laplacian.diag(),
                    laplacian.off_diag(),
                ));
            }
        }
        let jacobi = crate::solver::preconditioner::JacobiPreconditioner::new(laplacian.diag());
        let precond = &jacobi;
        let scratch_req = cg_scratch_size(&op, precond);

        let net_infos =
            demand::collect_nets_for_solve(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        let n_nets = net_infos.len();
        grad.fill(0.0);

        {
            struct Accum {
                pressure: Vec<f64>,
                grad: Vec<f64>,
            }

            let accum = solve_pool.install(|| {
                net_infos
                    .par_iter()
                    .fold(
                        || Accum {
                            pressure: vec![0.0; n_nodes],
                            grad: vec![0.0; 2 * n],
                        },
                        |mut local, net_info| {
                            // Thread-local AMG + CG scratch.
                            thread_local! {
                                static TL: std::cell::RefCell<Option<(AmgPreconditioner, dyn_stack::MemBuffer)>> =
                                    const { std::cell::RefCell::new(None) };
                            }
                            let driver_rhs = demand::build_driver_rhs(net_info, &network, cfg);
                            let mut pressure = vec![0.0; n_nodes];
                            thread_local! {
                                static BUF: std::cell::RefCell<Option<dyn_stack::MemBuffer>> =
                                    const { std::cell::RefCell::new(None) };
                            }
                            BUF.with(|cell| {
                                let mut borrow = cell.borrow_mut();
                                let buf = borrow.get_or_insert_with(|| {
                                    dyn_stack::MemBuffer::new(scratch_req)
                                });
                                let rhs_mat = faer::MatRef::from_column_major_slice(
                                    &driver_rhs, n_nodes, 1,
                                );
                                let p_mat = faer::MatMut::from_column_major_slice_mut(
                                    &mut pressure, n_nodes, 1,
                                );
                                solve_cg_reuse(
                                    &op, precond, rhs_mat, p_mat,
                                    cfg.cg_tol, cfg.cg_max_iters, buf,
                                );
                            });

                            for (dst, &src) in local.pressure.iter_mut().zip(pressure.iter()) {
                                *dst += src;
                            }
                            let (gx, gy) = local.grad.split_at_mut(n);
                            demand::accumulate_energy_gradient(
                                net_info, &pressure, &network, cfg, 1.0, gx, gy,
                            );
                            local
                        },
                    )
                    .reduce(
                        || Accum {
                            pressure: vec![0.0; n_nodes],
                            grad: vec![0.0; 2 * n],
                        },
                        |mut a, b| {
                            for (dst, &src) in a.pressure.iter_mut().zip(b.pressure.iter()) {
                                *dst += src;
                            }
                            for (dst, &src) in a.grad.iter_mut().zip(b.grad.iter()) {
                                *dst += src;
                            }
                            a
                        },
                    )
            });

            global_pressure.copy_from_slice(&accum.pressure);
            grad.copy_from_slice(&accum.grad);

            // Update node pressures and pipe flows.
            solve_pool.install(|| {
                network
                    .nodes
                    .par_iter_mut()
                    .zip(global_pressure.par_iter())
                    .for_each(|(node, &p)| {
                        node.pressure = p;
                    });

                network.pipes.par_iter_mut().for_each(|pipe| {
                    let dp = global_pressure[pipe.from] - global_pressure[pipe.to];
                    pipe.flow = dp * pipe.eff_conductance;
                });
            });
        }

        // c. Adam step (raw gradient — Adam adapts per-cell from scale).
        adam.step(&grad, &mut delta);

        let (dx, dy) = delta.split_at(n);
        solve_pool.install(|| {
            cell_x
                .par_iter_mut()
                .zip(dx.par_iter())
                .for_each(|(x, &d)| *x += d);
            cell_y
                .par_iter_mut()
                .zip(dy.par_iter())
                .for_each(|(y, &d)| *y += d);
        });

        let disp_rms = ((delta.par_iter().map(|v| v * v).sum::<f64>()) / (2.0 * n as f64)).sqrt();

        // Snap positions to nearest valid column (and optionally row) for each cell's type.
        // Cells optimize within their type's valid tile set, so legalization
        // only needs to resolve within-tile overlaps (much smaller displacement).
        for i in 0..n {
            let bucket = cell_buckets[i];
            cell_x[i] = type_aware.snap_x(bucket, cell_x[i]);
            let snapped_x = cell_x[i].round() as i32;
            cell_y[i] = type_aware.snap_y(bucket, snapped_x, cell_y[i]);
        }

        common::clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);
        demand::update_net_counts(&cell_to_idx, &cell_x, &cell_y, &mut network, ctx);

        // d. Track best.
        let chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
        if chpwl < best_chpwl {
            best_chpwl = chpwl;
            best_x.copy_from_slice(&cell_x);
            best_y.copy_from_slice(&cell_y);
            best_iter = iter;
        }

        let iter_ms = t_iter.elapsed().as_millis();
        if iter % cfg.report_interval == 0 || iter == cfg.max_iters - 1 {
            let grad_norm = grad.par_iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "OT {:3}: chpwl={:.0} rms={:.2} grad={:.3e} {}ms nets={}",
                iter, chpwl, disp_rms, grad_norm, iter_ms, n_nets,
            );
        }

        if disp_rms < 0.01 && iter >= 10 {
            eprintln!("Converged at iter {} (rms={:.3e})", iter, disp_rms);
            break;
        }
    }

    cell_x.copy_from_slice(&best_x);
    cell_y.copy_from_slice(&best_y);
    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Best iter {}: chpwl={:.0}", best_iter, pre_chpwl);

    // Legalize.
    crate::placer::legalize::snap_to_clb_grid(
        ctx,
        &mut cell_x,
        &mut cell_y,
        &idx_to_cell,
        &network,
    );

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();

    match cfg.legalization.as_str() {
        "sorted" => {
            crate::placer::legalize::sorted_legalize(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
        "bipartite" => {
            let cost = Box::new(crate::placer::legalize::DistanceCost);
            crate::placer::legalize::bipartite::legalize_bipartite(
                ctx,
                &idx_to_cell,
                &phys_x,
                &phys_y,
                &*cost,
                cfg.lap_max_cells,
            )?;
        }
        "greedy" => {
            crate::placer::legalize::legalize_electro(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
        _ => {
            crate::placer::legalize::legalize_ring(ctx, &idx_to_cell, &phys_x, &phys_y)?;
        }
    }

    let post_hpwl = total_hpwl(ctx);
    let post_line = total_line_estimate(ctx);
    eprintln!(
        "Post-legalization: HPWL={:.0}, line={:.0}, delta={:+.0} ({:+.1}%)",
        post_hpwl,
        post_line,
        post_hpwl - pre_chpwl,
        (post_hpwl - pre_chpwl) / pre_chpwl.max(1.0) * 100.0,
    );

    PlacerPipeline::validate(ctx)?;
    info!("OptTrans placement complete");
    Ok(())
}

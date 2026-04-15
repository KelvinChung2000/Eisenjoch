//! Main Beckmann OT placement algorithm.
//!
//! Uses inner/outer decomposition: expensive Dijkstra solves in the outer loop
//! cache distance fields, cheap discrete coordinate descent in the inner loop
//! moves cells to minimum-cost grid nodes.

use crate::common::IdString;
use crate::context::Context;
use crate::placer::common;
use crate::placer::pipeline::PlacerPipeline;
use crate::placer::report;
use crate::placer::PlacerError;
use rayon::{prelude::*, ThreadPoolBuilder};

use crate::placer::common::TypeAwarePlacement;

use super::config::OptTransPlacerCfg;
use super::demand;
use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Main Beckmann OT placement using inner/outer coordinate descent.
pub fn place_opt_trans(ctx: &mut Context, cfg: &OptTransPlacerCfg) -> Result<(), PlacerError> {
    let mut cfg = cfg.clone();
    cfg.apply_env_overrides();
    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;
    crate::solver::set_solver_threads(cfg.num_threads);

    let target_scale = 1.0;
    let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
    let alive_net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();
    let n = idx_to_cell.len();
    if n == 0 {
        PlacerPipeline::validate(ctx)?;
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    let phys_max_x = (ctx.chipdb().width() - 1) as f64;
    let phys_max_y = (ctx.chipdb().height() - 1) as f64;
    let phys_grid_w = ctx.chipdb().width() as usize;
    let phys_grid_h = ctx.chipdb().height() as usize;
    let solve_pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads.max(1))
        .build()
        .map_err(|e| PlacerError::PlacementFailed(format!("thread pool: {e}")))?;
    let mut network = PipeNetwork::from_context(ctx, target_scale, &cfg);

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

    let type_aware = TypeAwarePlacement::build(ctx, network.x0, network.y0);
    let cell_buckets: Vec<IdString> = idx_to_cell
        .iter()
        .map(|&ci| ctx.resolve_bucket(ctx.design.cell(ci).cell_type))
        .collect();
    let cell_pin_weights: Vec<f64> = idx_to_cell
        .iter()
        .map(|&ci| {
            let cell = ctx.cell(ci);
            cell.ports()
                .filter(|pin| cell.port_net(pin.port).is_some())
                .count() as f64
        })
        .collect();

    let resistance_model = ResistanceModel;

    eprintln!(
        "Coordinate descent mode: {} cells, scale {:.2}",
        n, target_scale
    );

    super::coord_descent::run_inner_outer(
        ctx,
        &mut cell_x,
        &mut cell_y,
        &mut network,
        &cell_to_idx,
        &idx_to_cell,
        &alive_net_ids,
        &cfg,
        &solve_pool,
        &resistance_model,
        &type_aware,
        &cell_buckets,
        &cell_pin_weights,
        phys_max_x,
        phys_max_y,
        phys_grid_w,
        phys_grid_h,
    );

    let pre_chpwl = demand::continuous_hpwl(ctx, &cell_to_idx, &cell_x, &cell_y, &network);
    eprintln!("Final chpwl={:.0}", pre_chpwl);

    let phys_x: Vec<f64> = cell_x.iter().map(|x| x + network.x0 as f64).collect();
    let phys_y: Vec<f64> = cell_y.iter().map(|y| y + network.y0 as f64).collect();
    crate::placer::legalize::legalize(ctx, &idx_to_cell, &phys_x, &phys_y, &cfg.legalization)?;

    report::report_post_legalization(ctx, pre_chpwl);
    report::report_top_net_hpwl(ctx, 10);
    report::dump_position_csv_from_env(ctx, "NPNR_OT_DUMP_CSV", &idx_to_cell, &phys_x, &phys_y)
        .expect("dump placement CSV");

    PlacerPipeline::validate(ctx)?;
    Ok(())
}

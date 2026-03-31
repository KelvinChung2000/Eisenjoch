use crate::context::Context;
use crate::placer::PlacerError;
use crate::placer::PlacerPipeline;
use log::{debug, info};

use super::config::PlacerHeapCfg;
use super::state::HeapState;

/// Run the HeAP placer on the given context.
///
/// Steps:
/// 1. Initial random placement of all unplaced cells.
/// 2. Iteratively: solve quadratic system, spread, legalize.
/// 3. Stop when spreading quality exceeds threshold or max iterations reached.
pub fn place_heap(ctx: &mut Context, cfg: &PlacerHeapCfg) -> Result<(), PlacerError> {
    info!("HeAP Placer: starting...");

    PlacerPipeline::prepare_discrete(ctx, cfg.seed)?;

    let mut state = HeapState::new(ctx, cfg)?;

    let num_cells = state.movable_cells.len();
    if num_cells == 0 {
        info!("HeAP Placer: no moveable cells, nothing to do.");
        return Ok(());
    }
    info!("HeAP Placer: {} moveable cells.", num_cells);

    state.sync_positions_from_bels(ctx);
    info!("HeAP Placer: initial placement done.");

    // Track whether the solver has run with congestion targets at least once.
    // When convergence is reached before congestion forces have been applied,
    // we force one additional iteration so that congestion-aware placement is
    // reflected in the final result.
    let mut solver_used_congestion = false;

    for iter in 0..cfg.max_iterations {
        solver_used_congestion |= state.congestion_targets.is_some();
        state.solve_analytical(ctx)?;

        let quality = state.spread(ctx)?;
        state.legalize(ctx)?;

        // Compute congestion forces for the next iteration.
        if cfg.congestion_weight > 0.0 {
            state.congestion_targets = Some(state.compute_congestion_targets(ctx));
        }

        debug!(
            "HeAP Placer: iter={}, quality={:.4}, alpha={:.4}",
            iter, quality, state.alpha
        );

        if quality > cfg.spreading_threshold {
            if cfg.congestion_weight > 0.0 && !solver_used_congestion {
                // Force one more iteration so congestion targets are applied.
                state.alpha *= 1.5;
                continue;
            }
            info!(
                "HeAP Placer: converged at iteration {} (quality={:.4}).",
                iter, quality
            );
            break;
        }

        state.alpha *= 1.5;
    }

    // Final validation: check all alive cells are placed and region constraints hold.
    PlacerPipeline::validate(ctx)?;

    info!("HeAP Placer: done.");
    Ok(())
}

/// Count how many BELs fall within the given rectangular region.
pub fn count_bels_in_region(ctx: &Context, x0: i32, y0: i32, x1: i32, y1: i32) -> usize {
    let mut count = 0;
    for bel in ctx.bels() {
        let loc = bel.loc();
        if loc.x >= x0 && loc.x <= x1 && loc.y >= y0 && loc.y <= y1 {
            count += 1;
        }
    }
    count
}

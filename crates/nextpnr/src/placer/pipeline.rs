//! Shared pre/post phases for all placement algorithms.

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashMap;

use super::common;
use super::opt_trans::InitStrategy;
use super::PlacerError;

/// Setup data returned by [`PlacerPipeline::prepare`].
pub struct PlacerSetup {
    /// Map from CellId to contiguous index.
    pub cell_to_idx: FxHashMap<CellId, usize>,
    /// Map from contiguous index to CellId.
    pub idx_to_cell: Vec<CellId>,
    /// Initial X positions (from BEL locations).
    pub cell_x: Vec<f64>,
    /// Initial Y positions (from BEL locations).
    pub cell_y: Vec<f64>,
}

/// Common pre/post phases shared by all placement algorithms.
pub struct PlacerPipeline;

impl PlacerPipeline {
    /// Standard pre-placement: seed RNG, initial random placement, collect movable
    /// cells, and initialize continuous positions from placed BEL locations.
    pub fn prepare(ctx: &mut Context, seed: u64) -> Result<PlacerSetup, PlacerError> {
        ctx.reseed_rng(seed);
        common::initial_placement(ctx)?;
        common::lock_boundary_cells(ctx);
        let (cell_to_idx, idx_to_cell) = common::collect_movable_cells(ctx);
        let mut cell_x = vec![0.0; idx_to_cell.len()];
        let mut cell_y = vec![0.0; idx_to_cell.len()];
        common::init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);
        Ok(PlacerSetup {
            cell_to_idx,
            idx_to_cell,
            cell_x,
            cell_y,
        })
    }

    /// Lightweight init for discrete placers (SA) that don't need continuous positions.
    ///
    /// `initial_placement` shuffles every cell onto a random BEL of its type.
    /// `init_strategy.place_boundary_cells` then refines BEL choice for boundary
    /// cells (IOBs, clock buffers) according to the strategy: random strategies
    /// keep the shuffle, graph-aware strategies (Centroid / Topological)
    /// re-bind each boundary cell to a BEL informed by its connected logic.
    ///
    /// `NPNR_PIN_BOUNDARY` (default on) freezes boundary cells at that BEL so
    /// the OT solver cannot relocate them. Set `NPNR_PIN_BOUNDARY=0` to leave
    /// boundary cells movable.
    pub fn prepare_discrete(ctx: &mut Context, seed: u64) -> Result<(), PlacerError> {
        ctx.reseed_rng(seed);
        let strategy = InitStrategy::from_env_or(InitStrategy::Topological);
        strategy.initial_placement(ctx)?;
        let do_pin = std::env::var("NPNR_PIN_BOUNDARY")
            .ok()
            .map(|v| v != "0" && v.to_ascii_lowercase() != "false")
            .unwrap_or(true);
        if do_pin {
            common::lock_boundary_cells(ctx);
        } else {
            eprintln!("NPNR_PIN_BOUNDARY=0: boundary cells left movable");
        }
        Ok(())
    }

    /// Standard post-placement validation.
    pub fn validate(ctx: &Context) -> Result<(), PlacerError> {
        common::validate_all_placed(ctx)?;
        common::validate_region_constraints(ctx)?;
        Ok(())
    }
}

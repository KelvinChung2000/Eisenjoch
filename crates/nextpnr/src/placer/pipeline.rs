//! Shared pre/post phases for all placement algorithms.

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashMap;

use super::common;
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
    pub fn prepare_discrete(ctx: &mut Context, seed: u64) -> Result<(), PlacerError> {
        ctx.reseed_rng(seed);
        common::initial_placement(ctx)?;
        common::lock_boundary_cells(ctx);
        Ok(())
    }

    /// Standard post-placement validation.
    pub fn validate(ctx: &Context) -> Result<(), PlacerError> {
        common::validate_all_placed(ctx)?;
        common::validate_region_constraints(ctx)?;
        Ok(())
    }
}

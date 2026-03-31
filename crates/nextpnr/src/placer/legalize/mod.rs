//! Legalization: converting continuous cell positions to discrete BEL assignments.
//!
//! All legalizers implement the [`Legalizer`] trait, which takes continuous
//! (x, y) positions and produces valid BEL bindings.

pub mod bipartite;
pub mod common;
pub mod greedy;
pub mod ring;
pub mod snap;
pub mod sorted;

use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common::TypeAwarePlacement;
use crate::placer::PlacerError;

pub use bipartite::{BipartiteLegalizer, DistanceCost, LegalizeCost};
pub use greedy::{legalize_electro, GreedyLegalizer};
pub use ring::{legalize_ring, RingLegalizer};
pub use snap::{snap_to_clb_grid, SnapLegalizer};
pub use sorted::{sorted_legalize, SortedLegalizer};

/// Trait for converting continuous cell positions to discrete BEL assignments.
///
/// All implementations unbind movable cells, find the nearest valid BEL
/// for each cell, bind them, and handle cluster children.
/// Returns total squared displacement.
///
/// The `type_aware` parameter provides pre-built tile compatibility and capacity
/// information, shared with the snap step so both use a single source of truth.
pub trait Legalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
        type_aware: &TypeAwarePlacement,
    ) -> Result<f64, PlacerError>;
}

/// Snap + legalize in one pass: build `TypeAwarePlacement` once, snap positions
/// to nearest valid tiles, then run the legalization algorithm. Both steps
/// share the same type-aware placement data.
pub fn snap_and_legalize(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    legalizer: &dyn Legalizer,
) -> Result<f64, PlacerError> {
    let type_aware = TypeAwarePlacement::build(ctx, 0, 0);
    let (snapped_x, snapped_y) = snap_to_clb_grid(ctx, idx_to_cell, cell_x, cell_y, &type_aware);
    legalizer.legalize(ctx, idx_to_cell, &snapped_x, &snapped_y, &type_aware)
}

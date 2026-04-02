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

/// Create a legalizer by name.
///
/// Available strategies:
/// - `"snap"` (default): type-aware snap + spread + BEL assignment in one pass
/// - `"ring"`: Manhattan ring search (nearest available BEL)
/// - `"sorted"`: distance-sorted greedy assignment
/// - `"bipartite"`: optimal LAPJV bipartite matching
/// - `"greedy"`: simple nearest-BEL greedy
pub fn create_legalizer(name: &str) -> Box<dyn Legalizer> {
    match name {
        "ring" => Box::new(RingLegalizer {
            x_offset: 0.0,
            y_offset: 0.0,
        }),
        "sorted" => Box::new(SortedLegalizer),
        "bipartite" => Box::new(BipartiteLegalizer {
            cost: Box::new(DistanceCost),
            lap_max_cells: 10000,
        }),
        "greedy" => Box::new(GreedyLegalizer),
        _ => Box::new(SnapLegalizer),
    }
}

/// Legalize cell positions: build TypeAwarePlacement, then run the selected strategy.
///
/// `cell_x`/`cell_y` are physical coordinates. The legalizer snaps to valid tiles,
/// resolves overcrowding, and assigns cells to specific BELs.
pub fn legalize(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
    strategy: &str,
) -> Result<f64, PlacerError> {
    let type_aware = TypeAwarePlacement::build(ctx, 0, 0);
    let legalizer = create_legalizer(strategy);
    legalizer.legalize(ctx, idx_to_cell, cell_x, cell_y, &type_aware)
}

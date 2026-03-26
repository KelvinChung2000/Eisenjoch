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
use crate::placer::PlacerError;

pub use bipartite::{BipartiteLegalizer, DistanceCost, LegalizeCost};
pub use greedy::{legalize_electro, GreedyLegalizer};
pub use ring::{legalize_ring, RingLegalizer};
pub use snap::snap_to_clb_grid;
pub use sorted::{sorted_legalize, SortedLegalizer};

/// Trait for converting continuous cell positions to discrete BEL assignments.
///
/// All implementations unbind movable cells, find the nearest valid BEL
/// for each cell, bind them, and handle cluster children.
/// Returns total squared displacement.
pub trait Legalizer {
    fn legalize(
        &self,
        ctx: &mut Context,
        idx_to_cell: &[CellId],
        cell_x: &[f64],
        cell_y: &[f64],
    ) -> Result<f64, PlacerError>;
}

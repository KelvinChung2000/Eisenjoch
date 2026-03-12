//! Placer trait and implementations.

pub mod common;
pub mod electro_place;
pub mod heap;
pub mod hydraulic_place;
pub mod sa;
pub mod solver;

pub use electro_place::PlacerElectro;
pub use heap::PlacerHeap;
pub use hydraulic_place::PlacerHydraulic;
pub use sa::PlacerSa;

use crate::context::Context;
use crate::netlist::CellId;

/// Errors that can occur during placement.
#[derive(Debug, thiserror::Error)]
pub enum PlacerError {
    #[error("No valid BELs available for cell type {0}")]
    NoBelsAvailable(String),
    #[error("Placement failed: {0}")]
    PlacementFailed(String),
    #[error("Initial placement failed: could not place cell {0}")]
    InitialPlacementFailed(String),
}

/// Trait for placement algorithms.
pub trait Placer {
    type Config;

    /// Full placement of all unplaced cells.
    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError>;

    /// Place only the specified cells, treating all other placed cells as fixed.
    ///
    /// Default: returns error indicating incremental placement is not supported.
    /// Algorithms that naturally handle locked cells can delegate to `place()`.
    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let _ = (ctx, cfg, cells);
        Err(PlacerError::PlacementFailed(
            "incremental placement not supported by this algorithm".into(),
        ))
    }
}

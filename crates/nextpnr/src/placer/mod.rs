//! Placer trait and implementations.

pub mod common;
pub mod heap;
pub mod sa;

pub use heap::PlacerHeap;
pub use sa::PlacerSa;

use crate::context::Context;

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
    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError>;
}

//! Placer trait and implementations.

pub mod common;
pub mod electro_place;
pub mod heap;
pub mod legalize;
pub mod opt_trans;
pub mod pipeline;
pub mod report;
pub mod routability;
pub mod sa;

pub use electro_place::PlacerElectro;
pub use heap::PlacerHeap;
pub use legalize::Legalizer;
pub use opt_trans::PlacerOptTrans;
pub use pipeline::{PlacerPipeline, PlacerSetup};
pub use sa::PlacerSa;

use crate::context::Context;
use crate::netlist::CellId;

/// Resident set size of this process in KiB, or 0 if /proc is unreadable.
///
/// Used by the placer's memory diagnostics; peak RSS sampled from outside the
/// process cannot say which phase allocated, so the phases report it directly.
pub(crate) fn process_rss_kb() -> usize {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
        }
    }
    0
}

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

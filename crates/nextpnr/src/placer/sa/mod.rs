//! Simulated Annealing (SA) placer for FPGA cell placement.
//!
//! This implements the Placer1/SA algorithm: cells are initially placed at random
//! valid BELs, then iteratively improved by proposing random swap moves and
//! accepting or rejecting them via the Metropolis criterion. The cost function
//! combines HPWL (Half-Perimeter Wire Length) with optional congestion-awareness
//! via edge-based demand tracking and optional timing-driven weighting via net
//! criticality.

pub mod algorithm;
pub mod config;
pub mod congestion;
pub mod swap;

pub use algorithm::place_sa;
pub use config::PlacerSaCfg;
pub use congestion::CongestionCache;
pub use swap::{SwapResult, try_swap, revert_swap};

use crate::context::Context;
use crate::netlist::CellId;

/// Simulated annealing placer.
pub struct PlacerSa;

impl super::Placer for PlacerSa {
    type Config = PlacerSaCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_sa(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), super::PlacerError> {
        use rustc_hash::FxHashSet;
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        super::common::with_locked_others(ctx, &cells_set, |ctx| place_sa(ctx, cfg))
    }
}

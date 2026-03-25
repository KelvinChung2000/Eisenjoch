//! HeAP (Heterogeneous Analytical Placer) for FPGA cell placement.
//!
//! This implements the HeAP algorithm: cells are positioned by solving a
//! quadratic optimization problem (analytical placement), then spread to
//! reduce overlap via recursive bisection, and finally legalized by snapping
//! each cell to the nearest available BEL of matching type.
//!
//! The cost function minimizes squared wirelength. For each net, connections
//! between cell pairs contribute quadratic terms. Multi-pin nets use a star
//! model with a virtual node at the net centroid. Anchor forces pull cells
//! toward their current spread positions, growing stronger each iteration.

pub mod algorithm;
pub mod analytical;
pub mod config;
pub mod legalize;
pub mod spreading;
pub mod state;

#[cfg(feature = "test-utils")]
pub use algorithm::count_bels_in_region;
pub use algorithm::place_heap;
pub use config::PlacerHeapCfg;
pub use state::HeapState;

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashSet;

use super::common;

/// HeAP analytical placer.
pub struct PlacerHeap;

impl super::Placer for PlacerHeap {
    type Config = PlacerHeapCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::PlacerError> {
        place_heap(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), super::PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        common::with_locked_others(ctx, &cells_set, |ctx| place_heap(ctx, cfg))
    }
}

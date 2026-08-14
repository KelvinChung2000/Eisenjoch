//! Post-legalisation placement refinement.
//!
//! Port of nextpnr's `placer1_refine` (`common/place/placer1.cc:1267`), the
//! pass HeAP runs unconditionally once strict legalisation finishes
//! (`common/place/placer_heap.cc:402-412`).
//!
//! It matters more than its size suggests. Measured on the 20x20 comparison
//! fabric, this stage is worth 7.5% of HeAP's wirelength when LUT/FF pairs are
//! packed and 24.7% when they are not, and it cuts the timing objective by
//! 37-57% -- converging on the same cost from starts that differ by a factor of
//! two. Any comparison against nextpnr that stops at legalisation is measuring
//! a pipeline one stage short.

mod algorithm;
mod config;
mod cost;

pub use algorithm::refine_placement;
pub use config::{RefineCfg, RefineStats, RefineTiming};

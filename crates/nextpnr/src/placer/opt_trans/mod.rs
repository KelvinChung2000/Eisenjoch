//! Beckmann optimal transport placer.
//!
//! Models placement as a single Kirchhoff system where:
//! - Nets inject flow demand at cell positions (bilinear interpolation)
//! - Pressure drives flow through the pipe network
//! - Cells are Lagrangian particles moving along -grad(P)
//! - Flow interference between nets sharing pipes causes natural spreading
//! - Resistance sigma encodes ALL physics: congestion, interference, timing
//!
//! Uses Anderson-accelerated fixed-point iteration (~20-30 Kirchhoff solves).

mod algorithm;
pub mod config;
mod demand;
pub(crate) mod network;
mod resistance;

pub use algorithm::place_opt_trans;
pub use config::OptTransPlacerCfg;

use rustc_hash::FxHashSet;

use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common;
use crate::placer::PlacerError;

pub struct PlacerOptTrans;

impl super::Placer for PlacerOptTrans {
    type Config = OptTransPlacerCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_opt_trans(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let target: FxHashSet<CellId> = cells.iter().copied().collect();
        common::with_locked_others(ctx, &target, |ctx| place_opt_trans(ctx, cfg))
    }
}

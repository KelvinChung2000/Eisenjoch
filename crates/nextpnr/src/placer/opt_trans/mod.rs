//! Beckmann optimal transport placer.
//!
//! Uses per-net Dijkstra paths on the pipe network to compute the placement
//! gradient, then steps floating-point cell positions with Adam. Adam's
//! learning rate is adapted from an EMA of relative energy progress.

mod algorithm;
pub mod config;
mod demand;
pub(crate) mod network;
pub mod path_solver;
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

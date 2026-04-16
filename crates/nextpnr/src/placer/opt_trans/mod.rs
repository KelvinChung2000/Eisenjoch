//! Beckmann optimal transport placer.
//!
//! Uses per-net Dial-logit soft path costs over a BPR pipe network, then
//! moves cells with discrete coordinate descent.

mod algorithm;
pub mod config;
pub mod congestion;
mod coord_descent;
mod demand;
pub(crate) mod diag;
pub(crate) mod network;
pub mod fsm;
pub mod path_solver;
pub(crate) mod region_min;
mod resistance;
pub(crate) mod tile_cache;

pub use algorithm::place_opt_trans;
pub use config::{GraphModel, OptTransPlacerCfg, PathModel, SweepMode};

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

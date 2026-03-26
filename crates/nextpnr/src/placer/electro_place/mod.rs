//! ElectroPlace: RePlAce-style analytical placer aligned with placer_static.cc.
//!
//! Uses Nesterov accelerated gradient descent with:
//! - Weighted-Average (WA) smooth wirelength (no gamma annealing)
//! - DCT-based density penalty (Poisson field)
//! - Growing density penalty (multiplicative then additive)
//! - Overlap-based convergence (not HPWL stagnation)
//! - Spacer insertion to target utilization
//! - Barzilai-Borwein step size

mod algorithm;
pub mod config;
pub mod density;

pub use algorithm::place_electro;
pub use config::ElectroPlaceCfg;

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashSet;

use super::common::with_locked_others;
use super::PlacerError;

pub struct PlacerElectro;

impl super::Placer for PlacerElectro {
    type Config = ElectroPlaceCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_electro(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_electro(ctx, cfg))
    }
}

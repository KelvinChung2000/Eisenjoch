//! Optimal transport placer: minimum transport energy placement.
//!
//! Minimizes score = CHPWL(x) + λ_avg · E_density(x) where:
//!
//! - CHPWL: continuous half-perimeter wirelength from cell positions.
//!   Kirchhoff system LP = S gives equilibrium potentials P for demand S,
//!   providing pressure gradients that drive cells toward shorter routes.
//!   This is a convex relaxation of routing — flow splits across parallel paths,
//!   automatically distributing demand and revealing congestion gradients.
//!
//! - E_density = Σ (ρ-1)²: symmetric quadratic capacity deviation penalty.
//!   Per-tile Augmented Lagrangian multipliers enforce capacity constraints:
//!   λ[tile] grows at overcrowded tiles via fixed α=0.1 dual step.
//!
//! - Congestion: R_eff = R·(1 + β·(Q/C)²)·(1 + density_penalty) couples both
//!   flow-based turbulence and density overflow into pipe resistance, steering
//!   the Kirchhoff solver around congested regions (Benamou-Brenier coupling).
//!
//! The gradient ∂F/∂x drives cell motion via Adam optimizer.
//! Kirchhoff gradient (asymmetric: drivers↓, sinks↑) minimizes transport energy.
//! Density gradient (symmetric: always repulsive) spreads overcrowded cells.

mod algorithm;
mod boundary;
pub mod config;
mod helmholtz;
pub mod kirchhoff;
pub mod legalize;
pub mod network;
pub mod state;
pub mod timing;

pub use algorithm::place_opt_trans;
pub use config::OptTransPlacerCfg;

use crate::context::Context;
use crate::netlist::CellId;
use rustc_hash::FxHashSet;

use super::common::with_locked_others;
use super::PlacerError;

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
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_opt_trans(ctx, cfg))
    }
}

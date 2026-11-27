//! Router trait and implementations.

pub mod astar;
pub mod common;
pub mod lookahead;
pub mod maze;
pub mod raster;
pub mod router2;
pub mod traits;

pub use common::{NegotiationCfg, NegotiationState};
pub use maze::Router1;
pub use raster::RasterRouter;
pub use router2::Router2;
pub use traits::Router;

// ---------------------------------------------------------------------------
// Unified error type
// ---------------------------------------------------------------------------

/// Errors that can occur during routing.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// A* search could not find any path for the named net.
    #[error("Failed to route net {0}: no path found")]
    NoPath(String),
    /// Routing did not converge within the iteration limit.
    #[error("Routing failed after {0} iterations, {1} nets still congested")]
    Congestion(usize, usize),
    /// Post-route validation against the cell netlist failed.
    ///
    /// The payload is a multi-line diagnostic produced by
    /// [`common::validate_all_routed`].
    #[error("{0}")]
    ValidationFailed(String),
    /// A converged plan could not bind a wire in the context.
    #[error("committing net {net} was rejected at wire {wire}")]
    CommitRejected { net: String, wire: String },
    /// A negotiated wire cost left the integer range the A* kernel sums in.
    #[error("negotiation at iteration {iteration} prices a wire at {wire_cost}, beyond the A* cost range")]
    CostRange { iteration: usize, wire_cost: f64 },
    /// The timing-driven cost needs a criticality cap in [0, 1).
    #[error("max_criticality must lie in [0, 1), got {0}")]
    CriticalityCap(f64),
    /// The timing-driven cost needs a positive, finite criticality exponent.
    #[error("criticality_exponent must be positive and finite, got {0}")]
    CriticalityExponent(f64),
    /// Timing-driven routing found no defined clock, so every connection is unconstrained.
    #[error("timing-driven routing needs a defined clock and the design has none; define one with add_clock or an SDC create_clock, or route with timing off")]
    NoClock,
    /// The STA behind the timing-driven cost failed.
    #[error(transparent)]
    Timing(#[from] crate::sta::StaSessionError),
    /// The viewer cancelled the flow.
    #[error(transparent)]
    Cancelled(#[from] crate::live::Cancelled),
    /// Generic router error.
    #[error("Router error: {0}")]
    Generic(String),
}

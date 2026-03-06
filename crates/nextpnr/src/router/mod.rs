//! Router trait and implementations.

pub mod common;
pub mod router1;
pub mod router2;

pub use router1::Router1;
pub use router2::Router2;

use crate::context::Context;

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
    /// Generic router error.
    #[error("Router error: {0}")]
    Generic(String),
}

/// Trait for routing algorithms.
pub trait Router {
    type Config;
    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), RouterError>;
}

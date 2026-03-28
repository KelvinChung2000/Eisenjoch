//! Router trait and implementations.

pub mod common;
pub mod lookahead;
pub mod maze;
pub mod negotiation;
pub mod traits;

pub use maze::Router1;
pub use negotiation::Router2;
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
    /// Generic router error.
    #[error("Router error: {0}")]
    Generic(String),
}

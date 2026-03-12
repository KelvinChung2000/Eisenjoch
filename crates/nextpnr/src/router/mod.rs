//! Router trait and implementations.

pub mod common;
pub mod router1;
pub mod router2;

pub use router1::Router1;
pub use router2::Router2;

use crate::context::Context;
use crate::netlist::NetId;

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

    /// Full routing of all unrouted nets.
    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), RouterError>;

    /// Route a single net. Used by incremental flows.
    ///
    /// Default: returns error indicating incremental routing is not supported.
    fn route_net(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        net: NetId,
    ) -> Result<(), RouterError> {
        let _ = (ctx, cfg, net);
        Err(RouterError::Generic(
            "incremental routing not supported by this algorithm".into(),
        ))
    }

    /// Route a set of nets. Used by incremental flows.
    ///
    /// Default: calls `route_net()` for each net.
    fn route_nets(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        nets: &[NetId],
    ) -> Result<(), RouterError> {
        for &net in nets {
            self.route_net(ctx, cfg, net)?;
        }
        Ok(())
    }
}

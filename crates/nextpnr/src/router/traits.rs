//! Trait definition for routing algorithms.

use crate::context::Context;
use crate::netlist::NetId;

use super::RouterError;

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

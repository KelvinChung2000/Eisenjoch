//! Trait definitions for the packer, placer, and router plugin stages,
//! plus default (no-op) implementations.

use crate::chipdb::BelId;

use super::manager::{PluginContext, PluginError};

// ---------------------------------------------------------------------------
// Plugin traits
// ---------------------------------------------------------------------------

/// Packer plugin -- transforms netlist cells into arch-specific packed cells.
pub trait PackerPlugin {
    /// Run the packing pass over the design.
    fn pack(&mut self, ctx: &mut PluginContext) -> Result<(), PluginError>;
}

/// Placer plugin -- hooks into the placement flow.
pub trait PlacerPlugin {
    /// Called before placement begins.
    fn pre_place(&mut self, ctx: &mut PluginContext);

    /// Called after placement completes.
    fn post_place(&mut self, ctx: &mut PluginContext);

    /// Check whether placing a cell at `bel` is valid (beyond basic type
    /// matching). Returns `true` if the placement is acceptable.
    fn check_placement_validity(&self, ctx: &PluginContext, bel: BelId) -> bool;
}

/// Router plugin -- hooks into the routing flow.
pub trait RouterPlugin {
    /// Called before routing begins.
    fn pre_route(&mut self, ctx: &mut PluginContext);

    /// Called after routing completes.
    fn post_route(&mut self, ctx: &mut PluginContext);
}

// ---------------------------------------------------------------------------
// Default (no-op) implementations
// ---------------------------------------------------------------------------

/// Default packer that performs no transformations.
pub struct DefaultPacker;

impl PackerPlugin for DefaultPacker {
    fn pack(&mut self, _ctx: &mut PluginContext) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Default placer hooks that accept all placements and do nothing.
pub struct DefaultPlacerHooks;

impl PlacerPlugin for DefaultPlacerHooks {
    fn pre_place(&mut self, _ctx: &mut PluginContext) {}

    fn post_place(&mut self, _ctx: &mut PluginContext) {}

    fn check_placement_validity(&self, _ctx: &PluginContext, _bel: BelId) -> bool {
        true
    }
}

/// Default router hooks that do nothing.
pub struct DefaultRouterHooks;

impl RouterPlugin for DefaultRouterHooks {
    fn pre_route(&mut self, _ctx: &mut PluginContext) {}

    fn post_route(&mut self, _ctx: &mut PluginContext) {}
}

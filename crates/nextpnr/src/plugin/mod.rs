//! Plugin system for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module provides trait definitions for the three main CAD stages:
//! packing, placement, and routing. Plugins can implement these traits to
//! customize behavior. A [`PluginManager`] holds the active plugin for each
//! stage (defaulting to no-op implementations) and provides stub methods
//! for future native shared-library and Python plugin loading.

use std::path::Path;

use crate::chipdb::{BelId, ChipDb};
use crate::common::{IdString, IdStringPool};
use crate::netlist::Design;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for plugin operations.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// A general plugin error with a description.
    #[error("Plugin error: {0}")]
    Generic(String),

    /// An error that occurred while loading a plugin.
    #[error("Plugin load error: {0}")]
    LoadError(String),

    /// The requested plugin was not found.
    #[error("Plugin not found: {0}")]
    NotFound(String),
}

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
// PluginContext
// ---------------------------------------------------------------------------

/// Context provided to plugins, wrapping access to the chip database and
/// design data.
///
/// This is the "safe window" through which plugins interact with the rest of
/// the tool. Keeping the surface area small makes it easier to maintain ABI
/// compatibility when native shared-library plugins are introduced later.
pub struct PluginContext<'a> {
    design: &'a mut Design,
    chipdb: &'a ChipDb,
    id_pool: &'a IdStringPool,
}

impl<'a> PluginContext<'a> {
    /// Create a new plugin context.
    pub fn new(design: &'a mut Design, chipdb: &'a ChipDb, id_pool: &'a IdStringPool) -> Self {
        Self {
            design,
            chipdb,
            id_pool,
        }
    }

    /// Get an immutable reference to the design.
    pub fn design(&self) -> &Design {
        self.design
    }

    /// Get a mutable reference to the design.
    pub fn design_mut(&mut self) -> &mut Design {
        self.design
    }

    /// Get an immutable reference to the chip database.
    pub fn chipdb(&self) -> &ChipDb {
        self.chipdb
    }

    /// Intern a string, returning its [`IdString`] handle.
    pub fn id(&self, s: &str) -> IdString {
        self.id_pool.intern(s)
    }

    /// Look up the string corresponding to an [`IdString`] handle.
    ///
    /// Returns the interned string, or `None` if the handle is out of range.
    pub fn name_of(&self, id: IdString) -> Option<&str> {
        self.id_pool.lookup(id)
    }
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

// ---------------------------------------------------------------------------
// PluginManager
// ---------------------------------------------------------------------------

/// Manages the active packer, placer, and router plugins.
///
/// Defaults to no-op implementations for all three stages. Plugins can be
/// set directly (for Rust callers) or, in the future, loaded from shared
/// libraries or Python modules.
pub struct PluginManager {
    packer: Box<dyn PackerPlugin>,
    placer: Box<dyn PlacerPlugin>,
    router: Box<dyn RouterPlugin>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new manager with default (no-op) plugins for all stages.
    pub fn new() -> Self {
        Self {
            packer: Box::new(DefaultPacker),
            placer: Box::new(DefaultPlacerHooks),
            router: Box::new(DefaultRouterHooks),
        }
    }

    // -- Native plugin loading stubs --------------------------------------

    /// Load a packer plugin from a shared library path.
    ///
    /// Not yet implemented -- will return [`PluginError::LoadError`].
    pub fn load_packer(&mut self, _path: &Path) -> Result<(), PluginError> {
        Err(PluginError::LoadError(
            "Native plugin loading not yet implemented".into(),
        ))
    }

    /// Load a placer plugin from a shared library path.
    ///
    /// Not yet implemented -- will return [`PluginError::LoadError`].
    pub fn load_placer(&mut self, _path: &Path) -> Result<(), PluginError> {
        Err(PluginError::LoadError(
            "Native plugin loading not yet implemented".into(),
        ))
    }

    /// Load a router plugin from a shared library path.
    ///
    /// Not yet implemented -- will return [`PluginError::LoadError`].
    pub fn load_router(&mut self, _path: &Path) -> Result<(), PluginError> {
        Err(PluginError::LoadError(
            "Native plugin loading not yet implemented".into(),
        ))
    }

    // -- Direct setters ---------------------------------------------------

    /// Replace the active packer plugin.
    pub fn set_packer(&mut self, packer: Box<dyn PackerPlugin>) {
        self.packer = packer;
    }

    /// Replace the active placer plugin.
    pub fn set_placer(&mut self, placer: Box<dyn PlacerPlugin>) {
        self.placer = placer;
    }

    /// Replace the active router plugin.
    pub fn set_router(&mut self, router: Box<dyn RouterPlugin>) {
        self.router = router;
    }

    // -- Accessors --------------------------------------------------------

    /// Get an immutable reference to the active packer plugin.
    pub fn packer(&self) -> &dyn PackerPlugin {
        self.packer.as_ref()
    }

    /// Get a mutable reference to the active packer plugin.
    pub fn packer_mut(&mut self) -> &mut dyn PackerPlugin {
        self.packer.as_mut()
    }

    /// Get an immutable reference to the active placer plugin.
    pub fn placer(&self) -> &dyn PlacerPlugin {
        self.placer.as_ref()
    }

    /// Get a mutable reference to the active placer plugin.
    pub fn placer_mut(&mut self) -> &mut dyn PlacerPlugin {
        self.placer.as_mut()
    }

    /// Get an immutable reference to the active router plugin.
    pub fn router(&self) -> &dyn RouterPlugin {
        self.router.as_ref()
    }

    /// Get a mutable reference to the active router plugin.
    pub fn router_mut(&mut self) -> &mut dyn RouterPlugin {
        self.router.as_mut()
    }
}

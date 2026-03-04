//! Plugin system for the nextpnr-rust FPGA place-and-route tool.
//!
//! This crate provides trait definitions for the three main CAD stages:
//! packing, placement, and routing. Plugins can implement these traits to
//! customize behavior. A [`PluginManager`] holds the active plugin for each
//! stage (defaulting to no-op implementations) and provides stub methods
//! for future native shared-library and Python plugin loading.

use std::path::Path;

use npnr_chipdb::ChipDb;
use npnr_netlist::Design;
use npnr_types::{BelId, IdString, IdStringPool};

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
    pub fn name_of(&self, id: IdString) -> Option<String> {
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
        &*self.packer
    }

    /// Get a mutable reference to the active packer plugin.
    pub fn packer_mut(&mut self) -> &mut dyn PackerPlugin {
        &mut *self.packer
    }

    /// Get an immutable reference to the active placer plugin.
    pub fn placer(&self) -> &dyn PlacerPlugin {
        &*self.placer
    }

    /// Get a mutable reference to the active placer plugin.
    pub fn placer_mut(&mut self) -> &mut dyn PlacerPlugin {
        &mut *self.placer
    }

    /// Get an immutable reference to the active router plugin.
    pub fn router(&self) -> &dyn RouterPlugin {
        &*self.router
    }

    /// Get a mutable reference to the active router plugin.
    pub fn router_mut(&mut self) -> &mut dyn RouterPlugin {
        &mut *self.router
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_netlist::Design;
    use npnr_types::IdStringPool;

    // -- Helper: build a minimal chipdb in memory -------------------------

    /// Create a minimal, valid ChipDb for testing.
    ///
    /// This relies on npnr-chipdb's `test-utils` feature and its `testutil`
    /// module which provides helpers for constructing in-memory chip databases.
    fn make_test_chipdb() -> ChipDb {
        npnr_chipdb::testutil::make_test_chipdb()
    }

    // -- PluginError tests ------------------------------------------------

    #[test]
    fn plugin_error_generic_display() {
        let err = PluginError::Generic("something went wrong".into());
        assert_eq!(err.to_string(), "Plugin error: something went wrong");
    }

    #[test]
    fn plugin_error_load_display() {
        let err = PluginError::LoadError("missing symbol".into());
        assert_eq!(err.to_string(), "Plugin load error: missing symbol");
    }

    #[test]
    fn plugin_error_not_found_display() {
        let err = PluginError::NotFound("my_plugin.so".into());
        assert_eq!(err.to_string(), "Plugin not found: my_plugin.so");
    }

    #[test]
    fn plugin_error_debug() {
        let err = PluginError::Generic("test".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Generic"));
        assert!(debug.contains("test"));
    }

    // -- PluginContext tests ----------------------------------------------

    #[test]
    fn plugin_context_design_access() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        design.top_module = pool.intern("top");

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        assert_eq!(ctx.design().top_module, pool.intern("top"));
    }

    #[test]
    fn plugin_context_design_mut_access() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let new_top = pool.intern("new_top");
        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            ctx.design_mut().top_module = new_top;
        }
        assert_eq!(design.top_module, new_top);
    }

    #[test]
    fn plugin_context_chipdb_access() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        // Just verify we can call a method on the chipdb without panicking.
        let _name = ctx.chipdb().name();
    }

    #[test]
    fn plugin_context_id_interning() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        let id = ctx.id("my_cell");
        assert!(!id.is_empty());

        // Same string yields same id.
        let id2 = ctx.id("my_cell");
        assert_eq!(id, id2);

        // Different string yields different id.
        let id3 = ctx.id("other_cell");
        assert_ne!(id, id3);
    }

    #[test]
    fn plugin_context_name_of() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        let id = ctx.id("hello");
        assert_eq!(ctx.name_of(id).as_deref(), Some("hello"));
    }

    #[test]
    fn plugin_context_name_of_empty() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        assert_eq!(ctx.name_of(IdString::EMPTY).as_deref(), Some(""));
    }

    #[test]
    fn plugin_context_name_of_invalid() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        assert_eq!(ctx.name_of(IdString(9999)), None);
    }

    // -- DefaultPacker tests ----------------------------------------------

    #[test]
    fn default_packer_returns_ok() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut packer = DefaultPacker;
        assert!(packer.pack(&mut ctx).is_ok());
    }

    // -- DefaultPlacerHooks tests -----------------------------------------

    #[test]
    fn default_placer_hooks_pre_place() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut placer = DefaultPlacerHooks;
        // Should not panic.
        placer.pre_place(&mut ctx);
    }

    #[test]
    fn default_placer_hooks_post_place() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut placer = DefaultPlacerHooks;
        placer.post_place(&mut ctx);
    }

    #[test]
    fn default_placer_hooks_validity_always_true() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let placer = DefaultPlacerHooks;
        // Should return true for any bel.
        assert!(placer.check_placement_validity(&ctx, BelId::new(0, 0)));
        assert!(placer.check_placement_validity(&ctx, BelId::INVALID));
        assert!(placer.check_placement_validity(&ctx, BelId::new(100, 200)));
    }

    // -- DefaultRouterHooks tests -----------------------------------------

    #[test]
    fn default_router_hooks_pre_route() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut router = DefaultRouterHooks;
        router.pre_route(&mut ctx);
    }

    #[test]
    fn default_router_hooks_post_route() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut router = DefaultRouterHooks;
        router.post_route(&mut ctx);
    }

    // -- PluginManager tests ----------------------------------------------

    #[test]
    fn manager_default_construction() {
        let _mgr = PluginManager::new();
    }

    #[test]
    fn manager_default_trait() {
        let _mgr = PluginManager::default();
    }

    #[test]
    fn manager_load_packer_stub_returns_error() {
        let mut mgr = PluginManager::new();
        let result = mgr.load_packer(Path::new("nonexistent.so"));
        assert!(result.is_err());
        match result.unwrap_err() {
            PluginError::LoadError(msg) => {
                assert!(msg.contains("not yet implemented"));
            }
            other => panic!("Expected LoadError, got {:?}", other),
        }
    }

    #[test]
    fn manager_load_placer_stub_returns_error() {
        let mut mgr = PluginManager::new();
        let result = mgr.load_placer(Path::new("nonexistent.so"));
        assert!(result.is_err());
        match result.unwrap_err() {
            PluginError::LoadError(msg) => {
                assert!(msg.contains("not yet implemented"));
            }
            other => panic!("Expected LoadError, got {:?}", other),
        }
    }

    #[test]
    fn manager_load_router_stub_returns_error() {
        let mut mgr = PluginManager::new();
        let result = mgr.load_router(Path::new("nonexistent.so"));
        assert!(result.is_err());
        match result.unwrap_err() {
            PluginError::LoadError(msg) => {
                assert!(msg.contains("not yet implemented"));
            }
            other => panic!("Expected LoadError, got {:?}", other),
        }
    }

    #[test]
    fn manager_default_packer_works() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut mgr = PluginManager::new();
        assert!(mgr.packer_mut().pack(&mut ctx).is_ok());
    }

    #[test]
    fn manager_default_placer_works() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut mgr = PluginManager::new();
        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            mgr.placer_mut().pre_place(&mut ctx);
        }
        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            mgr.placer_mut().post_place(&mut ctx);
        }
        {
            let ctx = PluginContext::new(&mut design, &chipdb, &pool);
            assert!(mgr.placer().check_placement_validity(&ctx, BelId::new(0, 0)));
        }
    }

    #[test]
    fn manager_default_router_works() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut mgr = PluginManager::new();
        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            mgr.router_mut().pre_route(&mut ctx);
        }
        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            mgr.router_mut().post_route(&mut ctx);
        }
    }

    // -- Custom plugin tests (verifies trait object dispatch) ---------------

    /// A test packer that records whether it was called.
    struct TestPacker {
        called: bool,
    }

    impl TestPacker {
        fn new() -> Self {
            Self { called: false }
        }
    }

    impl PackerPlugin for TestPacker {
        fn pack(&mut self, _ctx: &mut PluginContext) -> Result<(), PluginError> {
            self.called = true;
            Ok(())
        }
    }

    /// A test packer that always fails.
    struct FailingPacker;

    impl PackerPlugin for FailingPacker {
        fn pack(&mut self, _ctx: &mut PluginContext) -> Result<(), PluginError> {
            Err(PluginError::Generic("packing failed".into()))
        }
    }

    /// A test placer that tracks calls.
    struct TestPlacerHooks {
        pre_called: bool,
        post_called: bool,
        validity_result: bool,
    }

    impl TestPlacerHooks {
        fn new(validity_result: bool) -> Self {
            Self {
                pre_called: false,
                post_called: false,
                validity_result,
            }
        }
    }

    impl PlacerPlugin for TestPlacerHooks {
        fn pre_place(&mut self, _ctx: &mut PluginContext) {
            self.pre_called = true;
        }

        fn post_place(&mut self, _ctx: &mut PluginContext) {
            self.post_called = true;
        }

        fn check_placement_validity(&self, _ctx: &PluginContext, _bel: BelId) -> bool {
            self.validity_result
        }
    }

    /// A test router that tracks calls.
    struct TestRouterHooks {
        pre_called: bool,
        post_called: bool,
    }

    impl TestRouterHooks {
        fn new() -> Self {
            Self {
                pre_called: false,
                post_called: false,
            }
        }
    }

    impl RouterPlugin for TestRouterHooks {
        fn pre_route(&mut self, _ctx: &mut PluginContext) {
            self.pre_called = true;
        }

        fn post_route(&mut self, _ctx: &mut PluginContext) {
            self.post_called = true;
        }
    }

    #[test]
    fn custom_packer_is_called() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut packer = TestPacker::new();
        assert!(!packer.called);
        packer.pack(&mut ctx).unwrap();
        assert!(packer.called);
    }

    #[test]
    fn failing_packer_returns_error() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut packer = FailingPacker;
        let result = packer.pack(&mut ctx);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Plugin error: packing failed");
    }

    #[test]
    fn custom_placer_hooks_are_called() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut placer = TestPlacerHooks::new(false);
        assert!(!placer.pre_called);
        assert!(!placer.post_called);

        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            placer.pre_place(&mut ctx);
        }
        assert!(placer.pre_called);
        assert!(!placer.post_called);

        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            placer.post_place(&mut ctx);
        }
        assert!(placer.post_called);
    }

    #[test]
    fn custom_placer_validity_check() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let accept = TestPlacerHooks::new(true);
        assert!(accept.check_placement_validity(&ctx, BelId::new(0, 0)));

        let reject = TestPlacerHooks::new(false);
        assert!(!reject.check_placement_validity(&ctx, BelId::new(0, 0)));
    }

    #[test]
    fn custom_router_hooks_are_called() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut router = TestRouterHooks::new();
        assert!(!router.pre_called);
        assert!(!router.post_called);

        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            router.pre_route(&mut ctx);
        }
        assert!(router.pre_called);

        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            router.post_route(&mut ctx);
        }
        assert!(router.post_called);
    }

    #[test]
    fn manager_set_custom_packer() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        let mut mgr = PluginManager::new();
        mgr.set_packer(Box::new(FailingPacker));

        let result = mgr.packer_mut().pack(&mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn manager_set_custom_placer() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut mgr = PluginManager::new();
        mgr.set_placer(Box::new(TestPlacerHooks::new(false)));

        let ctx = PluginContext::new(&mut design, &chipdb, &pool);
        assert!(!mgr.placer().check_placement_validity(&ctx, BelId::new(0, 0)));
    }

    #[test]
    fn manager_set_custom_router() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let mut mgr = PluginManager::new();
        mgr.set_router(Box::new(TestRouterHooks::new()));

        {
            let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);
            mgr.router_mut().pre_route(&mut ctx);
        }
        // We cannot inspect the TestRouterHooks through the Box<dyn>, but at
        // least we verified it does not panic. For thorough checking, the
        // standalone custom_router_hooks_are_called test covers call tracking.
    }

    #[test]
    fn manager_replace_plugin_multiple_times() {
        let mut mgr = PluginManager::new();

        // Replace packer several times -- each old plugin is dropped.
        mgr.set_packer(Box::new(DefaultPacker));
        mgr.set_packer(Box::new(FailingPacker));
        mgr.set_packer(Box::new(DefaultPacker));

        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

        // The last one set was DefaultPacker, so pack should succeed.
        assert!(mgr.packer_mut().pack(&mut ctx).is_ok());
    }
}

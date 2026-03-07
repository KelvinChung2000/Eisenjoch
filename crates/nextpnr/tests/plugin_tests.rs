//! Integration tests for the plugin module.
//!
//! Tests for the plugin error types, PluginContext, default plugin
//! implementations, PluginManager, and custom plugin dispatch.

use std::path::Path;

use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::chipdb::{BelId, ChipDb};
use nextpnr::common::{IdString, IdStringPool};
use nextpnr::netlist::Design;
use nextpnr::plugin::{
    DefaultPacker, DefaultPlacerHooks, DefaultRouterHooks, PackerPlugin, PlacerPlugin,
    PluginContext, PluginError, PluginManager, RouterPlugin,
};

fn make_test_chipdb_helper() -> ChipDb {
    make_test_chipdb()
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    design.top_module = pool.intern("top");

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    assert_eq!(ctx.design().top_module, pool.intern("top"));
}

#[test]
fn plugin_context_design_mut_access() {
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    // Just verify we can call a method on the chipdb without panicking.
    let _name = ctx.chipdb().name();
}

#[test]
fn plugin_context_id_interning() {
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    let id = ctx.id("hello");
    assert_eq!(ctx.name_of(id), Some("hello"));
}

#[test]
fn plugin_context_name_of_empty() {
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    assert_eq!(ctx.name_of(IdString::EMPTY), Some(""));
}

#[test]
fn plugin_context_name_of_invalid() {
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    assert_eq!(ctx.name_of(IdString(9999)), None);
}

// -- DefaultPacker tests ----------------------------------------------

#[test]
fn default_packer_returns_ok() {
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    let mut packer = DefaultPacker;
    assert!(packer.pack(&mut ctx).is_ok());
}

// -- DefaultPlacerHooks tests -----------------------------------------

#[test]
fn default_placer_hooks_pre_place() {
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    let mut placer = DefaultPlacerHooks;
    // Should not panic.
    placer.pre_place(&mut ctx);
}

#[test]
fn default_placer_hooks_post_place() {
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    let mut placer = DefaultPlacerHooks;
    placer.post_place(&mut ctx);
}

#[test]
fn default_placer_hooks_validity_always_true() {
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    let mut router = DefaultRouterHooks;
    router.pre_route(&mut ctx);
}

#[test]
fn default_router_hooks_post_route() {
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    let mut mgr = PluginManager::new();
    assert!(mgr.packer_mut().pack(&mut ctx).is_ok());
}

#[test]
fn manager_default_placer_works() {
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
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
    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();

    let mut mgr = PluginManager::new();
    mgr.set_placer(Box::new(TestPlacerHooks::new(false)));

    let ctx = PluginContext::new(&mut design, &chipdb, &pool);
    assert!(!mgr.placer().check_placement_validity(&ctx, BelId::new(0, 0)));
}

#[test]
fn manager_set_custom_router() {
    let chipdb = make_test_chipdb_helper();
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

    let chipdb = make_test_chipdb_helper();
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let mut ctx = PluginContext::new(&mut design, &chipdb, &pool);

    // The last one set was DefaultPacker, so pack should succeed.
    assert!(mgr.packer_mut().pack(&mut ctx).is_ok());
}

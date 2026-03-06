//! Integration tests for the packer module (public API only).

use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::context::Context;
use nextpnr::packer::{pack, PackerError};
use nextpnr::plugin::{PackerPlugin, PluginContext, PluginError};

fn make_test_ctx() -> Context {
    let chipdb = make_test_chipdb();
    Context::new(chipdb)
}

// =====================================================================
// PackerError tests
// =====================================================================

#[test]
fn packer_error_generic_display() {
    let err = PackerError::Generic("something broke".into());
    assert_eq!(err.to_string(), "Packer error: something broke");
}

#[test]
fn packer_error_unsupported_cell_type_display() {
    let err = PackerError::UnsupportedCellType("WEIRD_CELL".into());
    assert_eq!(err.to_string(), "Unsupported cell type: WEIRD_CELL");
}

#[test]
fn packer_error_plugin_display() {
    let plugin_err = PluginError::Generic("plugin broke".into());
    let err = PackerError::Plugin(plugin_err);
    assert_eq!(err.to_string(), "Plugin error: Plugin error: plugin broke");
}

#[test]
fn packer_error_from_plugin_error() {
    let plugin_err = PluginError::Generic("test".into());
    let packer_err: PackerError = plugin_err.into();
    match packer_err {
        PackerError::Plugin(_) => {}
        other => panic!("Expected Plugin variant, got {:?}", other),
    }
}

// =====================================================================
// Plugin delegation tests
// =====================================================================

struct TrackingPacker {
    called: bool,
}

impl TrackingPacker {
    fn new() -> Self {
        Self { called: false }
    }
}

impl PackerPlugin for TrackingPacker {
    fn pack(&mut self, _ctx: &mut PluginContext) -> Result<(), PluginError> {
        self.called = true;
        Ok(())
    }
}

struct FailingPacker;

impl PackerPlugin for FailingPacker {
    fn pack(&mut self, _ctx: &mut PluginContext) -> Result<(), PluginError> {
        Err(PluginError::Generic("intentional failure".into()))
    }
}

#[test]
fn pack_delegates_to_plugin() {
    let mut ctx = make_test_ctx();
    let mut packer = TrackingPacker::new();
    let result = pack(&mut ctx, Some(&mut packer));
    assert!(result.is_ok());
    assert!(packer.called);
}

#[test]
fn pack_plugin_error_is_propagated() {
    let mut ctx = make_test_ctx();
    let mut packer = FailingPacker;
    let result = pack(&mut ctx, Some(&mut packer));
    assert!(result.is_err());
    match result.unwrap_err() {
        PackerError::Plugin(_) => {}
        other => panic!("Expected Plugin variant, got {:?}", other),
    }
}

#[test]
fn pack_without_plugin_uses_default() {
    let mut ctx = make_test_ctx();
    let result = pack(&mut ctx, None);
    assert!(result.is_ok());
}

//! Integration tests for the packer module (public API only).

mod common;

use nextpnr::packer::{pack, PackerError};
use nextpnr::plugin::{PackerPlugin, PluginContext, PluginError};

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
    assert_eq!(err.to_string(), "Plugin error: plugin broke");
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
    let mut ctx = common::make_context();
    let mut packer = TrackingPacker::new();
    let result = pack(&mut ctx, Some(&mut packer));
    assert!(result.is_ok());
    assert!(packer.called);
}

#[test]
fn pack_plugin_error_is_propagated() {
    let mut ctx = common::make_context();
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
    let mut ctx = common::make_context();
    let result = pack(&mut ctx, None);
    assert!(result.is_ok());
}

// =====================================================================
// Full pack integration tests
// =====================================================================

#[test]
fn full_pack_default_on_empty_design() {
    let mut ctx = common::make_context();
    assert!(nextpnr::packer::pack_default(&mut ctx).is_ok());
}

#[test]
fn full_pack_default_on_example_chipdb() {
    let mut ctx = common::make_example_context();
    assert!(nextpnr::packer::pack_default(&mut ctx).is_ok());
}

// =====================================================================
// remove_nextpnr_iobs
// =====================================================================
//
// Port of nextpnr's `HimbaechelHelpers::remove_nextpnr_iobs`, used when
// synthesis already inserted real IO buffers. Distinct from `pack_io`, which
// remaps the pseudo-cell onto an IOB bel instead of deleting it.

use nextpnr::context::Context;
use nextpnr::netlist::{CellPin, PortType};
use nextpnr::packer::passes::remove_nextpnr_iobs;

/// Build `pseudo -> real` (input side) or `real -> pseudo` (output side),
/// returning (pseudo cell, real cell, net, index of the user slot if any).
fn wire_pseudo_iob(
    ctx: &mut Context,
    pseudo_type: &str,
    pseudo_port: &str,
    pseudo_drives: bool,
) -> (nextpnr::netlist::CellId, nextpnr::netlist::CellId, nextpnr::netlist::NetId, Option<u32>) {
    let (pt, pp) = (ctx.id(pseudo_type), ctx.id(pseudo_port));
    let (rt, rp) = (ctx.id("INBUF"), ctx.id("PAD"));
    let net = ctx.design.add_net(ctx.id("pad_net"));

    let pseudo = ctx.design.add_cell(ctx.id("$io$pad"), pt);
    let real = ctx.design.add_cell(ctx.id("real_buf"), rt);

    if pseudo_drives {
        ctx.design.cell_edit(pseudo).add_port(pp, PortType::Out).set_port_net(pp, Some(net), None);
        ctx.design.net_edit(net).set_driver(pseudo, pp);
        let u = ctx.design.net_edit(net).add_user(real, rp);
        ctx.design.cell_edit(real).add_port(rp, PortType::In).set_port_net(rp, Some(net), Some(u));
        (pseudo, real, net, None)
    } else {
        ctx.design.cell_edit(real).add_port(rp, PortType::Out).set_port_net(rp, Some(net), None);
        ctx.design.net_edit(net).set_driver(real, rp);
        let u = ctx.design.net_edit(net).add_user(pseudo, pp);
        ctx.design.cell_edit(pseudo).add_port(pp, PortType::In).set_port_net(pp, Some(net), Some(u));
        (pseudo, real, net, Some(u))
    }
}

#[test]
fn remove_nextpnr_iobs_trims_every_pseudo_type() {
    // (type, port, does the pseudo-cell drive the pad net?)
    let cases = [
        ("$nextpnr_IBUF", "O", true),
        ("$nextpnr_OBUF", "I", false),
        ("$nextpnr_IOBUF", "IO", false),
    ];

    for (ty, port, drives) in cases {
        let mut ctx = common::make_context();
        let (pseudo, real, net, user) = wire_pseudo_iob(&mut ctx, ty, port, drives);

        assert_eq!(remove_nextpnr_iobs(&mut ctx).unwrap(), 1, "{ty} should be trimmed");
        assert!(!ctx.design.cell(pseudo).alive, "{ty} cell must be dead");
        assert!(ctx.design.cell(real).alive, "{ty}: the real buffer must survive");

        if drives {
            assert!(
                ctx.design.net(net).driver().is_none_or(|d| d.cell != pseudo),
                "{ty}: the pad net must no longer be driven by the pseudo-cell"
            );
        } else {
            let u = user.expect("sink case records a user slot");
            assert_eq!(
                ctx.design.net(net).users()[u as usize],
                CellPin::INVALID,
                "{ty}: the pseudo-cell's user slot must be invalidated"
            );
            assert_eq!(
                ctx.design.net(net).driver().map(|d| d.cell),
                Some(real),
                "{ty}: the real driver must be untouched"
            );
        }
    }
}

#[test]
fn remove_nextpnr_iobs_is_idempotent() {
    let mut ctx = common::make_context();
    wire_pseudo_iob(&mut ctx, "$nextpnr_IBUF", "O", true);
    assert_eq!(remove_nextpnr_iobs(&mut ctx).unwrap(), 1);
    assert_eq!(
        remove_nextpnr_iobs(&mut ctx).unwrap(),
        0,
        "a second pass must find nothing left to trim"
    );
}

#[test]
fn remove_nextpnr_iobs_leaves_real_cells_alone() {
    let mut ctx = common::make_context();
    let lut = ctx.design.add_cell(ctx.id("lut0"), ctx.id("LUT4"));
    assert_eq!(remove_nextpnr_iobs(&mut ctx).unwrap(), 0);
    assert!(ctx.design.cell(lut).alive);
}

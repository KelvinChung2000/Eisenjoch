mod common;

use nextpnr::packer::helpers::{connect_port, disconnect_port, get_net_for_port, is_single_fanout};
use nextpnr::packer::{pack_default, passes};
use nextpnr::types::PortType;

fn setup_simple_ctx(ctx: &mut nextpnr::context::Context) {
    let cell_name = ctx.id("my_cell");
    let cell_type = ctx.id("LUT4");
    let port_o = ctx.id("O");
    let port_i = ctx.id("I");
    let net_name = ctx.id("my_net");
    let cell_idx = ctx.design.add_cell(cell_name, cell_type);
    let net_idx = ctx.design.add_net(net_name);
    ctx.design.cell_edit(cell_idx).add_port(port_o, PortType::Out);
    connect_port(ctx, cell_idx, port_o, net_idx);
    ctx.design.cell_edit(cell_idx).add_port(port_i, PortType::In);
}

#[test]
fn get_net_for_port_connected() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    let net_idx = get_net_for_port(&ctx, cell_idx, ctx.id("O")).unwrap();
    assert_eq!(ctx.design.net(net_idx).name, ctx.id("my_net"));
}

#[test]
fn get_net_for_port_unconnected() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    assert!(get_net_for_port(&ctx, cell_idx, ctx.id("I")).is_none());
}

#[test]
fn get_net_for_port_nonexistent_port() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    assert!(get_net_for_port(&ctx, cell_idx, ctx.id("NONEXISTENT")).is_none());
}

#[test]
fn disconnect_port_removes_connection() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    let port_o = ctx.id("O");
    disconnect_port(&mut ctx, cell_idx, port_o);
    assert!(get_net_for_port(&ctx, cell_idx, port_o).is_none());
}

#[test]
fn disconnect_port_nonexistent_is_noop() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    let nonexistent = ctx.id("NONEXISTENT");
    disconnect_port(&mut ctx, cell_idx, nonexistent);
}

#[test]
fn connect_port_to_net() {
    let mut ctx = common::make_context();
    let cell_idx = ctx.design.add_cell(ctx.id("c1"), ctx.id("FF"));
    let net_idx = ctx.design.add_net(ctx.id("n1"));
    let port_d = ctx.id("D");
    ctx.design.cell_edit(cell_idx).add_port(port_d, PortType::In);
    connect_port(&mut ctx, cell_idx, port_d, net_idx);
    assert_eq!(get_net_for_port(&ctx, cell_idx, port_d), Some(net_idx));
}

#[test]
fn connect_port_as_driver() {
    let mut ctx = common::make_context();
    let cell_idx = ctx.design.add_cell(ctx.id("c1"), ctx.id("LUT4"));
    let net_idx = ctx.design.add_net(ctx.id("n1"));
    let port_o = ctx.id("O");
    ctx.design.cell_edit(cell_idx).add_port(port_o, PortType::Out);
    connect_port(&mut ctx, cell_idx, port_o, net_idx);
    let net = ctx.design.net(net_idx);
    assert!(net.driver.is_connected());
    assert_eq!(net.driver.cell, Some(cell_idx));
    assert_eq!(net.driver.port, port_o);
}

#[test]
fn rename_port_basic() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    let old_name = ctx.id("O");
    let new_name = ctx.id("Q");
    let net_before = get_net_for_port(&ctx, cell_idx, old_name);
    ctx.design.cell_edit(cell_idx).rename_port(old_name, new_name);
    assert!(ctx.design.cell(cell_idx).port(old_name).is_none());
    assert_eq!(get_net_for_port(&ctx, cell_idx, new_name), net_before);
}

#[test]
fn rename_port_nonexistent_is_noop() {
    let mut ctx = common::make_context();
    setup_simple_ctx(&mut ctx);
    let cell_idx = ctx.design.cell_by_name(ctx.id("my_cell")).unwrap();
    let nonexistent = ctx.id("NONEXISTENT");
    let q = ctx.id("Q");
    ctx.design.cell_edit(cell_idx).rename_port(nonexistent, q);
}

#[test]
fn is_single_fanout_true() {
    let mut ctx = common::make_context();
    let driver_idx = ctx.design.add_cell(ctx.id("driver"), ctx.id("LUT4"));
    let sink_idx = ctx.design.add_cell(ctx.id("sink"), ctx.id("FF"));
    let net_idx = ctx.design.add_net(ctx.id("n1"));
    let port_o = ctx.id("O");
    let port_d = ctx.id("D");
    ctx.design.cell_edit(driver_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(sink_idx).add_port(port_d, PortType::In);
    connect_port(&mut ctx, driver_idx, port_o, net_idx);
    connect_port(&mut ctx, sink_idx, port_d, net_idx);
    assert!(is_single_fanout(&ctx, net_idx));
}

#[test]
fn is_single_fanout_false_multi() {
    let mut ctx = common::make_context();
    let driver_idx = ctx.design.add_cell(ctx.id("driver"), ctx.id("LUT4"));
    let sink1_idx = ctx.design.add_cell(ctx.id("sink1"), ctx.id("FF"));
    let sink2_idx = ctx.design.add_cell(ctx.id("sink2"), ctx.id("FF"));
    let net_idx = ctx.design.add_net(ctx.id("n1"));
    let port_o = ctx.id("O");
    let port_d = ctx.id("D");
    let port_d2 = ctx.id("D2");
    ctx.design.cell_edit(driver_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(sink1_idx).add_port(port_d, PortType::In);
    ctx.design.cell_edit(sink2_idx).add_port(port_d2, PortType::In);
    connect_port(&mut ctx, driver_idx, port_o, net_idx);
    connect_port(&mut ctx, sink1_idx, port_d, net_idx);
    connect_port(&mut ctx, sink2_idx, port_d2, net_idx);
    assert!(!is_single_fanout(&ctx, net_idx));
}

#[test]
fn is_single_fanout_false_no_users() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("empty_net"));
    assert!(!is_single_fanout(&ctx, net_idx));
}

#[test]
fn pack_constants_creates_gnd_vcc() {
    let mut ctx = common::make_context();
    passes::pack_constants(&mut ctx).unwrap();
    assert!(ctx.design.cell_by_name(ctx.id("$PACKER_GND")).is_some());
    assert!(ctx.design.cell_by_name(ctx.id("$PACKER_VCC")).is_some());
    let gnd_net_idx = ctx.design.net_by_name(ctx.id("$PACKER_GND_NET")).unwrap();
    let vcc_net_idx = ctx.design.net_by_name(ctx.id("$PACKER_VCC_NET")).unwrap();
    assert!(ctx.design.net(gnd_net_idx).driver.is_connected());
    assert!(ctx.design.net(vcc_net_idx).driver.is_connected());
}

#[test]
fn pack_constants_idempotent() {
    let mut ctx = common::make_context();
    passes::pack_constants(&mut ctx).unwrap();
    passes::pack_constants(&mut ctx).unwrap();
    assert!(ctx.design.cell_by_name(ctx.id("$PACKER_GND")).is_some());
}

#[test]
fn pack_lut_ff_merges_single_fanout() {
    let mut ctx = common::make_context();
    let lut_idx = ctx.design.add_cell(ctx.id("lut0"), ctx.id("LUT4"));
    let ff_idx = ctx.design.add_cell(ctx.id("ff0"), ctx.id("DFF"));
    let net_idx = ctx.design.add_net(ctx.id("lut_to_ff"));
    let port_o = ctx.id("O");
    let port_d = ctx.id("D");
    let port_q = ctx.id("Q");
    let port_clk = ctx.id("CLK");
    ctx.design.cell_edit(lut_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(ff_idx).add_port(port_d, PortType::In);
    ctx.design.cell_edit(ff_idx).add_port(port_q, PortType::Out);
    ctx.design.cell_edit(ff_idx).add_port(port_clk, PortType::In);
    connect_port(&mut ctx, lut_idx, port_o, net_idx);
    connect_port(&mut ctx, ff_idx, port_d, net_idx);
    passes::pack_lut_ff(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(lut_idx).cluster, Some(lut_idx));
    assert_eq!(ctx.design.cell(ff_idx).cluster, Some(lut_idx));
}

#[test]
fn pack_carry_chains_simple() {
    let mut ctx = common::make_context();
    let carry0 = ctx.design.add_cell(ctx.id("carry0"), ctx.id("CARRY4"));
    let carry1 = ctx.design.add_cell(ctx.id("carry1"), ctx.id("CARRY4"));
    let net_idx = ctx.design.add_net(ctx.id("carry_chain"));
    let port_co = ctx.id("CO");
    let port_ci = ctx.id("CI");
    ctx.design.cell_edit(carry0).add_port(port_co, PortType::Out);
    ctx.design.cell_edit(carry1).add_port(port_ci, PortType::In);
    connect_port(&mut ctx, carry0, port_co, net_idx);
    connect_port(&mut ctx, carry1, port_ci, net_idx);
    passes::pack_carry(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(carry0).cluster, Some(carry0));
    assert_eq!(ctx.design.cell(carry1).cluster, Some(carry0));
}

#[test]
fn pack_remaining_passes_through() {
    let mut ctx = common::make_context();
    let cell_idx = ctx.design.add_cell(ctx.id("misc0"), ctx.id("BRAM"));
    passes::pack_remaining(&mut ctx).unwrap();
    assert!(ctx.design.cell(cell_idx).alive);
}

#[test]
fn full_pack_default_on_empty_design() {
    let mut ctx = common::make_context();
    assert!(pack_default(&mut ctx).is_ok());
}

// --- PackerError tests ---

use nextpnr::packer::{pack, PackerError};
use nextpnr::plugin::{PackerPlugin, PluginContext, PluginError};

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

// --- Plugin delegation tests ---

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

// --- Cell removal tests ---

#[test]
fn remove_cell_marks_dead() {
    let mut ctx = common::make_context();
    let cell_name = ctx.id("doomed");
    let _cell_idx = ctx.design.add_cell(cell_name, ctx.id("LUT4"));
    assert!(ctx.design.cell_by_name(cell_name).is_some());
    ctx.design.remove_cell(cell_name);
    assert!(ctx.design.cell_by_name(cell_name).is_none());
}

#[test]
fn remove_cell_nonexistent_is_noop() {
    let mut ctx = common::make_context();
    let bogus = ctx.id("nobody");
    ctx.design.remove_cell(bogus);
}

// --- Individual IO tests ---

fn setup_io_cell(
    ctx: &mut nextpnr::context::Context,
    name: &str,
    cell_type: &str,
) -> nextpnr::netlist::CellId {
    let port_o = ctx.id("O");
    let port_i = ctx.id("I");
    let cell_idx = ctx.design.add_cell(ctx.id(name), ctx.id(cell_type));
    ctx.design.cell_edit(cell_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(cell_idx).add_port(port_i, PortType::In);
    cell_idx
}

#[test]
fn pack_io_remaps_ibuf() {
    let mut ctx = common::make_context();
    let cell_idx = setup_io_cell(&mut ctx, "io0", "$nextpnr_IBUF");
    passes::pack_io(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(cell_idx).cell_type, ctx.id("IOB"));
}

#[test]
fn pack_io_remaps_obuf() {
    let mut ctx = common::make_context();
    let cell_idx = setup_io_cell(&mut ctx, "io1", "$nextpnr_OBUF");
    passes::pack_io(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(cell_idx).cell_type, ctx.id("IOB"));
}

#[test]
fn pack_io_remaps_iobuf() {
    let mut ctx = common::make_context();
    let cell_idx = setup_io_cell(&mut ctx, "io2", "$nextpnr_IOBUF");
    passes::pack_io(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(cell_idx).cell_type, ctx.id("IOB"));
}

#[test]
fn pack_io_leaves_non_io_cells_alone() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let cell_idx = ctx.design.add_cell(ctx.id("lut0"), lut_type);
    passes::pack_io(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(cell_idx).cell_type, lut_type);
}

// --- LUT+FF merge edge cases ---

#[test]
fn pack_lut_ff_no_merge_multi_fanout() {
    let mut ctx = common::make_context();
    let lut_idx = ctx.design.add_cell(ctx.id("lut0"), ctx.id("LUT4"));
    let ff_idx = ctx.design.add_cell(ctx.id("ff0"), ctx.id("DFF"));
    let other_idx = ctx.design.add_cell(ctx.id("other"), ctx.id("BUF"));
    let net_idx = ctx.design.add_net(ctx.id("lut_out"));
    let port_o = ctx.id("O");
    let port_d = ctx.id("D");
    let port_a = ctx.id("A");
    ctx.design.cell_edit(lut_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(ff_idx).add_port(port_d, PortType::In);
    ctx.design.cell_edit(other_idx).add_port(port_a, PortType::In);
    connect_port(&mut ctx, lut_idx, port_o, net_idx);
    connect_port(&mut ctx, ff_idx, port_d, net_idx);
    connect_port(&mut ctx, other_idx, port_a, net_idx);
    passes::pack_lut_ff(&mut ctx).unwrap();
    assert!(ctx.design.cell(lut_idx).cluster.is_none());
}

#[test]
fn pack_lut_ff_no_merge_ff_type_mismatch() {
    let mut ctx = common::make_context();
    let lut_idx = ctx.design.add_cell(ctx.id("lut0"), ctx.id("LUT4"));
    let buf_idx = ctx.design.add_cell(ctx.id("buf0"), ctx.id("BUF"));
    let net_idx = ctx.design.add_net(ctx.id("net0"));
    let port_o = ctx.id("O");
    let port_i = ctx.id("I");
    ctx.design.cell_edit(lut_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(buf_idx).add_port(port_i, PortType::In);
    connect_port(&mut ctx, lut_idx, port_o, net_idx);
    connect_port(&mut ctx, buf_idx, port_i, net_idx);
    passes::pack_lut_ff(&mut ctx).unwrap();
    assert!(ctx.design.cell(lut_idx).cluster.is_none());
}

// --- Full pack with cells ---

#[test]
fn full_pack_default_with_cells() {
    let mut ctx = common::make_context();
    let lut_idx = ctx.design.add_cell(ctx.id("lut0"), ctx.id("LUT4"));
    let ff_idx = ctx.design.add_cell(ctx.id("ff0"), ctx.id("DFF"));
    let net_idx = ctx.design.add_net(ctx.id("n0"));
    let port_o = ctx.id("O");
    let port_d = ctx.id("D");
    ctx.design.cell_edit(lut_idx).add_port(port_o, PortType::Out);
    ctx.design.cell_edit(ff_idx).add_port(port_d, PortType::In);
    connect_port(&mut ctx, lut_idx, port_o, net_idx);
    connect_port(&mut ctx, ff_idx, port_d, net_idx);
    assert!(pack_default(&mut ctx).is_ok());
    assert_eq!(ctx.design.cell(lut_idx).cluster, Some(lut_idx));
    assert_eq!(ctx.design.cell(ff_idx).cluster, Some(lut_idx));
}

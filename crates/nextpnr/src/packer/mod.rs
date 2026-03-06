//! Database-driven packer for the nextpnr-rust FPGA place-and-route tool.
//!
//! The packer transforms technology-mapped netlist cells (from Yosys) into
//! architecture-specific "packed" cells that map directly to BELs on the FPGA.
//!
//! The main entry point is [`pack`], which delegates to a plugin if one is
//! provided, or falls back to the built-in database-driven packer that performs:
//! 1. Constant driver handling (GND/VCC)
//! 2. IO buffer insertion/remapping
//! 3. LUT+FF merging into clusters
//! 4. Carry chain construction
//! 5. Remaining cell passthrough

mod helpers;
pub(crate) mod passes;

#[cfg(test)]
use helpers::{connect_port, disconnect_port, get_net_for_port, is_single_fanout, remove_cell, rename_port};

use crate::context::Context;
use crate::plugin::{PackerPlugin, PluginContext, PluginError};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during packing.
#[derive(Debug, thiserror::Error)]
pub enum PackerError {
    /// A general packer error with a description.
    #[error("Packer error: {0}")]
    Generic(String),

    /// A cell type that the packer does not know how to handle.
    #[error("Unsupported cell type: {0}")]
    UnsupportedCellType(String),

    /// An error originating from a packer plugin.
    #[error("Plugin error: {0}")]
    Plugin(#[from] PluginError),
}

// ---------------------------------------------------------------------------
// Main packer entry point
// ---------------------------------------------------------------------------

/// Run the packer on the design.
///
/// If a plugin is provided, delegates to it. Otherwise uses the built-in
/// database-driven packer.
pub fn pack(
    ctx: &mut Context,
    plugin: Option<&mut dyn PackerPlugin>,
) -> Result<(), PackerError> {
    if let Some(plugin) = plugin {
        let (design, chipdb, id_pool) = ctx.packer_parts();
        let mut plugin_ctx = PluginContext::new(design, chipdb, id_pool);
        plugin.pack(&mut plugin_ctx).map_err(PackerError::Plugin)?;
    } else {
        pack_default(ctx)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Default (database-driven) packer
// ---------------------------------------------------------------------------

/// The built-in packer that runs a series of standard passes.
pub(crate) fn pack_default(
    ctx: &mut Context,
) -> Result<(), PackerError> {
    // 1. Handle constant drivers (GND/VCC)
    passes::pack_constants(ctx)?;

    // 2. Pack IO buffers
    passes::pack_io(ctx)?;

    // 3. Pack LUTs (merge with FFs if possible)
    passes::pack_lut_ff(ctx)?;

    // 4. Pack carry chains
    passes::pack_carry(ctx)?;

    // 5. Pack remaining cells (generic passthrough)
    passes::pack_remaining(ctx)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::context::Context;
    use crate::plugin::{PackerPlugin, PluginContext, PluginError};
    use crate::types::PortType;

    fn make_test_ctx() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    // PackerError tests

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

    // Plugin delegation tests

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

    // Utility function tests

    fn setup_simple_ctx(ctx: &mut Context) {
        let cell_name = ctx.id("my_cell");
        let cell_type = ctx.id("LUT4");
        let port_o = ctx.id("O");
        let port_i = ctx.id("I");
        let net_name = ctx.id("my_net");

        let cell_idx = ctx.add_cell(cell_name, cell_type);
        let net_idx = ctx.add_net(net_name);

        ctx.cell_edit(cell_idx).add_port(port_o, PortType::Out);
        connect_port(ctx, cell_idx, port_o, net_idx);

        ctx.cell_edit(cell_idx).add_port(port_i, PortType::In);
    }

    #[test]
    fn get_net_for_port_connected() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let port_o = ctx.id("O");
        let net = get_net_for_port(&ctx, cell_idx, port_o);
        assert!(net.is_some());
        let net_idx = net.unwrap();
        assert_eq!(ctx.design().net(net_idx).name, ctx.id("my_net"));
    }

    #[test]
    fn get_net_for_port_unconnected() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let port_i = ctx.id("I");
        assert!(get_net_for_port(&ctx, cell_idx, port_i).is_none());
    }

    #[test]
    fn get_net_for_port_nonexistent_port() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let port_x = ctx.id("NONEXISTENT");
        assert!(get_net_for_port(&ctx, cell_idx, port_x).is_none());
    }

    #[test]
    fn disconnect_port_removes_connection() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let port_o = ctx.id("O");
        let net_name = ctx.id("my_net");
        assert!(get_net_for_port(&ctx, cell_idx, port_o).is_some());
        disconnect_port(&mut ctx, cell_idx, port_o);
        assert!(get_net_for_port(&ctx, cell_idx, port_o).is_none());
        let net_idx = ctx.design().net_by_name(net_name).unwrap();
        let net = ctx.design().net(net_idx);
        assert!(!net.driver.is_connected() || net.driver.cell != Some(cell_idx));
    }

    #[test]
    fn disconnect_port_nonexistent_is_noop() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let port_x = ctx.id("NONEXISTENT");
        disconnect_port(&mut ctx, cell_idx, port_x);
    }

    #[test]
    fn connect_port_to_net() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("c1");
        let cell_type = ctx.id("FF");
        let port_d = ctx.id("D");
        let net_name = ctx.id("n1");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(cell_idx).add_port(port_d, PortType::In);
        connect_port(&mut ctx, cell_idx, port_d, net_idx);
        assert_eq!(get_net_for_port(&ctx, cell_idx, port_d), Some(net_idx));
        let net = ctx.design().net(net_idx);
        assert!(net.users.iter().any(|u| u.cell == Some(cell_idx) && u.port == port_d));
    }

    #[test]
    fn connect_port_as_driver() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("c1");
        let cell_type = ctx.id("LUT4");
        let port_o = ctx.id("O");
        let net_name = ctx.id("n1");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(cell_idx).add_port(port_o, PortType::Out);
        connect_port(&mut ctx, cell_idx, port_o, net_idx);
        let net = ctx.design().net(net_idx);
        assert!(net.driver.is_connected());
        assert_eq!(net.driver.cell, Some(cell_idx));
        assert_eq!(net.driver.port, port_o);
    }

    #[test]
    fn rename_port_basic() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let old_name = ctx.id("O");
        let new_name = ctx.id("Q");
        let net_before = get_net_for_port(&ctx, cell_idx, old_name);
        assert!(net_before.is_some());
        rename_port(&mut ctx, cell_idx, old_name, new_name);
        assert!(ctx.design().cell(cell_idx).port(old_name).is_none());
        let net_after = get_net_for_port(&ctx, cell_idx, new_name);
        assert_eq!(net_before, net_after);
    }

    #[test]
    fn rename_port_nonexistent_is_noop() {
        let mut ctx = make_test_ctx();
        setup_simple_ctx(&mut ctx);
        let cell_idx = ctx.design().cell_by_name(ctx.id("my_cell")).unwrap();
        let old_name = ctx.id("NONEXISTENT");
        let new_name = ctx.id("Q");
        rename_port(&mut ctx, cell_idx, old_name, new_name);
    }

    #[test]
    fn is_single_fanout_true() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("driver");
        let cell_type = ctx.id("LUT4");
        let sink_name = ctx.id("sink");
        let sink_type = ctx.id("FF");
        let port_o = ctx.id("O");
        let port_d = ctx.id("D");
        let net_name = ctx.id("n1");
        let driver_idx = ctx.add_cell(cell_name, cell_type);
        let sink_idx = ctx.add_cell(sink_name, sink_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(driver_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(sink_idx).add_port(port_d, PortType::In);
        connect_port(&mut ctx, driver_idx, port_o, net_idx);
        connect_port(&mut ctx, sink_idx, port_d, net_idx);
        assert!(is_single_fanout(&ctx, net_idx));
    }

    #[test]
    fn is_single_fanout_false_multi() {
        let mut ctx = make_test_ctx();
        let driver_name = ctx.id("driver");
        let sink1_name = ctx.id("sink1");
        let sink2_name = ctx.id("sink2");
        let lut_type = ctx.id("LUT4");
        let ff_type = ctx.id("FF");
        let port_o = ctx.id("O");
        let port_d = ctx.id("D");
        let port_d2 = ctx.id("D2");
        let net_name = ctx.id("n1");
        let driver_idx = ctx.add_cell(driver_name, lut_type);
        let sink1_idx = ctx.add_cell(sink1_name, ff_type);
        let sink2_idx = ctx.add_cell(sink2_name, ff_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(driver_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(sink1_idx).add_port(port_d, PortType::In);
        ctx.cell_edit(sink2_idx).add_port(port_d2, PortType::In);
        connect_port(&mut ctx, driver_idx, port_o, net_idx);
        connect_port(&mut ctx, sink1_idx, port_d, net_idx);
        connect_port(&mut ctx, sink2_idx, port_d2, net_idx);
        assert!(!is_single_fanout(&ctx, net_idx));
    }

    #[test]
    fn is_single_fanout_false_no_users() {
        let mut ctx = make_test_ctx();
        let net_name = ctx.id("empty_net");
        let net_idx = ctx.add_net(net_name);
        assert!(!is_single_fanout(&ctx, net_idx));
    }

    #[test]
    fn remove_cell_marks_dead() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("doomed");
        let cell_type = ctx.id("LUT4");
        let _cell_idx = ctx.add_cell(cell_name, cell_type);
        assert!(ctx.design().cell_by_name(cell_name).is_some());
        remove_cell(&mut ctx, cell_name);
        assert!(ctx.design().cell_by_name(cell_name).is_none());
    }

    #[test]
    fn remove_cell_nonexistent_is_noop() {
        let mut ctx = make_test_ctx();
        let bogus = ctx.id("nobody");
        remove_cell(&mut ctx, bogus);
    }

    // Packing pass tests

    #[test]
    fn pack_constants_creates_gnd_vcc() {
        let mut ctx = make_test_ctx();
        passes::pack_constants(&mut ctx).unwrap();
        let gnd_cell_name = ctx.id("$PACKER_GND");
        let vcc_cell_name = ctx.id("$PACKER_VCC");
        let gnd_net_name = ctx.id("$PACKER_GND_NET");
        let vcc_net_name = ctx.id("$PACKER_VCC_NET");
        assert!(ctx.design().cell_by_name(gnd_cell_name).is_some());
        assert!(ctx.design().cell_by_name(vcc_cell_name).is_some());
        assert!(ctx.design().net_by_name(gnd_net_name).is_some());
        assert!(ctx.design().net_by_name(vcc_net_name).is_some());
        let gnd_net_idx = ctx.design().net_by_name(gnd_net_name).unwrap();
        assert!(ctx.design().net(gnd_net_idx).driver.is_connected());
        let vcc_net_idx = ctx.design().net_by_name(vcc_net_name).unwrap();
        assert!(ctx.design().net(vcc_net_idx).driver.is_connected());
    }

    #[test]
    fn pack_constants_idempotent() {
        let mut ctx = make_test_ctx();
        passes::pack_constants(&mut ctx).unwrap();
        passes::pack_constants(&mut ctx).unwrap();
        let gnd_cell_name = ctx.id("$PACKER_GND");
        assert!(ctx.design().cell_by_name(gnd_cell_name).is_some());
    }

    #[test]
    fn pack_io_remaps_ibuf() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("io0");
        let cell_type = ctx.id("$nextpnr_IBUF");
        let port_o = ctx.id("O");
        let port_i = ctx.id("I");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        ctx.cell_edit(cell_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(cell_idx).add_port(port_i, PortType::In);
        passes::pack_io(&mut ctx).unwrap();
        let iob_type = ctx.id("IOB");
        assert_eq!(ctx.design().cell(cell_idx).cell_type, iob_type);
    }

    #[test]
    fn pack_io_remaps_obuf() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("io1");
        let cell_type = ctx.id("$nextpnr_OBUF");
        let port_o = ctx.id("O");
        let port_i = ctx.id("I");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        ctx.cell_edit(cell_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(cell_idx).add_port(port_i, PortType::In);
        passes::pack_io(&mut ctx).unwrap();
        assert_eq!(ctx.design().cell(cell_idx).cell_type, ctx.id("IOB"));
    }

    #[test]
    fn pack_io_remaps_iobuf() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("io2");
        let cell_type = ctx.id("$nextpnr_IOBUF");
        let port_o = ctx.id("O");
        let port_i = ctx.id("I");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        ctx.cell_edit(cell_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(cell_idx).add_port(port_i, PortType::In);
        passes::pack_io(&mut ctx).unwrap();
        assert_eq!(ctx.design().cell(cell_idx).cell_type, ctx.id("IOB"));
    }

    #[test]
    fn pack_io_leaves_non_io_cells_alone() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("lut0");
        let lut_type = ctx.id("LUT4");
        let cell_idx = ctx.add_cell(cell_name, lut_type);
        passes::pack_io(&mut ctx).unwrap();
        assert_eq!(ctx.design().cell(cell_idx).cell_type, lut_type);
    }

    #[test]
    fn pack_lut_ff_merges_single_fanout() {
        let mut ctx = make_test_ctx();
        let lut_name = ctx.id("lut0");
        let ff_name = ctx.id("ff0");
        let lut_type = ctx.id("LUT4");
        let ff_type = ctx.id("DFF");
        let port_o = ctx.id("O");
        let port_d = ctx.id("D");
        let port_q = ctx.id("Q");
        let port_clk = ctx.id("CLK");
        let net_name = ctx.id("lut_to_ff");
        let lut_idx = ctx.add_cell(lut_name, lut_type);
        let ff_idx = ctx.add_cell(ff_name, ff_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(lut_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(ff_idx).add_port(port_d, PortType::In);
        ctx.cell_edit(ff_idx).add_port(port_q, PortType::Out);
        ctx.cell_edit(ff_idx).add_port(port_clk, PortType::In);
        connect_port(&mut ctx, lut_idx, port_o, net_idx);
        connect_port(&mut ctx, ff_idx, port_d, net_idx);
        passes::pack_lut_ff(&mut ctx).unwrap();
        assert_eq!(ctx.design().cell(lut_idx).cluster, Some(lut_idx));
        assert_eq!(ctx.design().cell(ff_idx).cluster, Some(lut_idx));
        let cluster = ctx.design().clusters.get(&lut_idx).expect("cluster should exist");
        assert!(cluster.members.contains(&ff_idx));
    }

    #[test]
    fn pack_lut_ff_no_merge_multi_fanout() {
        let mut ctx = make_test_ctx();
        let lut_name = ctx.id("lut0");
        let ff_name = ctx.id("ff0");
        let other_name = ctx.id("other");
        let lut_type = ctx.id("LUT4");
        let ff_type = ctx.id("DFF");
        let buf_type = ctx.id("BUF");
        let port_o = ctx.id("O");
        let port_d = ctx.id("D");
        let port_a = ctx.id("A");
        let net_name = ctx.id("lut_out");
        let lut_idx = ctx.add_cell(lut_name, lut_type);
        let ff_idx = ctx.add_cell(ff_name, ff_type);
        let other_idx = ctx.add_cell(other_name, buf_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(lut_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(ff_idx).add_port(port_d, PortType::In);
        ctx.cell_edit(other_idx).add_port(port_a, PortType::In);
        connect_port(&mut ctx, lut_idx, port_o, net_idx);
        connect_port(&mut ctx, ff_idx, port_d, net_idx);
        connect_port(&mut ctx, other_idx, port_a, net_idx);
        passes::pack_lut_ff(&mut ctx).unwrap();
        assert!(ctx.design().cell(lut_idx).cluster.is_none());
    }

    #[test]
    fn pack_lut_ff_no_merge_ff_type_mismatch() {
        let mut ctx = make_test_ctx();
        let lut_name = ctx.id("lut0");
        let buf_name = ctx.id("buf0");
        let lut_type = ctx.id("LUT4");
        let buf_type = ctx.id("BUF");
        let port_o = ctx.id("O");
        let port_i = ctx.id("I");
        let net_name = ctx.id("net0");
        let lut_idx = ctx.add_cell(lut_name, lut_type);
        let buf_idx = ctx.add_cell(buf_name, buf_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(lut_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(buf_idx).add_port(port_i, PortType::In);
        connect_port(&mut ctx, lut_idx, port_o, net_idx);
        connect_port(&mut ctx, buf_idx, port_i, net_idx);
        passes::pack_lut_ff(&mut ctx).unwrap();
        assert!(ctx.design().cell(lut_idx).cluster.is_none());
    }

    #[test]
    fn pack_carry_chains_simple() {
        let mut ctx = make_test_ctx();
        let carry0_name = ctx.id("carry0");
        let carry1_name = ctx.id("carry1");
        let carry_type = ctx.id("CARRY4");
        let port_co = ctx.id("CO");
        let port_ci = ctx.id("CI");
        let net_name = ctx.id("carry_chain");
        let carry0_idx = ctx.add_cell(carry0_name, carry_type);
        let carry1_idx = ctx.add_cell(carry1_name, carry_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(carry0_idx).add_port(port_co, PortType::Out);
        ctx.cell_edit(carry1_idx).add_port(port_ci, PortType::In);
        connect_port(&mut ctx, carry0_idx, port_co, net_idx);
        connect_port(&mut ctx, carry1_idx, port_ci, net_idx);
        passes::pack_carry(&mut ctx).unwrap();
        assert_eq!(ctx.design().cell(carry0_idx).cluster, Some(carry0_idx));
        assert_eq!(ctx.design().cell(carry1_idx).cluster, Some(carry0_idx));
        let cluster = ctx.design().clusters.get(&carry0_idx).expect("cluster should exist");
        assert!(cluster.members.contains(&carry1_idx));
    }

    #[test]
    fn pack_remaining_passes_through() {
        let mut ctx = make_test_ctx();
        let cell_name = ctx.id("misc0");
        let cell_type = ctx.id("BRAM");
        let cell_idx = ctx.add_cell(cell_name, cell_type);
        passes::pack_remaining(&mut ctx).unwrap();
        let cell = ctx.design().cell(cell_idx);
        assert!(cell.alive);
        assert_eq!(cell.cell_type, cell_type);
    }

    #[test]
    fn full_pack_default_on_empty_design() {
        let mut ctx = make_test_ctx();
        assert!(pack_default(&mut ctx).is_ok());
    }

    #[test]
    fn full_pack_default_with_cells() {
        let mut ctx = make_test_ctx();
        let lut_name = ctx.id("lut0");
        let ff_name = ctx.id("ff0");
        let lut_type = ctx.id("LUT4");
        let ff_type = ctx.id("DFF");
        let port_o = ctx.id("O");
        let port_d = ctx.id("D");
        let net_name = ctx.id("n0");
        let lut_idx = ctx.add_cell(lut_name, lut_type);
        let ff_idx = ctx.add_cell(ff_name, ff_type);
        let net_idx = ctx.add_net(net_name);
        ctx.cell_edit(lut_idx).add_port(port_o, PortType::Out);
        ctx.cell_edit(ff_idx).add_port(port_d, PortType::In);
        connect_port(&mut ctx, lut_idx, port_o, net_idx);
        connect_port(&mut ctx, ff_idx, port_d, net_idx);
        assert!(pack_default(&mut ctx).is_ok());
        assert_eq!(ctx.design().cell(lut_idx).cluster, Some(lut_idx));
        assert_eq!(ctx.design().cell(ff_idx).cluster, Some(lut_idx));
    }
}

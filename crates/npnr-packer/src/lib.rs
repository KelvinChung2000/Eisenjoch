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
mod passes;

pub use helpers::{connect_port, disconnect_port, get_net_for_port, is_single_fanout, remove_cell, rename_port};

use npnr_chipdb::ChipDb;
use npnr_netlist::Design;
use npnr_plugin::{PackerPlugin, PluginContext, PluginError};
use npnr_types::IdStringPool;

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
    design: &mut Design,
    chipdb: &ChipDb,
    id_pool: &IdStringPool,
    plugin: Option<&mut dyn PackerPlugin>,
) -> Result<(), PackerError> {
    if let Some(plugin) = plugin {
        let mut ctx = PluginContext::new(design, chipdb, id_pool);
        plugin.pack(&mut ctx).map_err(PackerError::Plugin)?;
    } else {
        pack_default(design, chipdb, id_pool)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Default (database-driven) packer
// ---------------------------------------------------------------------------

/// The built-in packer that runs a series of standard passes.
fn pack_default(
    design: &mut Design,
    _chipdb: &ChipDb,
    id_pool: &IdStringPool,
) -> Result<(), PackerError> {
    // 1. Handle constant drivers (GND/VCC)
    passes::pack_constants(design, id_pool)?;

    // 2. Pack IO buffers
    passes::pack_io(design, id_pool)?;

    // 3. Pack LUTs (merge with FFs if possible)
    passes::pack_lut_ff(design, id_pool)?;

    // 4. Pack carry chains
    passes::pack_carry(design, id_pool)?;

    // 5. Pack remaining cells (generic passthrough)
    passes::pack_remaining(design, id_pool)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_netlist::Design;
    use npnr_plugin::PluginError;
    use npnr_types::{IdStringPool, PortType};

    fn make_test_chipdb() -> ChipDb {
        npnr_chipdb::testutil::make_test_chipdb()
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
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut packer = TrackingPacker::new();

        let result = pack(&mut design, &chipdb, &pool, Some(&mut packer));
        assert!(result.is_ok());
        assert!(packer.called);
    }

    #[test]
    fn pack_plugin_error_is_propagated() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();
        let mut packer = FailingPacker;

        let result = pack(&mut design, &chipdb, &pool, Some(&mut packer));
        assert!(result.is_err());
        match result.unwrap_err() {
            PackerError::Plugin(_) => {}
            other => panic!("Expected Plugin variant, got {:?}", other),
        }
    }

    #[test]
    fn pack_without_plugin_uses_default() {
        let chipdb = make_test_chipdb();
        let pool = IdStringPool::new();
        let mut design = Design::new();

        // The default packer should succeed on an empty design.
        let result = pack(&mut design, &chipdb, &pool, None);
        assert!(result.is_ok());
    }

    // =====================================================================
    // Utility function tests
    // =====================================================================

    /// Build a design with one cell ("my_cell" / type "LUT4") with an output
    /// port "O" connected to a net "my_net", and one input port "I" unconnected.
    fn make_simple_design(pool: &IdStringPool) -> Design {
        let mut design = Design::new();

        let cell_name = pool.intern("my_cell");
        let cell_type = pool.intern("LUT4");
        let port_o = pool.intern("O");
        let port_i = pool.intern("I");
        let net_name = pool.intern("my_net");

        let cell_idx = design.add_cell(cell_name, cell_type);
        let net_idx = design.add_net(net_name);

        // Add output port O and connect it as driver of my_net
        {
            let cell = design.cell_mut(cell_idx);
            cell.add_port(port_o, PortType::Out);
        }
        connect_port(&mut design, cell_idx, port_o, net_idx);

        // Add input port I, unconnected
        {
            let cell = design.cell_mut(cell_idx);
            cell.add_port(port_i, PortType::In);
        }

        design
    }

    #[test]
    fn get_net_for_port_connected() {
        let pool = IdStringPool::new();
        let design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let port_o = pool.intern("O");

        let net = get_net_for_port(&design, cell_idx, port_o);
        assert!(net.is_some());
        let net_idx = net.unwrap();
        assert_eq!(design.net(net_idx).name, pool.intern("my_net"));
    }

    #[test]
    fn get_net_for_port_unconnected() {
        let pool = IdStringPool::new();
        let design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let port_i = pool.intern("I");

        let net = get_net_for_port(&design, cell_idx, port_i);
        assert!(net.is_none());
    }

    #[test]
    fn get_net_for_port_nonexistent_port() {
        let pool = IdStringPool::new();
        let design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let port_x = pool.intern("NONEXISTENT");

        let net = get_net_for_port(&design, cell_idx, port_x);
        assert!(net.is_none());
    }

    #[test]
    fn disconnect_port_removes_connection() {
        let pool = IdStringPool::new();
        let mut design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let port_o = pool.intern("O");
        let net_name = pool.intern("my_net");

        // Verify connected before disconnect
        assert!(get_net_for_port(&design, cell_idx, port_o).is_some());

        disconnect_port(&mut design, cell_idx, port_o);

        // After disconnect, port should have no net
        assert!(get_net_for_port(&design, cell_idx, port_o).is_none());

        // Net should no longer have a driver pointing to this cell
        let net_idx = *design.nets.get(&net_name).unwrap();
        let net = design.net(net_idx);
        assert!(!net.driver.is_connected() || net.driver.cell != cell_idx);
    }

    #[test]
    fn disconnect_port_nonexistent_is_noop() {
        let pool = IdStringPool::new();
        let mut design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let port_x = pool.intern("NONEXISTENT");

        // Should not panic
        disconnect_port(&mut design, cell_idx, port_x);
    }

    #[test]
    fn connect_port_to_net() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("c1");
        let cell_type = pool.intern("FF");
        let port_d = pool.intern("D");
        let net_name = pool.intern("n1");

        let cell_idx = design.add_cell(cell_name, cell_type);
        let net_idx = design.add_net(net_name);

        // Add an input port
        design.cell_mut(cell_idx).add_port(port_d, PortType::In);

        connect_port(&mut design, cell_idx, port_d, net_idx);

        // Verify the port is connected
        assert_eq!(get_net_for_port(&design, cell_idx, port_d), Some(net_idx));

        // Verify the net has this cell as a user
        let net = design.net(net_idx);
        assert!(net.users.iter().any(|u| u.cell == cell_idx && u.port == port_d));
    }

    #[test]
    fn connect_port_as_driver() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("c1");
        let cell_type = pool.intern("LUT4");
        let port_o = pool.intern("O");
        let net_name = pool.intern("n1");

        let cell_idx = design.add_cell(cell_name, cell_type);
        let net_idx = design.add_net(net_name);

        design.cell_mut(cell_idx).add_port(port_o, PortType::Out);

        connect_port(&mut design, cell_idx, port_o, net_idx);

        // Verify the net's driver is this cell
        let net = design.net(net_idx);
        assert!(net.driver.is_connected());
        assert_eq!(net.driver.cell, cell_idx);
        assert_eq!(net.driver.port, port_o);
    }

    #[test]
    fn rename_port_basic() {
        let pool = IdStringPool::new();
        let mut design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let old_name = pool.intern("O");
        let new_name = pool.intern("Q");

        // Port "O" is connected to my_net
        let net_before = get_net_for_port(&design, cell_idx, old_name);
        assert!(net_before.is_some());

        rename_port(&mut design, cell_idx, old_name, new_name);

        // Old port should be gone
        assert!(design.cell(cell_idx).port(old_name).is_none());

        // New port should exist and be connected to the same net
        let net_after = get_net_for_port(&design, cell_idx, new_name);
        assert_eq!(net_before, net_after);
    }

    #[test]
    fn rename_port_nonexistent_is_noop() {
        let pool = IdStringPool::new();
        let mut design = make_simple_design(&pool);

        let cell_idx = *design.cells.get(&pool.intern("my_cell")).unwrap();
        let old_name = pool.intern("NONEXISTENT");
        let new_name = pool.intern("Q");

        // Should not panic
        rename_port(&mut design, cell_idx, old_name, new_name);
    }

    #[test]
    fn is_single_fanout_true() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("driver");
        let cell_type = pool.intern("LUT4");
        let sink_name = pool.intern("sink");
        let port_o = pool.intern("O");
        let port_d = pool.intern("D");
        let net_name = pool.intern("n1");

        let driver_idx = design.add_cell(cell_name, cell_type);
        let sink_idx = design.add_cell(sink_name, pool.intern("FF"));
        let net_idx = design.add_net(net_name);

        design.cell_mut(driver_idx).add_port(port_o, PortType::Out);
        design.cell_mut(sink_idx).add_port(port_d, PortType::In);

        connect_port(&mut design, driver_idx, port_o, net_idx);
        connect_port(&mut design, sink_idx, port_d, net_idx);

        assert!(is_single_fanout(&design, net_idx));
    }

    #[test]
    fn is_single_fanout_false_multi() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let driver_name = pool.intern("driver");
        let sink1_name = pool.intern("sink1");
        let sink2_name = pool.intern("sink2");
        let port_o = pool.intern("O");
        let port_d = pool.intern("D");
        let port_d2 = pool.intern("D2");
        let net_name = pool.intern("n1");

        let driver_idx = design.add_cell(driver_name, pool.intern("LUT4"));
        let sink1_idx = design.add_cell(sink1_name, pool.intern("FF"));
        let sink2_idx = design.add_cell(sink2_name, pool.intern("FF"));
        let net_idx = design.add_net(net_name);

        design.cell_mut(driver_idx).add_port(port_o, PortType::Out);
        design.cell_mut(sink1_idx).add_port(port_d, PortType::In);
        design.cell_mut(sink2_idx).add_port(port_d2, PortType::In);

        connect_port(&mut design, driver_idx, port_o, net_idx);
        connect_port(&mut design, sink1_idx, port_d, net_idx);
        connect_port(&mut design, sink2_idx, port_d2, net_idx);

        assert!(!is_single_fanout(&design, net_idx));
    }

    #[test]
    fn is_single_fanout_false_no_users() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let net_name = pool.intern("empty_net");
        let net_idx = design.add_net(net_name);

        assert!(!is_single_fanout(&design, net_idx));
    }

    #[test]
    fn remove_cell_marks_dead() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("doomed");
        let cell_type = pool.intern("LUT4");
        let cell_idx = design.add_cell(cell_name, cell_type);

        assert!(design.cell(cell_idx).alive);
        assert!(design.cell_by_name(cell_name).is_some());

        remove_cell(&mut design, cell_name);

        assert!(!design.cell(cell_idx).alive);
        assert!(design.cell_by_name(cell_name).is_none());
    }

    #[test]
    fn remove_cell_nonexistent_is_noop() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let bogus = pool.intern("nobody");
        // Should not panic
        remove_cell(&mut design, bogus);
    }

    // =====================================================================
    // Packing pass tests
    // =====================================================================

    #[test]
    fn pack_constants_creates_gnd_vcc() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        passes::pack_constants(&mut design, &pool).unwrap();

        let gnd_cell_name = pool.intern("$PACKER_GND");
        let vcc_cell_name = pool.intern("$PACKER_VCC");
        let gnd_net_name = pool.intern("$PACKER_GND_NET");
        let vcc_net_name = pool.intern("$PACKER_VCC_NET");

        // Cells should exist
        assert!(design.cell_by_name(gnd_cell_name).is_some());
        assert!(design.cell_by_name(vcc_cell_name).is_some());

        // Nets should exist
        assert!(design.net_by_name(gnd_net_name).is_some());
        assert!(design.net_by_name(vcc_net_name).is_some());

        // GND cell should drive GND net
        let gnd_net_idx = design.net_by_name(gnd_net_name).unwrap();
        let gnd_net = design.net(gnd_net_idx);
        assert!(gnd_net.driver.is_connected());

        // VCC cell should drive VCC net
        let vcc_net_idx = design.net_by_name(vcc_net_name).unwrap();
        let vcc_net = design.net(vcc_net_idx);
        assert!(vcc_net.driver.is_connected());
    }

    #[test]
    fn pack_constants_idempotent() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        passes::pack_constants(&mut design, &pool).unwrap();
        // Running again should not panic or create duplicates
        passes::pack_constants(&mut design, &pool).unwrap();

        let gnd_cell_name = pool.intern("$PACKER_GND");
        assert!(design.cell_by_name(gnd_cell_name).is_some());
    }

    #[test]
    fn pack_io_remaps_ibuf() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("io0");
        let cell_type = pool.intern("$nextpnr_IBUF");
        let port_o = pool.intern("O");
        let port_i = pool.intern("I");

        let cell_idx = design.add_cell(cell_name, cell_type);
        design.cell_mut(cell_idx).add_port(port_o, PortType::Out);
        design.cell_mut(cell_idx).add_port(port_i, PortType::In);

        passes::pack_io(&mut design, &pool).unwrap();

        // Cell type should be remapped to IOB
        let iob_type = pool.intern("IOB");
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, iob_type);
    }

    #[test]
    fn pack_io_remaps_obuf() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("io1");
        let cell_type = pool.intern("$nextpnr_OBUF");
        let port_o = pool.intern("O");
        let port_i = pool.intern("I");

        let cell_idx = design.add_cell(cell_name, cell_type);
        design.cell_mut(cell_idx).add_port(port_o, PortType::Out);
        design.cell_mut(cell_idx).add_port(port_i, PortType::In);

        passes::pack_io(&mut design, &pool).unwrap();

        let iob_type = pool.intern("IOB");
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, iob_type);
    }

    #[test]
    fn pack_io_remaps_iobuf() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("io2");
        let cell_type = pool.intern("$nextpnr_IOBUF");
        let port_o = pool.intern("O");
        let port_i = pool.intern("I");

        let cell_idx = design.add_cell(cell_name, cell_type);
        design.cell_mut(cell_idx).add_port(port_o, PortType::Out);
        design.cell_mut(cell_idx).add_port(port_i, PortType::In);

        passes::pack_io(&mut design, &pool).unwrap();

        let iob_type = pool.intern("IOB");
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, iob_type);
    }

    #[test]
    fn pack_io_leaves_non_io_cells_alone() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("lut0");
        let lut_type = pool.intern("LUT4");
        let cell_idx = design.add_cell(cell_name, lut_type);

        passes::pack_io(&mut design, &pool).unwrap();

        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, lut_type);
    }

    #[test]
    fn pack_lut_ff_merges_single_fanout() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        // Create a LUT driving an FF via a net
        let lut_name = pool.intern("lut0");
        let ff_name = pool.intern("ff0");
        let lut_type = pool.intern("LUT4");
        let ff_type = pool.intern("DFF");
        let port_o = pool.intern("O");
        let port_d = pool.intern("D");
        let port_q = pool.intern("Q");
        let port_clk = pool.intern("CLK");
        let net_name = pool.intern("lut_to_ff");

        let lut_idx = design.add_cell(lut_name, lut_type);
        let ff_idx = design.add_cell(ff_name, ff_type);
        let net_idx = design.add_net(net_name);

        // LUT has output port O
        design.cell_mut(lut_idx).add_port(port_o, PortType::Out);
        // FF has input port D, output port Q, and clock port CLK
        design.cell_mut(ff_idx).add_port(port_d, PortType::In);
        design.cell_mut(ff_idx).add_port(port_q, PortType::Out);
        design.cell_mut(ff_idx).add_port(port_clk, PortType::In);

        // Connect LUT.O -> net -> FF.D
        connect_port(&mut design, lut_idx, port_o, net_idx);
        connect_port(&mut design, ff_idx, port_d, net_idx);

        passes::pack_lut_ff(&mut design, &pool).unwrap();

        // After merging, the LUT should be the cluster root and the FF
        // should be a member of the cluster.
        let lut = design.cell(lut_idx);
        let ff = design.cell(ff_idx);

        // LUT is cluster root (cluster == self)
        assert_eq!(lut.cluster, lut_idx);
        // FF belongs to the same cluster
        assert_eq!(ff.cluster, lut_idx);
        // LUT's cluster_next points to the FF
        assert_eq!(lut.cluster_next, ff_idx);
    }

    #[test]
    fn pack_lut_ff_no_merge_multi_fanout() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let lut_name = pool.intern("lut0");
        let ff_name = pool.intern("ff0");
        let other_name = pool.intern("other");
        let lut_type = pool.intern("LUT4");
        let ff_type = pool.intern("DFF");
        let port_o = pool.intern("O");
        let port_d = pool.intern("D");
        let port_a = pool.intern("A");
        let net_name = pool.intern("lut_out");

        let lut_idx = design.add_cell(lut_name, lut_type);
        let ff_idx = design.add_cell(ff_name, ff_type);
        let other_idx = design.add_cell(other_name, pool.intern("BUF"));
        let net_idx = design.add_net(net_name);

        design.cell_mut(lut_idx).add_port(port_o, PortType::Out);
        design.cell_mut(ff_idx).add_port(port_d, PortType::In);
        design.cell_mut(other_idx).add_port(port_a, PortType::In);

        connect_port(&mut design, lut_idx, port_o, net_idx);
        connect_port(&mut design, ff_idx, port_d, net_idx);
        connect_port(&mut design, other_idx, port_a, net_idx);

        passes::pack_lut_ff(&mut design, &pool).unwrap();

        // Should NOT merge because there are 2 users on the net
        let lut = design.cell(lut_idx);
        assert!(lut.cluster.is_none());
    }

    #[test]
    fn pack_lut_ff_no_merge_ff_type_mismatch() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        // LUT output drives a non-FF cell
        let lut_name = pool.intern("lut0");
        let buf_name = pool.intern("buf0");
        let lut_type = pool.intern("LUT4");
        let buf_type = pool.intern("BUF");
        let port_o = pool.intern("O");
        let port_i = pool.intern("I");
        let net_name = pool.intern("net0");

        let lut_idx = design.add_cell(lut_name, lut_type);
        let buf_idx = design.add_cell(buf_name, buf_type);
        let net_idx = design.add_net(net_name);

        design.cell_mut(lut_idx).add_port(port_o, PortType::Out);
        design.cell_mut(buf_idx).add_port(port_i, PortType::In);

        connect_port(&mut design, lut_idx, port_o, net_idx);
        connect_port(&mut design, buf_idx, port_i, net_idx);

        passes::pack_lut_ff(&mut design, &pool).unwrap();

        // Should NOT merge because the sink is not an FF type
        let lut = design.cell(lut_idx);
        assert!(lut.cluster.is_none());
    }

    #[test]
    fn pack_carry_chains_simple() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let carry0_name = pool.intern("carry0");
        let carry1_name = pool.intern("carry1");
        let carry_type = pool.intern("CARRY4");
        let port_co = pool.intern("CO");
        let port_ci = pool.intern("CI");
        let net_name = pool.intern("carry_chain");

        let carry0_idx = design.add_cell(carry0_name, carry_type);
        let carry1_idx = design.add_cell(carry1_name, carry_type);
        let net_idx = design.add_net(net_name);

        design.cell_mut(carry0_idx).add_port(port_co, PortType::Out);
        design.cell_mut(carry1_idx).add_port(port_ci, PortType::In);

        connect_port(&mut design, carry0_idx, port_co, net_idx);
        connect_port(&mut design, carry1_idx, port_ci, net_idx);

        passes::pack_carry(&mut design, &pool).unwrap();

        // carry0 should be the cluster root
        let c0 = design.cell(carry0_idx);
        assert_eq!(c0.cluster, carry0_idx);
        // carry1 should be in the same cluster
        let c1 = design.cell(carry1_idx);
        assert_eq!(c1.cluster, carry0_idx);
        // carry0's cluster_next should point to carry1
        assert_eq!(c0.cluster_next, carry1_idx);
    }

    #[test]
    fn pack_remaining_passes_through() {
        let pool = IdStringPool::new();
        let mut design = Design::new();

        let cell_name = pool.intern("misc0");
        let cell_type = pool.intern("BRAM");
        let cell_idx = design.add_cell(cell_name, cell_type);

        passes::pack_remaining(&mut design, &pool).unwrap();

        // Cell should still be alive and have the same type
        let cell = design.cell(cell_idx);
        assert!(cell.alive);
        assert_eq!(cell.cell_type, cell_type);
    }

    #[test]
    fn full_pack_default_on_empty_design() {
        let pool = IdStringPool::new();
        let chipdb = make_test_chipdb();
        let mut design = Design::new();

        let result = pack_default(&mut design, &chipdb, &pool);
        assert!(result.is_ok());
    }

    #[test]
    fn full_pack_default_with_cells() {
        let pool = IdStringPool::new();
        let chipdb = make_test_chipdb();
        let mut design = Design::new();

        // Add a LUT driving an FF
        let lut_name = pool.intern("lut0");
        let ff_name = pool.intern("ff0");
        let lut_type = pool.intern("LUT4");
        let ff_type = pool.intern("DFF");
        let port_o = pool.intern("O");
        let port_d = pool.intern("D");
        let net_name = pool.intern("n0");

        let lut_idx = design.add_cell(lut_name, lut_type);
        let ff_idx = design.add_cell(ff_name, ff_type);
        let net_idx = design.add_net(net_name);

        design.cell_mut(lut_idx).add_port(port_o, PortType::Out);
        design.cell_mut(ff_idx).add_port(port_d, PortType::In);

        connect_port(&mut design, lut_idx, port_o, net_idx);
        connect_port(&mut design, ff_idx, port_d, net_idx);

        let result = pack_default(&mut design, &chipdb, &pool);
        assert!(result.is_ok());

        // LUT+FF should be merged
        let lut = design.cell(lut_idx);
        assert_eq!(lut.cluster, lut_idx);
        let ff = design.cell(ff_idx);
        assert_eq!(ff.cluster, lut_idx);
    }
}

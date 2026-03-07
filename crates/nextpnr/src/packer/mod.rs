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

pub mod helpers;
pub mod passes;

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
    #[error("{0}")]
    Plugin(#[from] PluginError),
}

// ---------------------------------------------------------------------------
// Main packer entry point
// ---------------------------------------------------------------------------

/// Run the packer on the design.
///
/// If a plugin is provided, delegates to it. Otherwise uses the built-in
/// database-driven packer.
pub fn pack(ctx: &mut Context, plugin: Option<&mut dyn PackerPlugin>) -> Result<(), PackerError> {
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
pub fn pack_default(ctx: &mut Context) -> Result<(), PackerError> {
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

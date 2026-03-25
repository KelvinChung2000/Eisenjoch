//! Database-driven extractor/validator packer for the nextpnr-rust FPGA
//! place-and-route tool.
//!
//! The packer transforms technology-mapped netlist cells (from Yosys) into
//! architecture-specific "packed" cells that map directly to BELs on the FPGA.
//!
//! The main entry point is [`pack`], which delegates to a plugin if one is
//! provided, or falls back to the built-in database-driven packer that:
//! 1. Handles constant drivers (GND/VCC) and IO buffer remapping
//! 2. Extracts cell metadata from the chipdb (tile types, shared wires)
//! 3. Loads or derives packing rules from the chipdb
//! 4. Applies rules, validated by shared-wire and site-capacity checks
//! 5. Passes through remaining cells

pub mod extractor;
pub mod helpers;
pub mod passes;
pub mod pipeline;
pub mod rules;
pub mod tagger;
pub mod validator;

pub use pipeline::{pack, pack_default};

use crate::plugin::PluginError;

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

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
pub mod fabulous;
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

    /// A cell bound or pinned to a BEL that cannot hold it after packing.
    #[error("Cell {cell} cannot take BEL {bel}: {fault}")]
    IllegalBinding {
        cell: String,
        bel: String,
        fault: BindingFault,
    },
}

/// Why a cell's BEL binding cannot hold after packing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BindingFault {
    #[error("the BEL has type {bel_type}, but the cell packs to {packed_type}")]
    TypeMismatch {
        bel_type: String,
        packed_type: String,
    },
    #[error("the cell is already bound to {0}")]
    AlreadyBound(String),
    #[error("cell {0} holds it")]
    Occupied(String),
}

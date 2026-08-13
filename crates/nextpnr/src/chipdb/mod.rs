//! Memory-mapped zero-copy chip database reader for nextpnr-himbaechel.

mod access;
mod chip;
mod grid;
mod ids;
mod load;
mod pod;
mod relptr;
pub mod tile_template;

pub use access::RegArcInfo;
pub use chip::{parse_constids_inc, ChipDb};
pub use grid::Loc;
pub use ids::{BelId, PipId, WireId};
pub use pod::*;
pub use relptr::{RelPtr, RelSlice};
pub use tile_template::{
    port_key, span_bucket_of, Side, TileLocalWs, TileTypeTemplate, PIP_BASE_COST_INT,
};

pub const CHIPDB_MAGIC: i32 = 0x00ca7ca7u32 as i32;
pub const CHIPDB_VERSION: i32 = 6;

#[derive(Debug, thiserror::Error)]
pub enum ChipDbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("chip database file too small ({size} bytes, minimum {min} bytes)")]
    TooSmall { size: usize, min: usize },
    #[error("chip database magic mismatch: expected 0x{expected:08x}, got 0x{got:08x}")]
    MagicMismatch { expected: u32, got: u32 },
    #[error("chip database version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: i32, got: i32 },
    #[error("chip database root pointer out of bounds (offset {offset}, size {size})")]
    InvalidRootPointer { offset: i32, size: usize },
    #[error("chip database contains null required string pointer: {field}")]
    NullRequiredStringPointer { field: &'static str },
    #[error(
        "chip database expects {db_count} compiled-in constids but {supplied} were supplied; \
         load with the uarch's constids.inc (or none, for a database built with known_id_count=0)"
    )]
    KnownConstidMismatch { db_count: i32, supplied: i32 },
}

#[macro_export]
macro_rules! read_packed {
    ($base:expr, $field:ident) => {
        std::ptr::read_unaligned(std::ptr::addr_of!((*std::ptr::addr_of!($base)).$field))
    };
}

pub mod testutil;

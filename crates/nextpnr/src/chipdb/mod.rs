//! Memory-mapped zero-copy chip database reader for nextpnr-himbaechel.

mod access;
mod grid;
mod ids;
mod load;
mod pod;
mod relptr;

pub use access::RegArcInfo;
pub use grid::Loc;
pub use ids::{BelId, PipId, WireId};
pub use pod::*;
pub use relptr::{RelPtr, RelSlice};

use memmap2::Mmap;

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
        "chip database has {count} known constids without embedded strings; \
         regenerate with known_id_count=0 to embed all strings in the binary"
    )]
    MissingKnownConstids { count: i32 },
}

#[macro_export]
macro_rules! read_packed {
    ($base:expr, $field:ident) => {
        std::ptr::read_unaligned(std::ptr::addr_of!((*std::ptr::addr_of!($base)).$field))
    };
}

pub struct ChipDb {
    _mmap: Mmap,
    chip_info: *const ChipInfoPod,
    constid_strs: Vec<Option<*const u8>>,
}

impl std::fmt::Debug for ChipDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChipDb")
            .field("name", &self.name())
            .field("width", &self.width())
            .field("height", &self.height())
            .field("num_tiles", &self.num_tiles())
            .finish()
    }
}

unsafe impl Send for ChipDb {}
unsafe impl Sync for ChipDb {}

fn validate_and_follow_root_relptr(
    base: *const u8,
    size: usize,
) -> Result<*const ChipInfoPod, ChipDbError> {
    let root_relptr = base as *const RelPtr<ChipInfoPod>;
    let offset = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*root_relptr).offset)) };

    let root_addr = root_relptr as usize;
    let target_addr = if offset >= 0 {
        root_addr.checked_add(offset as usize)
    } else {
        root_addr.checked_sub((-offset) as usize)
    }
    .ok_or(ChipDbError::InvalidRootPointer { offset, size })?;

    let base_addr = base as usize;
    let end_addr = base_addr
        .checked_add(size)
        .ok_or(ChipDbError::InvalidRootPointer { offset, size })?;
    let chip_info_size = std::mem::size_of::<ChipInfoPod>();
    let target_end = target_addr
        .checked_add(chip_info_size)
        .ok_or(ChipDbError::InvalidRootPointer { offset, size })?;

    if target_addr < base_addr || target_end > end_addr {
        return Err(ChipDbError::InvalidRootPointer { offset, size });
    }

    Ok(target_addr as *const ChipInfoPod)
}

unsafe fn build_constid_table(
    chip_info: *const ChipInfoPod,
) -> Result<Vec<Option<*const u8>>, ChipDbError> {
    let extra_constids_ptr = (*chip_info).extra_constids.get();
    if extra_constids_ptr.is_null() || (*chip_info).extra_constids.is_null() {
        return Ok(Vec::new());
    }

    let known_id_count: i32 = read_packed!(*extra_constids_ptr, known_id_count);
    if known_id_count > 0 {
        return Err(ChipDbError::MissingKnownConstids {
            count: known_id_count,
        });
    }

    let bba_ids = (*extra_constids_ptr).bba_ids.get();
    let mut table = Vec::with_capacity(bba_ids.len());

    for relptr in bba_ids {
        if relptr.is_null() {
            table.push(None);
        } else {
            table.push(Some(relptr.get() as *const u8));
        }
    }

    Ok(table)
}

#[cfg(feature = "test-utils")]
pub mod testutil;

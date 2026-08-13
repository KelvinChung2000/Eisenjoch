//! ChipDb struct and associated validation/construction functions.

use memmap2::Mmap;

use super::pod::ChipInfoPod;
use super::relptr::RelPtr;
use super::ChipDbError;
use crate::read_packed;

/// One entry of the resolved constid table.
///
/// A himbaechel chipdb only embeds the strings the arch does *not* already
/// know: `known_id_count` ids are compiled into the C++ binary from the uarch's
/// `constids.inc`, and the file stores only what comes after them. So the table
/// is genuinely mixed -- borrowed pointers into the mmap for embedded strings,
/// owned strings for the ones the caller had to supply.
pub(super) enum ConstIdStr {
    /// Points into the mmap; valid for as long as the `ChipDb` lives.
    Embedded(*const u8),
    /// Supplied by the caller from the arch's `constids.inc`.
    Known(String),
    /// A null entry in the file's id table.
    Missing,
}

pub struct ChipDb {
    pub(super) _mmap: Mmap,
    pub(super) chip_info: *const ChipInfoPod,
    pub(super) constid_strs: Vec<ConstIdStr>,
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

pub(crate) fn validate_and_follow_root_relptr(
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

/// Resolve the chipdb's id table, splicing in the arch's compiled-in constids.
///
/// `known` holds the uarch's `constids.inc` entries, which occupy ids
/// `1..=known.len()`; id 0 is always the empty string. A database generated
/// with `known_id_count = 0` embeds every string itself and needs `known` to be
/// empty. Any other combination is a mismatch between the database and the arch
/// it was generated for, and is rejected rather than papered over -- a silent
/// off-by-one here would shift every bel type and wire name by a constant.
pub(crate) unsafe fn build_constid_table(
    chip_info: *const ChipInfoPod,
    known: &[String],
) -> Result<Vec<ConstIdStr>, ChipDbError> {
    let extra_constids_ptr = (*chip_info).extra_constids.get();
    if extra_constids_ptr.is_null() || (*chip_info).extra_constids.is_null() {
        return Ok(Vec::new());
    }

    let known_id_count: i32 = read_packed!(*extra_constids_ptr, known_id_count);

    // `known_id_count` counts id 0 (the empty string) as known, so a database
    // expecting N named constids reports N + 1.
    let expected_known = if known.is_empty() {
        0
    } else {
        known.len() as i32 + 1
    };
    if known_id_count != expected_known {
        return Err(ChipDbError::KnownConstidMismatch {
            db_count: known_id_count,
            supplied: known.len() as i32,
        });
    }

    let bba_ids = (*extra_constids_ptr).bba_ids.get();
    let mut table = Vec::with_capacity(known_id_count.max(0) as usize + bba_ids.len());

    if known_id_count > 0 {
        table.push(ConstIdStr::Known(String::new()));
        for name in known {
            table.push(ConstIdStr::Known(name.clone()));
        }
    }

    for relptr in bba_ids {
        if relptr.is_null() {
            table.push(ConstIdStr::Missing);
        } else {
            table.push(ConstIdStr::Embedded(relptr.get() as *const u8));
        }
    }

    Ok(table)
}

/// Parse a himbaechel `constids.inc`: one `X(NAME)` per line, blanks and
/// anything else ignored, in file order. The order *is* the id numbering, which
/// is why this mirrors the C++ macro expansion rather than sorting.
pub fn parse_constids_inc(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|line| {
            let l = line.trim();
            let rest = l.strip_prefix("X(")?;
            rest.strip_suffix(')').map(|name| name.trim().to_string())
        })
        .collect()
}

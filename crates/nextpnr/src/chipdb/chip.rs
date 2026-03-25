//! ChipDb struct and associated validation/construction functions.

use memmap2::Mmap;

use super::pod::ChipInfoPod;
use super::relptr::RelPtr;
use super::ChipDbError;
use crate::read_packed;

pub struct ChipDb {
    pub(super) _mmap: Mmap,
    pub(super) chip_info: *const ChipInfoPod,
    pub(super) constid_strs: Vec<Option<*const u8>>,
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

pub(crate) unsafe fn build_constid_table(
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

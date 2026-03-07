use super::*;
use crate::read_packed;
use std::path::Path;

impl ChipDb {
    pub fn load(path: &Path) -> Result<Self, ChipDbError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let min_size = std::mem::size_of::<RelPtr<ChipInfoPod>>();
        if mmap.len() < min_size {
            return Err(ChipDbError::TooSmall {
                size: mmap.len(),
                min: min_size,
            });
        }

        let chip_info = validate_and_follow_root_relptr(mmap.as_ptr(), mmap.len())?;

        let magic = unsafe { read_packed!(*chip_info, magic) };
        if magic != CHIPDB_MAGIC {
            return Err(ChipDbError::MagicMismatch {
                expected: CHIPDB_MAGIC as u32,
                got: magic as u32,
            });
        }

        let version = unsafe { read_packed!(*chip_info, version) };
        if version != CHIPDB_VERSION {
            return Err(ChipDbError::VersionMismatch {
                expected: CHIPDB_VERSION,
                got: version,
            });
        }

        if unsafe { (*chip_info).name.is_null() } {
            return Err(ChipDbError::NullRequiredStringPointer { field: "name" });
        }
        if unsafe { (*chip_info).uarch.is_null() } {
            return Err(ChipDbError::NullRequiredStringPointer { field: "uarch" });
        }

        let (constid_strs, known_id_count) = unsafe { build_constid_table(chip_info) };

        Ok(Self {
            _mmap: mmap,
            chip_info,
            constid_strs,
            known_id_count,
        })
    }

    #[cfg(feature = "test-utils")]
    pub unsafe fn from_bytes(bytes: &[u8]) -> Result<Self, ChipDbError> {
        let min_size = std::mem::size_of::<RelPtr<ChipInfoPod>>();
        if bytes.len() < min_size {
            return Err(ChipDbError::TooSmall {
                size: bytes.len(),
                min: min_size,
            });
        }

        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = std::env::temp_dir().join(format!(
            "nextpnr_chipdb_test_{}_{}.bin",
            std::process::id(),
            id,
        ));
        let mut tmpfile = std::fs::File::create(&tmp_path).map_err(ChipDbError::Io)?;
        tmpfile.write_all(bytes).map_err(ChipDbError::Io)?;
        drop(tmpfile);
        let file = std::fs::File::open(&tmp_path).map_err(ChipDbError::Io)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let _ = std::fs::remove_file(&tmp_path);

        let chip_info = validate_and_follow_root_relptr(mmap.as_ptr(), mmap.len())?;

        let magic = read_packed!(*chip_info, magic);
        if magic != CHIPDB_MAGIC {
            return Err(ChipDbError::MagicMismatch {
                expected: CHIPDB_MAGIC as u32,
                got: magic as u32,
            });
        }

        let version = read_packed!(*chip_info, version);
        if version != CHIPDB_VERSION {
            return Err(ChipDbError::VersionMismatch {
                expected: CHIPDB_VERSION,
                got: version,
            });
        }

        if (*chip_info).name.is_null() {
            return Err(ChipDbError::NullRequiredStringPointer { field: "name" });
        }
        if (*chip_info).uarch.is_null() {
            return Err(ChipDbError::NullRequiredStringPointer { field: "uarch" });
        }

        let (constid_strs, known_id_count) = build_constid_table(chip_info);

        Ok(Self {
            _mmap: mmap,
            chip_info,
            constid_strs,
            known_id_count,
        })
    }
}

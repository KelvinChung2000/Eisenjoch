use super::chip::{build_constid_table, validate_and_follow_root_relptr};
use super::*;
use crate::read_packed;
use memmap2::Mmap;
use std::path::Path;

fn compiled_constids(uarch: &str) -> Option<&'static str> {
    match uarch {
        "FABulous" => Some(include_str!("fabulous_constids.inc")),
        _ => None,
    }
}

impl ChipDb {
    /// Load a database, splicing in the `constids.inc` nextpnr compiles in for
    /// its uarch. A uarch Eisenjoch has no copy for must embed every string.
    pub fn load(path: &Path) -> Result<Self, ChipDbError> {
        Self::load_resolving(path, |uarch| {
            compiled_constids(uarch)
                .map(parse_constids_inc)
                .unwrap_or_default()
        })
    }

    /// Load a database generated for an arch whose `constids.inc` is compiled
    /// in, which is how every stock himbaechel chipdb is built.
    ///
    /// `known` must be the uarch's `constids.inc` in file order -- see
    /// [`parse_constids_inc`].
    pub fn load_with_known_constids(path: &Path, known: &[String]) -> Result<Self, ChipDbError> {
        Self::load_resolving(path, |_| known.to_vec())
    }

    fn load_resolving(
        path: &Path,
        known_for_uarch: impl FnOnce(&str) -> Vec<String>,
    ) -> Result<Self, ChipDbError> {
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

        let uarch = unsafe {
            std::ffi::CStr::from_ptr((*chip_info).uarch.get() as *const std::ffi::c_char)
        };
        let known = known_for_uarch(&uarch.to_string_lossy());
        let constid_strs = unsafe { build_constid_table(chip_info, &known)? };

        Ok(Self {
            _mmap: mmap,
            chip_info,
            constid_strs,
            constid_canon: std::sync::OnceLock::new(),
            global_network: std::sync::OnceLock::new(),
        })
    }

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

        let constid_strs = build_constid_table(chip_info, &[])?;

        Ok(Self {
            _mmap: mmap,
            chip_info,
            constid_strs,
            constid_canon: std::sync::OnceLock::new(),
            global_network: std::sync::OnceLock::new(),
        })
    }
}

//! Memory-mapped zero-copy chip database reader for nextpnr-himbaechel.
//!
//! This crate provides a zero-copy, memory-mapped reader for the binary chip
//! database format (.bin files) used by nextpnr-himbaechel. The database describes
//! all BELs (basic logic elements), wires, and PIPs (programmable interconnect
//! points) on an FPGA.
//!
//! All POD (Plain Old Data) structs are `#[repr(C, packed)]` to match the C++
//! binary format exactly. Self-referential pointers within the binary data are
//! represented using [`RelPtr`] and [`RelSlice`], which store offsets relative
//! to their own address.

mod pod;
mod relptr;

pub use pod::*;
pub use relptr::{RelPtr, RelSlice};

use std::ffi::CStr;
use std::path::Path;

use memmap2::Mmap;
use npnr_types::{BelId, Loc, PipId, WireId};

/// Expected magic/version number at the start of the chipdb binary file.
/// This matches the C++ HIMBAECHEL_CHIPDB_VERSION constant.
const CHIPDB_VERSION: i32 = 2;

/// Error type for chip database loading failures.
#[derive(Debug, thiserror::Error)]
pub enum ChipDbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("chip database file too small ({size} bytes, minimum {min} bytes)")]
    TooSmall { size: usize, min: usize },
    #[error("chip database version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: i32, got: i32 },
}

/// Read a field from a packed struct via `read_unaligned`.
///
/// This is necessary because creating a reference to a field in a `#[repr(packed)]`
/// struct may be undefined behavior if the field type has alignment > 1.
///
/// # Safety
/// The caller must wrap the call in an `unsafe` block and ensure the pointer
/// to the struct is valid and the struct is properly initialized.
///
/// # Example
/// ```ignore
/// let inst: &TileInstPod = ...;
/// let x: i16 = unsafe { read_packed!(*inst, x) };
/// ```
#[macro_export]
macro_rules! read_packed {
    ($base:expr, $field:ident) => {
        std::ptr::read_unaligned(std::ptr::addr_of!((*std::ptr::addr_of!($base)).$field))
    };
}

/// Memory-mapped chip database.
///
/// Owns the memory map and provides safe access to the chip information.
/// The database is read-only after construction.
pub struct ChipDb {
    _mmap: Mmap,
    chip_info: *const ChipInfoPod,
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

// SAFETY: The mmap is read-only and the pointer is derived from it.
// The data is never mutated after construction.
unsafe impl Send for ChipDb {}
unsafe impl Sync for ChipDb {}

impl ChipDb {
    /// Load a chip database from a `.bin` file path.
    pub fn load(path: &Path) -> Result<Self, ChipDbError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let min_size = std::mem::size_of::<ChipInfoPod>();
        if mmap.len() < min_size {
            return Err(ChipDbError::TooSmall {
                size: mmap.len(),
                min: min_size,
            });
        }

        let chip_info = mmap.as_ptr() as *const ChipInfoPod;

        // SAFETY: We verified the mmap is large enough for ChipInfoPod.
        let version = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*chip_info).version)) };
        if version != CHIPDB_VERSION {
            return Err(ChipDbError::VersionMismatch {
                expected: CHIPDB_VERSION,
                got: version,
            });
        }

        Ok(Self {
            _mmap: mmap,
            chip_info,
        })
    }

    /// Create a ChipDb from raw bytes (for testing).
    ///
    /// # Safety
    /// The caller must ensure the bytes represent a valid chipdb binary.
    #[cfg(any(test, feature = "test-utils"))]
    pub unsafe fn from_bytes(bytes: &[u8]) -> Result<Self, ChipDbError> {
        let min_size = std::mem::size_of::<ChipInfoPod>();
        if bytes.len() < min_size {
            return Err(ChipDbError::TooSmall {
                size: bytes.len(),
                min: min_size,
            });
        }

        let chip_info = bytes.as_ptr() as *const ChipInfoPod;
        let version = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*chip_info).version)) };
        if version != CHIPDB_VERSION {
            return Err(ChipDbError::VersionMismatch {
                expected: CHIPDB_VERSION,
                got: version,
            });
        }

        // Write bytes to a temporary file so we can mmap it.
        // Use a counter to avoid collisions between parallel tests.
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = std::env::temp_dir().join(format!(
            "npnr_chipdb_test_{}_{}.bin",
            std::process::id(),
            id,
        ));
        let mut tmpfile = std::fs::File::create(&tmp_path).map_err(ChipDbError::Io)?;
        tmpfile.write_all(bytes).map_err(ChipDbError::Io)?;
        drop(tmpfile);
        let file = std::fs::File::open(&tmp_path).map_err(ChipDbError::Io)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Clean up the temp file (the mmap keeps the data alive).
        let _ = std::fs::remove_file(&tmp_path);
        let chip_info = mmap.as_ptr() as *const ChipInfoPod;

        Ok(Self {
            _mmap: mmap,
            chip_info,
        })
    }

    /// Get the root chip info.
    #[inline]
    pub fn chip_info(&self) -> &ChipInfoPod {
        // SAFETY: chip_info was validated at construction time.
        unsafe { &*self.chip_info }
    }

    /// Get tile type info by tile index.
    ///
    /// Looks up the tile instance at the given index, then returns the
    /// corresponding tile type.
    #[inline]
    pub fn tile_type(&self, tile: i32) -> &TileTypePod {
        let ci = self.chip_info();
        let inst = &ci.tile_insts.get()[tile as usize];
        // SAFETY: inst is a valid reference into the mmap'd chipdb.
        let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
        &ci.tile_types.get()[tt_idx as usize]
    }

    /// Get the tile type index for a given tile instance.
    #[inline]
    pub fn tile_type_index(&self, tile: i32) -> i32 {
        let inst = &self.chip_info().tile_insts.get()[tile as usize];
        // SAFETY: inst is a valid reference into the mmap'd chipdb.
        unsafe { read_packed!(*inst, tile_type) }
    }

    /// Get bel info from a BelId.
    #[inline]
    pub fn bel_info(&self, bel: BelId) -> &BelDataPod {
        let tt = self.tile_type(bel.tile());
        &tt.bels.get()[bel.index() as usize]
    }

    /// Get wire info from a WireId.
    #[inline]
    pub fn wire_info(&self, wire: WireId) -> &TileWireDataPod {
        let tt = self.tile_type(wire.tile());
        &tt.wires.get()[wire.index() as usize]
    }

    /// Get pip info from a PipId.
    #[inline]
    pub fn pip_info(&self, pip: PipId) -> &PipDataPod {
        let tt = self.tile_type(pip.tile());
        &tt.pips.get()[pip.index() as usize]
    }

    /// Get tile coordinates (x, y) from a tile index.
    #[inline]
    pub fn tile_xy(&self, tile: i32) -> (i32, i32) {
        let inst = &self.chip_info().tile_insts.get()[tile as usize];
        // SAFETY: inst is a valid reference into the mmap'd chipdb.
        let x: i16 = unsafe { read_packed!(*inst, x) };
        let y: i16 = unsafe { read_packed!(*inst, y) };
        (x as i32, y as i32)
    }

    /// Find tile index by (x, y) coordinates.
    ///
    /// Computes `y * width + x` which is the standard row-major index.
    #[inline]
    pub fn tile_by_xy(&self, x: i32, y: i32) -> i32 {
        y * self.width() + x
    }

    /// Resolve a relative tile offset: given base tile + (dx, dy) delta,
    /// return the target tile index.
    #[inline]
    pub fn rel_tile(&self, base: i32, dx: i32, dy: i32) -> i32 {
        let (bx, by) = self.tile_xy(base);
        self.tile_by_xy(bx + dx, by + dy)
    }

    /// Grid width.
    #[inline]
    pub fn width(&self) -> i32 {
        // SAFETY: chip_info is a valid pointer established at construction.
        unsafe { read_packed!(*self.chip_info(), width) }
    }

    /// Grid height.
    #[inline]
    pub fn height(&self) -> i32 {
        // SAFETY: chip_info is a valid pointer established at construction.
        unsafe { read_packed!(*self.chip_info(), height) }
    }

    /// Total tile count.
    #[inline]
    pub fn num_tiles(&self) -> i32 {
        // SAFETY: chip_info is a valid pointer established at construction.
        unsafe { read_packed!(*self.chip_info(), num_tiles) }
    }

    /// Chip name as a string.
    pub fn name(&self) -> &str {
        let ptr = self.chip_info().name.get();
        // SAFETY: The chipdb binary is expected to contain valid null-terminated
        // ASCII strings.
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    /// Iterate all BELs across all tiles.
    pub fn bels(&self) -> impl Iterator<Item = BelId> + '_ {
        let ci = self.chip_info();
        let tile_insts = ci.tile_insts.get();
        let tile_types = ci.tile_types.get();

        tile_insts.iter().enumerate().flat_map(move |(tile_idx, inst)| {
            // SAFETY: inst is a valid reference into the mmap'd chipdb.
            let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
            let tt = &tile_types[tt_idx as usize];
            let num_bels = tt.bels.get().len();
            (0..num_bels).map(move |bel_idx| BelId::new(tile_idx as i32, bel_idx as i32))
        })
    }

    /// Iterate all wires across all tiles.
    pub fn wires(&self) -> impl Iterator<Item = WireId> + '_ {
        let ci = self.chip_info();
        let tile_insts = ci.tile_insts.get();
        let tile_types = ci.tile_types.get();

        tile_insts.iter().enumerate().flat_map(move |(tile_idx, inst)| {
            // SAFETY: inst is a valid reference into the mmap'd chipdb.
            let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
            let tt = &tile_types[tt_idx as usize];
            let num_wires = tt.wires.get().len();
            (0..num_wires).map(move |wire_idx| WireId::new(tile_idx as i32, wire_idx as i32))
        })
    }

    /// Iterate all PIPs across all tiles.
    pub fn pips(&self) -> impl Iterator<Item = PipId> + '_ {
        let ci = self.chip_info();
        let tile_insts = ci.tile_insts.get();
        let tile_types = ci.tile_types.get();

        tile_insts.iter().enumerate().flat_map(move |(tile_idx, inst)| {
            // SAFETY: inst is a valid reference into the mmap'd chipdb.
            let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
            let tt = &tile_types[tt_idx as usize];
            let num_pips = tt.pips.get().len();
            (0..num_pips).map(move |pip_idx| PipId::new(tile_idx as i32, pip_idx as i32))
        })
    }

    /// Get BEL name as a string.
    pub fn bel_name(&self, bel: BelId) -> &str {
        let info = self.bel_info(bel);
        let ptr = info.name.get();
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    /// Get BEL type as a string.
    pub fn bel_type(&self, bel: BelId) -> &str {
        let info = self.bel_info(bel);
        let ptr = info.bel_type.get();
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    /// Get BEL bucket (placement category) as a string.
    pub fn bel_bucket(&self, bel: BelId) -> &str {
        let info = self.bel_info(bel);
        let ptr = info.bucket.get();
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    /// Get BEL location (x, y, z).
    pub fn bel_loc(&self, bel: BelId) -> Loc {
        let (x, y) = self.tile_xy(bel.tile());
        let info = self.bel_info(bel);
        // SAFETY: info is a valid reference into the mmap'd chipdb.
        let z: i16 = unsafe { read_packed!(*info, z) };
        Loc::new(x, y, z as i32)
    }

    /// Get the source wire of a PIP.
    pub fn pip_src_wire(&self, pip: PipId) -> WireId {
        let info = self.pip_info(pip);
        // SAFETY: info is a valid reference into the mmap'd chipdb.
        let src_tile_delta: i16 = unsafe { read_packed!(*info, src_tile_delta) };
        let src_wire: i32 = unsafe { read_packed!(*info, src_wire) };
        let src_tile = self.rel_tile(pip.tile(), src_tile_delta as i32, 0);
        WireId::new(src_tile, src_wire)
    }

    /// Get the destination wire of a PIP.
    pub fn pip_dst_wire(&self, pip: PipId) -> WireId {
        let info = self.pip_info(pip);
        // SAFETY: info is a valid reference into the mmap'd chipdb.
        let dst_tile_delta: i16 = unsafe { read_packed!(*info, dst_tile_delta) };
        let dst_wire: i32 = unsafe { read_packed!(*info, dst_wire) };
        let dst_tile = self.rel_tile(pip.tile(), dst_tile_delta as i32, 0);
        WireId::new(dst_tile, dst_wire)
    }

    /// Get tile instance info.
    #[inline]
    pub fn tile_inst(&self, tile: i32) -> &TileInstPod {
        &self.chip_info().tile_insts.get()[tile as usize]
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub mod testutil;

#[cfg(test)]
mod tests;

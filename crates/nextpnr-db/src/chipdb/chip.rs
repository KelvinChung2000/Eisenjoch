//! ChipDb struct and associated validation/construction functions.

#[cfg(feature = "mmap")]
use memmap2::Mmap;
use zerocopy::IntoBytes;

use super::pod::ChipInfoPod;
use super::validate::validate;
use super::ChipDbError;
use super::WireId;
use crate::read_packed;

/// One entry of the resolved constid table.
///
/// A himbaechel chipdb only embeds the strings the arch does *not* already
/// know: `known_id_count` ids are compiled into the C++ binary from the uarch's
/// `constids.inc`, and the file stores only what comes after them. So the table
/// is genuinely mixed -- borrowed pointers into the backing bytes for embedded strings,
/// owned strings for the ones the caller had to supply.
pub(super) enum ConstIdStr {
    /// Points into the backing bytes; valid for as long as the `ChipDb` lives.
    Embedded(*const u8),
    /// Supplied by the caller from the arch's `constids.inc`.
    Known(String),
    /// A null entry in the file's id table.
    Missing,
}

/// The bytes a `ChipDb` points into, 8-byte aligned so `RelSlice<i32>` can be.
pub(super) enum Backing {
    Owned {
        words: Vec<u64>,
        len: usize,
    },
    #[cfg(feature = "mmap")]
    Mapped(Mmap),
}

impl Backing {
    fn copy_of(bytes: &[u8]) -> Self {
        let mut words = vec![0u64; bytes.len().div_ceil(size_of::<u64>())];
        words.as_mut_bytes()[..bytes.len()].copy_from_slice(bytes);
        Self::Owned {
            words,
            len: bytes.len(),
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned { words, len } => &words.as_bytes()[..*len],
            #[cfg(feature = "mmap")]
            Self::Mapped(mmap) => mmap,
        }
    }
}

pub struct ChipDb {
    pub(super) _backing: Backing,
    pub(super) chip_info: *const ChipInfoPod,
    pub(super) constid_strs: Vec<ConstIdStr>,
    /// Built on first use. See [`ChipDb::constid_canon`].
    pub(super) constid_canon: std::sync::OnceLock<ConstIdCanon>,
    /// Built on first use. See [`ChipDb::global_network`].
    pub(super) global_network: std::sync::OnceLock<rustc_hash::FxHashSet<WireId>>,
    pub(super) names: std::sync::OnceLock<super::names::NameIndex>,
}

/// Constids that spell the same string collapsed to one representative, so a
/// name comparison becomes an integer comparison.
///
/// `bel_pin_wire` matched a port against a BEL's pins by resolving both sides
/// to `&str` and calling `memcmp` for each pin of each candidate BEL, which the
/// legalizer does millions of times.
pub struct ConstIdCanon {
    /// `canon[i]` is the lowest constid spelling the same string as `i`.
    pub(super) canon: Vec<i32>,
    /// Name to canonical constid.
    pub(super) by_name: rustc_hash::FxHashMap<String, i32>,
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

impl ChipDb {
    /// Load a database from bytes, splicing in the `constids.inc` nextpnr
    /// compiles in for its uarch. A uarch Eisenjoch has no copy for must embed
    /// every string.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChipDbError> {
        Self::open(Backing::copy_of(bytes), compiled_constids)
    }

    /// Load a database from bytes with the uarch's `constids.inc` in file order.
    pub fn from_bytes_with_known_constids(
        bytes: &[u8],
        known: &[String],
    ) -> Result<Self, ChipDbError> {
        Self::open(Backing::copy_of(bytes), |_| known.to_vec())
    }

    pub(super) fn open(
        backing: Backing,
        known_for_uarch: impl FnOnce(&str) -> Vec<String>,
    ) -> Result<Self, ChipDbError> {
        let bytes = backing.bytes();
        let validated = validate(bytes)?;
        let known = known_for_uarch(&String::from_utf8_lossy(validated.uarch));
        let chip_info = bytes.as_ptr().wrapping_add(validated.chip_info) as *const ChipInfoPod;
        let constid_strs = unsafe { build_constid_table(chip_info, &known)? };
        Ok(Self {
            _backing: backing,
            chip_info,
            constid_strs,
            constid_canon: std::sync::OnceLock::new(),
            global_network: std::sync::OnceLock::new(),
            names: std::sync::OnceLock::new(),
        })
    }
}

pub(super) fn compiled_constids(uarch: &str) -> Vec<String> {
    match uarch {
        "FABulous" => parse_constids_inc(include_str!("fabulous_constids.inc")),
        _ => Vec::new(),
    }
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
            table.push(ConstIdStr::Embedded(relptr.get()));
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

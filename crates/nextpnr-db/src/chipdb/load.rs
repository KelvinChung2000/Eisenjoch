use super::chip::{compiled_constids, Backing};
use super::*;
use memmap2::Mmap;
use std::path::Path;

impl ChipDb {
    /// Load a database, splicing in the `constids.inc` nextpnr compiles in for
    /// its uarch. A uarch Eisenjoch has no copy for must embed every string.
    pub fn load(path: &Path) -> Result<Self, ChipDbError> {
        Self::open(map(path)?, compiled_constids)
    }

    /// Load a database generated for an arch whose `constids.inc` is compiled
    /// in, which is how every stock himbaechel chipdb is built.
    ///
    /// `known` must be the uarch's `constids.inc` in file order -- see
    /// [`parse_constids_inc`].
    pub fn load_with_known_constids(path: &Path, known: &[String]) -> Result<Self, ChipDbError> {
        Self::open(map(path)?, |_| known.to_vec())
    }
}

fn map(path: &Path) -> Result<Backing, ChipDbError> {
    let file = std::fs::File::open(path)?;
    Ok(Backing::Mapped(unsafe { Mmap::map(&file)? }))
}

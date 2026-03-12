//! Checkpoint save/load/restore for incremental place-and-route.
//!
//! Users manage the incremental flow explicitly:
//!
//! ```ignore
//! // Save after a successful P&R run:
//! checkpoint::save(&ctx, &path)?;
//!
//! // In a later session, restore and re-place/re-route:
//! let cp = Checkpoint::load(&path)?;
//! let report = checkpoint::restore(&mut ctx, &cp)?;
//! // Restored cells are Fixed, so normal place()/route() skips them.
//! placer.place(&mut ctx, &cfg)?;
//! router.route(&mut ctx, &cfg)?;
//! ```

mod restore;

pub use restore::{compute_fingerprint, restore, RestoreReport};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::context::Context;

/// Version of the checkpoint format.
pub const CHECKPOINT_VERSION: u32 = 1;

/// A saved snapshot of placement and routing state.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version for compatibility checking.
    pub version: u32,
    /// Saved cell placements.
    pub placements: Vec<CellPlacement>,
    /// Saved net routes.
    pub routes: Vec<NetRoute>,
    /// Design fingerprint for change detection.
    pub fingerprint: DesignFingerprint,
}

/// Saved placement of a single cell.
#[derive(Serialize, Deserialize)]
pub struct CellPlacement {
    /// Cell name (string, not IdString index, for cross-session stability).
    pub cell_name: String,
    /// Cell type name.
    pub cell_type: String,
    /// BEL tile index.
    pub bel_tile: i32,
    /// BEL index within the tile.
    pub bel_index: i32,
    /// Human-readable BEL name (e.g., "LUT4_0").
    #[serde(default)]
    pub bel_name: String,
    /// Human-readable tile name (e.g., "CLB_X1Y2").
    #[serde(default)]
    pub tile_name: String,
    /// Human-readable tile type (e.g., "CLB").
    #[serde(default)]
    pub tile_type: String,
    /// Placement strength (as u8, maps to PlaceStrength).
    pub strength: u8,
}

/// Saved route of a single net.
#[derive(Serialize, Deserialize)]
pub struct NetRoute {
    /// Net name.
    pub net_name: String,
    /// Source wire tile index.
    pub source_wire_tile: i32,
    /// Source wire index within the tile.
    pub source_wire_index: i32,
    /// Human-readable source wire name (e.g., "CLB_X1Y2/O0").
    #[serde(default)]
    pub source_wire_name: String,
    /// Sequence of PIPs as (tile, index) pairs.
    pub pips: Vec<(i32, i32)>,
    /// Human-readable PIP names (e.g., "CLB_X0Y3/dst.src").
    #[serde(default)]
    pub pip_names: Vec<String>,
}

/// Fingerprint of a design for detecting changes between sessions.
#[derive(Serialize, Deserialize)]
pub struct DesignFingerprint {
    /// Sorted cell signatures.
    pub cell_signatures: Vec<CellSig>,
    /// Sorted net signatures.
    pub net_signatures: Vec<NetSig>,
}

/// Signature of a single cell for change detection.
#[derive(Serialize, Deserialize, Clone)]
pub struct CellSig {
    /// Cell name.
    pub name: String,
    /// Cell type.
    pub cell_type: String,
    /// Number of ports.
    pub port_count: usize,
}

/// Signature of a single net for change detection.
#[derive(Serialize, Deserialize, Clone)]
pub struct NetSig {
    /// Net name.
    pub name: String,
    /// Driver cell name (empty if no driver).
    pub driver_cell: String,
    /// Driver port name (empty if no driver).
    pub driver_port: String,
    /// Number of users.
    pub user_count: usize,
}

/// Differences between two design fingerprints.
pub struct DesignDiff {
    /// Cells present in the new design but not the old.
    pub added_cells: Vec<String>,
    /// Cells present in the old design but not the new.
    pub removed_cells: Vec<String>,
    /// Cells present in both but with different signatures.
    pub changed_cells: Vec<String>,
    /// Nets present in the new design but not the old.
    pub added_nets: Vec<String>,
    /// Nets present in the old design but not the new.
    pub removed_nets: Vec<String>,
    /// Nets present in both but with different signatures.
    pub changed_nets: Vec<String>,
}

/// Diff two slices of named items, returning (added, removed, changed) name lists.
fn diff_by_name<'a, T>(
    old_items: &'a [T],
    new_items: &'a [T],
    name_of: impl Fn(&'a T) -> &'a str,
    is_equal: impl Fn(&T, &T) -> bool,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let old_map: FxHashMap<&str, &T> = old_items.iter().map(|item| (name_of(item), item)).collect();
    let new_map: FxHashMap<&str, &T> = new_items.iter().map(|item| (name_of(item), item)).collect();

    let mut added = Vec::new();
    let mut changed = Vec::new();

    for (&name, new_item) in &new_map {
        match old_map.get(name) {
            Some(old_item) if !is_equal(old_item, new_item) => {
                changed.push(name.to_string());
            }
            None => added.push(name.to_string()),
            _ => {}
        }
    }

    let removed: Vec<String> = old_map
        .keys()
        .filter(|name| !new_map.contains_key(*name))
        .map(|name| name.to_string())
        .collect();

    (added, removed, changed)
}

impl DesignDiff {
    /// Compute the diff between an old and new fingerprint.
    pub fn compute(old: &DesignFingerprint, new: &DesignFingerprint) -> Self {
        let (added_cells, removed_cells, changed_cells) = diff_by_name(
            &old.cell_signatures,
            &new.cell_signatures,
            |c| c.name.as_str(),
            |a, b| a.cell_type == b.cell_type && a.port_count == b.port_count,
        );

        let (added_nets, removed_nets, changed_nets) = diff_by_name(
            &old.net_signatures,
            &new.net_signatures,
            |n| n.name.as_str(),
            |a, b| {
                a.driver_cell == b.driver_cell
                    && a.driver_port == b.driver_port
                    && a.user_count == b.user_count
            },
        );

        Self {
            added_cells,
            removed_cells,
            changed_cells,
            added_nets,
            removed_nets,
            changed_nets,
        }
    }

    /// Returns true if there are any differences.
    pub fn has_changes(&self) -> bool {
        !self.added_cells.is_empty()
            || !self.removed_cells.is_empty()
            || !self.changed_cells.is_empty()
            || !self.added_nets.is_empty()
            || !self.removed_nets.is_empty()
            || !self.changed_nets.is_empty()
    }
}

impl Checkpoint {
    /// Save a checkpoint to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), CheckpointError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CheckpointError::SerializationFailed(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| CheckpointError::IoFailed(e.to_string()))?;
        Ok(())
    }

    /// Load a checkpoint from a JSON file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, CheckpointError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| CheckpointError::IoFailed(e.to_string()))?;
        let checkpoint: Self = serde_json::from_str(&json)
            .map_err(|e| CheckpointError::DeserializationFailed(e.to_string()))?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                found: checkpoint.version,
            });
        }
        Ok(checkpoint)
    }
}

/// Save the current placement and routing state as a checkpoint.
pub fn save(ctx: &Context, path: &std::path::Path) -> Result<(), CheckpointError> {
    let checkpoint = build_checkpoint(ctx);
    checkpoint.save_to_file(path)
}

/// Build a checkpoint from the current context state.
fn build_checkpoint(ctx: &Context) -> Checkpoint {
    let chipdb = ctx.chipdb();
    let placements: Vec<CellPlacement> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(_, cell)| {
            let bel = cell.bel?;
            Some(CellPlacement {
                cell_name: ctx.name_of(cell.name).to_owned(),
                cell_type: ctx.name_of(cell.cell_type).to_owned(),
                bel_tile: bel.tile(),
                bel_index: bel.index(),
                bel_name: chipdb.bel_name(bel).to_owned(),
                tile_name: chipdb.tile_name(bel.tile()),
                tile_type: chipdb.tile_type_name(bel.tile()).to_owned(),
                strength: cell.bel_strength as u8,
            })
        })
        .collect();

    let routes: Vec<NetRoute> = ctx
        .design
        .iter_alive_nets()
        .filter(|(_, net)| !net.wires.is_empty())
        .map(|(_, net)| {
            let mut source_wire_tile = 0i32;
            let mut source_wire_index = 0i32;
            let mut source_wire_name = String::new();
            let mut pips = Vec::new();
            let mut pip_names = Vec::new();

            for (&wire, pm) in &net.wires {
                match pm.pip {
                    None => {
                        source_wire_tile = wire.tile();
                        source_wire_index = wire.index();
                        source_wire_name = format!(
                            "{}/{}",
                            chipdb.tile_name(wire.tile()),
                            chipdb.wire_name(wire),
                        );
                    }
                    Some(pip) => {
                        pips.push((pip.tile(), pip.index()));
                        let src = chipdb.pip_src_wire(pip);
                        let dst = chipdb.pip_dst_wire(pip);
                        pip_names.push(format!(
                            "{}/{}.{}",
                            chipdb.tile_name(pip.tile()),
                            chipdb.wire_name(dst),
                            chipdb.wire_name(src),
                        ));
                    }
                }
            }

            NetRoute {
                net_name: ctx.name_of(net.name).to_owned(),
                source_wire_tile,
                source_wire_index,
                source_wire_name,
                pips,
                pip_names,
            }
        })
        .collect();

    Checkpoint {
        version: CHECKPOINT_VERSION,
        placements,
        routes,
        fingerprint: compute_fingerprint(ctx),
    }
}

/// Errors related to checkpoint operations.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("Checkpoint serialization failed: {0}")]
    SerializationFailed(String),
    #[error("Checkpoint deserialization failed: {0}")]
    DeserializationFailed(String),
    #[error("Checkpoint I/O failed: {0}")]
    IoFailed(String),
    #[error("Checkpoint version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },
    #[error("Checkpoint restore failed: {0}")]
    RestoreFailed(String),
}

//! Checkpoint types and data structures.

use serde::{Deserialize, Serialize};

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
    /// Human-readable BEL name (e.g., "LUT6_0").
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

impl DesignDiff {
    /// Compute the diff between an old and new fingerprint.
    pub fn compute(old: &DesignFingerprint, new: &DesignFingerprint) -> Self {
        let (added_cells, removed_cells, changed_cells) = super::diff::diff_by_name(
            &old.cell_signatures,
            &new.cell_signatures,
            |c| c.name.as_str(),
            |a, b| a.cell_type == b.cell_type && a.port_count == b.port_count,
        );

        let (added_nets, removed_nets, changed_nets) = super::diff::diff_by_name(
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
        std::fs::write(path, json).map_err(|e| CheckpointError::IoFailed(e.to_string()))?;
        Ok(())
    }

    /// Load a checkpoint from a JSON file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, CheckpointError> {
        let json =
            std::fs::read_to_string(path).map_err(|e| CheckpointError::IoFailed(e.to_string()))?;
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

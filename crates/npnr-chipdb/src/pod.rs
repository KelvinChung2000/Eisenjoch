//! Plain Old Data (POD) struct definitions matching the nextpnr-himbaechel
//! binary chip database format.
//!
//! All structs are `#[repr(C, packed)]` to guarantee exact memory layout
//! compatibility with the C++ definitions. Fields must never be reordered.

use crate::relptr::{RelPtr, RelSlice};

// =============================================================================
// Top-level chip info
// =============================================================================

/// Root structure of the chip database.
///
/// Contains grid dimensions, references to all tile types and instances,
/// routing nodes, package information, and speed grade timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct ChipInfoPod {
    /// Format version number (must match CHIPDB_VERSION).
    pub version: i32,
    /// Grid width in tiles.
    pub width: i32,
    /// Grid height in tiles.
    pub height: i32,
    /// Total number of tile instances.
    pub num_tiles: i32,
    /// Chip name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Generator tool name (null-terminated string).
    pub generator: RelPtr<u8>,
    /// Array of tile type definitions.
    pub tile_types: RelSlice<TileTypePod>,
    /// Array of tile instances (one per grid position).
    pub tile_insts: RelSlice<TileInstPod>,
    /// Global routing node shapes.
    pub nodes: RelSlice<NodeShapePod>,
    /// Package definitions (pin mappings).
    pub packages: RelSlice<PackageInfoPod>,
    /// Speed grade timing definitions.
    pub speed_grades: RelSlice<SpeedGradePod>,
    /// Optional extra data pointer.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Tile types and instances
// =============================================================================

/// Definition of a tile type.
///
/// A tile type defines the set of BELs, wires, and PIPs available in tiles
/// of this type. Multiple tile instances may share the same tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileTypePod {
    /// Tile type name (null-terminated string).
    pub name: RelPtr<u8>,
    /// BELs in this tile type.
    pub bels: RelSlice<BelDataPod>,
    /// Wires in this tile type.
    pub wires: RelSlice<TileWireDataPod>,
    /// PIPs in this tile type.
    pub pips: RelSlice<PipDataPod>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

/// A tile instance in the grid.
///
/// Each tile instance refers to a tile type and has a position in the grid.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileInstPod {
    /// Tile instance name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Index into `ChipInfoPod::tile_types`.
    pub tile_type: i32,
    /// Per-wire mapping to global routing node indices (-1 = no node).
    pub tilewire_to_node: RelSlice<i32>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
    /// X coordinate in the grid.
    pub x: i16,
    /// Y coordinate in the grid.
    pub y: i16,
}

// =============================================================================
// BEL (Basic Element of Logic)
// =============================================================================

/// BEL definition within a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelDataPod {
    /// BEL name (null-terminated string).
    pub name: RelPtr<u8>,
    /// BEL type (null-terminated string, e.g. "LUT4", "FF").
    pub bel_type: RelPtr<u8>,
    /// BEL bucket for placement grouping (null-terminated string).
    pub bucket: RelPtr<u8>,
    /// Pin definitions for this BEL.
    pub pins: RelSlice<BelPinPod>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
    /// Z position within the tile.
    pub z: i16,
    /// Padding for alignment.
    pub padding: i16,
}

/// A pin on a BEL.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelPinPod {
    /// Pin name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Index into the tile type's wire array.
    pub wire_index: i32,
    /// Port direction (PortType as i32).
    pub dir: i32,
}

// =============================================================================
// Wires
// =============================================================================

/// Wire definition within a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileWireDataPod {
    /// Wire name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Wire type/category (null-terminated string).
    pub wire_type: RelPtr<u8>,
    /// PIPs that drive this wire (uphill = towards this wire's destination).
    pub pips_uphill: RelSlice<PipRefPod>,
    /// PIPs driven by this wire (downhill = from this wire as source).
    pub pips_downhill: RelSlice<PipRefPod>,
    /// BEL pins connected to this wire.
    pub bel_pins: RelSlice<BelPinRefPod>,
    /// Wire flags.
    pub flags: i32,
}

/// Reference to a PIP from a wire's perspective.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PipRefPod {
    /// Relative tile offset from the wire's tile to the PIP's tile.
    pub tile_delta: i32,
    /// PIP index within the target tile's pip array.
    pub index: i32,
}

/// Reference to a BEL pin from a wire's perspective.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelPinRefPod {
    /// BEL index within the tile.
    pub bel: i32,
    /// Pin name (null-terminated string).
    pub pin: RelPtr<u8>,
}

// =============================================================================
// PIPs (Programmable Interconnect Points)
// =============================================================================

/// PIP definition within a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PipDataPod {
    /// Source wire index in the tile's wire array.
    pub src_wire: i32,
    /// Destination wire index in the tile's wire array.
    pub dst_wire: i32,
    /// Index into timing data (-1 if no timing data).
    pub timing_index: i32,
    /// PIP type/class.
    pub pip_type: i16,
    /// Padding for alignment.
    pub padding: i16,
    /// Delta for the source wire's tile (base + delta).
    pub src_tile_delta: i16,
    /// Delta for the destination wire's tile (base + delta).
    pub dst_tile_delta: i16,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Global routing nodes
// =============================================================================

/// A routing node shape: a set of wires across tiles that form a single
/// logical routing node.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct NodeShapePod {
    /// Wire references that make up this node.
    pub wires: RelSlice<RelNodeRefPod>,
}

/// A wire reference within a node shape, using tile deltas relative to
/// a reference tile.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct RelNodeRefPod {
    /// Delta X from the reference tile.
    pub tile_delta_x: i16,
    /// Delta Y from the reference tile.
    pub tile_delta_y: i16,
    /// Wire index in the target tile's wire array.
    pub wire_index: i16,
    /// Padding for alignment.
    pub padding: i16,
}

// =============================================================================
// Package info
// =============================================================================

/// Package definition.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PackageInfoPod {
    /// Package name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Pad mappings.
    pub pads: RelSlice<PadInfoPod>,
}

/// Pad/pin info within a package.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PadInfoPod {
    /// Tile index where this pad is located.
    pub tile: i32,
    /// BEL index within the tile.
    pub bel: i32,
    /// Pin function name (null-terminated string).
    pub function: RelPtr<u8>,
    /// I/O bank number.
    pub bank: i32,
}

// =============================================================================
// Speed grade / timing
// =============================================================================

/// Speed grade definition with timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct SpeedGradePod {
    /// Speed grade name (null-terminated string).
    pub name: RelPtr<u8>,
    /// PIP timing classes.
    pub pip_classes: RelSlice<PipTimingPod>,
    /// Cell timing entries.
    pub cell_timings: RelSlice<CellTimingPod>,
}

/// PIP timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PipTimingPod {
    /// Minimum delay in picoseconds.
    pub min_delay: i32,
    /// Maximum delay in picoseconds.
    pub max_delay: i32,
    /// Minimum fanout adder in picoseconds.
    pub min_fanout_adder: i32,
    /// Maximum fanout adder in picoseconds.
    pub max_fanout_adder: i32,
}

/// Cell timing data for a particular cell type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellTimingPod {
    /// Cell type name (null-terminated string).
    pub cell_type: RelPtr<u8>,
    /// Propagation delays.
    pub prop_delays: RelSlice<CellPropDelayPod>,
    /// Setup/hold constraints.
    pub setup_holds: RelSlice<CellSetupHoldPod>,
}

/// Cell propagation delay from one port to another.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellPropDelayPod {
    /// Source port name (null-terminated string).
    pub from_port: RelPtr<u8>,
    /// Destination port name (null-terminated string).
    pub to_port: RelPtr<u8>,
    /// Minimum delay in picoseconds.
    pub min_delay: i32,
    /// Maximum delay in picoseconds.
    pub max_delay: i32,
}

/// Cell setup/hold timing constraint.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellSetupHoldPod {
    /// Signal name (null-terminated string).
    pub signal: RelPtr<u8>,
    /// Clock name (null-terminated string).
    pub clock: RelPtr<u8>,
    /// Minimum setup time in picoseconds.
    pub min_setup: i32,
    /// Maximum setup time in picoseconds.
    pub max_setup: i32,
    /// Minimum hold time in picoseconds.
    pub min_hold: i32,
    /// Maximum hold time in picoseconds.
    pub max_hold: i32,
}

// =============================================================================
// Static size assertions
// =============================================================================

/// Compile-time size checks to ensure our struct layouts match the C++ binary format.
///
/// These use const assertions so they fail at compile time if sizes change.
const _: () = {
    assert!(std::mem::size_of::<RelPtr<u8>>() == 4);
    assert!(std::mem::size_of::<RelSlice<u8>>() == 8);

    // ChipInfoPod: 4+4+4+4 + 4+4 + 8+8+8+8+8 + 4 = 68
    assert!(std::mem::size_of::<ChipInfoPod>() == 68);

    // TileTypePod: 4 + 8+8+8 + 4 = 32
    assert!(std::mem::size_of::<TileTypePod>() == 32);

    // TileInstPod: 4 + 4 + 8 + 4 + 2+2 = 24
    assert!(std::mem::size_of::<TileInstPod>() == 24);

    // BelDataPod: 4+4+4 + 8 + 4 + 2+2 = 28
    assert!(std::mem::size_of::<BelDataPod>() == 28);

    // BelPinPod: 4+4+4 = 12
    assert!(std::mem::size_of::<BelPinPod>() == 12);

    // TileWireDataPod: 4+4 + 8+8+8 + 4 = 36
    assert!(std::mem::size_of::<TileWireDataPod>() == 36);

    // PipRefPod: 4+4 = 8
    assert!(std::mem::size_of::<PipRefPod>() == 8);

    // BelPinRefPod: 4+4 = 8
    assert!(std::mem::size_of::<BelPinRefPod>() == 8);

    // PipDataPod: 4+4+4 + 2+2+2+2 + 4 = 24
    assert!(std::mem::size_of::<PipDataPod>() == 24);

    // NodeShapePod: 8
    assert!(std::mem::size_of::<NodeShapePod>() == 8);

    // RelNodeRefPod: 2+2+2+2 = 8
    assert!(std::mem::size_of::<RelNodeRefPod>() == 8);

    // PackageInfoPod: 4 + 8 = 12
    assert!(std::mem::size_of::<PackageInfoPod>() == 12);

    // PadInfoPod: 4+4+4+4 = 16
    assert!(std::mem::size_of::<PadInfoPod>() == 16);

    // SpeedGradePod: 4 + 8+8 = 20
    assert!(std::mem::size_of::<SpeedGradePod>() == 20);

    // PipTimingPod: 4+4+4+4 = 16
    assert!(std::mem::size_of::<PipTimingPod>() == 16);

    // CellTimingPod: 4 + 8+8 = 20
    assert!(std::mem::size_of::<CellTimingPod>() == 20);

    // CellPropDelayPod: 4+4+4+4 = 16
    assert!(std::mem::size_of::<CellPropDelayPod>() == 16);

    // CellSetupHoldPod: 4+4+4+4+4+4 = 24
    assert!(std::mem::size_of::<CellSetupHoldPod>() == 24);
};

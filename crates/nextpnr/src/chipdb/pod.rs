//! Plain Old Data (POD) struct definitions matching the nextpnr-himbaechel
//! binary chip database format (chipdb.h, database_version = 6).
//!
//! All structs are `#[repr(C, packed)]` to guarantee exact memory layout
//! compatibility with the C++ definitions. Fields must never be reordered.
//!
//! String references in POD structs are `i32` constid indices, looked up via
//! the `ConstIdDataPod` table, not inline `RelPtr<u8>` strings.

use super::relptr::{RelPtr, RelSlice};

// =============================================================================
// BEL (Basic Element of Logic)
// =============================================================================

/// A pin on a BEL.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelPinPod {
    /// Pin name (constid index).
    pub name: i32,
    /// Wire index in the tile's wire array.
    pub wire: i32,
    /// Port direction (PortType as i32).
    pub dir: i32,
}

/// BEL definition within a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelDataPod {
    /// BEL name (constid index).
    pub name: i32,
    /// BEL type (constid index, e.g. "LUT4", "FF").
    pub bel_type: i32,
    /// Z position within the tile.
    pub z: i16,
    /// Padding for alignment.
    pub padding: i16,
    /// Flags: bits [7..0] for himbaechel use, [31..8] for user use.
    pub flags: u32,
    /// 64 bits of general data (first 32 bits).
    pub site: i32,
    /// 64 bits of general data (second 32 bits).
    pub checker_idx: i32,
    /// Pin definitions for this BEL.
    pub pins: RelSlice<BelPinPod>,
    /// Optional extra data pointer.
    pub extra_data: RelPtr<u8>,
}

impl BelDataPod {
    pub const FLAG_GLOBAL: u32 = 0x01;
    pub const FLAG_HIDDEN: u32 = 0x02;
}

/// Reference to a BEL pin from a wire's perspective.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct BelPinRefPod {
    /// BEL index within the tile.
    pub bel: i32,
    /// Pin name (constid index).
    pub pin: i32,
}

// =============================================================================
// Wires
// =============================================================================

/// Wire definition within a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileWireDataPod {
    /// Wire name (constid index).
    pub name: i32,
    /// Wire type/category (constid index).
    pub wire_type: i32,
    /// Tile wire index.
    pub tile_wire: i32,
    /// Constant value.
    pub const_value: i32,
    /// 32 bits of arbitrary flags.
    pub flags: i32,
    /// Timing index; used only when wire is not part of a node.
    pub timing_idx: i32,
    /// PIPs that drive this wire (indices into tile's pip array).
    pub pips_uphill: RelSlice<i32>,
    /// PIPs driven by this wire (indices into tile's pip array).
    pub pips_downhill: RelSlice<i32>,
    /// BEL pins connected to this wire.
    pub bel_pins: RelSlice<BelPinRefPod>,
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
    /// PIP type/class.
    pub pip_type: u32,
    /// PIP flags.
    pub flags: u32,
    /// Index into timing data (-1 if no timing data).
    pub timing_idx: i32,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Global routing nodes
// =============================================================================

/// A relative wire reference within a node shape, using tile deltas.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct RelTileWireRefPod {
    /// Delta X from the reference tile.
    pub dx: i16,
    /// Delta Y from the reference tile.
    pub dy: i16,
    /// Wire index in the target tile's wire array.
    pub wire: i16,
}

/// A routing node shape: a set of wires across tiles that form a single
/// logical routing node.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct NodeShapePod {
    /// Wire references that make up this node.
    pub tile_wires: RelSlice<RelTileWireRefPod>,
    /// Timing index for this node shape.
    pub timing_idx: i32,
}

/// Groups of related elements.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct GroupDataPod {
    /// Group name (constid index).
    pub name: i32,
    /// Group type (constid index).
    pub group_type: i32,
    /// BEL indices in this group.
    pub group_bels: RelSlice<i32>,
    /// Wire indices in this group.
    pub group_wires: RelSlice<i32>,
    /// PIP indices in this group.
    pub group_pips: RelSlice<i32>,
    /// Sub-group indices.
    pub group_groups: RelSlice<i32>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Tile types and instances
// =============================================================================

/// Definition of a tile type.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileTypePod {
    /// Tile type name (constid index).
    pub type_name: i32,
    /// BELs in this tile type.
    pub bels: RelSlice<BelDataPod>,
    /// Wires in this tile type.
    pub wires: RelSlice<TileWireDataPod>,
    /// PIPs in this tile type.
    pub pips: RelSlice<PipDataPod>,
    /// Groups in this tile type.
    pub groups: RelSlice<GroupDataPod>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

/// A wire reference within a tile routing shape, using relative node refs.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct RelNodeRefPod {
    /// Relative X-coord, or a special mode value.
    pub dx_mode: i16,
    /// Normally, relative Y-coord.
    pub dy: i16,
    /// Normally, node index in tile (x+dx, y+dy).
    pub wire: u16,
}

impl RelNodeRefPod {
    /// Wire is entirely internal to a single tile.
    pub const MODE_TILE_WIRE: i16 = 0x7000;
    /// This is the root; {wire, dy} form the node shape index.
    pub const MODE_IS_ROOT: i16 = 0x7001;
    /// Special case for row constant nets.
    pub const MODE_ROW_CONST: i16 = 0x7002;
    /// Special case for global constant nets.
    pub const MODE_GLB_CONST: i16 = 0x7003;
    /// Start of user-defined special modes.
    pub const MODE_USR_BEGIN: i16 = 0x7010;
}

/// Per-tile routing shape, mapping wires to nodes.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileRoutingShapePod {
    /// Per-wire mapping to node references.
    pub wire_to_node: RelSlice<RelNodeRefPod>,
    /// Timing index for this tile shape.
    pub timing_index: i32,
}

/// A tile instance in the grid.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TileInstPod {
    /// Tile name prefix (constid index).
    pub name_prefix: i32,
    /// Index into `ChipInfoPod::tile_types`.
    pub tile_type: i32,
    /// Index into `ChipInfoPod::tile_shapes`.
    pub shape: i32,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Package info
// =============================================================================

/// Pad/pin info within a package.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PadInfoPod {
    /// Package pin name (constid index).
    pub package_pin: i32,
    /// Tile index where this pad is located.
    pub tile: i32,
    /// BEL index within the tile.
    pub bel: i32,
    /// Pin function name (constid index).
    pub pad_function: i32,
    /// I/O bank number.
    pub pad_bank: i32,
    /// Extra pad flags.
    pub flags: u32,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

/// Package definition.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PackageInfoPod {
    /// Package name (constid index).
    pub name: i32,
    /// Pad mappings.
    pub pads: RelSlice<PadInfoPod>,
    /// Optional extra data.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Speed grade / timing
// =============================================================================

/// Four-corner timing value (fast/slow x min/max).
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct TimingValue {
    pub fast_min: i32,
    pub fast_max: i32,
    pub slow_min: i32,
    pub slow_max: i32,
}

/// PIP timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PipTimingPod {
    pub int_delay: TimingValue,
    pub in_cap: TimingValue,
    pub out_res: TimingValue,
    pub flags: u32,
}

impl PipTimingPod {
    pub const UNBUFFERED: u32 = 0x1;
}

/// Node timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct NodeTimingPod {
    pub cap: TimingValue,
    pub res: TimingValue,
    pub delay: TimingValue,
}

/// Combinational timing arc for a cell pin.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellPinCombArcPod {
    /// Input pin (constid index).
    pub input: i32,
    /// Propagation delay.
    pub delay: TimingValue,
}

/// Register timing arc for a cell pin.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellPinRegArcPod {
    /// Clock pin (constid index).
    pub clock: i32,
    /// Clock edge.
    pub edge: i32,
    /// Setup time.
    pub setup: TimingValue,
    /// Hold time.
    pub hold: TimingValue,
    /// Clock-to-Q delay.
    pub clk_q: TimingValue,
}

/// Cell pin timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellPinTimingPod {
    /// Pin name (constid index).
    pub pin: i32,
    /// Flags (FLAG_CLK = 1).
    pub flags: i32,
    /// Combinational timing arcs.
    pub comb_arcs: RelSlice<CellPinCombArcPod>,
    /// Register timing arcs.
    pub reg_arcs: RelSlice<CellPinRegArcPod>,
}

impl CellPinTimingPod {
    pub const FLAG_CLK: i32 = 1;
}

/// Cell timing data for a particular cell type variant.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellTimingPod {
    /// Type variant (constid index).
    pub type_variant: i32,
    /// Pin timing entries.
    pub pins: RelSlice<CellPinTimingPod>,
}

/// Speed grade definition with timing data.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct SpeedGradePod {
    /// Speed grade name (constid index).
    pub name: i32,
    /// PIP timing classes.
    pub pip_classes: RelSlice<PipTimingPod>,
    /// Node timing classes.
    pub node_classes: RelSlice<NodeTimingPod>,
    /// Cell timing entries.
    pub cell_types: RelSlice<CellTimingPod>,
}

// =============================================================================
// Const ID data
// =============================================================================

/// Constant ID string lookup table.
///
/// Maps integer constid indices to strings via `bba_ids`.
/// `known_id_count` must be 0 (all strings embedded in the binary).
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct ConstIdDataPod {
    /// Must be 0; loading rejects chipdb files with non-zero values.
    pub known_id_count: i32,
    /// ID strings, one RelPtr per constid index.
    pub bba_ids: RelSlice<RelPtr<u8>>,
}

// =============================================================================
// Top-level chip info
// =============================================================================

/// Root structure of the chip database.
///
/// The binary file starts with a `RelPtr<ChipInfoPod>` at offset 0.
/// Follow that pointer to reach this structure.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct ChipInfoPod {
    /// Magic number (must be 0x00ca7ca7).
    pub magic: i32,
    /// Format version number (must match CHIPDB_VERSION = 6).
    pub version: i32,
    /// Grid width in tiles.
    pub width: i32,
    /// Grid height in tiles.
    pub height: i32,
    /// Micro-architecture name (null-terminated string).
    pub uarch: RelPtr<u8>,
    /// Chip name (null-terminated string).
    pub name: RelPtr<u8>,
    /// Generator tool name (null-terminated string).
    pub generator: RelPtr<u8>,
    /// Array of tile type definitions.
    pub tile_types: RelSlice<TileTypePod>,
    /// Array of tile instances (one per grid position).
    pub tile_insts: RelSlice<TileInstPod>,
    /// Global routing node shapes.
    pub node_shapes: RelSlice<NodeShapePod>,
    /// Per-tile routing shapes (wire-to-node mapping).
    pub tile_shapes: RelSlice<TileRoutingShapePod>,
    /// Package definitions (pin mappings).
    pub packages: RelSlice<PackageInfoPod>,
    /// Speed grade timing definitions.
    pub speed_grades: RelSlice<SpeedGradePod>,
    /// Constant ID string lookup data.
    pub extra_constids: RelPtr<ConstIdDataPod>,
    /// Optional extra data pointer.
    pub extra_data: RelPtr<u8>,
}

// =============================================================================
// Packing rule POD types (upstream-compatible with nextpnr C++)
// =============================================================================

/// Cell type + port pair in a packing rule (matches C++ Cell_port_POD).
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct CellPortPod {
    /// Cell type (constid index).
    pub cell_type: i32,
    /// Port name (constid index).
    pub port: i32,
}

/// A packing rule stored in chipdb extra_data (matches C++ Packing_rule_POD).
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PackingRulePod {
    pub driver: CellPortPod,
    pub user: CellPortPod,
    pub width: i32,
    pub base_z: i32,
    pub rel_x: i32,
    pub rel_y: i32,
    pub rel_z: i32,
    pub flag: i32,
}

impl PackingRulePod {
    pub const FLAG_BASE_RULE: i32 = 0x01;
    pub const FLAG_ABS_RULE: i32 = 0x02;
}

/// Chip-level extra data containing packing rules (matches C++ Chip_extra_data_POD).
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct ChipExtraDataPod {
    pub context: i32,
    pub real_bel_count: i32,
    pub packing_rules: RelSlice<PackingRulePod>,
}

// =============================================================================
// Static size assertions
// =============================================================================

const _: () = {
    assert!(std::mem::size_of::<RelPtr<u8>>() == 4);
    assert!(std::mem::size_of::<RelSlice<u8>>() == 8);

    // BelPinPod: 4+4+4 = 12
    assert!(std::mem::size_of::<BelPinPod>() == 12);
    // BelDataPod: 4+4+2+2+4+4+4+8+4 = 36
    assert!(std::mem::size_of::<BelDataPod>() == 36);
    // BelPinRefPod: 4+4 = 8
    assert!(std::mem::size_of::<BelPinRefPod>() == 8);
    // TileWireDataPod: 4+4+4+4+4+4+8+8+8 = 48
    assert!(std::mem::size_of::<TileWireDataPod>() == 48);
    // PipDataPod: 4+4+4+4+4+4 = 24
    assert!(std::mem::size_of::<PipDataPod>() == 24);
    // RelTileWireRefPod: 2+2+2 = 6
    assert!(std::mem::size_of::<RelTileWireRefPod>() == 6);
    // NodeShapePod: 8+4 = 12
    assert!(std::mem::size_of::<NodeShapePod>() == 12);
    // GroupDataPod: 4+4+8+8+8+8+4 = 44
    assert!(std::mem::size_of::<GroupDataPod>() == 44);
    // TileTypePod: 4+8+8+8+8+4 = 40
    assert!(std::mem::size_of::<TileTypePod>() == 40);
    // RelNodeRefPod: 2+2+2 = 6
    assert!(std::mem::size_of::<RelNodeRefPod>() == 6);
    // TileRoutingShapePod: 8+4 = 12
    assert!(std::mem::size_of::<TileRoutingShapePod>() == 12);
    // TileInstPod: 4+4+4+4 = 16
    assert!(std::mem::size_of::<TileInstPod>() == 16);
    // PadInfoPod: 4+4+4+4+4+4+4 = 28
    assert!(std::mem::size_of::<PadInfoPod>() == 28);
    // PackageInfoPod: 4+8+4 = 16
    assert!(std::mem::size_of::<PackageInfoPod>() == 16);
    // TimingValue: 4+4+4+4 = 16
    assert!(std::mem::size_of::<TimingValue>() == 16);
    // PipTimingPod: 16+16+16+4 = 52
    assert!(std::mem::size_of::<PipTimingPod>() == 52);
    // NodeTimingPod: 16+16+16 = 48
    assert!(std::mem::size_of::<NodeTimingPod>() == 48);
    // CellPinCombArcPod: 4+16 = 20
    assert!(std::mem::size_of::<CellPinCombArcPod>() == 20);
    // CellPinRegArcPod: 4+4+16+16+16 = 56
    assert!(std::mem::size_of::<CellPinRegArcPod>() == 56);
    // CellPinTimingPod: 4+4+8+8 = 24
    assert!(std::mem::size_of::<CellPinTimingPod>() == 24);
    // CellTimingPod: 4+8 = 12
    assert!(std::mem::size_of::<CellTimingPod>() == 12);
    // SpeedGradePod: 4+8+8+8 = 28
    assert!(std::mem::size_of::<SpeedGradePod>() == 28);
    // ConstIdDataPod: 4+8 = 12
    assert!(std::mem::size_of::<ConstIdDataPod>() == 12);
    // ChipInfoPod: 4+4+4+4+4+4+4+8+8+8+8+8+8+4+4 = 84
    assert!(std::mem::size_of::<ChipInfoPod>() == 84);

    // CellPortPod: 4+4 = 8
    assert!(std::mem::size_of::<CellPortPod>() == 8);
    // PackingRulePod: 8+8+4+4+4+4+4+4 = 40
    assert!(std::mem::size_of::<PackingRulePod>() == 40);
    // ChipExtraDataPod: 4+4+8 = 16
    assert!(std::mem::size_of::<ChipExtraDataPod>() == 16);
};

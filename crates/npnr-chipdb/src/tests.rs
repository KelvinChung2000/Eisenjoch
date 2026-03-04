//! Unit tests for the npnr-chipdb crate.
//!
//! Tests use synthetic in-memory binary data to validate RelPtr/RelSlice
//! resolution, POD struct sizes, and ChipDb helper methods.

use std::marker::PhantomData;
use std::mem;

use crate::pod::*;
use crate::relptr::{RelPtr, RelSlice};
use crate::CHIPDB_VERSION;

// =============================================================================
// Struct size tests (compile-time assertions already exist, but runtime tests
// give clearer error messages if something changes)
// =============================================================================

#[test]
fn pod_struct_sizes() {
    assert_eq!(mem::size_of::<RelPtr<u8>>(), 4);
    assert_eq!(mem::size_of::<RelSlice<u8>>(), 8);
    assert_eq!(mem::size_of::<ChipInfoPod>(), 68);
    assert_eq!(mem::size_of::<TileTypePod>(), 32);
    assert_eq!(mem::size_of::<TileInstPod>(), 24);
    assert_eq!(mem::size_of::<BelDataPod>(), 28);
    assert_eq!(mem::size_of::<BelPinPod>(), 12);
    assert_eq!(mem::size_of::<TileWireDataPod>(), 36);
    assert_eq!(mem::size_of::<PipRefPod>(), 8);
    assert_eq!(mem::size_of::<BelPinRefPod>(), 8);
    assert_eq!(mem::size_of::<PipDataPod>(), 24);
    assert_eq!(mem::size_of::<NodeShapePod>(), 8);
    assert_eq!(mem::size_of::<RelNodeRefPod>(), 8);
    assert_eq!(mem::size_of::<PackageInfoPod>(), 12);
    assert_eq!(mem::size_of::<PadInfoPod>(), 16);
    assert_eq!(mem::size_of::<SpeedGradePod>(), 20);
    assert_eq!(mem::size_of::<PipTimingPod>(), 16);
    assert_eq!(mem::size_of::<CellTimingPod>(), 20);
    assert_eq!(mem::size_of::<CellPropDelayPod>(), 16);
    assert_eq!(mem::size_of::<CellSetupHoldPod>(), 24);
}

// =============================================================================
// Synthetic chipdb builder for in-memory testing
// =============================================================================

/// Helper to build a synthetic chipdb binary blob in memory.
///
/// This constructs a valid minimal chipdb with configurable grid size,
/// tile types, bels, wires, and pips. All relative pointers are computed
/// correctly.
struct SyntheticChipDb {
    buf: Vec<u8>,
}

impl SyntheticChipDb {
    /// Append a null-terminated string and return its offset.
    fn append_str(&mut self, s: &str) -> usize {
        let offset = self.buf.len();
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0); // null terminator
        offset
    }

    /// Append a value's raw bytes and return the offset.
    fn append_val<T>(&mut self, val: &T) -> usize {
        let offset = self.buf.len();
        let bytes = unsafe {
            std::slice::from_raw_parts(val as *const T as *const u8, mem::size_of::<T>())
        };
        self.buf.extend_from_slice(bytes);
        offset
    }

    /// Write a value at a specific offset.
    fn write_at<T>(&mut self, offset: usize, val: &T) {
        let bytes = unsafe {
            std::slice::from_raw_parts(val as *const T as *const u8, mem::size_of::<T>())
        };
        self.buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// Compute a relative offset: from `field_pos` to `target_pos`.
    fn rel_offset(field_pos: usize, target_pos: usize) -> i32 {
        (target_pos as isize - field_pos as isize) as i32
    }

    /// Build a minimal chipdb with:
    /// - 2x2 grid (4 tiles)
    /// - 1 tile type with 1 bel, 2 wires, 1 pip
    /// - Each tile is an instance of that tile type
    fn build_minimal() -> Vec<u8> {
        let mut db = SyntheticChipDb { buf: Vec::new() };

        // Reserve space for ChipInfoPod at the start
        let chip_info_offset = 0usize;
        let chip_info_size = mem::size_of::<ChipInfoPod>();
        db.buf.resize(chip_info_size, 0);

        // --- Strings ---
        let chip_name_offset = db.append_str("test_chip");
        let generator_offset = db.append_str("test_gen");
        let tile_type_name_offset = db.append_str("LOGIC");
        let bel_name_offset = db.append_str("LUT0");
        let bel_type_offset = db.append_str("LUT4");
        let bel_bucket_offset = db.append_str("LUT");
        let bel_pin_name_offset = db.append_str("I0");
        let wire0_name_offset = db.append_str("W0");
        let wire0_type_offset = db.append_str("LOCAL");
        let wire1_name_offset = db.append_str("W1");
        let wire1_type_offset = db.append_str("LOCAL");
        let tile_name_offsets = [
            db.append_str("TILE_0_0"),
            db.append_str("TILE_1_0"),
            db.append_str("TILE_0_1"),
            db.append_str("TILE_1_1"),
        ];

        // --- BelPinPod (1 pin) ---
        let bel_pin_offset = db.buf.len();
        let bel_pin_name_field_pos = bel_pin_offset;
        let bel_pin = BelPinPod {
            name: RelPtr {
                offset: Self::rel_offset(bel_pin_name_field_pos, bel_pin_name_offset),
                _phantom: PhantomData,
            },
            wire_index: 0,
            dir: 0, // PortType::In
        };
        db.append_val(&bel_pin);

        // --- BelDataPod (1 bel) ---
        let bel_data_offset = db.buf.len();
        let bel_name_field_pos = bel_data_offset;
        let bel_type_field_pos = bel_data_offset + 4;
        let bel_bucket_field_pos = bel_data_offset + 8;
        let bel_pins_offset_field_pos = bel_data_offset + 12;

        let bel = BelDataPod {
            name: RelPtr {
                offset: Self::rel_offset(bel_name_field_pos, bel_name_offset),
                _phantom: PhantomData,
            },
            bel_type: RelPtr {
                offset: Self::rel_offset(bel_type_field_pos, bel_type_offset),
                _phantom: PhantomData,
            },
            bucket: RelPtr {
                offset: Self::rel_offset(bel_bucket_field_pos, bel_bucket_offset),
                _phantom: PhantomData,
            },
            pins: RelSlice {
                offset: Self::rel_offset(bel_pins_offset_field_pos, bel_pin_offset),
                length: 1,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
            z: 0,
            padding: 0,
        };
        db.append_val(&bel);

        // --- PipDataPod (1 pip: wire 0 -> wire 1, in same tile) ---
        let pip_data_offset = db.buf.len();
        let pip = PipDataPod {
            src_wire: 0,
            dst_wire: 1,
            timing_index: -1,
            pip_type: 0,
            padding: 0,
            src_tile_delta: 0,
            dst_tile_delta: 0,
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };
        db.append_val(&pip);

        // --- PipRefPod for wires (referencing the pip above) ---
        let pip_ref_offset = db.buf.len();
        let pip_ref = PipRefPod {
            tile_delta: 0,
            index: 0,
        };
        db.append_val(&pip_ref);

        // --- BelPinRefPod for wires ---
        let bel_pin_ref_offset = db.buf.len();
        let bel_pin_ref_pin_field = bel_pin_ref_offset + 4; // after bel(i32)
        let bel_pin_ref = BelPinRefPod {
            bel: 0,
            pin: RelPtr {
                offset: Self::rel_offset(bel_pin_ref_pin_field, bel_pin_name_offset),
                _phantom: PhantomData,
            },
        };
        db.append_val(&bel_pin_ref);

        // --- TileWireDataPod (2 wires) ---
        let wire0_offset = db.buf.len();
        let w0_name_field = wire0_offset;
        let w0_type_field = wire0_offset + 4;
        let w0_downhill_field = wire0_offset + 16;
        let w0_belpins_field = wire0_offset + 24;

        let wire0 = TileWireDataPod {
            name: RelPtr {
                offset: Self::rel_offset(w0_name_field, wire0_name_offset),
                _phantom: PhantomData,
            },
            wire_type: RelPtr {
                offset: Self::rel_offset(w0_type_field, wire0_type_offset),
                _phantom: PhantomData,
            },
            pips_uphill: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            pips_downhill: RelSlice {
                offset: Self::rel_offset(w0_downhill_field, pip_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            bel_pins: RelSlice {
                offset: Self::rel_offset(w0_belpins_field, bel_pin_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            flags: 0,
        };
        db.append_val(&wire0);

        let wire1_offset = db.buf.len();
        let w1_name_field = wire1_offset;
        let w1_type_field = wire1_offset + 4;
        let w1_uphill_field = wire1_offset + 8;

        let wire1 = TileWireDataPod {
            name: RelPtr {
                offset: Self::rel_offset(w1_name_field, wire1_name_offset),
                _phantom: PhantomData,
            },
            wire_type: RelPtr {
                offset: Self::rel_offset(w1_type_field, wire1_type_offset),
                _phantom: PhantomData,
            },
            pips_uphill: RelSlice {
                offset: Self::rel_offset(w1_uphill_field, pip_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            pips_downhill: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            bel_pins: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            flags: 0,
        };
        db.append_val(&wire1);

        // --- TileTypePod (1 tile type) ---
        let tile_type_offset = db.buf.len();
        let tt_name_field = tile_type_offset;
        let tt_bels_field = tile_type_offset + 4;
        let tt_wires_field = tile_type_offset + 12;
        let tt_pips_field = tile_type_offset + 20;

        let tile_type = TileTypePod {
            name: RelPtr {
                offset: Self::rel_offset(tt_name_field, tile_type_name_offset),
                _phantom: PhantomData,
            },
            bels: RelSlice {
                offset: Self::rel_offset(tt_bels_field, bel_data_offset),
                length: 1,
                _phantom: PhantomData,
            },
            wires: RelSlice {
                offset: Self::rel_offset(tt_wires_field, wire0_offset),
                length: 2,
                _phantom: PhantomData,
            },
            pips: RelSlice {
                offset: Self::rel_offset(tt_pips_field, pip_data_offset),
                length: 1,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };
        db.append_val(&tile_type);

        // --- TileInstPods (4 tiles in a 2x2 grid) ---
        let tile_insts_offset = db.buf.len();
        let coords: [(i16, i16); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

        for (i, &(x, y)) in coords.iter().enumerate() {
            let inst_offset = db.buf.len();
            let inst_name_field = inst_offset;

            let inst = TileInstPod {
                name: RelPtr {
                    offset: Self::rel_offset(inst_name_field, tile_name_offsets[i]),
                    _phantom: PhantomData,
                },
                tile_type: 0, // all tiles use tile type 0
                tilewire_to_node: RelSlice {
                    offset: 0,
                    length: 0,
                    _phantom: PhantomData,
                },
                extra_data: RelPtr {
                    offset: 0,
                    _phantom: PhantomData,
                },
                x,
                y,
            };
            db.append_val(&inst);
        }

        // --- Fill in ChipInfoPod at offset 0 ---
        let ci_name_field = chip_info_offset + 16;
        let ci_gen_field = chip_info_offset + 20;
        let ci_tile_types_field = chip_info_offset + 24;
        let ci_tile_insts_field = chip_info_offset + 32;

        let chip_info = ChipInfoPod {
            version: CHIPDB_VERSION,
            width: 2,
            height: 2,
            num_tiles: 4,
            name: RelPtr {
                offset: Self::rel_offset(ci_name_field, chip_name_offset),
                _phantom: PhantomData,
            },
            generator: RelPtr {
                offset: Self::rel_offset(ci_gen_field, generator_offset),
                _phantom: PhantomData,
            },
            tile_types: RelSlice {
                offset: Self::rel_offset(ci_tile_types_field, tile_type_offset),
                length: 1,
                _phantom: PhantomData,
            },
            tile_insts: RelSlice {
                offset: Self::rel_offset(ci_tile_insts_field, tile_insts_offset),
                length: 4,
                _phantom: PhantomData,
            },
            nodes: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            packages: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            speed_grades: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };

        db.write_at(chip_info_offset, &chip_info);
        db.buf
    }
}

// =============================================================================
// ChipDb helper tests using synthetic data
// =============================================================================

fn make_chipdb() -> crate::ChipDb {
    let bytes = SyntheticChipDb::build_minimal();
    unsafe { crate::ChipDb::from_bytes(&bytes).expect("failed to load synthetic chipdb") }
}

#[test]
fn load_synthetic_chipdb() {
    let db = make_chipdb();
    assert_eq!(unsafe { read_packed!(*db.chip_info(), version) }, CHIPDB_VERSION);
}

#[test]
fn chip_dimensions() {
    let db = make_chipdb();
    assert_eq!(db.width(), 2);
    assert_eq!(db.height(), 2);
    assert_eq!(db.num_tiles(), 4);
}

#[test]
fn chip_name() {
    let db = make_chipdb();
    assert_eq!(db.name(), "test_chip");
}

#[test]
fn tile_xy_mapping() {
    let db = make_chipdb();
    assert_eq!(db.tile_xy(0), (0, 0));
    assert_eq!(db.tile_xy(1), (1, 0));
    assert_eq!(db.tile_xy(2), (0, 1));
    assert_eq!(db.tile_xy(3), (1, 1));
}

#[test]
fn tile_by_xy_mapping() {
    let db = make_chipdb();
    assert_eq!(db.tile_by_xy(0, 0), 0);
    assert_eq!(db.tile_by_xy(1, 0), 1);
    assert_eq!(db.tile_by_xy(0, 1), 2);
    assert_eq!(db.tile_by_xy(1, 1), 3);
}

#[test]
fn rel_tile_same_position() {
    let db = make_chipdb();
    assert_eq!(db.rel_tile(0, 0, 0), 0);
    assert_eq!(db.rel_tile(3, 0, 0), 3);
}

#[test]
fn rel_tile_with_delta() {
    let db = make_chipdb();
    // From tile 0 (0,0), dx=1 -> tile 1 (1,0)
    assert_eq!(db.rel_tile(0, 1, 0), 1);
    // From tile 0 (0,0), dy=1 -> tile 2 (0,1)
    assert_eq!(db.rel_tile(0, 0, 1), 2);
    // From tile 0 (0,0), dx=1,dy=1 -> tile 3 (1,1)
    assert_eq!(db.rel_tile(0, 1, 1), 3);
}

#[test]
fn bel_iteration() {
    let db = make_chipdb();
    let bels: Vec<_> = db.bels().collect();
    // 4 tiles, each with 1 bel = 4 bels total
    assert_eq!(bels.len(), 4);
    for (i, bel) in bels.iter().enumerate() {
        assert_eq!(bel.tile(), i as i32);
        assert_eq!(bel.index(), 0);
    }
}

#[test]
fn wire_iteration() {
    let db = make_chipdb();
    let wires: Vec<_> = db.wires().collect();
    // 4 tiles, each with 2 wires = 8 wires total
    assert_eq!(wires.len(), 8);
}

#[test]
fn pip_iteration() {
    let db = make_chipdb();
    let pips: Vec<_> = db.pips().collect();
    // 4 tiles, each with 1 pip = 4 pips total
    assert_eq!(pips.len(), 4);
}

#[test]
fn bel_info_access() {
    let db = make_chipdb();
    let bel = npnr_types::BelId::new(0, 0);
    let info = db.bel_info(bel);
    let z: i16 = unsafe { read_packed!(*info, z) };
    assert_eq!(z, 0);
}

#[test]
fn bel_name_access() {
    let db = make_chipdb();
    let bel = npnr_types::BelId::new(0, 0);
    assert_eq!(db.bel_name(bel), "LUT0");
}

#[test]
fn bel_type_access() {
    let db = make_chipdb();
    let bel = npnr_types::BelId::new(0, 0);
    assert_eq!(db.bel_type(bel), "LUT4");
}

#[test]
fn bel_loc_access() {
    let db = make_chipdb();
    // Bel in tile 0 -> (0,0,0)
    let bel0 = npnr_types::BelId::new(0, 0);
    assert_eq!(db.bel_loc(bel0), npnr_types::Loc::new(0, 0, 0));

    // Bel in tile 3 -> (1,1,0)
    let bel3 = npnr_types::BelId::new(3, 0);
    assert_eq!(db.bel_loc(bel3), npnr_types::Loc::new(1, 1, 0));
}

#[test]
fn pip_info_access() {
    let db = make_chipdb();
    let pip = npnr_types::PipId::new(0, 0);
    let info = db.pip_info(pip);
    let src_wire: i32 = unsafe { read_packed!(*info, src_wire) };
    let dst_wire: i32 = unsafe { read_packed!(*info, dst_wire) };
    let timing_index: i32 = unsafe { read_packed!(*info, timing_index) };
    assert_eq!(src_wire, 0);
    assert_eq!(dst_wire, 1);
    assert_eq!(timing_index, -1);
}

#[test]
fn pip_src_dst_wire() {
    let db = make_chipdb();
    let pip = npnr_types::PipId::new(0, 0);
    let src = db.pip_src_wire(pip);
    let dst = db.pip_dst_wire(pip);
    assert_eq!(src.tile(), 0);
    assert_eq!(src.index(), 0);
    assert_eq!(dst.tile(), 0);
    assert_eq!(dst.index(), 1);
}

#[test]
fn wire_info_access() {
    let db = make_chipdb();
    let wire = npnr_types::WireId::new(0, 0);
    let info = db.wire_info(wire);
    let flags: i32 = unsafe { read_packed!(*info, flags) };
    assert_eq!(flags, 0);
    // Wire 0 should have 1 downhill pip
    assert_eq!(info.pips_downhill.len(), 1);
    // Wire 0 should have 0 uphill pips
    assert_eq!(info.pips_uphill.len(), 0);
}

#[test]
fn tile_type_index() {
    let db = make_chipdb();
    for tile in 0..4 {
        assert_eq!(db.tile_type_index(tile), 0);
    }
}

#[test]
fn tile_inst_access() {
    let db = make_chipdb();
    let inst = db.tile_inst(0);
    let x: i16 = unsafe { read_packed!(*inst, x) };
    let y: i16 = unsafe { read_packed!(*inst, y) };
    let tt: i32 = unsafe { read_packed!(*inst, tile_type) };
    assert_eq!(x, 0);
    assert_eq!(y, 0);
    assert_eq!(tt, 0);
}

#[test]
fn version_mismatch_error() {
    let mut bytes = SyntheticChipDb::build_minimal();
    // Corrupt the version
    bytes[0] = 0xFF;
    bytes[1] = 0xFF;
    bytes[2] = 0xFF;
    bytes[3] = 0xFF;
    let result = unsafe { crate::ChipDb::from_bytes(&bytes) };
    assert!(result.is_err());
    match result.unwrap_err() {
        crate::ChipDbError::VersionMismatch { expected, got } => {
            assert_eq!(expected, CHIPDB_VERSION);
            assert_eq!(got, -1);
        }
        e => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn too_small_error() {
    let bytes = [0u8; 4]; // way too small
    let result = unsafe { crate::ChipDb::from_bytes(&bytes) };
    assert!(result.is_err());
    match result.unwrap_err() {
        crate::ChipDbError::TooSmall { size, min } => {
            assert_eq!(size, 4);
            assert_eq!(min, mem::size_of::<ChipInfoPod>());
        }
        e => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn bel_pins_accessible() {
    let db = make_chipdb();
    let bel = npnr_types::BelId::new(0, 0);
    let info = db.bel_info(bel);
    let pins = info.pins.get();
    assert_eq!(pins.len(), 1);
    let wire_index: i32 = unsafe { read_packed!(pins[0], wire_index) };
    let dir: i32 = unsafe { read_packed!(pins[0], dir) };
    assert_eq!(wire_index, 0);
    assert_eq!(dir, 0);
}

#[test]
fn wire_bel_pins_accessible() {
    let db = make_chipdb();
    let wire = npnr_types::WireId::new(0, 0);
    let info = db.wire_info(wire);
    let bel_pins = info.bel_pins.get();
    assert_eq!(bel_pins.len(), 1);
    let bel: i32 = unsafe { read_packed!(bel_pins[0], bel) };
    assert_eq!(bel, 0);
}

#[test]
fn wire_pip_refs_accessible() {
    let db = make_chipdb();
    // Wire 0 has 1 downhill pip
    let wire0 = npnr_types::WireId::new(0, 0);
    let info0 = db.wire_info(wire0);
    let downhill = info0.pips_downhill.get();
    assert_eq!(downhill.len(), 1);
    let tile_delta: i32 = unsafe { read_packed!(downhill[0], tile_delta) };
    let index: i32 = unsafe { read_packed!(downhill[0], index) };
    assert_eq!(tile_delta, 0);
    assert_eq!(index, 0);

    // Wire 1 has 1 uphill pip
    let wire1 = npnr_types::WireId::new(0, 1);
    let info1 = db.wire_info(wire1);
    let uphill = info1.pips_uphill.get();
    assert_eq!(uphill.len(), 1);
    let tile_delta: i32 = unsafe { read_packed!(uphill[0], tile_delta) };
    let index: i32 = unsafe { read_packed!(uphill[0], index) };
    assert_eq!(tile_delta, 0);
    assert_eq!(index, 0);
}

#[test]
fn all_tile_bels_have_same_info() {
    let db = make_chipdb();
    // Since all tiles use the same tile type, all bels should have the same name
    for tile in 0..4 {
        let bel = npnr_types::BelId::new(tile, 0);
        assert_eq!(db.bel_name(bel), "LUT0");
        assert_eq!(db.bel_type(bel), "LUT4");
    }
}

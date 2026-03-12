//! Integration tests for the chipdb module.
//!
//! Tests use synthetic in-memory binary data to validate RelPtr/RelSlice
//! resolution, POD struct sizes, and ChipDb helper methods.

use std::mem;

use nextpnr::chipdb::testutil::{make_test_chipdb, SyntheticChipDbBuilder};
use nextpnr::chipdb::*;

// =============================================================================
// Struct size tests
// =============================================================================

#[test]
fn pod_struct_sizes() {
    assert_eq!(mem::size_of::<RelPtr<u8>>(), 4);
    assert_eq!(mem::size_of::<RelSlice<u8>>(), 8);
    assert_eq!(mem::size_of::<BelPinPod>(), 12);
    assert_eq!(mem::size_of::<BelDataPod>(), 36);
    assert_eq!(mem::size_of::<BelPinRefPod>(), 8);
    assert_eq!(mem::size_of::<TileWireDataPod>(), 48);
    assert_eq!(mem::size_of::<PipDataPod>(), 24);
    assert_eq!(mem::size_of::<RelTileWireRefPod>(), 6);
    assert_eq!(mem::size_of::<NodeShapePod>(), 12);
    assert_eq!(mem::size_of::<GroupDataPod>(), 44);
    assert_eq!(mem::size_of::<TileTypePod>(), 40);
    assert_eq!(mem::size_of::<RelNodeRefPod>(), 6);
    assert_eq!(mem::size_of::<TileRoutingShapePod>(), 12);
    assert_eq!(mem::size_of::<TileInstPod>(), 16);
    assert_eq!(mem::size_of::<PadInfoPod>(), 28);
    assert_eq!(mem::size_of::<PackageInfoPod>(), 16);
    assert_eq!(mem::size_of::<TimingValue>(), 16);
    assert_eq!(mem::size_of::<PipTimingPod>(), 52);
    assert_eq!(mem::size_of::<NodeTimingPod>(), 48);
    assert_eq!(mem::size_of::<CellPinCombArcPod>(), 20);
    assert_eq!(mem::size_of::<CellPinRegArcPod>(), 56);
    assert_eq!(mem::size_of::<CellPinTimingPod>(), 24);
    assert_eq!(mem::size_of::<CellTimingPod>(), 12);
    assert_eq!(mem::size_of::<SpeedGradePod>(), 28);
    assert_eq!(mem::size_of::<ConstIdDataPod>(), 12);
    assert_eq!(mem::size_of::<ChipInfoPod>(), 84);
}

// =============================================================================
// ChipDb helper tests using synthetic data
// =============================================================================

#[test]
fn load_synthetic_chipdb() {
    let db = make_test_chipdb();
    assert_eq!(db.chip_magic(), CHIPDB_MAGIC);
    assert_eq!(db.chip_db_version(), CHIPDB_VERSION);
}

#[test]
fn chip_dimensions() {
    let db = make_test_chipdb();
    assert_eq!(db.width(), 2);
    assert_eq!(db.height(), 2);
    assert_eq!(db.num_tiles(), 4);
}

#[test]
fn chip_name() {
    let db = make_test_chipdb();
    assert_eq!(db.name(), "test_chip");
}

#[test]
fn chip_uarch() {
    let db = make_test_chipdb();
    assert_eq!(db.uarch(), "test_uarch");
}

#[test]
fn tile_xy_mapping() {
    let db = make_test_chipdb();
    assert_eq!(db.tile_xy(0), (0, 0));
    assert_eq!(db.tile_xy(1), (1, 0));
    assert_eq!(db.tile_xy(2), (0, 1));
    assert_eq!(db.tile_xy(3), (1, 1));
}

#[test]
fn tile_by_xy_mapping() {
    let db = make_test_chipdb();
    assert_eq!(db.tile_by_xy(0, 0), 0);
    assert_eq!(db.tile_by_xy(1, 0), 1);
    assert_eq!(db.tile_by_xy(0, 1), 2);
    assert_eq!(db.tile_by_xy(1, 1), 3);
}

#[test]
fn rel_tile_same_position() {
    let db = make_test_chipdb();
    assert_eq!(db.rel_tile(0, 0, 0), 0);
    assert_eq!(db.rel_tile(3, 0, 0), 3);
}

#[test]
fn rel_tile_with_delta() {
    let db = make_test_chipdb();
    assert_eq!(db.rel_tile(0, 1, 0), 1);
    assert_eq!(db.rel_tile(0, 0, 1), 2);
    assert_eq!(db.rel_tile(0, 1, 1), 3);
}

#[test]
fn bel_iteration() {
    let db = make_test_chipdb();
    let bels: Vec<_> = db.bels().collect();
    assert_eq!(bels.len(), 4);
    for (i, bel) in bels.iter().enumerate() {
        assert_eq!(bel.tile(), i as i32);
        assert_eq!(bel.index(), 0);
    }
}

#[test]
fn wire_iteration() {
    let db = make_test_chipdb();
    let wires: Vec<_> = db.wires().collect();
    assert_eq!(wires.len(), 8);
}

#[test]
fn pip_iteration() {
    let db = make_test_chipdb();
    let pips: Vec<_> = db.pips().collect();
    assert_eq!(pips.len(), 4);
}

#[test]
fn bel_info_access() {
    let db = make_test_chipdb();
    let bel = BelId::new(0, 0);
    let loc = db.bel_loc(bel);
    assert_eq!(loc.z, 0);
}

#[test]
fn bel_name_access() {
    let db = make_test_chipdb();
    let bel = BelId::new(0, 0);
    assert_eq!(db.bel_name(bel), "LUT0");
}

#[test]
fn bel_type_access() {
    let db = make_test_chipdb();
    let bel = BelId::new(0, 0);
    assert_eq!(db.bel_type(bel), "LUT4");
}

#[test]
fn bel_loc_access() {
    let db = make_test_chipdb();
    let bel0 = BelId::new(0, 0);
    assert_eq!(db.bel_loc(bel0), Loc::new(0, 0, 0));

    let bel3 = BelId::new(3, 0);
    assert_eq!(db.bel_loc(bel3), Loc::new(1, 1, 0));
}

#[test]
fn pip_info_access() {
    let db = make_test_chipdb();
    let pip = PipId::new(0, 0);
    let src = db.pip_src_wire(pip);
    let dst = db.pip_dst_wire(pip);
    assert_eq!(src.index(), 0);
    assert_eq!(dst.index(), 1);
    assert_eq!(db.pip_timing_index(pip), 0);
}

#[test]
fn pip_src_dst_wire() {
    let db = make_test_chipdb();
    let pip = PipId::new(0, 0);
    let src = db.pip_src_wire(pip);
    let dst = db.pip_dst_wire(pip);
    assert_eq!(src.tile(), 0);
    assert_eq!(src.index(), 0);
    assert_eq!(dst.tile(), 0);
    assert_eq!(dst.index(), 1);
}

#[test]
fn wire_info_access() {
    let db = make_test_chipdb();
    let wire = WireId::new(0, 0);
    assert_eq!(db.wire_flags(wire), 0);
    let info = db.wire_info(wire);
    assert_eq!(info.pips_downhill.len(), 1);
    assert_eq!(info.pips_uphill.len(), 0);
}

#[test]
fn tile_type_index() {
    let db = make_test_chipdb();
    for tile in 0..4 {
        assert_eq!(db.tile_type_index(tile), 0);
    }
}

#[test]
fn tile_inst_access() {
    let db = make_test_chipdb();
    assert_eq!(db.tile_type_index(0), 0);
    assert_eq!(db.tile_shape_index(0), 0);
}

#[test]
fn magic_mismatch_error() {
    let mut bytes = SyntheticChipDbBuilder::build_minimal();
    let chip_info_offset = 4usize;
    bytes[chip_info_offset] = 0xFF;
    bytes[chip_info_offset + 1] = 0xFF;
    bytes[chip_info_offset + 2] = 0xFF;
    bytes[chip_info_offset + 3] = 0xFF;
    let result = unsafe { ChipDb::from_bytes(&bytes) };
    assert!(result.is_err());
    match result.unwrap_err() {
        ChipDbError::MagicMismatch { .. } => {}
        e => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn version_mismatch_error() {
    let mut bytes = SyntheticChipDbBuilder::build_minimal();
    let version_offset = 4 + 4;
    bytes[version_offset] = 0xFF;
    bytes[version_offset + 1] = 0xFF;
    bytes[version_offset + 2] = 0xFF;
    bytes[version_offset + 3] = 0xFF;
    let result = unsafe { ChipDb::from_bytes(&bytes) };
    assert!(result.is_err());
    match result.unwrap_err() {
        ChipDbError::VersionMismatch { expected, got } => {
            assert_eq!(expected, CHIPDB_VERSION);
            assert_eq!(got, -1);
        }
        e => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn too_small_error() {
    let bytes = [0u8; 2];
    let result = unsafe { ChipDb::from_bytes(&bytes) };
    assert!(result.is_err());
    match result.unwrap_err() {
        ChipDbError::TooSmall { size, min } => {
            assert_eq!(size, 2);
            assert_eq!(min, mem::size_of::<RelPtr<ChipInfoPod>>());
        }
        e => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn bel_pins_accessible() {
    let db = make_test_chipdb();
    let bel = BelId::new(0, 0);
    let info = db.bel_info(bel);
    let pins = info.pins.get();
    assert_eq!(pins.len(), 2);
    let (_name, wire, dir) = db.bel_pin_fields(&pins[0]);
    assert_eq!(wire, 0); // I0 on wire W0
    assert_eq!(dir, 0);  // input
    let (_name, wire, dir) = db.bel_pin_fields(&pins[1]);
    assert_eq!(wire, 1); // O on wire W1
    assert_eq!(dir, 1);  // output
}

#[test]
fn wire_bel_pins_accessible() {
    let db = make_test_chipdb();
    let wire = WireId::new(0, 0);
    let info = db.wire_info(wire);
    let bel_pins = info.bel_pins.get();
    assert_eq!(bel_pins.len(), 1);
    assert_eq!(db.bel_pin_ref_bel(&bel_pins[0]), 0);
}

#[test]
fn wire_pip_indices_accessible() {
    let db = make_test_chipdb();
    let wire0 = WireId::new(0, 0);
    let info0 = db.wire_info(wire0);
    let downhill = info0.pips_downhill.get();
    assert_eq!(downhill.len(), 1);
    assert_eq!(downhill[0], 0);

    let wire1 = WireId::new(0, 1);
    let info1 = db.wire_info(wire1);
    let uphill = info1.pips_uphill.get();
    assert_eq!(uphill.len(), 1);
    assert_eq!(uphill[0], 0);
}

#[test]
fn all_tile_bels_have_same_info() {
    let db = make_test_chipdb();
    for tile in 0..4 {
        let bel = BelId::new(tile, 0);
        assert_eq!(db.bel_name(bel), "LUT0");
        assert_eq!(db.bel_type(bel), "LUT4");
    }
}

#[test]
fn constid_lookup() {
    let db = make_test_chipdb();
    assert_eq!(db.constid_str(0), Some("LOGIC"));
    assert_eq!(db.constid_str(1), Some("LUT0"));
    assert_eq!(db.constid_str(2), Some("LUT4"));
    assert_eq!(db.constid_str(3), Some("I0"));
    assert_eq!(db.constid_str(4), Some("W0"));
    assert_eq!(db.constid_str(5), Some("LOCAL"));
    assert_eq!(db.constid_str(6), Some("W1"));
    assert_eq!(db.constid_str(-1), None);
    assert_eq!(db.constid_str(100), None);
}

#[test]
fn tile_shape_access() {
    let db = make_test_chipdb();
    let shape = db.tile_shape(0);
    assert_eq!(shape.wire_to_node.len(), 0);
    assert_eq!(db.tile_shape_timing_index(0), 0);
}

// =============================================================================
// Timing data tests
// =============================================================================

#[test]
fn speed_grade_accessible() {
    let db = make_test_chipdb();
    assert_eq!(db.num_speed_grades(), 1);
    let sg = db.speed_grade(0).unwrap();
    assert_eq!(sg.pip_classes.len(), 1);
    assert_eq!(sg.node_classes.len(), 1);
    assert_eq!(sg.cell_types.len(), 1);
}

#[test]
fn pip_timing_class_accessible() {
    use nextpnr::read_packed;
    let db = make_test_chipdb();
    let sg = db.speed_grade(0).unwrap();
    let pip = PipId::new(0, 0);
    let tmg = db.pip_timing(sg, pip).unwrap();
    // Uniform pip delay: 100ps fast, 150ps slow
    let fast_min: i32 = unsafe { read_packed!(tmg.int_delay, fast_min) };
    let slow_min: i32 = unsafe { read_packed!(tmg.int_delay, slow_min) };
    assert_eq!(fast_min, 100);
    assert_eq!(slow_min, 150);
}

#[test]
fn node_timing_class_accessible() {
    use nextpnr::read_packed;
    let db = make_test_chipdb();
    let sg = db.speed_grade(0).unwrap();
    let wire = WireId::new(0, 0);
    let tmg = db.node_timing(sg, wire).unwrap();
    // Uniform node delay: 50ps fast, 75ps slow
    let fast_min: i32 = unsafe { read_packed!(tmg.delay, fast_min) };
    let slow_min: i32 = unsafe { read_packed!(tmg.delay, slow_min) };
    assert_eq!(fast_min, 50);
    assert_eq!(slow_min, 75);
}

#[test]
fn cell_timing_has_lut4_comb_arc() {
    use nextpnr::read_packed;
    let db = make_test_chipdb();
    let sg = db.speed_grade(0).unwrap();
    // LUT4 type variant index
    let ct_idx = db.cell_timing_index(sg, 2).unwrap(); // ID_LUT4 = 2
    let ct = &sg.cell_types.get()[ct_idx];
    assert_eq!(ct.pins.len(), 1);
    let pin_tmg = &ct.pins.get()[0];
    assert_eq!(pin_tmg.comb_arcs.len(), 1);
    let arc = &pin_tmg.comb_arcs.get()[0];
    // I0 -> O delay: 200ps fast, 300ps slow
    let fast_min: i32 = unsafe { read_packed!(arc.delay, fast_min) };
    let slow_min: i32 = unsafe { read_packed!(arc.delay, slow_min) };
    assert_eq!(fast_min, 200);
    assert_eq!(slow_min, 300);
}

// =============================================================================
// RelPtr / RelSlice unit tests (extracted from relptr.rs inline tests)
// =============================================================================

use std::marker::PhantomData;

#[test]
fn relptr_size() {
    // RelPtr should be exactly 4 bytes (i32 offset + zero-size PhantomData)
    assert_eq!(mem::size_of::<RelPtr<u8>>(), 4);
    assert_eq!(mem::size_of::<RelPtr<u32>>(), 4);
}

#[test]
fn relslice_size() {
    // RelSlice should be exactly 8 bytes (i32 offset + u32 length + zero-size PhantomData)
    assert_eq!(mem::size_of::<RelSlice<u8>>(), 8);
    assert_eq!(mem::size_of::<RelSlice<u32>>(), 8);
}

#[test]
fn relptr_resolve() {
    // Create a buffer: [offset: i32, target_data: u32]
    // offset = 4 (size of i32), pointing right after itself
    #[repr(C, packed)]
    struct TestData {
        ptr: RelPtr<u32>,
        value: u32,
    }
    let data = TestData {
        ptr: RelPtr {
            offset: 4, // points to the next field
            _phantom: PhantomData,
        },
        value: 0xDEADBEEF,
    };
    let resolved = data.ptr.get();
    let val = unsafe { std::ptr::read_unaligned(resolved) };
    assert_eq!(val, 0xDEADBEEF);
}

#[test]
fn relptr_self_reference() {
    // Test that RelPtr with offset 0 points to itself
    let ptr: RelPtr<i32> = RelPtr {
        offset: 0,
        _phantom: PhantomData,
    };
    let resolved = ptr.get();
    // The resolved pointer should point to the offset field itself
    assert_eq!(resolved as usize, std::ptr::addr_of!(ptr.offset) as usize);
}

#[test]
fn relptr_negative_offset() {
    // Create a buffer where the target is before the pointer
    #[repr(C, packed)]
    struct TestData {
        value: u32,
        ptr: RelPtr<u32>,
    }
    let data = TestData {
        value: 42,
        ptr: RelPtr {
            offset: -4, // points back to the previous field
            _phantom: PhantomData,
        },
    };
    let resolved = data.ptr.get();
    let val = unsafe { std::ptr::read_unaligned(resolved) };
    assert_eq!(val, 42);
}

#[test]
fn relslice_resolve() {
    // Layout: [offset: i32, length: u32, data: [u32; 3]]
    #[repr(C, packed)]
    struct TestData {
        slice: RelSlice<u32>,
        values: [u32; 3],
    }
    let data = TestData {
        slice: RelSlice {
            offset: 8, // skip past offset(4) + length(4)
            length: 3,
            _phantom: PhantomData,
        },
        values: [10, 20, 30],
    };
    let resolved = data.slice.get();
    assert_eq!(resolved.len(), 3);
    assert_eq!(resolved[0], 10);
    assert_eq!(resolved[1], 20);
    assert_eq!(resolved[2], 30);
}

#[test]
fn relslice_empty() {
    let slice: RelSlice<u32> = RelSlice {
        offset: 0,
        length: 0,
        _phantom: PhantomData,
    };
    assert!(slice.is_empty());
    assert_eq!(slice.len(), 0);
    assert_eq!(slice.get().len(), 0);
}

#[test]
fn relptr_is_null() {
    let null_ptr: RelPtr<u8> = RelPtr {
        offset: 0,
        _phantom: PhantomData,
    };
    assert!(null_ptr.is_null());

    let non_null_ptr: RelPtr<u8> = RelPtr {
        offset: 42,
        _phantom: PhantomData,
    };
    assert!(!non_null_ptr.is_null());
}

#[test]
fn relptr_debug() {
    let ptr: RelPtr<u8> = RelPtr {
        offset: 123,
        _phantom: PhantomData,
    };
    let debug = format!("{:?}", ptr);
    assert_eq!(debug, "RelPtr(offset=123)");
}

#[test]
fn relslice_debug() {
    let slice: RelSlice<u8> = RelSlice {
        offset: 10,
        length: 5,
        _phantom: PhantomData,
    };
    let debug = format!("{:?}", slice);
    assert_eq!(debug, "RelSlice(offset=10, length=5)");
}

//! A corrupt chip database is rejected with the field, byte offset and fault at fault.

use std::mem::offset_of;

use nextpnr_db::chipdb::testutil::SyntheticChipDbBuilder;
use nextpnr_db::chipdb::*;
use rstest::rstest;

#[derive(Debug, PartialEq)]
enum Rejection {
    Malformed {
        field: String,
        offset: usize,
        fault: Fault,
    },
    Magic {
        got: u32,
    },
    Version {
        got: i32,
    },
}

fn malformed(field: &str, offset: usize, fault: Fault) -> Rejection {
    Rejection::Malformed {
        field: field.to_owned(),
        offset,
        fault,
    }
}

fn rejection(err: ChipDbError) -> Rejection {
    match err {
        ChipDbError::Malformed {
            field,
            offset,
            fault,
        } => malformed(&field.to_string(), offset, fault),
        ChipDbError::MagicMismatch { got, .. } => Rejection::Magic { got },
        ChipDbError::VersionMismatch { got, .. } => Rejection::Version { got },
        other => panic!("unexpected error: {other}"),
    }
}

fn read_i32(bytes: &[u8], pos: usize) -> i32 {
    i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
}

fn write_i32(bytes: &mut [u8], pos: usize, value: i32) {
    bytes[pos..pos + 4].copy_from_slice(&value.to_le_bytes());
}

fn follow(bytes: &[u8], field: usize) -> usize {
    (field as i64 + i64::from(read_i32(bytes, field))) as usize
}

fn locate(bytes: &[u8], fields: &[usize]) -> usize {
    fields
        .iter()
        .fold(follow(bytes, 0), |pos, &field| follow(bytes, pos + field))
}

fn point_at_end(bytes: &mut Vec<u8>, field: usize, tail: &[u8]) -> usize {
    let target = bytes.len();
    bytes.extend_from_slice(tail);
    write_i32(bytes, field, (target - field) as i32);
    target
}

fn with_wire_to_node(bytes: &mut Vec<u8>, refs: &[RelNodeRefPod]) -> usize {
    let shape = locate(bytes, &[offset_of!(ChipInfoPod, tile_shapes)]);
    let field = shape + offset_of!(TileRoutingShapePod, wire_to_node);
    let encoded: Vec<u8> = refs
        .iter()
        .flat_map(|r| {
            let (dx_mode, dy, wire) = (r.dx_mode, r.dy, r.wire);
            [dx_mode.to_le_bytes(), dy.to_le_bytes(), wire.to_le_bytes()].concat()
        })
        .collect();
    let first = point_at_end(bytes, field, &encoded);
    write_i32(bytes, field + 4, refs.len() as i32);
    first
}

const CHIP: usize = 4;

#[rstest]
#[case::root_past_end(|b: &mut Vec<u8>| {
    let len = b.len();
    write_i32(b, 0, len as i32);
    let start = len as i64;
    malformed("root", 0, Fault::OutOfBounds { start, end: start + 84, len })
})]
#[case::slice_length_overflow(|b: &mut Vec<u8>| {
    let field = CHIP + offset_of!(ChipInfoPod, tile_types);
    write_i32(b, field + 4, -1);
    let start = follow(b, field) as i64;
    let end = start + i64::from(u32::MAX) * 40;
    malformed("chip_info.tile_types", field, Fault::OutOfBounds { start, end, len: b.len() })
})]
#[case::offset_before_buffer(|b: &mut Vec<u8>| {
    let field = CHIP + offset_of!(ChipInfoPod, tile_insts);
    write_i32(b, field, -1000);
    let start = field as i64 - 1000;
    malformed("chip_info.tile_insts", field, Fault::OutOfBounds { start, end: start + 64, len: b.len() })
})]
#[case::pip_src_wire_out_of_range(|b: &mut Vec<u8>| {
    let pip = locate(b, &[offset_of!(ChipInfoPod, tile_types), offset_of!(TileTypePod, pips)]);
    let field = pip + offset_of!(PipDataPod, src_wire);
    write_i32(b, field, 2);
    malformed("chip_info.tile_types[0].pips[0].src_wire", field, Fault::IndexOutOfRange { index: 2, len: 2 })
})]
#[case::unterminated_string(|b: &mut Vec<u8>| {
    let field = CHIP + offset_of!(ChipInfoPod, name);
    point_at_end(b, field, b"chip");
    malformed("chip_info.name", field, Fault::Unterminated)
})]
#[case::misaligned_i32_slice(|b: &mut Vec<u8>| {
    let wire = locate(b, &[offset_of!(ChipInfoPod, tile_types), offset_of!(TileTypePod, wires)]);
    let field = wire + offset_of!(TileWireDataPod, pips_downhill);
    let shifted = read_i32(b, field) + 1;
    write_i32(b, field, shifted);
    malformed("chip_info.tile_types[0].wires[0].pips_downhill", field, Fault::Misaligned { align: 4 })
})]
#[case::bad_magic(|b: &mut Vec<u8>| {
    write_i32(b, CHIP + offset_of!(ChipInfoPod, magic), 0x0bad_f00d);
    Rejection::Magic { got: 0x0bad_f00d }
})]
#[case::bad_version(|b: &mut Vec<u8>| {
    write_i32(b, CHIP + offset_of!(ChipInfoPod, version), CHIPDB_VERSION - 1);
    Rejection::Version { got: CHIPDB_VERSION - 1 }
})]
#[case::null_uarch(|b: &mut Vec<u8>| {
    let field = CHIP + offset_of!(ChipInfoPod, uarch);
    write_i32(b, field, 0);
    malformed("chip_info.uarch", field, Fault::Null)
})]
#[case::tile_name_names_null_constid(|b: &mut Vec<u8>| {
    let ids = locate(b, &[offset_of!(ChipInfoPod, extra_constids), offset_of!(ConstIdDataPod, bba_ids)]);
    write_i32(b, ids + 7 * 4, 0);
    let inst = locate(b, &[offset_of!(ChipInfoPod, tile_insts)]);
    let field = inst + offset_of!(TileInstPod, name_prefix);
    malformed("chip_info.tile_insts[0].name_prefix", field, Fault::MissingConstId { index: 7 })
})]
#[case::grid_disagrees_with_tiles(|b: &mut Vec<u8>| {
    write_i32(b, CHIP + offset_of!(ChipInfoPod, width), 3);
    let field = CHIP + offset_of!(ChipInfoPod, tile_insts);
    malformed("chip_info.tile_insts", field, Fault::GridMismatch { width: 3, height: 2, tiles: 4 })
})]
#[case::grid_area_overflow(|b: &mut Vec<u8>| {
    write_i32(b, CHIP + offset_of!(ChipInfoPod, width), 1 << 16);
    write_i32(b, CHIP + offset_of!(ChipInfoPod, height), 1 << 16);
    malformed("chip_info.width", CHIP + offset_of!(ChipInfoPod, width), Fault::Overflow)
})]
#[case::node_member_leaves_grid(|b: &mut Vec<u8>| {
    let entry = with_wire_to_node(b, &[RelNodeRefPod { dx_mode: -1, dy: 0, wire: 0 }]);
    let field = entry + offset_of!(RelNodeRefPod, dx_mode);
    malformed("chip_info.tile_shapes[0].wire_to_node[0].dx_mode", field, Fault::IndexOutOfRange { index: -1, len: 2 })
})]
#[case::node_member_root_is_a_member(|b: &mut Vec<u8>| {
    let entry = with_wire_to_node(b, &[RelNodeRefPod { dx_mode: 1, dy: 0, wire: 0 }]);
    malformed("chip_info.tile_shapes[0].wire_to_node[0]", entry, Fault::NotNodeRoot)
})]
fn corrupt_chipdb_is_rejected(#[case] corrupt: fn(&mut Vec<u8>) -> Rejection) {
    let mut bytes = SyntheticChipDbBuilder::build_minimal();
    let expected = corrupt(&mut bytes);
    let err = ChipDb::from_bytes(&bytes).expect_err("corrupt chipdb must not load");
    assert_eq!(rejection(err), expected);
}

#[cfg(feature = "mmap")]
#[rstest]
#[case::example("example.bin")]
#[case::fabulous("fabulous.bin")]
fn fixture_loads_alike_from_bytes_and_mmap(#[case] name: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../nextpnr/tests/fixtures")
        .join(name);
    let counts = |db: &ChipDb| {
        (
            db.num_tiles(),
            db.bels().count(),
            db.wires().count(),
            db.pips().count(),
        )
    };
    let owned = ChipDb::from_bytes(&std::fs::read(&path).unwrap()).expect("loads from bytes");
    let mapped = ChipDb::load(&path).expect("loads through mmap");
    assert_eq!(counts(&owned), counts(&mapped));
}

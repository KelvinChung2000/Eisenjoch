use nextpnr_db::chipdb::testutil::make_test_chipdb;
use nextpnr_db::chipdb::{BelId, NameError, NameKind, PipId, WireId};
use rstest::rstest;

#[rstest]
#[case::bel_in_first_tile("TILE_0_0X0Y0/LUT0", BelId::new(0, 0))]
#[case::bel_in_last_tile("TILE_1_1X1Y1/LUT0", BelId::new(3, 0))]
fn bel_names_resolve(#[case] name: &str, #[case] bel: BelId) {
    assert_eq!(make_test_chipdb().bel_by_name(name), Ok(bel));
}

#[rstest]
#[case::first_wire("TILE_1_0X1Y0/W0", WireId::new(1, 0))]
#[case::second_wire("TILE_0_1X0Y1/W1", WireId::new(2, 1))]
fn wire_names_resolve(#[case] name: &str, #[case] wire: WireId) {
    assert_eq!(make_test_chipdb().wire_by_name(name), Ok(wire));
}

#[test]
fn pip_names_put_the_driven_wire_first() {
    assert_eq!(
        make_test_chipdb().pip_by_name("TILE_1_1X1Y1/W1/W0"),
        Ok(PipId::new(3, 0))
    );
}

fn unknown(kind: NameKind, name: &str, segment_kind: NameKind, segment: &str) -> NameError {
    NameError::Unknown {
        kind,
        name: name.to_owned(),
        segment_kind,
        segment: segment.to_owned(),
    }
}

fn segment_count(kind: NameKind, name: &str, found: usize, expected: usize) -> NameError {
    NameError::SegmentCount {
        kind,
        name: name.to_owned(),
        found,
        expected,
    }
}

#[rstest]
#[case::unknown_tile(
    NameKind::Bel,
    "TILE_9_9X9Y9/LUT0",
    unknown(NameKind::Bel, "TILE_9_9X9Y9/LUT0", NameKind::Tile, "TILE_9_9X9Y9")
)]
#[case::unknown_bel(
    NameKind::Bel,
    "TILE_0_0X0Y0/FF0",
    unknown(NameKind::Bel, "TILE_0_0X0Y0/FF0", NameKind::Bel, "FF0")
)]
#[case::bel_without_tile(NameKind::Bel, "LUT0", segment_count(NameKind::Bel, "LUT0", 1, 2))]
#[case::unknown_wire(
    NameKind::Wire,
    "TILE_0_0X0Y0/W7",
    unknown(NameKind::Wire, "TILE_0_0X0Y0/W7", NameKind::Wire, "W7")
)]
#[case::wire_with_pip_segments(
    NameKind::Wire,
    "TILE_0_0X0Y0/W1/W0",
    segment_count(NameKind::Wire, "TILE_0_0X0Y0/W1/W0", 3, 2)
)]
#[case::unknown_pip_wire(
    NameKind::Pip,
    "TILE_0_0X0Y0/W1/W7",
    unknown(NameKind::Pip, "TILE_0_0X0Y0/W1/W7", NameKind::Wire, "W7")
)]
#[case::no_pip_between_wires(
    NameKind::Pip,
    "TILE_0_0X0Y0/W0/W1",
    unknown(NameKind::Pip, "TILE_0_0X0Y0/W0/W1", NameKind::Pip, "W0/W1")
)]
#[case::inverting_pip(
    NameKind::Pip,
    "TILE_0_0X0Y0/W1/W0/INV",
    NameError::InvertingPip { name: "TILE_0_0X0Y0/W1/W0/INV".to_owned() }
)]
fn bad_names_name_the_failing_segment(
    #[case] kind: NameKind,
    #[case] name: &str,
    #[case] error: NameError,
) {
    let chipdb = make_test_chipdb();
    let found = match kind {
        NameKind::Bel => chipdb.bel_by_name(name).map(|_| ()),
        NameKind::Wire => chipdb.wire_by_name(name).map(|_| ()),
        NameKind::Pip => chipdb.pip_by_name(name).map(|_| ()),
        NameKind::Tile => unreachable!("tiles are named only inside BEL, wire and pip names"),
    };
    assert_eq!(found, Err(error));
}

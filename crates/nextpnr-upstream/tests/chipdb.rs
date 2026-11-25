use std::fs;
use std::path::PathBuf;

use nextpnr_db::chipdb::{parse_constids_inc, ChipDb, CHIPDB_VERSION};
use nextpnr_upstream::{ChipdbSummary, Nextpnr};
use rstest::rstest;

fn rust_summary(db: &ChipDb) -> ChipdbSummary {
    let count = |n: usize| u64::try_from(n).unwrap();
    ChipdbSummary {
        version: db.chip_db_version(),
        width: db.width(),
        height: db.height(),
        bels: count(db.bels().count()),
        wires: count(db.wires().count()),
        pips: count(db.pips().count()),
        wire_checksum: db.wires().fold(0u64, |sum, wire| {
            let id = (u64::from(wire.tile() as u32) << 32) | u64::from(wire.index() as u32);
            sum.wrapping_add(id)
        }),
    }
}

#[rstest]
#[case::npnr_baseline("npnr_baseline/shared_chipdb.bin", "npnr_baseline/constids.inc")]
#[ignore = "requires NEXTPNR_BUILD with the bridge built from third_party/nextpnr"]
fn rust_loader_reads_what_upstream_reads(#[case] fixture: &str, #[case] constids: &str) {
    let build = PathBuf::from(std::env::var_os("NEXTPNR_BUILD").expect("set NEXTPNR_BUILD"));
    let bridge = unsafe { Nextpnr::load(&build.join("libeisenjoch-nextpnr.so")) }.unwrap();
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../nextpnr/tests/fixtures");
    let chipdb = fixtures.join(fixture);
    let known = parse_constids_inc(&fs::read_to_string(fixtures.join(constids)).unwrap());

    let upstream = unsafe { bridge.chipdb_summary("EXAMPLE", &chipdb) }.unwrap();
    let bytes = fs::read(&chipdb).unwrap();
    let rust = rust_summary(&ChipDb::from_bytes_with_known_constids(&bytes, &known).unwrap());

    assert_eq!(upstream.version, CHIPDB_VERSION);
    assert_eq!(rust, upstream);
}

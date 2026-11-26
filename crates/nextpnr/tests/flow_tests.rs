use std::path::{Path, PathBuf};

use nextpnr::flow::{run, FlowError, FlowOptions};
use nextpnr::live::{Cancelled, Live};
use rstest::rstest;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn blinky() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../demo/blinky.json")
}

fn flow(
    options: &FlowOptions,
    live: Option<std::sync::Arc<Live>>,
) -> (Result<(), FlowError>, Vec<String>) {
    let mut lines = Vec::new();
    let result = run(options, live, &mut |line| lines.push(line.to_owned())).map(|_| ());
    (result, lines)
}

#[test]
fn a_missing_chipdb_is_named() {
    let options = FlowOptions::new("/nonexistent/chipdb.bin", blinky());
    let (result, _) = flow(&options, None);
    assert!(
        matches!(&result, Err(FlowError::Chipdb { path, .. }) if path == Path::new("/nonexistent/chipdb.bin")),
        "{result:?}"
    );
}

#[rstest]
#[case::missing("/nonexistent/design.json")]
#[case::not_a_netlist(env!("CARGO_MANIFEST_DIR"))]
fn an_unreadable_netlist_is_named(#[case] netlist: &str) {
    let options = FlowOptions::new(fixture("example.bin"), netlist);
    let (result, _) = flow(&options, None);
    assert!(
        matches!(&result, Err(FlowError::Netlist { path, .. }) if path == Path::new(netlist)),
        "{result:?}"
    );
}

#[test]
fn a_cancelled_flow_stops_before_packing() {
    let live = Live::new();
    live.cancel();
    let options = FlowOptions::new(fixture("example.bin"), blinky());
    let (result, lines) = flow(&options, Some(live.clone()));
    assert!(
        matches!(result, Err(FlowError::Cancelled(Cancelled))),
        "{result:?}"
    );
    assert!(lines.is_empty());
    assert!(live.latest().is_none());
}

#[rstest]
#[case(0.0)]
#[case(-12.0)]
#[case(f64::NAN)]
fn a_clock_target_must_be_positive_and_finite(#[case] mhz: f64) {
    let live = Live::new();
    let mut options = FlowOptions::new(fixture("example.bin"), blinky());
    options.freq_mhz = Some(mhz);
    let (result, lines) = flow(&options, Some(live.clone()));
    assert!(matches!(result, Err(FlowError::Frequency(_))), "{result:?}");
    assert_eq!(lines, ["Packing"]);
    assert_eq!(live.latest().unwrap().frame.phase, "pack");
}

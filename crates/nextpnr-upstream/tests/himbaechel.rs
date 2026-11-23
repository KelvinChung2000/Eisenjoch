use std::fs;
use std::path::PathBuf;
use std::process::Command;

use nextpnr_upstream::{Error, Nextpnr, Placer, Request, Router, Stage};

const NETLIST: &str = r#"{
  "modules": {"top": {
    "ports": {"a": {"direction": "input", "bits": [2]}, "y": {"direction": "output", "bits": [4]}},
    "cells": {
      "in": {"type": "INBUF", "port_directions": {"PAD": "input", "O": "output"}, "connections": {"PAD": [2], "O": [3]}},
      "lut": {"type": "LUT4", "parameters": {"INIT": "1010101010101010"}, "port_directions": {"I": "input", "F": "output"}, "connections": {"I": [3, "0", "0", "0"], "F": [5]}},
      "out": {"type": "OUTBUF", "port_directions": {"PAD": "output", "I": "input"}, "connections": {"PAD": [4], "I": [5]}}
    },
    "netnames": {"a": {"bits": [2]}, "y": {"bits": [4]}, "in_wire": {"bits": [3]}, "lut_wire": {"bits": [5]}}
  }}
}"#;

#[test]
#[ignore = "requires NEXTPNR_BUILD with the bridge and upstream Himbächel executable"]
fn matches_upstream_cli() {
    let build = PathBuf::from(std::env::var_os("NEXTPNR_BUILD").expect("set NEXTPNR_BUILD"));
    let library = build.join("libeisenjoch-nextpnr.so");
    let bridge = unsafe { Nextpnr::load(&library) }.unwrap();
    let chipdb = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nextpnr/tests/fixtures/npnr_baseline/shared_chipdb.bin");
    let scratch = tempfile::tempdir().unwrap();
    let input = scratch.path().join("input.json");
    let reference = scratch.path().join("reference.json");
    let output = scratch.path().join("bridge.json");
    let mut request = Request {
        device: "EXAMPLE",
        chipdb: &chipdb,
        input: &input,
        output: &output,
        placer: None,
        router: None,
        seed: 1,
        stop_after: Stage::Route,
    };

    fs::write(&input, "invalid JSON").unwrap();
    fs::write(&output, "existing output").unwrap();
    let error = unsafe { bridge.run(&request) }.unwrap_err();
    assert!(matches!(error, Error::Nextpnr(message) if message.contains("Failed to parse JSON")));
    assert_eq!(fs::read_to_string(&output).unwrap(), "existing output");

    fs::write(&input, NETLIST).unwrap();
    for (placer, router, stop_after, flags) in [
        (None, None, Stage::Pack, ["--pack-only"].as_slice()),
        (None, None, Stage::Place, ["--no-route"].as_slice()),
        (None, None, Stage::Route, [].as_slice()),
        (
            Some(Placer::Heap),
            Some(Router::Router2),
            Stage::Route,
            ["--placer", "heap", "--router", "router2"].as_slice(),
        ),
        (
            Some(Placer::Sa),
            Some(Router::Router1),
            Stage::Route,
            ["--placer", "sa", "--router", "router1"].as_slice(),
        ),
    ] {
        request.placer = placer;
        request.router = router;
        request.stop_after = stop_after;
        let reference_run = Command::new(build.join("nextpnr-himbaechel"))
            .args(["--device", request.device, "--seed", "1", "--chipdb"])
            .arg(&chipdb)
            .arg("--json")
            .arg(&input)
            .arg("--write")
            .arg(&reference)
            .args(flags)
            .output()
            .unwrap();
        assert!(
            reference_run.status.success(),
            "{flags:?}: {}",
            String::from_utf8_lossy(&reference_run.stderr)
        );
        unsafe { bridge.run(&request) }.unwrap();
        let expected = fs::read_to_string(&reference).unwrap();
        let actual = fs::read_to_string(&output).unwrap();
        assert_eq!(actual, expected, "{flags:?}");
    }
}

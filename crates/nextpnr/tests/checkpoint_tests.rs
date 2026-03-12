mod common;

use nextpnr::checkpoint::{
    self, CellPlacement, CellSig, Checkpoint, DesignDiff, DesignFingerprint, NetRoute, NetSig,
    CHECKPOINT_VERSION,
};

// =============================================================
// Checkpoint save/load roundtrip
// =============================================================

#[test]
fn checkpoint_save_load_roundtrip() {
    let cp = Checkpoint {
        version: CHECKPOINT_VERSION,
        placements: vec![CellPlacement {
            cell_name: "cell_0".into(),
            cell_type: "LUT4".into(),
            bel_tile: 0,
            bel_index: 0,
            bel_name: "LUT4_0".into(),
            tile_name: "CLB_X0Y0".into(),
            tile_type: "CLB".into(),
            strength: 3,
        }],
        routes: vec![NetRoute {
            net_name: "net_0".into(),
            source_wire_tile: 0,
            source_wire_index: 0,
            source_wire_name: "CLB_X0Y0/O0".into(),
            pips: vec![(0, 1), (1, 2)],
            pip_names: vec!["CLB_X0Y0/I0.O0".into(), "CLB_X0Y1/I1.O1".into()],
        }],
        fingerprint: DesignFingerprint {
            cell_signatures: vec![CellSig {
                name: "cell_0".into(),
                cell_type: "LUT4".into(),
                port_count: 2,
            }],
            net_signatures: vec![NetSig {
                name: "net_0".into(),
                driver_cell: "cell_0".into(),
                driver_port: "Q".into(),
                user_count: 1,
            }],
        },
    };

    let tmp_dir = std::env::temp_dir().join("nextpnr_test_checkpoint");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let path = tmp_dir.join("test_checkpoint.json");

    cp.save_to_file(&path).expect("save should succeed");
    let loaded = Checkpoint::load_from_file(&path).expect("load should succeed");

    assert_eq!(loaded.version, CHECKPOINT_VERSION);
    assert_eq!(loaded.placements.len(), 1);
    assert_eq!(loaded.placements[0].cell_name, "cell_0");
    assert_eq!(loaded.routes.len(), 1);
    assert_eq!(loaded.routes[0].pips.len(), 2);
    assert_eq!(loaded.fingerprint.cell_signatures.len(), 1);
    assert_eq!(loaded.fingerprint.net_signatures.len(), 1);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// =============================================================
// Version mismatch
// =============================================================

#[test]
fn checkpoint_version_mismatch() {
    let cp = Checkpoint {
        version: 999,
        placements: vec![],
        routes: vec![],
        fingerprint: DesignFingerprint {
            cell_signatures: vec![],
            net_signatures: vec![],
        },
    };

    let tmp_dir = std::env::temp_dir().join("nextpnr_test_version");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let path = tmp_dir.join("bad_version.json");

    cp.save_to_file(&path).expect("save should succeed");
    let err = Checkpoint::load_from_file(&path);
    assert!(err.is_err());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// =============================================================
// DesignDiff tests
// =============================================================

#[test]
fn diff_no_changes() {
    let fp = DesignFingerprint {
        cell_signatures: vec![CellSig {
            name: "c0".into(),
            cell_type: "LUT4".into(),
            port_count: 2,
        }],
        net_signatures: vec![NetSig {
            name: "n0".into(),
            driver_cell: "c0".into(),
            driver_port: "Q".into(),
            user_count: 1,
        }],
    };

    let diff = DesignDiff::compute(&fp, &fp);
    assert!(!diff.has_changes());
}

#[test]
fn diff_added_cell() {
    let old = DesignFingerprint {
        cell_signatures: vec![],
        net_signatures: vec![],
    };
    let new = DesignFingerprint {
        cell_signatures: vec![CellSig {
            name: "new_cell".into(),
            cell_type: "LUT4".into(),
            port_count: 1,
        }],
        net_signatures: vec![],
    };

    let diff = DesignDiff::compute(&old, &new);
    assert!(diff.has_changes());
    assert_eq!(diff.added_cells.len(), 1);
    assert_eq!(diff.added_cells[0], "new_cell");
    assert!(diff.removed_cells.is_empty());
}

#[test]
fn diff_removed_cell() {
    let old = DesignFingerprint {
        cell_signatures: vec![CellSig {
            name: "old_cell".into(),
            cell_type: "LUT4".into(),
            port_count: 1,
        }],
        net_signatures: vec![],
    };
    let new = DesignFingerprint {
        cell_signatures: vec![],
        net_signatures: vec![],
    };

    let diff = DesignDiff::compute(&old, &new);
    assert!(diff.has_changes());
    assert_eq!(diff.removed_cells.len(), 1);
}

#[test]
fn diff_changed_cell() {
    let old = DesignFingerprint {
        cell_signatures: vec![CellSig {
            name: "c0".into(),
            cell_type: "LUT4".into(),
            port_count: 2,
        }],
        net_signatures: vec![],
    };
    let new = DesignFingerprint {
        cell_signatures: vec![CellSig {
            name: "c0".into(),
            cell_type: "LUT6".into(), // type changed
            port_count: 2,
        }],
        net_signatures: vec![],
    };

    let diff = DesignDiff::compute(&old, &new);
    assert!(diff.has_changes());
    assert_eq!(diff.changed_cells.len(), 1);
}

// =============================================================
// Fingerprint computation
// =============================================================

#[test]
fn fingerprint_from_context() {
    let ctx = common::make_context_with_cells(2);
    let fp = checkpoint::compute_fingerprint(&ctx);

    assert_eq!(fp.cell_signatures.len(), 2);
    assert_eq!(fp.net_signatures.len(), 1);
    // Sorted by name
    assert!(fp.cell_signatures[0].name < fp.cell_signatures[1].name);
}

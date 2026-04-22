//! Integration tests that assert the `xc7_large` chipdb encodes long wires
//! (LH/LV) as fully connected nodes. These tests load the real `.bin`
//! produced by the generator in `chip_database/`. If the chipdb has not
//! been built yet the tests skip themselves.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

fn chipdb_path() -> Option<PathBuf> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let candidates = [
        Path::new(manifest_dir).join("../../chip_database/xc7_large.bin"),
        Path::new("chip_database/xc7_large.bin").to_path_buf(),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn load_db() -> Option<ChipDb> {
    let path = chipdb_path()?;
    ChipDb::load(&path).ok()
}

fn find_mid_clb_tile(db: &ChipDb) -> i32 {
    let mid_x = db.width() / 2;
    let mid_y = db.height() / 2;
    let clb_tt = (0..db.num_tile_types() as i32)
        .find(|&idx| {
            (0..db.num_tiles())
                .any(|t| db.tile_type_index(t) == idx && db.tile_type_name(t) == "CLB")
        })
        .expect("chipdb has no CLB tile type");
    (0..db.num_tiles())
        .filter(|&t| db.tile_type_index(t) == clb_tt)
        .min_by_key(|&t| {
            let (x, y) = db.tile_xy(t);
            (x - mid_x).abs() + (y - mid_y).abs()
        })
        .expect("no CLB tile instance")
}

fn wire_peer_count(db: &ChipDb, wire: WireId) -> usize {
    let mut n = 0;
    db.node_wires_cb(wire, |_| n += 1);
    n
}

fn find_wires_by_name_prefix(db: &ChipDb, tile: i32, prefix: &str) -> Vec<(String, WireId)> {
    let tt = db.tile_type_by_index(db.tile_type_index(tile));
    let n_wires = tt.wires.len();
    let mut out = Vec::new();
    for wi in 0..n_wires {
        let wire = WireId::new(tile, wi as i32);
        let name = db.wire_name(wire).to_string();
        if name.starts_with(prefix) {
            out.push((name, wire));
        }
    }
    out
}

/// LH (long horizontal, 12-tile span) must form cross-composite nodes.
/// INT_L's LH wires land in the composite as `M1_LH*`. Each LH tap should
/// belong to a node spanning at least two composite columns.
#[test]
fn m1_lh_wires_have_node_peers() {
    let Some(db) = load_db() else {
        eprintln!("skipping: chip_database/xc7_large.bin not present");
        return;
    };
    let tile = find_mid_clb_tile(&db);
    let lh_wires = find_wires_by_name_prefix(&db, tile, "M1_LH");
    assert!(
        !lh_wires.is_empty(),
        "no M1_LH* wires found in mid-chip CLB tile"
    );

    let mut zero_peer = Vec::new();
    for (name, wire) in &lh_wires {
        let peers = wire_peer_count(&db, *wire);
        if peers == 0 {
            zero_peer.push(name.clone());
        }
    }
    assert!(
        zero_peer.is_empty(),
        "{} of {} M1_LH wires have zero node peers (chipdb LH connectivity broken): {:?}",
        zero_peer.len(),
        lh_wires.len(),
        zero_peer
    );
}

/// The combined LH tap set (M0/M1/M2/M3 long-horizontal wires at the same
/// tap index in one composite) should unify into a node reaching across
/// at least 3 composite columns. With the prior generator every LH tap
/// was isolated inside a single composite, so the max dx seen via node
/// equivalence was 0 or 1.
#[test]
fn lh_nodes_span_multiple_columns() {
    let Some(db) = load_db() else {
        eprintln!("skipping: chip_database/xc7_large.bin not present");
        return;
    };
    let tile = find_mid_clb_tile(&db);
    let (tx, _ty) = db.tile_xy(tile);

    let mut candidates: Vec<(String, WireId)> = Vec::new();
    for prefix in ["M0_CLBLL_LH", "M1_LH", "M2_LH", "M3_CLBLM_LH"] {
        candidates.extend(find_wires_by_name_prefix(&db, tile, prefix));
    }
    assert!(!candidates.is_empty(), "no LH wires found");

    let mut max_dx_span = 0i32;
    for (_name, wire) in &candidates {
        let mut xs: FxHashSet<i32> = FxHashSet::default();
        db.node_wires_cb(*wire, |nw| {
            let (nx, _) = db.tile_xy(nw.tile());
            xs.insert(nx - tx);
        });
        if let (Some(&lo), Some(&hi)) = (xs.iter().min(), xs.iter().max()) {
            let span = hi - lo;
            if span > max_dx_span {
                max_dx_span = span;
            }
        }
    }
    // Physical xc7 LH wire spans 12 INT tiles = 6 composite columns.
    // Allow a little slack, but a chipdb where the best LH node spans 1
    // column or less has definitely lost the long-wire topology.
    assert!(
        max_dx_span >= 3,
        "best LH node only spans {} composite columns; expected >= 3",
        max_dx_span
    );
}

/// Pure-chipdb BFS test: starting from a mid-row left-column IOB
/// `IO_BRIDGE_OUT*` wire, can we reach *any* right-column IOB
/// `IO_BRIDGE_IN*` wire via PIP hops + node equivalence? This catches
/// LH-chain gaps across DSP/BRAM composites (previously LH broke at the
/// first non-CLB column).
#[test]
fn left_io_reaches_right_io() {
    let Some(db) = load_db() else {
        eprintln!("skipping: chip_database/xc7_large.bin not present");
        return;
    };

    let w = db.width();
    let h = db.height();

    let (mut min_iob_x, mut max_iob_x) = (i32::MAX, i32::MIN);
    for t in 0..db.num_tiles() {
        if db.tile_type_name(t) != "IOB" {
            continue;
        }
        let (tx, _) = db.tile_xy(t);
        if tx < min_iob_x {
            min_iob_x = tx;
        }
        if tx > max_iob_x {
            max_iob_x = tx;
        }
    }
    assert!(
        max_iob_x > min_iob_x,
        "expected two IOB columns, got min={} max={}",
        min_iob_x,
        max_iob_x
    );

    let mid_y = h / 2;
    let collect_wires = |col: i32, prefix: &str| -> Vec<WireId> {
        let mut out = Vec::new();
        for t in 0..db.num_tiles() {
            if db.tile_type_name(t) != "IOB" {
                continue;
            }
            let (tx, ty) = db.tile_xy(t);
            if tx != col || (ty - mid_y).abs() >= 8 {
                continue;
            }
            let tt = db.tile_type_by_index(db.tile_type_index(t));
            for wi in 0..tt.wires.len() {
                let wire = WireId::new(t, wi as i32);
                let name = db.wire_name(wire).to_string();
                if name.starts_with(prefix) {
                    out.push(wire);
                }
            }
        }
        out
    };
    let left_sources = collect_wires(min_iob_x, "IO_BRIDGE_OUT");
    let right_sinks: FxHashSet<WireId> =
        collect_wires(max_iob_x, "IO_BRIDGE_IN").into_iter().collect();
    assert!(
        !left_sources.is_empty(),
        "no left IO_BRIDGE_OUT source wires found"
    );
    assert!(!right_sinks.is_empty(), "no right IO_BRIDGE_IN sinks found");

    // Cap the BFS — the chipdb has ~60M wires so unconditional BFS is slow.
    let cap: usize = std::env::var("BFS_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000_000);

    let src = left_sources[0];
    let (sx, _sy) = db.tile_xy(src.tile());
    let mut visited: FxHashSet<WireId> = FxHashSet::default();
    let mut queue: VecDeque<WireId> = VecDeque::new();
    visited.insert(src);
    queue.push_back(src);

    let mut popped = 0usize;
    let mut max_dx = 0i32;
    let mut reached: Option<WireId> = None;
    while let Some(wire) = queue.pop_front() {
        popped += 1;
        if popped > cap {
            break;
        }
        if right_sinks.contains(&wire) {
            reached = Some(wire);
            break;
        }
        let (wx, _) = db.tile_xy(wire.tile());
        let dx = wx - sx;
        if dx.abs() > max_dx {
            max_dx = dx.abs();
        }

        let info = db.wire_info(wire);
        for &pip_idx in info.pips_downhill.get() {
            let pip = PipId::new(wire.tile(), pip_idx);
            let dst = db.pip_dst_wire(pip);
            if visited.insert(dst) {
                queue.push_back(dst);
            }
        }
        db.node_wires_cb(wire, |nw| {
            if visited.insert(nw) {
                queue.push_back(nw);
            }
        });
    }

    assert!(
        reached.is_some(),
        "left IOB (x={}) cannot reach any right IOB (x={}); \
         BFS popped={} visited={} max_dx_seen={} \
         — LH chain is broken across DSP/BRAM composites",
        min_iob_x,
        max_iob_x,
        popped,
        visited.len(),
        max_dx
    );
}

/// Sanity check that LV (long vertical, 18-tile span) is already correct.
/// If this test fails, the whole generator regressed (not only LH is broken).
#[test]
fn lv_nodes_span_multiple_rows() {
    let Some(db) = load_db() else {
        eprintln!("skipping: chip_database/xc7_large.bin not present");
        return;
    };
    let tile = find_mid_clb_tile(&db);
    let (_tx, ty) = db.tile_xy(tile);

    let mut candidates: Vec<(String, WireId)> = Vec::new();
    for prefix in ["M1_LV", "M2_LV"] {
        candidates.extend(find_wires_by_name_prefix(&db, tile, prefix));
    }
    assert!(!candidates.is_empty(), "no LV wires found");

    let mut max_dy_span = 0i32;
    for (_name, wire) in &candidates {
        let mut ys: FxHashSet<i32> = FxHashSet::default();
        db.node_wires_cb(*wire, |nw| {
            let (_, ny) = db.tile_xy(nw.tile());
            ys.insert(ny - ty);
        });
        if let (Some(&lo), Some(&hi)) = (ys.iter().min(), ys.iter().max()) {
            let span = hi - lo;
            if span > max_dy_span {
                max_dy_span = span;
            }
        }
    }
    assert!(
        max_dy_span >= 10,
        "best LV node only spans {} rows; expected >= 10",
        max_dy_span
    );
}

//! Test the hypothesis: pip_delay correlates with the *effective tap
//! distance* of the pip's destination wire, with low within-bucket variance.
//!
//! Multi-tap-aware: a single physical wire that spans 4 tiles also taps at
//! 1, 2, and 3 tiles, so it serves as a routing resource at all those
//! distances at the same pip cost. We classify each (pip, tap) pair —
//! i.e. for every pip driving a wire, we record one (cost, tap_distance)
//! sample per node-tile-offset of that wire (excluding the zero offset).
//!
//! For each tile type:
//!   1. For each pip P with destination wire D, enumerate D's node members:
//!        for each (dx, dy) offset of node-member, record (pip_cost,
//!        |dx|+|dy|) into the bucket of |dx|+|dy|.
//!   2. Print min/p25/p50/p75/max per (tile_type, tap_distance_bucket).
//!
//! If pip_cost is uniform per tap-distance bucket, a flat
//! `(tile_type, side, tap_bucket) -> cost` lookahead is tight and
//! admissible.
//!
//! Usage: probe_pip_delay_vs_span [chipdb-path]

use nextpnr::chipdb::{ChipDb, PipId};
use nextpnr::context::Context;
use nextpnr::read_packed;
use nextpnr::router::astar::default_pip_cost;
use std::path::Path;

fn span_bucket(span: i32) -> u8 {
    match span {
        0 | 1 => 0,
        2 | 3 => 1,
        4..=6 => 2,
        7..=12 => 3,
        _ => 4,
    }
}

fn bucket_label(b: u8) -> &'static str {
    match b {
        0 => "<=1   ",
        1 => "2-3   ",
        2 => "4-6   ",
        3 => "7-12  ",
        4 => ">12   ",
        _ => "?     ",
    }
}

fn percentile(sorted: &[i32], p: f64) -> i32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let db_owned = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db_owned);
    let db = ctx.chipdb();
    let w = db.width();
    let h = db.height();
    let num_tt = db.num_tile_types();

    let is_interior = |tile: i32| -> bool {
        let (x, y) = db.tile_xy(tile);
        x > 0 && x < w - 1 && y > 0 && y < h - 1
    };
    let mut rep: Vec<Option<i32>> = vec![None; num_tt];
    for tile in 0..(w * h) {
        let tt = db.tile_type_index(tile) as usize;
        match rep[tt] {
            None => rep[tt] = Some(tile),
            Some(curr) if !is_interior(curr) && is_interior(tile) => rep[tt] = Some(tile),
            _ => {}
        }
    }

    println!(
        "{:<4}  {:<22}  {:<6}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>6}",
        "tt", "type_name", "bucket", "n", "min", "p25", "p50", "p75", "max"
    );

    for tt_idx in 0..num_tt {
        let Some(t) = rep[tt_idx] else { continue };
        let tt = db.tile_type_by_index(tt_idx as i32);
        let wires_pod = tt.wires.get();
        let pips_pod = tt.pips.get();
        let n_wires = wires_pod.len();
        if n_wires == 0 || pips_pod.is_empty() {
            continue;
        }
        let tt_name = db.tile_type_name(t).to_string();

        // Per-wire list of tap distances (|dx|+|dy| for every node-member,
        // including 0 to represent "stays in this tile"). For tile-local
        // wires the only tap is 0.
        let mut wire_taps: Vec<Vec<i32>> = vec![Vec::new(); n_wires];
        for wi in 0..n_wires {
            if let Some(shape) = db.wire_node_shape(t, wi) {
                for tw in shape.tile_wires.get() {
                    let dx: i16 = unsafe { read_packed!(*tw, dx) };
                    let dy: i16 = unsafe { read_packed!(*tw, dy) };
                    wire_taps[wi].push((dx as i32).abs() + (dy as i32).abs());
                }
            } else {
                wire_taps[wi].push(0);
            }
        }

        // For each (pip, tap-distance) pair, record (cost, tap-bucket).
        // A pip into a multi-tap wire contributes one sample per tap.
        let mut by_bucket: [Vec<i32>; 5] = Default::default();
        for (i, pip) in pips_pod.iter().enumerate() {
            let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if dst < 0 || (dst as usize) >= n_wires {
                continue;
            }
            let pid = PipId::new(t, i as i32);
            let cost = default_pip_cost(&ctx, pid) as i32 - 1;
            for &tap in &wire_taps[dst as usize] {
                if tap == 0 {
                    // Pure intra-tile arrival; tracked under <=1 bucket.
                    by_bucket[0].push(cost);
                } else {
                    by_bucket[span_bucket(tap) as usize].push(cost);
                }
            }
        }

        for b in 0..5 {
            let v = &mut by_bucket[b];
            if v.is_empty() {
                continue;
            }
            v.sort_unstable();
            let n = v.len();
            let min = v[0];
            let p25 = percentile(v, 0.25);
            let p50 = percentile(v, 0.50);
            let p75 = percentile(v, 0.75);
            let max = *v.last().unwrap();
            println!(
                "{:<4}  {:<22}  {}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>6}",
                tt_idx, tt_name, bucket_label(b as u8), n, min, p25, p50, p75, max,
            );
        }
    }
}

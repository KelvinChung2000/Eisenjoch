//! Find pips whose dst_wire node-share peers span unusually large
//! manhattan distances. Diagnoses the 209-tile span outlier seen in
//! the lookahead build.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::read_packed;
use nextpnr::router::astar::default_pip_cost;
use std::path::Path;

fn main() {
    let p = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let db = ChipDb::load(Path::new(p)).expect("load");
    let ctx = Context::new(db);
    let cdb = ctx.chipdb();
    let w = cdb.width();
    let h = cdb.height();
    let num_tile_types = cdb.num_tile_types();

    const REP_MARGIN: i32 = 13;
    let is_interior = |tile: i32| -> bool {
        let (x, y) = cdb.tile_xy(tile);
        x >= REP_MARGIN && x < w - REP_MARGIN && y >= REP_MARGIN && y < h - REP_MARGIN
    };
    let mut rep_of_type: Vec<Option<i32>> = vec![None; num_tile_types];
    for tile in 0..(w * h) {
        let tt = cdb.tile_type_index(tile) as usize;
        match rep_of_type[tt] {
            None => rep_of_type[tt] = Some(tile),
            Some(curr) if !is_interior(curr) && is_interior(tile) => {
                rep_of_type[tt] = Some(tile);
            }
            _ => {}
        }
    }

    // (rate_float, span, cost, tt_name, rep_xy, src_wire_name, dst_wire_name, peer_tile_xy, peer_wire_name)
    let mut hits: Vec<(
        f64,
        i64,
        i64,
        String,
        (i32, i32),
        String,
        String,
        (i32, i32),
        String,
    )> = Vec::new();

    for tt in 0..num_tile_types {
        let Some(rep_tile) = rep_of_type[tt] else {
            continue;
        };
        let (rep_x, rep_y) = cdb.tile_xy(rep_tile);
        let tt_name = cdb.tile_type_name(rep_tile).to_string();
        let pips = cdb.tile_type_by_index(tt as i32).pips.get();
        for (pip_idx, pip) in pips.iter().enumerate() {
            let dst_wi: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if dst_wi < 0 {
                continue;
            }
            let src_wi: i32 = unsafe { read_packed!(*pip, src_wire) };
            let pid = PipId::new(rep_tile, pip_idx as i32);
            let cost = default_pip_cost(&ctx, pid).max(1) as i64;
            let dst_wire = WireId::new(rep_tile, dst_wi);
            let mut max_span: i64 = 0;
            let mut peer_xy = (rep_x, rep_y);
            let mut peer_name = String::new();
            cdb.node_wires_cb(dst_wire, |peer| {
                let (px, py) = cdb.tile_xy(peer.tile());
                let s = ((px - rep_x).abs() + (py - rep_y).abs()) as i64;
                if s > max_span {
                    max_span = s;
                    peer_xy = (px, py);
                    let info = cdb.wire_info(peer);
                    let nid: i32 = unsafe { read_packed!(*info, name) };
                    peer_name = cdb
                        .constid_str(nid)
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                }
            });
            if max_span < 30 {
                continue;
            }
            let span = max_span.max(1);
            let rate = cost as f64 / span as f64;
            // Get src/dst wire names.
            let tt_tmpl = cdb.tile_type_by_index(tt as i32);
            let wires = tt_tmpl.wires.get();
            let src_name = if (src_wi as usize) < wires.len() {
                let nid: i32 = unsafe { read_packed!(wires[src_wi as usize], name) };
                cdb.constid_str(nid)
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            } else {
                "?".into()
            };
            let dst_name = if (dst_wi as usize) < wires.len() {
                let nid: i32 = unsafe { read_packed!(wires[dst_wi as usize], name) };
                cdb.constid_str(nid)
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            } else {
                "?".into()
            };
            hits.push((
                rate,
                span,
                cost,
                tt_name.clone(),
                (rep_x, rep_y),
                src_name,
                dst_name,
                peer_xy,
                peer_name,
            ));
        }
    }

    // Sort by rate ascending (cheapest = most suspicious for our heuristic).
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("=== top 30 lowest-rate pips with span >= 30 ===");
    for (rate, span, cost, tt_name, (rx, ry), sn, dn, (px, py), pn) in hits.iter().take(30) {
        println!(
            "  rate={rate:.3} span={span:>3} cost={cost:>3}  tt={tt_name:<10} rep=({rx},{ry})  {sn} -> {dn}  peer=({px},{py}):{pn}",
        );
    }

    println!("\n=== span histogram (>=30) ===");
    let mut bucket = [0usize; 12];
    for h in &hits {
        let s = h.1;
        let i = match s {
            30..=49 => 0,
            50..=69 => 1,
            70..=99 => 2,
            100..=129 => 3,
            130..=159 => 4,
            160..=199 => 5,
            200..=249 => 6,
            250..=299 => 7,
            _ => 8,
        };
        bucket[i] += 1;
    }
    let labels = [
        "30-49", "50-69", "70-99", "100-129", "130-159", "160-199", "200-249", "250-299", "300+",
        "", "", "",
    ];
    for (i, n) in bucket.iter().enumerate() {
        if *n > 0 {
            println!("  {}: {}", labels[i], n);
        }
    }
    println!("total long-span pips: {}", hits.len());
}

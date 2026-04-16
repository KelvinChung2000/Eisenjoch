//! Two-level cost-cache audit for `opt_trans`.
//!
//! Phase A gate for the v2 plan: verifies that chipdb PIP-level tile transit
//! is cheap enough to run on demand during cache misses.
//!
//! Gate criteria (printed as PASS/FAIL at the end):
//! - mean tile-local Dijkstra ≤ 1 ms
//! - max `n_pips` per tile type ≤ 20 000

use nextpnr::chipdb::ChipDb;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Side {
    North,
    East,
    South,
    West,
}

fn classify_side(dx: i32, dy: i32) -> Option<Side> {
    if dx == 0 && dy == 0 {
        return None;
    }
    if dx.abs() >= dy.abs() {
        if dx > 0 {
            Some(Side::East)
        } else {
            Some(Side::West)
        }
    } else if dy > 0 {
        Some(Side::South)
    } else {
        Some(Side::North)
    }
}

fn build_pip_csr(tt: &nextpnr::chipdb::TileTypePod) -> (Vec<usize>, Vec<u32>) {
    let n_wires = tt.wires.len();
    let pips = tt.pips.get();
    let mut out_deg = vec![0usize; n_wires];
    for pip in pips {
        let src: i32 = unsafe { nextpnr::read_packed!(*pip, src_wire) };
        if (src as usize) < n_wires {
            out_deg[src as usize] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(n_wires + 1);
    offsets.push(0usize);
    let mut running = 0usize;
    for d in &out_deg {
        running += d;
        offsets.push(running);
    }
    let mut cursors = offsets.clone();
    let mut targets = vec![0u32; running];
    for pip in pips {
        let src: i32 = unsafe { nextpnr::read_packed!(*pip, src_wire) };
        let dst: i32 = unsafe { nextpnr::read_packed!(*pip, dst_wire) };
        if (src as usize) < n_wires {
            let slot = cursors[src as usize];
            cursors[src as usize] = slot + 1;
            targets[slot] = dst.max(0) as u32;
        }
    }
    (offsets, targets)
}

fn dijkstra_nodes_reached(
    offsets: &[usize],
    targets: &[u32],
    source: usize,
    n_wires: usize,
) -> (usize, u32) {
    let mut dist = vec![u32::MAX; n_wires];
    dist[source] = 0;
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0, source as u32)));
    let mut reached = 0usize;
    let mut max_d = 0u32;
    while let Some(Reverse((d, u))) = heap.pop() {
        let u_us = u as usize;
        if d > dist[u_us] {
            continue;
        }
        reached += 1;
        if d > max_d {
            max_d = d;
        }
        let start = offsets[u_us];
        let end = offsets[u_us + 1];
        for &v in &targets[start..end] {
            let v_us = v as usize;
            if v_us >= n_wires {
                continue;
            }
            let nd = d.saturating_add(1);
            if nd < dist[v_us] {
                dist[v_us] = nd;
                heap.push(Reverse((nd, v)));
            }
        }
    }
    (reached, max_d)
}

fn main() {
    let chipdb = std::env::var("NPNR_AUDIT_CHIPDB")
        .unwrap_or_else(|_| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".to_string());
    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");

    // Prefer an interior representative tile so boundary wires on all four
    // sides are visible in wire_node_shape. Fall back to the first tile of
    // the type if no interior tile exists.
    let mut repr_tile = vec![None; db.num_tile_types()];
    let mut instances = vec![0usize; db.num_tile_types()];
    let w_interior = (db.width() / 4).max(10);
    let h_interior = (db.height() / 4).max(10);
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type_index(tile) as usize;
        instances[tt] += 1;
        let x = tile % db.width() as i32;
        let y = tile / db.width() as i32;
        let is_interior = x >= w_interior
            && x < db.width() - w_interior
            && y >= h_interior
            && y < db.height() - h_interior;
        if is_interior || repr_tile[tt].is_none() {
            if is_interior {
                repr_tile[tt] = Some(tile);
            } else {
                repr_tile[tt].get_or_insert(tile);
            }
        }
    }

    println!("chipdb={}", chipdb);
    println!(
        "tiles={} width={} height={} tile_types={}",
        db.num_tiles(),
        db.width(),
        db.height(),
        db.num_tile_types()
    );
    println!(
        "tt_idx,name,instances,wires,pips,bels,bnd_N,bnd_E,bnd_S,bnd_W,span_states,max_span,projected_span_keys,side_span_pairs,wire_types,wtN,wtE,wtS,wtW,wt_detail"
    );

    let mut total_projected = 0usize;
    let mut max_pips_per_type = 0usize;
    let mut type_infos: Vec<(usize, String, usize, usize)> = Vec::new(); // (tt_idx, name, n_wires, n_pips)
    let mut type_side_span: Vec<(usize, usize)> = Vec::new(); // (tt_idx, n_side_span)
    for tt_idx in 0..db.num_tile_types() {
        let Some(tile) = repr_tile[tt_idx] else {
            continue;
        };
        let tt = db.tile_type_by_index(tt_idx as i32);
        let mut spans: FxHashSet<(i32, i32)> = FxHashSet::default();
        let mut side_span_pairs: FxHashSet<(Side, i32)> = FxHashSet::default();
        let mut wire_types: FxHashSet<(Side, i32)> = FxHashSet::default();
        let mut max_span = 0i32;
        let mut boundary_by_side: FxHashMap<Side, usize> = FxHashMap::default();
        for wire_idx in 0..tt.wires.len() {
            let Some(shape) = db.wire_node_shape(tile, wire_idx) else {
                continue;
            };
            let mut is_boundary = false;
            let mut dominant_side: Option<Side> = None;
            let mut best_span = 0i32;
            for tw in shape.tile_wires.get() {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                let dx = dx as i32;
                let dy = dy as i32;
                let span = dx.abs() + dy.abs();
                if span == 0 {
                    continue;
                }
                is_boundary = true;
                if span > best_span {
                    if let Some(s) = classify_side(dx, dy) {
                        dominant_side = Some(s);
                        best_span = span;
                    }
                }
                if let Some(side) = classify_side(dx, dy) {
                    side_span_pairs.insert((side, span));
                }
                // Canonical-half filter for span_states (avoid double-count).
                if dx < 0 || (dx == 0 && dy <= 0) {
                    continue;
                }
                spans.insert((dx, dy));
                max_span = max_span.max(span);
            }
            if is_boundary {
                if let Some(side) = dominant_side {
                    *boundary_by_side.entry(side).or_insert(0) += 1;
                    wire_types.insert((side, best_span));
                }
            }
        }
        let n_side_span = side_span_pairs.len();
        let n_wire_types = wire_types.len();
        let mut wt_by_side: FxHashMap<Side, usize> = FxHashMap::default();
        for (side, _) in &wire_types {
            *wt_by_side.entry(*side).or_insert(0) += 1;
        }
        let mut side_span_by_side: FxHashMap<Side, usize> = FxHashMap::default();
        for (side, _span) in &side_span_pairs {
            *side_span_by_side.entry(*side).or_insert(0) += 1;
        }
        let mut wt_spans: Vec<_> = wire_types.iter().collect();
        wt_spans.sort_by_key(|(s, sp)| (*s as u8, *sp));
        let projected = spans.len() * 64;
        total_projected += projected;
        let name = db.tile_type_name(tile).to_string();
        let n_pips = tt.pips.len();
        let n_wires = tt.wires.len();
        if n_pips > max_pips_per_type {
            max_pips_per_type = n_pips;
        }
        type_infos.push((tt_idx, name.clone(), n_wires, n_pips));
        type_side_span.push((tt_idx, n_wire_types));
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            tt_idx,
            name,
            instances[tt_idx],
            n_wires,
            n_pips,
            tt.bels.len(),
            boundary_by_side.get(&Side::North).copied().unwrap_or(0),
            boundary_by_side.get(&Side::East).copied().unwrap_or(0),
            boundary_by_side.get(&Side::South).copied().unwrap_or(0),
            boundary_by_side.get(&Side::West).copied().unwrap_or(0),
            spans.len(),
            max_span,
            projected,
            n_side_span,
            n_wire_types,
            wt_by_side.get(&Side::North).copied().unwrap_or(0),
            wt_by_side.get(&Side::East).copied().unwrap_or(0),
            wt_by_side.get(&Side::South).copied().unwrap_or(0),
            wt_by_side.get(&Side::West).copied().unwrap_or(0),
            wt_spans.iter().map(|(s, sp)| format!("{:?}:{}", s, sp)).collect::<Vec<_>>().join("|"),
        );
    }
    println!("projected_total_span_keys={}", total_projected);
    println!("max_pips_per_type={}", max_pips_per_type);

    // Dump raw wire shapes for CLB to diagnose side/span distribution.
    // Pick an interior CLB (not at the edge) so we see bidirectional wires.
    let mut clb_candidate = None;
    for tile in 0..db.num_tiles() {
        if db.tile_type_name(tile).contains("CLB") {
            let (x, y) = (tile % 311, tile / 311);
            if x > 10 && x < 300 && y > 10 && y < 210 {
                clb_candidate = Some(tile);
                break;
            }
        }
    }
    if let Some(clb_tile) = clb_candidate.or_else(|| {
        repr_tile
            .iter()
            .find(|t| t.is_some() && db.tile_type_name(t.unwrap()).contains("CLB"))
            .and_then(|t| *t)
    }) {
        let tt = db.tile_type(clb_tile);
        println!("\n--- CLB wire dump (tile={}) ---", clb_tile);
        let mut wire_summary: FxHashMap<(i32, i32), usize> = FxHashMap::default();
        let mut all_offsets: FxHashMap<(i32, i32), usize> = FxHashMap::default();
        let mut n_boundary = 0usize;
        let mut n_internal = 0usize;
        for wire_idx in 0..tt.wires.len() {
            let Some(shape) = db.wire_node_shape(clb_tile, wire_idx) else {
                n_internal += 1;
                continue;
            };
            let tile_wires = shape.tile_wires.get();
            let mut max_dx = 0i32;
            let mut max_dy = 0i32;
            let mut max_span = 0i32;
            for tw in tile_wires {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                let dx = dx as i32;
                let dy = dy as i32;
                let span = dx.abs() + dy.abs();
                if span > 0 {
                    *all_offsets.entry((dx, dy)).or_insert(0) += 1;
                }
                if span > max_span {
                    max_span = span;
                    max_dx = dx;
                    max_dy = dy;
                }
            }
            if max_span > 0 {
                n_boundary += 1;
                *wire_summary.entry((max_dx, max_dy)).or_insert(0) += 1;
            } else {
                n_internal += 1;
            }
        }
        println!("  total_wires={} boundary={} internal(no_shape_or_span0)={}", tt.wires.len(), n_boundary, n_internal);
        println!("  max_reach (dx,dy) -> wire_count:");
        let mut summary: Vec<_> = wire_summary.into_iter().collect();
        summary.sort_by_key(|&((dx, dy), _)| (dy, dx));
        for ((dx, dy), count) in &summary {
            let side = classify_side(*dx, *dy);
            println!("    ({:+4},{:+4}) {:?} count={}", dx, dy, side, count);
        }
        // Print wire names for the longest-span wires (span-222 candidates).
        // Print ALL boundary wires with their names.
        println!("  All boundary wires (max_span > 0):");
        for wire_idx in 0..tt.wires.len() {
            let Some(shape) = db.wire_node_shape(clb_tile, wire_idx) else {
                continue;
            };
            let tile_wires = shape.tile_wires.get();
            let mut max_sp = 0i32;
            let mut max_dx = 0i32;
            let mut max_dy = 0i32;
            let mut n_tiles = 0usize;
            for tw in tile_wires {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                let sp = (dx as i32).abs() + (dy as i32).abs();
                n_tiles += 1;
                if sp > max_sp {
                    max_sp = sp;
                    max_dx = dx as i32;
                    max_dy = dy as i32;
                }
            }
            if max_sp > 0 {
                let wid = nextpnr::chipdb::WireId::new(clb_tile, wire_idx as i32);
                let wname = db.wire_name(wid);
                let wtype = db.wire_type(wid);
                println!("    wire_idx={:4} name={:30} type={:15} max_reach=({:+4},{:+4}) n_tiles={}", wire_idx, wname, wtype, max_dx, max_dy, n_tiles);
            }
        }
    }

    let side_span_by_tt: FxHashMap<usize, usize> =
        type_side_span.iter().copied().collect();
    let mut fabric_edges_total = 0usize;
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type_index(tile) as usize;
        if let Some(n) = side_span_by_tt.get(&tt) {
            fabric_edges_total += n;
        }
    }
    println!("fabric_edges_projected={}", fabric_edges_total);

    // ---- Phase A micro-bench: 50 tile-local Dijkstras across a sample of
    //      tile types with non-empty PIP graphs. Random sources via a tiny
    //      deterministic LCG so results are reproducible without adding a rand
    //      dep.
    let mut lcg: u64 = 0xDEADBEEF_CAFEBABE;
    let mut next_rand = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lcg
    };

    let sample_types: Vec<&(usize, String, usize, usize)> = type_infos
        .iter()
        .filter(|(_, _, _, n_pips)| *n_pips > 0)
        .collect();

    let mut samples = Vec::with_capacity(50);
    let n_samples = 50usize;
    for i in 0..n_samples {
        if sample_types.is_empty() {
            break;
        }
        let (tt_idx, name, n_wires, n_pips) = sample_types[i % sample_types.len()];
        let tt = db.tile_type_by_index(*tt_idx as i32);
        let (offsets, targets) = build_pip_csr(tt);
        // Pick a deterministic-random source wire with at least one outgoing PIP.
        // Fallback to 0 if no such wire exists.
        let mut source = (next_rand() as usize) % (*n_wires).max(1);
        for _ in 0..32 {
            if offsets[source + 1] > offsets[source] {
                break;
            }
            source = (next_rand() as usize) % (*n_wires).max(1);
        }
        let t0 = Instant::now();
        let (reached, max_d) = dijkstra_nodes_reached(&offsets, &targets, source, *n_wires);
        let elapsed = t0.elapsed();
        samples.push((
            tt_idx,
            name.clone(),
            *n_wires,
            *n_pips,
            source,
            reached,
            max_d,
            elapsed.as_secs_f64() * 1000.0,
        ));
    }

    println!("micro_bench_tile_local_dijkstra:");
    println!("  idx,tt,name,wires,pips,source,reached,max_hops,time_ms");
    let mut total_ms = 0.0f64;
    let mut max_ms = 0.0f64;
    for (i, (tt, name, nw, np, src, reached, max_d, t_ms)) in samples.iter().enumerate() {
        total_ms += *t_ms;
        if *t_ms > max_ms {
            max_ms = *t_ms;
        }
        println!(
            "  {},{},{},{},{},{},{},{},{:.3}",
            i, tt, name, nw, np, src, reached, max_d, t_ms
        );
    }
    let n = samples.len().max(1);
    let mean_ms = total_ms / n as f64;
    println!("  samples={} mean_ms={:.3} max_ms={:.3}", samples.len(), mean_ms, max_ms);

    // ---- Gate verdict ----
    let gate_mean_pass = mean_ms <= 1.0;
    let gate_pips_pass = max_pips_per_type <= 20_000;
    let gate_pass = gate_mean_pass && gate_pips_pass;
    println!(
        "TwoLevelPip gate: {}",
        if gate_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  mean_tile_local_dijkstra_ms={:.3} (target <=1.000 -> {})",
        mean_ms,
        if gate_mean_pass { "ok" } else { "FAIL" }
    );
    println!(
        "  max_pips_per_type={} (target <=20000 -> {})",
        max_pips_per_type,
        if gate_pips_pass { "ok" } else { "FAIL" }
    );

    let mut by_span: FxHashMap<i32, usize> = FxHashMap::default();
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type(tile);
        for wire_idx in 0..tt.wires.len() {
            let Some(shape) = db.wire_node_shape(tile, wire_idx) else {
                continue;
            };
            for tw in shape.tile_wires.get() {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                let dx = dx as i32;
                let dy = dy as i32;
                let span = dx.abs() + dy.abs();
                if span == 0 || dx < 0 || (dx == 0 && dy <= 0) {
                    continue;
                }
                *by_span.entry(span).or_insert(0) += 1;
            }
        }
    }
    let mut spans: Vec<_> = by_span.into_iter().collect();
    spans.sort_by_key(|&(span, _)| span);
    println!("span,count");
    for (span, count) in spans {
        println!("{},{}", span, count);
    }
}

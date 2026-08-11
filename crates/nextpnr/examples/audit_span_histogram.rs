//! Audit: what `build_span_histograms` actually derives per tile type.
//!
//! Replicates the `wire_reach` rule (one entry per non-internal wire at its
//! max-Manhattan-reach delta) and prints, per tile type, the resulting
//! reach-key histogram plus the distribution of raw node shapes that produced
//! it. Answers "why does a SLICE<->SLICE span-1 pipe have capacity 2 when the
//! device model declares 96 switch wires per tile".

use nextpnr::chipdb::{ChipDb, WireId};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::path::Path;

fn is_global_network_wire(name: &str) -> bool {
    name.contains("GCLK")
        || name.contains("HCLK")
        || name.contains("GND")
        || name.contains("VCC")
        || name.contains("CLK_HROW")
        || name.contains("CLK_BUFG")
        || is_clock_ladder_wire(name)
}

/// Mirrors the production filter: wires named exactly `CLK<n>` / `CLK<n>_PREV`.
fn is_clock_ladder_wire(name: &str) -> bool {
    let body = name.strip_suffix("_PREV").unwrap_or(name);
    let Some(digits) = body.strip_prefix("CLK") else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Full sorted delta-set of a wire's node, or None if global / no shape.
fn node_deltas(db: &ChipDb, tile: i32, wire_idx: usize) -> Option<Vec<(i32, i32)>> {
    let wid = WireId::new(tile, wire_idx as i32);
    if is_global_network_wire(db.wire_name(wid)) {
        return None;
    }
    let ns = db.wire_node_shape(tile, wire_idx)?;
    let mut offs: Vec<(i32, i32)> = ns
        .tile_wires
        .get()
        .iter()
        .map(|tw| {
            let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
            let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
            (dx as i32, dy as i32)
        })
        .collect();
    offs.sort();
    Some(offs)
}

fn canonical((dx, dy): (i32, i32)) -> (i32, i32) {
    if dx > 0 || (dx == 0 && dy > 0) {
        (dx, dy)
    } else {
        (-dx, -dy)
    }
}

fn nonzero_taps(offs: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut taps: Vec<(i32, i32)> = Vec::new();
    for &t in offs {
        if t != (0, 0) && !taps.contains(&t) {
            taps.push(t);
        }
    }
    taps
}

/// OLD rule: canonicalised max-Manhattan-reach delta, one unit, one key.
fn max_reach_key(offs: &[(i32, i32)]) -> Option<(i32, i32)> {
    let taps = nonzero_taps(offs);
    let far = taps
        .into_iter()
        .max_by_key(|&(dx, dy)| dx.abs() + dy.abs())?;
    Some(canonical(far))
}

/// NEW rule: collinear nodes keep max-reach; stars split 1/k over their arms.
fn reach_contributions(offs: &[(i32, i32)]) -> Vec<((i32, i32), f64)> {
    let taps = nonzero_taps(offs);
    if taps.is_empty() {
        return Vec::new();
    }
    let (ax, ay) = taps[0];
    if taps.iter().all(|&(bx, by)| ax * by - bx * ay == 0) {
        let far = taps
            .iter()
            .copied()
            .max_by_key(|&(dx, dy)| dx.abs() + dy.abs())
            .expect("non-empty");
        vec![(canonical(far), 1.0)]
    } else {
        let share = 1.0 / taps.len() as f64;
        taps.iter().map(|&t| (canonical(t), share)).collect()
    }
}

fn main() {
    let chipdb = std::env::var("NPNR_AUDIT_CHIPDB")
        .expect("set NPNR_AUDIT_CHIPDB to the chipdb path");
    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");
    println!("chipdb={}", chipdb);
    println!(
        "tiles={} width={} height={} tile_types={}",
        db.num_tiles(),
        db.width(),
        db.height(),
        db.num_tile_types()
    );

    // Richest (tile_type, shape) representative, exactly as production does.
    let mut seen: FxHashMap<(i32, i32), i32> = FxHashMap::default();
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type_index(tile);
        if tt < 0 {
            continue;
        }
        seen.entry((tt, db.tile_shape_index(tile))).or_insert(tile);
    }
    let num_tt = db.num_tile_types();
    let mut best: Vec<Option<(i32, usize)>> = vec![None; num_tt];
    for ((tt_idx, _sh), tile) in &seen {
        let tt = db.tile_type_by_index(*tt_idx);
        let mut distinct: FxHashMap<(i32, i32), ()> = FxHashMap::default();
        for wire_idx in 0..tt.wires.len() {
            if let Some(offs) = node_deltas(&db, *tile, wire_idx) {
                if let Some(k) = max_reach_key(&offs) {
                    distinct.insert(k, ());
                }
            }
        }
        let n = distinct.len();
        let slot = &mut best[*tt_idx as usize];
        match slot {
            Some((_, bn)) if *bn >= n => {}
            _ => *slot = Some((*tile, n)),
        }
    }

    for (tt_idx, entry) in best.iter().enumerate() {
        let Some((tile, _)) = entry else { continue };
        let tt = db.tile_type_by_index(tt_idx as i32);
        let name = db.tile_type_name(*tile).to_string();

        let mut reach_hist: BTreeMap<(i32, i32), usize> = BTreeMap::new();
        let mut new_hist: BTreeMap<(i32, i32), f64> = BTreeMap::new();
        let mut shape_hist: BTreeMap<String, (usize, String)> = BTreeMap::new();
        let mut internal = 0usize;
        let mut global = 0usize;
        let mut no_shape = 0usize;
        let mut contributing = 0usize;

        for wire_idx in 0..tt.wires.len() {
            let wid = WireId::new(*tile, wire_idx as i32);
            if is_global_network_wire(db.wire_name(wid)) {
                global += 1;
                continue;
            }
            let Some(offs) = node_deltas(&db, *tile, wire_idx) else {
                no_shape += 1;
                continue;
            };
            for (k, share) in reach_contributions(&offs) {
                *new_hist.entry(k).or_insert(0.0) += share;
            }
            match max_reach_key(&offs) {
                None => internal += 1,
                Some(k) => {
                    contributing += 1;
                    *reach_hist.entry(k).or_insert(0) += 1;
                    // Group by the raw delta-set so we can see stars vs lines.
                    let sig = format!("{:?}", offs);
                    let e = shape_hist
                        .entry(sig)
                        .or_insert((0, format!("{:?}", k)));
                    e.0 += 1;
                }
            }
        }

        println!(
            "\n=== tile_type[{}] {} : {} wires (global={} no_shape={} internal={}) ===",
            tt_idx,
            name,
            tt.wires.len(),
            global,
            no_shape,
            internal
        );
        println!("  OLD max-reach histogram: {:?}", reach_hist);
        let pretty: BTreeMap<(i32, i32), String> = new_hist
            .iter()
            .map(|(k, v)| (*k, format!("{:.2}", v)))
            .collect();
        println!("  NEW histogram:           {:?}", pretty);
        let booked: f64 = new_hist.values().sum();
        println!(
            "  conservation: booked {:.4} units from {} contributing wires -> {}",
            booked,
            contributing,
            if (booked - contributing as f64).abs() < 1e-6 { "OK" } else { "MISMATCH" }
        );
        println!("  distinct node shapes: {}", shape_hist.len());
        let mut shapes: Vec<_> = shape_hist.into_iter().collect();
        shapes.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (sig, (n, key)) in shapes.iter().take(6) {
            let arms = sig.matches('(').count();
            println!(
                "    {:5} wires  arms={:2}  -> key {}   shape={}",
                n,
                arms,
                key,
                if sig.len() > 140 { &sig[..140] } else { sig }
            );
        }
    }
}

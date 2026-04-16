//! Audit: composite chipdb node-shape correctness.
//!
//! For each key span-wire class (LH, LV, EE4, NN6, NN2, etc.), sample a
//! handful of CLB composite tiles at different grid positions and print the
//! node shape (list of (dx, dy) offsets). This reveals:
//!   - Whether shared wires correctly show offsets in BOTH directions
//!     (some instances should see (+1, 0), others (-1, 0)).
//!   - Whether long wires span the expected number of composite tiles.
//!   - Whether the placer's single-representative-tile assumption misses
//!     half the connectivity.

use nextpnr::chipdb::{ChipDb, WireId};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let chipdb = std::env::var("NPNR_AUDIT_CHIPDB").unwrap_or_else(|_| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".to_string()
    });
    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");
    println!("chipdb={}", chipdb);
    println!(
        "tiles={} width={} height={} tile_types={}",
        db.num_tiles(),
        db.width(),
        db.height(),
        db.num_tile_types()
    );

    // Locate the CLB composite tile type by name.
    let mut clb_tt = -1i32;
    for tt_idx in 0..db.num_tile_types() {
        if let Some(first_tile) = (0..db.num_tiles()).find(|&t| db.tile_type_index(t) == tt_idx as i32) {
            if db.tile_type_name(first_tile) == "CLB" {
                clb_tt = tt_idx as i32;
                break;
            }
        }
    }
    if clb_tt < 0 {
        println!("no CLB tile type found");
        return;
    }

    let clb_tiles: Vec<i32> = (0..db.num_tiles())
        .filter(|&t| db.tile_type_index(t) == clb_tt)
        .collect();
    println!("CLB composite tiles: {}", clb_tiles.len());

    // Build wire-name → wire-idx lookup using the first CLB tile's tile type.
    let first_clb = clb_tiles[0];
    let tt = db.tile_type_by_index(clb_tt);
    let wire_by_name: FxHashMap<String, usize> = (0..tt.wires.len())
        .map(|i| {
            (
                db.wire_name(WireId::new(first_clb, i as i32)).to_string(),
                i,
            )
        })
        .collect();

    // Pick 8 sample CLB tiles at different X coordinates and one Y row.
    let mut by_x: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for &t in &clb_tiles {
        let (x, _y) = db.tile_xy(t);
        by_x.entry(x).or_default().push(t);
    }
    let sample_xs: Vec<i32> = {
        let xs: Vec<i32> = by_x.keys().copied().collect();
        let step = xs.len().max(1) / 8;
        (0..xs.len()).step_by(step.max(1)).take(8).map(|i| xs[i]).collect()
    };
    // Fix a Y row near middle.
    let sample_y = db.height() / 2;
    let samples: Vec<i32> = sample_xs
        .iter()
        .filter_map(|&x| {
            by_x.get(&x).and_then(|ts| {
                ts.iter().copied().min_by_key(|&t| {
                    let (_, y) = db.tile_xy(t);
                    (y - sample_y).abs()
                })
            })
        })
        .collect();

    let target_names = [
        // M0 = CLBLL_L (left CLB), M3 = CLBLM_R (right CLB in this composite).
        ("M0_CLBLL_LH6", "M0 LH midpoint tap (CLBLL_L)"),
        ("M0_CLBLL_LH12", "M0 LH end tap"),
        ("M3_CLBLM_LH1", "M3 LH tap (CLBLM_R) — should mirror M0 east"),
        ("M3_CLBLM_LH6", "M3 LH midpoint"),
        ("M3_CLBLM_LH12", "M3 LH end"),
        ("M1_LH0", "M1 LH tap INT_L (intra-composite)"),
        ("M2_LH12", "M2 LH end INT_R"),
        // Short-east wires
        ("M0_CLBLL_EE4BEG0", "M0 EE4"),
        ("M3_CLBLM_EE4BEG0", "M3 EE4"),
        ("M1_EE4BEG0", "M1 EE4 (INT_L)"),
        ("M0_CLBLL_EE2BEG0", "M0 EE2"),
        ("M3_CLBLM_EE2BEG0", "M3 EE2"),
        // Vertical wires
        ("M1_NN2BEG0", "M1 NN2"),
        ("M1_NN6BEG0", "M1 NN6"),
        ("M1_SS6BEG0", "M1 SS6"),
        ("M1_LV_L0", "M1 LV"),
        ("M1_LVB_L0", "M1 LVB"),
        // Diagonals
        ("M1_NE6BEG0", "M1 NE6"),
        ("M1_NW6BEG0", "M1 NW6"),
    ];

    for (name, note) in &target_names {
        let Some(&idx) = wire_by_name.get(*name) else {
            println!("\nwire not in CLB: {}  ({})", name, note);
            continue;
        };
        println!("\n=== {} ({}) ===", name, note);
        let mut by_offset_histogram: BTreeMap<(i32, i32), usize> = BTreeMap::new();
        for &tile in &samples {
            let (x, y) = db.tile_xy(tile);
            let Some(ns) = db.wire_node_shape(tile, idx) else {
                println!("  tile=({:3},{:3}) NO node shape", x, y);
                continue;
            };
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
            for o in &offs {
                *by_offset_histogram.entry(*o).or_insert(0) += 1;
            }
            println!("  tile=({:3},{:3}) offsets={:?}", x, y, offs);
        }
        // Also run across ALL CLB tiles to see the full shape histogram
        let mut full_hist: BTreeMap<(i32, i32), usize> = BTreeMap::new();
        for &tile in &clb_tiles {
            let Some(ns) = db.wire_node_shape(tile, idx) else {
                continue;
            };
            for tw in ns.tile_wires.get() {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                *full_hist.entry((dx as i32, dy as i32)).or_insert(0) += 1;
            }
        }
        println!(
            "  histogram across ALL {} CLB tiles: {:?}",
            clb_tiles.len(),
            full_hist
        );
    }

    // Simulate build_span_histograms in BOTH the old and new variants:
    //   OLD: single-representative tile per tile type + canonical-half filter.
    //   NEW: aggregate across all unique (tile_type, tile_shape) variants
    //        with sign-mirroring, then halve counts.
    println!("\n=== OLD build_span_histograms (single-representative + canonical filter) ===");
    let mut repr_per_tt: Vec<Option<i32>> = vec![None; db.num_tile_types()];
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type_index(tile) as usize;
        if repr_per_tt[tt].is_none() {
            repr_per_tt[tt] = Some(tile);
        }
    }
    for tt_idx in 0..db.num_tile_types() {
        let Some(tile) = repr_per_tt[tt_idx] else { continue };
        let name = db.tile_type_name(tile).to_string();
        let tt = db.tile_type_by_index(tt_idx as i32);
        let mut hist: BTreeMap<(i32, i32), usize> = BTreeMap::new();
        for wire_idx in 0..tt.wires.len() {
            let Some(ns) = db.wire_node_shape(tile, wire_idx) else { continue };
            for tw in ns.tile_wires.get() {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                let dx = dx as i32;
                let dy = dy as i32;
                if dx < 0 || (dx == 0 && dy <= 0) {
                    continue;
                }
                if dx == 0 && dy == 0 {
                    continue;
                }
                *hist.entry((dx, dy)).or_insert(0) += 1;
            }
        }
        // Print summary: total entries, total wire contributions,
        // X-direction vs Y-direction breakdown.
        let total_wires: usize = hist.values().sum();
        let x_entries: usize = hist.iter().filter(|((dx,_),_)| *dx != 0).map(|(_,v)| v).sum();
        let y_entries: usize = hist.iter().filter(|((dx,_),_)| *dx == 0).map(|(_,v)| v).sum();
        println!(
            "  tt_idx={} name={:15} repr_tile={} n_offsets={} wires={} X_wires={} Y_wires={}",
            tt_idx, name, tile, hist.len(), total_wires, x_entries, y_entries
        );
    }

    println!("\n=== NEW build_span_histograms (Model A: one entry per wire at max reach, GCLK/HCLK name filter) ===");
    let is_global = |n: &str| -> bool {
        n.contains("GCLK") || n.contains("HCLK") || n.contains("GND")
            || n.contains("VCC") || n.contains("CLK_HROW") || n.contains("CLK_BUFG")
    };
    // Model A: each physical wire contributes one histogram entry at the tap
    // with the max Manhattan composite distance from its home tile. Wires
    // internal to the composite (all taps at dx=dy=0) are absorbed. The chipdb
    // is already composite-compressed so tile_wire offsets are in composite-
    // grid units directly.
    let wire_reach = |tile: i32, wire_idx: usize| -> Option<(i32, i32)> {
        let wid = WireId::new(tile, wire_idx as i32);
        if is_global(db.wire_name(wid)) { return None; }
        let ns = db.wire_node_shape(tile, wire_idx)?;
        let mut best: Option<(i32, i32, i32)> = None; // (mag, dx, dy)
        for tw in ns.tile_wires.get() {
            let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
            let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
            let dx = dx as i32;
            let dy = dy as i32;
            let mag = dx.abs() + dy.abs();
            match best {
                None => best = Some((mag, dx, dy)),
                Some((m, _, _)) if mag > m => best = Some((mag, dx, dy)),
                _ => {}
            }
        }
        let (mag, dx, dy) = best?;
        if mag == 0 { return None; }
        let (kdx, kdy) = if dx > 0 || (dx == 0 && dy > 0) { (dx, dy) } else { (-dx, -dy) };
        Some((kdx, kdy))
    };
    // Enumerate unique (tt, shape) pairs and pick the richest shape per type.
    let mut seen: FxHashMap<(i32, i32), i32> = FxHashMap::default();
    for tile in 0..db.num_tiles() {
        let tt = db.tile_type_index(tile);
        if tt < 0 { continue; }
        let sh = db.tile_shape_index(tile);
        seen.entry((tt, sh)).or_insert(tile);
    }
    let mut best: Vec<Option<(i32, usize)>> = vec![None; db.num_tile_types()];
    for ((tt_idx, _sh), tile) in &seen {
        let tt = db.tile_type_by_index(*tt_idx);
        let mut distinct: std::collections::HashSet<(i32, i32)> = Default::default();
        for wire_idx in 0..tt.wires.len() {
            if let Some(k) = wire_reach(*tile, wire_idx) {
                distinct.insert(k);
            }
        }
        let slot = &mut best[*tt_idx as usize];
        match slot {
            Some((_, bn)) if *bn >= distinct.len() => {}
            _ => *slot = Some((*tile, distinct.len())),
        }
    }
    let mut new_hist: Vec<BTreeMap<(i32, i32), usize>> = vec![BTreeMap::new(); db.num_tile_types()];
    let mut absorbed_counts: Vec<usize> = vec![0; db.num_tile_types()];
    for (tt_idx, entry) in best.iter().enumerate() {
        let Some((tile, _)) = entry else { continue };
        let tt = db.tile_type_by_index(tt_idx as i32);
        for wire_idx in 0..tt.wires.len() {
            let wid = WireId::new(*tile, wire_idx as i32);
            if is_global(db.wire_name(wid)) { continue; }
            match wire_reach(*tile, wire_idx) {
                Some(k) => { *new_hist[tt_idx].entry(k).or_insert(0) += 1; }
                None => { absorbed_counts[tt_idx] += 1; }
            }
        }
    }
    for tt_idx in 0..db.num_tile_types() {
        let hist = &new_hist[tt_idx];
        if hist.is_empty() && absorbed_counts[tt_idx] == 0 { continue; }
        let Some(tile) = repr_per_tt[tt_idx] else { continue };
        let name = db.tile_type_name(tile).to_string();
        let total_wires: usize = hist.values().sum();
        let x_entries: usize = hist.iter().filter(|((dx,_),_)| *dx != 0).map(|(_,v)| v).sum();
        let y_entries: usize = hist.iter().filter(|((dx,_),_)| *dx == 0).map(|(_,v)| v).sum();
        let distinct_x: usize = hist.keys().filter(|(dx,_)| *dx != 0).count();
        let distinct_y: usize = hist.keys().filter(|(dx,_)| *dx == 0).count();
        let absorbed = absorbed_counts[tt_idx];
        let total_physical = total_wires + absorbed;
        println!(
            "  tt_idx={} name={:15} distinct_keys={} exposed_wires={} absorbed={} total={} X_wires={} Y_wires={} distinct_X={} distinct_Y={}",
            tt_idx, name, hist.len(), total_wires, absorbed, total_physical,
            x_entries, y_entries, distinct_x, distinct_y
        );
        // Show the first 12 X offsets explicitly
        let x_sample: Vec<_> = hist.iter().filter(|((dx,_),_)| *dx != 0).take(12).collect();
        println!("    X-sample: {:?}", x_sample);
        // And first 12 Y offsets
        let y_sample: Vec<_> = hist.iter().filter(|((dx,_),_)| *dx == 0).take(12).collect();
        println!("    Y-sample: {:?}", y_sample);
    }

    // Drill-down: for CLB tile type, find surviving wires whose max |dy|
    // exceeds 18 (LV/LVB limit) or |dx| exceeds 3 (LH limit in composite).
    // With the GCLK/HCLK name filter these should be empty; any non-empty
    // output is a bug.
    println!("\n=== CLB pathology drill-down: non-global wires by max |dy| (should be empty) ===");
    if let Some((tile, _)) = best[clb_tt as usize] {
        let tt = db.tile_type_by_index(clb_tt);
        let mut wire_max_dy: Vec<(usize, i32, i32, String)> = Vec::new();
        for wire_idx in 0..tt.wires.len() {
            let wid = WireId::new(tile, wire_idx as i32);
            if is_global(db.wire_name(wid)) { continue; }
            let Some(ns) = db.wire_node_shape(tile, wire_idx) else { continue };
            let mut max_abs_dy = 0i32;
            let mut max_abs_dx = 0i32;
            for tw in ns.tile_wires.get() {
                let dx: i16 = unsafe { nextpnr::read_packed!(*tw, dx) };
                let dy: i16 = unsafe { nextpnr::read_packed!(*tw, dy) };
                max_abs_dx = max_abs_dx.max((dx as i32).abs());
                max_abs_dy = max_abs_dy.max((dy as i32).abs());
            }
            if max_abs_dy > 18 || max_abs_dx > 3 {
                let wname = db.wire_name(WireId::new(tile, wire_idx as i32)).to_string();
                wire_max_dy.push((wire_idx, max_abs_dx, max_abs_dy, wname));
            }
        }
        wire_max_dy.sort_by(|a, b| b.2.cmp(&a.2));
        println!("  non-global wires with |dy| > 18 or |dx| > 3: {}", wire_max_dy.len());
        for (wi, mdx, mdy, wn) in wire_max_dy.iter().take(30) {
            println!("    wire={:4} max|dx|={:3} max|dy|={:3}  {}", wi, mdx, mdy, wn);
        }
    }

    // Targeted dump: under Model A, which CLB wires' max-reach delta lands at
    // dy=1..18, grouped by dy? This shows the physical span class distribution
    // (expecting peaks at 1, 2, 4, 6, 12, 18 after the fix).
    println!("\n=== CLB max-reach histogram: wires grouped by final reach key ===");
    if let Some((tile, _)) = best[clb_tt as usize] {
        let tt = db.tile_type_by_index(clb_tt);
        let mut by_key: BTreeMap<(i32, i32), Vec<String>> = BTreeMap::new();
        for wire_idx in 0..tt.wires.len() {
            let Some(k) = wire_reach(tile, wire_idx) else { continue };
            let wname = db.wire_name(WireId::new(tile, wire_idx as i32)).to_string();
            by_key.entry(k).or_default().push(wname);
        }
        for (k, names) in &by_key {
            println!("  {:?}: {} wires", k, names.len());
            for n in names.iter().take(4) {
                println!("    {}", n);
            }
        }
    }
}

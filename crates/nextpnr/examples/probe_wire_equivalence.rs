//! Measure how far structural (bisimulation) refinement compresses the
//! per-tile-type wire class count. For each tile type:
//!   1. Start from the coarsest partition: every wire in class 0.
//!   2. Iteratively refine via Hopcroft: split class K if its members
//!      disagree on the signature
//!        - multiset of outgoing-pip (cost, dst_class) edges,
//!        - multiset of incoming-pip (cost, src_class) edges,
//!        - node-shape offset list (multiset of (dx, dy)).
//!      Signatures are computed w.r.t. the CURRENT class map, so refinement
//!      converges to the coarsest bisimulation relation — the strictly
//!      correct topology-only grouping. No name-prefix seed; names inject
//!      false distinctions that monotone refinement can never undo.
//!   3. Report `n_wires -> n_struct_classes` and the class-size histogram.
//!
//! If structural class count stays small (≤ ~50 per tile type), a
//! class-pair lookahead table (`transit[tt][side_in][side_out][c_in][c_out]`)
//! becomes tractable storage-wise.
//!
//! Usage: probe_wire_equivalence [chipdb-path]

use nextpnr::chipdb::{ChipDb, PipId};
use nextpnr::context::Context;
use nextpnr::read_packed;
use nextpnr::router::astar::default_pip_cost;
use std::collections::HashMap;
use std::path::Path;

fn histogram(class: &[u32]) -> Vec<(u32, u32)> {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &c in class {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut by_size: HashMap<u32, u32> = HashMap::new();
    for (_, n) in counts {
        *by_size.entry(n).or_insert(0) += 1;
    }
    let mut v: Vec<(u32, u32)> = by_size.into_iter().collect();
    v.sort_unstable();
    v
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

    // Pick a representative (interior) tile per tile type.
    let is_interior = |tile: i32| -> bool {
        let (x, y) = db.tile_xy(tile);
        x > 0 && x < w - 1 && y > 0 && y < h - 1
    };
    let mut rep: Vec<Option<i32>> = vec![None; num_tt];
    for tile in 0..(w * h) {
        let tt = db.tile_type_index(tile) as usize;
        match rep[tt] {
            None => rep[tt] = Some(tile),
            Some(curr) if !is_interior(curr) && is_interior(tile) => {
                rep[tt] = Some(tile);
            }
            _ => {}
        }
    }

    println!(
        "{:<4}  {:<28}  {:>7}  {:>6}  {:>5}  {}",
        "tt", "type_name", "n_wires", "n_strc", "iters", "class_size_histogram"
    );

    let mut total_wires = 0usize;
    let mut total_struct_classes = 0usize;
    let mut max_classes_per_tt = 0usize;

    for tt_idx in 0..num_tt {
        let Some(t) = rep[tt_idx] else { continue };
        let tt = db.tile_type_by_index(tt_idx as i32);
        let wires_pod = tt.wires.get();
        let pips_pod = tt.pips.get();
        let n_wires = wires_pod.len();
        if n_wires == 0 {
            continue;
        }
        let tt_name = db.tile_type_name(t).to_string();

        // Coarsest initial partition: all wires in one class. Bisimulation
        // refinement converges to the correct equivalence regardless of
        // seed, but only if the seed doesn't inject distinctions that
        // refinement (which can only split, never merge) cannot undo.
        let mut class: Vec<u32> = vec![0u32; n_wires];

        // Outgoing pip edges: per-src list of (cost, dst_wi).
        let mut out_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_wires];
        for (i, pip) in pips_pod.iter().enumerate() {
            let src: i32 = unsafe { read_packed!(*pip, src_wire) };
            let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if src < 0 || dst < 0 {
                continue;
            }
            let s = src as usize;
            let d = dst as usize;
            if s >= n_wires || d >= n_wires {
                continue;
            }
            let pid = PipId::new(t, i as i32);
            let c = default_pip_cost(&ctx, pid) as u32;
            out_edges[s].push((c, d as u32));
        }

        // Incoming pip edges: per-dst list of (cost, src_wi). Structural
        // equivalence needs both directions — a wire that looks identical
        // outbound but has a unique inbound source is NOT equivalent.
        let mut in_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_wires];
        for (i, pip) in pips_pod.iter().enumerate() {
            let src: i32 = unsafe { read_packed!(*pip, src_wire) };
            let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if src < 0 || dst < 0 {
                continue;
            }
            let s = src as usize;
            let d = dst as usize;
            if s >= n_wires || d >= n_wires {
                continue;
            }
            let pid = PipId::new(t, i as i32);
            let c = default_pip_cost(&ctx, pid) as u32;
            in_edges[d].push((c, s as u32));
        }

        // Node-shape signature per wire: sorted (dx, dy) list. Ignores peer
        // wire_idx because that's what class refinement would otherwise need
        // to fold in — and for the initial probe we just want to see if
        // name-prefix grouping is structurally consistent to first order.
        let mut node_sig: Vec<Vec<(i16, i16)>> = vec![Vec::new(); n_wires];
        for wi in 0..n_wires {
            if let Some(shape) = db.wire_node_shape(t, wi) {
                for tw in shape.tile_wires.get() {
                    let dx: i16 = unsafe { read_packed!(*tw, dx) };
                    let dy: i16 = unsafe { read_packed!(*tw, dy) };
                    node_sig[wi].push((dx, dy));
                }
                node_sig[wi].sort_unstable();
            }
        }

        // Hopcroft-style refinement: signature = (old_class, out-sig, in-sig, node-sig).
        // Out/in signatures translate peer wire index through the CURRENT
        // class map, so equivalence tightens monotonically.
        let mut iters = 0u32;
        loop {
            iters += 1;
            let mut sig_to_class: HashMap<
                (u32, Vec<(u32, u32)>, Vec<(u32, u32)>, Vec<(i16, i16)>),
                u32,
            > = HashMap::new();
            let mut new_class = vec![0u32; n_wires];
            for wi in 0..n_wires {
                let mut out_sig: Vec<(u32, u32)> = out_edges[wi]
                    .iter()
                    .map(|&(c, d)| (c, class[d as usize]))
                    .collect();
                out_sig.sort_unstable();
                let mut in_sig: Vec<(u32, u32)> = in_edges[wi]
                    .iter()
                    .map(|&(c, s)| (c, class[s as usize]))
                    .collect();
                in_sig.sort_unstable();
                let key = (class[wi], out_sig, in_sig, node_sig[wi].clone());
                let next = sig_to_class.len() as u32;
                new_class[wi] = *sig_to_class.entry(key).or_insert(next);
            }
            if new_class == class {
                break;
            }
            class = new_class;
            if iters > 100 {
                eprintln!(
                    "  !! tt={} failed to converge in 100 iters (still refining)",
                    tt_idx
                );
                break;
            }
        }

        let mut uniq = class.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let n_struct = uniq.len();

        let hist = histogram(&class);
        let hist_str: String = hist
            .iter()
            .map(|(sz, cnt)| format!("{}x{}", cnt, sz))
            .collect::<Vec<_>>()
            .join(",");

        println!(
            "{:<4}  {:<28}  {:>7}  {:>5}  {:>5}  {}",
            tt_idx, tt_name, n_wires, n_struct, iters, hist_str,
        );

        total_wires += n_wires;
        total_struct_classes += n_struct;
        if n_struct > max_classes_per_tt {
            max_classes_per_tt = n_struct;
        }
    }

    println!();
    println!(
        "TOTAL: wires={}  struct_classes={}",
        total_wires, total_struct_classes,
    );
    if total_struct_classes > 0 {
        println!(
            "Reduction vs. raw wires: {:.2}x",
            total_wires as f64 / total_struct_classes as f64,
        );
    }
    println!("Max struct classes in any tile type: {}", max_classes_per_tt);

    // Rough class-pair storage estimate at 4 bytes/entry, dense per
    // (tt, side_in, side_out) using each type's own class count.
    // Sum over types of (classes_in_type)^2 × 16 = 4 sides × 4 sides.
    // Without per-type counts available here in aggregate form, print the
    // upper bound using max_classes_per_tt.
    let upper = (max_classes_per_tt * max_classes_per_tt * 16 * 4) as f64 / (1024.0 * 1024.0);
    println!(
        "Dense class-pair storage upper bound: {:.1} MB per tile type (16 side-pairs × {}×{} × 4 B)",
        upper, max_classes_per_tt, max_classes_per_tt,
    );
}

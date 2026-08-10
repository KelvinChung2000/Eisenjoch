//! Dump (cost, span, cost/span) histogram for LUT6-reachable pips, to
//! understand why per-class min rate is 121/18 ≈ 6.72/tile and whether
//! cheaper-per-tile pips exist (and if so, why they aren't reachable).

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::read_packed;
use nextpnr::router::astar::default_pip_cost;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::path::Path;

fn main() {
    let p = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let db = ChipDb::load(Path::new(p)).expect("load");
    let ctx = Context::new(db);
    let cdb = ctx.chipdb();
    let w = cdb.width();
    let h = cdb.height();
    let num_tt = cdb.num_tile_types();

    const REP_MARGIN: i32 = 13;
    let is_interior = |t: i32| {
        let (x, y) = cdb.tile_xy(t);
        x >= REP_MARGIN && x < w - REP_MARGIN && y >= REP_MARGIN && y < h - REP_MARGIN
    };
    let mut rep_of_tt: Vec<Option<i32>> = vec![None; num_tt];
    for t in 0..(w * h) {
        let tt = cdb.tile_type_index(t) as usize;
        match rep_of_tt[tt] {
            None => rep_of_tt[tt] = Some(t),
            Some(c) if !is_interior(c) && is_interior(t) => rep_of_tt[tt] = Some(t),
            _ => {}
        }
    }

    // Forward adjacency per tt: src_wi -> [dst_wi]
    let mut adj: Vec<Vec<Vec<i32>>> = Vec::with_capacity(num_tt);
    let mut pip_data: Vec<Vec<(i64, i64, i32)>> = Vec::with_capacity(num_tt); // (cost, span, src)
    for tt_idx in 0..num_tt {
        let Some(rep) = rep_of_tt[tt_idx] else {
            adj.push(Vec::new());
            pip_data.push(Vec::new());
            continue;
        };
        let (rx, ry) = cdb.tile_xy(rep);
        let tt = cdb.tile_type_by_index(tt_idx as i32);
        let wires = tt.wires.get();
        let pips = tt.pips.get();
        let mut a: Vec<Vec<i32>> = vec![Vec::new(); wires.len()];
        let mut data: Vec<(i64, i64, i32)> = Vec::with_capacity(pips.len());
        for (pip_idx, pip) in pips.iter().enumerate() {
            let s: i32 = unsafe { read_packed!(*pip, src_wire) };
            let d: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if s < 0 || d < 0 {
                data.push((0, 1, -1));
                continue;
            }
            a[s as usize].push(d);
            let pid = PipId::new(rep, pip_idx as i32);
            let cost = default_pip_cost(&ctx, pid).max(1) as i64;
            let dst_wire = WireId::new(rep, d);
            let mut span: i64 = 0;
            cdb.node_wires_cb(dst_wire, |peer| {
                let (px, py) = cdb.tile_xy(peer.tile());
                let dist = ((px - rx).abs() + (py - ry).abs()) as i64;
                if dist > span {
                    span = dist;
                }
            });
            data.push((cost, span.max(1), s));
        }
        adj.push(a);
        pip_data.push(data);
    }

    // Node-share peer map across rep tiles: (tt, wi) -> [(tt, wi)]
    let mut node_share: FxHashMap<(usize, i32), Vec<(usize, i32)>> = FxHashMap::default();
    for tt_idx in 0..num_tt {
        let Some(rep) = rep_of_tt[tt_idx] else { continue };
        let tt = cdb.tile_type_by_index(tt_idx as i32);
        for wi in 0..tt.wires.get().len() {
            let wid = WireId::new(rep, wi as i32);
            let key = (tt_idx, wi as i32);
            cdb.node_wires_cb(wid, |peer| {
                let p_tt = cdb.tile_type_index(peer.tile()) as usize;
                let p_wi = peer.index();
                if (p_tt, p_wi) == key {
                    return;
                }
                if rep_of_tt[p_tt] != Some(peer.tile()) {
                    return;
                }
                node_share.entry(key).or_default().push((p_tt, p_wi));
            });
        }
    }

    // BFS from LUT6 BEL outputs.
    let target_bel = "LUT6";
    let mut reach: Vec<Vec<bool>> = (0..num_tt)
        .map(|tt| match rep_of_tt[tt] {
            Some(_) => vec![false; cdb.tile_type_by_index(tt as i32).wires.get().len()],
            None => Vec::new(),
        })
        .collect();
    let mut q: VecDeque<(usize, i32)> = VecDeque::new();
    for tt_idx in 0..num_tt {
        let Some(rep) = rep_of_tt[tt_idx] else { continue };
        let tt = cdb.tile_type_by_index(tt_idx as i32);
        for bel in tt.bels.get().iter() {
            let bt: i32 = unsafe { read_packed!(*bel, bel_type) };
            let Some(name) = cdb.constid_str(bt) else { continue };
            if name != target_bel { continue; }
            for pin in bel.pins.get().iter() {
                let dir: i32 = unsafe { read_packed!(*pin, dir) };
                if dir != 1 { continue; }
                let wi: i32 = unsafe { read_packed!(*pin, wire) };
                if wi < 0 { continue; }
                if !reach[tt_idx][wi as usize] {
                    reach[tt_idx][wi as usize] = true;
                    q.push_back((tt_idx, wi));
                }
            }
            // Only need one rep per tt
            let _ = rep;
        }
    }
    while let Some((tt, wi)) = q.pop_front() {
        for &dst in &adj[tt][wi as usize] {
            if !reach[tt][dst as usize] {
                reach[tt][dst as usize] = true;
                q.push_back((tt, dst));
            }
        }
        if let Some(peers) = node_share.get(&(tt, wi)) {
            for &(ptt, pwi) in peers {
                let pu = pwi as usize;
                if pu < reach[ptt].len() && !reach[ptt][pu] {
                    reach[ptt][pu] = true;
                    q.push_back((ptt, pwi));
                }
            }
        }
    }

    // Histogram of (cost, span, rate) for LUT6-reachable pips.
    let mut by_span: FxHashMap<i64, Vec<(i64, f64)>> = FxHashMap::default();
    let mut all: Vec<(f64, i64, i64, usize, &'static str)> = Vec::new();
    let mut min_overall: Option<(f64, i64, i64)> = None;
    for tt in 0..num_tt {
        for &(cost, span, src) in &pip_data[tt] {
            if src < 0 { continue; }
            let su = src as usize;
            if su >= reach[tt].len() || !reach[tt][su] { continue; }
            let rate = cost as f64 / span as f64;
            by_span.entry(span).or_default().push((cost, rate));
            all.push((rate, cost, span, tt, "LUT6"));
            min_overall = Some(match min_overall {
                None => (rate, cost, span),
                Some((br, _, _)) if rate < br => (rate, cost, span),
                Some(x) => x,
            });
        }
    }

    println!("=== LUT6-reachable pips: span -> (count, min_cost, min_rate, max_span_within_bucket) ===");
    let mut spans: Vec<i64> = by_span.keys().copied().collect();
    spans.sort();
    for s in spans {
        let v = &by_span[&s];
        let mn_c = v.iter().map(|x| x.0).min().unwrap();
        let mn_r = v.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        println!("  span={s:>3}: n={:<6} min_cost={mn_c:>4} min_rate={mn_r:.3}", v.len());
    }
    if let Some((r, c, s)) = min_overall {
        println!("\nGLOBAL MIN: rate={r:.3} cost={c} span={s}");
    }

    // Detailed: for span=18 and span=12, dump src/dst wire names and costs.
    println!("\n=== span=18 LUT6-reachable pips (full detail) ===");
    for tt in 0..num_tt {
        let Some(rep) = rep_of_tt[tt] else { continue };
        let pips = cdb.tile_type_by_index(tt as i32).pips.get();
        let wires = cdb.tile_type_by_index(tt as i32).wires.get();
        for (pidx, pip) in pips.iter().enumerate() {
            let s: i32 = unsafe { read_packed!(*pip, src_wire) };
            let d: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if s < 0 || d < 0 { continue; }
            let su = s as usize;
            if su >= reach[tt].len() || !reach[tt][su] { continue; }
            let (cost, span, _) = pip_data[tt][pidx];
            if span != 18 && span != 12 && span != 13 { continue; }
            let nm = |wi: i32| -> String {
                let wu = wi as usize;
                if wu >= wires.len() { return "?".into(); }
                let nid: i32 = unsafe { read_packed!(wires[wu], name) };
                cdb.constid_str(nid).map(|x| x.to_string()).unwrap_or_default()
            };
            println!(
                "  span={span:>2} cost={cost:>3} tt={:<10} {:<28} -> {}",
                cdb.tile_type_name(rep),
                nm(s),
                nm(d),
            );
        }
    }

    // Now: are there ANY pips in the chipdb (regardless of LUT6 reachability)
    // with cheaper rate than the LUT6 min? If yes, what gates them?
    println!("\n=== ALL pips in chipdb with rate < LUT6 min ===");
    let lut6_min = min_overall.map(|x| x.0).unwrap_or(f64::INFINITY);
    let mut cheaper: Vec<(f64, i64, i64, usize, i32)> = Vec::new();
    for tt in 0..num_tt {
        for (idx, &(cost, span, src)) in pip_data[tt].iter().enumerate() {
            if src < 0 { continue; }
            let rate = cost as f64 / span as f64;
            if rate < lut6_min {
                cheaper.push((rate, cost, span, tt, idx as i32));
            }
        }
    }
    cheaper.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("count = {}", cheaper.len());
    for (rate, cost, span, tt, _) in cheaper.iter().take(20) {
        let tt_name = match rep_of_tt[*tt] {
            Some(r) => cdb.tile_type_name(r).to_string(),
            None => "?".into(),
        };
        println!("  rate={rate:.3} cost={cost} span={span} tt={tt_name}");
    }
}

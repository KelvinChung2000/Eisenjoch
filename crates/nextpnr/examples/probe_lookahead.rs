//! Characterize the compositional lookahead: for a CLK_RELAY-like source
//! wire, print `estimate_delay` at a range of offsets. If composition is
//! firing, the value should scale ~linearly with offset; if we're hitting
//! the Manhattan-residual fallback (exit_class missing), it will jump
//! beyond the chain-walk's reach.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::router::lookahead::Lookahead;
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;
use std::path::Path;

struct DijkstraModel;

impl PathCostModel for DijkstraModel {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, _ctx: &Context, _wire: WireId, _dst: WireId) -> DelayT {
        0
    }
}

fn find_wire_by_name(chipdb: &ChipDb, tile: i32, name_str: &str) -> Option<WireId> {
    let tt = chipdb.tile_type(tile);
    let wires = tt.wires.get();
    for (idx, w) in wires.iter().enumerate() {
        let nid: i32 = unsafe { nextpnr::read_packed!(*w, name) };
        if chipdb.constid_str(nid).unwrap_or("") == name_str {
            return Some(WireId::new(tile, idx as i32));
        }
    }
    None
}

fn tile_at(chipdb: &ChipDb, x: i32, y: i32) -> i32 {
    y * chipdb.width() + x
}

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    let chipdb = ctx.chipdb();
    println!("Chipdb loaded: {}x{}", chipdb.width(), chipdb.height());

    // Global span enumeration.
    //
    // Each wire belongs to one node; the node's "span" is the max |Δ| from
    // any member to any other. For lookahead purposes, the relevant span of
    // a class is the max reach of the nodes accessible via one pip — but
    // as a first pass just look at per-wire node span (farthest peer from
    // the wire's own tile). If the chipdb represents long wires as single
    // nodes with all 16 tile-members, this histogram should have clean
    // peaks at the architecture's wire lengths (1, 2, 4, 6, 12, 16…)
    // rather than the continuous distribution we saw with per-peer-pair
    // counting.
    println!("\n=== per-wire max node-span ===");
    let t0 = std::time::Instant::now();
    let mut span_hist: BTreeMap<i32, usize> = BTreeMap::new();
    let mut wires_scanned: u64 = 0;
    for wire in chipdb.wires() {
        wires_scanned += 1;
        let (wx, wy) = chipdb.tile_xy(wire.tile());
        let mut max_span = 0;
        chipdb.node_wires_cb(wire, |peer| {
            let (px, py) = chipdb.tile_xy(peer.tile());
            let dx = (px - wx).abs();
            let dy = (py - wy).abs();
            let s = dx + dy;
            if s > max_span {
                max_span = s;
            }
        });
        *span_hist.entry(max_span).or_insert(0) += 1;
    }
    let elapsed = t0.elapsed();
    println!(
        "  scanned {} wires in {:?}",
        wires_scanned, elapsed,
    );
    println!("  max-node-span histogram:");
    for (span, count) in &span_hist {
        let pct = *count as f64 * 100.0 / wires_scanned as f64;
        println!("    span={span:>4}  wires={count:>12}  ({pct:.2}%)");
    }

    println!("Building lookahead...");
    let t0 = std::time::Instant::now();
    let lookahead = Lookahead::build(&ctx);
    println!("Lookahead built in {:?}", t0.elapsed());

    // Find a CLK_RELAY_OUT wire on a CLB near the chip center, and on an
    // IOB at x=0, to probe both class paths.
    let cx = chipdb.width() / 2;
    let cy = chipdb.height() / 2;

    let clb_tile = tile_at(chipdb, cx, cy);
    let io_tile = tile_at(chipdb, 0, cy);
    println!(
        "\nCenter tile ({},{})=idx {}  type={}",
        cx,
        cy,
        clb_tile,
        chipdb.tile_type_index(clb_tile),
    );
    println!(
        "IO tile    (0,{})=idx {}  type={}",
        cy,
        io_tile,
        chipdb.tile_type_index(io_tile),
    );

    // Try to find CLK_RELAY_OUT_0_E on both — this is the mesh wire the
    // clock distribution uses.
    let probe_names = [
        "CLK_RELAY_OUT_0_E",
        "CLK_RELAY_IN_0_E",
        "M1_GCLK_L_B0",
        "M2_GCLK_B0",
    ];

    for name in probe_names {
        println!("\n=== src='{name}' ===");
        for (label, tile) in [("CLB", clb_tile), ("IOB", io_tile)] {
            let wire = match find_wire_by_name(chipdb, tile, name) {
                Some(w) => w,
                None => {
                    println!("  [{label}] no wire '{name}' on this tile");
                    continue;
                }
            };
            let (sx, sy) = chipdb.tile_xy(wire.tile());
            println!("  [{label}] src @ ({sx},{sy}) idx={}", wire.index());
            // Probe increasing +x offsets.
            for dx in [0, 10, 20, 40, 60, 80, 120, 160, 240, 310] {
                let target_x = (sx + dx).min(chipdb.width() - 1);
                let target_tile = tile_at(chipdb, target_x, sy);
                // Destination wire: use the same-name wire on target tile
                // if present, else wire 0 as a stand-in.
                let dst = find_wire_by_name(chipdb, target_tile, name)
                    .unwrap_or(WireId::new(target_tile, 0));
                let est = lookahead.estimate_delay(chipdb, wire, dst, nextpnr::router::lookahead::UNKNOWN_CLASS);
                let manh = dx;
                let ratio = if manh > 0 {
                    est as f64 / manh as f64
                } else {
                    0.0
                };
                println!(
                    "    dx={dx:>3}  est={est:>6}   (est/manh = {ratio:.2})"
                );
            }
        }
    }

    // Now the exact failing pair: IO0_O at (0,98) → BUFG0_I at (310,111).
    println!("\n=== tm3_clk_v2 endpoints ===");
    let src_tile = tile_at(chipdb, 0, 98);
    let dst_tile = tile_at(chipdb, 310, 111);
    println!(
        "  src tile type={}  dst tile type={}",
        chipdb.tile_type_index(src_tile),
        chipdb.tile_type_index(dst_tile),
    );
    let src = find_wire_by_name(chipdb, src_tile, "IO0_O");
    let dst = find_wire_by_name(chipdb, dst_tile, "BUFG0_I");
    println!("  IO0_O found: {src:?}");
    println!("  BUFG0_I found: {dst:?}");
    if let (Some(s), Some(d)) = (src, dst) {
        let est = lookahead.estimate_delay(chipdb, s, d, nextpnr::router::lookahead::UNKNOWN_CLASS);
        println!("  estimate_delay = {est}  (manhattan=323)");
    }

    // Also probe from a CLK_RELAY_IN wire on the source IOB, since that's
    // what A* will pop first after expanding IO0_O's pips_downhill.
    if let Some(relay_src) = find_wire_by_name(chipdb, src_tile, "CLK_RELAY_IN_0_E") {
        if let Some(d) = dst {
            let est = lookahead.estimate_delay(chipdb, relay_src, d, nextpnr::router::lookahead::UNKNOWN_CLASS);
            println!("  from CLK_RELAY_IN_0_E@(0,98) → BUFG0_I: est={est}");
        }
    }

    // Direct Dijkstra from CLK_RELAY_OUT_0_E @ center (same seed the
    // lookahead builder uses for that class). Dump how far it actually
    // reached in 50k pops, grouped by offset, so we can see whether the
    // mesh got any traversal at all.
    println!("\n=== direct Dijkstra from CLK_RELAY_OUT_0_E @ center ===");
    let center_tile = tile_at(chipdb, cx, cy);
    if let Some(src) = find_wire_by_name(chipdb, center_tile, "CLK_RELAY_OUT_0_E") {
        let (rep_x, rep_y) = chipdb.tile_xy(src.tile());
        let mut src_set: FxHashSet<WireId> = FxHashSet::default();
        src_set.insert(src);
        let opts = AStarOptions {
            visit_limit: Some(50_000),
            exhaustive: true,
            retain_trace: true,
            stop_on_first_touch: false,
        };
        let result = astar_search(&ctx, &DijkstraModel, &src_set, src, &opts);
        println!(
            "  exit={:?}  visit_count={}  visited_len={}",
            result.trace.exit,
            result.trace.visit_count,
            result.trace.visited.len(),
        );

        // Bucket visited wires by tile-offset from rep tile. Keep min
        // cost per offset for ANY wire, plus the count of wires-per-tile
        // specifically on the CLK_RELAY_OUT_0_E class path.
        let mut offset_min: BTreeMap<(i32, i32), DelayT> = BTreeMap::new();
        let mut offset_count: BTreeMap<(i32, i32), usize> = BTreeMap::new();
        for (&wire, &(cost, _pen, _pip, _from)) in &result.trace.visited {
            let (wx, wy) = chipdb.tile_xy(wire.tile());
            let dx = wx - rep_x;
            let dy = wy - rep_y;
            *offset_count.entry((dx, dy)).or_insert(0) += 1;
            offset_min
                .entry((dx, dy))
                .and_modify(|v| {
                    if cost < *v {
                        *v = cost;
                    }
                })
                .or_insert(cost);
        }

        // Print min cost at horizontal offsets along dy=0.
        println!("  offset cost along dy=0:");
        for dx in [-40, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 40] {
            match offset_min.get(&(dx, 0)) {
                Some(c) => {
                    let n = offset_count.get(&(dx, 0)).copied().unwrap_or(0);
                    println!("    dx={dx:>4} dy=0  min_cost={c:>6}  n_wires={n}");
                }
                None => println!("    dx={dx:>4} dy=0  (not reached)"),
            }
        }
        println!("  total unique offsets visited: {}", offset_min.len());
    } else {
        println!("  CLK_RELAY_OUT_0_E not found on center tile!");
    }
}

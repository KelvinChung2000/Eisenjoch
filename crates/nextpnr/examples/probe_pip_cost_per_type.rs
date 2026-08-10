//! Measure per-tile-type pip_delay distribution. We want to know whether
//! clock-tile PIPs actually have higher delay than CLB-tile PIPs, since
//! that's a load-bearing assumption behind the per-type per_pip_cost
//! theory.

use nextpnr::chipdb::{ChipDb, PipId};
use nextpnr::context::Context;
use nextpnr::router::astar::default_pip_cost;
use std::path::Path;

fn main() {
    let chipdb_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into());
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let ctx = Context::new(db);
    let db = ctx.chipdb();
    let w = db.width();
    let h = db.height();
    let num_tt = db.num_tile_types();

    // First rep tile per tile type.
    let mut rep: Vec<Option<i32>> = vec![None; num_tt];
    for tile in 0..(w * h) {
        let tt = db.tile_type_index(tile) as usize;
        if rep[tt].is_none() {
            rep[tt] = Some(tile);
        }
    }

    println!(
        "{:>4}  {:>8}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {}",
        "tt", "n_pips", "min", "p25", "p50", "p75", "max", "example_tile"
    );
    let mut by_tt: Vec<(usize, usize, u64, i32, i32, i32, i32, i32, i32)> = Vec::new();
    for tt in 0..num_tt {
        let Some(t) = rep[tt] else { continue };
        let n_pips = db.tile_type_by_index(tt as i32).pips.get().len();
        if n_pips == 0 {
            continue;
        }
        let mut delays: Vec<i32> = Vec::with_capacity(n_pips);
        for i in 0..n_pips {
            let pip = PipId::new(t, i as i32);
            delays.push(default_pip_cost(&ctx, pip) as i32);
        }
        delays.sort_unstable();
        let pct = |p: f64| -> i32 {
            let idx = ((delays.len() - 1) as f64 * p).round() as usize;
            delays[idx]
        };
        let min = delays[0];
        let p25 = pct(0.25);
        let p50 = pct(0.50);
        let p75 = pct(0.75);
        let max = *delays.last().unwrap();
        let (tx, ty) = db.tile_xy(t);
        by_tt.push((tt, n_pips, p50 as u64, min, p25, p50, p75, max, t));
        println!(
            "{:>4}  {:>8}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  tile=({},{})",
            tt, n_pips, min, p25, p50, p75, max, tx, ty
        );
    }
    println!();

    // Highlight clock-adjacent tile types by name. Grep the pip_src wire
    // names of each type's first few pips to guess whether it's a clock tt.
    println!("Identifying candidate clock tile types by PIP source name prefix:");
    for tt in 0..num_tt {
        let Some(t) = rep[tt] else { continue };
        let n_pips = db.tile_type_by_index(tt as i32).pips.get().len();
        if n_pips == 0 {
            continue;
        }
        let mut has_clk = 0usize;
        let sample = n_pips.min(200);
        for i in 0..sample {
            let pip = PipId::new(t, i as i32);
            let src = db.pip_src_wire(pip);
            let info = db.wire_info(src);
            let name: i32 = unsafe { nextpnr::read_packed!(*info, name) };
            if let Some(s) = db.constid_str(name) {
                let sl = s.to_ascii_uppercase();
                if sl.contains("CLK") || sl.contains("GCLK") || sl.contains("BUFG") || sl.contains("HCLK") {
                    has_clk += 1;
                }
            }
        }
        if has_clk > 0 {
            let (tx, ty) = db.tile_xy(t);
            println!(
                "  tt={} tile=({},{}): {}/{} sampled pips have clock-ish src names",
                tt, tx, ty, has_clk, sample
            );
        }
    }
}

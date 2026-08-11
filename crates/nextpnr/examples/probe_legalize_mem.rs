//! Run the legalizer alone, on synthetic positions, to find where it allocates.
//!
//! The FPGA01 run that OOM-killed at 32GB spent 35 minutes in placement before
//! reaching legalization, which makes the legalizer's memory profile expensive
//! to observe. Legalization memory does not depend on placement quality — it
//! depends on cell count, BEL count and the candidate structures — so a
//! deterministic synthetic spread reaches the same allocation in seconds.
//!
//! Usage:
//!   NPNR_LEG_DIAG=1 probe_legalize_mem [chipdb] [design]

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use std::path::Path;
use std::time::Instant;

fn rss_mb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(kb) = rest.trim().split_whitespace().next() {
                return kb.parse::<f64>().unwrap_or(0.0) / 1024.0;
            }
        }
    }
    0.0
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let grid_w = ctx.chipdb().width() as f64;
    let grid_h = ctx.chipdb().height() as f64;

    // Movable cells, in the same sense opt_trans uses: alive and not locked.
    let mut idx_to_cell = Vec::new();
    for (cid, cell) in ctx.design.iter_alive_cells() {
        if !cell.bel_strength.is_locked() {
            idx_to_cell.push(cid);
        }
    }

    // Deterministic spread over the die. A xorshift on the index, not a real
    // placement: only the candidate-structure sizes matter here.
    let mut cell_x = Vec::with_capacity(idx_to_cell.len());
    let mut cell_y = Vec::with_capacity(idx_to_cell.len());
    for i in 0..idx_to_cell.len() {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
        cell_x.push(((h % 1_000_000) as f64 / 1_000_000.0) * (grid_w - 1.0));
        cell_y.push((((h >> 20) % 1_000_000) as f64 / 1_000_000.0) * (grid_h - 1.0));
    }

    println!(
        "packed: {} cells ({} movable), {} nets, grid {}x{}, rss={:.0}MB",
        ctx.design.num_cells(),
        idx_to_cell.len(),
        ctx.design.num_nets(),
        grid_w,
        grid_h,
        rss_mb()
    );

    let strategy = std::env::var("LEG_STRATEGY").unwrap_or_else(|_| "sorted".into());
    let t = Instant::now();
    let disp =
        nextpnr::placer::legalize::legalize(&mut ctx, &idx_to_cell, &cell_x, &cell_y, &strategy)
            .expect("legalize");
    println!(
        "LEGALIZE strategy={strategy} secs={:.1} displacement={disp:.0} rss_after={:.0}MB",
        t.elapsed().as_secs_f64(),
        rss_mb()
    );
}

use nextpnr::chipdb::{BelId, ChipDb};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use nextpnr::placer::Placer;
use std::env;
use std::path::Path;

/// If `NPNR_IO_EDGE` is set to one of left|right|bottom|top, after packing
/// unbind every locked IOB cell and rebind it to an IOB BEL on that edge.
/// Targets are assigned in sorted order (by the major coordinate orthogonal
/// to the chosen edge) so packing is deterministic. Pure diagnostic tool.
fn relocate_ios_to_edge(ctx: &mut Context, edge: &str) {
    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();

    // Find the IOB bucket id by scanning any cell with IOB type.
    let iob_bucket = {
        let mut found = None;
        for (_, cell) in ctx.design.iter_alive_cells() {
            let name = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
            if name == "IOB" {
                found = Some(ctx.resolve_bucket(cell.cell_type));
                break;
            }
        }
        match found {
            Some(b) => b,
            None => {
                eprintln!("no IOB cells found; skipping relocation");
                return;
            }
        }
    };

    // Enumerate all IOB BELs on the chosen edge.
    let ew = (w as f64 * 0.05).round() as i32;
    let eh = (h as f64 * 0.05).round() as i32;
    let mut candidates: Vec<(i32, i32, BelId)> = Vec::new();
    for bel in ctx.bels_for_bucket(iob_bucket) {
        let loc = bel.loc();
        let keep = match edge {
            "left" => loc.x <= ew,
            "right" => loc.x >= w - ew,
            "bottom" => loc.y <= eh,
            "top" => loc.y >= h - eh,
            _ => false,
        };
        if keep {
            candidates.push((loc.x, loc.y, bel.id()));
        }
    }

    // Sort so we fill the edge in a stable order.
    match edge {
        "left" | "right" => candidates.sort_by_key(|(_, y, _)| *y),
        "bottom" | "top" => candidates.sort_by_key(|(x, _, _)| *x),
        _ => {}
    }

    // At this point (right after packer::pack) IOBs have cell_type=IOB but no BEL.
    // Collect them and prepare to bind straight to the chosen edge.
    let mut iob_cells: Vec<nextpnr::netlist::CellId> = Vec::new();
    let mut old_bels: Vec<BelId> = Vec::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let bucket_name = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
        if bucket_name != "IOB" {
            continue;
        }
        iob_cells.push(cell_id);
        if let Some(bel) = cell.bel {
            old_bels.push(bel);
        }
    }

    if candidates.len() < iob_cells.len() {
        eprintln!(
            "ERROR: {} IOB BELs available on {} edge but {} IOBs to place; aborting",
            candidates.len(),
            edge,
            iob_cells.len()
        );
        return;
    }

    // Unbind all IOB cells from their current BELs.
    for bel in old_bels {
        ctx.unbind_bel(bel);
    }
    // Rebind to the chosen edge.
    for (i, cell_id) in iob_cells.iter().enumerate() {
        let (_x, _y, bel) = candidates[i];
        let ok = ctx.bind_bel(bel, *cell_id, PlaceStrength::Locked);
        if !ok {
            eprintln!("bind_bel failed for IOB cell {}; continuing", i);
        }
    }

    eprintln!(
        "IO relocation: placed {} IOBs on {} edge ({} candidate BELs on that edge)",
        iob_cells.len(),
        edge,
        candidates.len()
    );
}

fn main() {
    let chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json";

    let db = ChipDb::load(Path::new(chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    if let Ok(edge) = env::var("NPNR_IO_EDGE") {
        relocate_ios_to_edge(&mut ctx, &edge);
    }

    let mut cfg = OptTransPlacerCfg::default();
    cfg.seed = 42;
    cfg.timing_weight = env::var("NPNR_OT_TRACE_TIMING_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    cfg.report_interval = env::var("NPNR_OT_TRACE_REPORT_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    if let Some(num_threads) = env::var("NPNR_OT_TRACE_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.num_threads = num_threads.max(1);
    }
    if let Some(batch_size) = env::var("NPNR_OT_TRACE_NET_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.net_parallel_batch_size = batch_size.max(1);
    }
    if let Some(v) = env::var("NPNR_OT_MAX_OUTER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.max_outer_iters = v.max(1);
    }
    if let Some(v) = env::var("NPNR_OT_DCD_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.dcd_iters_per_cell = v.max(1);
    }
    if env::var("NPNR_OT_SEQUENTIAL_BISECTION").ok().as_deref() == Some("1") {
        cfg.sweep_mode = nextpnr::placer::opt_trans::SweepMode::SequentialBisection;
        eprintln!("sweep_mode: SequentialBisection");
    }
    if env::var("NPNR_OT_JACOBI_BISECTION").ok().as_deref() == Some("1") {
        cfg.sweep_mode = nextpnr::placer::opt_trans::SweepMode::JacobiBisection;
        eprintln!("sweep_mode: JacobiBisection");
    }
    if env::var("NPNR_OT_JACOBI_BB").ok().as_deref() == Some("1") {
        cfg.sweep_mode = nextpnr::placer::opt_trans::SweepMode::JacobiBB;
        eprintln!("sweep_mode: JacobiBB");
    }
    if let Some(v) = env::var("NPNR_OT_BISECT_REGION")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
    {
        cfg.bisect_refresh_region = v.max(1);
        eprintln!("bisect_refresh_region: {}", cfg.bisect_refresh_region);
    }
    if let Some(v) = env::var("NPNR_OT_BISECT_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.bisect_seed_k = v.max(2);
        eprintln!("bisect_seed_k: {}", cfg.bisect_seed_k);
    }
    PlacerOptTrans.place(&mut ctx, &cfg).expect("place");
}

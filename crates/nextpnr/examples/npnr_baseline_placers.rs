//! Placer-vs-placer on the shared nextpnr fabric.
//!
//! `tests/npnr_baseline_compare.rs` proved our `get_net_metric` is identical to
//! upstream nextpnr's on the same chipdb and the same placement. That makes the
//! metric a trustworthy referee, so this driver uses it to score *placements*:
//!
//!   - upstream nextpnr's own placement, read back from `placed.json`
//!   - eisenjoch's `opt_trans` (DCD) placing the same netlist from scratch
//!
//! Both are scored by the same validated function on the same fabric, so the
//! difference is placer quality and nothing else.
//!
//! Usage:
//!   npnr_baseline_placers <chipdb.bin> <constids.inc> <bench.json> <placed.json>

use nextpnr::chipdb::{parse_constids_inc, BelId, ChipDb};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::place_common::{get_net_metric, MetricType, WirelenT};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn load(chipdb: &str, constids: &str, design: &str) -> Context {
    let known = parse_constids_inc(&std::fs::read_to_string(constids).expect("constids.inc"));
    let db = ChipDb::load_with_known_constids(Path::new(chipdb), &known).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    // The example uarch's pack() trims nextpnr's IO pseudo-cells because
    // synthesis (iopadmap) already inserted real INBUF/OUTBUF. Without this the
    // pseudo-cells compete for the same IO bels.
    let trimmed = nextpnr::packer::passes::remove_nextpnr_iobs(&mut ctx).expect("trim iobs");
    eprintln!("  (trimmed {trimmed} nextpnr IO pseudo-cells)");
    adapt_cell_types(&mut ctx);
    ctx
}

/// Map cell types onto this fabric's bel types, the way the example uarch does.
///
/// Two distinct nextpnr mechanisms, both of which reduce to a retype here:
///
/// - `pack()`'s `replace_constants` turns constant nets into real `GND_DRV` /
///   `VCC_DRV` cells occupying actual bels; our frontend emits generic
///   `GND`/`VCC`. This matters — 88 LUT4 inputs on this benchmark tie to GND,
///   so it is a genuine high-fanout net both placers must handle.
/// - `getBelBucketForCellType` sends `INBUF`/`OUTBUF` to the `IOB` bucket. The
///   fabric's other bel types (LUT4, DFF, GND_DRV, VCC_DRV) are identity, so
///   retyping is exactly equivalent to the bucket rule on this arch.
///
/// This is arch-specific naming, so it lives in the driver, not the packer.
fn adapt_cell_types(ctx: &mut Context) {
    for (from, to) in [
        ("GND", "GND_DRV"),
        ("VCC", "VCC_DRV"),
        ("INBUF", "IOB"),
        ("OUTBUF", "IOB"),
    ] {
        let (from_id, to_id) = (ctx.id(from), ctx.id(to));
        let victims: Vec<_> = ctx
            .design
            .iter_alive_cells()
            .filter(|(_, c)| c.cell_type == from_id)
            .map(|(i, _)| i)
            .collect();
        for i in victims {
            ctx.design.cell_edit(i).set_type(to_id);
        }
    }
}

/// Sum of `get_net_metric(WIRELENGTH)` over every net -- the referee.
fn score(ctx: &Context) -> (WirelenT, usize) {
    let mut total: WirelenT = 0;
    let mut n = 0usize;
    for net in ctx.nets() {
        let mut tns = 0.0f32;
        total += get_net_metric(ctx, net.id(), MetricType::Wirelength, &mut tns);
        n += 1;
    }
    (total, n)
}

/// Split total wirelength by whether a net touches an IO cell.
///
/// DCD only moves cell types whose bels live on CLB tiles, so IO cells keep
/// whatever bel the initial discrete binding handed them. If that is the gap,
/// it shows up here as an IO-touching share far above nextpnr's.
fn score_by_io(ctx: &Context) -> (WirelenT, WirelenT, usize, usize) {
    let iob = ctx.id("IOB");
    let (mut io_wl, mut core_wl, mut io_n, mut core_n) = (0, 0, 0, 0);
    for net in ctx.nets() {
        let id = net.id();
        let info = ctx.design.net(id);
        let touches_io = info
            .driver()
            .into_iter()
            .chain(info.users().iter().copied())
            .filter(|p| p.cell.is_some())
            .any(|p| ctx.design.cell(p.cell).cell_type == iob);
        let mut tns = 0.0f32;
        let wl = get_net_metric(ctx, id, MetricType::Wirelength, &mut tns);
        if touches_io {
            io_wl += wl;
            io_n += 1;
        } else {
            core_wl += wl;
            core_n += 1;
        }
    }
    (io_wl, core_wl, io_n, core_n)
}

fn bound_count(ctx: &Context) -> (usize, usize) {
    let mut bound = 0usize;
    let mut total = 0usize;
    for cell_idx in ctx.design.iter_cell_indices() {
        total += 1;
        if ctx.design.cell(cell_idx).bel.is_some() {
            bound += 1;
        }
    }
    (bound, total)
}

/// Rebuild nextpnr's placement from the `NEXTPNR_BEL` attributes.
fn apply_nextpnr_placement(ctx: &mut Context) {
    let mut by_name: HashMap<String, BelId> = HashMap::new();
    for bel in ctx.chipdb().bels() {
        let loc = ctx.chipdb().bel_loc(bel);
        by_name.insert(
            format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel)),
            bel,
        );
    }
    let bel_attr = ctx.id("NEXTPNR_BEL");
    let bindings: Vec<_> = ctx
        .design
        .iter_cell_indices()
        .filter_map(|c| {
            let name = ctx.design.cell(c).attrs.get(&bel_attr)?.as_str();
            Some((*by_name.get(&name)?, c))
        })
        .collect();
    for (bel, c) in bindings {
        assert!(ctx.bind_bel(bel, c, PlaceStrength::Strong), "bind failed");
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: <chipdb.bin> <constids.inc> <bench.json> <placed.json>");
        std::process::exit(2);
    }
    let (chipdb, constids, bench, placed) = (&a[0], &a[1], &a[2], &a[3]);

    // --- Reference: upstream nextpnr's placement, scored by us ---------------
    let mut ref_ctx = load(chipdb, constids, placed);
    apply_nextpnr_placement(&mut ref_ctx);
    let (ref_wl, ref_nets) = score(&ref_ctx);
    let (rb, rt) = bound_count(&ref_ctx);
    let (r_io, r_core, r_ion, r_coren) = score_by_io(&ref_ctx);
    println!("nextpnr      : wirelength {ref_wl:>8}   nets {ref_nets:>5}   bound {rb}/{rt}");
    println!("               io-touching {r_io:>6} over {r_ion:>4} nets | core {r_core:>6} over {r_coren:>4} nets");

    // --- Ours: opt_trans placing the same netlist from scratch ---------------
    let mut ot_ctx = load(chipdb, constids, bench);
    let t = Instant::now();
    let envf = |k: &str, d: f64| -> f64 {
        std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let envu = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let mut cfg = OptTransPlacerCfg::default();
    cfg.num_threads = 8;
    cfg.max_outer_iters = envu("OT_ITERS", 50);
    cfg.dcd_iters_per_cell = envu("OT_DCD_ITERS", cfg.dcd_iters_per_cell);
    cfg.steiner_weight = envf("OT_STEINER", cfg.steiner_weight);
    cfg.mst_edge_weight = envf("OT_MST", cfg.mst_edge_weight);
    cfg.seed = envu("OT_SEED", cfg.seed as usize) as u64;
    eprintln!(
        "  cfg: iters={} dcd_iters={} steiner={} mst={} seed={}",
        cfg.max_outer_iters, cfg.dcd_iters_per_cell, cfg.steiner_weight, cfg.mst_edge_weight, cfg.seed
    );
    match place_opt_trans(&mut ot_ctx, &cfg) {
        Ok(()) => {
            let secs = t.elapsed().as_secs_f64();
            let (wl, nets) = score(&ot_ctx);
            let (b, tot) = bound_count(&ot_ctx);
            let (o_io, o_core, o_ion, o_coren) = score_by_io(&ot_ctx);
            println!("opt_trans    : wirelength {wl:>8}   nets {nets:>5}   bound {b}/{tot}   {secs:.1}s");
            println!("               io-touching {o_io:>6} over {o_ion:>4} nets | core {o_core:>6} over {o_coren:>4} nets");
            println!(
                "               excess: io {:>6}  core {:>6}",
                o_io - r_io,
                o_core - r_core
            );
            if ref_wl > 0 {
                println!(
                    "\nratio opt_trans / nextpnr = {:.3}x",
                    wl as f64 / ref_wl as f64
                );
            }
        }
        Err(e) => println!("opt_trans    : FAILED after {:.1}s: {e}", t.elapsed().as_secs_f64()),
    }
}

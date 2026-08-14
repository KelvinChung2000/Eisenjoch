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

/// Per-class wirelength: (total, net count) for each of three classes.
#[derive(Default, Clone, Copy)]
struct ClassStats {
    lutff: (WirelenT, usize),
    io: (WirelenT, usize),
    core: (WirelenT, usize),
}

/// Split wirelength three ways.
///
/// `lutff` is the class nextpnr *packs*: `constrain_cell_pairs(LUT4.F ->
/// DFF.D, delta_z=1, allow_fanout=false)` pins the DFF to constr_x/y=0,
/// constr_z=+1 -- the same tile as its LUT, making those nets 0.
///
/// It is tempting to call the excess here "packing" and discount it. Measured,
/// that is wrong: with the constraint disabled (patches/0002) HeAP still places
/// these 128 nets at total distance 32, and its overall wirelength *improves*
/// (1857 -> 1825). HeAP finds the co-location unaided, so this class measures
/// placement quality like any other -- see docs/dcd_vs_nextpnr_baseline.md.
///
/// `io` is any remaining net touching an IO cell; `core` is the rest.
fn score_by_class(ctx: &Context) -> ClassStats {
    let (iob, lut4, dff) = (ctx.id("IOB"), ctx.id("LUT4"), ctx.id("DFF"));
    let mut s = ClassStats::default();
    for net in ctx.nets() {
        let id = net.id();
        let info = ctx.design.net(id);
        let live: Vec<_> = info.users().iter().copied().filter(|p| p.cell.is_some()).collect();
        let drv = info.driver().filter(|p| p.cell.is_some());

        // Mirror constrain_cell_pairs: LUT4 output, single sink, sink is a DFF.
        let is_lutff = drv.is_some_and(|d| ctx.design.cell(d.cell).cell_type == lut4)
            && live.len() == 1
            && ctx.design.cell(live[0].cell).cell_type == dff;
        let touches_io = drv
            .into_iter()
            .chain(live.iter().copied())
            .any(|p| ctx.design.cell(p.cell).cell_type == iob);

        let mut tns = 0.0f32;
        let wl = get_net_metric(ctx, id, MetricType::Wirelength, &mut tns);
        let slot = if is_lutff {
            &mut s.lutff
        } else if touches_io {
            &mut s.io
        } else {
            &mut s.core
        };
        slot.0 += wl;
        slot.1 += 1;
    }
    s
}

fn report_classes(label: &str, s: ClassStats) {
    println!(
        "               {label:<9} lutff-pair {:>6} / {:>4} nets | io {:>6} / {:>4} | core {:>6} / {:>4}",
        s.lutff.0, s.lutff.1, s.io.0, s.io.1, s.core.0, s.core.1
    );
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
    let rs = score_by_class(&ref_ctx);
    println!("nextpnr      : wirelength {ref_wl:>8}   nets {ref_nets:>5}   bound {rb}/{rt}");
    report_classes("", rs);

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
    cfg.num_threads = envu("OT_THREADS", 8);
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
            let os = score_by_class(&ot_ctx);
            println!("opt_trans    : wirelength {wl:>8}   nets {nets:>5}   bound {b}/{tot}   {secs:.1}s");
            report_classes("", os);
            println!(
                "               excess:   lutff-pair {:>6} | io {:>6} | core {:>6}",
                os.lutff.0 - rs.lutff.0,
                os.io.0 - rs.io.0,
                os.core.0 - rs.core.0
            );
            let ex_lutff = os.lutff.0 - rs.lutff.0;
            let ex_rest = (os.io.0 - rs.io.0) + (os.core.0 - rs.core.0);
            println!(
                "               lutff-class share of excess: {ex_lutff} of {} ({:.0}%)",
                ex_lutff + ex_rest,
                100.0 * ex_lutff as f64 / (ex_lutff + ex_rest).max(1) as f64
            );
            println!(
                "               excluding lutff class: ours {} vs nextpnr {} = {:.3}x",
                wl - os.lutff.0,
                ref_wl - rs.lutff.0,
                (wl - os.lutff.0) as f64 / (ref_wl - rs.lutff.0).max(1) as f64
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

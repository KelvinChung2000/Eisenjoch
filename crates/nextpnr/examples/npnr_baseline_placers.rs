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
    apply_bel_constraints(&mut ctx);
    install_slice_valid(&mut ctx);
    ctx
}

/// Honour the `BEL` attribute as a locked pre-placement.
///
/// nextpnr applies these as placer constraints; without them the clock buffer
/// is free to move, and on the example fabric that is fatal -- its clock ladder
/// is fed by a single `GCLK_OUT` pip that exists only in the tile at X1Y0, so
/// `clk_buf` carries `(* BEL = "X1Y0/IO0" *)` and routing fails anywhere else.
///
/// `Locked` is what makes this stick: `lock_boundary_cells` and the warmup
/// re-shuffle both skip cells that are already locked, and `initial_placement`
/// skips anything already bound.
fn apply_bel_constraints(ctx: &mut Context) {
    let bel_attr = ctx.id("BEL");
    let by_name = bel_name_map(ctx);
    let wanted: Vec<_> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(c, cell)| Some((c, cell.attrs.get(&bel_attr)?.as_str())))
        .collect();
    for (c, name) in wanted {
        let bel = *by_name
            .get(&name)
            .unwrap_or_else(|| panic!("BEL constraint names no such bel: {name}"));
        assert!(
            ctx.bind_bel(bel, c, PlaceStrength::Locked),
            "BEL constraint {name} could not be bound"
        );
        eprintln!("  (locked {name} from BEL constraint)");
    }
}

/// Install the example uarch's `slice_valid` as this arch's validity rule.
///
/// Port of `ExampleImpl::slice_valid` / `isBelLocationValid` (upstream
/// `4d235150`, `himbaechel/uarch/example/example.cc:134`). A slice holds a LUT
/// at z and its FF at z+1; if both are occupied, the FF's D net must be the
/// LUT's F net, or the LUT's I3 must be unused.
///
/// The I3 escape hatch is nearly dead in practice: `lut_i3_used` tests the
/// *net*, and a constant tie is a real net once constants are driven, so an I3
/// tied to '0' counts as used.
///
/// Keyed off bel *type*, not z parity: IO tiles put IOBs at z = i, so parity
/// would misread them as slices.
fn install_slice_valid(ctx: &mut Context) {
    let mut by_loc: HashMap<(i32, i32, i32), BelId> = HashMap::new();
    for bel in ctx.chipdb().bels() {
        let t = ctx.chipdb().bel_type(bel);
        if t == "LUT4" || t == "DFF" {
            let l = ctx.chipdb().bel_loc(bel);
            by_loc.insert((l.x, l.y, l.z), bel);
        }
    }
    let (f, d, i3) = (ctx.id("F"), ctx.id("D"), ctx.id("I[3]"));

    ctx.set_validity_check(std::sync::Arc::new(move |ctx: &Context, bel: BelId| {
        let loc = ctx.chipdb().bel_loc(bel);
        let (lut_z, ff_z) = match ctx.chipdb().bel_type(bel) {
            "LUT4" => (loc.z, loc.z + 1),
            "DFF" => (loc.z - 1, loc.z),
            _ => return true,
        };
        let lookup = |z| {
            by_loc
                .get(&(loc.x, loc.y, z))
                .and_then(|&b| ctx.bel(b).bound_cell().map(|c| c.id()))
        };
        let (Some(lut), Some(ff)) = (lookup(lut_z), lookup(ff_z)) else {
            return true; // only one half of the slice is used
        };
        if ctx.design.cell(lut).port_net(f) == ctx.design.cell(ff).port_net(d) {
            return true;
        }
        ctx.design.cell(lut).port_net(i3).is_none()
    }));
}

/// `"X{x}Y{y}/{bel}"` -> bel, the name form nextpnr reads and writes.
fn bel_name_map(ctx: &Context) -> HashMap<String, BelId> {
    let mut by_name = HashMap::new();
    for bel in ctx.chipdb().bels() {
        let loc = ctx.chipdb().bel_loc(bel);
        by_name.insert(
            format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel)),
            bel,
        );
    }
    by_name
}

/// Write the input netlist back out with our placement as `NEXTPNR_BEL` attrs.
///
/// This is what lets nextpnr *route* our placement. `attributesToArchInfo()`
/// runs in upstream's JSON frontend, before `pack()`, and calls
/// `bindBel(bel, cell, strength)` with the strength from `BEL_STRENGTH`,
/// defaulting to `STRENGTH_USER` (6). Every placer skips cells above
/// `STRENGTH_STRONG` (2), so omitting `BEL_STRENGTH` pins the placement: the
/// binary re-places nothing and goes straight to routing ours.
///
/// Cells nextpnr's own `pack()` creates -- the constant drivers -- are not in
/// the input JSON and so get no attribute; nextpnr places those itself.
///
/// Fails loudly on any mismatch: a partially-injected placement would still
/// route, and would silently be a different experiment.
fn write_placement_json(ctx: &Context, bench: &str, out: &str) {
    let mut root: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bench).expect("read bench")).expect("json");
    let top = ctx.name_of(ctx.design.top_module).to_owned();
    let cells = root["modules"][&top]["cells"]
        .as_object_mut()
        .unwrap_or_else(|| panic!("no cells in module {top}"));

    let mut injected = 0usize;
    let mut unplaced = Vec::new();
    for (name, cell_json) in cells.iter_mut() {
        let id = ctx.id(name);
        let Some(c) = ctx.design.cell_by_name(id) else {
            panic!("JSON cell {name} is absent from our design")
        };
        match ctx.design.cell(c).bel {
            Some(bel) => {
                let loc = ctx.chipdb().bel_loc(bel);
                let bel_name = format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel));
                cell_json["attributes"]["NEXTPNR_BEL"] = serde_json::Value::String(bel_name);
                // Drop any `BEL` constraint, exactly as upstream's
                // archInfoToAttributes() does. Left in place it is applied a
                // second time by the placer's constraint pass, which then
                // errors: "cannot be bound ... already bound to cell".
                if let Some(a) = cell_json["attributes"].as_object_mut() {
                    a.remove("BEL");
                }
                injected += 1;
            }
            None => unplaced.push(name.clone()),
        }
    }
    assert!(unplaced.is_empty(), "cells left unplaced: {unplaced:?}");
    std::fs::write(out, serde_json::to_string(&root).expect("serialize")).expect("write placement");
    eprintln!("  (wrote {injected} NEXTPNR_BEL attrs to {out})");
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
    let by_name = bel_name_map(ctx);
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

    // Optional: make slice legality structural instead of a filter, the way
    // nextpnr's example pack() does. Off by default so the primary comparison
    // is unpacked-vs-unpacked against placed*_nopack.json; set OT_PAIR_LUTFF=1
    // to compare packed-vs-packed instead.
    // Value-based, not presence-based: `OT_PAIR_LUTFF=` sets the variable to an
    // empty string, which `is_ok()` would accept and silently enable pairing.
    if std::env::var("OT_PAIR_LUTFF").as_deref() == Ok("1") {
        let (lut4, dff) = (ot_ctx.id("LUT4"), ot_ctx.id("DFF"));
        let (f, d) = (ot_ctx.id("F"), ot_ctx.id("D"));
        let n = nextpnr::packer::passes::constrain_cell_pairs(
            &mut ot_ctx,
            &[nextpnr::packer::passes::CellTypePort::new(lut4, f)],
            &[nextpnr::packer::passes::CellTypePort::new(dff, d)],
            1,
            false,
        );
        eprintln!("  (constrained {n} LUTFF pairs)");
    }

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
            if let Ok(path) = std::env::var("OT_WRITE_PLACEMENT") {
                write_placement_json(&ot_ctx, bench, &path);
            }
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

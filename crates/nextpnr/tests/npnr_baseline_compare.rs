//! Cross-tool baseline: does our ported wirelength model agree with nextpnr's?
//!
//! Both sides read the *same* himbaechel chipdb and the *same* placement, so
//! nothing here depends on the two tools agreeing about anything else. The
//! fixtures are produced by `tools/npnr_compare/regen.sh`:
//!
//! - `shared_chipdb.bin` -- a synthetic 12x12 fabric built by upstream's own
//!   `example_arch_gen.py`, loaded by nextpnr and by us.
//! - `placed.json` -- upstream nextpnr's place+route output, carrying a
//!   `NEXTPNR_BEL` attribute per cell.
//! - `golden_net_metric.txt` -- per-net wirelength, emitted by upstream nextpnr
//!   calling its *own* `get_net_metric`. It is not a reimplementation: the
//!   numbers come from `common/place/place_common.cc` at the pinned hash.
//!
//! `MetricType::WIRELENGTH` short-circuits the timing weighting in both
//! implementations (`timing_driven` requires `type == COST`), so these are pure
//! integer HPWL values. They must match **exactly** -- an off-by-one is a
//! defect, not noise.

use nextpnr::chipdb::{parse_constids_inc, BelId, ChipDb};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::placer::place_common::{get_net_metric, MetricType, WirelenT};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npnr_baseline")
        .join(name)
}

/// `name -> wirelength`, plus the total nextpnr reported.
fn load_golden() -> (HashMap<String, WirelenT>, WirelenT) {
    let raw = std::fs::read_to_string(fixture("golden_net_metric.txt")).expect("golden fixture");
    let mut per_net = HashMap::new();
    let mut total = None;

    for line in raw.lines() {
        let Some((name, value)) = line.rsplit_once('\t') else {
            continue;
        };
        let value: WirelenT = value.trim().parse().expect("golden value is an integer");
        if name == "# total" {
            total = Some(value);
        } else {
            per_net.insert(name.to_string(), value);
        }
    }
    (per_net, total.expect("golden file records a total"))
}

/// The fixture's netlist on the shared chipdb, with no cell bound.
fn load_unplaced_context() -> Context {
    let known = parse_constids_inc(&std::fs::read_to_string(fixture("constids.inc")).unwrap());
    let chipdb = ChipDb::load_with_known_constids(&fixture("shared_chipdb.bin"), &known)
        .expect("the shared chipdb must load with the arch's constids");
    let mut ctx = Context::new(chipdb);

    let json = std::fs::read_to_string(fixture("placed.json")).expect("placed fixture");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("nextpnr output must parse");
    ctx
}

/// Rebuild the placement nextpnr produced, from the `NEXTPNR_BEL` attributes.
fn load_placed_context() -> Context {
    let mut ctx = load_unplaced_context();

    // nextpnr names bels "X<x>Y<y>/<name>"; we index them the same way so the
    // attribute can be resolved without depending on bel ordering.
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
        .filter_map(|cell_idx| {
            let name = ctx.design.cell(cell_idx).attrs.get(&bel_attr)?.as_str();
            let bel = *by_name
                .get(&name)
                .unwrap_or_else(|| panic!("placed cell references unknown bel {name}"));
            Some((bel, cell_idx))
        })
        .collect();

    assert!(
        !bindings.is_empty(),
        "no cell carried a NEXTPNR_BEL attribute -- the fixture is not a placed design"
    );

    for (bel, cell_idx) in bindings {
        assert!(
            ctx.bind_bel(bel, cell_idx, PlaceStrength::Strong),
            "binding the placement nextpnr already found must succeed"
        );
    }
    ctx
}

#[test]
fn shared_chipdb_is_the_fabric_nextpnr_placed_on() {
    let ctx = load_placed_context();
    assert_eq!(ctx.chipdb().width(), 12);
    assert_eq!(ctx.chipdb().height(), 12);
    assert_eq!(ctx.chipdb().num_tiles(), 144);
}

#[test]
fn every_net_matches_nextpnrs_own_get_net_metric() {
    let ctx = load_placed_context();
    let (golden, golden_total) = load_golden();

    let mut ours_total: WirelenT = 0;
    let mut checked = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for net in ctx.nets() {
        let name = net.name();
        let mut tns = 0.0f32;
        let ours = get_net_metric(&ctx, net.id(), MetricType::Wirelength, &mut tns);
        ours_total += ours;

        let Some(&want) = golden.get(name) else {
            continue;
        };
        checked += 1;
        if ours != want {
            mismatches.push(format!("  {name}: ours {ours}, nextpnr {want}"));
        }
    }

    assert!(
        checked >= golden.len(),
        "only matched {checked} of {} golden nets by name -- the netlists have diverged",
        golden.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} of {checked} nets disagree with nextpnr:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert_eq!(
        ours_total, golden_total,
        "total wirelength must match nextpnr exactly"
    );
    println!("matched {checked} nets, total wirelength {ours_total} (nextpnr: {golden_total})");
}

/// The default opt_trans flow (JacobiFullscan + DialLogit) must keep producing
/// this exact placement. Memory work on the placer may change how labels and
/// usage are stored, never what they are, so drift here is a numeric change
/// that has to be justified on its own.
///
/// The default sweep commits moves racily, so whichever thread reaches a
/// contested tile slot first wins it and the placement differs run to run even
/// on one solve thread. `NPNR_OT_DET_SWEEP=1` orders those commits by cell
/// index and touches nothing else, so it pins everything the default path
/// computes: labels, usage, costs and candidates.
#[test]
fn opt_trans_default_placement_is_pinned() {
    // Read once per process by the placer; no other test in this binary
    // runs opt_trans.
    std::env::set_var("NPNR_OT_DET_SWEEP", "1");
    // Several runs and thread counts: a pin that only holds on some runs would
    // say nothing about the code under test.
    let runs: Vec<_> = [1, 4, 4]
        .map(|threads| place_fixture_opt_trans(|cfg| cfg.num_threads = threads))
        .to_vec();
    for (cells, hash, hpwl) in &runs {
        println!("opt_trans placement: cells={cells} hash={hash:#018x} hpwl={hpwl}");
    }
    for run in &runs {
        assert_eq!(*run, (151, 0xc393_a8dd_d642_60bf, 531));
    }
}

/// Place the fixture's netlist from scratch with the default opt_trans config.
/// Returns `(cells, FNV-1a of the sorted cell->bel list, total HPWL)`.
fn place_fixture_opt_trans(
    tweak: impl FnOnce(&mut nextpnr::placer::opt_trans::OptTransPlacerCfg),
) -> (usize, u64, WirelenT) {
    use nextpnr::packer::passes::remove_nextpnr_iobs;
    use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};

    // Start from nextpnr's placement so that cells whose type has no bucket
    // (constant drivers) keep their bel, fixed; everything else is unbound
    // and left to the placer.
    let mut ctx = load_placed_context();
    // The fixture is packed with real IO buffers, so the frontend's port
    // pseudo-cells would only double-book the IO sites.
    remove_nextpnr_iobs(&mut ctx).unwrap();
    ctx.populate_bel_buckets();
    let bound: Vec<_> = ctx
        .design
        .iter_cell_indices()
        .filter_map(|ci| ctx.design.cell(ci).bel.map(|bel| (ci, bel)))
        .collect();
    for (ci, bel) in bound {
        let cell_type = ctx.design.cell(ci).cell_type;
        let has_bucket = ctx.bels_for_bucket(cell_type).next().is_some();
        ctx.unbind_bel(bel);
        if !has_bucket {
            assert!(ctx.bind_bel(bel, ci, PlaceStrength::Fixed));
        }
    }
    // The JSON frontend adds its own constant drivers; the packed fixture
    // already carries placed ones, so these drive nothing and have no bel.
    let dangling: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .filter_map(|(_, net)| {
            let cell = ctx.design.cell(net.driver()?.cell);
            let orphan = net.num_users() == 0
                && cell.bel.is_none()
                && ctx.bels_for_bucket(cell.cell_type).next().is_none();
            orphan.then_some((net.name, cell.name))
        })
        .collect();
    for (net, cell) in dangling {
        ctx.design.remove_net(net);
        ctx.design.remove_cell(cell);
    }
    let mut cfg = OptTransPlacerCfg {
        num_threads: 4,
        max_outer_iters: 6,
        ..OptTransPlacerCfg::default()
    };
    tweak(&mut cfg);
    place_opt_trans(&mut ctx, &cfg).expect("opt_trans must place the fixture");

    let mut placed: Vec<(String, String)> = ctx
        .cells()
        .filter(|c| c.is_alive())
        .map(|c| {
            let bel = c.bel_id().expect("every cell is placed");
            let loc = ctx.chipdb().bel_loc(bel);
            let bel = format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel));
            (c.name().to_string(), bel)
        })
        .collect();
    placed.sort();
    // FNV-1a over the sorted cell->bel list.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (cell, bel) in &placed {
        for b in cell.bytes().chain([0]).chain(bel.bytes()).chain([0]) {
            hash = (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut hpwl: WirelenT = 0;
    for net in ctx.nets() {
        let mut tns = 0.0f32;
        hpwl += get_net_metric(&ctx, net.id(), MetricType::Wirelength, &mut tns);
    }
    (placed.len(), hash, hpwl)
}

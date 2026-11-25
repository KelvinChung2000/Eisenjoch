//! Golden cell -> BEL maps for every placer and legalizer on the example
//! chipdb.
//!
//! The BEL indexes these flows sample from are pure data-structure choices:
//! swapping one for another must not move a single cell. This pins the final
//! assignment of each flow on a design that exercises the paths an index
//! change can disturb -- several cell types, a type rare enough to trip
//! refinement's `min_bels_for_grid_pick`, LUT/FF clusters, a region, and a
//! cell type aliased onto another type's BELs.
//!
//! Regenerate with `NPNR_BLESS_PLACEMENT=1` after an intentional change.

mod common;

use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::{CellId, PortType, Rect};
use nextpnr::placer::common::TypeAwarePlacement;
use nextpnr::placer::electro_place::{place_electro, ElectroPlaceCfg};
use nextpnr::placer::heap::{place_heap, PlacerHeapCfg};
use nextpnr::placer::legalize::create_legalizer;
use nextpnr::placer::place_common::legalise_relative_constraints;
use nextpnr::placer::sa::{place_sa, PlacerSaCfg};
use nextpnr::placer::{refine_placement, RefineCfg};

const GOLDEN: &str = "tests/fixtures/placement_golden.txt";

fn connect(ctx: &mut Context, name: &str, drv: (CellId, &str), users: &[(CellId, &str)]) {
    let net = ctx.design.add_net(ctx.id(name));
    let port = ctx.id(drv.1);
    ctx.design.cell_edit(drv.0).add_port(port, PortType::Out);
    ctx.design
        .cell_edit(drv.0)
        .set_port_net(port, Some(net), None);
    ctx.design.net_edit(net).set_driver(drv.0, port);
    for &(cell, p) in users {
        let port = ctx.id(p);
        ctx.design.cell_edit(cell).add_port(port, PortType::In);
        let user = ctx.design.net_edit(net).add_user(cell, port);
        ctx.design
            .cell_edit(cell)
            .set_port_net(port, Some(net), Some(user));
    }
}

/// 24 LUTs (the first 10 each clustered with an FF one slot up), 6 free FFs
/// (the last 4 of an alias type that shares the FF BELs), 2 BRAMs, and 4 LUTs
/// held to a 2x2 region away from the fabric's centre.
fn build_design() -> Context {
    let mut ctx = common::make_example_context();
    ctx.add_cell_type_alias("DFF_ALT", "DFF");
    let lut_t = ctx.id("LUT6");
    let ff_t = ctx.id("DFF");
    let ff_alt_t = ctx.id("DFF_ALT");
    let ram_t = ctx.id("BRAM_512X16");

    let luts: Vec<CellId> = (0..24)
        .map(|i| ctx.design.add_cell(ctx.id(&format!("lut{i:02}")), lut_t))
        .collect();
    let ffs: Vec<CellId> = (0..16)
        .map(|i| {
            let ty = if i < 12 { ff_t } else { ff_alt_t };
            ctx.design.add_cell(ctx.id(&format!("ff{i:02}")), ty)
        })
        .collect();
    let rams: Vec<CellId> = (0..2)
        .map(|i| ctx.design.add_cell(ctx.id(&format!("ram{i}")), ram_t))
        .collect();

    for i in 0..10 {
        let (root, child) = (luts[i], ffs[i]);
        ctx.design.cell_edit(root).set_cluster(Some(root));
        ctx.design.cell_edit(root).set_constraints(0, 0, 0, false);
        ctx.design.cell_edit(child).set_cluster(Some(root));
        ctx.design.cell_edit(child).set_constraints(0, 0, 1, false);
        let cluster = ctx
            .design
            .clusters
            .entry(root)
            .or_insert_with(|| nextpnr::netlist::Cluster::new(root));
        cluster.members.push(child);
        cluster.constr_children.push(child);
    }

    for i in 0..24 {
        let mut users = vec![
            (luts[(i + 1) % 24], "I[0]"),
            (luts[(i * 7 + 3) % 24], "I[1]"),
        ];
        if i < 16 {
            users.push((ffs[i], "D"));
        }
        if i < 2 {
            users.push((rams[i], "DI[0]"));
        }
        connect(&mut ctx, &format!("n_lut{i}"), (luts[i], "F"), &users);
    }
    for i in 0..16 {
        connect(
            &mut ctx,
            &format!("n_ff{i}"),
            (ffs[i], "Q"),
            &[(luts[(i * 5 + 2) % 24], "I[2]")],
        );
    }
    for i in 0..2 {
        connect(
            &mut ctx,
            &format!("n_ram{i}"),
            (rams[i], "DO[0]"),
            &[(luts[i * 11 + 4], "I[3]")],
        );
    }

    let region = ctx.design.add_region(ctx.id("corner"));
    ctx.design
        .region_mut(region)
        .rects
        .push(Rect::new(1, 1, 2, 2));
    for &c in &luts[20..24] {
        ctx.design.cell_edit(c).set_region(Some(region));
    }

    ctx.populate_bel_buckets();
    ctx
}

/// One line per cell, by name. BEL names repeat in every tile, so the
/// location is what identifies the BEL.
fn snapshot(ctx: &Context) -> String {
    let mut rows: Vec<(String, String)> = ctx
        .design
        .iter_alive_cells()
        .map(|(_, cell)| {
            let bel = cell.bel.map_or_else(
                || "-".to_owned(),
                |b| {
                    let loc = ctx.bel(b).loc();
                    format!("{} {} {} {}", loc.x, loc.y, loc.z, ctx.bel(b).name())
                },
            );
            (ctx.name_of(cell.name).to_owned(), bel)
        })
        .collect();
    rows.sort();
    rows.into_iter()
        .map(|(c, b)| format!("{c} {b}\n"))
        .collect()
}

/// A flow that fails still leaves cells where it put them, and that state
/// depends on the indexes as much as a success does.
fn record<E: std::fmt::Display>(out: &mut String, label: &str, ctx: &Context, r: Result<(), E>) {
    out.push_str(&format!("# {label}\n"));
    if let Err(e) = r {
        out.push_str(&format!("error: {e}\n"));
    }
    out.push_str(&snapshot(ctx));
}

/// Movable cells and deterministic, deliberately crowded target positions.
fn legalizer_inputs(ctx: &Context) -> (Vec<CellId>, Vec<f64>, Vec<f64>) {
    let cells: Vec<CellId> = ctx
        .design
        .iter_alive_cells()
        .filter(|(id, c)| c.cluster.is_none_or(|root| root == *id))
        .map(|(id, _)| id)
        .collect();
    let xs = (0..cells.len())
        .map(|i| 2.0 + ((i * 37) % 50) as f64 / 10.0)
        .collect();
    let ys = (0..cells.len())
        .map(|i| 2.5 + ((i * 53) % 40) as f64 / 10.0)
        .collect();
    (cells, xs, ys)
}

fn run_all() -> String {
    let mut out = String::new();

    let mut ctx = build_design();
    let r = place_sa(
        &mut ctx,
        &PlacerSaCfg {
            seed: 11,
            cooling_rate: 0.8,
            inner_iters_per_cell: 4,
            min_temp: 0.05,
            ..PlacerSaCfg::default()
        },
    );
    record(&mut out, "sa", &ctx, r);

    // Break one cluster by moving its root, so the constraint legaliser has
    // something to repair.
    let root = ctx.design.cell_by_name(ctx.id("lut00")).expect("lut00");
    if let Some(old) = ctx.design.cell(root).bel {
        let lut_t = ctx.id("LUT6");
        let free = ctx
            .bels_for_bucket(lut_t)
            .map(|b| b.id())
            .filter(|&b| ctx.bel(b).is_available())
            .last();
        if let Some(free) = free {
            ctx.unbind_bel(old);
            assert!(ctx.bind_bel(free, root, PlaceStrength::Weak));
        }
    }
    let r = legalise_relative_constraints(&mut ctx).map(|_| ());
    record(&mut out, "legalise_relative_constraints", &ctx, r);

    let mut ctx = build_design();
    let r = place_sa(
        &mut ctx,
        &PlacerSaCfg {
            seed: 12,
            cooling_rate: 0.8,
            inner_iters_per_cell: 4,
            min_temp: 0.05,
            congestion_weight: 1.0,
            ..PlacerSaCfg::default()
        },
    );
    record(&mut out, "sa congestion", &ctx, r);

    let mut ctx = build_design();
    let r = place_heap(
        &mut ctx,
        &PlacerHeapCfg {
            seed: 5,
            max_iterations: 6,
            ..PlacerHeapCfg::default()
        },
    );
    record(&mut out, "heap", &ctx, r);
    let r = refine_placement(&mut ctx, &RefineCfg::default()).map(|_| ());
    record(&mut out, "heap+refine", &ctx, r);

    let mut ctx = build_design();
    let r = place_electro(
        &mut ctx,
        &ElectroPlaceCfg {
            seed: 3,
            max_iters: 20,
            legalize_interval: 5,
            ..ElectroPlaceCfg::default()
        },
    );
    record(&mut out, "electro", &ctx, r);

    let mut ctx = build_design();
    let r = nextpnr::placer::common::initial_placement_topological(&mut ctx);
    record(&mut out, "initial_placement_topological", &ctx, r);

    for name in ["sorted", "ring", "snap", "bipartite", "greedy"] {
        let mut ctx = build_design();
        let (cells, xs, ys) = legalizer_inputs(&ctx);
        let type_aware = TypeAwarePlacement::build(&ctx, 0, 0);
        let r = create_legalizer(name)
            .legalize(&mut ctx, &cells, &xs, &ys, &type_aware)
            .map(|_| ());
        record(&mut out, &format!("legalize {name}"), &ctx, r);
    }

    out
}

/// A cluster child whose type is an alias must land in its slot. The sorted
/// legaliser keyed its slot index by raw cell type but looked children up by
/// resolved bucket, so it failed on every such cluster.
#[test]
fn sorted_legalizer_places_an_aliased_cluster_child() {
    let mut ctx = common::make_example_context();
    ctx.add_cell_type_alias("DFF_ALT", "DFF");
    let root = ctx.design.add_cell(ctx.id("root"), ctx.id("LUT6"));
    let child = ctx.design.add_cell(ctx.id("child"), ctx.id("DFF_ALT"));
    ctx.design.cell_edit(root).set_cluster(Some(root));
    ctx.design.cell_edit(root).set_constraints(0, 0, 0, false);
    ctx.design.cell_edit(child).set_cluster(Some(root));
    ctx.design.cell_edit(child).set_constraints(0, 0, 1, false);
    let cluster = ctx
        .design
        .clusters
        .entry(root)
        .or_insert_with(|| nextpnr::netlist::Cluster::new(root));
    cluster.members.push(child);
    cluster.constr_children.push(child);
    ctx.populate_bel_buckets();

    let type_aware = TypeAwarePlacement::build(&ctx, 0, 0);
    create_legalizer("sorted")
        .legalize(&mut ctx, &[root], &[4.0], &[4.0], &type_aware)
        .expect("sorted legalisation");

    let root_loc = ctx
        .bel(ctx.design.cell(root).bel.expect("root bound"))
        .loc();
    let child_loc = ctx
        .bel(ctx.design.cell(child).bel.expect("child bound"))
        .loc();
    assert_eq!(
        (child_loc.x, child_loc.y, child_loc.z),
        (root_loc.x, root_loc.y, root_loc.z + 1)
    );
}

#[test]
fn placements_match_the_golden_maps() {
    let got = run_all();
    assert_eq!(got, run_all(), "a flow is not deterministic run to run");

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN);
    if std::env::var("NPNR_BLESS_PLACEMENT").as_deref() == Ok("1") {
        std::fs::write(&path, &got).expect("write golden");
        return;
    }
    let want =
        std::fs::read_to_string(&path).expect("golden file; bless with NPNR_BLESS_PLACEMENT=1");
    for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        assert_eq!(g, w, "first divergence at line {}", i + 1);
    }
    assert_eq!(got.lines().count(), want.lines().count(), "line count");
}

mod common;

use nextpnr::chipdb::BelId;
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::CellId;
use nextpnr::placer::heap::{place_heap, PlacerHeapCfg};
use nextpnr::placer::{refine_placement, Placer, PlacerHeap, PlacerPipeline, RefineCfg};
use rstest::rstest;

// =====================================================================
// Configuration tests
// =====================================================================

#[test]
fn default_heap_config() {
    let cfg = PlacerHeapCfg::default();
    assert_eq!(cfg.seed, 1);
    assert_eq!(cfg.max_iterations, 20);
    assert_eq!(cfg.solver_tolerance, 1e-5);
    assert_eq!(cfg.max_solver_iters, 100);
    assert_eq!(cfg.spreading_threshold, 0.95);
    assert_eq!(cfg.alpha, 0.1);
    assert_eq!(cfg.beta, 1.0);
}

// =====================================================================
// Integration tests: full HeAP run
// =====================================================================

#[test]
fn full_heap_placement_2_cells() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).expect("HeAP placement should succeed");
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() {
            continue;
        }
        assert!(cell_view.bel_id().is_some());
    }
}

#[test]
fn full_heap_placement_4_cells() {
    let mut ctx = common::make_context_with_cells(4);
    let cfg = PlacerHeapCfg {
        seed: 123,
        max_iterations: 10,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).expect("HeAP placement should succeed");
    let mut used_bels = std::collections::HashSet::new();
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() {
            continue;
        }
        let bel = cell_view.bel_id().expect("alive cell should be placed");
        assert!(used_bels.insert(bel));
    }
}

#[test]
fn full_heap_placement_single_cell() {
    let mut ctx = common::make_context_with_cells(1);
    let cfg = PlacerHeapCfg {
        seed: 1,
        max_iterations: 3,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).expect("HeAP placement should succeed");
    let cell_name = ctx.id("cell_0");
    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    assert!(ctx.cell(cell_idx).bel_id().is_some());
}

#[test]
fn full_heap_deterministic() {
    let cfg = PlacerHeapCfg {
        seed: 99,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };

    let mut ctx1 = common::make_context_with_cells(3);
    place_heap(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = common::make_context_with_cells(3);
    place_heap(&mut ctx2, &cfg).expect("run 2");

    for c1 in ctx1.cells() {
        if !c1.is_alive() {
            continue;
        }
        let c2 = ctx2.cell(c1.id());
        assert_eq!(
            c1.bel_id(),
            c2.bel_id(),
            "cell {} placed differently",
            c1.id().raw()
        );
    }
}

#[test]
fn heap_no_moveable_cells_is_ok() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cfg = PlacerHeapCfg::default();
    place_heap(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn place_heap_via_trait() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    PlacerHeap
        .place(&mut ctx, &cfg)
        .expect("trait dispatch should work");
}

const LUT_COUNT: usize = 32;

fn lut_design(
    lock_every: usize,
    free_strength: Option<PlaceStrength>,
) -> (Context, Vec<(CellId, BelId)>) {
    let mut ctx = common::make_example_context();
    let cells: Vec<CellId> = (0..LUT_COUNT)
        .map(|i| common::add_cell(&mut ctx, &format!("lut{i}"), "LUT6"))
        .collect();
    for (i, &cell) in cells.iter().enumerate() {
        let net = common::add_net(&mut ctx, &format!("n{i}"));
        common::drive(&mut ctx, net, cell, "F");
        common::sink(&mut ctx, net, cells[(i + 1) % LUT_COUNT], "I[0]");
        common::sink(&mut ctx, net, cells[(i + 5) % LUT_COUNT], "I[1]");
    }
    ctx.populate_bel_buckets();
    let lut6 = ctx.id("LUT6");
    let bels: Vec<BelId> = ctx.bels_for_bucket(lut6).map(|bel| bel.id()).collect();
    assert!(bels.len() >= 4 * LUT_COUNT, "the fixture spreads the locks");
    let mut locked = Vec::new();
    for (i, &cell) in cells.iter().enumerate() {
        let (bel, strength) = match i % lock_every == 0 {
            true => (bels[bels.len() - 1 - 3 * i], Some(PlaceStrength::User)),
            false => (bels[i], free_strength),
        };
        if let Some(strength) = strength {
            assert!(ctx.bind_bel(bel, cell, strength));
        }
        if strength == Some(PlaceStrength::User) {
            locked.push((cell, bel));
        }
    }
    (ctx, locked)
}

fn assert_user_locked(ctx: &Context, locked: &[(CellId, BelId)]) {
    for &(cell, bel) in locked {
        let info = ctx.design.cell(cell);
        assert_eq!(
            (info.bel, info.bel_strength),
            (Some(bel), PlaceStrength::User)
        );
    }
}

#[rstest]
#[case::every_second_cell(2)]
#[case::every_seventh_cell(7)]
fn heap_places_every_unbound_cell_around_user_locks(#[case] lock_every: usize) {
    let (mut ctx, locked) = lut_design(lock_every, None);
    PlacerHeap
        .place(&mut ctx, &PlacerHeapCfg::default())
        .expect("HeAP places the unbound cells");
    PlacerPipeline::validate(&ctx).expect("every cell is placed");
    assert_user_locked(&ctx, &locked);
}

#[rstest]
#[case::every_second_cell(2)]
#[case::every_seventh_cell(7)]
fn refine_never_moves_user_locked_cells(#[case] lock_every: usize) {
    let (mut ctx, locked) = lut_design(lock_every, Some(PlaceStrength::Weak));
    let stats = refine_placement(&mut ctx, &RefineCfg::default()).expect("refinement runs");
    assert_eq!(stats.autoplaced, LUT_COUNT - locked.len());
    assert!(stats.moves_accepted > 0, "refinement moved a free cell");
    assert_user_locked(&ctx, &locked);
}

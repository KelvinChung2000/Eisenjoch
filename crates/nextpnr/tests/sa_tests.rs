mod common;

use nextpnr::placer::sa::{place_sa, PlacerSaCfg};
use nextpnr::placer::{Placer, PlacerSa};

// =====================================================================
// Temperature cooling tests
// =====================================================================

#[test]
fn cooling_rate_reduces_temperature() {
    let cfg = PlacerSaCfg::default();
    let mut temp = 1.0;
    let initial = temp;
    for _ in 0..100 {
        temp *= cfg.cooling_rate;
    }
    assert!(temp < initial);
    assert!(temp > 0.0);
}

#[test]
fn temperature_converges_to_zero() {
    let cfg = PlacerSaCfg::default();
    let mut temp = 1000.0;
    let mut iters = 0;
    while temp > cfg.min_temp {
        temp *= cfg.cooling_rate;
        iters += 1;
        assert!(iters < 100_000, "temperature did not converge");
    }
}

#[test]
fn default_config_values() {
    let cfg = PlacerSaCfg::default();
    assert_eq!(cfg.seed, 1);
    assert_eq!(cfg.cooling_rate, 0.995);
    assert_eq!(cfg.inner_iters_per_cell, 10);
    assert_eq!(cfg.initial_temp_factor, 1.5);
    assert_eq!(cfg.min_temp, 1e-6);
    assert_eq!(cfg.timing_weight, 0.5);
    assert!(cfg.slack_redistribution);
    assert_eq!(cfg.congestion_weight, 0.0);
}

// =====================================================================
// Integration test: full SA run
// =====================================================================

#[test]
fn full_sa_placement_2_cells() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerSaCfg {
        seed: 42,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        assert!(cell_view.bel_id().is_some());
    }
}

#[test]
fn full_sa_placement_4_cells() {
    let mut ctx = common::make_context_with_cells(4);
    let cfg = PlacerSaCfg {
        seed: 123,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    let mut used_bels = std::collections::HashSet::new();
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        let bel = cell_view.bel_id().expect("alive cell should be placed");
        assert!(used_bels.insert(bel));
    }
}

#[test]
fn full_sa_placement_single_cell() {
    let mut ctx = common::make_context_with_cells(1);
    let cfg = PlacerSaCfg {
        seed: 1,
        cooling_rate: 0.9,
        inner_iters_per_cell: 2,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    let cell_name = ctx.id("cell_0");
    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    assert!(ctx.cell(cell_idx).bel_id().is_some());
}

#[test]
fn full_sa_deterministic() {
    let cfg = PlacerSaCfg {
        seed: 99,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };

    let mut ctx1 = common::make_context_with_cells(3);
    place_sa(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = common::make_context_with_cells(3);
    place_sa(&mut ctx2, &cfg).expect("run 2");

    for c1 in ctx1.cells() {
        if !c1.is_alive() { continue; }
        let c2 = ctx2.cell(c1.id());
        assert_eq!(c1.bel_id(), c2.bel_id(), "cell {} placed differently", c1.id().raw());
    }
}

#[test]
fn sa_no_moveable_cells_is_ok() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cfg = PlacerSaCfg::default();
    place_sa(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn place_sa_via_trait() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerSaCfg {
        seed: 42,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    PlacerSa.place(&mut ctx, &cfg).expect("trait dispatch should work");
}

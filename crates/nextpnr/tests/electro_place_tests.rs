mod common;

use nextpnr::placer::electro_place::{place_electro, ElectroPlaceCfg};
use nextpnr::placer::{Placer, PlacerElectro};

#[test]
fn place_electro_2_cells() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = ElectroPlaceCfg {
        seed: 42,
        max_iters: 10,
        legalize_interval: 2,
        ..ElectroPlaceCfg::default()
    };
    place_electro(&mut ctx, &cfg).expect("ElectroPlace should succeed");
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        assert!(cell_view.bel_id().is_some());
    }
}

#[test]
fn place_electro_4_cells() {
    let mut ctx = common::make_context_with_cells(4);
    let cfg = ElectroPlaceCfg {
        seed: 123,
        max_iters: 10,
        legalize_interval: 2,
        ..ElectroPlaceCfg::default()
    };
    place_electro(&mut ctx, &cfg).expect("ElectroPlace should succeed");
    let mut used_bels = std::collections::HashSet::new();
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        let bel = cell_view.bel_id().expect("alive cell should be placed");
        assert!(used_bels.insert(bel));
    }
}

#[test]
fn place_electro_single_cell() {
    let mut ctx = common::make_context_with_cells(1);
    let cfg = ElectroPlaceCfg {
        seed: 1,
        max_iters: 5,
        legalize_interval: 2,
        ..ElectroPlaceCfg::default()
    };
    place_electro(&mut ctx, &cfg).expect("ElectroPlace should succeed");
    let cell_name = ctx.id("cell_0");
    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    assert!(ctx.cell(cell_idx).bel_id().is_some());
}

#[test]
fn place_electro_deterministic() {
    let cfg = ElectroPlaceCfg {
        seed: 99,
        max_iters: 10,
        legalize_interval: 2,
        ..ElectroPlaceCfg::default()
    };

    let mut ctx1 = common::make_context_with_cells(3);
    place_electro(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = common::make_context_with_cells(3);
    place_electro(&mut ctx2, &cfg).expect("run 2");

    for c1 in ctx1.cells() {
        if !c1.is_alive() { continue; }
        let c2 = ctx2.cell(c1.id());
        assert_eq!(c1.bel_id(), c2.bel_id(), "cell {} placed differently", c1.id().raw());
    }
}

#[test]
fn place_electro_via_trait() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = ElectroPlaceCfg {
        seed: 42,
        max_iters: 10,
        legalize_interval: 2,
        ..ElectroPlaceCfg::default()
    };
    PlacerElectro.place(&mut ctx, &cfg).expect("trait dispatch should work");
}

#[test]
fn electro_no_cells_is_ok() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cfg = ElectroPlaceCfg::default();
    place_electro(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn default_config_values() {
    let cfg = ElectroPlaceCfg::default();
    assert_eq!(cfg.seed, 1);
    assert_eq!(cfg.wl_coeff, 0.5);
    assert_eq!(cfg.target_util, 0.7);
    assert!(!cfg.timing_driven);
    assert_eq!(cfg.target_density, 1.0);
    assert_eq!(cfg.timing_weight, 0.0);
    assert_eq!(cfg.nesterov_step_size, 0.1);
    assert_eq!(cfg.max_iters, 500);
    assert_eq!(cfg.legalize_interval, 5);
}

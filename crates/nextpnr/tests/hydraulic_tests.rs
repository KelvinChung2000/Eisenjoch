mod common;

use nextpnr::placer::hydraulic_place::{place_hydraulic, HydraulicPlacerCfg};
use nextpnr::placer::{Placer, PlacerHydraulic};

#[test]
fn place_hydraulic_2_cells() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = HydraulicPlacerCfg {
        seed: 42,
        max_outer_iters: 10,
        report_interval: 2,
        ..HydraulicPlacerCfg::default()
    };
    place_hydraulic(&mut ctx, &cfg).expect("Hydraulic placement should succeed");
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() {
            continue;
        }
        assert!(cell_view.bel_id().is_some());
    }
}

#[test]
fn place_hydraulic_4_cells() {
    let mut ctx = common::make_context_with_cells(4);
    let cfg = HydraulicPlacerCfg {
        seed: 123,
        max_outer_iters: 10,
        report_interval: 2,
        ..HydraulicPlacerCfg::default()
    };
    place_hydraulic(&mut ctx, &cfg).expect("Hydraulic placement should succeed");
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
fn place_hydraulic_single_cell() {
    let mut ctx = common::make_context_with_cells(1);
    let cfg = HydraulicPlacerCfg {
        seed: 1,
        max_outer_iters: 5,
        report_interval: 2,
        ..HydraulicPlacerCfg::default()
    };
    place_hydraulic(&mut ctx, &cfg).expect("Hydraulic placement should succeed");
    let cell_name = ctx.id("cell_0");
    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    assert!(ctx.cell(cell_idx).bel_id().is_some());
}

#[test]
fn place_hydraulic_deterministic() {
    let cfg = HydraulicPlacerCfg {
        seed: 99,
        max_outer_iters: 10,
        report_interval: 2,
        ..HydraulicPlacerCfg::default()
    };

    let mut ctx1 = common::make_context_with_cells(3);
    place_hydraulic(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = common::make_context_with_cells(3);
    place_hydraulic(&mut ctx2, &cfg).expect("run 2");

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
fn place_hydraulic_via_trait() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = HydraulicPlacerCfg {
        seed: 42,
        max_outer_iters: 10,
        report_interval: 2,
        ..HydraulicPlacerCfg::default()
    };
    PlacerHydraulic
        .place(&mut ctx, &cfg)
        .expect("trait dispatch should work");
}

#[test]
fn hydraulic_no_cells_is_ok() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cfg = HydraulicPlacerCfg::default();
    place_hydraulic(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn default_config_values() {
    let cfg = HydraulicPlacerCfg::default();
    assert_eq!(cfg.seed, 1);
    assert_eq!(cfg.turbulence_beta, 4.0);
    assert_eq!(cfg.newton_iters, 2);
    assert_eq!(cfg.cg_max_iters, 500);
    assert_eq!(cfg.cg_tolerance, 1e-6);
    assert_eq!(cfg.cfl_number, 0.5);
    assert_eq!(cfg.max_outer_iters, 500);
    assert_eq!(cfg.report_interval, 5);
    assert_eq!(cfg.timing_weight, 0.0);
    assert_eq!(cfg.gas_temperature, 1.0);
    assert_eq!(cfg.lap_max_cells, 10000);
    assert_eq!(cfg.star_weight, 1.0);
    assert_eq!(cfg.pressure_weight_start, 0.0);
    assert_eq!(cfg.pressure_weight_end, 2.0);
    assert_eq!(cfg.io_boost, 4.0);
    assert_eq!(cfg.nesterov_step_size, 0.1);
    assert_eq!(cfg.momentum, None);
    assert_eq!(cfg.legalize_interval, 5);
    assert_eq!(cfg.wl_coeff, 0.5);
    assert_eq!(cfg.enable_expanding_box, true);
    assert_eq!(cfg.pump_gain, 10.0);
}

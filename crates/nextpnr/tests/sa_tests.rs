use nextpnr::placer::sa::{place_sa, PlacerSaCfg};
use nextpnr::placer::{Placer, PlacerSa};
use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::context::Context;
use nextpnr::netlist::PortRef;
use nextpnr::types::PortType;

fn make_context() -> Context {
    let chipdb = make_test_chipdb();
    Context::new(chipdb)
}

fn make_context_with_cells(n: usize) -> Context {
    assert!(n <= 4, "synthetic chipdb only has 4 BELs");
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let cell_type = ctx.id("LUT4");
    let mut cell_names = Vec::new();

    for i in 0..n {
        let name = ctx.id(&format!("cell_{}", i));
        ctx.design_mut().add_cell(name, cell_type);
        cell_names.push(name);
    }

    if n >= 2 {
        let net_name = ctx.id("net_0");
        let net_idx = ctx.design_mut().add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        let cell0_idx = ctx.design().cell_by_name(cell_names[0]).unwrap();
        ctx.design_mut().cell_mut(cell0_idx).add_port(q_port, PortType::Out);
        ctx.design_mut().cell_mut(cell0_idx).port_mut(q_port).unwrap().net = Some(net_idx);

        ctx.design_mut().net_mut(net_idx).driver = PortRef {
            cell: Some(cell0_idx),
            port: q_port,
            budget: 0,
        };

        for i in 1..n {
            let cell_idx = ctx.design().cell_by_name(cell_names[i]).unwrap();
            ctx.design_mut().cell_mut(cell_idx).add_port(a_port, PortType::In);
            ctx.design_mut().cell_mut(cell_idx).port_mut(a_port).unwrap().net = Some(net_idx);

            let user_idx = ctx.design().net(net_idx).users.len() as u32;
            ctx.design_mut().cell_mut(cell_idx).port_mut(a_port).unwrap().user_idx = Some(user_idx);
            ctx.design_mut().net_mut(net_idx).users.push(PortRef {
                cell: Some(cell_idx),
                port: a_port,
                budget: 0,
            });
        }
    }

    ctx
}

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
}

// =====================================================================
// Integration test: full SA run
// =====================================================================

#[test]
fn full_sa_placement_2_cells() {
    let mut ctx = make_context_with_cells(2);
    let cfg = PlacerSaCfg {
        seed: 42,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    for (_cell_idx, cell) in ctx.design().iter_alive_cells() {
        assert!(cell.bel.is_some());
    }
}

#[test]
fn full_sa_placement_4_cells() {
    let mut ctx = make_context_with_cells(4);
    let cfg = PlacerSaCfg {
        seed: 123,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    let mut used_bels = std::collections::HashSet::new();
    for (_cell_idx, cell) in ctx.design().iter_alive_cells() {
        let bel = cell.bel.expect("alive cell should be placed");
        assert!(used_bels.insert(bel));
    }
}

#[test]
fn full_sa_placement_single_cell() {
    let mut ctx = make_context_with_cells(1);
    let cfg = PlacerSaCfg {
        seed: 1,
        cooling_rate: 0.9,
        inner_iters_per_cell: 2,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    place_sa(&mut ctx, &cfg).expect("SA placement should succeed");
    let cell_name = ctx.id("cell_0");
    let cell_idx = ctx.design().cell_by_name(cell_name).unwrap();
    assert!(ctx.design().cell(cell_idx).bel.is_some());
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

    let mut ctx1 = make_context_with_cells(3);
    place_sa(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = make_context_with_cells(3);
    place_sa(&mut ctx2, &cfg).expect("run 2");

    for (cidx, c1) in ctx1.design().iter_alive_cells() {
        let c2 = ctx2.design().cell(cidx);
        assert_eq!(c1.bel, c2.bel, "cell {} placed differently", cidx.raw());
    }
}

#[test]
fn sa_no_moveable_cells_is_ok() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    let cfg = PlacerSaCfg::default();
    place_sa(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn place_sa_via_trait() {
    let mut ctx = make_context_with_cells(2);
    let cfg = PlacerSaCfg {
        seed: 42,
        cooling_rate: 0.9,
        inner_iters_per_cell: 5,
        min_temp: 0.01,
        ..PlacerSaCfg::default()
    };
    PlacerSa.place(&mut ctx, &cfg).expect("trait dispatch should work");
}

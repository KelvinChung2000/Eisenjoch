use nextpnr::placer::heap::{place_heap, PlacerHeapCfg};
use nextpnr::placer::{Placer, PlacerHeap};
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
        ctx.design.add_cell(name, cell_type);
        cell_names.push(name);
    }

    if n >= 2 {
        let net_name = ctx.id("net_0");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        let cell0_idx = ctx.design.cell_by_name(cell_names[0]).unwrap();
        ctx.design.cell_edit(cell0_idx).add_port(q_port, PortType::Out);
        ctx.design.cell_edit(cell0_idx).set_port_net(q_port, Some(net_idx), None);

        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(cell0_idx),
            port: q_port,
            budget: 0,
        });

        for i in 1..n {
            let cell_idx = ctx.design.cell_by_name(cell_names[i]).unwrap();
            ctx.design.cell_edit(cell_idx).add_port(a_port, PortType::In);

            let user_idx = ctx.design.net_edit(net_idx).add_user_raw(PortRef {
                cell: Some(cell_idx),
                port: a_port,
                budget: 0,
            });
            ctx.design.cell_edit(cell_idx).set_port_net(a_port, Some(net_idx), Some(user_idx));
        }
    }

    ctx
}

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
    let mut ctx = make_context_with_cells(2);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).expect("HeAP placement should succeed");
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        assert!(cell_view.bel_id().is_some());
    }
}

#[test]
fn full_heap_placement_4_cells() {
    let mut ctx = make_context_with_cells(4);
    let cfg = PlacerHeapCfg {
        seed: 123,
        max_iterations: 10,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).expect("HeAP placement should succeed");
    let mut used_bels = std::collections::HashSet::new();
    for cell_view in ctx.cells() {
        if !cell_view.is_alive() { continue; }
        let bel = cell_view.bel_id().expect("alive cell should be placed");
        assert!(used_bels.insert(bel));
    }
}

#[test]
fn full_heap_placement_single_cell() {
    let mut ctx = make_context_with_cells(1);
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

    let mut ctx1 = make_context_with_cells(3);
    place_heap(&mut ctx1, &cfg).expect("run 1");

    let mut ctx2 = make_context_with_cells(3);
    place_heap(&mut ctx2, &cfg).expect("run 2");

    for c1 in ctx1.cells() {
        if !c1.is_alive() { continue; }
        let c2 = ctx2.cell(c1.id());
        assert_eq!(c1.bel_id(), c2.bel_id(), "cell {} placed differently", c1.id().raw());
    }
}

#[test]
fn heap_no_moveable_cells_is_ok() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    let cfg = PlacerHeapCfg::default();
    place_heap(&mut ctx, &cfg).expect("no cells should be OK");
}

#[test]
fn place_heap_via_trait() {
    let mut ctx = make_context_with_cells(2);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    PlacerHeap.place(&mut ctx, &cfg).expect("trait dispatch should work");
}

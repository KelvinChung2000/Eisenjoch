mod common;

use nextpnr::chipdb::{BelId, PipId, WireId};
use nextpnr::common::PlaceStrength;
use nextpnr::netlist::NetId;
use nextpnr::netlist::PortType;
use nextpnr::router::common::{
    apply_route_plan, bind_route, collect_routable_nets, find_congested_wires, unroute_net,
    RoutePlan, SinkRoute,
};
use nextpnr::router::router1::{
    astar_route, compute_route_r1, find_congested_nets, route_net, QueueEntry, Router1Cfg,
    Router1State,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;

/// Helper: create an FxHashSet from a slice of WireIds.
fn wire_set(wires: &[WireId]) -> FxHashSet<WireId> {
    wires.iter().copied().collect()
}

#[test]
fn queue_entry_min_heap_ordering() {
    let mut heap = BinaryHeap::new();
    heap.push(QueueEntry {
        wire: WireId::new(0, 0),
        cost: 10,
        estimate: 50,
    });
    heap.push(QueueEntry {
        wire: WireId::new(0, 1),
        cost: 5,
        estimate: 20,
    });
    heap.push(QueueEntry {
        wire: WireId::new(1, 0),
        cost: 8,
        estimate: 35,
    });
    assert_eq!(heap.pop().unwrap().estimate, 20);
    assert_eq!(heap.pop().unwrap().estimate, 35);
    assert_eq!(heap.pop().unwrap().estimate, 50);
}

#[test]
fn queue_entry_tiebreak_by_cost() {
    let mut heap = BinaryHeap::new();
    heap.push(QueueEntry {
        wire: WireId::new(0, 0),
        cost: 30,
        estimate: 50,
    });
    heap.push(QueueEntry {
        wire: WireId::new(0, 1),
        cost: 10,
        estimate: 50,
    });
    assert_eq!(heap.pop().unwrap().cost, 10);
}

#[test]
fn astar_same_wire_returns_empty_path() {
    let ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let penalty = FxHashMap::default();
    let path = astar_route(&ctx, &wire_set(&[wire]), wire, &penalty, None);
    assert!(path.is_some());
    assert!(path.unwrap().is_empty());
}

#[test]
fn astar_single_pip_path() {
    let ctx = common::make_context();
    let src = WireId::new(0, 0);
    let dst = WireId::new(0, 1);
    let penalty = FxHashMap::default();
    let path = astar_route(&ctx, &wire_set(&[src]), dst, &penalty, None).unwrap();
    assert_eq!(path, vec![PipId::new(0, 0)]);
}

#[test]
fn astar_verifies_pip_connectivity() {
    let ctx = common::make_context();
    let src = WireId::new(0, 0);
    let dst = WireId::new(0, 1);
    let penalty = FxHashMap::default();
    let path = astar_route(&ctx, &wire_set(&[src]), dst, &penalty, None).unwrap();
    let pip = path[0];
    assert_eq!(ctx.pip(pip).src_wire().id(), src);
    assert_eq!(ctx.pip(pip).dst_wire().id(), dst);
}

#[test]
fn astar_no_path_returns_none() {
    let ctx = common::make_context();
    assert!(astar_route(
        &ctx,
        &wire_set(&[WireId::new(0, 1)]),
        WireId::new(0, 0),
        &FxHashMap::default(),
        None,
    )
    .is_none());
}

#[test]
fn astar_cross_tile_no_path() {
    let ctx = common::make_context();
    assert!(astar_route(
        &ctx,
        &wire_set(&[WireId::new(0, 0)]),
        WireId::new(1, 0),
        &FxHashMap::default(),
        None,
    )
    .is_none());
}

#[test]
fn astar_with_penalty_still_finds_path() {
    let ctx = common::make_context();
    let src = WireId::new(0, 0);
    let dst = WireId::new(0, 1);
    let mut penalty = FxHashMap::default();
    penalty.insert(dst, 1000);
    assert_eq!(astar_route(&ctx, &wire_set(&[src]), dst, &penalty, None).unwrap().len(), 1);
}

#[test]
fn astar_multi_source_picks_closest() {
    let ctx = common::make_context();
    let path = astar_route(
        &ctx,
        &wire_set(&[WireId::new(0, 0), WireId::new(1, 0)]),
        WireId::new(0, 1),
        &FxHashMap::default(),
        None,
    )
    .unwrap();
    assert_eq!(path, vec![PipId::new(0, 0)]);
}

#[test]
fn astar_empty_sources_returns_none() {
    let ctx = common::make_context();
    assert!(astar_route(&ctx, &wire_set(&[]), WireId::new(0, 1), &FxHashMap::default(), None).is_none());
}

#[test]
fn bind_route_records_wires_and_pips() {
    let mut ctx = common::make_context();
    let net_name = ctx.id("net_bind");
    let net_idx = ctx.design.add_net(net_name);
    let pip = PipId::new(0, 0);
    let dst_wire = ctx.pip(pip).dst_wire().id();
    bind_route(&mut ctx, net_idx, &[pip]);
    assert!(!ctx.wire(dst_wire).is_available());
    assert_eq!(
        ctx.wire(dst_wire).bound_net().map(|n| n.name_id()),
        Some(net_name)
    );
    assert!(!ctx.pip(pip).is_available());
    assert!(ctx.net(net_idx).wires().contains_key(&dst_wire));
    assert_eq!(ctx.net(net_idx).wires()[&dst_wire].pip, Some(pip));
}

#[test]
fn bind_empty_route() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_empty"));
    bind_route(&mut ctx, net_idx, &[]);
    assert!(ctx.net(net_idx).wires().is_empty());
}

#[test]
fn unroute_clears_net_wires() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_rip"));
    let wire = WireId::new(0, 1);
    let pip = PipId::new(0, 0);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Strong);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_idx)
        .add_wire(wire, Some(pip), PlaceStrength::Strong);
    unroute_net(&mut ctx, net_idx);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.pip(pip).is_available());
    assert!(ctx.net(net_idx).wires().is_empty());
}

#[test]
fn unroute_handles_invalid_pip() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_src"));
    let wire = WireId::new(0, 0);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    unroute_net(&mut ctx, net_idx);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.net(net_idx).wires().is_empty());
}

#[test]
fn unroute_multiple_wires() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_multi"));
    let wire0 = WireId::new(0, 0);
    let wire1 = WireId::new(0, 1);
    let pip = PipId::new(0, 0);
    ctx.bind_wire(wire0, net_idx, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_idx)
        .add_wire(wire0, None, PlaceStrength::Strong);
    ctx.bind_wire(wire1, net_idx, PlaceStrength::Strong);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_idx)
        .add_wire(wire1, Some(pip), PlaceStrength::Strong);
    unroute_net(&mut ctx, net_idx);
    assert!(ctx.wire(wire0).is_available());
    assert!(ctx.wire(wire1).is_available());
    assert!(ctx.pip(pip).is_available());
    assert!(ctx.net(net_idx).wires().is_empty());
}

#[test]
fn no_congestion_with_no_nets() {
    let ctx = common::make_context();
    assert!(find_congested_wires(&ctx).is_empty());
    assert!(find_congested_nets(&ctx).is_empty());
}

#[test]
fn no_congestion_with_single_net() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_a"));
    ctx.design
        .net_edit(net_idx)
        .add_wire(WireId::new(0, 0), None, PlaceStrength::Strong);
    assert!(find_congested_wires(&ctx).is_empty());
}

#[test]
fn congestion_detected_with_shared_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let net_a_idx = ctx.design.add_net(ctx.id("net_a"));
    ctx.design
        .net_edit(net_a_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    let net_b_idx = ctx.design.add_net(ctx.id("net_b"));
    ctx.design
        .net_edit(net_b_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    let congested = find_congested_wires(&ctx);
    assert_eq!(congested, vec![wire]);
}

#[test]
fn congested_nets_identified() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let net_a_idx = ctx.design.add_net(ctx.id("net_a"));
    ctx.design
        .net_edit(net_a_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    let net_b_idx = ctx.design.add_net(ctx.id("net_b"));
    ctx.design
        .net_edit(net_b_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    let net_set: FxHashSet<NetId> = find_congested_nets(&ctx).into_iter().collect();
    assert!(net_set.contains(&net_a_idx));
    assert!(net_set.contains(&net_b_idx));
}

#[test]
fn non_shared_wires_not_congested() {
    let mut ctx = common::make_context();
    let net_a_idx = ctx.design.add_net(ctx.id("net_a"));
    ctx.design
        .net_edit(net_a_idx)
        .add_wire(WireId::new(0, 0), None, PlaceStrength::Strong);
    let net_b_idx = ctx.design.add_net(ctx.id("net_b"));
    ctx.design
        .net_edit(net_b_idx)
        .add_wire(WireId::new(1, 0), None, PlaceStrength::Strong);
    assert!(find_congested_wires(&ctx).is_empty());
    assert!(find_congested_nets(&ctx).is_empty());
}

#[test]
fn route_net_same_pin_driver_and_sink() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port_name = ctx.id("I0");
    let cell_idx = ctx.design.add_cell(ctx.id("cell_a"), lut_type);
    ctx.design
        .cell_edit(cell_idx)
        .add_port(port_name, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("net_self"));
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port_name);
    ctx.design.net_edit(net_idx).add_user(cell_idx, port_name);
    assert!(route_net(&mut ctx, net_idx, &FxHashMap::default()).is_ok());
}

#[test]
fn route_net_cross_tile_fails_in_minimal_chipdb() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");
    let driver_idx = ctx.design.add_cell(ctx.id("driver"), lut_type);
    ctx.design
        .cell_edit(driver_idx)
        .add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), driver_idx, PlaceStrength::Placer);
    let sink_idx = ctx.design.add_cell(ctx.id("sink"), lut_type);
    ctx.design.cell_edit(sink_idx).add_port(port, PortType::In);
    ctx.bind_bel(BelId::new(1, 0), sink_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("test_net"));
    ctx.design.net_edit(net_idx).set_driver(driver_idx, port);
    ctx.design.net_edit(net_idx).add_user(sink_idx, port);
    assert!(route_net(&mut ctx, net_idx, &FxHashMap::default()).is_err());
}

#[test]
fn bel_pin_wire_valid() {
    let ctx = common::make_context();
    assert_eq!(
        ctx.bel(BelId::new(0, 0))
            .pin_wire(ctx.id("I0"))
            .map(|w| w.id()),
        Some(WireId::new(0, 0))
    );
}

#[test]
fn bel_pin_wire_different_tiles() {
    let ctx = common::make_context();
    for tile in 0..4 {
        assert_eq!(
            ctx.bel(BelId::new(tile, 0))
                .pin_wire(ctx.id("I0"))
                .map(|w| w.id()),
            Some(WireId::new(tile, 0))
        );
    }
}

#[test]
fn bel_pin_wire_invalid_port() {
    let ctx = common::make_context();
    assert!(ctx
        .bel(BelId::new(0, 0))
        .pin_wire(ctx.id("NONEXISTENT"))
        .is_none());
}

#[test]
fn wire_penalty_accumulates() {
    let cfg = Router1Cfg::default();
    let mut state = Router1State::new();
    let wire = WireId::new(0, 0);
    *state.wire_penalty.entry(wire).or_insert(0) += cfg.rip_up_penalty;
    *state.wire_penalty.entry(wire).or_insert(0) += cfg.rip_up_penalty;
    assert_eq!(state.wire_penalty[&wire], 20);
}

#[test]
fn collect_routable_nets_empty_design() {
    let ctx = common::make_context();
    assert!(collect_routable_nets(&ctx).is_empty());
}

#[test]
fn collect_routable_nets_skips_no_driver() {
    let mut ctx = common::make_context();
    ctx.design.add_net(ctx.id("no_driver"));
    assert!(collect_routable_nets(&ctx).is_empty());
}

#[test]
fn collect_routable_nets_finds_valid_net() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");
    let cell_idx = ctx.design.add_cell(ctx.id("cell"), lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    let net_idx = ctx.design.add_net(ctx.id("routable"));
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);
    ctx.design.net_edit(net_idx).add_user(cell_idx, port);
    assert_eq!(collect_routable_nets(&ctx), vec![net_idx]);
}

#[test]
fn compute_route_produces_valid_plan() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");
    let cell_idx = ctx.design.add_cell(ctx.id("cell_a"), lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("test_net"));
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);
    ctx.design.net_edit(net_idx).add_user(cell_idx, port);

    let penalty = FxHashMap::default();
    let plan = compute_route_r1(&ctx, net_idx, &penalty, 0).unwrap();
    assert_eq!(plan.net, net_idx);
    assert!(plan.source_wire.is_valid());
    // Driver and sink use the same wire, so sink_routes should have empty pips
    assert!(!plan.sink_routes.is_empty());
    for sr in &plan.sink_routes {
        assert!(sr.pips.is_empty(), "same-pin route should have no PIPs");
    }
}

#[test]
fn apply_route_plan_binds_wires() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_plan"));
    let src_wire = WireId::new(0, 0);
    let pip = PipId::new(0, 0);
    let dst_wire = ctx.pip(pip).dst_wire().id();

    let plan = RoutePlan {
        net: net_idx,
        source_wire: src_wire,
        sink_routes: vec![SinkRoute {
            sink_wire: dst_wire,
            pips: vec![pip],
        }],
    };

    apply_route_plan(&mut ctx, &plan);

    // Source wire should be bound
    assert!(!ctx.wire(src_wire).is_available());
    // Destination wire should be bound
    assert!(!ctx.wire(dst_wire).is_available());
    // PIP should be bound
    assert!(!ctx.pip(pip).is_available());
    // Net should have both wires
    assert!(ctx.net(net_idx).wires().contains_key(&src_wire));
    assert!(ctx.net(net_idx).wires().contains_key(&dst_wire));
}

#[test]
fn compute_then_apply_matches_route_net() {
    // Compare compute+apply vs route_net for the same setup
    let mut ctx1 = common::make_context();
    let mut ctx2 = common::make_context();

    // Setup: driver BEL(0,0), sink BEL(0,0) same pin
    for ctx in [&mut ctx1, &mut ctx2] {
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");
        let cell_idx = ctx.design.add_cell(ctx.id("cell_a"), lut_type);
        ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
        let net_idx = ctx.design.add_net(ctx.id("test_net"));
        ctx.design.net_edit(net_idx).set_driver(cell_idx, port);
        ctx.design.net_edit(net_idx).add_user(cell_idx, port);
    }

    let penalty = FxHashMap::default();
    let net_idx = ctx1.design.net_by_name(ctx1.id("test_net")).unwrap();

    // Method 1: compute + apply
    let plan = compute_route_r1(&ctx1, net_idx, &penalty, 0).unwrap();
    if plan.source_wire.is_valid() {
        apply_route_plan(&mut ctx1, &plan);
    }

    // Method 2: route_net
    let net_idx2 = ctx2.design.net_by_name(ctx2.id("test_net")).unwrap();
    route_net(&mut ctx2, net_idx2, &penalty).unwrap();

    // Both should have the same wires on the net
    let wires1: FxHashSet<WireId> = ctx1.net(net_idx).wire_ids().collect();
    let wires2: FxHashSet<WireId> = ctx2.net(net_idx2).wire_ids().collect();
    assert_eq!(wires1, wires2);
}

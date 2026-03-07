mod common;

use nextpnr::netlist::NetId;
use nextpnr::router::router2::{
    astar_route_r2, compute_bbox, BoundingBox, R2QueueEntry, Router2Cfg, Router2State,
};
use nextpnr::types::{BelId, PipId, PlaceStrength, PortType, WireId};
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;

#[test]
fn bbox_contains_within() {
    let bb = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 3,
    };
    assert!(bb.contains(0, 0));
    assert!(bb.contains(1, 2));
    assert!(bb.contains(3, 3));
}

#[test]
fn bbox_contains_boundary() {
    let bb = BoundingBox {
        x0: 1,
        y0: 1,
        x1: 5,
        y1: 5,
    };
    assert!(bb.contains(1, 1));
    assert!(bb.contains(5, 1));
    assert!(bb.contains(1, 5));
    assert!(bb.contains(5, 5));
}

#[test]
fn bbox_excludes_outside() {
    let bb = BoundingBox {
        x0: 1,
        y0: 1,
        x1: 3,
        y1: 3,
    };
    assert!(!bb.contains(0, 0));
    assert!(!bb.contains(4, 2));
    assert!(!bb.contains(2, 4));
}

#[test]
fn bbox_single_point() {
    let bb = BoundingBox {
        x0: 2,
        y0: 3,
        x1: 2,
        y1: 3,
    };
    assert!(bb.contains(2, 3));
    assert!(!bb.contains(1, 3));
}

#[test]
fn compute_bbox_no_placed_cells() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("unplaced_net"));
    let bb = compute_bbox(&ctx, net_idx, 0);
    assert_eq!(bb.x0, 0);
    assert_eq!(bb.y0, 0);
    assert_eq!(bb.x1, ctx.chipdb().width() - 1);
    assert_eq!(bb.y1, ctx.chipdb().height() - 1);
}

#[test]
fn compute_bbox_single_cell() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");
    let cell_idx = ctx.design.add_cell(ctx.id("driver"), lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("test_net"));
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);
    let bb = compute_bbox(&ctx, net_idx, 0);
    assert_eq!((bb.x0, bb.y0, bb.x1, bb.y1), (0, 0, 0, 0));
}

#[test]
fn compute_bbox_with_margin() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");
    let cell_idx = ctx.design.add_cell(ctx.id("driver"), lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("test_net"));
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);
    let bb = compute_bbox(&ctx, net_idx, 1);
    assert_eq!((bb.x0, bb.y0, bb.x1, bb.y1), (0, 0, 1, 1));
}

#[test]
fn wire_cost_base_only() {
    let state = Router2State::new(&Router2Cfg::default());
    assert!((state.wire_cost(WireId::new(0, 0), NetId::from_raw(0)) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn wire_cost_with_congestion() {
    let cfg = Router2Cfg {
        base_cost: 1.0,
        present_cost_multiplier: 2.0,
        initial_present_cost: 1.0,
        history_cost_multiplier: 1.0,
        ..Router2Cfg::default()
    };
    let mut state = Router2State::new(&cfg);
    let wire = WireId::new(0, 0);
    let net_a = NetId::from_raw(0);
    let net_b = NetId::from_raw(1);
    state.wire_usage.insert(wire, 1);
    state.wire_owner.insert(wire, net_a);
    assert!((state.wire_cost(wire, net_a) - 1.0).abs() < f64::EPSILON);
    assert!((state.wire_cost(wire, net_b) - 3.0).abs() < f64::EPSILON);
}

#[test]
fn wire_cost_with_history() {
    let cfg = Router2Cfg {
        base_cost: 1.0,
        present_cost_multiplier: 2.0,
        initial_present_cost: 1.0,
        history_cost_multiplier: 3.0,
        ..Router2Cfg::default()
    };
    let mut state = Router2State::new(&cfg);
    let wire = WireId::new(0, 0);
    state.wire_history.insert(wire, 5.0);
    assert!((state.wire_cost(wire, NetId::from_raw(0)) - 16.0).abs() < f64::EPSILON);
}

#[test]
fn wire_cost_combined() {
    let cfg = Router2Cfg {
        base_cost: 1.0,
        present_cost_multiplier: 2.0,
        initial_present_cost: 1.0,
        history_cost_multiplier: 1.0,
        ..Router2Cfg::default()
    };
    let mut state = Router2State::new(&cfg);
    let wire = WireId::new(0, 0);
    state.wire_usage.insert(wire, 2);
    state.wire_owner.insert(wire, NetId::from_raw(0));
    state.wire_history.insert(wire, 1.0);
    assert!((state.wire_cost(wire, NetId::from_raw(1)) - 6.0).abs() < f64::EPSILON);
}

#[test]
fn update_history_no_congestion() {
    let mut state = Router2State::new(&Router2Cfg::default());
    let wire = WireId::new(0, 0);
    state.wire_usage.insert(wire, 1);
    state.update_history();
    assert!(!state.wire_history.contains_key(&wire));
}

#[test]
fn update_history_with_congestion() {
    let mut state = Router2State::new(&Router2Cfg::default());
    let wire = WireId::new(0, 0);
    state.wire_usage.insert(wire, 3);
    state.update_history();
    assert!((state.wire_history[&wire] - 2.0).abs() < f64::EPSILON);
}

#[test]
fn update_history_accumulates() {
    let mut state = Router2State::new(&Router2Cfg::default());
    let wire = WireId::new(0, 0);
    state.wire_usage.insert(wire, 2);
    state.update_history();
    state.update_history();
    assert!((state.wire_history[&wire] - 2.0).abs() < f64::EPSILON);
}

#[test]
fn update_usage_empty_design() {
    let ctx = common::make_context();
    let mut state = Router2State::new(&Router2Cfg::default());
    state.update_usage(&ctx.design);
    assert!(state.wire_usage.is_empty());
    assert!(state.wire_owner.is_empty());
}

#[test]
fn update_usage_single_net() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_a"));
    let wire = WireId::new(0, 0);
    ctx.design
        .net_edit(net_idx)
        .add_wire(wire, None, PlaceStrength::Strong);
    let mut state = Router2State::new(&Router2Cfg::default());
    state.update_usage(&ctx.design);
    assert_eq!(state.wire_usage[&wire], 1);
    assert_eq!(state.wire_owner[&wire], net_idx);
}

#[test]
fn update_usage_multiple_nets_same_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let net_a = ctx.design.add_net(ctx.id("net_a"));
    let net_b = ctx.design.add_net(ctx.id("net_b"));
    ctx.design
        .net_edit(net_a)
        .add_wire(wire, None, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_b)
        .add_wire(wire, None, PlaceStrength::Strong);
    let mut state = Router2State::new(&Router2Cfg::default());
    state.update_usage(&ctx.design);
    assert_eq!(state.wire_usage[&wire], 2);
    let owner = state.wire_owner[&wire];
    assert!(owner == net_a || owner == net_b);
}

#[test]
fn find_congested_nets_none() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("net_a"));
    ctx.design
        .net_edit(net_idx)
        .add_wire(WireId::new(0, 0), None, PlaceStrength::Strong);
    let mut state = Router2State::new(&Router2Cfg::default());
    state.update_usage(&ctx.design);
    assert!(state.find_congested_nets(&ctx.design).is_empty());
}

#[test]
fn find_congested_nets_shared_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let net_a = ctx.design.add_net(ctx.id("net_a"));
    let net_b = ctx.design.add_net(ctx.id("net_b"));
    ctx.design
        .net_edit(net_a)
        .add_wire(wire, None, PlaceStrength::Strong);
    ctx.design
        .net_edit(net_b)
        .add_wire(wire, None, PlaceStrength::Strong);
    let mut state = Router2State::new(&Router2Cfg::default());
    state.update_usage(&ctx.design);
    let set: FxHashSet<NetId> = state.find_congested_nets(&ctx.design).into_iter().collect();
    assert!(set.contains(&net_a));
    assert!(set.contains(&net_b));
}

#[test]
fn r2_queue_min_heap_ordering() {
    let mut heap = BinaryHeap::new();
    heap.push(R2QueueEntry {
        wire: WireId::new(0, 0),
        cost: 10.0,
        estimate: 50.0,
    });
    heap.push(R2QueueEntry {
        wire: WireId::new(0, 1),
        cost: 5.0,
        estimate: 20.0,
    });
    heap.push(R2QueueEntry {
        wire: WireId::new(1, 0),
        cost: 8.0,
        estimate: 35.0,
    });
    assert!((heap.pop().unwrap().estimate - 20.0).abs() < f64::EPSILON);
    assert!((heap.pop().unwrap().estimate - 35.0).abs() < f64::EPSILON);
    assert!((heap.pop().unwrap().estimate - 50.0).abs() < f64::EPSILON);
}

#[test]
fn r2_queue_tiebreak_by_cost() {
    let mut heap = BinaryHeap::new();
    heap.push(R2QueueEntry {
        wire: WireId::new(0, 0),
        cost: 30.0,
        estimate: 50.0,
    });
    heap.push(R2QueueEntry {
        wire: WireId::new(0, 1),
        cost: 10.0,
        estimate: 50.0,
    });
    assert!((heap.pop().unwrap().cost - 10.0).abs() < f64::EPSILON);
}

#[test]
fn astar_r2_same_wire_returns_empty_path() {
    let ctx = common::make_context();
    let state = Router2State::new(&Router2Cfg::default());
    let wire = WireId::new(0, 0);
    let bbox = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    assert!(
        astar_route_r2(&ctx, &[wire], wire, NetId::from_raw(0), &state, &bbox)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn astar_r2_single_pip_path() {
    let ctx = common::make_context();
    let state = Router2State::new(&Router2Cfg::default());
    let bbox = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    assert_eq!(
        astar_route_r2(
            &ctx,
            &[WireId::new(0, 0)],
            WireId::new(0, 1),
            NetId::from_raw(0),
            &state,
            &bbox
        )
        .unwrap(),
        vec![PipId::new(0, 0)]
    );
}

#[test]
fn astar_r2_no_path_returns_none() {
    let ctx = common::make_context();
    let state = Router2State::new(&Router2Cfg::default());
    let bbox = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    assert!(astar_route_r2(
        &ctx,
        &[WireId::new(0, 1)],
        WireId::new(0, 0),
        NetId::from_raw(0),
        &state,
        &bbox
    )
    .is_none());
}

#[test]
fn astar_r2_bbox_prunes_out_of_range() {
    let ctx = common::make_context();
    let state = Router2State::new(&Router2Cfg::default());
    let bbox = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };
    assert!(astar_route_r2(
        &ctx,
        &[WireId::new(0, 0)],
        WireId::new(1, 0),
        NetId::from_raw(0),
        &state,
        &bbox
    )
    .is_none());
}

#[test]
fn astar_r2_empty_sources_returns_none() {
    let ctx = common::make_context();
    let state = Router2State::new(&Router2Cfg::default());
    let bbox = BoundingBox {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    assert!(astar_route_r2(
        &ctx,
        &[],
        WireId::new(0, 1),
        NetId::from_raw(0),
        &state,
        &bbox
    )
    .is_none());
}

#[test]
fn present_cost_initialized_from_config() {
    let cfg = Router2Cfg {
        initial_present_cost: 2.5,
        ..Router2Cfg::default()
    };
    let state = Router2State::new(&cfg);
    assert!((state.present_cost - 2.5).abs() < f64::EPSILON);
}

#[test]
fn present_cost_grows() {
    let cfg = Router2Cfg {
        initial_present_cost: 1.0,
        present_cost_growth: 2.0,
        ..Router2Cfg::default()
    };
    let mut state = Router2State::new(&cfg);
    state.present_cost *= state.cfg.present_cost_growth;
    state.present_cost *= state.cfg.present_cost_growth;
    assert!((state.present_cost - 4.0).abs() < f64::EPSILON);
}

#[test]
fn count_congested_wires_none() {
    let mut state = Router2State::new(&Router2Cfg::default());
    state.wire_usage.insert(WireId::new(0, 0), 1);
    state.wire_usage.insert(WireId::new(0, 1), 1);
    assert_eq!(state.count_congested_wires(), 0);
}

#[test]
fn count_congested_wires_some() {
    let mut state = Router2State::new(&Router2Cfg::default());
    state.wire_usage.insert(WireId::new(0, 0), 2);
    state.wire_usage.insert(WireId::new(0, 1), 1);
    state.wire_usage.insert(WireId::new(1, 0), 3);
    assert_eq!(state.count_congested_wires(), 2);
}

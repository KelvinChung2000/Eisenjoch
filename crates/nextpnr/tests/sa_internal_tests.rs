mod common;

use nextpnr::chipdb::BelId;
use nextpnr::common::PlaceStrength;
use nextpnr::netlist::PortType;
use nextpnr::placer::common::{initial_placement, net_hpwl, total_hpwl};
use nextpnr::placer::sa::{revert_swap, try_swap};

#[test]
fn hpwl_no_driver_is_zero() {
    let mut ctx = common::make_context();
    let net_idx = ctx.design.add_net(ctx.id("floating"));
    assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
}

#[test]
fn hpwl_no_users_is_zero() {
    let mut ctx = common::make_context();
    let cell_type = ctx.id("LUT4");
    let cell_idx = ctx.design.add_cell(ctx.id("drv"), cell_type);
    let net_idx = ctx.design.add_net(ctx.id("n0"));
    let q_port = ctx.id("Q");
    ctx.design
        .cell_edit(cell_idx)
        .add_port(q_port, PortType::Out);
    ctx.design.net_edit(net_idx).set_driver(cell_idx, q_port);
    assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
}

#[test]
fn hpwl_adjacent_tiles() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cell_type = ctx.id("LUT4");
    let drv_idx = ctx.design.add_cell(ctx.id("drv"), cell_type);
    let usr_idx = ctx.design.add_cell(ctx.id("usr"), cell_type);
    ctx.bind_bel(BelId::new(0, 0), drv_idx, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), usr_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("n"));
    let q_port = ctx.id("Q");
    let a_port = ctx.id("A");
    ctx.design
        .cell_edit(drv_idx)
        .add_port(q_port, PortType::Out);
    ctx.design
        .cell_edit(drv_idx)
        .set_port_net(q_port, Some(net_idx), None);
    ctx.design.net_edit(net_idx).set_driver(drv_idx, q_port);
    ctx.design.cell_edit(usr_idx).add_port(a_port, PortType::In);
    let user_idx = ctx.design.net_edit(net_idx).add_user(usr_idx, a_port);
    ctx.design
        .cell_edit(usr_idx)
        .set_port_net(a_port, Some(net_idx), Some(user_idx));
    assert_eq!(net_hpwl(&ctx, net_idx), 1.0);
}

#[test]
fn hpwl_diagonal_placement() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cell_type = ctx.id("LUT4");
    let drv_idx = ctx.design.add_cell(ctx.id("drv"), cell_type);
    let usr_idx = ctx.design.add_cell(ctx.id("usr"), cell_type);
    ctx.bind_bel(BelId::new(0, 0), drv_idx, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), usr_idx, PlaceStrength::Placer);
    let net_idx = ctx.design.add_net(ctx.id("n"));
    let q_port = ctx.id("Q");
    let a_port = ctx.id("A");
    ctx.design
        .cell_edit(drv_idx)
        .add_port(q_port, PortType::Out);
    ctx.design
        .cell_edit(drv_idx)
        .set_port_net(q_port, Some(net_idx), None);
    ctx.design.net_edit(net_idx).set_driver(drv_idx, q_port);
    ctx.design.cell_edit(usr_idx).add_port(a_port, PortType::In);
    let user_idx = ctx.design.net_edit(net_idx).add_user(usr_idx, a_port);
    ctx.design
        .cell_edit(usr_idx)
        .set_port_net(a_port, Some(net_idx), Some(user_idx));
    assert_eq!(net_hpwl(&ctx, net_idx), 2.0);
}

#[test]
fn total_hpwl_sums_all_nets() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cell_type = ctx.id("LUT4");
    let ids: Vec<_> = (0..4)
        .map(|i| ctx.design.add_cell(ctx.id(&format!("c{}", i)), cell_type))
        .collect();
    for (i, &ci) in ids.iter().enumerate() {
        ctx.bind_bel(BelId::new(i as i32, 0), ci, PlaceStrength::Placer);
    }
    let q_port = ctx.id("Q");
    let a_port = ctx.id("A");
    let net_a = ctx.design.add_net(ctx.id("net_a"));
    ctx.design.cell_edit(ids[0]).add_port(q_port, PortType::Out);
    ctx.design
        .cell_edit(ids[0])
        .set_port_net(q_port, Some(net_a), None);
    ctx.design.net_edit(net_a).set_driver(ids[0], q_port);
    ctx.design.cell_edit(ids[3]).add_port(a_port, PortType::In);
    let user_a = ctx.design.net_edit(net_a).add_user(ids[3], a_port);
    ctx.design
        .cell_edit(ids[3])
        .set_port_net(a_port, Some(net_a), Some(user_a));
    let net_b = ctx.design.add_net(ctx.id("net_b"));
    let b_port = ctx.id("B");
    let c_port = ctx.id("C");
    ctx.design.cell_edit(ids[1]).add_port(b_port, PortType::Out);
    ctx.design
        .cell_edit(ids[1])
        .set_port_net(b_port, Some(net_b), None);
    ctx.design.net_edit(net_b).set_driver(ids[1], b_port);
    ctx.design.cell_edit(ids[2]).add_port(c_port, PortType::In);
    let user_b = ctx.design.net_edit(net_b).add_user(ids[2], c_port);
    ctx.design
        .cell_edit(ids[2])
        .set_port_net(c_port, Some(net_b), Some(user_b));
    assert_eq!(total_hpwl(&ctx), 4.0);
}

#[test]
fn initial_placement_places_all_cells() {
    let mut ctx = common::make_context_with_cells(3);
    initial_placement(&mut ctx).unwrap();
    for (_id, cell) in ctx.design.iter_alive_cells() {
        assert!(cell.bel.is_some());
    }
}

#[test]
fn initial_placement_no_duplicate_bels() {
    let mut ctx = common::make_context_with_cells(4);
    initial_placement(&mut ctx).unwrap();
    let mut used = std::collections::HashSet::new();
    for (_id, cell) in ctx.design.iter_alive_cells() {
        assert!(used.insert(cell.bel.unwrap()));
    }
}

#[test]
fn initial_placement_too_many_cells_fails() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let cell_type = ctx.id("LUT4");
    for i in 0..5 {
        ctx.design
            .add_cell(ctx.id(&format!("cell_{}", i)), cell_type);
    }
    assert!(initial_placement(&mut ctx).is_err());
}

#[test]
fn initial_placement_skips_already_placed() {
    let mut ctx = common::make_context_with_cells(2);
    let cell_idx = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let bel = BelId::new(0, 0);
    ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer);
    initial_placement(&mut ctx).unwrap();
    assert_eq!(ctx.design.cell(cell_idx).bel, Some(bel));
}

#[test]
fn swap_to_empty_bel() {
    let mut ctx = common::make_context_with_cells(1);
    initial_placement(&mut ctx).unwrap();
    let cell_idx = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let old_bel = ctx.design.cell(cell_idx).bel.unwrap();
    let empty_bel = ctx.bels().map(|b| b.id()).find(|&b| b != old_bel).unwrap();
    let result = try_swap(&mut ctx, cell_idx, empty_bel);
    assert!(result.performed);
    assert_eq!(ctx.design.cell(cell_idx).bel, Some(empty_bel));
}

#[test]
fn swap_two_cells() {
    let mut ctx = common::make_context_with_cells(2);
    initial_placement(&mut ctx).unwrap();
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();
    let bel0 = ctx.design.cell(cell0).bel.unwrap();
    let bel1 = ctx.design.cell(cell1).bel.unwrap();
    let result = try_swap(&mut ctx, cell0, bel1);
    assert!(result.performed);
    assert_eq!(ctx.design.cell(cell0).bel, Some(bel1));
    assert_eq!(ctx.design.cell(cell1).bel, Some(bel0));
}

#[test]
fn swap_same_bel_is_noop() {
    let mut ctx = common::make_context_with_cells(1);
    initial_placement(&mut ctx).unwrap();
    let cell_idx = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let bel = ctx.design.cell(cell_idx).bel.unwrap();
    let result = try_swap(&mut ctx, cell_idx, bel);
    assert!(!result.performed);
    assert_eq!(result.delta_cost, 0.0);
}

#[test]
fn revert_swap_restores_state() {
    let mut ctx = common::make_context_with_cells(2);
    initial_placement(&mut ctx).unwrap();
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();
    let bel0 = ctx.design.cell(cell0).bel.unwrap();
    let bel1 = ctx.design.cell(cell1).bel.unwrap();
    let _ = try_swap(&mut ctx, cell0, bel1);
    revert_swap(&mut ctx, cell0, bel0, Some(cell1), bel1);
    assert_eq!(ctx.design.cell(cell0).bel, Some(bel0));
    assert_eq!(ctx.design.cell(cell1).bel, Some(bel1));
}

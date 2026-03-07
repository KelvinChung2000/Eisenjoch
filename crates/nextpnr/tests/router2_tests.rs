mod common;

use nextpnr::chipdb::BelId;
use nextpnr::common::PlaceStrength;
use nextpnr::netlist::PortType;
use nextpnr::router::router2::{route_router2, Router2Cfg};
use nextpnr::router::{Router, Router2, RouterError};

// =====================================================================
// Config defaults
// =====================================================================

#[test]
fn default_config() {
    let cfg = Router2Cfg::default();
    assert_eq!(cfg.max_iterations, 50);
    assert!((cfg.base_cost - 1.0).abs() < f64::EPSILON);
    assert!((cfg.present_cost_multiplier - 2.0).abs() < f64::EPSILON);
    assert!((cfg.history_cost_multiplier - 1.0).abs() < f64::EPSILON);
    assert!((cfg.initial_present_cost - 1.0).abs() < f64::EPSILON);
    assert!((cfg.present_cost_growth - 1.5).abs() < f64::EPSILON);
    assert_eq!(cfg.bb_margin, 3);
    assert!(!cfg.verbose);
}

// =====================================================================
// Error display
// =====================================================================

#[test]
fn router2_error_no_path() {
    let err = RouterError::NoPath("my_net".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("my_net"));
    assert!(msg.contains("no path"));
}

#[test]
fn router2_error_congestion() {
    let err = RouterError::Congestion(42, 7);
    let msg = format!("{}", err);
    assert!(msg.contains("42"));
    assert!(msg.contains("7"));
}

#[test]
fn router2_error_generic() {
    let err = RouterError::Generic("something went wrong".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("something went wrong"));
}

// =====================================================================
// Integration: route_router2
// =====================================================================

#[test]
fn route_r2_empty_design() {
    let mut ctx = common::make_context();
    let cfg = Router2Cfg::default();
    let result = route_router2(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_design_with_no_routable_nets() {
    let mut ctx = common::make_context();
    let net_name = ctx.id("no_driver");
    ctx.design.add_net(net_name);

    let cfg = Router2Cfg::default();
    let result = route_router2(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_design_with_no_users() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");

    let cell_name = ctx.id("driver");
    let cell_idx = ctx.design.add_cell(cell_name, lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);

    let net_name = ctx.id("driveronly");
    let net_idx = ctx.design.add_net(net_name);
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);

    let cfg = Router2Cfg::default();
    let result = route_router2(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_same_pin_driver_and_sink() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT4");
    let port_name = ctx.id("I0");

    let cell_name = ctx.id("cell_a");
    let cell_idx = ctx.design.add_cell(cell_name, lut_type);
    ctx.design
        .cell_edit(cell_idx)
        .add_port(port_name, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);

    let net_name = ctx.id("net_self");
    let net_idx = ctx.design.add_net(net_name);

    ctx.design.net_edit(net_idx).set_driver(cell_idx, port_name);
    ctx.design.net_edit(net_idx).add_user(cell_idx, port_name);

    let cfg = Router2Cfg::default();
    let result = route_router2(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_via_trait() {
    let mut ctx = common::make_context();
    let cfg = Router2Cfg::default();
    Router2
        .route(&mut ctx, &cfg)
        .expect("trait dispatch should work");
}

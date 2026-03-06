use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::context::Context;
use nextpnr::netlist::PortRef;
use nextpnr::router::router1::{route_router1, Router1Cfg};
use nextpnr::router::{Router, Router1, RouterError};
use nextpnr::types::{BelId, PlaceStrength, PortType};

/// Create a fresh Context backed by the synthetic 2x2 chipdb.
fn make_context() -> Context {
    let chipdb = make_test_chipdb();
    Context::new(chipdb)
}

// =====================================================================
// Router1Cfg defaults
// =====================================================================

#[test]
fn default_config() {
    let cfg = Router1Cfg::default();
    assert_eq!(cfg.max_iterations, 500);
    assert_eq!(cfg.rip_up_penalty, 10);
    assert!((cfg.congestion_weight - 1.0).abs() < f64::EPSILON);
    assert!(!cfg.verbose);
}

// =====================================================================
// RouterError display
// =====================================================================

#[test]
fn router_error_no_path() {
    let err = RouterError::NoPath("my_net".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("my_net"));
    assert!(msg.contains("no path"));
}

#[test]
fn router_error_congestion() {
    let err = RouterError::Congestion(100, 5);
    let msg = format!("{}", err);
    assert!(msg.contains("100"));
    assert!(msg.contains("5"));
}

#[test]
fn router_error_generic() {
    let err = RouterError::Generic("oops".to_string());
    assert!(format!("{}", err).contains("oops"));
}

// =====================================================================
// Integration: route_router1
// =====================================================================

#[test]
fn route_empty_design() {
    let mut ctx = make_context();
    let cfg = Router1Cfg::default();
    let result = route_router1(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_design_with_no_routable_nets() {
    let mut ctx = make_context();
    let net_name = ctx.id("no_driver");
    ctx.design_mut().add_net(net_name);

    let cfg = Router1Cfg::default();
    let result = route_router1(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_design_with_no_users() {
    let mut ctx = make_context();
    let lut_type = ctx.id("LUT4");
    let port = ctx.id("I0");

    let cell_name = ctx.id("driver");
    let cell_idx = ctx.design_mut().add_cell(cell_name, lut_type);
    ctx.design_mut().cell_mut(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);

    let net_name = ctx.id("driveronly");
    let net_idx = ctx.design_mut().add_net(net_name);
    ctx.design_mut().net_mut(net_idx).driver = PortRef {
        cell: Some(cell_idx),
        port,
        budget: 0,
    };

    let cfg = Router1Cfg::default();
    let result = route_router1(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_via_trait() {
    let mut ctx = make_context();
    let cfg = Router1Cfg::default();
    Router1.route(&mut ctx, &cfg).expect("trait dispatch should work");
}

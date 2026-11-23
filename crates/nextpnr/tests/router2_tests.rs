mod common;

use nextpnr::chipdb::BelId;
use nextpnr::common::PlaceStrength;
use nextpnr::netlist::PortType;
use nextpnr::router::common::{collect_routable_nets, validate_all_routed};
use nextpnr::router::router2::Router2Cfg;
use nextpnr::router::{Router, Router2};
use rstest::rstest;

// =====================================================================
// Config defaults
// =====================================================================

#[test]
fn default_config() {
    let cfg = Router2Cfg::default();
    assert_eq!(cfg.max_iterations, 50);
    assert!((cfg.negotiation.base_cost - 1.0).abs() < f64::EPSILON);
    assert!((cfg.negotiation.present_cost_multiplier - 2.0).abs() < f64::EPSILON);
    assert!((cfg.negotiation.history_cost_multiplier - 1.0).abs() < f64::EPSILON);
    assert!((cfg.negotiation.initial_present_cost - 1.0).abs() < f64::EPSILON);
    assert!((cfg.negotiation.present_cost_growth - 1.5).abs() < f64::EPSILON);
    assert_eq!(cfg.bb_margin, 3);
    assert!(!cfg.verbose);
}

// RouterError display tests are in router1_tests.rs (shared enum).

// =====================================================================
// Integration: Router2.route()
// =====================================================================

#[test]
fn route_r2_empty_design() {
    let mut ctx = common::make_context();
    let cfg = Router2Cfg::default();
    let result = Router2.route(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_design_with_no_routable_nets() {
    let mut ctx = common::make_context();
    let net_name = ctx.id("no_driver");
    ctx.design.add_net(net_name);

    let cfg = Router2Cfg::default();
    let result = Router2.route(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_design_with_no_users() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT6");
    let port = ctx.id("I0");

    let cell_name = ctx.id("driver");
    let cell_idx = ctx.design.add_cell(cell_name, lut_type);
    ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
    ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);

    let net_name = ctx.id("driveronly");
    let net_idx = ctx.design.add_net(net_name);
    ctx.design.net_edit(net_idx).set_driver(cell_idx, port);

    let cfg = Router2Cfg::default();
    let result = Router2.route(&mut ctx, &cfg);
    assert!(result.is_ok());
}

#[test]
fn route_r2_same_pin_driver_and_sink() {
    let mut ctx = common::make_context();
    let lut_type = ctx.id("LUT6");
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
    let result = Router2.route(&mut ctx, &cfg);
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

// =====================================================================
// Negotiation on a placed design
// =====================================================================

#[rstest]
#[case::default_margin(3)]
#[case::tight_margin(1)]
fn negotiation_resolves_npnr_baseline_congestion(#[case] bb_margin: i32) {
    let mut ctx = common::placed_npnr_baseline();
    let nets = collect_routable_nets(&ctx);
    let cfg = Router2Cfg {
        bb_margin,
        ..Router2Cfg::default()
    };
    let report = Router2
        .route_nets_report(&mut ctx, &cfg, &nets)
        .expect("negotiation converges");
    validate_all_routed(&ctx).expect("every net is routed and no wire is shared");
    let (first, last) = (report.iterations[0], *report.iterations.last().unwrap());
    assert!(first.overused_wires > 0, "the initial pass overuses wires");
    assert!(report.iterations.len() > 1, "negotiation replans");
    assert_eq!(last.overused_wires, 0);
}

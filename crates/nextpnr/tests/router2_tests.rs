mod common;

use nextpnr::chipdb::BelId;
use nextpnr::chipdb::{PipId, WireId};
use nextpnr::common::DelayT;
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::PortType;
use nextpnr::router::common::{collect_routable_nets, validate_all_routed};
use nextpnr::router::router2::{Router2Cfg, Router2Report, TimingDrivenCfg};
use nextpnr::router::{Router, Router2};
use nextpnr::sta::{
    CritNorm, InterconnectModel, InterconnectStage, NetSink, PlacedEstimate, StaSession,
    NOTIONAL_CLASSLESS_PIPS,
};
use rstest::rstest;
use rustc_hash::FxHashMap;

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
    assert!(cfg.timing.is_none());
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

fn clocked_baseline(period_ps: Option<DelayT>) -> Context {
    let mut ctx = common::placed_npnr_baseline();
    if let Some(period_ps) = period_ps {
        let clk_buf = ctx
            .design
            .cell_by_name(ctx.id("clk_buf"))
            .expect("fixture has clk_buf");
        let clk = ctx
            .design
            .cell(clk_buf)
            .port_net(ctx.id("O"))
            .expect("clk_buf drives the clock");
        ctx.design.net_edit(clk).set_clock_constraint(period_ps);
    }
    ctx
}

fn route_baseline(ctx: &mut Context, cfg: &Router2Cfg) -> Router2Report {
    let nets = collect_routable_nets(ctx);
    let report = Router2
        .route_nets_report(ctx, cfg, &nets)
        .expect("negotiation converges");
    validate_all_routed(ctx).expect("every net is routed and no wire is shared");
    report
}

fn timing_cfg(timing: TimingDrivenCfg) -> Router2Cfg {
    Router2Cfg {
        timing: Some(timing),
        ..Router2Cfg::default()
    }
}

fn session(ctx: &Context, stage: InterconnectStage) -> StaSession {
    let model = InterconnectModel::new(ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::DelayTable)
        .expect("fixture chipdb has pip classes");
    StaSession::build(ctx, model, stage).expect("session builds the fixture")
}

fn routed_trees(ctx: &Context) -> Vec<FxHashMap<WireId, Option<PipId>>> {
    collect_routable_nets(ctx)
        .into_iter()
        .map(|net| {
            ctx.design
                .net(net)
                .wires
                .iter()
                .map(|(&wire, entry)| (wire, entry.pip))
                .collect()
        })
        .collect()
}

fn timed_sinks(ctx: &Context) -> Vec<NetSink> {
    collect_routable_nets(ctx)
        .into_iter()
        .flat_map(|net| {
            (0..ctx.design.net(net).users().len()).map(move |user| NetSink {
                net,
                user: u32::try_from(user).expect("user index fits in u32"),
            })
        })
        .collect()
}

#[rstest]
#[case::default_margin(3, None)]
#[case::tight_margin(1, None)]
#[case::timing_driven_4ns(3, Some(4000))]
fn negotiation_resolves_npnr_baseline_congestion(
    #[case] bb_margin: i32,
    #[case] timed_period_ps: Option<DelayT>,
) {
    let mut ctx = clocked_baseline(timed_period_ps);
    let cfg = Router2Cfg {
        bb_margin,
        timing: timed_period_ps.map(|_| TimingDrivenCfg::default()),
        ..Router2Cfg::default()
    };
    let report = route_baseline(&mut ctx, &cfg);
    let (first, last) = (report.iterations[0], *report.iterations.last().unwrap());
    assert!(first.overused_wires > 0, "the initial pass overuses wires");
    assert!(report.iterations.len() > 1, "negotiation replans");
    assert_eq!(last.overused_wires, 0);
}

#[rstest]
#[case::met_20ns(20000)]
#[case::failing_4ns(4000)]
fn zero_criticality_cap_routes_like_timing_off(#[case] period_ps: DelayT) {
    let mut off = clocked_baseline(Some(period_ps));
    route_baseline(&mut off, &Router2Cfg::default());
    let mut capped = clocked_baseline(Some(period_ps));
    let cap = TimingDrivenCfg {
        max_criticality: 0.0,
        ..TimingDrivenCfg::default()
    };
    route_baseline(&mut capped, &timing_cfg(cap));
    assert!(routed_trees(&off) == routed_trees(&capped));
}

#[rstest]
#[case::failing_4ns(4000)]
#[case::failing_3ns(3000)]
fn criticality_refresh_equals_fresh_routed_build(#[case] period_ps: DelayT) {
    let mut ctx = clocked_baseline(Some(period_ps));
    let report = route_baseline(&mut ctx, &timing_cfg(TimingDrivenCfg::default()));
    let refreshed = report
        .timing
        .expect("a timing-driven run returns its session");
    let fresh = session(&ctx, InterconnectStage::Routed);
    assert_eq!(refreshed.domains(), fresh.domains());
    for sink in timed_sinks(&ctx) {
        assert_eq!(
            refreshed.sink_timing(sink).unwrap(),
            fresh.sink_timing(sink).unwrap()
        );
        let bits = |session: &StaSession| {
            session
                .normalised_criticality(sink, CritNorm::RequiredTime)
                .unwrap()
                .map(f64::to_bits)
        };
        assert_eq!(bits(&refreshed), bits(&fresh), "{sink:?}");
    }
}

#[rstest]
#[case::failing_4ns(4000)]
#[case::failing_3ns(3000)]
fn timing_takes_a_detouring_critical_connection_to_its_fastest_route(#[case] period_ps: DelayT) {
    let placed_ctx = clocked_baseline(Some(period_ps));
    let placed = session(&placed_ctx, InterconnectStage::Placed);
    let routed_delays = |cfg: &Router2Cfg| {
        let mut ctx = clocked_baseline(Some(period_ps));
        route_baseline(&mut ctx, cfg);
        let routed = session(&ctx, InterconnectStage::Routed);
        move |sink: NetSink| routed.connection_delay(sink).unwrap().late_ps
    };
    let off = routed_delays(&Router2Cfg::default());
    let on = routed_delays(&timing_cfg(TimingDrivenCfg::default()));
    let critical: Vec<(NetSink, DelayT)> = timed_sinks(&placed_ctx)
        .into_iter()
        .filter(|&sink| placed.connection_delay(sink).is_ok())
        .filter(|&sink| {
            placed
                .normalised_criticality(sink, CritNorm::RequiredTime)
                .unwrap()
                .is_some_and(|crit| crit >= TimingDrivenCfg::default().max_criticality)
        })
        .map(|sink| (sink, placed.connection_delay(sink).unwrap().late_ps))
        .collect();
    assert!(
        critical
            .iter()
            .any(|&(sink, fastest)| off(sink) > fastest && on(sink) == fastest),
        "a critical connection that detours with timing off takes its fastest route with timing on"
    );
    let total = |delay: &dyn Fn(NetSink) -> DelayT| -> DelayT {
        critical.iter().map(|&(sink, _)| delay(sink)).sum()
    };
    assert!(total(&on) < total(&off));
}

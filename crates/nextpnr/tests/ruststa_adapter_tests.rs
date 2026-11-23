mod common;

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::netlist::{CellId, CellPin, NetId, PortType};
use nextpnr::sta::{RustSta, StaBuildError};
use nextpnr::timing::DelayT;
use rstest::rstest;

const SLACK_TOLERANCE_NS: f64 = 1e-9;

fn add_cell(ctx: &mut Context, name: &str, cell_type: &str) -> CellId {
    let (name, cell_type) = (ctx.id(name), ctx.id(cell_type));
    ctx.design.add_cell(name, cell_type)
}

fn add_net(ctx: &mut Context, name: &str) -> NetId {
    let name = ctx.id(name);
    ctx.design.add_net(name)
}

fn drive(ctx: &mut Context, net: NetId, cell: CellId, port: &str) {
    let port = ctx.id(port);
    ctx.design
        .cell_edit(cell)
        .add_port(port, PortType::Out)
        .set_port_net(port, Some(net), None);
    ctx.design.net_edit(net).set_driver(cell, port);
}

fn sink(ctx: &mut Context, net: NetId, cell: CellId, port: &str) {
    let port = ctx.id(port);
    let user = ctx.design.net_edit(net).add_user(cell, port);
    ctx.design
        .cell_edit(cell)
        .add_port(port, PortType::In)
        .set_port_net(port, Some(net), Some(user));
}

fn register_chain(ctx: &mut Context, lut_inputs: &[usize], period_ps: DelayT) -> CellId {
    let clk = add_net(ctx, "clk");
    let clk_src = add_cell(ctx, "clk_src", "LUT6");
    drive(ctx, clk, clk_src, "F");
    ctx.design.net_edit(clk).set_clock_constraint(period_ps);
    let launch = add_cell(ctx, "launch", "DFF");
    let capture = add_cell(ctx, "capture", "DFF");
    sink(ctx, clk, launch, "CLK");
    sink(ctx, clk, capture, "CLK");
    let mut source = (launch, String::from("Q"));
    for (stage, input) in lut_inputs.iter().enumerate() {
        let net = add_net(ctx, &format!("d{stage}"));
        drive(ctx, net, source.0, &source.1);
        let lut = add_cell(ctx, &format!("lut{stage}"), "LUT6");
        sink(ctx, net, lut, &format!("I[{input}]"));
        source = (lut, String::from("F"));
    }
    let d = add_net(ctx, "d_capture");
    drive(ctx, d, source.0, &source.1);
    sink(ctx, d, capture, "D");
    capture
}

struct ChipTiming {
    lut_input_to_f_ps: Vec<DelayT>,
    clk_to_q_ps: DelayT,
    setup_ps: DelayT,
    hold_ps: DelayT,
}

impl ChipTiming {
    fn read(ctx: &Context) -> Self {
        let chipdb = ctx.chipdb();
        let sg = ctx.speed_grade().expect("example chipdb has a speed grade");
        let type_idx = |name: &str| {
            chipdb
                .cell_timing_index(sg, ctx.id(name).index())
                .expect("example chipdb times LUT6 and DFF")
        };
        let (lut, dff) = (type_idx("LUT6"), type_idx("DFF"));
        let reg_arc = |port: &str| {
            let arcs = chipdb
                .cell_reg_arcs(sg, dff, ctx.id(port).index())
                .expect("DFF port has register arcs");
            ChipDb::reg_arc_info(&arcs[0])
        };
        let lut_input_to_f_ps = (0..6)
            .map(|input| {
                chipdb
                    .cell_delay(
                        sg,
                        lut,
                        ctx.id(format!("I[{input}]").as_str()).index(),
                        ctx.id("F").index(),
                    )
                    .expect("LUT6 input reaches F")
                    .max_delay()
            })
            .collect();
        Self {
            lut_input_to_f_ps,
            clk_to_q_ps: reg_arc("Q").clock_to_q.max_delay(),
            setup_ps: reg_arc("D").setup.max_delay,
            hold_ps: reg_arc("D").hold.max_delay,
        }
    }
}

fn assert_slack_ns(actual_ns: Option<f64>, expected_ps: DelayT) {
    let actual_ns = actual_ns.expect("endpoint has the check");
    let expected_ns = f64::from(expected_ps) / 1000.0;
    assert!(
        (actual_ns - expected_ns).abs() < SLACK_TOLERANCE_NS,
        "RustSTA slack {actual_ns} ns, hand-computed {expected_ns} ns"
    );
}

#[rstest]
#[case::register_to_register(&[], 1000)]
#[case::one_lut(&[0], 1000)]
#[case::two_luts(&[5, 2], 1000)]
#[case::three_luts(&[1, 3, 4], 1000)]
#[case::three_luts_violating(&[5, 4, 3], 800)]
fn capture_slack_is_chipdb_arc_sum(#[case] lut_inputs: &[usize], #[case] period_ps: DelayT) {
    let mut ctx = common::make_example_context();
    let capture = register_chain(&mut ctx, lut_inputs, period_ps);
    let timing = ChipTiming::read(&ctx);

    let lut_sum_ps: DelayT = lut_inputs
        .iter()
        .map(|&input| timing.lut_input_to_f_ps[input])
        .sum();
    let data_arrival_ps = timing.clk_to_q_ps + lut_sum_ps;
    let expected_setup_ps = period_ps - (data_arrival_ps + timing.setup_ps);
    let expected_hold_ps = data_arrival_ps - timing.hold_ps;

    let sta = RustSta::build(&ctx).expect("adapter builds the chain");
    let slack = sta
        .endpoint_slack(CellPin::new(capture, ctx.id("D")))
        .expect("capture D is an endpoint");
    assert_slack_ns(slack.setup_ns, expected_setup_ps);
    assert_slack_ns(slack.hold_ns, expected_hold_ps);
}

#[test]
fn net_sinks_map_to_their_wire_arcs() {
    let mut ctx = common::make_example_context();
    register_chain(&mut ctx, &[0, 1], 1000);
    let sta = RustSta::build(&ctx).expect("adapter builds the chain");
    let (ids, arcs) = (sta.ids(), sta.session().graph().arcs());

    let sinks: usize = ctx
        .design
        .iter_alive_nets()
        .map(|(_, net)| net.users().len())
        .sum();
    assert_eq!(ids.net_arcs.len(), sinks);
    for (net_sink, &arc) in &ids.net_arcs {
        let net = ctx.design.net(net_sink.net);
        let driver = net.driver().expect("net with a sink has a driver");
        let user = net.users()[net_sink.user as usize];
        assert_eq!(arcs[arc].src, ids.pins[&driver].node);
        assert_eq!(arcs[arc].dst, ids.pins[&user].node);
        assert_eq!(ids.node_pins[arcs[arc].dst.index()], Some(user));
    }
}

#[derive(Debug)]
enum ExpectedError {
    CellTypeWithoutTiming(&'static str),
    ClockNetWithoutDriver(&'static str),
}

fn untimed_cell(ctx: &mut Context) {
    add_cell(ctx, "pad", "IOB");
}

fn undriven_clock(ctx: &mut Context) {
    let clk = add_net(ctx, "clk");
    ctx.design.net_edit(clk).set_clock_constraint(1000);
    let ff = add_cell(ctx, "ff", "DFF");
    sink(ctx, clk, ff, "CLK");
}

#[rstest]
#[case::cell_type_without_timing(untimed_cell, ExpectedError::CellTypeWithoutTiming("IOB"))]
#[case::clock_net_without_driver(undriven_clock, ExpectedError::ClockNetWithoutDriver("clk"))]
fn build_rejects_design(#[case] design: fn(&mut Context), #[case] expected: ExpectedError) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let err = RustSta::build(&ctx).expect_err("adapter rejects the design");
    match (&err, &expected) {
        (
            StaBuildError::CellTypeWithoutTiming { cell_type },
            ExpectedError::CellTypeWithoutTiming(name),
        ) => assert_eq!(cell_type, name),
        (
            StaBuildError::ClockNetWithoutDriver { net },
            ExpectedError::ClockNetWithoutDriver(name),
        ) => {
            assert_eq!(net, name)
        }
        _ => panic!("expected {expected:?}, got {err}"),
    }
}

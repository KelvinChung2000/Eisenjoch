mod common;

use std::collections::BTreeMap;

use common::{add_cell, add_net, drive, sink, sink_as};

use nextpnr::chipdb::{ChipDb, TimingValue};
use nextpnr::common::DelayT;
use nextpnr::context::Context;
use nextpnr::netlist::{CellId, CellPin, NetId, PortType};
use nextpnr::sta::{DelayBounds, RustSta, StaBuildError};
use rstest::rstest;
use rusteda_core::netlist::PortDir;

const SLACK_TOLERANCE_NS: f64 = 1e-9;

#[derive(Debug, Clone, Copy)]
enum PadCells {
    Pseudo,
    Packed,
}

impl PadCells {
    fn cell_type(self, dir: PortDir) -> &'static str {
        match (self, dir) {
            (PadCells::Packed, _) => "IOB",
            (PadCells::Pseudo, PortDir::Input) => "$nextpnr_IBUF",
            (PadCells::Pseudo, PortDir::Output) => "$nextpnr_OBUF",
            (PadCells::Pseudo, PortDir::Inout) => "$nextpnr_IOBUF",
        }
    }
}

fn pad(ctx: &mut Context, cells: PadCells, dir: PortDir, port_name: &str, net: NetId) -> CellId {
    let cell = add_cell(ctx, &format!("$io${port_name}"), cells.cell_type(dir));
    match dir {
        PortDir::Input => drive(ctx, net, cell, "O"),
        PortDir::Output => sink(ctx, net, cell, "I"),
        PortDir::Inout => sink_as(ctx, net, cell, "IO", PortType::InOut),
    }
    cell
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
    lut_input_to_f: Vec<DelayBounds>,
    clk_to_q: DelayBounds,
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
        let lut_input_to_f = (0..6)
            .map(|input| {
                bounds(
                    chipdb
                        .cell_comb_timing(
                            sg,
                            lut,
                            ctx.id(format!("I[{input}]").as_str()).index(),
                            ctx.id("F").index(),
                        )
                        .expect("LUT6 input reaches F"),
                )
            })
            .collect();
        let q_arcs = chipdb
            .cell_reg_arcs(sg, dff, ctx.id("Q").index())
            .expect("DFF Q has a launch arc");
        Self {
            lut_input_to_f,
            clk_to_q: bounds(ChipDb::reg_arc_clock_to_q(&q_arcs[0])),
            setup_ps: reg_arc("D").setup.max_delay,
            hold_ps: reg_arc("D").hold.max_delay,
        }
    }
}

fn bounds(value: TimingValue) -> DelayBounds {
    DelayBounds {
        early_ps: value.fast_min,
        late_ps: value.slow_max,
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

fn padded_register_chain(
    ctx: &mut Context,
    cells: PadCells,
    clock_port: &str,
    lut_input: usize,
    period_ps: DelayT,
) -> CellId {
    let clk = add_net(ctx, "clk");
    pad(ctx, cells, PortDir::Input, clock_port, clk);
    ctx.design.net_edit(clk).set_clock_constraint(period_ps);
    let (din, q, d, dout) = (
        add_net(ctx, "din"),
        add_net(ctx, "q"),
        add_net(ctx, "d"),
        add_net(ctx, "dout"),
    );
    pad(ctx, cells, PortDir::Input, "din", din);
    let launch = add_cell(ctx, "launch", "DFF");
    let lut = add_cell(ctx, "lut", "LUT6");
    let capture = add_cell(ctx, "capture", "DFF");
    sink(ctx, clk, launch, "CLK");
    sink(ctx, clk, capture, "CLK");
    sink(ctx, din, launch, "D");
    drive(ctx, q, launch, "Q");
    sink(ctx, q, lut, &format!("I[{lut_input}]"));
    drive(ctx, d, lut, "F");
    sink(ctx, d, capture, "D");
    drive(ctx, dout, capture, "Q");
    pad(ctx, cells, PortDir::Output, "dout", dout);
    capture
}

fn assert_chain_slack(ctx: &Context, capture: CellId, lut_inputs: &[usize], period_ps: DelayT) {
    let timing = ChipTiming::read(ctx);
    let arrival = lut_inputs
        .iter()
        .map(|&input| timing.lut_input_to_f[input])
        .fold(timing.clk_to_q, |sum, arc| sum + arc);
    let expected_setup_ps = period_ps - (arrival.late_ps + timing.setup_ps);
    let expected_hold_ps = arrival.early_ps - timing.hold_ps;

    let sta = RustSta::build(ctx).expect("adapter builds the chain");
    let slack = sta
        .endpoint_slack(CellPin::new(capture, ctx.id("D")))
        .expect("capture D is an endpoint");
    assert_slack_ns(slack.setup_ns, expected_setup_ps);
    assert_slack_ns(slack.hold_ns, expected_hold_ps);
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
    assert_chain_slack(&ctx, capture, lut_inputs, period_ps);
}

#[rstest]
#[case::pseudo_pads(PadCells::Pseudo, "clk", 2, 1000)]
#[case::packed_pads(PadCells::Packed, "clk", 2, 1000)]
#[case::packed_pads_violating(PadCells::Packed, "clk", 5, 300)]
#[case::bus_bit_clock_port(PadCells::Packed, "sys_clk[0]", 0, 1000)]
fn padded_capture_slack_is_chipdb_arc_sum(
    #[case] cells: PadCells,
    #[case] clock_port: &str,
    #[case] lut_input: usize,
    #[case] period_ps: DelayT,
) {
    let mut ctx = common::make_example_context();
    let capture = padded_register_chain(&mut ctx, cells, clock_port, lut_input, period_ps);
    assert_chain_slack(&ctx, capture, &[lut_input], period_ps);
}

#[rstest]
#[case::pseudo_input(PadCells::Pseudo, PortDir::Input)]
#[case::pseudo_output(PadCells::Pseudo, PortDir::Output)]
#[case::pseudo_inout(PadCells::Pseudo, PortDir::Inout)]
#[case::packed_input(PadCells::Packed, PortDir::Input)]
#[case::packed_output(PadCells::Packed, PortDir::Output)]
fn pads_become_top_level_ports(#[case] cells: PadCells, #[case] dir: PortDir) {
    let mut ctx = common::make_example_context();
    let net = add_net(&mut ctx, "pad_net");
    let lut = add_cell(&mut ctx, "lut", "LUT6");
    match dir {
        PortDir::Input => sink(&mut ctx, net, lut, "I[0]"),
        PortDir::Output | PortDir::Inout => drive(&mut ctx, net, lut, "F"),
    }
    pad(&mut ctx, cells, dir, "data[3]", net);

    let sta = RustSta::build(&ctx).expect("adapter builds the padded design");
    let netlist = sta.session().netlist();
    let port = netlist
        .find_port("data[3]")
        .expect("the pad is a port named after the design port");
    assert_eq!(port.dir, dir);
    assert_eq!(netlist.instance_count(), 1);
}

#[rstest]
#[case::gnd("GND", "Y")]
#[case::vcc("VCC", "Y")]
#[case::gnd_drv("GND_DRV", "GND")]
#[case::vcc_drv("VCC_DRV", "VCC")]
fn constant_lut_input_leaves_chain_slack(#[case] cell_type: &str, #[case] port: &str) {
    let mut ctx = common::make_example_context();
    let capture = register_chain(&mut ctx, &[0], 1000);
    let tie = add_net(&mut ctx, "tie");
    let tie_cell = add_cell(&mut ctx, "$PACKER_TIE", cell_type);
    drive(&mut ctx, tie, tie_cell, port);
    let lut = ctx
        .design
        .cell_by_name(ctx.id("lut0"))
        .expect("chain has lut0");
    sink(&mut ctx, tie, lut, "I[3]");
    assert_chain_slack(&ctx, capture, &[0], 1000);
}

fn unpadded_design(ctx: &mut Context) {
    register_chain(ctx, &[0, 1], 1000);
}

fn padded_design(ctx: &mut Context) {
    padded_register_chain(ctx, PadCells::Packed, "clk", 1, 1000);
}

#[rstest]
#[case::unpadded(unpadded_design)]
#[case::padded(padded_design)]
fn net_sinks_map_to_their_wire_arcs(#[case] design: fn(&mut Context)) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
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

fn untimed_lut_in_chain(ctx: &mut Context) -> CellId {
    let capture = register_chain(ctx, &[0], 1000);
    let lut = ctx.design.cell_by_name(ctx.id("lut0")).expect("chain has lut0");
    let untimed = ctx.id("UNTIMED");
    ctx.design.cell_edit(lut).set_type(untimed);
    capture
}

fn register_without_clock_port(ctx: &mut Context) -> CellId {
    let (lut, register, d) = (add_cell(ctx, "lut", "LUT6"), add_cell(ctx, "reg", "DFF"), add_net(ctx, "d"));
    drive(ctx, d, lut, "F");
    sink(ctx, d, register, "D");
    register
}

#[rstest]
#[case::untimed_cell_type(untimed_lut_in_chain, &[("UNTIMED", 1)])]
#[case::register_without_clock_port(register_without_clock_port, &[])]
fn build_leaves_the_capture_input_unconstrained(
    #[case] design: fn(&mut Context) -> CellId,
    #[case] black_boxes: &[(&str, usize)],
) {
    let mut ctx = common::make_example_context();
    let capture = design(&mut ctx);

    let sta = RustSta::build(&ctx).expect("adapter builds the design");
    let expected: BTreeMap<String, usize> =
        black_boxes.iter().map(|&(name, cells)| (name.to_owned(), cells)).collect();
    assert_eq!(sta.black_boxes(), &expected);
    let setup = sta
        .endpoint_slack(CellPin::new(capture, ctx.id("D")))
        .and_then(|slack| slack.setup_ns);
    assert_eq!(setup, None);
}

fn pad_without_port_name(ctx: &mut Context) {
    let net = add_net(ctx, "pad_net");
    let pad = add_cell(ctx, "pad", "IOB");
    drive(ctx, net, pad, "O");
    let ff = add_cell(ctx, "ff", "DFF");
    sink(ctx, net, ff, "D");
}

fn buffer_without_fabric_side(ctx: &mut Context) {
    let buf = add_cell(ctx, "buf", "INBUF");
    let pad_net = add_net(ctx, "pad_net");
    sink(ctx, pad_net, buf, "PAD");
    let dangling = add_net(ctx, "dangling");
    drive(ctx, dangling, buf, "O");
}

fn undriven_clock(ctx: &mut Context) {
    let clk = add_net(ctx, "clk");
    ctx.design.net_edit(clk).set_clock_constraint(1000);
    let ff = add_cell(ctx, "ff", "DFF");
    sink(ctx, clk, ff, "CLK");
}

fn constant_clock(ctx: &mut Context) {
    let clk = add_net(ctx, "clk");
    ctx.design.net_edit(clk).set_clock_constraint(1000);
    let gnd = add_cell(ctx, "$PACKER_GND", "GND");
    drive(ctx, clk, gnd, "Y");
    let ff = add_cell(ctx, "ff", "DFF");
    sink(ctx, clk, ff, "CLK");
}

#[rstest]
#[case::pad_without_port_name(
    pad_without_port_name,
    |err: &StaBuildError| matches!(err, StaBuildError::PadWithoutPortName { cell } if cell == "pad")
)]
#[case::io_buffer_without_fabric_side(
    buffer_without_fabric_side,
    |err: &StaBuildError| matches!(err, StaBuildError::IoBufferSides { cell, pad_side: 2, fabric_side: 0 } if cell == "buf")
)]
#[case::clock_net_without_driver(
    undriven_clock,
    |err: &StaBuildError| matches!(err, StaBuildError::ClockNetWithoutDriver { net } if net == "clk")
)]
#[case::clock_net_driven_by_constant(
    constant_clock,
    |err: &StaBuildError| matches!(err, StaBuildError::ClockNetDrivenByConstant { net } if net == "clk")
)]
fn build_rejects_design(
    #[case] design: fn(&mut Context),
    #[case] expected: fn(&StaBuildError) -> bool,
) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let err = RustSta::build(&ctx).expect_err("adapter rejects the design");
    assert!(expected(&err), "unexpected error: {err}");
}

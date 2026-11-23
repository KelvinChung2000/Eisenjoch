mod common;

use common::{add_cell, add_net, drive, sink};

use nextpnr::chipdb::{BelId, PipId, WireId};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::{CellId, NetId};
use nextpnr::router::common::{bind_route, unroute_net};
use nextpnr::router::maze::Router1Cfg;
use nextpnr::router::{Router, Router1};
use nextpnr::sta::{
    ArcTiming, ClasslessPips, DelayBounds, InterconnectError, InterconnectModel, InterconnectStage,
    NetSink, SinkTiming, StaSession, StaSessionError, Untimed,
};
use nextpnr::timing::DelayT;
use rstest::rstest;

const SLACK_TOLERANCE_NS: f64 = 1e-9;
const LOGIC_TILE: (i32, i32) = (3, 3);
const DETOUR_TILE: (i32, i32) = (2, 4);
const MOVED_LUT_TILE: (i32, i32) = (4, 4);
const CLOCK_PAD_TILE: (i32, i32) = (1, 0);
const OUTPUT_PAD_TILE: (i32, i32) = (9, 3);
const CAPTURE_SLICE: usize = 3;

const CLK_TO_Q_PS: DelayT = 200;
const SETUP_PS: DelayT = 150;
const HOLD_PS: DelayT = 25;
const SWINPUT_PIP_PS: DelayT = 80;
const SWNEIGH_PIP_PS: DelayT = 120;
const NOTIONAL_PIP_PS: DelayT = 100;

const NOTIONAL: ClasslessPips = ClasslessPips::Notional(DelayBounds {
    early_ps: NOTIONAL_PIP_PS,
    late_ps: NOTIONAL_PIP_PS,
});

fn lut_input_to_f_ps(input: usize) -> DelayT {
    150 + 15 * DelayT::try_from(input).expect("LUT input index is small")
}

fn bel_at(ctx: &Context, (x, y): (i32, i32), name: &str) -> BelId {
    let tile = ctx.chipdb().tile_by_xy(x, y);
    ctx.chipdb()
        .bels()
        .find(|&bel| bel.tile() == tile && ctx.chipdb().bel_name(bel) == name)
        .unwrap_or_else(|| panic!("tile ({x}, {y}) has BEL {name}"))
}

fn wire_at(ctx: &Context, (x, y): (i32, i32), name: &str) -> WireId {
    let tile = ctx.chipdb().tile_by_xy(x, y);
    ctx.chipdb()
        .wires()
        .find(|&wire| wire.tile() == tile && ctx.chipdb().wire_name(wire) == name)
        .unwrap_or_else(|| panic!("tile ({x}, {y}) has wire {name}"))
}

fn pip_at(ctx: &Context, tile: (i32, i32), src: &str, dst: &str) -> PipId {
    let chipdb = ctx.chipdb();
    let src_wire = wire_at(ctx, tile, src);
    chipdb
        .wire_info(src_wire)
        .pips_downhill
        .get()
        .iter()
        .map(|&index| PipId::new(src_wire.tile(), index))
        .find(|&pip| {
            let dst_wire = chipdb.pip_dst_wire(pip);
            dst_wire.tile() == src_wire.tile() && chipdb.wire_name(dst_wire) == dst
        })
        .unwrap_or_else(|| panic!("tile {tile:?} has pip {src} -> {dst}"))
}

fn place(ctx: &mut Context, cell: CellId, tile: (i32, i32), bel: &str) {
    let bel = bel_at(ctx, tile, bel);
    assert!(ctx.bind_bel(bel, cell, PlaceStrength::Placer));
}

struct Chain {
    clk: NetId,
    q: NetId,
    d: NetId,
    lut: CellId,
    lut_input: usize,
}

impl Chain {
    fn q_sink(&self) -> NetSink {
        NetSink {
            net: self.q,
            user: 0,
        }
    }

    fn d_sink(&self) -> NetSink {
        NetSink {
            net: self.d,
            user: 0,
        }
    }
}

fn placed_chain(ctx: &mut Context, launch_slice: usize, period_ps: DelayT) -> Chain {
    let lut_input = (2 * launch_slice + 1) % 6;
    let clk = add_net(ctx, "clk");
    ctx.design.net_edit(clk).set_clock_constraint(period_ps);
    let pad = add_cell(ctx, "$io$clk", "IOB");
    drive(ctx, clk, pad, "O");
    place(ctx, pad, CLOCK_PAD_TILE, "IO0");
    let (launch, lut, capture) = (
        add_cell(ctx, "launch", "DFF"),
        add_cell(ctx, "lut", "LUT6"),
        add_cell(ctx, "capture", "DFF"),
    );
    sink(ctx, clk, launch, "CLK");
    sink(ctx, clk, capture, "CLK");
    let (q, d) = (add_net(ctx, "q"), add_net(ctx, "d"));
    drive(ctx, q, launch, "Q");
    sink(ctx, q, lut, &format!("I[{lut_input}]"));
    drive(ctx, d, lut, "F");
    sink(ctx, d, capture, "D");
    place(ctx, launch, LOGIC_TILE, &format!("L{launch_slice}_FF"));
    place(ctx, lut, LOGIC_TILE, &format!("L{CAPTURE_SLICE}_LUT"));
    place(ctx, capture, LOGIC_TILE, &format!("L{CAPTURE_SLICE}_FF"));
    Chain {
        clk,
        q,
        d,
        lut,
        lut_input,
    }
}

#[derive(Debug, Clone, Copy)]
enum QRoute {
    Direct,
    Detour,
}

impl QRoute {
    fn pips(self, ctx: &Context) -> Vec<PipId> {
        let input = format!("L{CAPTURE_SLICE}_I1");
        match self {
            QRoute::Direct => vec![
                pip_at(ctx, LOGIC_TILE, "L0_Q", "SWITCH1"),
                pip_at(ctx, LOGIC_TILE, "SWITCH1", &input),
            ],
            QRoute::Detour => vec![
                pip_at(ctx, LOGIC_TILE, "L0_Q", "SWITCH1"),
                pip_at(ctx, DETOUR_TILE, "NE1", "SWITCH0"),
                pip_at(ctx, LOGIC_TILE, "SW0", "SWITCH67"),
                pip_at(ctx, LOGIC_TILE, "SWITCH67", &input),
            ],
        }
    }

    fn delay_ps(self) -> DelayT {
        match self {
            QRoute::Direct => 2 * SWINPUT_PIP_PS,
            QRoute::Detour => 2 * SWINPUT_PIP_PS + 2 * SWNEIGH_PIP_PS,
        }
    }
}

fn route_by_hand(ctx: &mut Context, net: NetId, source: WireId, pips: &[PipId]) {
    ctx.bind_wire(source, net, PlaceStrength::Strong);
    ctx.design
        .net_edit(net)
        .add_wire(source, None, PlaceStrength::Strong);
    bind_route(ctx, net, pips);
}

fn route_q(ctx: &mut Context, chain: &Chain, route: QRoute) {
    let pips = route.pips(ctx);
    let source = wire_at(ctx, LOGIC_TILE, "L0_Q");
    route_by_hand(ctx, chain.q, source, &pips);
}

fn routed_chain(ctx: &mut Context, route: QRoute, period_ps: DelayT) -> Chain {
    let chain = placed_chain(ctx, 0, period_ps);
    route_q(ctx, &chain, route);
    let lut_out = format!("L{CAPTURE_SLICE}_O");
    let ff_data = format!("L{CAPTURE_SLICE}_D");
    let d_pip = pip_at(ctx, LOGIC_TILE, &lut_out, &ff_data);
    let source = wire_at(ctx, LOGIC_TILE, &lut_out);
    route_by_hand(ctx, chain.d, source, &[d_pip]);
    Router1
        .route_net(ctx, &Router1Cfg::default(), chain.clk)
        .expect("router reaches both clock sinks");
    chain
}

fn session(ctx: &Context, stage: InterconnectStage) -> StaSession {
    let model = InterconnectModel::new(ctx, NOTIONAL).expect("example chipdb has pip classes");
    StaSession::build(ctx, model, stage).expect("session builds the chain")
}

fn assert_ns(actual_ns: f64, expected_ps: DelayT) {
    let expected_ns = f64::from(expected_ps) / 1000.0;
    assert!(
        (actual_ns - expected_ns).abs() < SLACK_TOLERANCE_NS,
        "RustSTA {actual_ns} ns, hand-computed {expected_ns} ns"
    );
}

struct Expected {
    setup_ps: DelayT,
    hold_ps: DelayT,
    criticality: f64,
}

fn expected(period_ps: DelayT, lut_input: usize, q: DelayBounds, d: DelayBounds) -> Expected {
    let lut_ps = lut_input_to_f_ps(lut_input);
    let late_arrival = CLK_TO_Q_PS + q.late_ps + lut_ps + d.late_ps;
    let early_arrival = CLK_TO_Q_PS + q.early_ps + lut_ps + d.early_ps;
    let setup_ps = period_ps - (late_arrival + SETUP_PS);
    let required_ps = period_ps - SETUP_PS;
    Expected {
        setup_ps,
        hold_ps: early_arrival - HOLD_PS,
        criticality: (1.0 - f64::from(setup_ps) / f64::from(required_ps)).clamp(0.0, 1.0),
    }
}

fn assert_chain_timing(session: &StaSession, chain: &Chain, expected: &Expected) {
    for sink in [chain.q_sink(), chain.d_sink()] {
        let timing = session.sink_timing(sink).expect("chain sink is timed");
        assert_ns(
            timing.setup_slack_ns().expect("setup reaches the sink"),
            expected.setup_ps,
        );
        assert_ns(
            timing.hold_slack_ns().expect("hold reaches the sink"),
            expected.hold_ps,
        );
        let criticality = session
            .criticality(sink)
            .expect("chain sink is timed")
            .expect("chain sink has setup slack")
            .value();
        assert!(
            (criticality - expected.criticality).abs() < SLACK_TOLERANCE_NS,
            "criticality {criticality}, hand-computed {}",
            expected.criticality
        );
    }
    let [domain] = session.domains() else {
        panic!("one clock gives one domain pair: {:?}", session.domains());
    };
    assert_eq!((domain.launch, domain.capture), (chain.clk, chain.clk));
    assert_ns(domain.wns_ns, expected.setup_ps);
    assert_ns(domain.tns_ns, expected.setup_ps.min(0));
}

fn uniform(delay_ps: DelayT) -> DelayBounds {
    DelayBounds {
        early_ps: delay_ps,
        late_ps: delay_ps,
    }
}

#[rstest]
#[case::slice0_input1(0, 1000)]
#[case::slice1_input3(1, 1000)]
#[case::slice2_input5_violating(2, 600)]
fn pre_route_slack_is_shortest_pip_path_sum(
    #[case] launch_slice: usize,
    #[case] period_ps: DelayT,
) {
    let mut ctx = common::make_example_context();
    let chain = placed_chain(&mut ctx, launch_slice, period_ps);
    let session = session(&ctx, InterconnectStage::Placed);

    let (q_delay, d_delay) = (uniform(2 * SWINPUT_PIP_PS), uniform(NOTIONAL_PIP_PS));
    assert_eq!(session.connection_delay(chain.q_sink()).unwrap(), q_delay);
    assert_eq!(session.connection_delay(chain.d_sink()).unwrap(), d_delay);
    let expected = expected(period_ps, chain.lut_input, q_delay, d_delay);
    assert_chain_timing(&session, &chain, &expected);
}

#[rstest]
#[case::direct(QRoute::Direct, 1000)]
#[case::detour(QRoute::Detour, 1000)]
#[case::detour_violating(QRoute::Detour, 800)]
fn routed_slack_is_routed_pip_sum(#[case] route: QRoute, #[case] period_ps: DelayT) {
    let mut ctx = common::make_example_context();
    let chain = routed_chain(&mut ctx, route, period_ps);
    let session = session(&ctx, InterconnectStage::Routed);

    let (q_delay, d_delay) = (uniform(route.delay_ps()), uniform(NOTIONAL_PIP_PS));
    assert_eq!(session.connection_delay(chain.q_sink()).unwrap(), q_delay);
    assert_eq!(session.connection_delay(chain.d_sink()).unwrap(), d_delay);
    let expected = expected(period_ps, chain.lut_input, q_delay, d_delay);
    assert_chain_timing(&session, &chain, &expected);
}

#[test]
fn hold_reads_early_bound_and_setup_reads_late_bound() {
    let mut ctx = common::make_example_context();
    let chain = placed_chain(&mut ctx, 0, 1000);
    let mut session = session(&ctx, InterconnectStage::Placed);
    let q_delay = DelayBounds {
        early_ps: 50,
        late_ps: 500,
    };

    session
        .set_connection_delays(&[(chain.q_sink(), q_delay)])
        .expect("q is a timed connection");

    let expected = expected(1000, chain.lut_input, q_delay, uniform(NOTIONAL_PIP_PS));
    assert_chain_timing(&session, &chain, &expected);
}

#[test]
fn constant_input_and_output_pad_leave_chain_slack() {
    let mut ctx = common::make_example_context();
    let chain = placed_chain(&mut ctx, 0, 1000);
    let tie = add_net(&mut ctx, "tie");
    let gnd = add_cell(&mut ctx, "$PACKER_GND", "GND");
    drive(&mut ctx, tie, gnd, "Y");
    sink(&mut ctx, tie, chain.lut, "I[4]");
    let capture = ctx
        .design
        .cell_by_name(ctx.id("capture"))
        .expect("chain has a capture register");
    let dout = add_net(&mut ctx, "dout");
    drive(&mut ctx, dout, capture, "Q");
    let pad = add_cell(&mut ctx, "$io$dout", "IOB");
    sink(&mut ctx, dout, pad, "I");
    place(&mut ctx, pad, OUTPUT_PAD_TILE, "IO0");
    let session = session(&ctx, InterconnectStage::Placed);

    assert_eq!(
        session.sink_timing(NetSink { net: tie, user: 0 }).unwrap(),
        SinkTiming::Untimed(Untimed::ConstantDriver)
    );
    assert!(matches!(
        session.sink_timing(NetSink { net: dout, user: 0 }).unwrap(),
        SinkTiming::Arc(ArcTiming::Propagated { setup: None, .. })
    ));
    let expected = expected(
        1000,
        chain.lut_input,
        uniform(2 * SWINPUT_PIP_PS),
        uniform(NOTIONAL_PIP_PS),
    );
    assert_chain_timing(&session, &chain, &expected);
}

#[derive(Debug, Clone, Copy)]
enum Edit {
    MoveLut,
    RerouteQ,
}

impl Edit {
    fn stage(self) -> InterconnectStage {
        match self {
            Edit::MoveLut => InterconnectStage::Placed,
            Edit::RerouteQ => InterconnectStage::Routed,
        }
    }

    fn design(self, ctx: &mut Context) -> Chain {
        match self {
            Edit::MoveLut => placed_chain(ctx, 0, 1000),
            Edit::RerouteQ => routed_chain(ctx, QRoute::Direct, 1000),
        }
    }

    fn apply(self, ctx: &mut Context, chain: &Chain) {
        match self {
            Edit::MoveLut => {
                let bel = ctx.design.cell(chain.lut).bel.expect("lut is placed");
                ctx.unbind_bel(bel);
                place(
                    ctx,
                    chain.lut,
                    MOVED_LUT_TILE,
                    &format!("L{CAPTURE_SLICE}_LUT"),
                );
            }
            Edit::RerouteQ => {
                unroute_net(ctx, chain.q);
                route_q(ctx, chain, QRoute::Detour);
            }
        }
    }

    fn update(
        self,
        session: &mut StaSession,
        ctx: &Context,
        chain: &Chain,
    ) -> Result<(), StaSessionError> {
        match self {
            Edit::MoveLut => session.update_cells(ctx, self.stage(), &[chain.lut]),
            Edit::RerouteQ => session.update_nets(ctx, self.stage(), &[chain.q]),
        }
        .map(|_| ())
    }
}

fn snapshot(ctx: &Context, session: &StaSession) -> Vec<String> {
    let mut lines: Vec<String> = ctx
        .design
        .iter_alive_nets()
        .map(|(net, info)| {
            let delays: Vec<_> = (0..info.users().len())
                .map(|user| {
                    let user = u32::try_from(user).expect("few users");
                    session.connection_delay(NetSink { net, user }).ok()
                })
                .collect();
            format!("{net:?} {:?} {delays:?}", session.net_sinks(net).unwrap())
        })
        .collect();
    lines.push(format!("{:?}", session.domains()));
    lines
}

#[rstest]
#[case::move_lut(Edit::MoveLut)]
#[case::reroute_q(Edit::RerouteQ)]
fn incremental_update_equals_fresh_build(#[case] edit: Edit) {
    let mut ctx = common::make_example_context();
    let chain = edit.design(&mut ctx);
    let mut incremental = session(&ctx, edit.stage());
    let before = snapshot(&ctx, &incremental);

    edit.apply(&mut ctx, &chain);
    edit.update(&mut incremental, &ctx, &chain)
        .expect("the edited design times");

    let after = snapshot(&ctx, &incremental);
    assert_ne!(before, after, "the edit changes the timing");
    assert_eq!(after, snapshot(&ctx, &session(&ctx, edit.stage())));
}

fn routed_direct(ctx: &mut Context) {
    routed_chain(ctx, QRoute::Direct, 1000);
}

fn placed(ctx: &mut Context) {
    placed_chain(ctx, 0, 1000);
}

fn lut_unplaced(ctx: &mut Context) {
    let chain = placed_chain(ctx, 0, 1000);
    let bel = ctx.design.cell(chain.lut).bel.expect("lut is placed");
    ctx.unbind_bel(bel);
}

#[rstest]
#[case::classless_pip_on_route(
    routed_direct,
    ClasslessPips::Reject,
    InterconnectStage::Routed,
    |err: &StaSessionError| matches!(err, StaSessionError::Interconnect(InterconnectError::ClasslessPip { .. }))
)]
#[case::only_path_has_classless_pip(
    placed,
    ClasslessPips::Reject,
    InterconnectStage::Placed,
    |err: &StaSessionError| matches!(err, StaSessionError::Interconnect(InterconnectError::Unreachable { .. }))
)]
#[case::unplaced_cell(
    lut_unplaced,
    NOTIONAL,
    InterconnectStage::Placed,
    |err: &StaSessionError| matches!(err, StaSessionError::Interconnect(InterconnectError::UnplacedCell { cell }) if cell == "lut")
)]
#[case::unrouted_sink(
    placed,
    NOTIONAL,
    InterconnectStage::Routed,
    |err: &StaSessionError| matches!(err, StaSessionError::Interconnect(InterconnectError::Unrouted { .. }))
)]
fn build_rejects_missing_interconnect_delay(
    #[case] design: fn(&mut Context),
    #[case] classless_pips: ClasslessPips,
    #[case] stage: InterconnectStage,
    #[case] expected: fn(&StaSessionError) -> bool,
) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let model =
        InterconnectModel::new(&ctx, classless_pips).expect("example chipdb has pip classes");
    let err = StaSession::build(&ctx, model, stage).expect_err("a connection has no delay");
    assert!(expected(&err), "unexpected error: {err}");
}

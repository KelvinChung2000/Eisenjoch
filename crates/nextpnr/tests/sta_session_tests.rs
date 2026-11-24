mod common;

use common::{add_cell, add_net, drive, sink};

use std::collections::HashMap;

use nextpnr::chipdb::{parse_constids_inc, BelId, ChipDb, PipId, WireId};
use nextpnr::common::DelayT;
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::netlist::{CellId, NetId};
use nextpnr::packer::passes::remove_nextpnr_iobs;
use nextpnr::router::common::{bind_route, unroute_net};
use nextpnr::router::maze::Router1Cfg;
use nextpnr::router::{Router, Router1};
use nextpnr::sta::{
    ArcTiming, ClasslessPips, ClockId, CritNorm, DelayBounds, InterconnectError, InterconnectModel,
    InterconnectStage, NetSink, PairBudget, PlacedEstimate, Reach, SinkTiming, StaSession,
    StaSessionError, TableEntry, Untimed,
};
use rayon::prelude::*;
use rstest::rstest;
use rustc_hash::FxHashMap;
use ruststa::engine::incr::SessionNodeKey;

const SLACK_TOLERANCE_NS: f64 = 1e-9;
const LOGIC_TILE: (i32, i32) = (3, 3);
const DETOUR_TILE: (i32, i32) = (2, 4);
const MOVED_LUT_TILE: (i32, i32) = (4, 4);
const CLOCK_PAD_TILE: (i32, i32) = (1, 0);
const OUTPUT_PAD_TILE: (i32, i32) = (9, 3);
const CAPTURE_SLICE: usize = 3;

const CLK_TO_Q: DelayBounds = DelayBounds {
    early_ps: 120,
    late_ps: 200,
};
const SETUP_PS: DelayT = 150;
const HOLD_PS: DelayT = 25;
const SWINPUT_PIP_PS: DelayT = 80;
const SWNEIGH_PIP_PS: DelayT = 120;
const NOTIONAL_PIP_PS: DelayT = 100;

const NOTIONAL: ClasslessPips = ClasslessPips::Notional(DelayBounds {
    early_ps: NOTIONAL_PIP_PS,
    late_ps: NOTIONAL_PIP_PS,
});

fn lut_input_to_f(input: usize) -> DelayBounds {
    let input = DelayT::try_from(input).expect("LUT input index is small");
    DelayBounds {
        early_ps: 100 + 10 * input,
        late_ps: 150 + 15 * input,
    }
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

fn capture_register(ctx: &Context) -> CellId {
    ctx.design
        .cell_by_name(ctx.id("capture"))
        .expect("chain has a capture register")
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
    StaSession::build(ctx, model(ctx, NOTIONAL, PlacedEstimate::DelayTable), stage)
        .expect("session builds the chain")
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
    fmax_mhz: f64,
}

fn expected(period_ps: DelayT, lut_input: usize, q: DelayBounds, d: DelayBounds) -> Expected {
    let arrival = CLK_TO_Q + q + lut_input_to_f(lut_input) + d;
    let (late_arrival, early_arrival) = (arrival.late_ps, arrival.early_ps);
    let setup_ps = period_ps - (late_arrival + SETUP_PS);
    let required_ps = period_ps - SETUP_PS;
    Expected {
        setup_ps,
        hold_ps: early_arrival - HOLD_PS,
        criticality: (1.0 - f64::from(setup_ps) / f64::from(required_ps)).clamp(0.0, 1.0),
        fmax_mhz: 1e6 / f64::from(late_arrival + SETUP_PS),
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
    assert_eq!(domain.launch, domain.capture);
    assert_ns(domain.wns_ns, expected.setup_ps);
    assert_ns(domain.tns_ns, expected.setup_ps.min(0));
    assert_eq!(
        (domain.endpoints, domain.failing_endpoints),
        (1, usize::from(expected.setup_ps < 0))
    );
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
    assert_chain_report(&ctx, &session, &expected);
}

#[rstest]
#[case::launch("launch")]
#[case::capture("capture")]
fn aliased_register_times_as_its_bel_bucket(#[case] register: &str) {
    let mut ctx = common::make_example_context();
    let chain = placed_chain(&mut ctx, 0, 1000);
    ctx.add_cell_type_alias("DFF_ALT", "DFF");
    let (cell, alias) = (ctx.design.cell_by_name(ctx.id(register)).unwrap(), ctx.id("DFF_ALT"));
    ctx.design.cell_edit(cell).set_type(alias);
    let session = session(&ctx, InterconnectStage::Placed);

    let (q_delay, d_delay) = (uniform(2 * SWINPUT_PIP_PS), uniform(NOTIONAL_PIP_PS));
    assert_chain_timing(&session, &chain, &expected(1000, chain.lut_input, q_delay, d_delay));
}

fn assert_chain_report(ctx: &Context, session: &StaSession, expected: &Expected) {
    let report = session.report(ctx, 1).expect("the chain reports");
    let [domain] = &report.domains[..] else {
        panic!("one clock gives one domain pair: {report}");
    };
    let fmax_mhz = domain.fmax_mhz.expect("a same-clock pair has an Fmax");
    assert!(
        (fmax_mhz - expected.fmax_mhz).abs() < 1e-6 * expected.fmax_mhz,
        "Fmax {fmax_mhz} MHz, hand-computed {} MHz",
        expected.fmax_mhz
    );
    let [path] = &domain.worst_paths[..] else {
        panic!("one endpoint gives one worst path: {report}");
    };
    assert_ns(path.slack_ns, expected.setup_ps);
    let pins: Vec<&str> = path.pins.iter().map(|pin| pin.pin.as_str()).collect();
    assert_eq!(
        (pins.first(), pins.last()),
        (Some(&"launch/Q"), Some(&"capture/D"))
    );
}

#[rstest]
#[case::met(1000)]
#[case::violating(400)]
fn ideal_wires_time_an_unplaced_chain_at_zero_interconnect_delay(#[case] period_ps: DelayT) {
    let mut ctx = common::make_example_context();
    let chain = placed_chain(&mut ctx, 0, period_ps);
    let bel = ctx.design.cell(chain.lut).bel.expect("lut is placed");
    ctx.unbind_bel(bel);
    let model = InterconnectModel::new(&ctx, NOTIONAL, PlacedEstimate::DelayTable)
        .expect("example chipdb has pip classes");
    let session = StaSession::build_ideal(&ctx, model).expect("ideal wires need no placement");

    for sink in [chain.q_sink(), chain.d_sink()] {
        assert_eq!(session.connection_delay(sink).unwrap(), DelayBounds::ZERO);
    }
    let expected = expected(
        period_ps,
        chain.lut_input,
        DelayBounds::ZERO,
        DelayBounds::ZERO,
    );
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
    let capture = capture_register(&ctx);
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
    MoveCapture,
    RerouteQ,
}

impl Edit {
    fn stage(self) -> InterconnectStage {
        match self {
            Edit::MoveLut | Edit::MoveCapture => InterconnectStage::Placed,
            Edit::RerouteQ => InterconnectStage::Routed,
        }
    }

    fn design(self, ctx: &mut Context) -> Chain {
        match self {
            Edit::MoveLut | Edit::MoveCapture => placed_chain(ctx, 0, 1000),
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
            Edit::MoveCapture => {
                let capture = capture_register(ctx);
                let bel = ctx.design.cell(capture).bel.expect("capture is placed");
                ctx.unbind_bel(bel);
                place(
                    ctx,
                    capture,
                    MOVED_LUT_TILE,
                    &format!("L{CAPTURE_SLICE}_FF"),
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
            Edit::MoveCapture => session.update_cells(ctx, self.stage(), &[capture_register(ctx)]),
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
#[case::move_capture_on_clock_net(Edit::MoveCapture)]
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

fn routed_direct(ctx: &mut Context) -> Chain {
    routed_chain(ctx, QRoute::Direct, 1000)
}

fn placed(ctx: &mut Context) -> Chain {
    placed_chain(ctx, 0, 1000)
}

fn lut_unplaced(ctx: &mut Context) -> Chain {
    let chain = placed_chain(ctx, 0, 1000);
    let bel = ctx.design.cell(chain.lut).bel.expect("lut is placed");
    ctx.unbind_bel(bel);
    chain
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
    #[case] design: fn(&mut Context) -> Chain,
    #[case] classless_pips: ClasslessPips,
    #[case] stage: InterconnectStage,
    #[values(PlacedEstimate::DelayTable, PlacedEstimate::ExactSearch)]
    placed_estimate: PlacedEstimate,
    #[case] expected: fn(&StaSessionError) -> bool,
) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let err = StaSession::build(&ctx, model(&ctx, classless_pips, placed_estimate), stage)
        .expect_err("a connection has no delay");
    assert!(expected(&err), "unexpected error: {err}");
}

fn npnr_baseline_fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npnr_baseline")
        .join(name)
}

fn placed_npnr_baseline(trim_top_ports: bool) -> Context {
    let constids = std::fs::read_to_string(npnr_baseline_fixture("constids.inc"))
        .expect("fixture has constids");
    let chipdb = ChipDb::load_with_known_constids(
        &npnr_baseline_fixture("shared_chipdb.bin"),
        &parse_constids_inc(&constids),
    )
    .expect("fixture chipdb loads");
    let mut ctx = Context::new(chipdb);
    let json = std::fs::read_to_string(npnr_baseline_fixture("placed.json"))
        .expect("fixture has a placed design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("placed design parses");
    if trim_top_ports {
        remove_nextpnr_iobs(&mut ctx).expect("pseudo-cells trim");
    }
    let bels: HashMap<String, BelId> = ctx
        .chipdb()
        .bels()
        .map(|bel| {
            let loc = ctx.chipdb().bel_loc(bel);
            let name = format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel));
            (name, bel)
        })
        .collect();
    let bel_attr = ctx.id("NEXTPNR_BEL");
    let bindings: Vec<(BelId, CellId)> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(cell, info)| Some((bels[info.attrs.get(&bel_attr)?.as_str().as_str()], cell)))
        .collect();
    for (bel, cell) in bindings {
        assert!(ctx.bind_bel(bel, cell, PlaceStrength::Placer));
    }
    ctx
}

fn constrain_baseline_clock_net(ctx: &mut Context, period_ps: DelayT) {
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

const BASELINE_SDC_CLOCK: &str = "create_clock -name sys -period 4 [get_ports {clk_pad}]\n";

const BASELINE_PERIOD_PS: DelayT = 4000;
const BASELINE_MET_PERIOD_PS: DelayT = 20000;
const BASELINE_VIOLATING_PERIOD_PS: DelayT = 3000;

#[rstest]
#[case::with_top_port_cells(false)]
#[case::top_port_cells_trimmed(true)]
fn io_buffers_time_as_top_level_ports(#[case] trim_top_ports: bool) {
    let mut ctx = placed_npnr_baseline(trim_top_ports);
    constrain_baseline_clock_net(&mut ctx, BASELINE_PERIOD_PS);
    let session = session(&ctx, InterconnectStage::Placed);
    let timing = session.sta().session();
    let top_ports = ["clk_pad", "rst", "din[0]", "din[15]", "dout[0]", "dout[15]"];
    for name in top_ports {
        assert!(
            matches!(timing.key_of_name(name), Some(SessionNodeKey::PortBit(_))),
            "{name} is a top-level port"
        );
    }
    let [domain] = session.domains() else {
        panic!("one clock gives one domain pair: {:?}", session.domains());
    };
    assert!(domain.wns_ns.is_finite());
    let din_buf = ctx
        .design
        .cell_by_name(ctx.id("$iopadmap$top.din_9"))
        .expect("fixture has an input buffer");
    let pad_net = ctx
        .design
        .cell(din_buf)
        .port_net(ctx.id("PAD"))
        .expect("the buffer has a pad net");
    let pad_sinks = session
        .net_sinks(pad_net)
        .expect("the pad net is a design net");
    assert!(pad_sinks
        .iter()
        .all(|sink| *sink == SinkTiming::Untimed(Untimed::OffChip)));
}

#[test]
fn user_sdc_reads_ns_like_the_net_clock_constraint() {
    let mut by_net = placed_npnr_baseline(false);
    constrain_baseline_clock_net(&mut by_net, BASELINE_PERIOD_PS);
    let mut by_sdc = placed_npnr_baseline(false);
    by_sdc.design.sdc = BASELINE_SDC_CLOCK.to_owned();
    let wns = |ctx: &Context| {
        let session = session(ctx, InterconnectStage::Placed);
        let [domain] = session.domains() else {
            panic!("one clock gives one domain pair: {:?}", session.domains());
        };
        domain.wns_ns
    };
    assert_eq!(wns(&by_net), wns(&by_sdc));
}

#[rstest]
#[case::launched_by_the_capture_clock("", "sys")]
#[case::launched_by_a_virtual_clock("create_clock -name vclk -period 4\n", "vclk")]
fn input_delay_from_user_sdc_seeds_the_pad_arrival(
    #[case] launch_clock: &str,
    #[case] launch: &str,
) {
    const INPUT_DELAY_NS: f64 = 10.0;
    let mut ctx = placed_npnr_baseline(false);
    ctx.design.sdc = format!(
        "{BASELINE_SDC_CLOCK}{launch_clock}set_input_delay -clock {launch} {INPUT_DELAY_NS} [get_ports {{din\\[*\\]}}]\n"
    );
    let report = session(&ctx, InterconnectStage::Placed)
        .report(&ctx, usize::MAX)
        .expect("the fixture reports");
    let from_inputs = report
        .domains
        .iter()
        .filter(|domain| domain.launch == launch)
        .flat_map(|domain| &domain.worst_paths)
        .find(|path| path.pins[0].pin.starts_with("$iopadmap$top.din_"))
        .unwrap_or_else(|| panic!("{launch} launches a path from din: {report}"));
    assert!(
        (from_inputs.pins[0].arrival_ns - INPUT_DELAY_NS).abs() < SLACK_TOLERANCE_NS,
        "the input delay is the pad arrival: {report}"
    );
}

const LOGIC_ROWS: [i32; 7] = [1, 2, 3, 4, 5, 6, 8];
const LOGIC_COLUMNS: std::ops::RangeInclusive<i32> = 1..=8;
const FANOUT_STRIDES: [usize; 3] = [1, 7, 29];
const INPUT_PAD_TILES: [(i32, i32); 3] = [(0, 4), (5, 0), (9, 8)];
const OUTPUT_PAD_TILES: [(i32, i32); 2] = [(0, 2), (6, 9)];
const BRAM_TILE: (i32, i32) = (4, 7);

struct Scattered {
    luts: Vec<CellId>,
    ffs: Vec<CellId>,
}

fn scattered(ctx: &mut Context) -> Scattered {
    let clk = add_net(ctx, "clk");
    ctx.design.net_edit(clk).set_clock_constraint(5000);
    let clock_pad = add_cell(ctx, "$io$clk", "IOB");
    drive(ctx, clk, clock_pad, "O");
    place(ctx, clock_pad, CLOCK_PAD_TILE, "IO0");
    let tiles: Vec<(i32, i32)> = LOGIC_ROWS
        .iter()
        .flat_map(|&y| LOGIC_COLUMNS.map(move |x| (x, y)))
        .collect();
    let (mut luts, mut ffs) = (Vec::new(), Vec::new());
    for (k, &tile) in tiles.iter().enumerate() {
        let slice = k % 8;
        let ff = add_cell(ctx, &format!("ff{k}"), "DFF");
        let lut = add_cell(ctx, &format!("lut{k}"), "LUT6");
        place(ctx, ff, tile, &format!("L{slice}_FF"));
        place(ctx, lut, tile, &format!("L{slice}_LUT"));
        sink(ctx, clk, ff, "CLK");
        let d = add_net(ctx, &format!("d{k}"));
        drive(ctx, d, lut, "F");
        sink(ctx, d, ff, "D");
        luts.push(lut);
        ffs.push(ff);
    }
    for (k, &ff) in ffs.iter().enumerate() {
        let q = add_net(ctx, &format!("q{k}"));
        drive(ctx, q, ff, "Q");
        for (input, stride) in FANOUT_STRIDES.iter().enumerate() {
            let lut = luts[(k + stride) % luts.len()];
            sink(ctx, q, lut, &format!("I[{input}]"));
        }
    }
    for (k, &tile) in INPUT_PAD_TILES.iter().enumerate() {
        let pad = add_cell(ctx, &format!("$io$in{k}"), "IOB");
        place(ctx, pad, tile, "IO1");
        let net = add_net(ctx, &format!("in{k}"));
        drive(ctx, net, pad, "O");
        for lut in luts.iter().skip(k).step_by(11) {
            sink(ctx, net, *lut, "I[3]");
        }
    }
    for (k, &tile) in OUTPUT_PAD_TILES.iter().enumerate() {
        let pad = add_cell(ctx, &format!("$io$out{k}"), "IOB");
        place(ctx, pad, tile, "IO0");
        let q = ctx
            .design
            .net_by_name(ctx.id(&format!("q{}", 17 * k)))
            .expect("q nets exist");
        sink(ctx, q, pad, "I");
    }
    Scattered { luts, ffs }
}

fn scattered_with_bram(ctx: &mut Context) -> Scattered {
    let design = scattered(ctx);
    let bram = add_cell(ctx, "bram", "BRAM_512X16");
    place(ctx, bram, BRAM_TILE, "RAM");
    let dout = add_net(ctx, "dout");
    drive(ctx, dout, bram, "DO[0]");
    for lut in design.luts.iter().step_by(5) {
        sink(ctx, dout, *lut, "I[4]");
    }
    let q = ctx
        .design
        .cell(design.ffs[30])
        .port_net(ctx.id("Q"))
        .expect("every register drives a q net");
    sink(ctx, q, bram, "DI[3]");
    design
}

fn model(
    ctx: &Context,
    classless_pips: ClasslessPips,
    placed: PlacedEstimate,
) -> InterconnectModel {
    InterconnectModel::new(ctx, classless_pips, placed).expect("example chipdb has pip classes")
}

#[rstest]
#[case::notional(NOTIONAL)]
#[case::reject(ClasslessPips::Reject)]
fn table_entries_equal_exact_search_at_their_sample(#[case] classless_pips: ClasslessPips) {
    let ctx = common::make_example_context();
    let table_model = model(&ctx, classless_pips, PlacedEstimate::DelayTable);
    let table = table_model
        .delay_table(ctx.chipdb())
        .expect("the model holds a delay table");
    let chipdb = ctx.chipdb();
    let mut by_source: FxHashMap<WireId, Vec<(WireId, Reach)>> = FxHashMap::default();
    for (key, entry) in table.entries() {
        let TableEntry::Sampled { sample, reach } = *entry else {
            continue;
        };
        let (x, y) = chipdb.tile_xy(sample);
        let sink_tile = chipdb.tile_by_xy(x + key.offset.dx, y + key.offset.dy);
        by_source
            .entry(WireId::new(sample, key.source.wire))
            .or_default()
            .push((WireId::new(sink_tile, key.sink.wire), reach));
    }
    let mismatches: Vec<(WireId, WireId, Reach, Option<DelayBounds>)> = by_source
        .par_iter()
        .flat_map_iter(|(&source, sinks)| {
            let exact = table_model.exact_delays_from(chipdb, source);
            sinks.iter().filter_map(move |&(sink, reach)| {
                let searched = exact.get(&sink).copied();
                let agrees = match reach {
                    Reach::Delay(delay) => searched == Some(delay),
                    Reach::Unreachable => searched.is_none(),
                };
                (!agrees).then_some((source, sink, reach, searched))
            })
        })
        .collect();
    assert!(!by_source.is_empty(), "the table holds sampled entries");
    assert_eq!(mismatches, Vec::new());
}

#[rstest]
#[case::luts_ffs_pads(scattered)]
#[case::luts_ffs_pads_bram(scattered_with_bram)]
fn table_estimate_equals_exact_search_on_placed_design(
    #[case] design: fn(&mut Context) -> Scattered,
) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let table = model(&ctx, NOTIONAL, PlacedEstimate::DelayTable);
    let exact = model(&ctx, NOTIONAL, PlacedEstimate::ExactSearch);
    let mut connections = 0;
    for (net, info) in ctx.design.iter_alive_nets() {
        if info.clock_constraint > 0 {
            continue;
        }
        let users: Vec<u32> = (0..info.users().len())
            .filter(|&user| info.users()[user].is_valid())
            .map(|user| u32::try_from(user).expect("few users"))
            .collect();
        let searched = exact
            .net_delays(&ctx, InterconnectStage::Placed, net, &users)
            .expect("every data connection has a timed path");
        let looked_up = table
            .net_delays(&ctx, InterconnectStage::Placed, net, &users)
            .expect("the table holds every data connection");
        connections += searched.len();
        assert_eq!(looked_up, searched, "net {}", ctx.name_of(info.name));
    }
    assert!(
        connections > 150,
        "the design has {connections} connections"
    );
}

#[rstest]
#[case::chain(|ctx: &mut Context| { placed(ctx); })]
#[case::scattered(|ctx: &mut Context| { scattered(ctx); })]
fn table_session_equals_exact_session(#[case] design: fn(&mut Context)) {
    let mut ctx = common::make_example_context();
    design(&mut ctx);
    let build = |placed_estimate| {
        StaSession::build(
            &ctx,
            model(&ctx, NOTIONAL, placed_estimate),
            InterconnectStage::Placed,
        )
        .expect("session builds the design")
    };
    let table = build(PlacedEstimate::DelayTable);
    let exact = build(PlacedEstimate::ExactSearch);
    assert_eq!(snapshot(&ctx, &table), snapshot(&ctx, &exact));
}

#[derive(Debug, Clone, Copy)]
enum ClockDefinition {
    NetConstraint,
    UserSdcNamedApart,
    Undefined,
}

impl ClockDefinition {
    const USER_SDC_NAME: &str = "sys";

    fn apply(self, ctx: &mut Context, clk: NetId) {
        let period_ps = ctx.design.net(clk).clock_constraint;
        match self {
            ClockDefinition::NetConstraint => {}
            ClockDefinition::UserSdcNamedApart => {
                ctx.design.net_edit(clk).set_clock_constraint(0);
                ctx.design.sdc = format!(
                    "create_clock -name {} -period {} [get_ports {{clk}}]\n",
                    Self::USER_SDC_NAME,
                    f64::from(period_ps) / 1000.0
                );
            }
            ClockDefinition::Undefined => {
                ctx.design.net_edit(clk).set_clock_constraint(0);
            }
        }
    }

    fn rejects_clock_sink(
        self,
        result: Result<impl Sized, StaSessionError>,
        sink: NetSink,
    ) -> bool {
        match (self, result) {
            (
                ClockDefinition::NetConstraint | ClockDefinition::UserSdcNamedApart,
                Err(StaSessionError::IdealClockSink(rejected)),
            )
            | (ClockDefinition::Undefined, Err(StaSessionError::UnclockedClockSink(rejected))) => {
                rejected == sink
            }
            _ => false,
        }
    }

    fn clock_sink_timing(self) -> SinkTiming {
        match self {
            ClockDefinition::NetConstraint | ClockDefinition::UserSdcNamedApart => {
                SinkTiming::Arc(ArcTiming::Propagated {
                    setup: None,
                    hold_slack_ns: None,
                })
            }
            ClockDefinition::Undefined => SinkTiming::Arc(ArcTiming::NoArrival),
        }
    }
}

#[rstest]
#[case::placed(placed, InterconnectStage::Placed)]
#[case::routed(routed_direct, InterconnectStage::Routed)]
fn clock_sinks_carry_no_delay(
    #[case] design: fn(&mut Context) -> Chain,
    #[case] stage: InterconnectStage,
    #[values(
        ClockDefinition::NetConstraint,
        ClockDefinition::UserSdcNamedApart,
        ClockDefinition::Undefined
    )]
    clock: ClockDefinition,
) {
    let mut ctx = common::make_example_context();
    let chain = design(&mut ctx);
    clock.apply(&mut ctx, chain.clk);
    let mut session = session(&ctx, stage);
    for user in 0..2 {
        let clock_sink = NetSink {
            net: chain.clk,
            user,
        };
        assert!(clock.rejects_clock_sink(session.connection_delay(clock_sink), clock_sink));
        assert_eq!(
            session.sink_timing(clock_sink).unwrap(),
            clock.clock_sink_timing()
        );
        assert!(clock.rejects_clock_sink(
            session.set_connection_delays(&[(clock_sink, uniform(1))]),
            clock_sink
        ));
    }
}

#[rstest]
#[case::placed(placed, InterconnectStage::Placed)]
#[case::routed(routed_direct, InterconnectStage::Routed)]
fn unclocked_register_is_unconstrained(
    #[case] design: fn(&mut Context) -> Chain,
    #[case] stage: InterconnectStage,
) {
    let mut ctx = common::make_example_context();
    let chain = design(&mut ctx);
    ClockDefinition::Undefined.apply(&mut ctx, chain.clk);
    let session = session(&ctx, stage);
    for sink in [chain.q_sink(), chain.d_sink()] {
        assert_eq!(
            session.sink_timing(sink).unwrap(),
            SinkTiming::Arc(ArcTiming::NoArrival)
        );
    }
    assert!(session.domains().is_empty());
}

fn clock_also_drives_lut(ctx: &mut Context) -> (Chain, NetSink) {
    let chain = placed(ctx);
    let user = u32::try_from(ctx.design.net(chain.clk).users().len()).expect("few users");
    sink(ctx, chain.clk, chain.lut, "I[4]");
    let data_sink = NetSink {
        net: chain.clk,
        user,
    };
    (chain, data_sink)
}

#[rstest]
fn clock_net_data_sink_keeps_its_delay(
    #[values(
        ClockDefinition::NetConstraint,
        ClockDefinition::UserSdcNamedApart,
        ClockDefinition::Undefined
    )]
    clock: ClockDefinition,
) {
    let mut ctx = common::make_example_context();
    let (chain, data_sink) = clock_also_drives_lut(&mut ctx);
    clock.apply(&mut ctx, chain.clk);
    let build = |placed_estimate| {
        StaSession::build(
            &ctx,
            model(&ctx, NOTIONAL, placed_estimate),
            InterconnectStage::Placed,
        )
        .expect("session builds the design")
    };
    let (table, exact) = (
        build(PlacedEstimate::DelayTable),
        build(PlacedEstimate::ExactSearch),
    );
    for user in 0..data_sink.user {
        let clock_sink = NetSink {
            net: chain.clk,
            user,
        };
        assert!(clock.rejects_clock_sink(table.connection_delay(clock_sink), clock_sink));
    }
    assert_eq!(
        table.connection_delay(data_sink).unwrap(),
        exact.connection_delay(data_sink).unwrap()
    );
    assert!(table.connection_delay(data_sink).unwrap().late_ps > 0);
}

#[rstest]
#[case::net_constraint(ClockDefinition::NetConstraint, "clk")]
#[case::user_sdc_named_apart(ClockDefinition::UserSdcNamedApart, ClockDefinition::USER_SDC_NAME)]
fn propagated_clock_is_rejected(#[case] clock: ClockDefinition, #[case] name: &str) {
    let mut ctx = common::make_example_context();
    let chain = placed(&mut ctx);
    clock.apply(&mut ctx, chain.clk);
    ctx.design
        .sdc
        .push_str(&format!("set_propagated_clock [get_clocks {{{name}}}]\n"));
    let built = StaSession::build(
        &ctx,
        model(&ctx, NOTIONAL, PlacedEstimate::DelayTable),
        InterconnectStage::Placed,
    );
    assert!(matches!(
        built,
        Err(StaSessionError::PropagatedClock { clock }) if clock == name
    ));
}

#[rstest]
#[case::met_required_time(CritNorm::RequiredTime, 1.0, 4.0, 0.5, 0.75)]
#[case::met_wns_relaxed(CritNorm::WnsRelaxed, 1.0, 4.0, 0.5, 0.75)]
#[case::failing_required_time_saturates(CritNorm::RequiredTime, -0.5, 4.0, -1.0, 1.0)]
#[case::failing_wns_relaxed_is_marquardt_eq6(CritNorm::WnsRelaxed, -0.5, 4.0, -1.0, 0.9)]
#[case::worst_wns_relaxed(CritNorm::WnsRelaxed, -1.0, 4.0, -1.0, 1.0)]
#[case::slack_above_budget(CritNorm::WnsRelaxed, 6.0, 4.0, -1.0, 0.0)]
#[case::no_budget(CritNorm::RequiredTime, 0.0, 0.0, 0.0, 1.0)]
fn pair_criticality_normalises_slack(
    #[case] norm: CritNorm,
    #[case] slack_ns: f64,
    #[case] d_req_ns: f64,
    #[case] wns_ns: f64,
    #[case] expected: f64,
) {
    let budget = PairBudget {
        launch: ClockId(0),
        capture: ClockId(0),
        d_req_ns,
        wns_ns,
    };
    let actual = norm.pair_criticality(slack_ns, budget);
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn baseline_timed_sinks(ctx: &Context) -> Vec<NetSink> {
    ctx.design
        .iter_alive_nets()
        .flat_map(|(net, info)| {
            (0..info.users().len()).map(move |user| NetSink {
                net,
                user: u32::try_from(user).expect("net user index fits in u32"),
            })
        })
        .collect()
}

fn baseline_session_at(period_ps: DelayT) -> (Context, StaSession) {
    let mut ctx = placed_npnr_baseline(true);
    constrain_baseline_clock_net(&mut ctx, period_ps);
    let session = session(&ctx, InterconnectStage::Placed);
    (ctx, session)
}

#[rstest]
#[case::met("create_clock -name sys -period 20 [get_ports {clk_pad}]\n", 1)]
#[case::violating("create_clock -name sys -period 3 [get_ports {clk_pad}]\n", 1)]
#[case::two_pairs(
    "create_clock -name sys -period 3 [get_ports {clk_pad}]\n\
     create_clock -name vclk -period 3\n\
     set_input_delay -clock vclk 1 [get_ports {din\\[*\\]}]\n",
    2
)]
fn required_time_norm_is_the_ruststa_criticality(#[case] sdc: &str, #[case] pairs: usize) {
    let mut ctx = placed_npnr_baseline(false);
    ctx.design.sdc = sdc.to_owned();
    let session = session(&ctx, InterconnectStage::Placed);
    assert_eq!(session.pair_budgets().len(), pairs);
    for sink in baseline_timed_sinks(&ctx) {
        let ruststa = session
            .sink_timing(sink)
            .expect("a live user is a sink")
            .criticality()
            .map(|crit| crit.value());
        let normalised = session
            .normalised_criticality(sink, CritNorm::RequiredTime)
            .expect("a live user is a sink");
        assert_eq!(normalised, ruststa, "{sink:?}");
    }
}

#[rstest]
#[case::met(BASELINE_MET_PERIOD_PS, true)]
#[case::violating(BASELINE_VIOLATING_PERIOD_PS, false)]
fn wns_relaxed_norm_desaturates_only_a_failing_pair(#[case] period_ps: DelayT, #[case] met: bool) {
    const SATURATED: f64 = 1.0 - 1e-9;
    let (ctx, session) = baseline_session_at(period_ps);
    let [budget] = session.pair_budgets() else {
        panic!(
            "one clock gives one domain pair: {:?}",
            session.pair_budgets()
        );
    };
    let crits = |norm| -> Vec<f64> {
        baseline_timed_sinks(&ctx)
            .into_iter()
            .filter_map(|sink| {
                session
                    .normalised_criticality(sink, norm)
                    .expect("a live user is a sink")
            })
            .collect()
    };
    let (required, relaxed) = (crits(CritNorm::RequiredTime), crits(CritNorm::WnsRelaxed));
    let saturated = |crits: &[f64]| crits.iter().filter(|&&crit| crit >= SATURATED).count();
    assert_eq!(budget.wns_ns >= 0.0, met, "pair WNS {}", budget.wns_ns);
    if met {
        assert_eq!(relaxed, required);
        return;
    }
    let worst = relaxed.iter().copied().fold(0.0, f64::max);
    assert!(
        worst >= SATURATED,
        "the worst connection is critical: {worst}"
    );
    assert!(
        saturated(&relaxed) < saturated(&required),
        "{} relaxed against {} saturated",
        saturated(&relaxed),
        saturated(&required)
    );
}

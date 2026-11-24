use std::path::{Path, PathBuf};

use nextpnr::chipdb::{CellPinRegArcPod, ChipDb, TimingValue};
use nextpnr::common::DelayT;
use nextpnr::context::{BelPin, Context};
use nextpnr::frontend::parse_json;
use nextpnr::netlist::CellPin;
use nextpnr::packer;
use nextpnr::placer::heap::PlacerHeapCfg;
use nextpnr::placer::{Placer, PlacerHeap};
use nextpnr::read_packed;
use nextpnr::sta::{
    DelayBounds, InterconnectModel, InterconnectStage, PlacedEstimate, StaSession,
    NOTIONAL_CLASSLESS_PIPS,
};
use rstest::rstest;

fn repo_file(relative: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    assert!(path.exists(), "{} is missing", path.display());
    path
}

fn xc7_context() -> Context {
    let db = ChipDb::load(&repo_file("chip_database/xc7_standard.bin")).expect("load xc7 chipdb");
    Context::new(db)
}

#[derive(Debug, Clone, Copy)]
enum Arc {
    Comb { from: &'static str },
    ClockToQ,
    Setup,
    Hold,
}

fn corners(value: TimingValue) -> [i32; 4] {
    [
        value.fast_min,
        value.fast_max,
        value.slow_min,
        value.slow_max,
    ]
}

fn rising_clk_arc<'a>(ctx: &Context, arcs: &'a [CellPinRegArcPod]) -> &'a CellPinRegArcPod {
    let [arc] = arcs else {
        panic!("expected one register arc, got {}", arcs.len())
    };
    let clock: i32 = unsafe { read_packed!(*arc, clock) };
    let edge: i32 = unsafe { read_packed!(*arc, edge) };
    assert_eq!((clock, edge), (ctx.id("CLK").index(), 0));
    arc
}

#[rstest]
#[case::lut_a1_o6(
    "LUT6",
    "F",
    Arc::Comb { from: "I[0]" },
    [45, 56, 100, 124]
)]
#[case::lut_a6_o6(
    "LUT6",
    "F",
    Arc::Comb { from: "I[5]" },
    [45, 56, 100, 124]
)]
#[case::bufgctrl_i_o("BUFG", "O", Arc::Comb { from: "I" }, [26, 29, 91, 96])]
#[case::ff_clk_q("DFF", "Q", Arc::ClockToQ, [99, 147, 244, 362])]
#[case::ff_setup_din("DFF", "D", Arc::Setup, [-74, -45, -74, -45])]
#[case::ff_hold_din("DFF", "D", Arc::Hold, [181, 262, 181, 262])]
#[ignore = "needs chip_database/xc7_standard.bin from chip_database.gen_xc7"]
fn chipdb_cell_arc_is_the_prjxray_sdf_envelope(
    #[case] cell: &str,
    #[case] pin: &str,
    #[case] arc: Arc,
    #[case] expected_ps: [i32; 4],
) {
    let ctx = xc7_context();
    let chipdb = ctx.chipdb();
    let speed_grade = chipdb.speed_grade(0).expect("speed grade 0");
    let cell_type = chipdb
        .cell_timing_index(speed_grade, ctx.id(cell).index())
        .unwrap_or_else(|| panic!("{cell} has cell timing"));
    let pin_id = ctx.id(pin).index();
    let value = match arc {
        Arc::Comb { from } => chipdb
            .cell_comb_timing(speed_grade, cell_type, ctx.id(from).index(), pin_id)
            .unwrap_or_else(|| panic!("{cell} has {from} -> {pin}")),
        Arc::ClockToQ | Arc::Setup | Arc::Hold => {
            let arcs = chipdb
                .cell_reg_arcs(speed_grade, cell_type, pin_id)
                .unwrap_or_else(|| panic!("{cell}.{pin} has register arcs"));
            let reg = rising_clk_arc(&ctx, arcs);
            match arc {
                Arc::ClockToQ => unsafe { read_packed!(*reg, clk_q) },
                Arc::Setup => unsafe { read_packed!(*reg, setup) },
                Arc::Hold => unsafe { read_packed!(*reg, hold) },
                Arc::Comb { .. } => unreachable!(),
            }
        }
    };
    assert_eq!(corners(value), expected_ps);
}

fn stereovision3(period_ps: DelayT) -> Context {
    let mut ctx = xc7_context();
    let json = std::fs::read_to_string(repo_file("benchmark/output/stereovision3.json"))
        .expect("read stereovision3");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse stereovision3");
    packer::pack(&mut ctx, None).expect("pack");
    for clock in ["tm3_clk_v0", "tm3_clk_v2"] {
        let net = ctx
            .design
            .net_by_name(ctx.id(clock))
            .unwrap_or_else(|| panic!("net {clock}"));
        ctx.design.net_edit(net).set_clock_constraint(period_ps);
    }
    ctx
}

#[rstest]
#[case::ideal_wires_10ns(10_000)]
#[ignore = "needs benchmark/output/stereovision3.json and the xc7_standard.bin"]
fn stereovision3_session_has_finite_setup_and_hold_slack(#[case] period_ps: DelayT) {
    let ctx = stereovision3(period_ps);
    let model = InterconnectModel::new(&ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)
        .expect("interconnect model");
    let session = StaSession::build_ideal(&ctx, model).expect("session");

    let mut setup = Vec::new();
    let mut hold = Vec::new();
    for (net, _) in ctx.design.iter_alive_nets() {
        for timing in session.net_sinks(net).expect("net sinks") {
            setup.extend(timing.setup_slack_ns());
            hold.extend(timing.hold_slack_ns());
        }
    }
    assert_eq!(session.domains().len(), 2);
    assert!(!setup.is_empty() && !hold.is_empty());
    assert!(setup.iter().chain(&hold).all(|slack| slack.is_finite()));
    assert!(session.domains().iter().all(|d| d.wns_ns.is_finite()));
}

fn pin_wire(ctx: &Context, pin: CellPin) -> nextpnr::chipdb::WireId {
    let bel = ctx.design.cell(pin.cell).bel.expect("placed cell");
    ctx.bel_pin_wire(BelPin::new(bel, pin.port))
        .expect("placed pin has a wire")
        .id()
}

#[rstest]
#[case::wide_fanout("iic_stop")]
#[case::register_fanout("temp_reg1")]
#[case::single_sink("$abc$5650$new_n276")]
#[ignore = "needs benchmark/output/stereovision3.json and the xc7_standard.bin"]
fn stereovision3_placed_session_matches_exhaustive_search(#[case] net_name: &str) {
    let mut ctx = stereovision3(10_000);
    PlacerHeap
        .place(&mut ctx, &PlacerHeapCfg::default())
        .expect("heap placement");
    let model = || {
        InterconnectModel::new(&ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)
            .expect("interconnect model")
    };
    let session =
        StaSession::build(&ctx, model(), InterconnectStage::Placed).expect("placed session");
    let setup: Vec<f64> = ctx
        .design
        .iter_alive_nets()
        .flat_map(|(net, _)| session.net_sinks(net).expect("net sinks"))
        .filter_map(|timing| timing.setup_slack_ns())
        .collect();
    assert!(!setup.is_empty() && setup.iter().all(|slack| slack.is_finite()));

    let net = ctx
        .design
        .net_by_name(ctx.id(net_name))
        .unwrap_or_else(|| panic!("net {net_name}"));
    let info = ctx.design.net(net);
    let users: Vec<u32> = (0..u32::try_from(info.users().len()).expect("few users")).collect();
    let estimated = model()
        .net_delays(&ctx, InterconnectStage::Placed, net, &users)
        .expect("every sink is reachable");
    let exhaustive = model().exact_delays_from(
        ctx.chipdb(),
        pin_wire(&ctx, info.driver().expect("driven net")),
    );
    let searched: Vec<(u32, DelayBounds)> = users
        .iter()
        .map(|&user| {
            (
                user,
                exhaustive[&pin_wire(&ctx, info.users()[user as usize])],
            )
        })
        .collect();
    assert_eq!(estimated, searched);
}

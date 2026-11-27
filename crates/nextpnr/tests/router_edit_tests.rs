mod common;

use common::{add_cell, add_net, drive, make_example_context, sink};

use nextpnr::chipdb::{BelId, PipId, WireId};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::{CellId, CellPin, NetId};
use nextpnr::router::edit::{bind_arc, route_arc_via, unroute_arc, ArcEditError};
use rstest::rstest;

const DRIVER_TILE: (i32, i32) = (1, 1);
const SINK_TILE: (i32, i32) = (4, 3);

struct Fixture {
    ctx: Context,
    net: NetId,
    blocker: NetId,
    driver: CellId,
    s0: CellPin,
    s1: CellPin,
    arc0: Vec<PipId>,
    arc1: Vec<PipId>,
}

fn lut_bels(ctx: &Context, (x, y): (i32, i32)) -> Vec<BelId> {
    let chipdb = ctx.chipdb();
    let tile = chipdb.tile_by_xy(x, y);
    let mut bels: Vec<BelId> = chipdb
        .bels()
        .filter(|&bel| bel.tile() == tile && chipdb.bel_type(bel) == "LUT6")
        .collect();
    bels.sort_by_key(|bel| bel.index());
    bels
}

fn placed_lut(ctx: &mut Context, name: &str, bel: BelId) -> CellId {
    let cell = add_cell(ctx, name, "LUT6");
    assert!(ctx.bind_bel(bel, cell, PlaceStrength::Placer));
    cell
}

fn pin(ctx: &Context, cell: CellId, port: &str) -> CellPin {
    CellPin::new(cell, ctx.id(port))
}

fn fixture() -> Fixture {
    let mut ctx = make_example_context();
    let driver_bel = lut_bels(&ctx, DRIVER_TILE)[0];
    let sink_bels = lut_bels(&ctx, SINK_TILE);
    let driver = placed_lut(&mut ctx, "drv", driver_bel);
    let c0 = placed_lut(&mut ctx, "s0", sink_bels[0]);
    let c1 = placed_lut(&mut ctx, "s1", sink_bels[1]);
    let net = add_net(&mut ctx, "n");
    let blocker = add_net(&mut ctx, "blocker");
    drive(&mut ctx, net, driver, "F");
    sink(&mut ctx, net, c0, "I[0]");
    sink(&mut ctx, net, c1, "I[0]");
    let (s0, s1) = (pin(&ctx, c0, "I[0]"), pin(&ctx, c1, "I[0]"));
    let arc0 = route_arc_via(&mut ctx, net, s0, &[], PlaceStrength::Strong).expect("s0 routes");
    let arc1 = route_arc_via(&mut ctx, net, s1, &[], PlaceStrength::Strong).expect("s1 routes");
    unroute_arc(&mut ctx, net, s1).expect("s1 unroutes");
    unroute_arc(&mut ctx, net, s0).expect("s0 unroutes");
    assert!(arc0.len() >= 3, "arc0 has room to cut, got {arc0:?}");
    Fixture {
        ctx,
        net,
        blocker,
        driver,
        s0,
        s1,
        arc0,
        arc1,
    }
}

fn wire_name(ctx: &Context, wire: WireId) -> String {
    let chipdb = ctx.chipdb();
    format!(
        "{}/{}",
        chipdb.tile_name(wire.tile()),
        chipdb.wire_name(wire)
    )
}

fn pip_name(ctx: &Context, pip: PipId) -> String {
    let chipdb = ctx.chipdb();
    format!(
        "{}/{}/{}",
        chipdb.tile_name(pip.tile()),
        chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)),
        chipdb.wire_name(chipdb.pip_src_tile_wire(pip)),
    )
}

fn sink_wire(ctx: &Context, sink: CellPin) -> WireId {
    let bel = ctx.cell(sink.cell).bel().expect("sink is placed");
    bel.pin_wire(sink.port).expect("sink pin has a wire").id()
}

fn source_wire(f: &Fixture) -> WireId {
    f.ctx.chipdb().pip_src_wire(f.arc0[0])
}

fn dst_wires(ctx: &Context, pips: &[PipId]) -> Vec<WireId> {
    pips.iter()
        .map(|&pip| ctx.chipdb().pip_dst_wire(pip))
        .collect()
}

type TreeEntry = (WireId, Option<PipId>, PlaceStrength, Option<NetId>);

fn tree(ctx: &Context, net: NetId) -> Vec<TreeEntry> {
    let mut entries: Vec<TreeEntry> = ctx
        .net(net)
        .wires()
        .iter()
        .map(|(&wire, entry)| {
            let owner = ctx.wire_binding(wire).map(|(owner, _)| owner);
            (wire, entry.pip, entry.strength, owner)
        })
        .collect();
    entries.sort_by_key(|&(wire, ..)| wire.raw());
    entries
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    trees: Vec<Vec<TreeEntry>>,
    bound_wires: usize,
    pip_owners: Vec<Option<String>>,
}

fn snapshot(f: &Fixture) -> Snapshot {
    Snapshot {
        trees: [f.net, f.blocker]
            .iter()
            .map(|&net| tree(&f.ctx, net))
            .collect(),
        bound_wires: f.ctx.bound_wire_count(),
        pip_owners: f
            .arc0
            .iter()
            .chain(&f.arc1)
            .map(|&pip| f.ctx.pip(pip).bound_net().map(|net| net.name().to_owned()))
            .collect(),
    }
}

fn assert_tree_consistent(ctx: &Context, net: NetId) {
    let chipdb = ctx.chipdb();
    let wires = ctx.net(net).wires();
    for (&wire, entry) in wires {
        assert_eq!(ctx.wire_binding(wire).map(|(owner, _)| owner), Some(net));
        if let Some(pip) = entry.pip {
            assert_eq!(chipdb.pip_dst_wire(pip), wire);
            assert!(wires.contains_key(&chipdb.pip_src_wire(pip)));
            assert_eq!(
                ctx.pip(pip).bound_net().map(|n| n.name()),
                Some(ctx.net(net).name())
            );
        }
    }
}

fn assert_chain(ctx: &Context, pips: &[PipId], from: WireId, to: WireId) {
    let chipdb = ctx.chipdb();
    let mut at = from;
    for &pip in pips {
        assert_eq!(
            chipdb.pip_src_wire(pip),
            at,
            "chain is continuous at {pip:?}"
        );
        at = chipdb.pip_dst_wire(pip);
    }
    assert_eq!(at, to);
}

#[test]
fn route_arc_via_binds_a_chain_from_the_source_to_the_sink() {
    let mut f = fixture();
    let pips = route_arc_via(&mut f.ctx, f.net, f.s0, &[], PlaceStrength::Strong).unwrap();
    assert_chain(&f.ctx, &pips, source_wire(&f), sink_wire(&f.ctx, f.s0));
    assert_eq!(f.ctx.net(f.net).wires().len(), pips.len() + 1);
    assert_tree_consistent(&f.ctx, f.net);
}

#[test]
fn second_arc_branches_from_the_existing_tree() {
    let mut f = fixture();
    route_arc_via(&mut f.ctx, f.net, f.s0, &[], PlaceStrength::Strong).unwrap();
    let first = tree(&f.ctx, f.net);
    let pips = route_arc_via(&mut f.ctx, f.net, f.s1, &[], PlaceStrength::Strong).unwrap();
    let start = f.ctx.chipdb().pip_src_wire(pips[0]);
    assert!(first.iter().any(|&(wire, ..)| wire == start));
    let second = tree(&f.ctx, f.net);
    assert!(first.iter().all(|entry| second.contains(entry)));
    assert_tree_consistent(&f.ctx, f.net);
}

#[rstest]
#[case::node_wire(false)]
#[case::tile_wire_of_the_node(true)]
fn route_arc_via_passes_through_each_waypoint_in_order(#[case] joint_as_tile_wire: bool) {
    let mut f = fixture();
    let chipdb = f.ctx.chipdb();
    let default_hop = chipdb.pip_dst_wire(f.arc0[0]);
    let mut detour = None;
    chipdb.pips_downhill_cb(source_wire(&f), |pip| {
        let wire = chipdb.pip_dst_wire(pip);
        let mut drives_on = false;
        chipdb.pips_downhill_cb(wire, |_| drives_on = true);
        if wire != default_hop && drives_on && detour.is_none() {
            detour = Some(wire);
        }
    });
    let detour = detour.expect("source wire drives more than one node");
    let joint_pip = f.arc0[f.arc0.len() - 2];
    let joint = chipdb.pip_src_wire(joint_pip);
    let tile_wire = chipdb.pip_src_tile_wire(joint_pip);
    assert_ne!(tile_wire, joint, "joint node spans more than one tile");
    let waypoint = match joint_as_tile_wire {
        true => tile_wire,
        false => joint,
    };
    let pips = route_arc_via(
        &mut f.ctx,
        f.net,
        f.s0,
        &[detour, waypoint],
        PlaceStrength::Strong,
    )
    .unwrap();
    let reached = dst_wires(&f.ctx, &pips);
    let at = |wire: WireId| reached.iter().position(|&w| w == wire);
    assert!(at(detour).unwrap() < at(joint).unwrap());
    assert_chain(&f.ctx, &pips, source_wire(&f), sink_wire(&f.ctx, f.s0));
    assert_tree_consistent(&f.ctx, f.net);
}

#[test]
fn unroute_arc_keeps_the_wires_another_sink_uses() {
    let mut f = fixture();
    route_arc_via(&mut f.ctx, f.net, f.s0, &[], PlaceStrength::Strong).unwrap();
    let with_s0 = snapshot(&f);
    route_arc_via(&mut f.ctx, f.net, f.s1, &[], PlaceStrength::Strong).unwrap();
    unroute_arc(&mut f.ctx, f.net, f.s1).unwrap();
    assert_eq!(snapshot(&f), with_s0);
}

#[test]
fn unroute_arc_removes_the_whole_tree_with_the_last_arc() {
    let mut f = fixture();
    route_arc_via(&mut f.ctx, f.net, f.s0, &[], PlaceStrength::Strong).unwrap();
    unroute_arc(&mut f.ctx, f.net, f.s0).unwrap();
    assert!(f.ctx.net(f.net).wires().is_empty());
    assert_eq!(f.ctx.bound_wire_count(), 0);
    assert!(f
        .arc0
        .iter()
        .all(|&pip| f.ctx.pip(pip).bound_net().is_none()));
}

#[rstest]
#[case::onto_an_empty_tree(false)]
#[case::onto_another_arc(true)]
fn bind_arc_binds_the_given_chain_at_the_given_strength(#[case] s0_bound_first: bool) {
    let mut f = fixture();
    let (sink, pips) = match s0_bound_first {
        true => {
            bind_arc(
                &mut f.ctx,
                f.net,
                f.s0,
                &f.arc0.clone(),
                PlaceStrength::Strong,
            )
            .unwrap();
            (f.s1, f.arc1.clone())
        }
        false => (f.s0, f.arc0.clone()),
    };
    bind_arc(&mut f.ctx, f.net, sink, &pips, PlaceStrength::User).unwrap();
    let entries = tree(&f.ctx, f.net);
    for wire in dst_wires(&f.ctx, &pips) {
        let entry = entries
            .iter()
            .find(|&&(w, ..)| w == wire)
            .expect("chain wire is in the tree");
        assert_eq!(entry.2, PlaceStrength::User);
    }
    assert_tree_consistent(&f.ctx, f.net);
}

type Edit = Box<dyn FnOnce(&mut Context) -> Result<(), ArcEditError>>;

fn bind(net: NetId, sink: CellPin, pips: Vec<PipId>) -> Edit {
    Box::new(move |ctx| bind_arc(ctx, net, sink, &pips, PlaceStrength::Strong))
}

fn arc_names(f: &Fixture, sink: CellPin) -> (String, String) {
    let cell = f.ctx.cell(sink.cell).name().to_owned();
    (
        "n".to_owned(),
        format!("{cell}.{}", f.ctx.name_of(sink.port)),
    )
}

fn not_a_sink(f: &mut Fixture) -> (Edit, ArcEditError) {
    let driver = pin(&f.ctx, f.driver, "F");
    let (net, sink) = arc_names(f, driver);
    (
        bind(f.net, driver, f.arc0.clone()),
        ArcEditError::NotASink { net, sink },
    )
}

fn no_driver(f: &mut Fixture) -> (Edit, ArcEditError) {
    let floating = add_net(&mut f.ctx, "floating");
    sink(&mut f.ctx, floating, f.s0.cell, "I[1]");
    let user = pin(&f.ctx, f.s0.cell, "I[1]");
    let expected = ArcEditError::NoDriver {
        net: "floating".to_owned(),
    };
    (bind(floating, user, f.arc0.clone()), expected)
}

fn unplaced(f: &mut Fixture) -> (Edit, ArcEditError) {
    let loose = add_cell(&mut f.ctx, "loose", "LUT6");
    sink(&mut f.ctx, f.net, loose, "I[0]");
    let user = pin(&f.ctx, loose, "I[0]");
    let (net, pin) = arc_names(f, user);
    (
        bind(f.net, user, f.arc0.clone()),
        ArcEditError::Unplaced { net, pin },
    )
}

fn no_pin_wire(f: &mut Fixture) -> (Edit, ArcEditError) {
    sink(&mut f.ctx, f.net, f.s0.cell, "NOPE");
    let user = pin(&f.ctx, f.s0.cell, "NOPE");
    let (net, pin) = arc_names(f, user);
    (
        bind(f.net, user, f.arc0.clone()),
        ArcEditError::NoPinWire { net, pin },
    )
}

fn already_routed(f: &mut Fixture) -> (Edit, ArcEditError) {
    bind_arc(
        &mut f.ctx,
        f.net,
        f.s0,
        &f.arc0.clone(),
        PlaceStrength::Strong,
    )
    .unwrap();
    let (net, sink) = arc_names(f, f.s0);
    let (n, s0) = (f.net, f.s0);
    let edit: Edit =
        Box::new(move |ctx| route_arc_via(ctx, n, s0, &[], PlaceStrength::Strong).map(drop));
    (edit, ArcEditError::AlreadyRouted { net, sink })
}

fn not_routed(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let (n, s0) = (f.net, f.s0);
    let edit: Edit = Box::new(move |ctx| unroute_arc(ctx, n, s0));
    (edit, ArcEditError::NotRouted { net, sink })
}

fn empty_chain(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    (
        bind(f.net, f.s0, Vec::new()),
        ArcEditError::EmptyChain { net, sink },
    )
}

fn unknown_pip(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let pip = PipId::new(0, 1_000_000);
    let mut pips = f.arc0.clone();
    pips.push(pip);
    (
        bind(f.net, f.s0, pips),
        ArcEditError::UnknownPip { net, sink, pip },
    )
}

fn unknown_wire(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let wire = WireId::new(0, 1_000_000);
    let (n, s0) = (f.net, f.s0);
    let edit: Edit =
        Box::new(move |ctx| route_arc_via(ctx, n, s0, &[wire], PlaceStrength::Strong).map(drop));
    (edit, ArcEditError::UnknownWire { net, sink, wire })
}

fn detached_start(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let first = f.arc0[1];
    let expected = ArcEditError::DetachedStart {
        net,
        sink,
        pip: pip_name(&f.ctx, first),
        wire: wire_name(&f.ctx, f.ctx.chipdb().pip_src_wire(first)),
    };
    (bind(f.net, f.s0, f.arc0[1..].to_vec()), expected)
}

fn discontinuous(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let mut pips = f.arc0.clone();
    pips.remove(1);
    let expected = ArcEditError::Discontinuous {
        net,
        sink,
        pip: pip_name(&f.ctx, pips[1]),
        wire: wire_name(&f.ctx, f.ctx.chipdb().pip_src_wire(pips[1])),
        expected: wire_name(&f.ctx, f.ctx.chipdb().pip_dst_wire(pips[0])),
    };
    (bind(f.net, f.s0, pips), expected)
}

fn wrong_end(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let pips = f.arc0[..f.arc0.len() - 1].to_vec();
    let expected = ArcEditError::WrongEnd {
        net,
        sink,
        wire: wire_name(&f.ctx, f.ctx.chipdb().pip_dst_wire(*pips.last().unwrap())),
        sink_wire: wire_name(&f.ctx, sink_wire(&f.ctx, f.s0)),
    };
    (bind(f.net, f.s0, pips), expected)
}

fn wire_taken(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let wire = f.ctx.chipdb().pip_dst_wire(f.arc0[2]);
    f.ctx.bind_wire(wire, f.blocker, PlaceStrength::Strong);
    let expected = ArcEditError::WireTaken {
        net,
        sink,
        wire: wire_name(&f.ctx, wire),
        owner: "blocker".to_owned(),
    };
    (bind(f.net, f.s0, f.arc0.clone()), expected)
}

fn re_enters_tree(f: &mut Fixture) -> (Edit, ArcEditError) {
    bind_arc(
        &mut f.ctx,
        f.net,
        f.s0,
        &f.arc0.clone(),
        PlaceStrength::Strong,
    )
    .unwrap();
    let (net, sink) = arc_names(f, f.s1);
    let wire = wire_name(&f.ctx, f.ctx.chipdb().pip_dst_wire(f.arc0[0]));
    (
        bind(f.net, f.s1, f.arc0.clone()),
        ArcEditError::Loop { net, sink, wire },
    )
}

fn no_path(f: &mut Fixture) -> (Edit, ArcEditError) {
    let (net, sink) = arc_names(f, f.s0);
    let target = sink_wire(&f.ctx, f.s0);
    f.ctx.bind_wire(target, f.blocker, PlaceStrength::Strong);
    let wire = wire_name(&f.ctx, target);
    let (n, s0) = (f.net, f.s0);
    let edit: Edit =
        Box::new(move |ctx| route_arc_via(ctx, n, s0, &[], PlaceStrength::Strong).map(drop));
    (edit, ArcEditError::NoPath { net, sink, wire })
}

#[rstest]
#[case::not_a_sink(not_a_sink)]
#[case::no_driver(no_driver)]
#[case::unplaced(unplaced)]
#[case::no_pin_wire(no_pin_wire)]
#[case::already_routed(already_routed)]
#[case::not_routed(not_routed)]
#[case::empty_chain(empty_chain)]
#[case::unknown_pip(unknown_pip)]
#[case::unknown_wire(unknown_wire)]
#[case::detached_start(detached_start)]
#[case::discontinuous(discontinuous)]
#[case::wrong_end(wrong_end)]
#[case::wire_taken(wire_taken)]
#[case::re_enters_tree(re_enters_tree)]
#[case::no_path(no_path)]
fn rejected_edit_leaves_the_context_unchanged(
    #[case] setup: fn(&mut Fixture) -> (Edit, ArcEditError),
) {
    let mut f = fixture();
    let (edit, expected) = setup(&mut f);
    let before = snapshot(&f);
    assert_eq!(edit(&mut f.ctx), Err(expected));
    assert_eq!(snapshot(&f), before);
}

mod common;

use common::{add_cell, add_net, drive, make_example_context, sink};

use nextpnr::chipdb::{BelId, ChipDb, NameError, NameKind, PipId, WireId};
use nextpnr::common::{IdStringPool, PlaceStrength};
use nextpnr::context::Context;
use nextpnr::frontend::{parse_routed_json, AttributeFault, RoutedAttribute};
use nextpnr::netlist::{CellId, Cluster, Design, NetId, Property, Rect};
use nextpnr::placer::heap::{place_heap, PlacerHeapCfg};
use nextpnr::routed_json::{reopen, routed_json, ReopenError, RoutedJsonError};
use nextpnr::router::common::{
    bind_route, collect_routable_nets, unroute_net, validate_all_routed,
};
use nextpnr::router::router2::Router2Cfg;
use nextpnr::router::Router2;
use rstest::rstest;
use rustc_hash::FxHashMap;
use serde_json::{json, Value};

const LOGIC_TILE: (i32, i32) = (3, 3);

fn routed_lut(ctx: &mut Context) {
    let chipdb = ctx.chipdb();
    let tile = chipdb.tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1);
    let bel = chipdb
        .bels()
        .find(|&bel| bel.tile() == tile && chipdb.bel_name(bel) == "L0_LUT")
        .expect("logic tile has BEL L0_LUT");
    let pip = chipdb
        .pips()
        .find(|&pip| {
            pip.tile() == tile
                && chipdb.wire_name(chipdb.pip_src_tile_wire(pip)) == "L0_O"
                && chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)) == "L0_D"
        })
        .expect("logic tile has pip L0_O -> L0_D");
    let lut = add_cell(ctx, "lut", "LUT6");
    let bel_attr = ctx.id("BEL");
    ctx.design
        .cell_edit(lut)
        .set_attr(bel_attr, Property::string("stale"));
    assert!(ctx.bind_bel(bel, lut, PlaceStrength::Placer));
    let net = add_net(ctx, "o");
    drive(ctx, net, lut, "O");
    let source = ctx.chipdb().pip_src_wire(pip);
    ctx.bind_wire(source, net, PlaceStrength::Strong);
    ctx.design
        .net_edit(net)
        .add_wire(source, None, PlaceStrength::Strong);
    bind_route(ctx, net, &[pip]);
}

fn top_module(json: &Value) -> &Value {
    let modules = json["modules"].as_object().expect("modules object");
    assert_eq!(modules.len(), 1);
    modules.values().next().expect("one module")
}

#[test]
fn placement_and_routing_travel_in_nextpnr_attributes() {
    let mut ctx = make_example_context();
    routed_lut(&mut ctx);
    let tile = ctx
        .chipdb()
        .tile_name(ctx.chipdb().tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1));
    let json = routed_json(&ctx).expect("routed JSON");
    let top = top_module(&json);
    assert_eq!(
        top["cells"]["lut"]["attributes"],
        json!({
            "NEXTPNR_BEL": format!("{tile}/L0_LUT"),
            "BEL_STRENGTH": "00000000000000000000000000000011",
        })
    );
    let bit = top["netnames"]["o"]["bits"][0].clone();
    assert_eq!(top["cells"]["lut"]["connections"]["O"], json!([bit]));
    assert_eq!(top["cells"]["lut"]["port_directions"]["O"], "output");
    let mut routing: Vec<String> = top["netnames"]["o"]["attributes"]["ROUTING"]
        .as_str()
        .expect("ROUTING string")
        .split(';')
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|entry| entry.join(";"))
        .collect();
    routing.sort();
    assert_eq!(
        routing,
        vec![
            format!("{tile}/L0_D;{tile}/L0_D/L0_O;2"),
            format!("{tile}/L0_O;;2"),
        ]
    );
}

#[rstest]
#[case::plain_string(Property::string("FAST"), "FAST")]
#[case::bit_like_string_is_padded(Property::string("0101"), "0101 ")]
#[case::empty_string_is_padded(Property::string(""), " ")]
#[case::bit_vector(Property::bit_vector("0110"), "0110")]
#[case::integer_is_32_bits(Property::int(-2), "11111111111111111111111111111110")]
fn properties_encode_as_nextpnr_writes_them(#[case] value: Property, #[case] encoded: &str) {
    let mut ctx = make_example_context();
    let cell = add_cell(&mut ctx, "c", "LUT6");
    let key = ctx.id("P");
    ctx.design.cell_edit(cell).set_param(key, value);
    let json = routed_json(&ctx).expect("routed JSON");
    assert_eq!(top_module(&json)["cells"]["c"]["parameters"]["P"], encoded);
}

#[test]
fn integer_wider_than_32_bits_is_rejected() {
    let mut ctx = make_example_context();
    let cell = add_cell(&mut ctx, "c", "LUT6");
    let key = ctx.id("P");
    ctx.design
        .cell_edit(cell)
        .set_param(key, Property::int(1 << 40));
    assert_eq!(
        routed_json(&ctx).err(),
        Some(RoutedJsonError::IntegerOutOfRange {
            owner: "Parameter P of cell c".to_owned(),
            value: 1 << 40,
        })
    );
}

type Placements = FxHashMap<String, (Option<BelId>, PlaceStrength)>;
type Routes = FxHashMap<String, FxHashMap<WireId, (Option<PipId>, PlaceStrength)>>;

fn placements(design: &Design, name: impl Fn(nextpnr::common::IdString) -> String) -> Placements {
    design
        .iter_alive_cells()
        .map(|(_, cell)| (name(cell.name), (cell.bel, cell.bel_strength)))
        .collect()
}

fn routes(design: &Design, name: impl Fn(nextpnr::common::IdString) -> String) -> Routes {
    design
        .iter_alive_nets()
        .map(|(_, net)| {
            let wires = net
                .wires
                .iter()
                .map(|(&wire, binding)| (wire, (binding.pip, binding.strength)))
                .collect();
            (name(net.name), wires)
        })
        .collect()
}

fn routed_baseline_with_varied_strengths() -> Context {
    let mut ctx = common::placed_npnr_baseline();
    let nets = collect_routable_nets(&ctx);
    Router2
        .route_nets_report(&mut ctx, &Router2Cfg::default(), &nets)
        .expect("negotiation converges");
    validate_all_routed(&ctx).expect("every net is routed");
    let (cell, bel) = ctx
        .design
        .iter_alive_cells()
        .find_map(|(id, cell)| Some((id, cell.bel?)))
        .expect("baseline has a placed cell");
    ctx.unbind_bel(bel);
    assert!(ctx.bind_bel(bel, cell, PlaceStrength::Fixed));
    let (net, source) = ctx
        .design
        .iter_alive_nets()
        .find_map(|(id, net)| {
            let source = net
                .wires
                .iter()
                .find(|(_, binding)| binding.pip.is_none())?;
            Some((id, *source.0))
        })
        .expect("baseline has a routed net");
    ctx.bind_wire(source, net, PlaceStrength::Locked);
    ctx.design
        .net_edit(net)
        .add_wire(source, None, PlaceStrength::Locked);
    ctx
}

#[test]
fn routed_json_decodes_to_the_placement_and_routing_it_was_written_from() {
    let ctx = routed_baseline_with_varied_strengths();
    let json = serde_json::to_string(&routed_json(&ctx).expect("routed JSON")).expect("serialises");
    let pool = IdStringPool::new();
    let decoded = parse_routed_json(&json, ctx.chipdb(), &pool).expect("routed JSON decodes");
    let decoded_name = |id| pool.lookup(id).expect("interned").to_owned();
    let written_name = |id| ctx.name_of(id).to_owned();
    assert_eq!(
        placements(&decoded, decoded_name),
        placements(&ctx.design, written_name)
    );
    assert_eq!(
        routes(&decoded, decoded_name),
        routes(&ctx.design, written_name)
    );
}

fn example_chipdb() -> ChipDb {
    ChipDb::load(&common::example_chipdb_path()).expect("example chipdb loads")
}

fn reopen_json(chipdb: ChipDb, json: &Value) -> Result<Context, ReopenError> {
    reopen(chipdb, &json.to_string())
}

fn reopened(ctx: &Context) -> Context {
    reopen_json(example_chipdb(), &routed_json(ctx).expect("routed JSON")).expect("reopens")
}

fn cell_named(ctx: &Context, name: &str) -> CellId {
    ctx.design.cell_by_name(ctx.id(name)).expect("cell exists")
}

fn net_named(ctx: &Context, name: &str) -> NetId {
    ctx.design.net_by_name(ctx.id(name)).expect("net exists")
}

fn restrengthen(ctx: &mut Context, strength: PlaceStrength) {
    let lut = cell_named(ctx, "lut");
    let bel = ctx.design.cell(lut).bel;
    ctx.design.cell_edit(lut).set_bel(bel, strength);
    let net = net_named(ctx, "o");
    let wires: Vec<_> = ctx
        .design
        .net(net)
        .wires
        .iter()
        .map(|(&wire, binding)| (wire, binding.pip))
        .collect();
    for (wire, pip) in wires {
        ctx.design.net_edit(net).add_wire(wire, pip, strength);
    }
}

#[rstest]
#[case::user_stays_user(PlaceStrength::User, PlaceStrength::User)]
#[case::locked_stays_locked(PlaceStrength::Locked, PlaceStrength::Locked)]
#[case::fixed_becomes_strong(PlaceStrength::Fixed, PlaceStrength::Strong)]
#[case::placer_becomes_strong(PlaceStrength::Placer, PlaceStrength::Strong)]
#[case::weak_becomes_strong(PlaceStrength::Weak, PlaceStrength::Strong)]
fn reopen_keeps_user_and_locked_bindings_and_makes_the_rest_strong(
    #[case] written: PlaceStrength,
    #[case] expected: PlaceStrength,
) {
    let mut ctx = make_example_context();
    routed_lut(&mut ctx);
    restrengthen(&mut ctx, written);
    let ctx = reopened(&ctx);
    assert_eq!(
        ctx.design.cell(cell_named(&ctx, "lut")).bel_strength,
        expected
    );
    let net = net_named(&ctx, "o");
    let wires: Vec<_> = ctx
        .design
        .net(net)
        .wires
        .iter()
        .map(|(&wire, binding)| (binding.strength, ctx.wire_binding(wire)))
        .collect();
    assert_eq!(wires, vec![(expected, Some((net, expected))); 2]);
}

#[test]
fn reopened_baseline_binds_every_bel_and_wire_and_stays_routed() {
    let ctx = routed_baseline_with_varied_strengths();
    let json = routed_json(&ctx).expect("routed JSON");
    let reopened = reopen_json(common::npnr_baseline_chipdb(), &json).expect("reopens");
    for (id, cell) in reopened.design.iter_alive_cells() {
        if let Some(bel) = cell.bel {
            assert_eq!(reopened.conflicting_bel_cell(bel), Some(id));
        }
    }
    for (id, net) in reopened.design.iter_alive_nets() {
        for (&wire, binding) in &net.wires {
            assert_eq!(reopened.wire_binding(wire), Some((id, binding.strength)));
        }
    }
    validate_all_routed(&reopened).expect("every net stays routed");
}

#[test]
fn a_net_ripped_up_after_reopen_routes_again() {
    let ctx = routed_baseline_with_varied_strengths();
    let json = routed_json(&ctx).expect("routed JSON");
    let mut ctx = reopen_json(common::npnr_baseline_chipdb(), &json).expect("reopens");
    let net = collect_routable_nets(&ctx)[0];
    unroute_net(&mut ctx, net);
    Router2
        .route_nets_report(&mut ctx, &Router2Cfg::default(), &[net])
        .expect("the ripped-up net routes");
    validate_all_routed(&ctx).expect("every net is routed");
}

fn names(ctx: &Context, cells: &[CellId]) -> Vec<String> {
    cells
        .iter()
        .map(|&cell| ctx.name_of(ctx.design.cell(cell).name).to_owned())
        .collect()
}

fn lut_ff_cluster(ctx: &mut Context) {
    let lut = add_cell(ctx, "lut", "LUT6");
    let ff = add_cell(ctx, "ff", "DFF");
    let d = add_net(ctx, "d");
    drive(ctx, d, lut, "O");
    sink(ctx, d, ff, "D");
    ctx.design
        .cell_edit(lut)
        .set_cluster(Some(lut))
        .set_constraints(0, 0, 0, false);
    ctx.design
        .cell_edit(ff)
        .set_cluster(Some(lut))
        .set_constraints(-1, 2, 3, true);
    let mut cluster = Cluster::new(lut);
    cluster.add_member(ff);
    cluster.add_constrained_child(ff);
    ctx.design.clusters.insert(lut, cluster);
}

fn clusters(ctx: &Context) -> String {
    let mut cells: Vec<_> = ctx
        .design
        .iter_alive_cells()
        .map(|(_, cell)| {
            let root = cell.cluster.map(|root| names(ctx, &[root]));
            let offset = (
                cell.constr_x,
                cell.constr_y,
                cell.constr_z,
                cell.constr_abs_z,
            );
            (ctx.name_of(cell.name).to_owned(), root, offset)
        })
        .collect();
    cells.sort();
    let mut clusters: Vec<_> = ctx
        .design
        .clusters
        .values()
        .map(|cluster| {
            (
                names(ctx, &[cluster.root]),
                names(ctx, &cluster.members),
                names(ctx, &cluster.constr_children),
            )
        })
        .collect();
    clusters.sort();
    format!("{cells:?} {clusters:?}")
}

fn pblock(ctx: &mut Context) {
    let region = ctx.design.add_region(ctx.id("pblock"));
    ctx.design.region_mut(region).rects = vec![Rect::new(1, 1, 2, 2), Rect::new(4, 0, 5, 3)];
    ctx.design.add_region(ctx.id("empty"));
    let lut = add_cell(ctx, "lut", "LUT6");
    let o = add_net(ctx, "o");
    drive(ctx, o, lut, "O");
    ctx.design.cell_edit(lut).set_region(Some(region));
    ctx.design.net_edit(o).set_region(Some(region));
}

fn regions(ctx: &Context) -> String {
    let name =
        |region: Option<u32>| region.map(|region| ctx.name_of(ctx.design.region(region).name));
    let mut regions: Vec<_> = ctx
        .design
        .regions
        .iter()
        .map(|region| (ctx.name_of(region.name), region.rects.clone()))
        .collect();
    regions.sort_by_key(|&(name, _)| name);
    let mut cells: Vec<_> = ctx
        .design
        .iter_alive_cells()
        .map(|(_, cell)| (ctx.name_of(cell.name), name(cell.region)))
        .collect();
    cells.sort();
    let mut nets: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(_, net)| (ctx.name_of(net.name), name(net.region)))
        .collect();
    nets.sort();
    format!("{regions:?} {cells:?} {nets:?}")
}

fn constrained_clock(ctx: &mut Context) {
    let source = add_cell(ctx, "source", "LUT6");
    let clk = add_net(ctx, "clk");
    drive(ctx, clk, source, "O");
    ctx.design.net_edit(clk).set_clock_constraint(4_000);
}

fn clock_periods(ctx: &Context) -> String {
    let mut periods: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(_, net)| (ctx.name_of(net.name), net.clock_constraint))
        .collect();
    periods.sort();
    format!("{periods:?}")
}

fn user_sdc(ctx: &mut Context) {
    ctx.design.sdc = "set_false_path -from [get_cells a] -to [get_cells b]\n".to_owned();
}

fn sdc(ctx: &Context) -> String {
    ctx.design.sdc.clone()
}

#[rstest]
#[case::clusters(lut_ff_cluster, clusters)]
#[case::regions(pblock, regions)]
#[case::clock_periods(constrained_clock, clock_periods)]
#[case::sdc(user_sdc, sdc)]
fn design_state_survives_reopen(
    #[case] set: fn(&mut Context),
    #[case] observe: fn(&Context) -> String,
) {
    let mut ctx = make_example_context();
    set(&mut ctx);
    let observed = observe(&ctx);
    assert_ne!(
        observed,
        observe(&make_example_context()),
        "the case sets state"
    );
    assert_eq!(observe(&reopened(&ctx)), observed);
}

#[test]
fn a_cell_unplaced_after_reopen_stays_unplaced_after_the_next_reopen() {
    let mut ctx = make_example_context();
    routed_lut(&mut ctx);
    let mut ctx = reopened(&ctx);
    let lut = cell_named(&ctx, "lut");
    let bel = ctx.design.cell(lut).bel.expect("lut is placed");
    ctx.unbind_bel(bel);
    let ctx = reopened(&ctx);
    assert_eq!(ctx.design.cell(cell_named(&ctx, "lut")).bel, None);
}

#[test]
fn a_reopened_cluster_places_again_with_heap() {
    let mut ctx = make_example_context();
    lut_ff_cluster(&mut ctx);
    let chipdb = ctx.chipdb();
    let tile = chipdb.tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1);
    let z = |name| {
        let bel = chipdb
            .bels()
            .find(|&bel| bel.tile() == tile && chipdb.bel_name(bel) == name)
            .expect("logic tile has the BEL");
        chipdb.bel_loc(bel).z
    };
    let ff_dz = z("L0_FF") - z("L0_LUT");
    let ff = cell_named(&ctx, "ff");
    ctx.design.cell_edit(ff).set_constraints(0, 0, ff_dz, false);
    let heap = PlacerHeapCfg {
        seed: 7,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &heap).expect("HeAP places the cluster");
    let mut ctx = reopened(&ctx);
    let placed: Vec<BelId> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(_, cell)| cell.bel)
        .collect();
    for bel in placed {
        ctx.unbind_bel(bel);
    }
    place_heap(&mut ctx, &heap).expect("HeAP places the reopened cluster");
    let bel_of = |name| ctx.design.cell(cell_named(&ctx, name)).bel.expect("placed");
    let (lut, ff) = (
        ctx.chipdb().bel_loc(bel_of("lut")),
        ctx.chipdb().bel_loc(bel_of("ff")),
    );
    assert_eq!((ff.x, ff.y, ff.z), (lut.x, lut.y, lut.z + ff_dz));
}

fn two_luts(ctx: &mut Context) {
    routed_lut(ctx);
    let lut2 = add_cell(ctx, "lut2", "LUT6");
    let p = add_net(ctx, "p");
    drive(ctx, p, lut2, "O");
}

fn top_module_mut(json: &mut Value) -> &mut Value {
    json["modules"]
        .as_object_mut()
        .expect("modules object")
        .values_mut()
        .next()
        .expect("one module")
}

fn share_bel(json: &mut Value) {
    let top = top_module_mut(json);
    top["cells"]["lut2"]["attributes"] = top["cells"]["lut"]["attributes"].clone();
}

fn share_source_wire(json: &mut Value) {
    let top = top_module_mut(json);
    let routing = top["netnames"]["o"]["attributes"]["ROUTING"]
        .as_str()
        .expect("ROUTING string")
        .split(';')
        .skip(3)
        .collect::<Vec<_>>()
        .join(";");
    top["netnames"]["p"]["attributes"]["ROUTING"] = Value::String(routing);
}

fn rename_bel(json: &mut Value) {
    top_module_mut(json)["cells"]["lut"]["attributes"]["NEXTPNR_BEL"] = json!("nowhere/L0_LUT");
}

fn bel_held(tile: &str) -> ReopenError {
    ReopenError::BelHeld {
        cell: "lut2".to_owned(),
        bel: format!("{tile}/L0_LUT"),
        holder: "lut".to_owned(),
    }
}

fn wire_held(tile: &str) -> ReopenError {
    ReopenError::WireHeld {
        net: "p".to_owned(),
        wire: format!("{tile}/L0_O"),
        holder: "o".to_owned(),
    }
}

fn unknown_bel(_: &str) -> ReopenError {
    ReopenError::Json(nextpnr::frontend::RoutedJsonError::Cell {
        cell: "lut".to_owned(),
        attribute: RoutedAttribute::NextpnrBel,
        fault: AttributeFault::Name(NameError::Unknown {
            kind: NameKind::Bel,
            name: "nowhere/L0_LUT".to_owned(),
            segment_kind: NameKind::Tile,
            segment: "nowhere".to_owned(),
        }),
    })
}

#[rstest]
#[case::two_cells_on_one_bel(share_bel, bel_held)]
#[case::two_nets_on_one_wire(share_source_wire, wire_held)]
#[case::unknown_bel(rename_bel, unknown_bel)]
fn reopen_rejects_a_conflicting_or_unknown_binding(
    #[case] corrupt: fn(&mut Value),
    #[case] error: fn(&str) -> ReopenError,
) {
    let mut ctx = make_example_context();
    two_luts(&mut ctx);
    let tile = ctx
        .chipdb()
        .tile_name(ctx.chipdb().tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1))
        .to_owned();
    let mut json = routed_json(&ctx).expect("routed JSON");
    corrupt(&mut json);
    assert_eq!(
        reopen_json(example_chipdb(), &json).err(),
        Some(error(&tile))
    );
}

mod common;

use common::{add_cell, add_net, drive, make_example_context};

use nextpnr::chipdb::{BelId, PipId, WireId};
use nextpnr::common::{IdStringPool, PlaceStrength};
use nextpnr::context::Context;
use nextpnr::frontend::parse_routed_json;
use nextpnr::netlist::{Design, Property};
use nextpnr::routed_json::{routed_json, RoutedJsonError};
use nextpnr::router::common::{bind_route, collect_routable_nets, validate_all_routed};
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

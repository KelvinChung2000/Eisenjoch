mod common;

use common::{add_cell, add_net, drive, make_example_context};

use nextpnr::chipdb::{BelId, PipId};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::fasm::{fasm_features, FasmError, FasmFeature, FeatureSetting};
use nextpnr::netlist::{CellId, Property};
use nextpnr::router::common::bind_route;
use rstest::rstest;

const LOGIC_TILE: (i32, i32) = (3, 3);

fn bel_at(ctx: &Context, name: &str) -> BelId {
    let tile = ctx.chipdb().tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1);
    ctx.chipdb()
        .bels()
        .find(|&bel| bel.tile() == tile && ctx.chipdb().bel_name(bel) == name)
        .unwrap_or_else(|| panic!("logic tile has BEL {name}"))
}

fn pip_at(ctx: &Context, src: &str, dst: &str) -> PipId {
    let chipdb = ctx.chipdb();
    let tile = chipdb.tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1);
    chipdb
        .pips()
        .find(|&pip| {
            pip.tile() == tile
                && chipdb.wire_name(chipdb.pip_src_tile_wire(pip)) == src
                && chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)) == dst
        })
        .unwrap_or_else(|| panic!("logic tile has pip {src} -> {dst}"))
}

fn routed_lut(ctx: &mut Context, param: Option<(&str, Property)>) -> CellId {
    let lut = add_cell(ctx, "lut", "LUT6");
    if let Some((key, value)) = param {
        let key = ctx.id(key);
        ctx.design.cell_edit(lut).set_param(key, value);
    }
    assert!(ctx.bind_bel(bel_at(ctx, "L0_LUT"), lut, PlaceStrength::Placer));
    let net = add_net(ctx, "o");
    drive(ctx, net, lut, "O");
    let pip = pip_at(ctx, "L0_O", "L0_D");
    let source = ctx.chipdb().pip_src_wire(pip);
    ctx.bind_wire(source, net, PlaceStrength::Strong);
    ctx.design
        .net_edit(net)
        .add_wire(source, None, PlaceStrength::Strong);
    bind_route(ctx, net, &[pip]);
    lut
}

fn logic_feature(ctx: &Context, feature: &str, setting: FeatureSetting) -> FasmFeature {
    FasmFeature {
        tile: ctx
            .chipdb()
            .tile_name(ctx.chipdb().tile_by_xy(LOGIC_TILE.0, LOGIC_TILE.1)),
        feature: feature.to_owned(),
        setting,
    }
}

#[rstest]
#[case::no_param(None, None)]
#[case::vector(
    Some(("INIT", Property::bit_vector("0110"))),
    Some(("L0_LUT.INIT", FeatureSetting::Bits { lsb_first: vec![false, true, true, false] }))
)]
#[case::set_bit(Some(("FLAG", Property::bit_vector("1"))), Some(("L0_LUT.FLAG", FeatureSetting::Enabled)))]
#[case::clear_bit(Some(("FLAG", Property::bit_vector("0"))), None)]
#[case::enum_value(Some(("MODE", Property::string("FAST"))), Some(("L0_LUT.MODE.FAST", FeatureSetting::Enabled)))]
fn features_name_the_pip_and_bel_from_the_chipdb(
    #[case] param: Option<(&str, Property)>,
    #[case] cell_feature: Option<(&str, FeatureSetting)>,
) {
    let mut ctx = make_example_context();
    routed_lut(&mut ctx, param);
    let mut expected: Vec<FasmFeature> = cell_feature
        .into_iter()
        .map(|(feature, setting)| logic_feature(&ctx, feature, setting))
        .collect();
    expected.push(logic_feature(&ctx, "L0_D.L0_O", FeatureSetting::Enabled));
    expected.sort_by(|a, b| a.feature.cmp(&b.feature));
    assert_eq!(fasm_features(&ctx), Ok(expected));
}

#[rstest]
#[case::integer(
    Property::int(5),
    FasmError::UnsizedParameter { cell: "lut".into(), param: "INIT".into() }
)]
#[case::undefined_bits(
    Property::string("01x0"),
    FasmError::UndefinedBits { cell: "lut".into(), param: "INIT".into(), bits: "01x0".into() }
)]
#[case::undefined_vector_bits(
    Property::bit_vector("01x0"),
    FasmError::UndefinedBits { cell: "lut".into(), param: "INIT".into(), bits: "01x0".into() }
)]
#[case::empty_vector(
    Property::bit_vector(""),
    FasmError::UnsizedParameter { cell: "lut".into(), param: "INIT".into() }
)]
fn parameters_without_exact_bits_are_rejected(#[case] value: Property, #[case] error: FasmError) {
    let mut ctx = make_example_context();
    routed_lut(&mut ctx, Some(("INIT", value)));
    assert_eq!(fasm_features(&ctx), Err(error));
}

#[test]
fn unplaced_cell_is_rejected() {
    let mut ctx = make_example_context();
    add_cell(&mut ctx, "floating", "LUT6");
    assert_eq!(
        fasm_features(&ctx),
        Err(FasmError::UnplacedCell {
            cell: "floating".into()
        })
    );
}

#[rstest]
#[case::pip_is_src_dot_dst("LC_O.N1BEG0")]
#[case::bel_prefix_is_bel_name("C.FF")]
fn fabulous_features_follow_the_bitstream_spec_names(#[case] feature: &str) {
    let (src, dst) = ("LC_O", "N1BEG0");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fabulous.bin");
    let mut ctx =
        Context::new(nextpnr::chipdb::ChipDb::load(&path).expect("FABulous chipdb loads"));
    let lc = add_cell(&mut ctx, "lc", "FABULOUS_LC");
    let ff = ctx.id("FF");
    ctx.design
        .cell_edit(lc)
        .set_param(ff, Property::bit_vector("1"));
    let chipdb = ctx.chipdb();
    let tile = chipdb.tile_by_xy(1, 1);
    let bel = chipdb
        .bels()
        .find(|&bel| bel.tile() == tile && chipdb.bel_name(bel) == "C")
        .expect("X1Y1 has LC C");
    let pip = chipdb
        .pips()
        .find(|&pip| {
            pip.tile() == tile
                && chipdb.wire_name(chipdb.pip_src_tile_wire(pip)) == src
                && chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)) == dst
        })
        .expect("X1Y1 has the pip");
    assert!(ctx.bind_bel(bel, lc, PlaceStrength::Placer));
    let net = add_net(&mut ctx, "o");
    drive(&mut ctx, net, lc, "O");
    let source = ctx.chipdb().pip_src_wire(pip);
    ctx.bind_wire(source, net, PlaceStrength::Strong);
    ctx.design
        .net_edit(net)
        .add_wire(source, None, PlaceStrength::Strong);
    bind_route(&mut ctx, net, &[pip]);
    let features = fasm_features(&ctx).expect("features");
    assert!(
        features
            .iter()
            .any(|f| f.tile == "X1Y1" && f.feature == feature),
        "{feature} in {features:?}"
    );
}

use nextpnr_db::chipdb::testutil::make_test_chipdb;
use nextpnr_db::chipdb::{BelId, NameError, NameKind, PipId, WireId};
use nextpnr_db::common::{IdStringPool, PlaceStrength};
use nextpnr_db::frontend::{parse_routed_json, AttributeFault, RoutedAttribute, RoutedJsonError};
use nextpnr_db::netlist::Design;
use rstest::rstest;
use serde_json::json;

const PLACER: &str = "00000000000000000000000000000011";
const ROUTED: &str = "TILE_0_0X0Y0/W0;;2;TILE_0_0X0Y0/W1;TILE_0_0X0Y0/W1/W0;5";

fn decode(bel: &str, strength: &str, routing: &str) -> Result<Design, RoutedJsonError> {
    let json = json!({
        "modules": {
            "top": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "cells": {
                    "lut": {
                        "type": "LUT6",
                        "attributes": { "NEXTPNR_BEL": bel, "BEL_STRENGTH": strength },
                        "port_directions": { "O": "output" },
                        "connections": { "O": [2] },
                    },
                },
                "netnames": {
                    "o": { "bits": [2], "attributes": { "ROUTING": routing } },
                },
            },
        },
    });
    parse_routed_json(&json.to_string(), &make_test_chipdb(), &IdStringPool::new())
}

fn wires(design: &Design) -> Vec<(WireId, Option<PipId>, PlaceStrength)> {
    let (_, net) = design.iter_alive_nets().next().expect("one net");
    let mut wires: Vec<_> = net
        .wires
        .iter()
        .map(|(&wire, binding)| (wire, binding.pip, binding.strength))
        .collect();
    wires.sort_by_key(|&(wire, _, _)| wire.index());
    wires
}

#[rstest]
#[case::eisenjoch_joins_entries(ROUTED)]
#[case::upstream_ends_each_entry("TILE_0_0X0Y0/W0;;2;TILE_0_0X0Y0/W1;TILE_0_0X0Y0/W1/W0;5;")]
fn routing_binds_each_wire_with_its_pip_and_strength(#[case] routing: &str) {
    let design = decode("TILE_1_0X1Y0/LUT0", PLACER, routing).expect("decodes");
    let (_, cell) = design.iter_alive_cells().next().expect("one cell");
    assert_eq!(
        (cell.bel, cell.bel_strength),
        (Some(BelId::new(1, 0)), PlaceStrength::Placer)
    );
    assert_eq!(
        wires(&design),
        vec![
            (WireId::new(0, 0), None, PlaceStrength::Strong),
            (
                WireId::new(0, 1),
                Some(PipId::new(0, 0)),
                PlaceStrength::Locked
            ),
        ]
    );
}

#[test]
fn padded_empty_routing_binds_no_wire() {
    let design = decode("TILE_0_0X0Y0/LUT0", PLACER, " ").expect("decodes");
    assert_eq!(wires(&design), vec![]);
}

#[test]
fn constant_bits_are_rejected_because_nextpnr_writes_constants_as_nets() {
    let json = json!({
        "modules": {
            "top": {
                "cells": {
                    "lut": {
                        "type": "LUT6",
                        "port_directions": { "I0": "input" },
                        "connections": { "I0": ["0"] },
                    },
                },
            },
        },
    });
    let decoded = parse_routed_json(&json.to_string(), &make_test_chipdb(), &IdStringPool::new());
    assert!(matches!(decoded, Err(RoutedJsonError::Netlist(_))));
}

fn cell(attribute: RoutedAttribute, fault: AttributeFault) -> RoutedJsonError {
    RoutedJsonError::Cell {
        cell: "lut".to_owned(),
        attribute,
        fault,
    }
}

fn net(fault: AttributeFault) -> RoutedJsonError {
    RoutedJsonError::Net {
        net: "o".to_owned(),
        attribute: RoutedAttribute::Routing,
        fault,
    }
}

fn unknown(kind: NameKind, name: &str, segment_kind: NameKind, segment: &str) -> AttributeFault {
    AttributeFault::Name(NameError::Unknown {
        kind,
        name: name.to_owned(),
        segment_kind,
        segment: segment.to_owned(),
    })
}

fn strength(value: &str) -> AttributeFault {
    AttributeFault::Strength {
        value: value.to_owned(),
    }
}

#[rstest]
#[case::unknown_tile(
    "TILE_9_9X9Y9/LUT0",
    PLACER,
    ROUTED,
    cell(
        RoutedAttribute::NextpnrBel,
        unknown(NameKind::Bel, "TILE_9_9X9Y9/LUT0", NameKind::Tile, "TILE_9_9X9Y9")
    )
)]
#[case::wrong_segment_count(
    "LUT0", PLACER, ROUTED,
    cell(RoutedAttribute::NextpnrBel, AttributeFault::Name(NameError::SegmentCount {
        kind: NameKind::Bel, name: "LUT0".to_owned(), found: 1, expected: 2,
    }))
)]
#[case::bel_strength_out_of_range(
    "TILE_0_0X0Y0/LUT0",
    "00000000000000000000000000000111",
    ROUTED,
    cell(
        RoutedAttribute::BelStrength,
        strength("00000000000000000000000000000111")
    )
)]
#[case::bel_strength_not_bits(
    "TILE_0_0X0Y0/LUT0",
    "PLACER",
    ROUTED,
    cell(RoutedAttribute::BelStrength, strength("PLACER"))
)]
#[case::unknown_wire(
    "TILE_0_0X0Y0/LUT0",
    PLACER,
    "TILE_0_0X0Y0/W7;;2",
    net(unknown(NameKind::Wire, "TILE_0_0X0Y0/W7", NameKind::Wire, "W7"))
)]
#[case::wire_strength_out_of_range(
    "TILE_0_0X0Y0/LUT0",
    PLACER,
    "TILE_0_0X0Y0/W0;;9",
    net(strength("9"))
)]
#[case::partial_entry(
    "TILE_0_0X0Y0/LUT0", PLACER, "TILE_0_0X0Y0/W0;;2;TILE_0_0X0Y0/W1",
    net(AttributeFault::EntryCount { pieces: 4 })
)]
#[case::pip_drives_another_wire(
    "TILE_0_0X0Y0/LUT0", PLACER, "TILE_0_0X0Y0/W0;TILE_0_0X0Y0/W1/W0;2",
    net(AttributeFault::PipDrivesOtherWire {
        wire: "TILE_0_0X0Y0/W0".to_owned(), pip: "TILE_0_0X0Y0/W1/W0".to_owned(),
    })
)]
#[case::wire_bound_twice(
    "TILE_0_0X0Y0/LUT0", PLACER, "TILE_0_0X0Y0/W0;;2;TILE_0_0X0Y0/W0;;3",
    net(AttributeFault::RepeatedWire { wire: "TILE_0_0X0Y0/W0".to_owned() })
)]
fn bad_attributes_name_the_cell_or_net(
    #[case] bel: &str,
    #[case] bel_strength: &str,
    #[case] routing: &str,
    #[case] error: RoutedJsonError,
) {
    assert_eq!(decode(bel, bel_strength, routing).err(), Some(error));
}

const ONE: &str = "00000000000000000000000000000001";
const TWO: &str = "00000000000000000000000000000010";

fn decode_state(
    lut: serde_json::Value,
    ff: serde_json::Value,
    net: serde_json::Value,
    settings: serde_json::Value,
) -> Result<Design, RoutedJsonError> {
    let json = json!({
        "modules": {
            "top": {
                "attributes": { "top": ONE },
                "settings": settings,
                "cells": {
                    "lut": {
                        "type": "LUT6",
                        "attributes": lut,
                        "port_directions": { "O": "output" },
                        "connections": { "O": [2] },
                    },
                    "ff": { "type": "DFF", "attributes": ff },
                },
                "netnames": { "o": { "bits": [2], "attributes": net } },
            },
        },
    });
    parse_routed_json(&json.to_string(), &make_test_chipdb(), &IdStringPool::new())
}

fn clustered(root: &str, abs_z: &str) -> serde_json::Value {
    json!({
        "EISENJOCH_CLUSTER": root,
        "EISENJOCH_CONSTR_X": ONE,
        "EISENJOCH_CONSTR_Y": ONE,
        "EISENJOCH_CONSTR_Z": ONE,
        "EISENJOCH_CONSTR_ABS_Z": abs_z,
    })
}

fn ff_cell(attribute: RoutedAttribute, fault: AttributeFault) -> RoutedJsonError {
    RoutedJsonError::Cell {
        cell: "ff".to_owned(),
        attribute,
        fault,
    }
}

#[rstest]
#[case::constraint_not_integer(
    json!({}), json!({ "EISENJOCH_CLUSTER": "ff", "EISENJOCH_CONSTR_X": "left",
        "EISENJOCH_CONSTR_Y": ONE, "EISENJOCH_CONSTR_Z": ONE, "EISENJOCH_CONSTR_ABS_Z": ONE }),
    json!({}), json!({}),
    ff_cell(RoutedAttribute::ConstrX, AttributeFault::Integer { value: "left".to_owned() })
)]
#[case::constraint_missing(
    json!({}), json!({ "EISENJOCH_CLUSTER": "ff" }), json!({}), json!({}),
    ff_cell(RoutedAttribute::ConstrAbsZ, AttributeFault::Missing)
)]
#[case::abs_z_not_a_flag(
    json!({}), clustered("ff", TWO), json!({}), json!({}),
    ff_cell(RoutedAttribute::ConstrAbsZ, AttributeFault::Flag { value: 2 })
)]
#[case::root_not_in_netlist(
    json!({}), clustered("gone", ONE), json!({}), json!({}),
    ff_cell(RoutedAttribute::Cluster, AttributeFault::NoSuchCell { name: "gone".to_owned() })
)]
#[case::root_not_rooting_a_cluster(
    json!({}), clustered("lut", ONE), json!({}), json!({}),
    ff_cell(RoutedAttribute::Cluster, AttributeFault::NotClusterRoot { root: "lut".to_owned() })
)]
#[case::constraint_without_cluster(
    json!({}), json!({ "EISENJOCH_CONSTR_Y": ONE }), json!({}), json!({}),
    ff_cell(RoutedAttribute::ConstrY, AttributeFault::WithoutCluster)
)]
#[case::undefined_region(
    json!({}), json!({ "EISENJOCH_REGION": "pblock" }), json!({}), json!({}),
    ff_cell(RoutedAttribute::Region, AttributeFault::NoSuchRegion { name: "pblock".to_owned() })
)]
#[case::clock_period_not_positive(
    json!({}), json!({}), json!({ "EISENJOCH_CLOCK_PERIOD_PS": "0" }), json!({}),
    RoutedJsonError::Net {
        net: "o".to_owned(),
        attribute: RoutedAttribute::ClockPeriod,
        fault: AttributeFault::ClockPeriod { value: 0 },
    }
)]
#[case::malformed_region_rects(
    json!({}), json!({}), json!({}), json!({ "EISENJOCH_REGION:pblock": "1,1,2" }),
    RoutedJsonError::Setting {
        key: "EISENJOCH_REGION:pblock".to_owned(),
        fault: AttributeFault::Rects { value: "1,1,2".to_owned() },
    }
)]
fn bad_eisenjoch_state_names_the_cell_net_or_setting(
    #[case] lut: serde_json::Value,
    #[case] ff: serde_json::Value,
    #[case] net: serde_json::Value,
    #[case] settings: serde_json::Value,
    #[case] error: RoutedJsonError,
) {
    assert_eq!(decode_state(lut, ff, net, settings).err(), Some(error));
}

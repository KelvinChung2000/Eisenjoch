//! The routed design in nextpnr's `write_json` form. Placement and routing travel in the
//! `NEXTPNR_BEL`, `BEL_STRENGTH` and `ROUTING` attributes that nextpnr's `archInfoToAttributes` sets.

use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use crate::chipdb::{BelId, PipId, WireId};
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellInfo, NetId, NetInfo, PortType, Property};

const CREATOR: &str = "nextpnr-himbaechel-rust";
const YOSYS_TRUE: &str = "00000000000000000000000000000001";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RoutedJsonError {
    #[error("{owner} is the integer {value}, which does not fit the 32-bit property nextpnr writes. Give it as a bit vector in the netlist.")]
    IntegerOutOfRange { owner: String, value: i64 },
}

type Properties = BTreeMap<String, String>;

#[derive(Serialize)]
struct Root {
    creator: &'static str,
    modules: BTreeMap<String, Module>,
}

#[derive(Serialize)]
struct Module {
    settings: Properties,
    attributes: Properties,
    ports: serde_json::Map<String, serde_json::Value>,
    cells: BTreeMap<String, Cell>,
    netnames: BTreeMap<String, NetName>,
}

#[derive(Serialize)]
struct Cell {
    hide_name: u8,
    #[serde(rename = "type")]
    cell_type: String,
    parameters: Properties,
    attributes: Properties,
    port_directions: BTreeMap<String, Direction>,
    connections: BTreeMap<String, Vec<u32>>,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Input,
    Output,
    Inout,
}

#[derive(Serialize)]
struct NetName {
    hide_name: u8,
    bits: Vec<u32>,
    attributes: Properties,
}

/// The routed design as one top module, cells carrying their BEL and nets their wires and pips.
///
/// Fails when a property is an integer wider than the 32 bits nextpnr writes.
pub fn routed_json(ctx: &Context) -> Result<serde_json::Value, RoutedJsonError> {
    let cells = ctx
        .design
        .iter_alive_cells()
        .map(|(_, cell)| Ok((ctx.name_of(cell.name).to_owned(), routed_cell(ctx, cell)?)))
        .collect::<Result<_, RoutedJsonError>>()?;
    let netnames = ctx
        .design
        .iter_alive_nets()
        .map(|(id, net)| Ok((ctx.name_of(net.name).to_owned(), routed_net(ctx, id, net)?)))
        .collect::<Result<_, RoutedJsonError>>()?;
    let module = Module {
        settings: Properties::new(),
        attributes: Properties::from([("top".to_owned(), YOSYS_TRUE.to_owned())]),
        ports: serde_json::Map::new(),
        cells,
        netnames,
    };
    let root = Root {
        creator: CREATOR,
        modules: BTreeMap::from([(ctx.name_of(ctx.design.top_module).to_owned(), module)]),
    };
    Ok(serde_json::to_value(root).expect("string-keyed maps of strings serialise"))
}

fn routed_cell(ctx: &Context, cell: &CellInfo) -> Result<Cell, RoutedJsonError> {
    let name = ctx.name_of(cell.name);
    let mut attributes = encode_properties(ctx, &cell.attrs, |key| {
        format!("Attribute {key} of cell {name}")
    })?;
    if let Some(bel) = cell.bel {
        attributes.remove("BEL");
        attributes.insert("NEXTPNR_BEL".to_owned(), bel_name(ctx, bel));
        attributes.insert(
            "BEL_STRENGTH".to_owned(),
            yosys_int(cell.bel_strength as i32),
        );
    }
    let mut port_directions = BTreeMap::new();
    let mut connections = BTreeMap::new();
    for (&port, data) in &cell.ports {
        let port = ctx.name_of(port).to_owned();
        port_directions.insert(port.clone(), direction(data.port_type));
        connections.insert(port, data.net.map(net_bit).into_iter().collect());
    }
    Ok(Cell {
        hide_name: hide_name(name),
        cell_type: ctx.name_of(cell.cell_type).to_owned(),
        parameters: encode_properties(ctx, &cell.params, |key| {
            format!("Parameter {key} of cell {name}")
        })?,
        attributes,
        port_directions,
        connections,
    })
}

fn routed_net(ctx: &Context, id: NetId, net: &NetInfo) -> Result<NetName, RoutedJsonError> {
    let name = ctx.name_of(net.name);
    let mut attributes = encode_properties(ctx, &net.attrs, |key| {
        format!("Attribute {key} of net {name}")
    })?;
    let mut routing: Vec<String> = net
        .wires
        .iter()
        .map(|(&wire, binding)| {
            let pip = binding
                .pip
                .map(|pip| pip_name(ctx, pip))
                .unwrap_or_default();
            format!("{};{pip};{}", wire_name(ctx, wire), binding.strength as i32)
        })
        .collect();
    routing.sort();
    attributes.insert("ROUTING".to_owned(), yosys_string(&routing.join(";")));
    Ok(NetName {
        hide_name: hide_name(name),
        bits: vec![net_bit(id)],
        attributes,
    })
}

fn encode_properties(
    ctx: &Context,
    properties: &rustc_hash::FxHashMap<IdString, Property>,
    owner: impl Fn(&str) -> String,
) -> Result<Properties, RoutedJsonError> {
    properties
        .iter()
        .map(|(&key, value)| {
            let key = ctx.name_of(key);
            Ok((key.to_owned(), encode_property(value, || owner(key))?))
        })
        .collect()
}

fn encode_property(
    value: &Property,
    owner: impl FnOnce() -> String,
) -> Result<String, RoutedJsonError> {
    match value {
        Property::String(text) => Ok(yosys_string(text)),
        Property::BitVector(bits) => Ok(bits.clone()),
        Property::Int(value) => {
            i32::try_from(*value)
                .map(yosys_int)
                .map_err(|_| RoutedJsonError::IntegerOutOfRange {
                    owner: owner(),
                    value: *value,
                })
        }
    }
}

fn yosys_string(text: &str) -> String {
    let reads_as_bits = text
        .trim_end_matches(' ')
        .chars()
        .all(|c| matches!(c, '0' | '1' | 'x' | 'z'));
    if reads_as_bits {
        format!("{text} ")
    } else {
        text.to_owned()
    }
}

fn yosys_int(value: i32) -> String {
    format!("{:032b}", value as u32)
}

fn direction(port_type: PortType) -> Direction {
    match port_type {
        PortType::In => Direction::Input,
        PortType::Out => Direction::Output,
        PortType::InOut => Direction::Inout,
    }
}

fn hide_name(name: &str) -> u8 {
    u8::from(name.starts_with('$'))
}

fn net_bit(net: NetId) -> u32 {
    net.slot() + 2
}

fn bel_name(ctx: &Context, bel: BelId) -> String {
    let chipdb = ctx.chipdb();
    format!("{}/{}", chipdb.tile_name(bel.tile()), chipdb.bel_name(bel))
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
        chipdb.wire_name(chipdb.pip_src_tile_wire(pip))
    )
}

//! The routed design in nextpnr's `write_json` form. Placement and routing travel in the
//! `NEXTPNR_BEL`, `BEL_STRENGTH` and `ROUTING` attributes that nextpnr's `archInfoToAttributes` sets.
//! Clusters, regions, clock periods and SDC, which upstream does not write, travel in `EISENJOCH_`
//! attributes and module settings, so [`reopen`] rebuilds an editable context from the file alone.

use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use crate::chipdb::{BelId, ChipDb, PipId, WireId};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::frontend::{
    parse_routed_json, region_rects, RoutedAttribute, REGION_SETTING_PREFIX, SDC_SETTING,
};
use crate::netlist::{CellId, CellInfo, NetId, NetInfo, PortType, Property};

const CREATOR: &str = "nextpnr-himbaechel-rust";
const YOSYS_TRUE: &str = "00000000000000000000000000000001";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RoutedJsonError {
    #[error("{owner} is the integer {value}, which does not fit the 32-bit property nextpnr writes. Give it as a bit vector in the netlist.")]
    IntegerOutOfRange { owner: String, value: i64 },
    #[error("Two regions are named {name}, and the routed JSON keys regions by name. Rename one.")]
    DuplicateRegion { name: String },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReopenError {
    #[error(transparent)]
    Json(#[from] crate::frontend::RoutedJsonError),
    #[error("cell {cell} is placed on BEL {bel}, which cell {holder} already holds")]
    BelHeld {
        cell: String,
        bel: String,
        holder: String,
    },
    #[error("net {net} routes through wire {wire}, whose node net {holder} already holds")]
    WireHeld {
        net: String,
        wire: String,
        holder: String,
    },
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
        settings: settings(ctx)?,
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

/// An editable context on `chipdb` holding the design that `routed_json` describes, bound in place.
///
/// User and Locked bindings keep their strength and every other binding becomes Strong. Fails on a
/// file `parse_routed_json` rejects, two cells on one BEL, or two nets on one wire node.
pub fn reopen(chipdb: ChipDb, routed_json: &str) -> Result<Context, ReopenError> {
    let mut ctx = Context::new(chipdb);
    ctx.design = parse_routed_json(routed_json, ctx.chipdb(), &ctx.id_pool)?;
    let placements: Vec<(CellId, BelId, PlaceStrength)> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(id, cell)| Some((id, cell.bel?, reopened(cell.bel_strength))))
        .collect();
    for (cell, bel, strength) in placements {
        if let Some(holder) = ctx.conflicting_bel_cell(bel) {
            return Err(ReopenError::BelHeld {
                cell: cell_name(&ctx, cell),
                bel: bel_name(&ctx, bel),
                holder: cell_name(&ctx, holder),
            });
        }
        assert!(ctx.bind_bel(bel, cell, strength), "a free chipdb BEL binds");
    }
    let routes: Vec<(NetId, WireId, Option<PipId>, PlaceStrength)> = ctx
        .design
        .iter_alive_nets()
        .flat_map(|(id, net)| {
            net.wires
                .iter()
                .map(move |(&wire, binding)| (id, wire, binding.pip, reopened(binding.strength)))
        })
        .collect();
    for (net, wire, pip, strength) in routes {
        if let Err(wire) = ctx.try_bind_wire_node(wire, net, strength) {
            let (holder, _) = ctx.wire_binding(wire).expect("a refused wire is bound");
            return Err(ReopenError::WireHeld {
                net: net_name(&ctx, net),
                wire: wire_name(&ctx, wire),
                holder: net_name(&ctx, holder),
            });
        }
        if let Some(pip) = pip {
            ctx.bind_pip(pip, net, strength);
        }
        ctx.design.net_edit(net).add_wire(wire, pip, strength);
    }
    Ok(ctx)
}

fn reopened(strength: PlaceStrength) -> PlaceStrength {
    match strength {
        PlaceStrength::User | PlaceStrength::Locked => strength,
        PlaceStrength::None
        | PlaceStrength::Weak
        | PlaceStrength::Strong
        | PlaceStrength::Placer
        | PlaceStrength::Fixed => PlaceStrength::Strong,
    }
}

fn cell_name(ctx: &Context, cell: CellId) -> String {
    ctx.name_of(ctx.design.cell(cell).name).to_owned()
}

fn net_name(ctx: &Context, net: NetId) -> String {
    ctx.name_of(ctx.design.net(net).name).to_owned()
}

fn settings(ctx: &Context) -> Result<Properties, RoutedJsonError> {
    let mut settings = Properties::new();
    if !ctx.design.sdc.is_empty() {
        settings.insert(SDC_SETTING.to_owned(), yosys_string(&ctx.design.sdc));
    }
    for region in &ctx.design.regions {
        let name = ctx.name_of(region.name);
        let rects = yosys_string(&region_rects(&region.rects));
        if settings
            .insert(format!("{REGION_SETTING_PREFIX}{name}"), rects)
            .is_some()
        {
            return Err(RoutedJsonError::DuplicateRegion {
                name: name.to_owned(),
            });
        }
    }
    Ok(settings)
}

fn region_name(ctx: &Context, region: u32) -> String {
    yosys_string(ctx.name_of(ctx.design.region(region).name))
}

fn routed_cell(ctx: &Context, cell: &CellInfo) -> Result<Cell, RoutedJsonError> {
    let name = ctx.name_of(cell.name);
    let mut attributes = encode_properties(ctx, &cell.attrs, |key| {
        format!("Attribute {key} of cell {name}")
    })?;
    if let Some(bel) = cell.bel {
        attributes.remove("BEL");
        attributes.insert(
            RoutedAttribute::NextpnrBel.key().to_owned(),
            bel_name(ctx, bel),
        );
        attributes.insert(
            RoutedAttribute::BelStrength.key().to_owned(),
            yosys_int(cell.bel_strength as i32),
        );
    }
    if let Some(root) = cell.cluster {
        attributes.insert(
            RoutedAttribute::Cluster.key().to_owned(),
            yosys_string(&cell_name(ctx, root)),
        );
        for (attribute, value) in [
            (RoutedAttribute::ConstrX, cell.constr_x),
            (RoutedAttribute::ConstrY, cell.constr_y),
            (RoutedAttribute::ConstrZ, cell.constr_z),
            (RoutedAttribute::ConstrAbsZ, i32::from(cell.constr_abs_z)),
        ] {
            attributes.insert(attribute.key().to_owned(), yosys_int(value));
        }
    }
    if let Some(region) = cell.region {
        attributes.insert(
            RoutedAttribute::Region.key().to_owned(),
            region_name(ctx, region),
        );
    }
    let mut port_directions = BTreeMap::new();
    let mut connections = BTreeMap::new();
    for (&port, data) in cell.ports() {
        let port = ctx.name_of(port).to_owned();
        port_directions.insert(port.clone(), direction(data.port_type()));
        connections.insert(port, data.net().map(net_bit).into_iter().collect());
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
    let wires = net
        .wires
        .iter()
        .map(|(&wire, binding)| (wire, binding.pip, binding.strength));
    attributes.insert(
        RoutedAttribute::Routing.key().to_owned(),
        yosys_string(&routing(ctx, wires)),
    );
    if net.clock_constraint > 0 {
        attributes.insert(
            RoutedAttribute::ClockPeriod.key().to_owned(),
            yosys_int(net.clock_constraint),
        );
    }
    if let Some(region) = net.region {
        attributes.insert(
            RoutedAttribute::Region.key().to_owned(),
            region_name(ctx, region),
        );
    }
    Ok(NetName {
        hide_name: hide_name(name),
        bits: vec![net_bit(id)],
        attributes,
    })
}

/// A net's wires in the `ROUTING` form, `wire;pip;strength` triples sorted and joined by `;`.
pub(crate) fn routing(
    ctx: &Context,
    wires: impl Iterator<Item = (WireId, Option<PipId>, PlaceStrength)>,
) -> String {
    let mut routing: Vec<String> = wires
        .map(|(wire, pip, strength)| {
            let pip = pip.map(|pip| pip_name(ctx, pip)).unwrap_or_default();
            format!("{};{pip};{}", wire_name(ctx, wire), strength as i32)
        })
        .collect();
    routing.sort();
    routing.join(";")
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

pub(crate) fn bel_name(ctx: &Context, bel: BelId) -> String {
    let chipdb = ctx.chipdb();
    format!("{}/{}", chipdb.tile_name(bel.tile()), chipdb.bel_name(bel))
}

pub(crate) fn wire_name(ctx: &Context, wire: WireId) -> String {
    let chipdb = ctx.chipdb();
    format!(
        "{}/{}",
        chipdb.tile_name(wire.tile()),
        chipdb.wire_name(wire)
    )
}

pub(crate) fn pip_name(ctx: &Context, pip: PipId) -> String {
    let chipdb = ctx.chipdb();
    format!(
        "{}/{}/{}",
        chipdb.tile_name(pip.tile()),
        chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)),
        chipdb.wire_name(chipdb.pip_src_tile_wire(pip))
    )
}

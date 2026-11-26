use std::fmt;

use rustc_hash::FxHashMap;
use thiserror::Error;

use super::parser::{parse_netlist, ConstantDrivers};
use crate::chipdb::{BelId, ChipDb, NameError, WireId};
use crate::common::{IdString, IdStringPool, PlaceStrength};
use crate::netlist::{CellInfo, Design, NetInfo, PipMap, Property};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedAttribute {
    NextpnrBel,
    BelStrength,
    Routing,
}

impl RoutedAttribute {
    const fn key(self) -> &'static str {
        match self {
            Self::NextpnrBel => "NEXTPNR_BEL",
            Self::BelStrength => "BEL_STRENGTH",
            Self::Routing => "ROUTING",
        }
    }
}

impl fmt::Display for RoutedAttribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AttributeFault {
    #[error("is missing")]
    Missing,
    #[error("is not a string")]
    NotString,
    #[error("holds {value:?}, which is not a placement strength")]
    Strength { value: String },
    #[error("names nothing in the chipdb: {0}")]
    Name(#[from] NameError),
    #[error("splits into {pieces} ';'-separated fields, not whole wire;pip;strength entries")]
    EntryCount { pieces: usize },
    #[error("binds wire {wire:?} through pip {pip:?}, which drives another wire")]
    PipDrivesOtherWire { wire: String, pip: String },
    #[error("binds wire {wire:?} twice")]
    RepeatedWire { wire: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RoutedJsonError {
    #[error("routed JSON is not a netlist: {0}")]
    Netlist(String),
    #[error("cell {cell} attribute {attribute} {fault}")]
    Cell {
        cell: String,
        attribute: RoutedAttribute,
        fault: AttributeFault,
    },
    #[error("net {net} attribute {} {fault}", RoutedAttribute::Routing)]
    Net { net: String, fault: AttributeFault },
}

struct Keys {
    bel: IdString,
    strength: IdString,
    routing: IdString,
}

/// The netlist nextpnr writes with `--write`, its cells placed and its nets routed on `chipdb`.
pub fn parse_routed_json(
    json_str: &str,
    chipdb: &ChipDb,
    pool: &IdStringPool,
) -> Result<Design, RoutedJsonError> {
    let mut design = parse_netlist(json_str, pool, ConstantDrivers::InNetlist)
        .map_err(|error| RoutedJsonError::Netlist(format!("{error:#}")))?;
    let keys = Keys {
        bel: pool.intern(RoutedAttribute::NextpnrBel.key()),
        strength: pool.intern(RoutedAttribute::BelStrength.key()),
        routing: pool.intern(RoutedAttribute::Routing.key()),
    };
    let name = |id: IdString| {
        pool.lookup(id)
            .expect("the frontend interns every name in the pool")
            .to_owned()
    };
    let placements = design
        .iter_alive_cells()
        .filter_map(|(id, cell)| {
            placement(chipdb, &keys, cell)
                .map_err(|(attribute, fault)| RoutedJsonError::Cell {
                    cell: name(cell.name),
                    attribute,
                    fault,
                })
                .transpose()
                .map(|placement| placement.map(|placement| (id, placement)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let routes = design
        .iter_alive_nets()
        .map(|(id, net)| {
            routing(chipdb, &keys, net)
                .map(|wires| (id, wires))
                .map_err(|fault| RoutedJsonError::Net {
                    net: name(net.name),
                    fault,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (cell, (bel, strength)) in placements {
        design.cell_edit(cell).set_bel(Some(bel), strength);
    }
    for (net, wires) in routes {
        let mut net = design.net_edit(net);
        for (wire, PipMap { pip, strength }) in wires {
            net.add_wire(wire, pip, strength);
        }
    }
    Ok(design)
}

fn placement(
    chipdb: &ChipDb,
    keys: &Keys,
    cell: &CellInfo,
) -> Result<Option<(BelId, PlaceStrength)>, (RoutedAttribute, AttributeFault)> {
    let Some(bel) = cell.attrs.get(&keys.bel) else {
        return Ok(None);
    };
    let bel = text(bel)
        .and_then(|name| chipdb.bel_by_name(name).map_err(AttributeFault::from))
        .map_err(|fault| (RoutedAttribute::NextpnrBel, fault))?;
    let strength = cell
        .attrs
        .get(&keys.strength)
        .ok_or(AttributeFault::Missing)
        .and_then(bel_strength)
        .map_err(|fault| (RoutedAttribute::BelStrength, fault))?;
    Ok(Some((bel, strength)))
}

fn bel_strength(value: &Property) -> Result<PlaceStrength, AttributeFault> {
    let Property::BitVector(bits) = value else {
        return Err(AttributeFault::Strength {
            value: value.as_str(),
        });
    };
    u32::from_str_radix(bits, 2)
        .ok()
        .and_then(|value| PlaceStrength::try_from(i64::from(value)).ok())
        .ok_or_else(|| AttributeFault::Strength {
            value: bits.clone(),
        })
}

fn routing(
    chipdb: &ChipDb,
    keys: &Keys,
    net: &NetInfo,
) -> Result<FxHashMap<WireId, PipMap>, AttributeFault> {
    let Some(value) = net.attrs.get(&keys.routing) else {
        return Ok(FxHashMap::default());
    };
    let fields: Vec<&str> = unpad(text(value)?).split_terminator(';').collect();
    let (entries, remainder) = fields.as_chunks::<3>();
    if !remainder.is_empty() {
        return Err(AttributeFault::EntryCount {
            pieces: fields.len(),
        });
    }
    let mut wires = FxHashMap::default();
    for &[wire_name, pip_name, strength] in entries {
        let wire = chipdb.wire_by_name(wire_name)?;
        let pip = match pip_name {
            "" => None,
            pip_name => Some(chipdb.pip_by_name(pip_name)?),
        };
        if pip.is_some_and(|pip| chipdb.pip_dst_wire(pip) != wire) {
            return Err(AttributeFault::PipDrivesOtherWire {
                wire: wire_name.to_owned(),
                pip: pip_name.to_owned(),
            });
        }
        let strength = strength
            .parse::<i64>()
            .ok()
            .and_then(|value| PlaceStrength::try_from(value).ok())
            .ok_or_else(|| AttributeFault::Strength {
                value: strength.to_owned(),
            })?;
        if wires.insert(wire, PipMap { pip, strength }).is_some() {
            return Err(AttributeFault::RepeatedWire {
                wire: wire_name.to_owned(),
            });
        }
    }
    Ok(wires)
}

fn text(value: &Property) -> Result<&str, AttributeFault> {
    match value {
        Property::String(text) => Ok(text),
        Property::Int(_) | Property::BitVector(_) => Err(AttributeFault::NotString),
    }
}

fn unpad(text: &str) -> &str {
    let padding = text.trim_start_matches(['0', '1', 'x', 'z']);
    if !padding.is_empty() && padding.bytes().all(|byte| byte == b' ') {
        &text[..text.len() - 1]
    } else {
        text
    }
}

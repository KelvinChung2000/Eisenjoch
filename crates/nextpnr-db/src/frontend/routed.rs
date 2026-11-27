use std::fmt;

use rustc_hash::FxHashMap;
use serde_json::Value;
use thiserror::Error;

use super::helpers::parse_property;
use super::parser::{parse_netlist_value, top_module, ConstantDrivers};
use crate::chipdb::{BelId, ChipDb, NameError, WireId};
use crate::common::{DelayT, IdString, IdStringPool, PlaceStrength};
use crate::netlist::{CellId, CellInfo, Cluster, Design, NetInfo, PipMap, Property, Rect};

pub const SDC_SETTING: &str = "EISENJOCH_SDC";
pub const REGION_SETTING_PREFIX: &str = "EISENJOCH_REGION:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedAttribute {
    NextpnrBel,
    BelStrength,
    Routing,
    Cluster,
    ConstrX,
    ConstrY,
    ConstrZ,
    ConstrAbsZ,
    Region,
    ClockPeriod,
}

impl RoutedAttribute {
    pub const ALL: [Self; 10] = [
        Self::NextpnrBel,
        Self::BelStrength,
        Self::Routing,
        Self::Cluster,
        Self::ConstrX,
        Self::ConstrY,
        Self::ConstrZ,
        Self::ConstrAbsZ,
        Self::Region,
        Self::ClockPeriod,
    ];

    const CONSTRAINTS: [Self; 4] = [
        Self::ConstrX,
        Self::ConstrY,
        Self::ConstrZ,
        Self::ConstrAbsZ,
    ];

    pub const fn key(self) -> &'static str {
        match self {
            Self::NextpnrBel => "NEXTPNR_BEL",
            Self::BelStrength => "BEL_STRENGTH",
            Self::Routing => "ROUTING",
            Self::Cluster => "EISENJOCH_CLUSTER",
            Self::ConstrX => "EISENJOCH_CONSTR_X",
            Self::ConstrY => "EISENJOCH_CONSTR_Y",
            Self::ConstrZ => "EISENJOCH_CONSTR_Z",
            Self::ConstrAbsZ => "EISENJOCH_CONSTR_ABS_Z",
            Self::Region => "EISENJOCH_REGION",
            Self::ClockPeriod => "EISENJOCH_CLOCK_PERIOD_PS",
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
    #[error("holds {value:?}, which is not a 32-bit integer")]
    Integer { value: String },
    #[error("holds {value}, which is neither 0 nor 1")]
    Flag { value: i32 },
    #[error("names cell {name:?}, which the netlist does not have")]
    NoSuchCell { name: String },
    #[error(
        "names cell {root:?}, whose own {} does not name itself",
        RoutedAttribute::Cluster
    )]
    NotClusterRoot { root: String },
    #[error("is set on a cell without {}", RoutedAttribute::Cluster)]
    WithoutCluster,
    #[error("names region {name:?}, which no {REGION_SETTING_PREFIX} setting defines")]
    NoSuchRegion { name: String },
    #[error("holds {value} ps, which is not a positive clock period")]
    ClockPeriod { value: DelayT },
    #[error("holds {value:?}, which is not ';'-separated x0,y0,x1,y1 rectangles")]
    Rects { value: String },
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
    #[error("net {net} attribute {attribute} {fault}")]
    Net {
        net: String,
        attribute: RoutedAttribute,
        fault: AttributeFault,
    },
    #[error("module setting {key} {fault}")]
    Setting { key: String, fault: AttributeFault },
}

type Fault = (RoutedAttribute, AttributeFault);

struct Keys([IdString; RoutedAttribute::ALL.len()]);

impl Keys {
    fn new(pool: &IdStringPool) -> Self {
        Self(RoutedAttribute::ALL.map(|attribute| pool.intern(attribute.key())))
    }

    fn get(&self, attribute: RoutedAttribute) -> IdString {
        self.0[attribute as usize]
    }

    fn is_reserved(&self, key: IdString) -> bool {
        self.0.contains(&key)
    }
}

struct ClusterSlot {
    root: CellId,
    x: i32,
    y: i32,
    z: i32,
    abs_z: bool,
}

struct CellState {
    placement: Option<(BelId, PlaceStrength)>,
    cluster: Option<ClusterSlot>,
    region: Option<u32>,
}

struct NetState {
    wires: FxHashMap<WireId, PipMap>,
    clock_period: Option<DelayT>,
    region: Option<u32>,
}

struct Decoder<'a> {
    chipdb: &'a ChipDb,
    pool: &'a IdStringPool,
    design: &'a Design,
    keys: &'a Keys,
}

/// The netlist nextpnr writes with `--write`, its cells placed and its nets routed on `chipdb`.
///
/// Also restores the clusters, regions, clock periods and SDC that `EISENJOCH_` attributes and
/// settings carry, and strips every attribute it decodes. Fails naming the cell, net or setting.
pub fn parse_routed_json(
    json_str: &str,
    chipdb: &ChipDb,
    pool: &IdStringPool,
) -> Result<Design, RoutedJsonError> {
    let netlist = |error: anyhow::Error| RoutedJsonError::Netlist(format!("{error:#}"));
    let json: Value = serde_json::from_str(json_str)
        .map_err(|error| RoutedJsonError::Netlist(error.to_string()))?;
    let mut design =
        parse_netlist_value(&json, pool, ConstantDrivers::InNetlist).map_err(netlist)?;
    let (_, top) = top_module(&json).map_err(netlist)?;
    apply_settings(&mut design, pool, top)?;
    let keys = Keys::new(pool);
    let decoder = Decoder {
        chipdb,
        pool,
        design: &design,
        keys: &keys,
    };
    let name = |id: IdString| {
        pool.lookup(id)
            .expect("the frontend interns every name in the pool")
            .to_owned()
    };
    let cell_name = |id: CellId| name(design.cell(id).name);
    let cells = design
        .iter_alive_cells()
        .map(|(id, cell)| {
            decoder
                .cell_state(cell)
                .map(|state| (id, state))
                .map_err(|(attribute, fault)| RoutedJsonError::Cell {
                    cell: name(cell.name),
                    attribute,
                    fault,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let roots: FxHashMap<CellId, CellId> = cells
        .iter()
        .filter_map(|(id, state)| Some((*id, state.cluster.as_ref()?.root)))
        .collect();
    if let Some((id, root)) = roots
        .iter()
        .find(|&(_, root)| roots.get(root) != Some(root))
    {
        return Err(RoutedJsonError::Cell {
            cell: cell_name(*id),
            attribute: RoutedAttribute::Cluster,
            fault: AttributeFault::NotClusterRoot {
                root: cell_name(*root),
            },
        });
    }
    let nets = design
        .iter_alive_nets()
        .map(|(id, net)| {
            decoder
                .net_state(net)
                .map(|state| (id, state))
                .map_err(|(attribute, fault)| RoutedJsonError::Net {
                    net: name(net.name),
                    attribute,
                    fault,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (id, state) in cells {
        apply_cell(&mut design, &keys, id, state);
    }
    for (id, state) in nets {
        let mut net = design.net_edit(id);
        for (wire, PipMap { pip, strength }) in state.wires {
            net.add_wire(wire, pip, strength);
        }
        if let Some(period) = state.clock_period {
            net.set_clock_constraint(period);
        }
        net.set_region(state.region);
        design
            .net_mut(id)
            .attrs
            .retain(|&key, _| !keys.is_reserved(key));
    }
    Ok(design)
}

/// The `x0,y0,x1,y1` rectangles of a region joined by `;`, as its `EISENJOCH_REGION:` setting holds them.
pub fn region_rects(rects: &[Rect]) -> String {
    rects
        .iter()
        .map(|rect| format!("{},{},{},{}", rect.x0, rect.y0, rect.x1, rect.y1))
        .collect::<Vec<_>>()
        .join(";")
}

fn apply_settings(
    design: &mut Design,
    pool: &IdStringPool,
    top: &Value,
) -> Result<(), RoutedJsonError> {
    let Some(settings) = top.get("settings") else {
        return Ok(());
    };
    let settings = settings
        .as_object()
        .ok_or_else(|| RoutedJsonError::Netlist("module settings is not an object".to_owned()))?;
    for (key, value) in settings {
        let fault = |fault| RoutedJsonError::Setting {
            key: key.clone(),
            fault,
        };
        let value = parse_property(value)
            .map_err(|error| RoutedJsonError::Netlist(format!("setting {key}: {error:#}")))?;
        if key == SDC_SETTING {
            design.sdc = string(&value).map_err(fault)?.to_owned();
        } else if let Some(name) = key.strip_prefix(REGION_SETTING_PREFIX) {
            let rects = string(&value).and_then(rects).map_err(fault)?;
            let region = design.add_region(pool.intern(name));
            design.region_mut(region).rects = rects;
        }
    }
    Ok(())
}

fn apply_cell(design: &mut Design, keys: &Keys, id: CellId, state: CellState) {
    let mut cell = design.cell_edit(id);
    if let Some((bel, strength)) = state.placement {
        cell.set_bel(Some(bel), strength);
    }
    cell.set_region(state.region);
    if let Some(slot) = state.cluster {
        cell.set_cluster(Some(slot.root))
            .set_constraints(slot.x, slot.y, slot.z, slot.abs_z);
        let cluster = design
            .clusters
            .entry(slot.root)
            .or_insert_with(|| Cluster::new(slot.root));
        if id != slot.root {
            cluster.add_member(id);
            cluster.add_constrained_child(id);
        }
    }
    design
        .cell_mut(id)
        .attrs
        .retain(|&key, _| !keys.is_reserved(key));
}

impl Decoder<'_> {
    fn attr<'p>(
        &self,
        attrs: &'p FxHashMap<IdString, Property>,
        attribute: RoutedAttribute,
    ) -> Option<&'p Property> {
        attrs.get(&self.keys.get(attribute))
    }

    fn cell_state(&self, cell: &CellInfo) -> Result<CellState, Fault> {
        Ok(CellState {
            placement: self.placement(cell)?,
            cluster: self.cluster_slot(&cell.attrs)?,
            region: self.region(&cell.attrs)?,
        })
    }

    fn net_state(&self, net: &NetInfo) -> Result<NetState, Fault> {
        let clock_period = self
            .attr(&net.attrs, RoutedAttribute::ClockPeriod)
            .map(clock_period)
            .transpose()
            .map_err(|fault| (RoutedAttribute::ClockPeriod, fault))?;
        Ok(NetState {
            wires: self
                .routing(net)
                .map_err(|fault| (RoutedAttribute::Routing, fault))?,
            clock_period,
            region: self.region(&net.attrs)?,
        })
    }

    fn placement(&self, cell: &CellInfo) -> Result<Option<(BelId, PlaceStrength)>, Fault> {
        let Some(bel) = self.attr(&cell.attrs, RoutedAttribute::NextpnrBel) else {
            return Ok(None);
        };
        let bel = text(bel)
            .and_then(|name| self.chipdb.bel_by_name(name).map_err(AttributeFault::from))
            .map_err(|fault| (RoutedAttribute::NextpnrBel, fault))?;
        let strength = self
            .attr(&cell.attrs, RoutedAttribute::BelStrength)
            .ok_or(AttributeFault::Missing)
            .and_then(bel_strength)
            .map_err(|fault| (RoutedAttribute::BelStrength, fault))?;
        Ok(Some((bel, strength)))
    }

    fn cluster_slot(
        &self,
        attrs: &FxHashMap<IdString, Property>,
    ) -> Result<Option<ClusterSlot>, Fault> {
        let Some(root) = self.attr(attrs, RoutedAttribute::Cluster) else {
            return match RoutedAttribute::CONSTRAINTS
                .into_iter()
                .find(|&attribute| self.attr(attrs, attribute).is_some())
            {
                Some(attribute) => Err((attribute, AttributeFault::WithoutCluster)),
                None => Ok(None),
            };
        };
        let root = string(root)
            .and_then(|name| self.cell_named(name))
            .map_err(|fault| (RoutedAttribute::Cluster, fault))?;
        let constraint = |attribute| {
            self.attr(attrs, attribute)
                .ok_or(AttributeFault::Missing)
                .and_then(int32)
                .map_err(|fault| (attribute, fault))
        };
        let abs_z = match constraint(RoutedAttribute::ConstrAbsZ)? {
            0 => false,
            1 => true,
            value => return Err((RoutedAttribute::ConstrAbsZ, AttributeFault::Flag { value })),
        };
        Ok(Some(ClusterSlot {
            root,
            x: constraint(RoutedAttribute::ConstrX)?,
            y: constraint(RoutedAttribute::ConstrY)?,
            z: constraint(RoutedAttribute::ConstrZ)?,
            abs_z,
        }))
    }

    fn cell_named(&self, name: &str) -> Result<CellId, AttributeFault> {
        self.design
            .cell_by_name(self.pool.intern(name))
            .ok_or_else(|| AttributeFault::NoSuchCell {
                name: name.to_owned(),
            })
    }

    fn region(&self, attrs: &FxHashMap<IdString, Property>) -> Result<Option<u32>, Fault> {
        self.attr(attrs, RoutedAttribute::Region)
            .map(|value| {
                let name = string(value)?;
                self.design
                    .region_by_name(self.pool.intern(name))
                    .ok_or_else(|| AttributeFault::NoSuchRegion {
                        name: name.to_owned(),
                    })
            })
            .transpose()
            .map_err(|fault| (RoutedAttribute::Region, fault))
    }

    fn routing(&self, net: &NetInfo) -> Result<FxHashMap<WireId, PipMap>, AttributeFault> {
        let Some(value) = self.attr(&net.attrs, RoutedAttribute::Routing) else {
            return Ok(FxHashMap::default());
        };
        let fields: Vec<&str> = string(value)?.split_terminator(';').collect();
        let (entries, remainder) = fields.as_chunks::<3>();
        if !remainder.is_empty() {
            return Err(AttributeFault::EntryCount {
                pieces: fields.len(),
            });
        }
        let mut wires = FxHashMap::default();
        for &[wire_name, pip_name, strength] in entries {
            let wire = self.chipdb.wire_by_name(wire_name)?;
            let pip = match pip_name {
                "" => None,
                pip_name => Some(self.chipdb.pip_by_name(pip_name)?),
            };
            if pip.is_some_and(|pip| self.chipdb.pip_dst_wire(pip) != wire) {
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

fn int32(value: &Property) -> Result<i32, AttributeFault> {
    let integer = |value: String| AttributeFault::Integer { value };
    let Property::BitVector(bits) = value else {
        return Err(integer(value.as_str()));
    };
    u32::from_str_radix(bits, 2)
        .map(|value| value as i32)
        .map_err(|_| integer(bits.clone()))
}

fn clock_period(value: &Property) -> Result<DelayT, AttributeFault> {
    match int32(value)? {
        period if period > 0 => Ok(period),
        value => Err(AttributeFault::ClockPeriod { value }),
    }
}

fn rects(value: &str) -> Result<Vec<Rect>, AttributeFault> {
    let malformed = || AttributeFault::Rects {
        value: value.to_owned(),
    };
    value
        .split_terminator(';')
        .map(|rect| {
            let corners: Vec<i32> = rect
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|_| malformed())?;
            match corners[..] {
                [x0, y0, x1, y1] => Ok(Rect::new(x0, y0, x1, y1)),
                _ => Err(malformed()),
            }
        })
        .collect()
}

fn text(value: &Property) -> Result<&str, AttributeFault> {
    match value {
        Property::String(text) => Ok(text),
        Property::Int(_) | Property::BitVector(_) => Err(AttributeFault::NotString),
    }
}

fn string(value: &Property) -> Result<&str, AttributeFault> {
    text(value).map(unpad)
}

fn unpad(text: &str) -> &str {
    let padding = text.trim_start_matches(['0', '1', 'x', 'z']);
    if !padding.is_empty() && padding.bytes().all(|byte| byte == b' ') {
        &text[..text.len() - 1]
    } else {
        text
    }
}

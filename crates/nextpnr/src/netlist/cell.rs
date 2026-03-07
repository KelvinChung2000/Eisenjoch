use crate::types::{BelId, IdString, PlaceStrength, PortType, Property};
use rustc_hash::FxHashMap;

use super::{CellId, FlatIndex, PortInfo, TimingIndex};

pub struct CellInfo {
    pub name: IdString,
    pub cell_type: IdString,
    pub ports: FxHashMap<IdString, PortInfo>,
    pub attrs: FxHashMap<IdString, Property>,
    pub params: FxHashMap<IdString, Property>,

    pub bel: Option<BelId>,
    pub bel_strength: PlaceStrength,

    pub cluster: Option<CellId>,
    pub region: Option<u32>,

    pub flat_index: Option<FlatIndex>,
    pub timing_index: Option<TimingIndex>,

    pub alive: bool,
}

impl CellInfo {
    pub fn new(name: IdString, cell_type: IdString) -> Self {
        Self {
            name,
            cell_type,
            ports: FxHashMap::default(),
            attrs: FxHashMap::default(),
            params: FxHashMap::default(),
            bel: None,
            bel_strength: PlaceStrength::None,
            cluster: None,
            region: None,
            flat_index: None,
            timing_index: None,
            alive: true,
        }
    }

    pub fn add_port(&mut self, name: IdString, port_type: PortType) -> &mut PortInfo {
        self.ports
            .entry(name)
            .or_insert_with(|| PortInfo::new(name, port_type))
    }

    pub fn port(&self, name: IdString) -> Option<&PortInfo> {
        self.ports.get(&name)
    }

    pub fn port_mut(&mut self, name: IdString) -> Option<&mut PortInfo> {
        self.ports.get_mut(&name)
    }
}

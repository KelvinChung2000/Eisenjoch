use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use rustc_hash::FxHashMap;

use super::{CellId, CellPin, FlatIndex, PortData, PortType, Property, TimingIndex};

pub struct CellInfo {
    pub name: IdString,
    pub cell_type: IdString,
    pub(crate) ports: FxHashMap<IdString, PortData>,
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

    pub fn add_port(&mut self, name: IdString, port_type: PortType) {
        self.ports
            .entry(name)
            .or_insert_with(|| PortData::new(port_type));
    }

    #[inline]
    pub fn has_port(&self, name: IdString) -> bool {
        self.ports.contains_key(&name)
    }

    #[inline]
    pub fn num_ports(&self) -> usize {
        self.ports.len()
    }

    #[inline]
    pub fn port(&self, name: IdString) -> Option<CellPin> {
        self.ports.contains_key(&name).then(|| CellPin::new(CellId::NONE, name))
    }

    #[inline]
    pub fn port_net(&self, name: IdString) -> Option<super::NetId> {
        self.port_data(name).and_then(|port| port.net())
    }

    #[inline]
    pub fn port_type(&self, name: IdString) -> Option<PortType> {
        self.port_data(name).map(|port| port.port_type())
    }

    #[inline]
    pub fn port_user_idx(&self, name: IdString) -> Option<u32> {
        self.port_data(name).and_then(|port| port.user_idx())
    }

    #[inline]
    pub fn port_budget(&self, name: IdString) -> Option<crate::timing::DelayT> {
        self.port_data(name).map(|port| port.budget())
    }

    #[inline]
    pub fn set_port_net(&mut self, name: IdString, net: Option<super::NetId>) {
        if let Some(port) = self.port_data_mut(name) {
            port.set_net(net);
        }
    }

    #[inline]
    pub fn set_port_user_idx(&mut self, name: IdString, user_idx: Option<u32>) {
        if let Some(port) = self.port_data_mut(name) {
            port.set_user_idx(user_idx);
        }
    }

    #[inline]
    pub fn set_port_budget(&mut self, name: IdString, budget: crate::timing::DelayT) {
        if let Some(port) = self.port_data_mut(name) {
            port.set_budget(budget);
        }
    }

    #[inline]
    pub(crate) fn port_data(&self, name: IdString) -> Option<&PortData> {
        self.ports.get(&name)
    }

    #[inline]
    pub(crate) fn port_data_mut(&mut self, name: IdString) -> Option<&mut PortData> {
        self.ports.get_mut(&name)
    }
}

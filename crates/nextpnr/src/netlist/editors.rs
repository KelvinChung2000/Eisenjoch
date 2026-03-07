use crate::types::{BelId, DelayT, IdString, PipId, PlaceStrength, PortType, Property, WireId};

use super::{CellId, CellInfo, CellPin, FlatIndex, NetId, NetInfo, PipMap, TimingIndex};

pub struct CellEditor<'a> {
    cell: &'a mut CellInfo,
}

impl<'a> CellEditor<'a> {
    pub(crate) fn new(cell: &'a mut CellInfo) -> Self {
        Self { cell }
    }

    pub fn set_bel(&mut self, bel: Option<BelId>, strength: PlaceStrength) -> &mut Self {
        self.cell.bel = bel;
        self.cell.bel_strength = strength;
        self
    }

    pub fn add_port(&mut self, name: IdString, port_type: PortType) -> &mut Self {
        self.cell.add_port(name, port_type);
        self
    }

    pub fn set_port_net(
        &mut self,
        port: IdString,
        net: Option<NetId>,
        user_idx: Option<u32>,
    ) -> &mut Self {
        if let Some(p) = self.cell.port_data_mut(port) {
            p.set_net(net);
            p.set_user_idx(user_idx);
        }
        self
    }

    pub fn set_port_budget(&mut self, port: IdString, budget: DelayT) -> &mut Self {
        if let Some(p) = self.cell.port_data_mut(port) {
            p.set_budget(budget);
        }
        self
    }

    pub fn rename_port(&mut self, old: IdString, new: IdString) -> &mut Self {
        if let Some(port_info) = self.cell.ports.remove(&old) {
            self.cell.ports.insert(new, port_info);
        }
        self
    }

    pub fn set_type(&mut self, cell_type: IdString) -> &mut Self {
        self.cell.cell_type = cell_type;
        self
    }

    pub fn set_attr(&mut self, key: IdString, value: Property) -> &mut Self {
        self.cell.attrs.insert(key, value);
        self
    }

    pub fn set_param(&mut self, key: IdString, value: Property) -> &mut Self {
        self.cell.params.insert(key, value);
        self
    }

    pub fn set_cluster(&mut self, root: Option<CellId>) -> &mut Self {
        self.cell.cluster = root;
        self
    }

    pub fn set_region(&mut self, region: Option<u32>) -> &mut Self {
        self.cell.region = region;
        self
    }

    pub fn set_flat_index(&mut self, idx: Option<FlatIndex>) -> &mut Self {
        self.cell.flat_index = idx;
        self
    }

    pub fn set_timing_index(&mut self, idx: Option<TimingIndex>) -> &mut Self {
        self.cell.timing_index = idx;
        self
    }

    pub fn mark_dead(&mut self) -> &mut Self {
        self.cell.alive = false;
        self
    }
}

pub struct NetEditor<'a> {
    net: &'a mut NetInfo,
}

impl<'a> NetEditor<'a> {
    pub(crate) fn new(net: &'a mut NetInfo) -> Self {
        Self { net }
    }

    pub fn set_driver_raw(&mut self, driver: CellPin) -> &mut Self {
        self.net.driver = driver;
        self
    }

    pub fn set_driver(&mut self, cell: CellId, port: IdString) -> &mut Self {
        self.net.driver = CellPin::new(cell, port);
        self
    }

    pub fn clear_driver(&mut self) -> &mut Self {
        self.net.driver = CellPin::INVALID;
        self
    }

    pub fn add_user(&mut self, cell: CellId, port: IdString) -> u32 {
        let idx = self.net.users.len() as u32;
        self.net.users.push(CellPin::new(cell, port));
        idx
    }

    pub fn add_user_raw(&mut self, user: CellPin) -> u32 {
        let idx = self.net.users.len() as u32;
        self.net.users.push(user);
        idx
    }

    pub fn disconnect_user(&mut self, user_idx: usize) -> &mut Self {
        if user_idx < self.net.users.len() {
            self.net.users[user_idx] = CellPin::INVALID;
        }
        self
    }

    pub fn add_wire(
        &mut self,
        wire: WireId,
        pip: Option<PipId>,
        strength: PlaceStrength,
    ) -> &mut Self {
        self.net.wires.insert(wire, PipMap { pip, strength });
        self
    }

    pub fn clear_wires(&mut self) -> &mut Self {
        self.net.wires.clear();
        self
    }

    pub fn set_name(&mut self, name: IdString) -> &mut Self {
        self.net.name = name;
        self
    }

    pub fn set_clock_constraint(&mut self, period_ps: DelayT) -> &mut Self {
        self.net.clock_constraint = period_ps;
        self
    }

    pub fn set_attr(&mut self, key: IdString, value: Property) -> &mut Self {
        self.net.attrs.insert(key, value);
        self
    }

    pub fn set_region(&mut self, region: Option<u32>) -> &mut Self {
        self.net.region = region;
        self
    }

    pub fn mark_dead(&mut self) -> &mut Self {
        self.net.alive = false;
        self
    }
}

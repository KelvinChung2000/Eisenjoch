use crate::types::{BelId, DelayT, IdString, PipId, PlaceStrength, PortType, Property, WireId};

use super::{CellId, CellInfo, FlatIndex, NetId, NetInfo, PipMap, PortInfo, PortRef, TimingIndex};

pub struct CellEditor<'a> {
    cell: &'a mut CellInfo,
}

impl<'a> CellEditor<'a> {
    pub(crate) fn new(cell: &'a mut CellInfo) -> Self {
        Self { cell }
    }

    pub fn set_bel(&mut self, bel: Option<BelId>, strength: PlaceStrength) {
        self.cell.bel = bel;
        self.cell.bel_strength = strength;
    }

    pub fn add_port(&mut self, name: IdString, port_type: PortType) -> &mut PortInfo {
        self.cell.add_port(name, port_type)
    }

    pub fn set_port_net(&mut self, port: IdString, net: Option<NetId>, user_idx: Option<u32>) {
        if let Some(p) = self.cell.port_mut(port) {
            p.net = net;
            p.user_idx = user_idx;
        }
    }

    pub fn rename_port(&mut self, old: IdString, new: IdString) {
        if let Some(mut port_info) = self.cell.ports.remove(&old) {
            port_info.name = new;
            self.cell.ports.insert(new, port_info);
        }
    }

    pub fn set_type(&mut self, cell_type: IdString) {
        self.cell.cell_type = cell_type;
    }

    pub fn set_attr(&mut self, key: IdString, value: Property) {
        self.cell.attrs.insert(key, value);
    }

    pub fn set_param(&mut self, key: IdString, value: Property) {
        self.cell.params.insert(key, value);
    }

    pub fn set_cluster(&mut self, root: Option<CellId>) {
        self.cell.cluster = root;
    }

    pub fn set_region(&mut self, region: Option<u32>) {
        self.cell.region = region;
    }

    pub fn set_flat_index(&mut self, idx: Option<FlatIndex>) {
        self.cell.flat_index = idx;
    }

    pub fn set_timing_index(&mut self, idx: Option<TimingIndex>) {
        self.cell.timing_index = idx;
    }

    pub fn mark_dead(&mut self) {
        self.cell.alive = false;
    }
}

pub struct NetEditor<'a> {
    net: &'a mut NetInfo,
}

impl<'a> NetEditor<'a> {
    pub(crate) fn new(net: &'a mut NetInfo) -> Self {
        Self { net }
    }

    pub fn set_driver_raw(&mut self, driver: PortRef) {
        self.net.driver = driver;
    }

    pub fn clear_driver(&mut self) {
        self.net.driver = PortRef::unconnected();
    }

    pub fn add_user(&mut self, cell: CellId, port: IdString) -> u32 {
        let idx = self.net.users.len() as u32;
        self.net.users.push(PortRef::connected(cell, port, 0));
        idx
    }

    pub fn add_user_raw(&mut self, user: PortRef) -> u32 {
        let idx = self.net.users.len() as u32;
        self.net.users.push(user);
        idx
    }

    pub fn disconnect_user(&mut self, user_idx: usize) {
        if user_idx < self.net.users.len() {
            self.net.users[user_idx] = PortRef::unconnected();
        }
    }

    pub fn add_wire(&mut self, wire: WireId, pip: Option<PipId>, strength: PlaceStrength) {
        self.net.wires.insert(wire, PipMap { pip, strength });
    }

    pub fn clear_wires(&mut self) {
        self.net.wires.clear();
    }

    pub fn set_name(&mut self, name: IdString) {
        self.net.name = name;
    }

    pub fn set_clock_constraint(&mut self, period_ps: DelayT) {
        self.net.clock_constraint = period_ps;
    }

    pub fn set_attr(&mut self, key: IdString, value: Property) {
        self.net.attrs.insert(key, value);
    }

    pub fn set_region(&mut self, region: Option<u32>) {
        self.net.region = region;
    }

    pub fn mark_dead(&mut self) {
        self.net.alive = false;
    }
}

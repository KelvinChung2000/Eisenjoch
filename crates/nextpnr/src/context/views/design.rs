use std::fmt;

use crate::chipdb::{BelId, WireId};
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellId, CellInfo, FlatIndex, NetId, NetInfo, PipMap, TimingIndex};
use crate::netlist::{PortType, Property};
use rustc_hash::FxHashMap;

use super::common::define_view;
use super::hardware::{Bel, Wire};
use super::pins::CellPinView;

define_view!(Cell, CellId);
define_view!(Net, NetId);

impl fmt::Display for Cell<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for Net<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl<'a> Net<'a> {
    #[inline]
    pub fn name(&self) -> &'a str {
        self.ctx.name_of(self.info().name)
    }

    #[inline]
    pub(crate) fn info(&self) -> &'a NetInfo {
        self.ctx.design.net(self.id)
    }

    #[inline]
    pub fn wire_ids(&self) -> impl Iterator<Item = WireId> + 'a {
        self.info().wires.keys().copied()
    }

    #[inline]
    pub fn wire_views(&self) -> impl Iterator<Item = Wire<'a>> + 'a {
        self.wire_ids().map(|wire| Wire::new(self.ctx, wire))
    }

    #[inline]
    pub fn name_id(&self) -> IdString { self.info().name }

    #[inline]
    pub fn driver(&self) -> Option<crate::netlist::CellPin> { self.info().driver() }

    #[inline]
    pub fn driver_view(&self) -> Option<CellPinView<'a>> {
        self.driver().map(|pin| pin.view(self.ctx))
    }

    #[inline]
    pub fn driver_cell_port(&self) -> Option<crate::netlist::CellPin> { self.driver() }

    #[inline]
    pub fn users(&self) -> &'a [crate::netlist::CellPin] { self.info().users() }

    #[inline]
    pub fn wires(&self) -> &'a FxHashMap<WireId, PipMap> { &self.info().wires }

    #[inline]
    pub fn is_alive(&self) -> bool { self.info().alive }

    #[inline]
    pub fn has_driver(&self) -> bool { self.info().has_driver() }

    #[inline]
    pub fn num_users(&self) -> usize { self.info().num_users() }

    #[inline]
    pub fn connected_users(&self) -> impl Iterator<Item = crate::netlist::CellPin> + 'a {
        self.info().users().iter().copied().filter(|u| u.is_valid())
    }

    #[inline]
    pub fn fanout(&self) -> usize {
        self.connected_users().count()
    }

    #[inline]
    pub fn clock_constraint(&self) -> crate::timing::DelayT { self.info().clock_constraint }

    #[inline]
    pub fn region(&self) -> Option<u32> { self.info().region }

    #[inline]
    pub fn attrs(&self) -> &'a FxHashMap<IdString, Property> { &self.info().attrs }
}

impl<'a> Cell<'a> {
    #[inline]
    pub fn name(&self) -> &'a str {
        self.ctx.name_of(self.info().name)
    }

    #[inline]
    pub fn cell_type(&self) -> &'a str {
        self.ctx.name_of(self.info().cell_type)
    }

    #[inline]
    pub(crate) fn info(&self) -> &'a CellInfo {
        self.ctx.design.cell(self.id)
    }

    #[inline]
    pub fn bel(&self) -> Option<Bel<'a>> {
        self.info().bel.map(|bel| Bel::new(self.ctx, bel))
    }

    #[inline]
    pub fn name_id(&self) -> IdString { self.info().name }

    #[inline]
    pub fn cell_type_id(&self) -> IdString { self.info().cell_type }

    #[inline]
    pub fn bel_id(&self) -> Option<BelId> { self.info().bel }

    #[inline]
    pub fn bel_strength(&self) -> crate::common::PlaceStrength { self.info().bel_strength }

    #[inline]
    pub fn is_alive(&self) -> bool { self.info().alive }

    #[inline]
    pub fn ports(&self) -> impl Iterator<Item = crate::netlist::CellPin> + '_ {
        self.info().ports.keys().copied().map(move |port| crate::netlist::CellPin::new(self.id, port))
    }

    #[inline]
    pub fn port(&self, name: IdString) -> Option<crate::netlist::CellPin> {
        self.info().port_data(name).map(|_| crate::netlist::CellPin::new(self.id, name))
    }

    #[inline]
    pub fn port_view(&self, name: IdString) -> Option<CellPinView<'a>> {
        self.port(name).map(|pin| pin.view(self.ctx))
    }

    #[inline]
    pub fn port_net(&self, name: IdString) -> Option<NetId> {
        self.info().port_data(name).and_then(|p| p.net())
    }

    #[inline]
    pub fn port_type(&self, name: IdString) -> Option<PortType> {
        self.info().port_data(name).map(|p| p.port_type())
    }

    #[inline]
    pub fn attrs(&self) -> &'a FxHashMap<IdString, Property> { &self.info().attrs }

    #[inline]
    pub fn params(&self) -> &'a FxHashMap<IdString, Property> { &self.info().params }

    #[inline]
    pub fn cluster(&self) -> Option<CellId> { self.info().cluster }

    #[inline]
    pub fn region(&self) -> Option<u32> { self.info().region }

    #[inline]
    pub fn flat_index(&self) -> Option<FlatIndex> { self.info().flat_index }

    #[inline]
    pub fn timing_index(&self) -> Option<TimingIndex> { self.info().timing_index }
}

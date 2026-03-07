use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use super::Context;
use crate::netlist::{
    CellId, CellInfo, CellPin, FlatIndex, NetId, NetInfo, PipMap, TimingIndex,
};
use crate::types::{
    BelId, DelayQuad, DelayT, IdString, Loc, PipId, PlaceStrength, PortType, Property, WireId,
};
use rustc_hash::FxHashMap;

pub struct IdStringView<'a> {
    ctx: &'a Context,
    id: IdString,
}

impl<'a> IdStringView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: IdString) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> IdString {
        self.id
    }

    #[inline]
    pub fn index(&self) -> i32 {
        self.id.index()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.ctx.name_of(self.id)
    }
}

#[derive(Clone, Copy)]
pub struct BelPin {
    bel: BelId,
    port: IdString,
}

impl BelPin {
    pub fn new(bel: BelId, port: IdString) -> Self {
        Self { bel, port }
    }

    #[inline]
    pub fn bel(&self) -> BelId {
        self.bel
    }

    #[inline]
    pub fn port(&self) -> IdString {
        self.port
    }
}

#[derive(Clone, Copy)]
pub struct BelPinView<'a> {
    ctx: &'a Context,
    pin: BelPin,
}

impl<'a> BelPinView<'a> {
    pub(crate) fn new(ctx: &'a Context, pin: BelPin) -> Self {
        Self { ctx, pin }
    }

    #[inline]
    pub fn id(&self) -> BelPin {
        self.pin
    }

    #[inline]
    pub fn bel(&self) -> Bel<'a> {
        Bel::new(self.ctx, self.pin.bel())
    }

    #[inline]
    pub fn port(&self) -> IdStringView<'a> {
        IdStringView::new(self.ctx, self.pin.port())
    }

    #[inline]
    pub fn wire(&self) -> Option<Wire<'a>> {
        self.ctx.bel_pin_wire(self.pin)
    }
}

#[derive(Clone, Copy)]
pub struct CellPinView<'a> {
    ctx: &'a Context,
    pin: CellPin,
}

impl<'a> CellPinView<'a> {
    pub(crate) fn new(ctx: &'a Context, pin: CellPin) -> Self {
        Self { ctx, pin }
    }

    #[inline]
    pub fn id(&self) -> CellPin {
        self.pin
    }

    #[inline]
    pub fn cell(&self) -> Cell<'a> {
        Cell::new(self.ctx, self.pin.cell)
    }

    #[inline]
    pub fn port(&self) -> IdStringView<'a> {
        IdStringView::new(self.ctx, self.pin.port)
    }

    #[inline]
    fn data(&self) -> &'a crate::netlist::PortData {
        self.ctx
            .design
            .cell(self.pin.cell)
            .port_data(self.pin.port)
            .expect("invalid CellPin")
    }

    #[inline]
    pub fn port_type(&self) -> PortType {
        self.data().port_type()
    }

    #[inline]
    pub fn net_id(&self) -> Option<NetId> {
        self.data().net()
    }

    #[inline]
    pub fn net(&self) -> Option<Net<'a>> {
        self.net_id().map(|net| Net::new(self.ctx, net))
    }

    #[inline]
    pub fn user_idx(&self) -> Option<u32> {
        self.data().user_idx()
    }

    #[inline]
    pub fn budget(&self) -> DelayT {
        self.data().budget()
    }

    #[inline]
    pub fn is_connected(&self) -> bool {
        self.data().is_connected()
    }
}

impl CellPin {
    #[inline]
    pub fn view<'a>(self, ctx: &'a Context) -> CellPinView<'a> {
        CellPinView::new(ctx, self)
    }
}

impl BelPin {
    #[inline]
    pub fn view<'a>(self, ctx: &'a Context) -> BelPinView<'a> {
        BelPinView::new(ctx, self)
    }
}

#[derive(Clone, Copy)]
pub struct TileView {
    ctx: *const Context,
    id: i32,
}

impl TileView {
    pub(crate) fn new(ctx: &Context, id: i32) -> Self {
        Self { ctx, id }
    }

    #[inline]
    fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }

    #[inline]
    pub fn id(&self) -> i32 {
        self.id
    }

    #[inline]
    pub fn x(&self) -> i32 {
        self.ctx().chipdb().tile_xy(self.id).0
    }

    #[inline]
    pub fn y(&self) -> i32 {
        self.ctx().chipdb().tile_xy(self.id).1
    }
}

macro_rules! define_view {
    ($name:ident, $id_type:ty) => {
        #[derive(Clone, Copy)]
        pub struct $name<'a> {
            ctx: &'a Context,
            id: $id_type,
        }

        impl<'a> $name<'a> {
            pub(crate) fn new(ctx: &'a Context, id: $id_type) -> Self {
                Self { ctx, id }
            }

            #[inline]
            pub fn id(&self) -> $id_type { self.id }
        }

        impl Deref for $name<'_> {
            type Target = $id_type;
            #[inline]
            fn deref(&self) -> &$id_type { &self.id }
        }

        impl PartialEq for $name<'_> {
            fn eq(&self, other: &Self) -> bool { self.id == other.id }
        }
        impl Eq for $name<'_> {}

        impl Hash for $name<'_> {
            fn hash<H: Hasher>(&self, state: &mut H) { self.id.hash(state); }
        }

        impl PartialEq<$id_type> for $name<'_> {
            fn eq(&self, other: &$id_type) -> bool { self.id == *other }
        }
        impl PartialEq<$name<'_>> for $id_type {
            fn eq(&self, other: &$name<'_>) -> bool { *self == other.id }
        }

        impl From<$name<'_>> for $id_type {
            fn from(v: $name<'_>) -> $id_type { v.id }
        }
        impl From<&$name<'_>> for $id_type {
            fn from(v: &$name<'_>) -> $id_type { v.id }
        }
    };
}

macro_rules! define_hardware_view {
    ($name:ident, $id_type:ty) => {
        define_view!($name, $id_type);

        impl fmt::Display for $name<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.id.fmt(f)
            }
        }
    };
}

define_hardware_view!(Bel, BelId);
define_hardware_view!(Wire, WireId);
define_hardware_view!(Pip, PipId);
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

impl<'a> Bel<'a> {
    #[inline]
    pub fn name(&self) -> &'a str {
        self.ctx.chipdb().bel_name(self.id)
    }

    #[inline]
    pub fn bel_type(&self) -> &'a str {
        self.ctx.chipdb().bel_type(self.id)
    }

    /// BEL bucket name (same as `bel_type` in himbaechel).
    #[inline]
    pub fn bucket(&self) -> &'a str {
        self.bel_type()
    }

    #[inline]
    pub fn loc(&self) -> Loc {
        self.ctx.chipdb().bel_loc(self.id)
    }

    #[inline]
    pub fn tile(&self) -> TileView {
        TileView::new(self.ctx, self.id.tile())
    }

    #[inline]
    pub fn is_available(&self) -> bool {
        self.ctx.bel_slot(self.id).is_some_and(Option::is_none)
    }

    #[inline]
    pub fn bound_cell(&self) -> Option<Cell<'a>> {
        self.ctx.bel_slot(self.id)
            .copied()
            .flatten()
            .map(|cell_idx| Cell::new(self.ctx, cell_idx))
    }

    #[inline]
    pub fn pin_wire(&self, port: IdString) -> Option<Wire<'a>> {
        self.ctx.bel_pin_wire(BelPin::new(self.id, port))
    }

    #[inline]
    pub fn is_valid_for_cell_type(&self, cell_type: IdString) -> bool {
        let bucket = self.ctx.chipdb().bel_type(self.id);
        let cell_type_str = self.ctx.name_of(cell_type);
        bucket == cell_type_str
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
    pub fn driver(&self) -> Option<CellPin> { self.info().driver() }

    #[inline]
    pub fn driver_view(&self) -> Option<CellPinView<'a>> {
        self.driver().map(|pin| pin.view(self.ctx))
    }

    #[inline]
    pub fn driver_cell_port(&self) -> Option<CellPin> { self.driver() }

    #[inline]
    pub fn users(&self) -> &'a [CellPin] { self.info().users() }

    #[inline]
    pub fn wires(&self) -> &'a FxHashMap<WireId, PipMap> { &self.info().wires }

    #[inline]
    pub fn is_alive(&self) -> bool { self.info().alive }

    #[inline]
    pub fn has_driver(&self) -> bool { self.info().has_driver() }

    #[inline]
    pub fn num_users(&self) -> usize { self.info().num_users() }

    #[inline]
    pub fn connected_users(&self) -> impl Iterator<Item = CellPin> + 'a {
        self.info().users().iter().copied().filter(|u| u.is_valid())
    }

    #[inline]
    pub fn fanout(&self) -> usize {
        self.connected_users().count()
    }

    #[inline]
    pub fn clock_constraint(&self) -> DelayT { self.info().clock_constraint }

    #[inline]
    pub fn region(&self) -> Option<u32> { self.info().region }

    #[inline]
    pub fn attrs(&self) -> &'a FxHashMap<IdString, Property> { &self.info().attrs }
}

impl<'a> Wire<'a> {
    #[inline]
    pub fn tile(&self) -> TileView {
        TileView::new(self.ctx, self.id.tile())
    }

    #[inline]
    pub fn bound_net(&self) -> Option<Net<'a>> {
        self.ctx
            .wire_slot(self.id)
            .and_then(|slot| slot.map(|(net_idx, _)| Net::new(self.ctx, net_idx)))
    }

    #[inline]
    pub fn strength(&self) -> Option<PlaceStrength> {
        self.ctx
            .wire_slot(self.id)
            .copied()
            .flatten()
            .map(|(_, strength)| strength)
    }

    #[inline]
    pub fn is_available(&self) -> bool {
        self.ctx.wire_slot(self.id).is_some_and(Option::is_none)
    }

    #[inline]
    pub fn delay(&self) -> DelayQuad {
        match self.ctx.speed_grade() {
            Some(sg) => self.ctx.chipdb().compute_wire_delay(sg, self.id),
            None => DelayQuad::default(),
        }
    }
}

impl<'a> Pip<'a> {
    #[inline]
    pub fn tile(&self) -> TileView {
        TileView::new(self.ctx, self.id.tile())
    }

    #[inline]
    pub fn src_wire(&self) -> Wire<'a> {
        Wire::new(self.ctx, self.ctx.chipdb().pip_src_wire(self.id))
    }

    #[inline]
    pub fn dst_wire(&self) -> Wire<'a> {
        Wire::new(self.ctx, self.ctx.chipdb().pip_dst_wire(self.id))
    }

    #[inline]
    pub fn bound_net(&self) -> Option<Net<'a>> {
        self.ctx
            .pip_slot(self.id)
            .copied()
            .flatten()
            .map(|(net_idx, _)| Net::new(self.ctx, net_idx))
    }

    #[inline]
    pub fn strength(&self) -> Option<PlaceStrength> {
        self.ctx
            .pip_slot(self.id)
            .copied()
            .flatten()
            .map(|(_, strength)| strength)
    }

    #[inline]
    pub fn is_available(&self) -> bool {
        self.ctx.pip_slot(self.id).is_some_and(Option::is_none)
    }

    #[inline]
    pub fn delay(&self) -> DelayQuad {
        match self.ctx.speed_grade() {
            Some(sg) => DelayQuad::uniform(self.ctx.chipdb().compute_pip_delay(sg, self.id)),
            None => DelayQuad::default(),
        }
    }
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
    pub fn bel_strength(&self) -> PlaceStrength { self.info().bel_strength }

    #[inline]
    pub fn is_alive(&self) -> bool { self.info().alive }

    #[inline]
    pub fn ports(&self) -> impl Iterator<Item = CellPin> + '_ {
        self.info()
            .ports
            .keys()
            .copied()
            .map(move |port| CellPin::new(self.id, port))
    }

    #[inline]
    pub fn port(&self, name: IdString) -> Option<CellPin> {
        self.info().port_data(name).map(|_| CellPin::new(self.id, name))
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


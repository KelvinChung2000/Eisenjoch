use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellPin, NetId};
use crate::netlist::PortData;
use crate::timing::DelayT;

use super::common::IdStringView;
use super::design::{Cell, Net};
use super::hardware::{Bel, Wire};

#[derive(Clone, Copy)]
pub struct BelPin {
    bel: crate::chipdb::BelId,
    port: IdString,
}

impl BelPin {
    pub fn new(bel: crate::chipdb::BelId, port: IdString) -> Self {
        Self { bel, port }
    }

    #[inline]
    pub fn bel(&self) -> crate::chipdb::BelId {
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
    fn data(&self) -> &'a PortData {
        self.ctx
            .design
            .cell(self.pin.cell)
            .port_data(self.pin.port)
            .expect("invalid CellPin")
    }

    #[inline]
    pub fn port_type(&self) -> crate::netlist::PortType {
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

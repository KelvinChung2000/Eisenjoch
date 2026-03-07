use crate::chipdb::{BelId, Loc, PipId, WireId};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::timing::DelayQuad;

use super::common::define_hardware_view;
use super::design::{Cell, Net};
use super::pins::BelPin;

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

define_hardware_view!(Bel, BelId);
define_hardware_view!(Wire, WireId);
define_hardware_view!(Pip, PipId);

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
        self.ctx
            .bel_slot(self.id)
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

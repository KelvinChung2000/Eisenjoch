use super::Context;
use crate::netlist::{CellIdx, CellInfo, Cluster, Design, HierarchicalCell, NetIdx, NetInfo};
use crate::types::{BelId, DelayQuad, IdString, IdStringPool, Loc, PipId, PlaceStrength, WireId};
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

pub struct BelView<'a> {
    ctx: &'a Context,
    id: BelId,
}

impl<'a> BelView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: BelId) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> BelId {
        self.id
    }

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
        self.ctx.is_bel_available(self.id)
    }

    #[inline]
    pub fn bound_cell(&self) -> Option<CellView<'a>> {
        self.ctx.bound_bel_cell(self.id)
    }

    #[inline]
    pub fn pin_wire(&self, port: IdString) -> Option<WireView<'a>> {
        self.ctx
            .bel_pin_wire(self.id, port)
            .map(|w| WireView::new(self.ctx, w))
    }

    #[inline]
    pub fn is_valid_for_cell_type(&self, cell_type: IdString) -> bool {
        self.ctx.is_valid_bel_for_cell(self.id, cell_type)
    }
}

pub struct NetView<'a> {
    ctx: &'a Context,
    id: NetIdx,
}

impl<'a> NetView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: NetIdx) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> NetIdx {
        self.id
    }

    #[inline]
    pub fn name(&self) -> &'a str {
        self.ctx.name_of(self.info().name)
    }

    #[inline]
    pub fn info(&self) -> &'a NetInfo {
        self.ctx.design().net(self.id)
    }

    #[inline]
    pub fn has_wire(&self, wire: &WireView<'_>) -> bool {
        self.info().wires.contains_key(&wire.id())
    }

    #[inline]
    pub fn wire_ids(&self) -> impl Iterator<Item = WireId> + 'a {
        self.info().wires.keys().copied()
    }

    #[inline]
    pub fn wire_views(&self) -> impl Iterator<Item = WireView<'a>> + 'a {
        self.wire_ids().map(|wire| WireView::new(self.ctx, wire))
    }
}

pub struct WireView<'a> {
    ctx: &'a Context,
    id: WireId,
}

impl<'a> WireView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: WireId) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> WireId {
        self.id
    }

    #[inline]
    pub fn tile(&self) -> TileView {
        TileView::new(self.ctx, self.id.tile())
    }

    #[inline]
    pub fn bound_net(&self) -> Option<NetView<'a>> {
        self.ctx
            .bound_wire_net_idx(self.id)
            .map(|net_idx| NetView::new(self.ctx, net_idx))
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
        self.ctx.is_wire_available(self.id)
    }

    #[inline]
    pub fn delay(&self) -> DelayQuad {
        self.ctx.wire_delay(self.id)
    }
}

pub struct PipView<'a> {
    ctx: &'a Context,
    id: PipId,
}

impl<'a> PipView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: PipId) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> PipId {
        self.id
    }

    #[inline]
    pub fn tile(&self) -> TileView {
        TileView::new(self.ctx, self.id.tile())
    }

    #[inline]
    pub fn src_wire(&self) -> WireView<'a> {
        WireView::new(self.ctx, self.ctx.chipdb().pip_src_wire(self.id))
    }

    #[inline]
    pub fn dst_wire(&self) -> WireView<'a> {
        WireView::new(self.ctx, self.ctx.chipdb().pip_dst_wire(self.id))
    }

    #[inline]
    pub fn bound_net(&self) -> Option<NetView<'a>> {
        self.ctx
            .pip_slot(self.id)
            .copied()
            .flatten()
            .map(|(net_idx, _)| NetView::new(self.ctx, net_idx))
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
        self.ctx.is_pip_available(self.id)
    }

    #[inline]
    pub fn delay(&self) -> DelayQuad {
        self.ctx.pip_delay(self.id)
    }
}

pub struct CellView<'a> {
    ctx: &'a Context,
    id: CellIdx,
}

impl<'a> CellView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: CellIdx) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> CellIdx {
        self.id
    }

    #[inline]
    pub fn name(&self) -> &'a str {
        self.ctx.name_of(self.info().name)
    }

    #[inline]
    pub fn cell_type(&self) -> &'a str {
        self.ctx.name_of(self.info().cell_type)
    }

    #[inline]
    pub fn info(&self) -> &'a CellInfo {
        self.ctx.design().cell(self.id)
    }

    #[inline]
    pub fn bel(&self) -> Option<BelView<'a>> {
        self.info().bel.map(|bel| BelView::new(self.ctx, bel))
    }
}

impl From<BelView<'_>> for BelId {
    fn from(v: BelView<'_>) -> BelId { v.id }
}

impl From<&BelView<'_>> for BelId {
    fn from(v: &BelView<'_>) -> BelId { v.id }
}

impl From<WireView<'_>> for WireId {
    fn from(v: WireView<'_>) -> WireId { v.id }
}

impl From<&WireView<'_>> for WireId {
    fn from(v: &WireView<'_>) -> WireId { v.id }
}

impl From<PipView<'_>> for PipId {
    fn from(v: PipView<'_>) -> PipId { v.id }
}

impl From<&PipView<'_>> for PipId {
    fn from(v: &PipView<'_>) -> PipId { v.id }
}

impl From<CellView<'_>> for CellIdx {
    fn from(v: CellView<'_>) -> CellIdx { v.id }
}

impl From<&CellView<'_>> for CellIdx {
    fn from(v: &CellView<'_>) -> CellIdx { v.id }
}

impl From<NetView<'_>> for NetIdx {
    fn from(v: NetView<'_>) -> NetIdx { v.id }
}

impl From<&NetView<'_>> for NetIdx {
    fn from(v: &NetView<'_>) -> NetIdx { v.id }
}

pub struct DesignView<'a> {
    design: &'a Design,
    pool: &'a IdStringPool,
}

impl<'a> DesignView<'a> {
    pub(crate) fn new(design: &'a Design, pool: &'a IdStringPool) -> Self {
        Self { design, pool }
    }

    #[inline]
    pub fn num_cells(&self) -> usize { self.design.num_cells() }

    #[inline]
    pub fn num_nets(&self) -> usize { self.design.num_nets() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.design.is_empty() }

    #[inline]
    pub fn cell_by_name(&self, name: IdString) -> Option<CellIdx> { self.design.cell_by_name(name) }

    #[inline]
    pub fn net_by_name(&self, name: IdString) -> Option<NetIdx> { self.design.net_by_name(name) }

    #[inline]
    pub fn iter_cell_indices(&self) -> impl Iterator<Item = CellIdx> + '_ {
        self.design.iter_cell_indices()
    }

    #[inline]
    pub fn iter_net_indices(&self) -> impl Iterator<Item = NetIdx> + '_ {
        self.design.iter_net_indices()
    }

    #[inline]
    pub fn cell_slots_len(&self) -> usize { self.design.cell_slots_len() }

    #[inline]
    pub fn net_slots_len(&self) -> usize { self.design.net_slots_len() }

    #[inline]
    pub fn cell_idx_at_slot(&self, slot: usize) -> Option<CellIdx> { self.design.cell_idx_at_slot(slot) }

    #[inline]
    pub fn net_idx_at_slot(&self, slot: usize) -> Option<NetIdx> { self.design.net_idx_at_slot(slot) }

    #[inline]
    pub fn top_module(&self) -> IdString { self.design.top_module }

    #[inline]
    pub fn hierarchy(&self) -> &'a FxHashMap<IdString, HierarchicalCell> { &self.design.hierarchy }

    #[inline]
    pub fn clusters(&self) -> &'a FxHashMap<CellIdx, Cluster> { &self.design.clusters }
}

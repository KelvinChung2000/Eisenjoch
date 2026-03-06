//! Central context struct and architecture API for the nextpnr-rust FPGA
//! place-and-route tool.
//!
//! The [`Context`] ties together the read-only chip database ([`ChipDb`]) with the
//! mutable design netlist ([`Design`]), and maintains the placement and routing
//! state maps that track which hardware resources (bels, wires, pips) are bound
//! to which design elements (cells, nets).
//!
//! All placer, router, and timing code operates through the `Context`.

mod rng;
mod storage;
mod utils;
mod views;

pub use rng::DeterministicRng;
pub use views::{
    BelPin, BelView, CellView, DesignView, IdStringView, NetView, PipView, TileView, WireView,
};

use crate::chipdb::ChipDb;
use crate::netlist::{CellEditor, CellIdx, Design, NetEditor, NetIdx};
use crate::read_packed;
use crate::types::{
    BelId, IdString, IdStringPool, PipId, PlaceStrength, Property, WireId,
};
use log::warn;
use rustc_hash::FxHashMap;
use storage::TileSlotMap;

pub type BelPinWireMap = FxHashMap<(BelId, IdString), WireId>;

/// The central context for the nextpnr place-and-route flow.
///
/// Owns the string pool, chip database, design netlist, and all placement/routing
/// state. Every operation that queries or modifies the hardware mapping goes
/// through this struct.
pub struct Context {
    /// String interning pool shared across the whole flow.
    id_pool: IdStringPool,
    /// Read-only chip database describing the FPGA hardware.
    chipdb: ChipDb,
    /// Mutable design netlist being placed and routed.
    design: Design,

    // -- Placement state --
    /// For each tile, occupancy of BEL slots by cell index.
    bel_to_cell: TileSlotMap<Option<CellIdx>>,
    // -- Routing state --
    /// For each tile, occupancy of wire slots by (net index, strength).
    wire_to_net: TileSlotMap<Option<(NetIdx, PlaceStrength)>>,
    /// For each tile, occupancy of pip slots by (net index, strength).
    pip_to_net: TileSlotMap<Option<(NetIdx, PlaceStrength)>>,

    // -- Caches (populated on demand) --
    /// Unique bel bucket names across the whole chip.
    bel_buckets_cache: Vec<IdString>,
    /// For each bucket, the list of all BelIds belonging to it.
    bucket_bels_cache: FxHashMap<IdString, Vec<BelId>>,

    // -- Settings and flags --
    /// Arbitrary key-value settings (e.g. from command-line options).
    settings: FxHashMap<IdString, Property>,
    /// Deterministic RNG for reproducible results.
    rng: DeterministicRng,
    /// Enable verbose output.
    verbose: bool,
    /// Enable debug output.
    debug: bool,
    /// Force operations even when validity checks fail.
    force: bool,
}

impl Context {
    #[inline]
    fn bel_slot(&self, bel: BelId) -> Option<&Option<CellIdx>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get(tile, index)
    }

    #[inline]
    fn bel_slot_mut(&mut self, bel: BelId) -> Option<&mut Option<CellIdx>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get_mut(tile, index)
    }

    #[inline]
    fn wire_slot(&self, wire: WireId) -> Option<&Option<(NetIdx, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get(tile, index)
    }

    #[inline]
    fn wire_slot_mut(&mut self, wire: WireId) -> Option<&mut Option<(NetIdx, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get_mut(tile, index)
    }

    #[inline]
    fn pip_slot(&self, pip: PipId) -> Option<&Option<(NetIdx, PlaceStrength)>> {
        let tile = usize::try_from(pip.tile()).ok()?;
        let index = usize::try_from(pip.index()).ok()?;
        self.pip_to_net.get(tile, index)
    }

    #[inline]
    fn pip_slot_mut(&mut self, pip: PipId) -> Option<&mut Option<(NetIdx, PlaceStrength)>> {
        let tile = usize::try_from(pip.tile()).ok()?;
        let index = usize::try_from(pip.index()).ok()?;
        self.pip_to_net.get_mut(tile, index)
    }

    /// Create a new context from a chip database.
    ///
    /// The design starts empty; cells and nets should be loaded by the frontend
    /// before placement and routing.
    pub fn new(chipdb: ChipDb) -> Self {
        let mut bel_lengths = Vec::new();
        let mut wire_lengths = Vec::new();
        let mut pip_lengths = Vec::new();
        for tile in 0..chipdb.num_tiles() {
            let tt = chipdb.tile_type(tile);
            bel_lengths.push(tt.bels.get().len());
            wire_lengths.push(tt.wires.get().len());
            pip_lengths.push(tt.pips.get().len());
        }

        let bel_to_cell = TileSlotMap::with_fill(&bel_lengths, None);
        let wire_to_net = TileSlotMap::with_fill(&wire_lengths, None);
        let pip_to_net = TileSlotMap::with_fill(&pip_lengths, None);

        Self {
            id_pool: IdStringPool::new(),
            chipdb,
            design: Design::new(),
            bel_to_cell,
            wire_to_net,
            pip_to_net,
            bel_buckets_cache: Vec::new(),
            bucket_bels_cache: FxHashMap::default(),
            settings: FxHashMap::default(),
            rng: DeterministicRng::new(1),
            verbose: false,
            debug: false,
            force: false,
        }
    }

    // =====================================================================
    // String interning helpers
    // =====================================================================

    /// Intern a string, returning its IdString handle.
    #[inline]
    pub fn id(&self, s: &str) -> IdString {
        self.id_pool.intern(s)
    }

    #[inline]
    pub fn id_pool(&self) -> &IdStringPool {
        &self.id_pool
    }

    /// Look up the string for an IdString handle.
    ///
    /// Returns `"<unknown>"` if the index is out of range.
    #[inline]
    pub fn name_of(&self, id: IdString) -> &str {
        self.id_pool.lookup(id).unwrap_or("<unknown>")
    }

    #[inline]
    pub fn chipdb(&self) -> &ChipDb {
        &self.chipdb
    }

    #[inline]
    pub(crate) fn design(&self) -> &Design {
        &self.design
    }

    /// Split borrow: returns mutable design + immutable chipdb + immutable id pool.
    pub(crate) fn packer_parts(&mut self) -> (&mut Design, &ChipDb, &IdStringPool) {
        (&mut self.design, &self.chipdb, &self.id_pool)
    }

    /// Read-only design view through DesignView proxy.
    #[inline]
    pub fn design_view(&self) -> DesignView<'_> {
        DesignView::new(&self.design, &self.id_pool)
    }

    /// Get a CellEditor for mutating a cell.
    #[inline]
    pub fn cell_edit(&mut self, idx: CellIdx) -> CellEditor<'_> {
        self.design.cell_edit(idx)
    }

    /// Get a NetEditor for mutating a net.
    #[inline]
    pub fn net_edit(&mut self, idx: NetIdx) -> NetEditor<'_> {
        self.design.net_edit(idx)
    }

    /// Add a cell to the design.
    #[inline]
    pub fn add_cell(&mut self, name: IdString, cell_type: IdString) -> CellIdx {
        self.design.add_cell(name, cell_type)
    }

    /// Add a net to the design.
    #[inline]
    pub fn add_net(&mut self, name: IdString) -> NetIdx {
        self.design.add_net(name)
    }

    /// Remove a cell from the design by name.
    #[inline]
    pub fn remove_cell(&mut self, name: IdString) {
        self.design.remove_cell(name)
    }

    /// Remove a net from the design by name.
    #[inline]
    pub fn remove_net(&mut self, name: IdString) {
        self.design.remove_net(name)
    }

    /// Rename a net.
    #[inline]
    pub fn rename_net(&mut self, net_idx: NetIdx, new_name: IdString) {
        self.design.rename_net(net_idx, new_name)
    }

    /// Replace the entire design (used by frontend loading).
    #[inline]
    pub fn set_design(&mut self, design: Design) {
        self.design = design;
    }

    /// Set the top module name.
    #[inline]
    pub fn set_top_module(&mut self, name: IdString) {
        self.design.top_module = name;
    }

    /// Run timing analysis on the current design.
    #[inline]
    pub fn analyse_timing(&self, timing: &mut crate::timing::TimingAnalyser) {
        timing.analyse(&self.design, &self.id_pool);
    }

    /// Mutable access to clusters map (for packer).
    #[inline]
    pub(crate) fn clusters_mut(&mut self) -> &mut FxHashMap<CellIdx, crate::netlist::Cluster> {
        &mut self.design.clusters
    }

    #[inline]
    pub fn rng(&self) -> &DeterministicRng {
        &self.rng
    }

    #[inline]
    pub fn rng_mut(&mut self) -> &mut DeterministicRng {
        &mut self.rng
    }

    #[inline]
    pub fn reseed_rng(&mut self, seed: u64) {
        self.rng = DeterministicRng::new(seed);
    }

    #[inline]
    pub fn settings(&self) -> &FxHashMap<IdString, Property> {
        &self.settings
    }

    #[inline]
    pub fn settings_mut(&mut self) -> &mut FxHashMap<IdString, Property> {
        &mut self.settings
    }

    #[inline]
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    #[inline]
    pub fn debug(&self) -> bool {
        self.debug
    }

    #[inline]
    pub fn force(&self) -> bool {
        self.force
    }

    /// Create a lazy read-only view over an interned string handle.
    #[inline]
    pub fn id_ref(&self, id: IdString) -> IdStringView<'_> {
        IdStringView::new(self, id)
    }

    // =====================================================================
    // Grid dimensions
    // =====================================================================

    /// Grid width in tiles.
    #[inline]
    pub fn width(&self) -> i32 {
        self.chipdb.width()
    }

    /// Grid height in tiles.
    #[inline]
    pub fn height(&self) -> i32 {
        self.chipdb.height()
    }

    // =====================================================================
    // Property-style object views
    // =====================================================================

    #[inline]
    pub fn bel(&self, bel: BelId) -> BelView<'_> {
        BelView::new(self, bel)
    }

    #[inline]
    pub fn bels(&self) -> impl Iterator<Item = BelView<'_>> {
        self.chipdb.bels().map(|bel| self.bel(bel))
    }

    #[inline]
    pub fn wire(&self, wire: WireId) -> WireView<'_> {
        WireView::new(self, wire)
    }

    #[inline]
    pub fn wires(&self) -> impl Iterator<Item = WireView<'_>> + '_ {
        self.chipdb.wires().map(|wire| self.wire(wire))
    }

    #[inline]
    pub fn pip(&self, pip: PipId) -> PipView<'_> {
        PipView::new(self, pip)
    }

    #[inline]
    pub fn pips(&self) -> impl Iterator<Item = PipView<'_>> + '_ {
        self.chipdb.pips().map(|pip| self.pip(pip))
    }

    #[inline]
    pub fn net(&self, net_idx: NetIdx) -> NetView<'_> {
        NetView::new(self, net_idx)
    }

    #[inline]
    pub fn nets(&self) -> impl Iterator<Item = NetView<'_>> {
        self.design
            .iter_net_indices()
            .map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn net_by_name(&self, net_name: IdString) -> Option<NetView<'_>> {
        self.design
            .net_by_name(net_name)
            .map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn cell(&self, cell_idx: CellIdx) -> CellView<'_> {
        CellView::new(self, cell_idx)
    }

    #[inline]
    pub fn cells(&self) -> impl Iterator<Item = CellView<'_>> {
        self.design
            .iter_cell_indices()
            .map(|cell_idx| self.cell(cell_idx))
    }

    #[inline]
    pub fn cell_by_name(&self, cell_name: IdString) -> Option<CellView<'_>> {
        self.design
            .cell_by_name(cell_name)
            .map(|cell_idx| self.cell(cell_idx))
    }

    // =====================================================================
    // BEL operations
    // =====================================================================

    /// Bind a cell to a BEL.
    ///
    /// Updates both the `bel_to_cell` map and the cell's own `bel` / `bel_strength`
    /// fields. Returns `true` if the binding succeeded, `false` if the BEL was
    /// already occupied.
    pub fn bind_bel(&mut self, bel: impl Into<BelId>, cell_idx: impl Into<CellIdx>, strength: PlaceStrength) -> bool {
        let bel = bel.into();
        let cell_idx = cell_idx.into();
        if self.bel_slot(bel).and_then(|slot| *slot).is_some() {
            warn!("bind_bel: bel {} already occupied", bel);
            return false;
        }
        let Some(slot) = self.bel_slot_mut(bel) else {
            warn!("bind_bel: bel {} out of range", bel);
            return false;
        };

        *slot = Some(cell_idx);

        let cell = self.design.cell_mut(cell_idx);
        cell.bel = Some(bel);
        cell.bel_strength = strength;

        true
    }

    /// Unbind a cell from its BEL.
    ///
    /// Clears both the `bel_to_cell` map entry and the cell's placement fields.
    pub fn unbind_bel(&mut self, bel: impl Into<BelId>) {
        let bel = bel.into();
        if let Some(slot) = self.bel_slot_mut(bel) {
            if let Some(cell_idx) = slot.take() {
                let cell = self.design.cell_mut(cell_idx);
                cell.bel = None;
                cell.bel_strength = PlaceStrength::None;
            }
        }
    }

    /// Check if a BEL is available (not bound to any cell).
    #[inline]
    pub(crate) fn is_bel_available(&self, bel: BelId) -> bool {
        self.bel_slot(bel).is_some_and(Option::is_none)
    }

    /// View of the cell bound to a BEL, or `None` if unoccupied.
    #[inline]
    pub(crate) fn bound_bel_cell(&self, bel: BelId) -> Option<CellView<'_>> {
        self.bound_bel_cell_idx(bel)
            .map(|cell_idx| self.cell(cell_idx))
    }

    /// Cell index bound to a BEL, or `None` if unoccupied.
    #[inline]
    pub(crate) fn bound_bel_cell_idx(&self, bel: BelId) -> Option<CellIdx> {
        self.bel_slot(bel).copied().flatten()
    }

    /// All BELs belonging to a given bucket.
    ///
    /// The bucket caches must be populated first via [`populate_bel_buckets`].
    /// Returns an empty slice if the bucket is unknown or caches are not populated.
    pub fn bels_for_bucket(&self, bucket: &str) -> impl Iterator<Item = BelView<'_>> {
        self.bel_ids_for_bucket(bucket)
            .iter()
            .copied()
            .map(|bel| self.bel(bel))
    }

    pub(crate) fn bel_ids_for_bucket(&self, bucket: &str) -> &[BelId] {
        let id = self.id_pool.intern(bucket);
        self.bucket_bels_cache
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Find the wire connected to a specific BEL pin.
    pub(crate) fn bel_pin_wire(&self, bel: BelId, port: IdString) -> Option<WireId> {
        let port_name = self.name_of(port);

        let bel_info = self.chipdb.bel_info(bel);
        for pin in bel_info.pins.get() {
            let name_id: i32 = unsafe { read_packed!(*pin, name) };
            let pin_name = self.chipdb.constid_str(name_id).unwrap_or("");
            if pin_name == port_name {
                let wire: i32 = unsafe { read_packed!(*pin, wire) };
                return Some(WireId::new(bel.tile(), wire));
            }
        }
        None
    }

    #[inline]
    pub(crate) fn bel_pin_wire_map(&self) -> BelPinWireMap {
        let mut map = BelPinWireMap::default();
        for bel in self.bels() {
            let bel_id = bel.id();
            let bel_info = self.chipdb.bel_info(bel_id);
            for pin in bel_info.pins.get() {
                let name_id: i32 = unsafe { read_packed!(*pin, name) };
                let Some(pin_name) = self.chipdb.constid_str(name_id) else {
                    continue;
                };
                let wire: i32 = unsafe { read_packed!(*pin, wire) };
                let port_id = self.id(pin_name);
                map.insert((bel_id, port_id), WireId::new(bel_id.tile(), wire));
            }
        }
        map
    }

    // =====================================================================
    // Wire operations
    // =====================================================================

    /// Bind a wire to a net.
    pub fn bind_wire(&mut self, wire: impl Into<WireId>, net_idx: impl Into<NetIdx>, strength: PlaceStrength) {
        let wire = wire.into();
        let net_idx = net_idx.into();
        if let Some(slot) = self.wire_slot_mut(wire) {
            *slot = Some((net_idx, strength));
        }
    }

    /// Unbind a wire.
    pub fn unbind_wire(&mut self, wire: impl Into<WireId>) {
        let wire = wire.into();
        if let Some(slot) = self.wire_slot_mut(wire) {
            *slot = None;
        }
    }

    /// Check if a wire is available (not bound to any net).
    #[inline]
    pub(crate) fn is_wire_available(&self, wire: WireId) -> bool {
        self.wire_slot(wire).is_some_and(Option::is_none)
    }

    /// Net index bound to a wire, or `None` if unbound.
    #[inline]
    pub(crate) fn bound_wire_net_idx(&self, wire: WireId) -> Option<NetIdx> {
        self.wire_slot(wire)
            .and_then(|slot| slot.map(|(net_idx, _)| net_idx))
    }

    // =====================================================================
    // PIP operations
    // =====================================================================

    /// Bind a PIP to a net.
    pub fn bind_pip(&mut self, pip: impl Into<PipId>, net_idx: impl Into<NetIdx>, strength: PlaceStrength) {
        let pip = pip.into();
        let net_idx = net_idx.into();
        if let Some(slot) = self.pip_slot_mut(pip) {
            *slot = Some((net_idx, strength));
        }
    }

    /// Unbind a PIP.
    pub fn unbind_pip(&mut self, pip: impl Into<PipId>) {
        let pip = pip.into();
        if let Some(slot) = self.pip_slot_mut(pip) {
            *slot = None;
        }
    }

    /// Check if a PIP is available (not bound to any net).
    #[inline]
    pub(crate) fn is_pip_available(&self, pip: PipId) -> bool {
        self.pip_slot(pip).is_some_and(Option::is_none)
    }

    /// Source wire of a PIP.
    #[inline]
    pub(crate) fn pip_src_wire(&self, pip: PipId) -> WireId {
        self.chipdb.pip_src_wire(pip)
    }

    /// Destination wire of a PIP.
    #[inline]
    pub(crate) fn pip_dst_wire(&self, pip: PipId) -> WireId {
        self.chipdb.pip_dst_wire(pip)
    }

    // =====================================================================
    // Placement validity
    // =====================================================================

    /// Check if a cell type is valid for placement at a given BEL.
    ///
    /// Uses database-driven matching: the cell's type (as an IdString) is
    /// compared against the BEL's bucket. The cell type name must match the
    /// BEL bucket string for the placement to be valid.
    pub(crate) fn is_valid_bel_for_cell(&self, bel: BelId, cell_type: IdString) -> bool {
        let bucket = self.chipdb.bel_type(bel);
        let cell_type_str = self.name_of(cell_type);
        bucket == cell_type_str
    }

    // =====================================================================
    // BEL bucket operations
    // =====================================================================

    /// Populate the bel bucket caches by scanning all BELs in the chip database.
    ///
    /// This builds two caches:
    /// - `bel_buckets_cache`: a sorted list of unique bucket IdStrings
    /// - `bucket_bels_cache`: a map from bucket IdString to the list of BelIds
    ///
    /// This should be called once after loading the chip database and before
    /// any placement operations that need bucket information.
    pub fn populate_bel_buckets(&mut self) {
        self.bucket_bels_cache.clear();
        self.bel_buckets_cache.clear();

        for bel in self.chipdb.bels() {
            let bucket_str = self.chipdb.bel_type(bel);
            let bucket_id = self.id_pool.intern(bucket_str);
            self.bucket_bels_cache
                .entry(bucket_id)
                .or_default()
                .push(bel);
        }

        // Collect unique bucket IDs in a deterministic order.
        let mut buckets: Vec<IdString> = self.bucket_bels_cache.keys().copied().collect();
        buckets.sort_by_key(|id| id.index());
        self.bel_buckets_cache = buckets;
    }

    /// Get all unique bel bucket names.
    ///
    /// Returns an empty slice if [`populate_bel_buckets`] has not been called.
    #[inline]
    pub fn bel_buckets(&self) -> &[IdString] {
        &self.bel_buckets_cache
    }

}

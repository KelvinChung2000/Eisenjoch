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
pub use views::{Bel, BelPin, Cell, IdStringView, Net, Pip, TileView, Wire};

use crate::chipdb::ChipDb;
use crate::netlist::{CellId, Design, NetId};
use crate::read_packed;
use crate::types::{BelId, IdString, IdStringPool, PipId, PlaceStrength, Property, WireId};
use log::warn;
use rustc_hash::FxHashMap;
use storage::TileSlotMap;

/// The central context for the nextpnr place-and-route flow.
///
/// Owns the string pool, chip database, design netlist, and all placement/routing
/// state. Every operation that queries or modifies the hardware mapping goes
/// through this struct.
pub struct Context {
    /// String interning pool shared across the whole flow.
    pub id_pool: IdStringPool,
    /// Read-only chip database describing the FPGA hardware.
    chipdb: ChipDb,
    /// Mutable design netlist being placed and routed.
    pub design: Design,

    // -- Placement state --
    /// For each tile, occupancy of BEL slots by cell index.
    bel_to_cell: TileSlotMap<Option<CellId>>,
    // -- Routing state --
    /// For each tile, occupancy of wire slots by (net index, strength).
    wire_to_net: TileSlotMap<Option<(NetId, PlaceStrength)>>,
    /// For each tile, occupancy of pip slots by (net index, strength).
    pip_to_net: TileSlotMap<Option<(NetId, PlaceStrength)>>,

    // -- Caches (populated on demand) --
    /// For each bucket (bel type), the list of all BelIds belonging to it.
    bucket_bels: FxHashMap<IdString, Vec<BelId>>,

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
    pub(crate) fn bel_slot(&self, bel: BelId) -> Option<&Option<CellId>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get(tile, index)
    }

    #[inline]
    pub(crate) fn bel_slot_mut(&mut self, bel: BelId) -> Option<&mut Option<CellId>> {
        let tile = usize::try_from(bel.tile()).ok()?;
        let index = usize::try_from(bel.index()).ok()?;
        self.bel_to_cell.get_mut(tile, index)
    }

    #[inline]
    pub(crate) fn wire_slot(&self, wire: WireId) -> Option<&Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get(tile, index)
    }

    #[inline]
    pub(crate) fn wire_slot_mut(
        &mut self,
        wire: WireId,
    ) -> Option<&mut Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(wire.tile()).ok()?;
        let index = usize::try_from(wire.index()).ok()?;
        self.wire_to_net.get_mut(tile, index)
    }

    #[inline]
    pub(crate) fn pip_slot(&self, pip: PipId) -> Option<&Option<(NetId, PlaceStrength)>> {
        let tile = usize::try_from(pip.tile()).ok()?;
        let index = usize::try_from(pip.index()).ok()?;
        self.pip_to_net.get(tile, index)
    }

    #[inline]
    pub(crate) fn pip_slot_mut(
        &mut self,
        pip: PipId,
    ) -> Option<&mut Option<(NetId, PlaceStrength)>> {
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
            bucket_bels: FxHashMap::default(),
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

    /// Split borrow: returns mutable design + immutable chipdb + immutable id pool.
    ///
    /// Needed when callers require `&mut Design` alongside `&ChipDb`/`&IdStringPool`,
    /// which can't be done through the pub fields + `chipdb()` accessor due to
    /// borrow-checker limitations.
    pub fn packer_parts(&mut self) -> (&mut Design, &ChipDb, &IdStringPool) {
        (&mut self.design, &self.chipdb, &self.id_pool)
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
    // Property-style object views
    // =====================================================================

    #[inline]
    pub fn bel(&self, bel: BelId) -> Bel<'_> {
        Bel::new(self, bel)
    }

    #[inline]
    pub fn bels(&self) -> impl Iterator<Item = Bel<'_>> {
        self.chipdb.bels().map(|bel| self.bel(bel))
    }

    #[inline]
    pub fn wire(&self, wire: WireId) -> Wire<'_> {
        Wire::new(self, wire)
    }

    #[inline]
    pub fn wires(&self) -> impl Iterator<Item = Wire<'_>> + '_ {
        self.chipdb.wires().map(|wire| self.wire(wire))
    }

    #[inline]
    pub fn pip(&self, pip: PipId) -> Pip<'_> {
        Pip::new(self, pip)
    }

    #[inline]
    pub fn pips(&self) -> impl Iterator<Item = Pip<'_>> + '_ {
        self.chipdb.pips().map(|pip| self.pip(pip))
    }

    #[inline]
    pub fn net(&self, net_idx: NetId) -> Net<'_> {
        Net::new(self, net_idx)
    }

    #[inline]
    pub fn nets(&self) -> impl Iterator<Item = Net<'_>> {
        self.design
            .iter_net_indices()
            .map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn net_by_name(&self, net_name: IdString) -> Option<Net<'_>> {
        self.design
            .net_by_name(net_name)
            .map(|net_idx| self.net(net_idx))
    }

    #[inline]
    pub fn cell(&self, cell_idx: CellId) -> Cell<'_> {
        Cell::new(self, cell_idx)
    }

    #[inline]
    pub fn cells(&self) -> impl Iterator<Item = Cell<'_>> {
        self.design
            .iter_cell_indices()
            .map(|cell_idx| self.cell(cell_idx))
    }

    #[inline]
    pub fn cell_by_name(&self, cell_name: IdString) -> Option<Cell<'_>> {
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
    pub fn bind_bel(
        &mut self,
        bel: impl Into<BelId>,
        cell_idx: impl Into<CellId>,
        strength: PlaceStrength,
    ) -> bool {
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

    /// All BELs belonging to a given bucket.
    ///
    /// The cache must be populated first via [`populate_bel_buckets`].
    pub fn bels_for_bucket(&self, bucket: &str) -> impl Iterator<Item = Bel<'_>> {
        let id = self.id_pool.intern(bucket);
        self.bucket_bels
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(|bel| self.bel(bel))
    }

    /// Find the wire connected to a specific BEL pin.
    pub fn bel_pin_wire(&self, bp: BelPin) -> Option<Wire<'_>> {
        let port_name = self.name_of(bp.port());
        let bel_info = self.chipdb.bel_info(bp.bel());
        for pin in bel_info.pins.get() {
            let name_id: i32 = unsafe { read_packed!(*pin, name) };
            let pin_name = self.chipdb.constid_str(name_id).unwrap_or("");
            if pin_name == port_name {
                let wire: i32 = unsafe { read_packed!(*pin, wire) };
                return Some(Wire::new(self, WireId::new(bp.bel().tile(), wire)));
            }
        }
        None
    }

    // =====================================================================
    // Wire operations
    // =====================================================================

    /// Bind a wire to a net.
    pub fn bind_wire(
        &mut self,
        wire: impl Into<WireId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
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

    // =====================================================================
    // PIP operations
    // =====================================================================

    /// Bind a PIP to a net.
    pub fn bind_pip(
        &mut self,
        pip: impl Into<PipId>,
        net_idx: impl Into<NetId>,
        strength: PlaceStrength,
    ) {
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

    // =====================================================================
    // BEL bucket operations
    // =====================================================================

    /// Populate the bel bucket cache by scanning all BELs in the chip database.
    ///
    /// Should be called once before any placement operations that need bucket information.
    pub fn populate_bel_buckets(&mut self) {
        self.bucket_bels.clear();
        for bel in self.chipdb.bels() {
            let bucket_id = self.id_pool.intern(self.chipdb.bel_type(bel));
            self.bucket_bels.entry(bucket_id).or_default().push(bel);
        }
    }

    /// Get all unique bel bucket names (sorted by IdString index for determinism).
    pub fn bel_buckets(&self) -> Vec<IdString> {
        let mut buckets: Vec<IdString> = self.bucket_bels.keys().copied().collect();
        buckets.sort_by_key(|id| id.index());
        buckets
    }
}

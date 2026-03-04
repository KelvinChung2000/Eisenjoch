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

pub use rng::DeterministicRng;

use log::warn;
use npnr_chipdb::ChipDb;
use npnr_netlist::Design;
use npnr_types::{
    BelId, DelayQuad, DelayT, IdString, IdStringPool, Loc, PipId, PlaceStrength, Property, WireId,
};
use rustc_hash::FxHashMap;

/// Delay scaling factor: picoseconds per Manhattan grid unit.
/// Used for delay estimation when detailed routing data is unavailable.
const DELAY_SCALE: i32 = 100;

/// The central context for the nextpnr place-and-route flow.
///
/// Owns the string pool, chip database, design netlist, and all placement/routing
/// state. Every operation that queries or modifies the hardware mapping goes
/// through this struct.
pub struct Context {
    /// String interning pool shared across the whole flow.
    pub id_pool: IdStringPool,
    /// Read-only chip database describing the FPGA hardware.
    pub chipdb: ChipDb,
    /// Mutable design netlist being placed and routed.
    pub design: Design,

    // -- Placement state --
    /// Maps each occupied BEL to the name of the cell placed on it.
    pub bel_to_cell: FxHashMap<BelId, IdString>,
    // -- Routing state --
    /// Maps each bound wire to (net name, strength).
    pub wire_to_net: FxHashMap<WireId, (IdString, PlaceStrength)>,
    /// Maps each bound pip to (net name, strength).
    pub pip_to_net: FxHashMap<PipId, (IdString, PlaceStrength)>,

    // -- Caches (populated on demand) --
    /// Unique bel bucket names across the whole chip.
    pub bel_buckets_cache: Vec<IdString>,
    /// For each bucket, the list of all BelIds belonging to it.
    pub bucket_bels_cache: FxHashMap<IdString, Vec<BelId>>,

    // -- Settings and flags --
    /// Arbitrary key-value settings (e.g. from command-line options).
    pub settings: FxHashMap<IdString, Property>,
    /// Deterministic RNG for reproducible results.
    pub rng: DeterministicRng,
    /// Enable verbose output.
    pub verbose: bool,
    /// Enable debug output.
    pub debug: bool,
    /// Force operations even when validity checks fail.
    pub force: bool,
}

impl Context {
    /// Create a new context from a chip database.
    ///
    /// The design starts empty; cells and nets should be loaded by the frontend
    /// before placement and routing.
    pub fn new(chipdb: ChipDb) -> Self {
        Self {
            id_pool: IdStringPool::new(),
            chipdb,
            design: Design::new(),
            bel_to_cell: FxHashMap::default(),
            wire_to_net: FxHashMap::default(),
            pip_to_net: FxHashMap::default(),
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

    /// Look up the string for an IdString handle.
    ///
    /// Returns `"<unknown>"` if the index is out of range.
    #[inline]
    pub fn name_of(&self, id: IdString) -> String {
        self.id_pool
            .lookup(id)
            .unwrap_or_else(|| "<unknown>".to_owned())
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
    // BEL operations
    // =====================================================================

    /// Bind a cell to a BEL.
    ///
    /// Updates both the `bel_to_cell` map and the cell's own `bel` / `bel_strength`
    /// fields. Returns `true` if the binding succeeded, `false` if the BEL was
    /// already occupied.
    pub fn bind_bel(
        &mut self,
        bel: BelId,
        cell_name: IdString,
        strength: PlaceStrength,
    ) -> bool {
        if self.bel_to_cell.contains_key(&bel) {
            warn!("bind_bel: bel {} already occupied", bel);
            return false;
        }

        self.bel_to_cell.insert(bel, cell_name);

        // Update the cell's placement info in the design.
        if let Some(cell_idx) = self.design.cell_by_name(cell_name) {
            let cell = self.design.cell_mut(cell_idx);
            cell.bel = bel;
            cell.bel_strength = strength;
        }

        true
    }

    /// Unbind a cell from its BEL.
    ///
    /// Clears both the `bel_to_cell` map entry and the cell's placement fields.
    pub fn unbind_bel(&mut self, bel: BelId) {
        if let Some(cell_name) = self.bel_to_cell.remove(&bel) {
            if let Some(cell_idx) = self.design.cell_by_name(cell_name) {
                let cell = self.design.cell_mut(cell_idx);
                cell.bel = BelId::INVALID;
                cell.bel_strength = PlaceStrength::None;
            }
        }
    }

    /// Check if a BEL is available (not bound to any cell).
    #[inline]
    pub fn is_bel_available(&self, bel: BelId) -> bool {
        !self.bel_to_cell.contains_key(&bel)
    }

    /// Get the name of the cell bound to a BEL, or `None` if unoccupied.
    #[inline]
    pub fn get_bound_bel_cell(&self, bel: BelId) -> Option<IdString> {
        self.bel_to_cell.get(&bel).copied()
    }

    /// Iterate over all BELs on the chip.
    pub fn get_bels(&self) -> impl Iterator<Item = BelId> + '_ {
        self.chipdb.bels()
    }

    /// Get the name string for a BEL.
    #[inline]
    pub fn get_bel_name(&self, bel: BelId) -> &str {
        self.chipdb.bel_name(bel)
    }

    /// Get the type string for a BEL (e.g. "LUT4", "FF").
    #[inline]
    pub fn get_bel_type(&self, bel: BelId) -> &str {
        self.chipdb.bel_type(bel)
    }

    /// Get the grid location (x, y, z) of a BEL.
    #[inline]
    pub fn get_bel_location(&self, bel: BelId) -> Loc {
        self.chipdb.bel_loc(bel)
    }

    /// Get the bucket (placement category) string for a BEL.
    #[inline]
    pub fn get_bel_bucket(&self, bel: BelId) -> &str {
        self.chipdb.bel_bucket(bel)
    }

    /// Get all BELs belonging to a given bucket.
    ///
    /// The bucket caches must be populated first via [`populate_bel_buckets`].
    /// Returns an empty slice if the bucket is unknown or caches are not populated.
    pub fn get_bels_for_bucket(&self, bucket: &str) -> &[BelId] {
        let id = self.id_pool.intern(bucket);
        self.bucket_bels_cache
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    // =====================================================================
    // Wire operations
    // =====================================================================

    /// Bind a wire to a net.
    pub fn bind_wire(&mut self, wire: WireId, net_name: IdString, strength: PlaceStrength) {
        self.wire_to_net.insert(wire, (net_name, strength));
    }

    /// Unbind a wire.
    pub fn unbind_wire(&mut self, wire: WireId) {
        self.wire_to_net.remove(&wire);
    }

    /// Check if a wire is available (not bound to any net).
    #[inline]
    pub fn is_wire_available(&self, wire: WireId) -> bool {
        !self.wire_to_net.contains_key(&wire)
    }

    /// Get the name of the net bound to a wire, or `None` if unbound.
    #[inline]
    pub fn get_bound_wire_net(&self, wire: WireId) -> Option<IdString> {
        self.wire_to_net.get(&wire).map(|(name, _)| *name)
    }

    // =====================================================================
    // PIP operations
    // =====================================================================

    /// Bind a PIP to a net.
    pub fn bind_pip(&mut self, pip: PipId, net_name: IdString, strength: PlaceStrength) {
        self.pip_to_net.insert(pip, (net_name, strength));
    }

    /// Unbind a PIP.
    pub fn unbind_pip(&mut self, pip: PipId) {
        self.pip_to_net.remove(&pip);
    }

    /// Check if a PIP is available (not bound to any net).
    #[inline]
    pub fn is_pip_available(&self, pip: PipId) -> bool {
        !self.pip_to_net.contains_key(&pip)
    }

    /// Get the source wire of a PIP.
    #[inline]
    pub fn get_pip_src_wire(&self, pip: PipId) -> WireId {
        self.chipdb.pip_src_wire(pip)
    }

    /// Get the destination wire of a PIP.
    #[inline]
    pub fn get_pip_dst_wire(&self, pip: PipId) -> WireId {
        self.chipdb.pip_dst_wire(pip)
    }

    // =====================================================================
    // Delay estimation
    // =====================================================================

    /// Estimate the delay between two wires using Manhattan distance.
    ///
    /// This is a coarse estimate suitable for early placement and routing
    /// before detailed timing data is available. The delay is proportional
    /// to the Manhattan distance between the tiles, scaled by [`DELAY_SCALE`]
    /// picoseconds per grid unit.
    pub fn estimate_delay(&self, src: WireId, dst: WireId) -> DelayT {
        let src_loc = self.chipdb.tile_xy(src.tile());
        let dst_loc = self.chipdb.tile_xy(dst.tile());
        let dx = (src_loc.0 - dst_loc.0).abs();
        let dy = (src_loc.1 - dst_loc.1).abs();
        (dx + dy) * DELAY_SCALE
    }

    /// Get the delay of a PIP.
    ///
    /// Currently returns a zero delay quad since detailed timing data is not
    /// yet wired up. In the future this will look up the PIP's timing class
    /// in the speed grade data.
    pub fn get_pip_delay(&self, _pip: PipId) -> DelayQuad {
        // TODO: Look up actual PIP timing from speed grade data.
        DelayQuad::default()
    }

    /// Get the delay of a wire.
    ///
    /// Currently returns a zero delay quad. In the future this will account
    /// for wire capacitance and resistance from the chip database.
    pub fn get_wire_delay(&self, _wire: WireId) -> DelayQuad {
        // TODO: Look up actual wire delay from chip database.
        DelayQuad::default()
    }

    // =====================================================================
    // Placement validity
    // =====================================================================

    /// Check if a cell type is valid for placement at a given BEL.
    ///
    /// Uses database-driven matching: the cell's type (as an IdString) is
    /// compared against the BEL's bucket. The cell type name must match the
    /// BEL bucket string for the placement to be valid.
    pub fn is_valid_bel_for_cell(&self, bel: BelId, cell_type: IdString) -> bool {
        let bucket = self.chipdb.bel_bucket(bel);
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
            let bucket_str = self.chipdb.bel_bucket(bel);
            let bucket_id = self.id_pool.intern(bucket_str);
            self.bucket_bels_cache
                .entry(bucket_id)
                .or_insert_with(Vec::new)
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
    pub fn get_bel_buckets(&self) -> &[IdString] {
        &self.bel_buckets_cache
    }
}

#[cfg(test)]
mod tests;

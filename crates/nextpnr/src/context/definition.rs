//! Context struct definition and constructor.

use super::storage::TileSlotMap;
use super::DeterministicRng;
use crate::chipdb::{BelId, ChipDb};
use crate::common::{IdString, IdStringPool, PlaceStrength};
use crate::netlist::{CellId, Design, NetId, Property};
use rustc_hash::FxHashMap;

/// The central context for the nextpnr place-and-route flow.
///
/// Owns the string pool, chip database, design netlist, and all placement/routing
/// state. Every operation that queries or modifies the hardware mapping goes
/// through this struct.
pub struct Context {
    /// String interning pool shared across the whole flow.
    pub id_pool: IdStringPool,
    /// Read-only chip database describing the FPGA hardware.
    pub(crate) chipdb: ChipDb,
    /// Mutable design netlist being placed and routed.
    pub design: Design,

    // -- Placement state --
    /// For each tile, occupancy of BEL slots by cell index.
    pub(super) bel_to_cell: TileSlotMap<Option<CellId>>,
    // -- Routing state (sparse; grows as bindings are added) --
    /// Wire occupancy keyed by (tile, slot_index). Only bound wires are
    /// stored, so memory scales with routed bindings rather than chip size.
    /// A full XC7 chipdb has ~278 M wire slots; a routed design touches a
    /// tiny fraction, so the sparse representation is dramatically smaller
    /// than a dense per-slot table.
    pub(super) wire_to_net: FxHashMap<(u32, u32), (NetId, PlaceStrength)>,
    /// Pip occupancy keyed by (tile, slot_index). Same rationale as
    /// `wire_to_net`: placement-only flows keep this empty, routing grows
    /// it proportionally to the number of bound pips.
    pub(super) pip_to_net: FxHashMap<(u32, u32), (NetId, PlaceStrength)>,

    // -- Caches (populated on demand) --
    /// For each bucket (bel type), the list of all BelIds belonging to it.
    pub(super) bucket_bels: FxHashMap<IdString, Vec<BelId>>,
    /// Cache of BELs per (region_idx, bucket). Populated on demand.
    pub(super) region_bel_cache: FxHashMap<(u32, IdString), Vec<BelId>>,
    /// Cell type aliases: maps a cell type (e.g. "FDRE") to the BEL bucket
    /// it should be placed on (e.g. "AFF"). Used for architectures where
    /// cell types in the netlist don't match BEL type names in the chipdb.
    pub(super) cell_type_aliases: FxHashMap<IdString, IdString>,

    // -- Settings and flags --
    /// Arbitrary key-value settings (e.g. from command-line options).
    pub(super) settings: FxHashMap<IdString, Property>,
    /// Active speed grade index for timing lookups.
    pub(super) speed_grade_idx: usize,
    /// Deterministic RNG for reproducible results.
    pub(super) rng: DeterministicRng,
    /// Enable verbose output.
    pub(super) verbose: bool,
    /// Enable debug output.
    pub(super) debug: bool,
    /// Force operations even when validity checks fail.
    pub(super) force: bool,
    /// Architecture validity rule, if one has been installed.
    ///
    /// nextpnr's `Arch::isBelLocationValid` delegates to the uarch; this is
    /// where that uarch rule lives for us. `None` means "no rule", which is
    /// `BaseArch`'s default of always-valid -- see
    /// [`Context::is_bel_location_valid`].
    pub(super) validity_check: Option<ArchValidityCheck>,
}

/// An architecture's `isBelLocationValid` rule.
///
/// Takes the whole context because these rules are tile-level: whether a bel is
/// valid depends on what else is currently bound around it.
pub type ArchValidityCheck = std::sync::Arc<dyn Fn(&Context, BelId) -> bool + Send + Sync>;

impl Context {
    /// Create a new context from a chip database.
    ///
    /// The design starts empty; cells and nets should be loaded by the frontend
    /// before placement and routing.
    pub fn new(chipdb: ChipDb) -> Self {
        let mut bel_lengths = Vec::with_capacity(chipdb.num_tiles() as usize);
        for tile in 0..chipdb.num_tiles() {
            let tt = chipdb.tile_type(tile);
            bel_lengths.push(tt.bels.get().len());
        }

        let bel_to_cell = TileSlotMap::with_fill(&bel_lengths, None);
        // wire_to_net / pip_to_net are sparse hash maps: a dense per-slot
        // table would be ~7 GB on XC7, but real designs only bind a few
        // thousand of those slots. The sparse maps start empty and grow
        // one entry at a time as binds arrive.

        // nextpnr has a single global IdString space, and the chipdb's strings
        // occupy their own file indices in it (that is what `known_id_count`
        // is for). Ids stored *inside* the database -- cell timing
        // `type_variant`, timing pin names -- are indices into that space, so a
        // pool that starts empty and interns design names first produces
        // IdStrings that silently miss every such lookup. Seeding in file order
        // makes a pool IdString usable directly as a database key.
        let id_pool = IdStringPool::new();
        for i in 1..chipdb.num_constids() {
            // A null entry still consumes an id; give it a placeholder so the
            // rest of the table does not shift.
            let name = chipdb
                .constid_str(i as i32)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("$unnamed_constid_{i}$"));
            let id = id_pool.intern(&name);
            assert_eq!(
                id.index(),
                i as i32,
                "chipdb constid table is not a dense ordered list: entry {i} ({name:?}) \
                 interned as {}, so pool ids no longer match database ids",
                id.index(),
            );
        }

        Self {
            id_pool,
            chipdb,
            design: Design::new(),
            bel_to_cell,
            wire_to_net: FxHashMap::default(),
            pip_to_net: FxHashMap::default(),
            bucket_bels: FxHashMap::default(),
            region_bel_cache: FxHashMap::default(),
            cell_type_aliases: FxHashMap::default(),
            settings: FxHashMap::default(),
            speed_grade_idx: 0,
            rng: DeterministicRng::new(1),
            verbose: false,
            debug: false,
            force: false,
            validity_check: None,
        }
    }
}

//! Central context struct and architecture API for the nextpnr-rust FPGA
//! place-and-route tool.
//!
//! The [`Context`] ties together the read-only chip database ([`ChipDb`]) with the
//! mutable design netlist ([`Design`]), and maintains the placement and routing
//! state maps that track which hardware resources (bels, wires, pips) are bound
//! to which design elements (cells, nets).
//!
//! All placer, router, and timing code operates through the `Context`.

mod buckets;
mod core;
mod occupancy;
mod rng;
mod storage;
mod timing;
mod views;

pub use rng::DeterministicRng;
pub use views::{Bel, BelPin, BelPinView, Cell, CellPinView, IdStringView, Net, Pip, TileView, Wire};

use crate::chipdb::{BelId, ChipDb};
use crate::common::{IdString, IdStringPool, PlaceStrength};
use crate::netlist::{CellId, Design, NetId, Property};
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
    /// Active speed grade index for timing lookups.
    speed_grade_idx: usize,
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
            speed_grade_idx: 0,
            rng: DeterministicRng::new(1),
            verbose: false,
            debug: false,
            force: false,
        }
    }
}

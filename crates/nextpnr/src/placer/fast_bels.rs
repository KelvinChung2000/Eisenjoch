//! Faithful port of nextpnr's `FastBels`.
//!
//! Source: upstream YosysHQ nextpnr `main` @ `4d235150`,
//! `common/place/fast_bels.h`.
//!
//! ```text
//!  nextpnr -- Next Generation Place and Route
//!
//!  Copyright (C) 2018  Claire Xenia Wolf <claire@yosyshq.com>
//!  Copyright (C) 2018  gatecat <gatecat@ds0.me>
//!
//!  Permission to use, copy, modify, and/or distribute this software for any
//!  purpose with or without fee is hereby granted, provided that the above
//!  copyright notice and this permission notice appear in all copies.
//!
//!  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//!  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//!  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//!  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//!  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//!  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//!  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//! ```
//!
//! A grid-indexed lookup of the BELs that can host a given cell type, so the
//! placers can sample candidates near a location without rescanning the chip.

use rustc_hash::FxHashMap;

use crate::chipdb::BelId;
use crate::common::IdString;
use crate::context::Context;

/// `FastBels::FastBelsData` -- BELs indexed by `[x][y]`.
///
/// Ragged on purpose: nextpnr grows the outer vector to `x + 1` and each inner
/// one to `y + 1` as BELs arrive, so trailing empty rows and columns simply do
/// not exist. Callers bounds-check with `.len()` before indexing, and the ports
/// keep doing that.
pub type FastBelsData = Vec<Vec<Vec<BelId>>>;

/// `FastBels::TypeData`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TypeData {
    /// Index into the per-type BEL grid store.
    pub type_index: usize,
    /// How many BELs on the whole chip could host this type, ignoring
    /// availability. Drives the `min_bels_for_grid_pick` collapse below.
    pub number_of_possible_bels: i32,
}

/// `FastBels`.
pub struct FastBels {
    /// `check_bel_available` -- when set, BELs already bound are left out of
    /// the grid entirely.
    check_bel_available: bool,
    /// `minBelsForGridPick` -- when a type has fewer candidate BELs than this,
    /// its entire grid is collapsed into cell `(0, 0)`.
    ///
    /// Rare types are scattered so thinly that sampling by location almost
    /// always lands on an empty cell; collapsing turns that into a flat list.
    /// A negative value disables the collapse.
    min_bels_for_grid_pick: i32,

    cell_types: FxHashMap<IdString, TypeData>,
    fast_bels_by_cell_type: Vec<FastBelsData>,

    partition_types: FxHashMap<IdString, TypeData>,
    fast_bels_by_partition_type: Vec<FastBelsData>,
}

impl FastBels {
    /// `FastBels(ctx, check_bel_available, minBelsForGridPick)`.
    pub fn new(check_bel_available: bool, min_bels_for_grid_pick: i32) -> Self {
        Self {
            check_bel_available,
            min_bels_for_grid_pick,
            cell_types: FxHashMap::default(),
            fast_bels_by_cell_type: Vec::new(),
            partition_types: FxHashMap::default(),
            fast_bels_by_partition_type: Vec::new(),
        }
    }

    /// Insert `bel` at its location, growing the ragged grid to fit.
    ///
    /// Shared by both `add_cell_type` and `add_bel_bucket`; the C++ repeats
    /// this block verbatim in each.
    fn place_into_grid(
        bel_data: &mut FastBelsData,
        ctx: &Context,
        bel: BelId,
        collapse: bool,
    ) {
        let mut loc = ctx.chipdb().bel_loc(bel);
        if collapse {
            loc.x = 0;
            loc.y = 0;
        }

        if (bel_data.len() as i32) < loc.x + 1 {
            bel_data.resize((loc.x + 1) as usize, Vec::new());
        }
        let column = &mut bel_data[loc.x as usize];
        if (column.len() as i32) < loc.y + 1 {
            column.resize((loc.y + 1) as usize, Vec::new());
        }
        column[loc.y as usize].push(bel);
    }

    /// `addCellType`.
    pub fn add_cell_type(&mut self, ctx: &Context, cell_type: IdString) {
        if self.cell_types.contains_key(&cell_type) {
            return;
        }

        let type_idx = self.cell_types.len();

        // Two passes over the BELs, exactly as in nextpnr: the first counts
        // every BEL that *could* host the type, and that count decides whether
        // the second pass collapses the grid. Counting and filling in one pass
        // would make the collapse depend on how far the scan had got.
        let mut number_of_possible_bels = 0;
        for bel in ctx.chipdb().bels() {
            if ctx.bel(bel).is_valid_for_cell_type(cell_type) {
                number_of_possible_bels += 1;
            }
        }

        let collapse = self.min_bels_for_grid_pick >= 0
            && number_of_possible_bels < self.min_bels_for_grid_pick;

        let mut bel_data = FastBelsData::new();
        for bel in ctx.chipdb().bels() {
            if self.check_bel_available && !ctx.bel(bel).is_available() {
                continue;
            }
            if !ctx.bel(bel).is_valid_for_cell_type(cell_type) {
                continue;
            }
            Self::place_into_grid(&mut bel_data, ctx, bel, collapse);
        }

        self.fast_bels_by_cell_type.push(bel_data);
        self.cell_types.insert(
            cell_type,
            TypeData {
                type_index: type_idx,
                number_of_possible_bels,
            },
        );
    }

    /// `addBelBucket`.
    pub fn add_bel_bucket(&mut self, ctx: &Context, partition: IdString) {
        if self.partition_types.contains_key(&partition) {
            return;
        }

        let type_idx = self.partition_types.len();
        let partition_name = ctx.name_of(partition).to_string();

        let mut number_of_possible_bels = 0;
        for bel in ctx.chipdb().bels() {
            if ctx.bel(bel).bucket() == partition_name {
                number_of_possible_bels += 1;
            }
        }

        let collapse = self.min_bels_for_grid_pick >= 0
            && number_of_possible_bels < self.min_bels_for_grid_pick;

        let mut bel_data = FastBelsData::new();
        for bel in ctx.chipdb().bels() {
            if self.check_bel_available && !ctx.bel(bel).is_available() {
                continue;
            }
            if ctx.bel(bel).bucket() != partition_name {
                continue;
            }
            Self::place_into_grid(&mut bel_data, ctx, bel, collapse);
        }

        self.fast_bels_by_partition_type.push(bel_data);
        self.partition_types.insert(
            partition,
            TypeData {
                type_index: type_idx,
                number_of_possible_bels,
            },
        );
    }

    /// `getBelsForCellType` -- returns the BEL grid and the number of possible
    /// BELs, building the entry on first use.
    pub fn bels_for_cell_type(
        &mut self,
        ctx: &Context,
        cell_type: IdString,
    ) -> (&FastBelsData, i32) {
        if !self.cell_types.contains_key(&cell_type) {
            self.add_cell_type(ctx, cell_type);
        }
        let data = self.cell_types[&cell_type];
        (
            &self.fast_bels_by_cell_type[data.type_index],
            data.number_of_possible_bels,
        )
    }

    /// `getBelsForBelBucket`.
    pub fn bels_for_bel_bucket(
        &mut self,
        ctx: &Context,
        partition: IdString,
    ) -> (&FastBelsData, i32) {
        if !self.partition_types.contains_key(&partition) {
            self.add_bel_bucket(ctx, partition);
        }
        let data = self.partition_types[&partition];
        (
            &self.fast_bels_by_partition_type[data.type_index],
            data.number_of_possible_bels,
        )
    }
}

//! Spatial index over one cell type's BELs, and a nearest-first walk over it.

use crate::chipdb::BelId;

/// Tile `(x, y)` -> the BELs of one cell type in that tile, as CSR.
///
/// Stores only each BEL's *enumeration index* -- its position in the type's
/// `bel_data_cache` entry -- because that is both the key the walk's tie-break
/// needs and the way the `BelId` is recovered. Storing the `BelId` too would
/// double the index for nothing.
pub(crate) struct BelGrid {
    width: i32,
    height: i32,
    /// Prefix sums into `entries`; `len() == width * height + 1`.
    offsets: Vec<u32>,
    /// Enumeration indices, grouped by tile, ascending within each tile.
    entries: Vec<u32>,
}

impl BelGrid {
    /// Index one cell type's `(BelId, x, y, z)` cache entries by tile.
    ///
    /// Panics if any BEL sits outside `width x height`: such a BEL would be
    /// unreachable by the ring walk yet visible to `candidate_list`, which is
    /// exactly the silent divergence this whole design exists to rule out.
    pub(crate) fn build(bels: &[(BelId, i32, i32, i32)], width: i32, height: i32) -> Self {
        assert!(width > 0 && height > 0, "degenerate grid {width}x{height}");
        assert!(
            bels.len() <= u32::MAX as usize,
            "{} BELs overflows the u32 enumeration index",
            bels.len()
        );
        let n_cells = (width as usize) * (height as usize);
        let cell_of = |x: i32, y: i32| -> usize {
            assert!(
                x >= 0 && x < width && y >= 0 && y < height,
                "BEL at ({x}, {y}) is outside the {width}x{height} grid"
            );
            (y as usize) * (width as usize) + (x as usize)
        };

        let mut offsets = vec![0u32; n_cells + 1];
        for &(_, x, y, _) in bels {
            offsets[cell_of(x, y) + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        // Counting-sort fill. Walking `bels` in enumeration order is what
        // leaves each tile's entries ascending.
        let mut cursor = offsets.clone();
        let mut entries = vec![0u32; bels.len()];
        for (enum_index, &(_, x, y, _)) in bels.iter().enumerate() {
            let c = cell_of(x, y);
            entries[cursor[c] as usize] = enum_index as u32;
            cursor[c] += 1;
        }

        Self {
            width,
            height,
            offsets,
            entries,
        }
    }

    /// Enumeration indices in tile `(x, y)`, ascending. Empty off-grid: the
    /// walk probes ring cells that fall past the device edge.
    #[inline]
    pub(crate) fn at(&self, x: i32, y: i32) -> &[u32] {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return &[];
        }
        let c = (y as usize) * (self.width as usize) + (x as usize);
        &self.entries[self.offsets[c] as usize..self.offsets[c + 1] as usize]
    }

    #[inline]
    pub(crate) fn width(&self) -> i32 {
        self.width
    }

    #[inline]
    pub(crate) fn height(&self) -> i32 {
        self.height
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `per_tile` BELs at every (x, y) of a `dim x dim` grid, enumerated in
    /// the order `bels_for_bucket` yields them: tile-major, then slot.
    fn grid_bels(dim: i32, per_tile: i32) -> Vec<(BelId, i32, i32, i32)> {
        let mut bels = Vec::new();
        let mut tile = 0;
        for x in 0..dim {
            for y in 0..dim {
                for z in 0..per_tile {
                    bels.push((BelId::new(tile, z), x, y, z));
                }
                tile += 1;
            }
        }
        bels
    }

    /// Every BEL must land in its own tile, and nowhere else.
    #[test]
    fn build_places_every_bel_in_its_own_tile() {
        let bels = grid_bels(6, 3);
        let grid = BelGrid::build(&bels, 6, 6);
        let mut seen = 0;
        for x in 0..6 {
            for y in 0..6 {
                let entries = grid.at(x, y);
                assert_eq!(entries.len(), 3, "tile ({x},{y})");
                for &e in entries {
                    let (_, bx, by, _) = bels[e as usize];
                    assert_eq!((bx, by), (x, y), "entry {e} filed under the wrong tile");
                }
                seen += entries.len();
            }
        }
        assert_eq!(seen, bels.len(), "grid dropped or duplicated entries");
    }

    /// A tile's entries must stay in enumeration order.
    ///
    /// Every BEL in a tile shares one `dist_sq`, so this ordering *is* the
    /// tie-break the walk relies on to match `candidate_list`. A counting-sort
    /// fill that iterated the source out of order would silently break it.
    #[test]
    fn tile_entries_are_in_enumeration_order() {
        let bels = grid_bels(5, 4);
        let grid = BelGrid::build(&bels, 5, 5);
        for x in 0..5 {
            for y in 0..5 {
                let entries = grid.at(x, y);
                assert!(
                    entries.windows(2).all(|w| w[0] < w[1]),
                    "tile ({x},{y}) entries out of order: {entries:?}"
                );
            }
        }
    }

    /// Sparse layouts are the interesting case: most tiles hold nothing.
    #[test]
    fn empty_tiles_return_an_empty_slice() {
        let bels = vec![(BelId::new(0, 0), 2, 3, 0), (BelId::new(1, 0), 2, 3, 1)];
        let grid = BelGrid::build(&bels, 8, 8);
        assert_eq!(grid.at(2, 3).len(), 2);
        assert!(grid.at(0, 0).is_empty());
        assert!(grid.at(7, 7).is_empty());
    }

    /// The walk probes ring cells that fall off the edge of the device;
    /// `at` must answer those without a bounds panic.
    #[test]
    fn out_of_bounds_lookups_are_empty_not_a_panic() {
        let grid = BelGrid::build(&grid_bels(4, 2), 4, 4);
        for (x, y) in [(-1, 0), (0, -1), (4, 0), (0, 4), (-7, 9)] {
            assert!(grid.at(x, y).is_empty(), "({x},{y})");
        }
    }

    #[test]
    fn dimensions_are_reported() {
        let grid = BelGrid::build(&grid_bels(4, 2), 4, 9);
        assert_eq!((grid.width(), grid.height()), (4, 9));
        assert!(!grid.is_empty());
        assert!(BelGrid::build(&[], 4, 9).is_empty());
    }

    /// Fail loudly. A BEL outside the declared grid would be unreachable by
    /// the walk but present in `candidate_list`, i.e. a silent divergence.
    #[test]
    #[should_panic(expected = "outside the")]
    fn a_bel_outside_the_grid_panics() {
        BelGrid::build(&[(BelId::new(0, 0), 9, 0, 0)], 4, 4);
    }
}

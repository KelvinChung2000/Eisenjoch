use std::ops::Range;
use std::sync::Arc;

use crate::chipdb::{BelId, Loc};
use crate::common::IdString;
use crate::netlist::Rect;
use rustc_hash::FxHashMap;

use super::Context;

/// One bucket's BELs seen as a tile grid.
///
/// A bucket list is in chipdb order -- ascending tile index `y * width + x`,
/// then slot -- so each tile's BELs are one contiguous run of it, found by
/// binary search. That is why no bucket carries a per-tile table.
#[derive(Clone, Copy)]
pub struct BucketGrid<'a> {
    bels: &'a [BelId],
    width: i32,
    height: i32,
}

impl<'a> BucketGrid<'a> {
    /// `bels` must be in chipdb order, as every list from the context is.
    pub fn new(bels: &'a [BelId], width: i32, height: i32) -> Self {
        Self {
            bels,
            width,
            height,
        }
    }

    pub fn bels(&self) -> &'a [BelId] {
        self.bels
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    /// Positions in [`Self::bels`] of the BELs in tile `(x, y)`; empty off-grid.
    pub fn span(&self, x: i32, y: i32) -> Range<usize> {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 0..0;
        }
        let tile = y * self.width + x;
        self.tiles_span(tile, tile)
    }

    /// The tile coordinates of one of the grid's BELs.
    pub fn xy(&self, bel: BelId) -> (i32, i32) {
        (bel.tile() % self.width, bel.tile() / self.width)
    }

    /// The BELs in tile `(x, y)`, in bucket order; empty off-grid.
    pub fn at(&self, x: i32, y: i32) -> &'a [BelId] {
        &self.bels[self.span(x, y)]
    }

    /// One span per grid row that `rect` covers, ascending, so concatenated
    /// they stay in bucket order.
    pub fn row_spans(self, rect: Rect) -> impl Iterator<Item = Range<usize>> + 'a {
        let (x0, x1) = (rect.x0.max(0), rect.x1.min(self.width - 1));
        let rows = if x0 <= x1 {
            rect.y0.max(0)..=rect.y1.min(self.height - 1)
        } else {
            1..=0
        };
        rows.map(move |y| self.tiles_span(y * self.width + x0, y * self.width + x1))
    }

    fn tiles_span(&self, first: i32, last: i32) -> Range<usize> {
        let lo = self.bels.partition_point(|b| b.tile() < first);
        let hi = lo + self.bels[lo..].partition_point(|b| b.tile() <= last);
        lo..hi
    }
}

impl Context {
    /// Resolve a cell type to its BEL bucket, following aliases if set.
    #[inline]
    pub fn resolve_bucket(&self, cell_type: IdString) -> IdString {
        self.cell_type_aliases
            .get(&cell_type)
            .copied()
            .unwrap_or(cell_type)
    }

    /// Register a cell type alias: cells of type `cell_type` will be placed
    /// on BELs of type `bel_type`.
    pub fn add_cell_type_alias(&mut self, cell_type: &str, bel_type: &str) {
        let ct = self.id(cell_type);
        let bt = self.id(bel_type);
        self.cell_type_aliases.insert(ct, bt);
    }

    /// The BELs of a bucket (resolves aliases), in chipdb order.
    fn bucket_bel_ids(&self, bucket: IdString) -> &[BelId] {
        self.bucket_bels
            .get(&self.resolve_bucket(bucket))
            .map_or(&[], |v| v)
    }

    /// All BELs belonging to a given bucket (resolves aliases).
    pub fn bels_for_bucket(&self, bucket: IdString) -> impl Iterator<Item = super::Bel<'_>> {
        self.bucket_bel_ids(bucket)
            .iter()
            .copied()
            .map(|bel| self.bel(bel))
    }

    /// The context's own list for a bucket (resolves aliases), shared rather
    /// than copied so a placer can hold it across `&mut` uses of the context.
    pub fn bucket_bel_list(&self, bucket: IdString) -> Arc<[BelId]> {
        self.shared_list(self.resolve_bucket(bucket))
    }

    /// The BELs [`Bel::is_valid_for_cell_type`](super::Bel::is_valid_for_cell_type)
    /// accepts for `cell_type`. That check compares type names and does not
    /// follow aliases, so neither does this.
    pub fn bels_valid_for_cell_type(&self, cell_type: IdString) -> Arc<[BelId]> {
        self.shared_list(cell_type)
    }

    fn shared_list(&self, bucket: IdString) -> Arc<[BelId]> {
        self.bucket_bels
            .get(&bucket)
            .cloned()
            .unwrap_or_else(|| Arc::from([]))
    }

    /// A tile-grid view of a list obtained from this context.
    pub fn bucket_grid<'a>(&self, bels: &'a [BelId]) -> BucketGrid<'a> {
        BucketGrid::new(bels, self.chipdb.width(), self.chipdb.height())
    }

    /// The BEL of a bucket (resolves aliases) at an exact `(x, y, z)`.
    pub fn bucket_bel_at(&self, bucket: IdString, loc: Loc) -> Option<BelId> {
        self.bucket_grid(self.bucket_bel_ids(bucket))
            .at(loc.x, loc.y)
            .iter()
            .copied()
            .find(|&bel| self.chipdb.bel_loc(bel).z == loc.z)
    }

    /// Populate the bel bucket cache by scanning all BELs in the chip database.
    pub fn populate_bel_buckets(&mut self) {
        let mut lists: FxHashMap<IdString, Vec<BelId>> = FxHashMap::default();
        for bel in self.chipdb.bels() {
            let bucket_id = self.id_pool.intern(self.chipdb.bel_type(bel));
            lists.entry(bucket_id).or_default().push(bel);
        }
        self.bucket_bels = lists.into_iter().map(|(k, v)| (k, v.into())).collect();
    }

    /// Populate the buckets unless that is done. They derive from the chipdb
    /// alone, so a filled table is never stale.
    pub fn ensure_bel_buckets(&mut self) {
        if self.bucket_bels.is_empty() {
            self.populate_bel_buckets();
        }
    }

    /// Get all unique bel bucket names (sorted by IdString index for determinism).
    pub fn bel_buckets(&self) -> Vec<IdString> {
        let mut buckets: Vec<IdString> = self.bucket_bels.keys().copied().collect();
        buckets.sort_by_key(|id| id.index());
        buckets
    }

    /// Check whether a BEL falls within a region constraint.
    pub fn is_bel_in_region(&self, bel: BelId, region_idx: u32) -> bool {
        let region = self.design.region(region_idx);
        let loc = self.bel(bel).loc();
        region.contains(loc.x, loc.y)
    }

    /// Positions in `bels` (a list from this context) of the BELs inside a
    /// region, ascending. Only the rows of the region's bounding box are
    /// searched, so the cost follows the region, not the bucket.
    pub fn bucket_positions_in_region<'a>(
        &'a self,
        bels: &'a [BelId],
        region_idx: u32,
    ) -> impl Iterator<Item = usize> + 'a {
        let region = self.design.region(region_idx);
        let grid = self.bucket_grid(bels);
        region
            .bounding_box()
            .into_iter()
            .flat_map(move |bbox| grid.row_spans(bbox))
            .flatten()
            .filter(move |&i| {
                let loc = self.chipdb.bel_loc(bels[i]);
                region.contains(loc.x, loc.y)
            })
    }

    /// The BELs of a bucket (resolves aliases) inside a region, in bucket order.
    pub fn bels_for_bucket_in_region(
        &self,
        bucket: IdString,
        region_idx: u32,
    ) -> impl Iterator<Item = BelId> + '_ {
        let bels = self.bucket_bel_ids(bucket);
        self.bucket_positions_in_region(bels, region_idx)
            .map(move |i| bels[i])
    }

    /// Check whether any BEL of the given type exists in the chipdb (resolves aliases).
    ///
    /// Requires `populate_bel_buckets()` to have been called first.
    pub fn has_bel_type(&self, bel_type: IdString) -> bool {
        !self.bucket_bel_ids(bel_type).is_empty()
    }
}

/// The example-arch context the integration tests use, buckets populated.
#[cfg(test)]
pub(crate) fn example_context() -> Context {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/example.bin");
    let mut ctx = Context::new(crate::chipdb::ChipDb::load(&path).expect("example chipdb"));
    ctx.populate_bel_buckets();
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_lists_are_in_chipdb_order() {
        let ctx = example_context();
        for bucket in ctx.bel_buckets() {
            let bels = ctx.bucket_bel_list(bucket);
            assert!(
                bels.windows(2)
                    .all(|w| (w[0].tile(), w[0].index()) < (w[1].tile(), w[1].index())),
                "{} is not tile-sorted",
                ctx.name_of(bucket)
            );
        }
    }

    /// Aliases borrow their bucket's list, so however many there are the
    /// context holds each BEL once.
    #[test]
    fn bucket_storage_does_not_scale_with_alias_count() {
        let mut ctx = example_context();
        let list = ctx.bucket_bel_list(ctx.id("DFF"));
        assert!(!list.is_empty());
        for i in 0..100 {
            ctx.add_cell_type_alias(&format!("ALIAS{i}"), "DFF");
        }
        for i in 0..100 {
            let alias = ctx.id(format!("ALIAS{i}").as_str());
            assert!(Arc::ptr_eq(&ctx.bucket_bel_list(alias), &list), "ALIAS{i}");
        }
        let stored: usize = ctx
            .bel_buckets()
            .iter()
            .map(|&b| ctx.bucket_bel_list(b).len())
            .sum();
        assert_eq!(stored, ctx.chipdb.bels().count());
    }

    #[test]
    fn bucket_bel_at_finds_each_bel_by_location_and_nothing_else() {
        let ctx = example_context();
        let lut = ctx.id("LUT6");
        let mut lut_slots = 0;
        for bel in ctx.chipdb.bels() {
            let loc = ctx.chipdb.bel_loc(bel);
            let bucket = ctx.id(ctx.chipdb.bel_type(bel));
            assert_eq!(ctx.bucket_bel_at(bucket, loc), Some(bel), "{bel}");
            if bucket == lut {
                lut_slots += 1;
            } else {
                assert_eq!(ctx.bucket_bel_at(lut, loc), None, "{bel} is not a LUT slot");
            }
        }
        assert!(lut_slots > 0);
        let (w, h) = (ctx.chipdb.width(), ctx.chipdb.height());
        for (x, y) in [(-1, 1), (1, -1), (w, 1), (1, h)] {
            assert_eq!(ctx.bucket_bel_at(lut, Loc::new(x, y, 0)), None, "({x},{y})");
        }
    }

    /// Searching only the bounding box's rows must find exactly what a
    /// filter over the whole bucket finds, in the same order.
    #[test]
    fn region_positions_match_a_filter_over_the_bucket() {
        let mut ctx = example_context();
        let overlapping = ctx.design.add_region(ctx.id("overlapping"));
        let rects = &mut ctx.design.region_mut(overlapping).rects;
        rects.push(Rect::new(1, 1, 3, 2));
        rects.push(Rect::new(2, 2, 4, 5));
        rects.push(Rect::new(8, -2, 12, 1));
        let empty = ctx.design.add_region(ctx.id("empty"));

        for bucket in ctx.bel_buckets() {
            let bels = ctx.bucket_bel_list(bucket);
            let want: Vec<usize> = (0..bels.len())
                .filter(|&i| ctx.is_bel_in_region(bels[i], overlapping))
                .collect();
            let got: Vec<usize> = ctx.bucket_positions_in_region(&bels, overlapping).collect();
            assert_eq!(got, want, "{}", ctx.name_of(bucket));
            assert_eq!(ctx.bucket_positions_in_region(&bels, empty).count(), 0);
        }
    }
}

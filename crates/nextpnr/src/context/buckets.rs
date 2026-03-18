use crate::chipdb::BelId;
use crate::common::IdString;

use super::Context;

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

    /// All BELs belonging to a given bucket (resolves aliases).
    pub fn bels_for_bucket(&self, bucket: IdString) -> impl Iterator<Item = super::Bel<'_>> {
        let resolved = self.resolve_bucket(bucket);
        self.bucket_bels
            .get(&resolved)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(|bel| self.bel(bel))
    }

    /// Populate the bel bucket cache by scanning all BELs in the chip database.
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

    /// Check whether a BEL falls within a region constraint.
    pub fn is_bel_in_region(&self, bel: BelId, region_idx: u32) -> bool {
        let region = self.design.region(region_idx);
        let loc = self.bel(bel).loc();
        region.contains(loc.x, loc.y)
    }

    /// Get all BELs for a given bucket that fall within a region.
    ///
    /// Results are cached. Call `invalidate_region_cache()` if regions change.
    pub fn bels_for_bucket_in_region(
        &mut self,
        bucket: IdString,
        region_idx: u32,
    ) -> &[BelId] {
        let resolved = self.resolve_bucket(bucket);
        let key = (region_idx, resolved);
        self.region_bel_cache.entry(key).or_insert_with(|| {
            let region = &self.design.regions[region_idx as usize];
            self.bucket_bels
                .get(&resolved)
                .map(|v| v.as_slice())
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(|&bel| {
                    let loc = self.chipdb.bel_loc(bel);
                    region.contains(loc.x, loc.y)
                })
                .collect()
        })
    }

    /// Invalidate the region BEL cache (call after modifying region constraints).
    pub fn invalidate_region_cache(&mut self) {
        self.region_bel_cache.clear();
    }

    /// Check whether any BEL of the given type exists in the chipdb (resolves aliases).
    ///
    /// Requires `populate_bel_buckets()` to have been called first.
    pub fn has_bel_type(&self, bel_type: IdString) -> bool {
        let resolved = self.resolve_bucket(bel_type);
        self.bucket_bels
            .get(&resolved)
            .is_some_and(|v| !v.is_empty())
    }
}

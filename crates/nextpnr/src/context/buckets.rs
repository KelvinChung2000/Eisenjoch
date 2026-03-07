use crate::common::IdString;

use super::Context;

impl Context {
    /// All BELs belonging to a given bucket.
    pub fn bels_for_bucket(&self, bucket: IdString) -> impl Iterator<Item = super::Bel<'_>> {
        self.bucket_bels
            .get(&bucket)
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
}

use crate::chipdb::ChipDb;
use crate::common::{IdString, IdStringPool};

/// Read-only view of chip database + string interning.
pub struct ChipView<'a> {
    pub chipdb: &'a ChipDb,
    pub id_pool: &'a IdStringPool,
}

impl<'a> ChipView<'a> {
    /// Look up a string by IdString.
    pub fn name_of(&self, id: IdString) -> &str {
        self.id_pool.lookup(id).unwrap_or("<unknown>")
    }
}

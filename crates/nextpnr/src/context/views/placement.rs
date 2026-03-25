use crate::chipdb::{BelId, ChipDb};
use crate::context::storage::TileSlotMap;
use crate::netlist::CellId;

/// Read-only view of placement state.
pub struct PlacementView<'a> {
    pub chipdb: &'a ChipDb,
    pub(crate) bel_to_cell: &'a TileSlotMap<Option<CellId>>,
}

impl<'a> PlacementView<'a> {
    /// Check if a BEL is occupied.
    pub fn is_bel_occupied(&self, bel: BelId) -> bool {
        let tile = usize::try_from(bel.tile()).ok();
        let index = usize::try_from(bel.index()).ok();
        match (tile, index) {
            (Some(t), Some(i)) => self.bel_to_cell.get(t, i).map_or(false, |v| v.is_some()),
            _ => false,
        }
    }
}

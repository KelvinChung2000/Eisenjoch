/// Flat tile-slot container used for compact internal resource occupancy maps.
///
/// Stores all tile-local slots in one contiguous vector and a per-tile offset
/// table, avoiding `Vec<Vec<_>>` overhead while preserving tile/index access.
pub(crate) struct TileSlotMap<T> {
    offsets: Vec<usize>,
    data: Vec<T>,
}

impl<T: Clone> TileSlotMap<T> {
    pub(crate) fn with_fill(tile_lengths: &[usize], fill: T) -> Self {
        let mut offsets = Vec::with_capacity(tile_lengths.len() + 1);
        offsets.push(0);

        let mut total = 0usize;
        for &len in tile_lengths {
            total += len;
            offsets.push(total);
        }

        let data = vec![fill; total];
        Self { offsets, data }
    }
}

impl<T> TileSlotMap<T> {
    #[inline]
    fn index_of(&self, tile: usize, slot: usize) -> Option<usize> {
        let start = *self.offsets.get(tile)?;
        let end = *self.offsets.get(tile + 1)?;
        let idx = start.checked_add(slot)?;
        if idx < end {
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn get(&self, tile: usize, slot: usize) -> Option<&T> {
        let idx = self.index_of(tile, slot)?;
        self.data.get(idx)
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, tile: usize, slot: usize) -> Option<&mut T> {
        let idx = self.index_of(tile, slot)?;
        self.data.get_mut(idx)
    }
}

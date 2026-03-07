use super::Context;
use crate::types::{DelayT, WireId};

/// Delay scaling factor: picoseconds per Manhattan grid unit.
const DELAY_SCALE: i32 = 100;

impl Context {
    /// Estimate the delay between two wires using Manhattan distance.
    pub fn estimate_delay(&self, src: impl Into<WireId>, dst: impl Into<WireId>) -> DelayT {
        let src = src.into();
        let dst = dst.into();
        let src_loc = self.chipdb.tile_xy(src.tile());
        let dst_loc = self.chipdb.tile_xy(dst.tile());
        let dx = (src_loc.0 - dst_loc.0).abs();
        let dy = (src_loc.1 - dst_loc.1).abs();
        (dx + dy) * DELAY_SCALE
    }
}

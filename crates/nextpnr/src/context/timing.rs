use crate::chipdb::SpeedGradePod;
use crate::chipdb::WireId;
use crate::timing::DelayT;
use crate::netlist::NetId;

use super::Context;

/// Delay scaling factor: picoseconds per Manhattan grid unit.
const DELAY_SCALE: i32 = 100;

/// Compute Manhattan-distance delay between two tile locations.
fn manhattan_delay(loc_a: (i32, i32), loc_b: (i32, i32)) -> DelayT {
    let dx = (loc_a.0 - loc_b.0).abs();
    let dy = (loc_a.1 - loc_b.1).abs();
    (dx + dy) * DELAY_SCALE
}

impl Context {
    /// Set the active speed grade index.
    pub fn set_speed_grade(&mut self, index: usize) {
        self.speed_grade_idx = index;
    }

    /// Get the active speed grade index.
    pub fn speed_grade_idx(&self) -> usize {
        self.speed_grade_idx
    }

    /// Get the active speed grade POD, if available.
    pub fn speed_grade(&self) -> Option<&SpeedGradePod> {
        self.chipdb.speed_grade(self.speed_grade_idx)
    }

    /// Estimate the delay between two wires using Manhattan distance.
    pub fn estimate_delay(&self, src: impl Into<WireId>, dst: impl Into<WireId>) -> DelayT {
        let src_loc = self.chipdb.tile_xy(src.into().tile());
        let dst_loc = self.chipdb.tile_xy(dst.into().tile());
        manhattan_delay(src_loc, dst_loc)
    }

    /// Estimate delay for a net based on placed driver/user BEL locations.
    pub fn estimate_delay_for_net(&self, net_idx: NetId) -> DelayT {
        let net = self.design.net(net_idx);
        if !net.driver.is_valid() {
            return 0;
        }
        let driver_cell = net.driver.cell;
        let Some(driver_bel) = self.design.cell(driver_cell).bel else {
            return 0;
        };
        let driver_loc = self.chipdb.tile_xy(driver_bel.tile());
        let mut max_delay: DelayT = 0;
        for user in &net.users {
            if !user.is_valid() {
                continue;
            }
            let user_cell = user.cell;
            let Some(user_bel) = self.design.cell(user_cell).bel else {
                continue;
            };
            let user_loc = self.chipdb.tile_xy(user_bel.tile());
            max_delay = max_delay.max(manhattan_delay(driver_loc, user_loc));
        }
        max_delay
    }
}

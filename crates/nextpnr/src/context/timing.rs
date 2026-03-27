use crate::chipdb::SpeedGradePod;
use crate::chipdb::WireId;
use crate::timing::DelayT;
use crate::netlist::NetId;

use super::Context;

/// Delay scaling factor: estimated cost per Manhattan grid unit.
/// Must be ≤ the cheapest PIP cost per tile for A* admissibility.
/// XC7 SPAN PIPs cost 10 per tile, MUX PIPs cost 150.
/// Using 10 ensures A* is admissible and SPAN-preferring.
const DELAY_SCALE: i32 = 10;

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

    /// Node-aware delay estimate: considers the closest tile reachable
    /// via the wire's routing node (if any). More expensive than
    /// `estimate_delay` but gives tighter bounds for wires on long-range nodes.
    pub fn estimate_delay_node_aware(&self, src: impl Into<WireId>, dst: impl Into<WireId>) -> DelayT {
        let src_wire = src.into();
        let dst_loc = self.chipdb.tile_xy(dst.into().tile());
        let mut best = manhattan_delay(self.chipdb.tile_xy(src_wire.tile()), dst_loc);
        self.chipdb.node_wires_cb(src_wire, |nw| {
            let d = manhattan_delay(self.chipdb.tile_xy(nw.tile()), dst_loc);
            if d < best {
                best = d;
            }
        });
        best
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

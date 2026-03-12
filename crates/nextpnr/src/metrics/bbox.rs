//! Axis-aligned bounding box computation for nets.

use crate::context::Context;
use crate::netlist::NetId;

/// Axis-aligned bounding box in tile coordinates.
pub struct BoundingBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl BoundingBox {
    /// Check whether a tile coordinate (x, y) falls within this bounding box.
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Compute a bounding box for a net based on its connected cells' BEL locations.
///
/// The box is expanded by `margin` tiles in each direction and clamped to the
/// chip grid boundaries.
pub fn compute_bbox(ctx: &Context, net_idx: NetId, margin: i32) -> BoundingBox {
    let net = ctx.net(net_idx);

    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;
    let mut found_any = false;

    let cell_indices = net
        .driver()
        .into_iter()
        .map(|pin| pin.cell)
        .chain(net.users().iter().filter(|u| u.is_valid()).map(|u| u.cell));

    for cell_idx in cell_indices {
        if let Some(bel) = ctx.cell(cell_idx).bel() {
            let loc = bel.loc();
            x0 = x0.min(loc.x);
            y0 = y0.min(loc.y);
            x1 = x1.max(loc.x);
            y1 = y1.max(loc.y);
            found_any = true;
        }
    }

    if !found_any {
        return BoundingBox {
            x0: 0,
            y0: 0,
            x1: ctx.chipdb().width() - 1,
            y1: ctx.chipdb().height() - 1,
        };
    }

    BoundingBox {
        x0: (x0 - margin).max(0),
        y0: (y0 - margin).max(0),
        x1: (x1 + margin).min(ctx.chipdb().width() - 1),
        y1: (y1 + margin).min(ctx.chipdb().height() - 1),
    }
}

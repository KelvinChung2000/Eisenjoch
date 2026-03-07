//! Grid location type for FPGA placement.

use std::fmt;

/// A location on the FPGA grid.
///
/// `x` and `y` represent the tile coordinates, while `z` represents the
/// position within the tile (e.g., which BEL slot).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Loc {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Loc {
    /// Create a new location.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Returns the Manhattan distance between two locations (ignoring z).
    #[inline]
    pub fn manhattan_distance(self, other: Loc) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

impl fmt::Debug for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Loc({}, {}, {})", self.x, self.y, self.z)
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

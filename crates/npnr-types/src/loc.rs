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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_fields() {
        let loc = Loc::new(1, 2, 3);
        assert_eq!(loc.x, 1);
        assert_eq!(loc.y, 2);
        assert_eq!(loc.z, 3);
    }

    #[test]
    fn default_is_origin() {
        let loc = Loc::default();
        assert_eq!(loc.x, 0);
        assert_eq!(loc.y, 0);
        assert_eq!(loc.z, 0);
    }

    #[test]
    fn equality() {
        let a = Loc::new(1, 2, 3);
        let b = Loc::new(1, 2, 3);
        let c = Loc::new(1, 2, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn copy_semantics() {
        let a = Loc::new(5, 6, 7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Loc::new(1, 2, 3));
        set.insert(Loc::new(4, 5, 6));
        set.insert(Loc::new(1, 2, 3));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn manhattan_distance_same_point() {
        let a = Loc::new(5, 5, 0);
        assert_eq!(a.manhattan_distance(a), 0);
    }

    #[test]
    fn manhattan_distance_horizontal() {
        let a = Loc::new(0, 0, 0);
        let b = Loc::new(5, 0, 0);
        assert_eq!(a.manhattan_distance(b), 5);
    }

    #[test]
    fn manhattan_distance_vertical() {
        let a = Loc::new(0, 0, 0);
        let b = Loc::new(0, 3, 0);
        assert_eq!(a.manhattan_distance(b), 3);
    }

    #[test]
    fn manhattan_distance_diagonal() {
        let a = Loc::new(0, 0, 0);
        let b = Loc::new(3, 4, 0);
        assert_eq!(a.manhattan_distance(b), 7);
    }

    #[test]
    fn manhattan_distance_ignores_z() {
        let a = Loc::new(0, 0, 0);
        let b = Loc::new(0, 0, 100);
        assert_eq!(a.manhattan_distance(b), 0);
    }

    #[test]
    fn manhattan_distance_negative_coords() {
        let a = Loc::new(-3, -4, 0);
        let b = Loc::new(3, 4, 0);
        assert_eq!(a.manhattan_distance(b), 14);
    }

    #[test]
    fn debug_format() {
        let loc = Loc::new(1, 2, 3);
        assert_eq!(format!("{:?}", loc), "Loc(1, 2, 3)");
    }

    #[test]
    fn display_format() {
        let loc = Loc::new(1, 2, 3);
        assert_eq!(format!("{}", loc), "(1, 2, 3)");
    }
}

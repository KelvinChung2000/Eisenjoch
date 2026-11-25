//! Region constraints for floorplanning.

use crate::common::IdString;

/// Axis-aligned rectangle in tile coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    /// Create a new rectangle. Coordinates are inclusive on both ends.
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Check whether a tile coordinate falls within this rectangle.
    #[inline]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    /// Compute the area (number of tiles) covered by this rectangle.
    pub fn area(&self) -> i32 {
        (self.x1 - self.x0 + 1).max(0) * (self.y1 - self.y0 + 1).max(0)
    }
}

/// A named region constraint (pblock) that confines cells to one or more rectangles.
pub struct RegionConstraint {
    /// Region name.
    pub name: IdString,
    /// One or more rectangles defining the allowed area.
    pub rects: Vec<Rect>,
}

impl RegionConstraint {
    /// Create a new empty region with the given name.
    pub fn new(name: IdString) -> Self {
        Self {
            name,
            rects: Vec::new(),
        }
    }

    /// Check whether a tile coordinate falls within any rectangle of this region.
    #[inline]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        self.rects.iter().any(|r| r.contains(x, y))
    }

    /// Compute the bounding box enclosing all rectangles in this region.
    ///
    /// Returns `None` if the region has no rectangles.
    pub fn bounding_box(&self) -> Option<Rect> {
        if self.rects.is_empty() {
            return None;
        }
        let mut x0 = i32::MAX;
        let mut y0 = i32::MAX;
        let mut x1 = i32::MIN;
        let mut y1 = i32::MIN;
        for r in &self.rects {
            x0 = x0.min(r.x0);
            y0 = y0.min(r.y0);
            x1 = x1.max(r.x1);
            y1 = y1.max(r.y1);
        }
        Some(Rect { x0, y0, x1, y1 })
    }
}

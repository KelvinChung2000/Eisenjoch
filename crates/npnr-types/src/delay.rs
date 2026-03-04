//! Delay types for timing analysis.
//!
//! All delays are represented in picoseconds as i32 values.

use std::fmt;
use std::ops::{Add, Sub};

/// Delay value in picoseconds.
pub type DelayT = i32;

/// A pair of min/max delays, used to represent setup/hold or similar constraints.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DelayPair {
    pub min_delay: DelayT,
    pub max_delay: DelayT,
}

impl DelayPair {
    /// Create a new delay pair with explicit min and max values.
    #[inline]
    pub const fn new(min_delay: DelayT, max_delay: DelayT) -> Self {
        Self { min_delay, max_delay }
    }

    /// Create a delay pair where min and max are the same value.
    #[inline]
    pub const fn uniform(delay: DelayT) -> Self {
        Self {
            min_delay: delay,
            max_delay: delay,
        }
    }

    /// Returns the average of min and max delays.
    #[inline]
    pub const fn average(self) -> DelayT {
        (self.min_delay + self.max_delay) / 2
    }
}

impl Add for DelayPair {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            min_delay: self.min_delay + rhs.min_delay,
            max_delay: self.max_delay + rhs.max_delay,
        }
    }
}

impl Sub for DelayPair {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            min_delay: self.min_delay - rhs.min_delay,
            max_delay: self.max_delay - rhs.max_delay,
        }
    }
}

impl fmt::Debug for DelayPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DelayPair(min={}, max={})", self.min_delay, self.max_delay)
    }
}

impl fmt::Display for DelayPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.min_delay == self.max_delay {
            write!(f, "{}ps", self.min_delay)
        } else {
            write!(f, "{}-{}ps", self.min_delay, self.max_delay)
        }
    }
}

/// A quad of rise/fall delay pairs, capturing both min/max and rise/fall variations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DelayQuad {
    pub rise: DelayPair,
    pub fall: DelayPair,
}

impl DelayQuad {
    /// Create a new delay quad with explicit rise and fall pairs.
    #[inline]
    pub const fn new(rise: DelayPair, fall: DelayPair) -> Self {
        Self { rise, fall }
    }

    /// Create a delay quad where rise and fall are the same pair.
    #[inline]
    pub const fn uniform_pair(pair: DelayPair) -> Self {
        Self {
            rise: pair,
            fall: pair,
        }
    }

    /// Create a delay quad where all four values are the same.
    #[inline]
    pub const fn uniform(delay: DelayT) -> Self {
        let pair = DelayPair::uniform(delay);
        Self {
            rise: pair,
            fall: pair,
        }
    }

    /// Returns the minimum delay across all four values.
    #[inline]
    pub fn min_delay(self) -> DelayT {
        self.rise.min_delay.min(self.fall.min_delay)
    }

    /// Returns the maximum delay across all four values.
    #[inline]
    pub fn max_delay(self) -> DelayT {
        self.rise.max_delay.max(self.fall.max_delay)
    }

    /// Returns the overall min/max as a DelayPair.
    #[inline]
    pub fn as_delay_pair(self) -> DelayPair {
        DelayPair::new(self.min_delay(), self.max_delay())
    }
}

impl Add for DelayQuad {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            rise: self.rise + rhs.rise,
            fall: self.fall + rhs.fall,
        }
    }
}

impl fmt::Debug for DelayQuad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DelayQuad")
            .field("rise", &self.rise)
            .field("fall", &self.fall)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === DelayPair tests ===

    #[test]
    fn delay_pair_new() {
        let dp = DelayPair::new(100, 200);
        assert_eq!(dp.min_delay, 100);
        assert_eq!(dp.max_delay, 200);
    }

    #[test]
    fn delay_pair_uniform() {
        let dp = DelayPair::uniform(150);
        assert_eq!(dp.min_delay, 150);
        assert_eq!(dp.max_delay, 150);
    }

    #[test]
    fn delay_pair_default() {
        let dp = DelayPair::default();
        assert_eq!(dp.min_delay, 0);
        assert_eq!(dp.max_delay, 0);
    }

    #[test]
    fn delay_pair_average() {
        let dp = DelayPair::new(100, 200);
        assert_eq!(dp.average(), 150);
    }

    #[test]
    fn delay_pair_average_uniform() {
        let dp = DelayPair::uniform(300);
        assert_eq!(dp.average(), 300);
    }

    #[test]
    fn delay_pair_add() {
        let a = DelayPair::new(100, 200);
        let b = DelayPair::new(10, 20);
        let c = a + b;
        assert_eq!(c.min_delay, 110);
        assert_eq!(c.max_delay, 220);
    }

    #[test]
    fn delay_pair_sub() {
        let a = DelayPair::new(100, 200);
        let b = DelayPair::new(10, 20);
        let c = a - b;
        assert_eq!(c.min_delay, 90);
        assert_eq!(c.max_delay, 180);
    }

    #[test]
    fn delay_pair_equality() {
        let a = DelayPair::new(100, 200);
        let b = DelayPair::new(100, 200);
        let c = DelayPair::new(100, 300);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn delay_pair_copy() {
        let a = DelayPair::new(100, 200);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn delay_pair_debug() {
        let dp = DelayPair::new(100, 200);
        let s = format!("{:?}", dp);
        assert!(s.contains("min=100"));
        assert!(s.contains("max=200"));
    }

    #[test]
    fn delay_pair_display_uniform() {
        let dp = DelayPair::uniform(100);
        assert_eq!(format!("{}", dp), "100ps");
    }

    #[test]
    fn delay_pair_display_range() {
        let dp = DelayPair::new(100, 200);
        assert_eq!(format!("{}", dp), "100-200ps");
    }

    #[test]
    fn delay_pair_negative() {
        let dp = DelayPair::new(-50, -10);
        assert_eq!(dp.min_delay, -50);
        assert_eq!(dp.max_delay, -10);
        assert_eq!(dp.average(), -30);
    }

    // === DelayQuad tests ===

    #[test]
    fn delay_quad_new() {
        let rise = DelayPair::new(100, 200);
        let fall = DelayPair::new(150, 250);
        let dq = DelayQuad::new(rise, fall);
        assert_eq!(dq.rise, rise);
        assert_eq!(dq.fall, fall);
    }

    #[test]
    fn delay_quad_uniform() {
        let dq = DelayQuad::uniform(100);
        assert_eq!(dq.rise.min_delay, 100);
        assert_eq!(dq.rise.max_delay, 100);
        assert_eq!(dq.fall.min_delay, 100);
        assert_eq!(dq.fall.max_delay, 100);
    }

    #[test]
    fn delay_quad_uniform_pair() {
        let pair = DelayPair::new(100, 200);
        let dq = DelayQuad::uniform_pair(pair);
        assert_eq!(dq.rise, pair);
        assert_eq!(dq.fall, pair);
    }

    #[test]
    fn delay_quad_default() {
        let dq = DelayQuad::default();
        assert_eq!(dq.rise, DelayPair::default());
        assert_eq!(dq.fall, DelayPair::default());
    }

    #[test]
    fn delay_quad_min_delay() {
        let dq = DelayQuad::new(
            DelayPair::new(100, 200),
            DelayPair::new(50, 250),
        );
        assert_eq!(dq.min_delay(), 50);
    }

    #[test]
    fn delay_quad_max_delay() {
        let dq = DelayQuad::new(
            DelayPair::new(100, 200),
            DelayPair::new(50, 250),
        );
        assert_eq!(dq.max_delay(), 250);
    }

    #[test]
    fn delay_quad_as_delay_pair() {
        let dq = DelayQuad::new(
            DelayPair::new(100, 200),
            DelayPair::new(50, 250),
        );
        let dp = dq.as_delay_pair();
        assert_eq!(dp.min_delay, 50);
        assert_eq!(dp.max_delay, 250);
    }

    #[test]
    fn delay_quad_add() {
        let a = DelayQuad::new(
            DelayPair::new(100, 200),
            DelayPair::new(150, 250),
        );
        let b = DelayQuad::new(
            DelayPair::new(10, 20),
            DelayPair::new(15, 25),
        );
        let c = a + b;
        assert_eq!(c.rise.min_delay, 110);
        assert_eq!(c.rise.max_delay, 220);
        assert_eq!(c.fall.min_delay, 165);
        assert_eq!(c.fall.max_delay, 275);
    }

    #[test]
    fn delay_quad_copy() {
        let a = DelayQuad::uniform(42);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn delay_quad_debug() {
        let dq = DelayQuad::uniform(100);
        let s = format!("{:?}", dq);
        assert!(s.contains("DelayQuad"));
        assert!(s.contains("rise"));
        assert!(s.contains("fall"));
    }
}

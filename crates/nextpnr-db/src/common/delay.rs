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
        Self {
            min_delay,
            max_delay,
        }
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
        write!(
            f,
            "DelayPair(min={}, max={})",
            self.min_delay, self.max_delay
        )
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

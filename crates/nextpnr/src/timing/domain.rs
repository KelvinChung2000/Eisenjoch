//! Clock domain types for timing analysis.

use crate::types::{ClockEdge, DelayT, IdString};

/// A clock domain in the design.
///
/// Each sequential element belongs to a clock domain defined by its clock net
/// and the edge it is sensitive to. Combinational logic inherits domains from
/// the registers it connects.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClockDomain {
    /// Name of the clock net. `IdString::EMPTY` means unclocked.
    pub clock_net: IdString,
    /// Active edge (rising or falling).
    pub edge: ClockEdge,
    /// Clock period in picoseconds. 0 means unconstrained.
    pub period: DelayT,
}

impl ClockDomain {
    /// Create an unclocked domain (for combinational-only paths).
    pub fn unclocked() -> Self {
        Self {
            clock_net: IdString::EMPTY,
            edge: ClockEdge::Rising,
            period: 0,
        }
    }

    /// Returns `true` if this domain has a valid clock (is not unclocked).
    pub fn is_clocked(&self) -> bool {
        !self.clock_net.is_empty()
    }
}

impl Default for ClockDomain {
    fn default() -> Self {
        Self::unclocked()
    }
}

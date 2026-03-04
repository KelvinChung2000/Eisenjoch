//! Clock domain types for timing analysis.

use npnr_types::{ClockEdge, DelayT, IdString};

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

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_types::IdStringPool;

    #[test]
    fn unclocked_domain() {
        let d = ClockDomain::unclocked();
        assert!(!d.is_clocked());
        assert!(d.clock_net.is_empty());
        assert_eq!(d.period, 0);
        assert_eq!(d.edge, ClockEdge::Rising);
    }

    #[test]
    fn default_is_unclocked() {
        let d = ClockDomain::default();
        assert!(!d.is_clocked());
    }

    #[test]
    fn clocked_domain() {
        let pool = IdStringPool::new();
        let clk = pool.intern("sys_clk");
        let d = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Falling,
            period: 5000,
        };
        assert!(d.is_clocked());
        assert_eq!(d.period, 5000);
        assert_eq!(d.edge, ClockEdge::Falling);
    }

    #[test]
    fn equality() {
        let pool = IdStringPool::new();
        let clk = pool.intern("clk");
        let a = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 10_000,
        };
        let b = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 10_000,
        };
        let c = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Falling,
            period: 10_000,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn hashing() {
        use std::collections::HashSet;
        let pool = IdStringPool::new();
        let clk = pool.intern("clk");
        let d1 = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 10_000,
        };
        let d2 = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 10_000,
        };
        let d3 = ClockDomain::unclocked();

        let mut set = HashSet::new();
        set.insert(d1);
        set.insert(d2);
        set.insert(d3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn clone() {
        let pool = IdStringPool::new();
        let clk = pool.intern("fast_clk");
        let d = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 2000,
        };
        let d2 = d.clone();
        assert_eq!(d, d2);
    }
}

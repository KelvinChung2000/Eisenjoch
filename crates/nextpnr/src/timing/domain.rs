//! Clock domain types and registry for timing analysis.

use crate::common::IdString;
use crate::timing::{ClockEdge, DelayQuad, DelayT};
use rustc_hash::FxHashMap;

/// Unique identifier for a clock domain within the timing analyser.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockDomainId(pub u32);

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

/// A (launch, capture) clock domain pair for cross-domain analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockDomainPair {
    pub launch: ClockDomainId,
    pub capture: ClockDomainId,
}

/// Registry that assigns unique IDs to clock domains.
pub struct DomainRegistry {
    domains: Vec<ClockDomain>,
    domain_map: FxHashMap<(IdString, ClockEdge), ClockDomainId>,
    /// Special domain ID for unclocked / async paths.
    pub async_domain: ClockDomainId,
}

impl DomainRegistry {
    pub fn new() -> Self {
        let async_domain = ClockDomain::unclocked();
        let async_id = ClockDomainId(0);
        let mut domain_map = FxHashMap::default();
        domain_map.insert((IdString::EMPTY, ClockEdge::Rising), async_id);
        Self {
            domains: vec![async_domain],
            domain_map,
            async_domain: async_id,
        }
    }

    /// Get or create a domain ID for the given clock net and edge.
    pub fn domain_id(&mut self, clock_net: IdString, edge: ClockEdge, period: DelayT) -> ClockDomainId {
        let key = (clock_net, edge);
        if let Some(&id) = self.domain_map.get(&key) {
            return id;
        }
        let id = ClockDomainId(self.domains.len() as u32);
        self.domains.push(ClockDomain {
            clock_net,
            edge,
            period,
        });
        self.domain_map.insert(key, id);
        id
    }

    /// Look up a domain by ID.
    pub fn get(&self, id: ClockDomainId) -> &ClockDomain {
        &self.domains[id.0 as usize]
    }

    /// Number of registered domains.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    /// Iterate over all (id, domain) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (ClockDomainId, &ClockDomain)> {
        self.domains
            .iter()
            .enumerate()
            .map(|(i, d)| (ClockDomainId(i as u32), d))
    }
}

impl Default for DomainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of a cached cell timing arc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellArcType {
    /// Combinational path through cell (input -> output).
    Combinational,
    /// Setup time arc (data input -> clock port).
    Setup,
    /// Hold time arc (data input -> clock port).
    Hold,
    /// Clock-to-Q arc (clock port -> data output).
    ClockToQ,
}

/// A cached cell timing arc, reducing repeated chipdb lookups.
///
/// Follows the C++ `CellArc` pattern from timing.h.
#[derive(Clone, Debug)]
pub struct CellArc {
    /// Type of timing relationship.
    pub arc_type: CellArcType,
    /// The other port involved in this arc.
    /// For input ports: the output port (Combinational) or clock port (Setup/Hold).
    /// For output ports: the input port (Combinational) or clock port (ClockToQ).
    pub other_port: IdString,
    /// Delay value for this arc.
    pub value: DelayQuad,
    /// Clock edge (only meaningful for Setup/Hold/ClockToQ arcs).
    pub edge: ClockEdge,
}

impl CellArc {
    pub fn combinational(other_port: IdString, delay: DelayQuad) -> Self {
        Self {
            arc_type: CellArcType::Combinational,
            other_port,
            value: delay,
            edge: ClockEdge::Rising,
        }
    }

    pub fn setup(clock_port: IdString, delay: DelayQuad, edge: ClockEdge) -> Self {
        Self {
            arc_type: CellArcType::Setup,
            other_port: clock_port,
            value: delay,
            edge,
        }
    }

    pub fn hold(clock_port: IdString, delay: DelayQuad, edge: ClockEdge) -> Self {
        Self {
            arc_type: CellArcType::Hold,
            other_port: clock_port,
            value: delay,
            edge,
        }
    }

    pub fn clock_to_q(clock_port: IdString, delay: DelayQuad, edge: ClockEdge) -> Self {
        Self {
            arc_type: CellArcType::ClockToQ,
            other_port: clock_port,
            value: delay,
            edge,
        }
    }
}

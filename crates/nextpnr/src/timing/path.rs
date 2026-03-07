//! Timing path and report types.

use super::domain::ClockDomain;
use crate::netlist::{CellId, NetId};
use crate::types::{DelayPair, DelayT, IdString, TimingPortClass};

/// Classification of a cell port for timing analysis.
///
/// Describes the timing characteristics of a port: whether it is combinational,
/// a register input/output, or a clock port, along with associated delays.
#[derive(Clone, Debug)]
pub struct TimingPortInfo {
    /// Timing class of this port.
    pub port_class: TimingPortClass,
    /// Clock domain this port belongs to (for sequential ports).
    pub clock_domain: ClockDomain,
    /// Setup time constraint in picoseconds (register inputs only).
    pub setup: DelayT,
    /// Hold time constraint in picoseconds (register inputs only).
    pub hold: DelayT,
    /// Clock-to-output delay (register outputs only).
    pub clock_to_out: DelayPair,
    /// Combinational delay from this port to the related output.
    pub comb_delay: DelayPair,
}

impl Default for TimingPortInfo {
    fn default() -> Self {
        Self {
            port_class: TimingPortClass::Combinational,
            clock_domain: ClockDomain::unclocked(),
            setup: 0,
            hold: 0,
            clock_to_out: DelayPair::default(),
            comb_delay: DelayPair::default(),
        }
    }
}

/// A timing endpoint (source or destination of a timing path).
#[derive(Clone, Debug)]
pub struct TimingEndpoint {
    /// Cell containing this endpoint.
    pub cell: CellId,
    /// Port name on the cell.
    pub port: IdString,
    /// Clock domain of this endpoint.
    pub domain: ClockDomain,
}

/// A segment along a timing path (one net + cell traversal).
#[derive(Clone, Debug)]
pub struct PathSegment {
    /// Net traversed in this segment.
    pub net: NetId,
    /// Destination cell of this segment.
    pub cell: CellId,
    /// Destination port name.
    pub port: IdString,
    /// Delay contribution of this segment in picoseconds.
    pub delay: DelayT,
}

/// A complete timing path from source to destination.
#[derive(Clone, Debug)]
pub struct TimingPath {
    /// Source endpoint (e.g., register output or primary input).
    pub from: TimingEndpoint,
    /// Destination endpoint (e.g., register input or primary output).
    pub to: TimingEndpoint,
    /// Total path delay in picoseconds.
    pub delay: DelayT,
    /// Timing budget (available time from clock constraint).
    pub budget: DelayT,
    /// Slack = budget - delay. Negative means timing violation.
    pub slack: DelayT,
    /// Detailed segments along the path (may be empty for summary paths).
    pub segments: Vec<PathSegment>,
}

/// Summary report of timing analysis results.
#[derive(Clone, Debug)]
pub struct TimingReport {
    /// Maximum achievable frequency in MHz.
    pub fmax: f64,
    /// Worst negative slack in picoseconds.
    pub worst_slack: DelayT,
    /// Number of endpoints with negative slack (failing timing).
    pub num_failing: usize,
    /// Total number of timing endpoints analysed.
    pub num_endpoints: usize,
    /// Critical paths sorted by slack (ascending = worst first).
    pub critical_paths: Vec<TimingPath>,
}

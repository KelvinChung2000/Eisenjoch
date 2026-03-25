//! Core struct definitions and lifecycle methods for the timing analyser.

use rustc_hash::FxHashMap;

use super::constraints::SdcConstraints;
use super::delay::{DelayPair, DelayT};
use super::domain::{ClockDomain, ClockDomainPair, DomainRegistry};
use super::kinds::TimingPortClass;
use super::path::TimingPath;
use crate::common::IdString;
use crate::netlist::{CellPin, NetId, PortType};

/// Default combinational cell delay in picoseconds when no chipdb data is available.
pub(super) const DEFAULT_COMB_DELAY: DelayT = 100;

// ---------------------------------------------------------------------------
// Per-port timing data
// ---------------------------------------------------------------------------

/// Cached timing data for a single cell port.
pub(super) struct PerPort {
    /// Port direction.
    pub(super) port_type: PortType,
    /// Timing classification from chipdb.
    pub(super) port_class: TimingPortClass,
    /// Cell timing arcs involving this port.
    pub(super) cell_arcs: Vec<super::domain::CellArc>,
    /// Routing delay into this port (input ports only).
    pub(super) route_delay: DelayPair,
}

impl PerPort {
    pub(super) fn new(port_type: PortType) -> Self {
        Self {
            port_type,
            port_class: TimingPortClass::Combinational,
            cell_arcs: Vec::new(),
            route_delay: DelayPair::default(),
        }
    }
}

/// Per-domain startpoint/endpoint tracking.
pub(super) struct PerDomain {
    /// (signal_port, clock_port) pairs that are startpoints in this domain.
    pub(super) startpoints: Vec<(CellPin, IdString)>,
    /// (signal_port, clock_port) pairs that are endpoints in this domain.
    pub(super) endpoints: Vec<(CellPin, IdString)>,
}

impl PerDomain {
    pub(super) fn new() -> Self {
        Self {
            startpoints: Vec::new(),
            endpoints: Vec::new(),
        }
    }
}

/// Per-domain-pair result data.
pub(super) struct PerDomainPair {
    pub(super) pair: ClockDomainPair,
    /// Period for this domain pair (adjusted for edge relationships).
    pub(super) period: DelayPair,
    pub(super) worst_setup_slack: DelayT,
    pub(super) worst_hold_slack: DelayT,
}

// ---------------------------------------------------------------------------
// TimingAnalyser
// ---------------------------------------------------------------------------

/// Static timing analyser.
///
/// Performs forward (arrival-time) and backward (required-time) propagation
/// through a design netlist, then computes slack and criticality for every net.
pub struct TimingAnalyser {
    /// Net criticality values (0.0 = not critical, 1.0 = most critical).
    pub(super) net_criticality: FxHashMap<NetId, f32>,
    /// Port arrival times (forward pass): CellPin -> arrival time.
    pub(super) arrival_times: FxHashMap<CellPin, DelayT>,
    /// Port required times (backward pass): CellPin -> required time.
    pub(super) required_times: FxHashMap<CellPin, DelayT>,
    /// Computed timing paths, sorted by slack (ascending = worst first).
    pub(super) paths: Vec<TimingPath>,
    /// Clock domain constraints: clock net name -> period in picoseconds.
    pub(super) clock_constraints: FxHashMap<IdString, DelayT>,
    /// Worst negative slack across all endpoints (most negative = worst).
    pub(super) worst_slack: DelayT,
    /// Worst setup slack across all endpoints.
    pub(super) worst_setup_slack: DelayT,
    /// Worst hold slack across all endpoints.
    pub(super) worst_hold_slack: DelayT,
    /// Whether timing has been computed and is up-to-date.
    pub(super) is_valid: bool,
    /// Enable clock skew analysis (routing delay on clock pins).
    pub with_clock_skew: bool,

    // -- Domain infrastructure --
    /// Clock domain registry: assigns unique IDs to (clock_net, edge) pairs.
    pub(super) domain_registry: DomainRegistry,
    /// Per-port cached timing data.
    pub(super) port_data: FxHashMap<CellPin, PerPort>,
    /// Per-domain data (startpoints/endpoints).
    pub(super) per_domain: Vec<PerDomain>,
    /// Per-domain-pair data.
    pub(super) domain_pairs: Vec<PerDomainPair>,
    /// Map from (launch, capture) domain IDs to domain pair index.
    pub(super) pair_to_id: FxHashMap<ClockDomainPair, usize>,
    /// Clock-to-clock delays for related domains.
    pub(super) clock_delays: FxHashMap<(IdString, IdString), DelayT>,
    /// SDC constraints.
    pub sdc: SdcConstraints,
    /// Topological order of cell ports (cached).
    pub(super) topological_order: Vec<CellPin>,
    /// Legacy: per-port domain assignment (used by heuristic classify_ports).
    pub(super) legacy_port_domains: FxHashMap<CellPin, ClockDomain>,
    /// Predecessor map for path reconstruction: pin -> (source_pin, net_id, delay_contribution).
    pub(super) predecessors: FxHashMap<CellPin, (CellPin, Option<NetId>, DelayT)>,
    /// Maximum number of critical paths to keep per analysis.
    pub(super) max_critical_paths: usize,
}

impl TimingAnalyser {
    /// Create a new, empty timing analyser.
    pub fn new() -> Self {
        Self {
            net_criticality: FxHashMap::default(),
            arrival_times: FxHashMap::default(),
            required_times: FxHashMap::default(),
            paths: Vec::new(),
            clock_constraints: FxHashMap::default(),
            worst_slack: 0,
            worst_setup_slack: DelayT::MAX,
            worst_hold_slack: DelayT::MAX,
            is_valid: false,
            with_clock_skew: false,
            domain_registry: DomainRegistry::new(),
            port_data: FxHashMap::default(),
            per_domain: vec![PerDomain::new()], // index 0 = async domain
            domain_pairs: Vec::new(),
            pair_to_id: FxHashMap::default(),
            clock_delays: FxHashMap::default(),
            sdc: SdcConstraints::new(),
            topological_order: Vec::new(),
            legacy_port_domains: FxHashMap::default(),
            predecessors: FxHashMap::default(),
            max_critical_paths: 10,
        }
    }

    /// Clear all internal state.
    pub(super) fn clear_all(&mut self) {
        self.arrival_times.clear();
        self.required_times.clear();
        self.net_criticality.clear();
        self.paths.clear();
        self.port_data.clear();
        self.domain_registry = DomainRegistry::new();
        self.per_domain = vec![PerDomain::new()];
        self.domain_pairs.clear();
        self.pair_to_id.clear();
        self.clock_delays.clear();
        self.topological_order.clear();
        self.legacy_port_domains.clear();
        self.predecessors.clear();
        self.worst_slack = 0;
        self.worst_setup_slack = DelayT::MAX;
        self.worst_hold_slack = DelayT::MAX;
    }

    /// Invalidate timing results (must re-analyse after design changes).
    pub fn invalidate(&mut self) {
        self.is_valid = false;
    }

    /// Check if timing is valid.
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
}

impl Default for TimingAnalyser {
    fn default() -> Self {
        Self::new()
    }
}

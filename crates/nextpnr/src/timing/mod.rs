//! Static timing analysis engine for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module performs static timing analysis (STA) on a placed design to determine
//! whether it meets frequency constraints and to provide criticality values used by
//! timing-driven placement and routing.
//!
//! The central type is [`TimingAnalyser`], which performs forward and backward
//! propagation through the netlist to compute arrival times, required times, slack,
//! and criticality for every net.

pub mod constraints;
mod delay;
mod domain;
mod kinds;
mod path;
pub mod report;
pub mod sort;

pub use constraints::SdcConstraints;
pub use delay::{DelayPair, DelayQuad, DelayT};
pub use domain::{
    CellArc, CellArcType, ClockDomain, ClockDomainId, ClockDomainPair, DomainRegistry,
};
pub use kinds::{ClockEdge, TimingPortClass};
pub use path::{PathSegment, TimingEndpoint, TimingPath, TimingPortInfo, TimingReport};
pub use report::{
    format_constraint_coverage, format_cross_domain_report, format_path_detail, TimingSummary,
};
pub use sort::topological_sort;

use crate::chipdb::ChipDb;
use crate::common::{IdString, IdStringPool};
use crate::context::Context;
use crate::netlist::{CellId, CellPin, Design, NetId};
use crate::netlist::PortType;
use log::debug;
use rustc_hash::{FxHashMap, FxHashSet};

/// Default combinational cell delay in picoseconds when no chipdb data is available.
const DEFAULT_COMB_DELAY: DelayT = 100;

// ---------------------------------------------------------------------------
// Per-port timing data
// ---------------------------------------------------------------------------

/// Cached timing data for a single cell port.
struct PerPort {
    /// Port direction.
    port_type: PortType,
    /// Timing classification from chipdb.
    port_class: TimingPortClass,
    /// Cell timing arcs involving this port.
    cell_arcs: Vec<CellArc>,
    /// Routing delay into this port (input ports only).
    route_delay: DelayPair,
}

impl PerPort {
    fn new(port_type: PortType) -> Self {
        Self {
            port_type,
            port_class: TimingPortClass::Combinational,
            cell_arcs: Vec::new(),
            route_delay: DelayPair::default(),
        }
    }
}

/// Per-domain startpoint/endpoint tracking.
struct PerDomain {
    /// (signal_port, clock_port) pairs that are startpoints in this domain.
    startpoints: Vec<(CellPin, IdString)>,
    /// (signal_port, clock_port) pairs that are endpoints in this domain.
    endpoints: Vec<(CellPin, IdString)>,
}

impl PerDomain {
    fn new() -> Self {
        Self {
            startpoints: Vec::new(),
            endpoints: Vec::new(),
        }
    }
}

/// Per-domain-pair result data.
struct PerDomainPair {
    pair: ClockDomainPair,
    /// Period for this domain pair (adjusted for edge relationships).
    period: DelayPair,
    worst_setup_slack: DelayT,
    worst_hold_slack: DelayT,
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
    net_criticality: FxHashMap<NetId, f32>,
    /// Port arrival times (forward pass): CellPin -> arrival time.
    arrival_times: FxHashMap<CellPin, DelayT>,
    /// Port required times (backward pass): CellPin -> required time.
    required_times: FxHashMap<CellPin, DelayT>,
    /// Computed timing paths, sorted by slack (ascending = worst first).
    paths: Vec<TimingPath>,
    /// Clock domain constraints: clock net name -> period in picoseconds.
    clock_constraints: FxHashMap<IdString, DelayT>,
    /// Worst negative slack across all endpoints (most negative = worst).
    worst_slack: DelayT,
    /// Worst setup slack across all endpoints.
    worst_setup_slack: DelayT,
    /// Worst hold slack across all endpoints.
    worst_hold_slack: DelayT,
    /// Whether timing has been computed and is up-to-date.
    is_valid: bool,
    /// Enable clock skew analysis (routing delay on clock pins).
    pub with_clock_skew: bool,

    // -- Domain infrastructure (Task 2) --
    /// Clock domain registry: assigns unique IDs to (clock_net, edge) pairs.
    domain_registry: DomainRegistry,
    /// Per-port cached timing data.
    port_data: FxHashMap<CellPin, PerPort>,
    /// Per-domain data (startpoints/endpoints).
    per_domain: Vec<PerDomain>,
    /// Per-domain-pair data.
    domain_pairs: Vec<PerDomainPair>,
    /// Map from (launch, capture) domain IDs to domain pair index.
    pair_to_id: FxHashMap<ClockDomainPair, usize>,
    /// Clock-to-clock delays for related domains.
    clock_delays: FxHashMap<(IdString, IdString), DelayT>,
    /// SDC constraints.
    pub sdc: SdcConstraints,
    /// Topological order of cell ports (cached).
    topological_order: Vec<CellPin>,
    /// Legacy: per-port domain assignment (used by heuristic classify_ports).
    legacy_port_domains: FxHashMap<CellPin, ClockDomain>,
    /// Predecessor map for path reconstruction: pin -> (source_pin, net_id, delay_contribution).
    predecessors: FxHashMap<CellPin, (CellPin, Option<NetId>, DelayT)>,
    /// Maximum number of critical paths to keep per analysis.
    max_critical_paths: usize,
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

    /// Add a clock constraint given a frequency in MHz.
    ///
    /// Converts frequency to period: period_ps = 1_000_000 / freq_mhz.
    pub fn add_clock_constraint(&mut self, clock_net: IdString, freq_mhz: f64) {
        let period_ps = (1_000_000.0 / freq_mhz) as DelayT;
        self.add_clock_constraint_ps(clock_net, period_ps);
    }

    /// Add a clock constraint directly in picoseconds.
    pub fn add_clock_constraint_ps(&mut self, clock_net: IdString, period_ps: DelayT) {
        self.clock_constraints.insert(clock_net, period_ps);
        self.is_valid = false;
    }

    /// Full setup + run: init ports, get cell delays, topo sort, domain setup, then analyse.
    ///
    /// This is the primary entry point. Takes `&Context` to access chipdb timing data.
    pub fn setup_and_run(&mut self, ctx: &Context) {
        self.clear_all();
        self.init_ports(&ctx.design);
        self.get_cell_delays(ctx);
        self.topo_sort_ports(&ctx.design);
        self.setup_port_domains(ctx);
        self.identify_related_domains(ctx);
        self.run(ctx);
    }

    /// Re-run timing with existing port/domain setup (e.g. after placement changes).
    pub fn run(&mut self, ctx: &Context) {
        self.arrival_times.clear();
        self.required_times.clear();
        self.net_criticality.clear();
        self.paths.clear();
        self.predecessors.clear();
        self.worst_slack = 0;
        self.worst_setup_slack = DelayT::MAX;
        self.worst_hold_slack = DelayT::MAX;
        // Reset domain pair slacks.
        for dp in &mut self.domain_pairs {
            dp.worst_setup_slack = DelayT::MAX;
            dp.worst_hold_slack = DelayT::MAX;
        }

        self.get_route_delays(ctx);
        self.forward_propagation(&ctx.design);
        self.backward_propagation(&ctx.design);
        self.compute_slack_and_paths(&ctx.design);
        self.compute_criticality(&ctx.design);

        self.paths.sort_by_key(|p| p.slack);
        self.is_valid = true;
    }

    /// Legacy entry point: analyse using only Design + IdStringPool (no chipdb timing).
    ///
    /// Uses heuristic port classification and DEFAULT_COMB_DELAY.
    /// Prefer `setup_and_run(&Context)` when chipdb is available.
    pub fn analyse(&mut self, design: &Design, _id_pool: &IdStringPool) {
        self.clear_all();

        // Legacy: classify ports heuristically.
        self.classify_ports_heuristic(design);

        let sorted_cells = topological_sort(design);
        debug!("Topological sort: {} cells", sorted_cells.len());

        self.forward_propagation_legacy(design, &sorted_cells);
        self.backward_propagation_legacy(design, &sorted_cells);
        self.compute_slack_and_paths(design);
        self.compute_criticality(design);

        self.paths.sort_by_key(|p| p.slack);
        self.is_valid = true;
    }

    /// Get criticality of a net (0.0 to 1.0).
    pub fn net_criticality(&self, net: NetId) -> f32 {
        self.net_criticality.get(&net).copied().unwrap_or(0.0)
    }

    /// Get criticality of a specific port.
    pub fn port_criticality(&self, cell: CellId, port: IdString) -> f32 {
        if self.worst_slack >= 0 {
            return 0.0;
        }

        let pin = CellPin::new(cell, port);
        let arrival = self.arrival_times.get(&pin).copied().unwrap_or(0);
        let required = self.required_times.get(&pin).copied().unwrap_or(0);
        let slack = required - arrival;

        let neg_ws = -self.worst_slack as f64;
        let crit = 1.0 - ((slack - self.worst_slack) as f64 / neg_ws);
        crit.clamp(0.0, 1.0) as f32
    }

    /// Get worst negative slack across all endpoints.
    pub fn worst_slack(&self) -> DelayT {
        self.worst_slack
    }

    /// Get worst setup slack across all endpoints.
    pub fn worst_setup_slack(&self) -> DelayT {
        if self.worst_setup_slack == DelayT::MAX {
            0
        } else {
            self.worst_setup_slack
        }
    }

    /// Get worst hold slack across all endpoints.
    pub fn worst_hold_slack(&self) -> DelayT {
        if self.worst_hold_slack == DelayT::MAX {
            0
        } else {
            self.worst_hold_slack
        }
    }

    /// Get the N most critical paths (sorted by slack, ascending = worst first).
    pub fn critical_paths(&self, limit: usize) -> &[TimingPath] {
        let n = limit.min(self.paths.len());
        &self.paths[..n]
    }

    /// Set maximum number of critical paths to retain.
    pub fn set_max_critical_paths(&mut self, n: usize) {
        self.max_critical_paths = n;
    }

    // =====================================================================
    // Path query API (Task 6)
    // =====================================================================

    /// Get arrival time at a specific pin.
    pub fn pin_arrival(&self, cell: CellId, port: IdString) -> Option<DelayT> {
        self.arrival_times.get(&CellPin::new(cell, port)).copied()
    }

    /// Get required time at a specific pin.
    pub fn pin_required(&self, cell: CellId, port: IdString) -> Option<DelayT> {
        self.required_times.get(&CellPin::new(cell, port)).copied()
    }

    /// Get slack at a specific endpoint pin.
    pub fn endpoint_slack(&self, cell: CellId, port: IdString) -> Option<DelayT> {
        let pin = CellPin::new(cell, port);
        let arr = self.arrival_times.get(&pin)?;
        let req = self.required_times.get(&pin)?;
        Some(req - arr)
    }

    /// Get all timing paths passing through a given net.
    pub fn paths_through_net(&self, net: NetId) -> Vec<&TimingPath> {
        self.paths
            .iter()
            .filter(|p| p.segments.iter().any(|s| s.net == net))
            .collect()
    }

    /// Re-run timing after placement changes (uses estimated wire delays).
    pub fn update_after_placement(&mut self, ctx: &Context) {
        self.run(ctx);
    }

    /// Re-run timing after routing (uses actual routed delays).
    pub fn update_after_routing(&mut self, ctx: &Context) {
        self.run(ctx);
    }

    /// Compute Fmax from worst slack and clock period.
    pub fn fmax_mhz(&self) -> f64 {
        if self.clock_constraints.is_empty() {
            return 0.0;
        }
        let min_period = self.clock_constraints.values().copied().min().unwrap_or(0);
        if min_period <= 0 {
            return 0.0;
        }
        let effective_period = if self.worst_slack < 0 {
            min_period + self.worst_slack
        } else {
            min_period
        };
        if effective_period <= 0 {
            return 0.0;
        }
        1_000_000.0 / effective_period as f64
    }

    /// Get a timing report summarizing the analysis results.
    pub fn report(&self) -> TimingReport {
        let num_failing = self.paths.iter().filter(|p| p.slack < 0).count();
        let num_endpoints = self.paths.len();
        TimingReport {
            fmax: self.fmax_mhz(),
            worst_slack: self.worst_slack,
            num_failing,
            num_endpoints,
            critical_paths: self.paths.clone(),
        }
    }

    /// Invalidate timing results (must re-analyse after design changes).
    pub fn invalidate(&mut self) {
        self.is_valid = false;
    }

    /// Check if timing is valid.
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// Access the domain registry.
    pub fn domain_registry(&self) -> &DomainRegistry {
        &self.domain_registry
    }

    /// Access clock-to-clock delays.
    pub fn clock_delays(&self) -> &FxHashMap<(IdString, IdString), DelayT> {
        &self.clock_delays
    }

    /// Get or create a domain pair ID.
    fn domain_pair_id(&mut self, launch: ClockDomainId, capture: ClockDomainId) -> usize {
        let pair = ClockDomainPair { launch, capture };
        if let Some(&id) = self.pair_to_id.get(&pair) {
            return id;
        }
        let id = self.domain_pairs.len();
        self.domain_pairs.push(PerDomainPair {
            pair,
            period: DelayPair::default(),
            worst_setup_slack: DelayT::MAX,
            worst_hold_slack: DelayT::MAX,
        });
        self.pair_to_id.insert(pair, id);
        id
    }

    // =====================================================================
    // Internal: Route delay update (C++ get_route_delays)
    // =====================================================================

    /// Update route delays for input ports from the routed design.
    fn get_route_delays(&mut self, ctx: &Context) {
        for (net_idx, net) in ctx.design.iter_alive_nets() {
            if !net.driver.is_connected() {
                continue;
            }
            let driver_cell = net.driver.cell;
            if ctx.design.cell(driver_cell).bel.is_none() {
                continue;
            }
            for user in &net.users {
                if !user.is_valid() {
                    continue;
                }
                let user_cell = user.cell;
                if ctx.design.cell(user_cell).bel.is_none() {
                    continue;
                }
                let pin = CellPin::new(user_cell, user.port);
                if let Some(pd) = self.port_data.get_mut(&pin) {
                    // Use net delay estimate. With full routing, this would be
                    // the actual routed delay. For now, use estimate_delay.
                    let delay = ctx.estimate_delay_for_net(net_idx);
                    pd.route_delay = DelayPair::uniform(delay);
                }
            }
        }
    }

    // =====================================================================
    // Internal: Clear state
    // =====================================================================

    fn clear_all(&mut self) {
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

    // =====================================================================
    // Internal: Port initialization (C++ init_ports)
    // =====================================================================

    /// Initialize per-port structures from the design netlist.
    fn init_ports(&mut self, design: &Design) {
        for (cell_idx, cell) in design.iter_alive_cells() {
            for (port_name, port_info) in &cell.ports {
                let pin = CellPin::new(cell_idx, *port_name);
                self.port_data.insert(pin, PerPort::new(port_info.port_type()));
            }
        }
    }

    // =====================================================================
    // Internal: Cell delay caching (C++ get_cell_delays)
    // =====================================================================

    /// Cache all cell timing arcs from chipdb, following the C++ pattern.
    fn get_cell_delays(&mut self, ctx: &Context) {
        let speed_grade = match ctx.speed_grade() {
            Some(sg) => sg,
            None => {
                self.get_cell_delays_heuristic(&ctx.design);
                return;
            }
        };

        // Collect port pins first to avoid borrow conflicts.
        let port_pins: Vec<(CellPin, PortType)> = self
            .port_data
            .iter()
            .map(|(pin, pd)| (*pin, pd.port_type))
            .collect();

        for (pin, port_type) in port_pins {
            let cell = ctx.design.cell(pin.cell);
            let port_info = match cell.ports.get(&pin.port) {
                Some(pi) => pi,
                None => continue,
            };

            // Skip dangling ports.
            if port_info.net().is_none() {
                continue;
            }

            // Get cell timing index from chipdb.
            let type_idx = match cell
                .timing_index
                .map(|ti| ti.0 as usize)
                .or_else(|| {
                    ctx.chipdb()
                        .cell_timing_index(speed_grade, cell.cell_type.index())
                })
            {
                Some(idx) => idx,
                None => continue,
            };

            let port_class = ctx.chipdb().port_timing_class(
                speed_grade,
                type_idx,
                pin.port.index(),
                port_type,
            );

            let mut arcs = Vec::new();

            match port_type {
                PortType::In => {
                    if port_class == TimingPortClass::ClockInput
                        || port_class == TimingPortClass::GenClock
                        || port_class == TimingPortClass::Ignore
                    {
                        // No arcs for clock/ignore ports.
                    } else {
                        // Register inputs have setup/hold arcs.
                        if port_class == TimingPortClass::RegisterInput {
                            if let Some(reg_arc_pods) =
                                ctx.chipdb().cell_reg_arcs(speed_grade, type_idx, pin.port.index())
                            {
                                for arc_pod in reg_arc_pods {
                                    let info = ChipDb::reg_arc_info(arc_pod);
                                    let clock_port = IdString(info.clock_port);
                                    // Check clock port is connected.
                                    if cell.ports.get(&clock_port).and_then(|p| p.net()).is_none() {
                                        continue;
                                    }
                                    arcs.push(CellArc::setup(
                                        clock_port,
                                        DelayQuad::uniform_pair(info.setup),
                                        info.edge,
                                    ));
                                    arcs.push(CellArc::hold(
                                        clock_port,
                                        DelayQuad::uniform_pair(info.hold),
                                        info.edge,
                                    ));
                                }
                            }
                        }
                        // Combinational arcs: input -> output.
                        for (other_name, other_port) in &cell.ports {
                            if other_port.port_type() != PortType::Out || other_port.net().is_none()
                            {
                                continue;
                            }
                            if let Some(delay) = ctx.chipdb().cell_delay(
                                speed_grade,
                                type_idx,
                                pin.port.index(),
                                other_name.index(),
                            ) {
                                arcs.push(CellArc::combinational(*other_name, delay));
                            }
                        }
                    }
                }
                PortType::Out | PortType::InOut => {
                    if port_class == TimingPortClass::ClockInput
                        || port_class == TimingPortClass::GenClock
                        || port_class == TimingPortClass::Ignore
                    {
                        // No arcs for these classes.
                    } else {
                        // Register outputs have clock-to-Q arcs.
                        if port_class == TimingPortClass::RegisterOutput {
                            if let Some(reg_arc_pods) =
                                ctx.chipdb().cell_reg_arcs(speed_grade, type_idx, pin.port.index())
                            {
                                for arc_pod in reg_arc_pods {
                                    let info = ChipDb::reg_arc_info(arc_pod);
                                    let clock_port = IdString(info.clock_port);
                                    if cell.ports.get(&clock_port).and_then(|p| p.net()).is_none() {
                                        continue;
                                    }
                                    arcs.push(CellArc::clock_to_q(
                                        clock_port,
                                        info.clock_to_q,
                                        info.edge,
                                    ));
                                }
                            }
                        }
                        // Combinational arcs: output <- input.
                        for (other_name, other_port) in &cell.ports {
                            if other_port.port_type() != PortType::In || other_port.net().is_none()
                            {
                                continue;
                            }
                            if let Some(delay) = ctx.chipdb().cell_delay(
                                speed_grade,
                                type_idx,
                                other_name.index(),
                                pin.port.index(),
                            ) {
                                arcs.push(CellArc::combinational(*other_name, delay));
                            }
                        }
                    }
                }
            }

            if let Some(pd) = self.port_data.get_mut(&pin) {
                pd.port_class = port_class;
                pd.cell_arcs = arcs;
            }
        }
    }

    /// Fallback: heuristic cell delay caching when no chipdb is available.
    fn get_cell_delays_heuristic(&mut self, design: &Design) {
        for (cell_idx, cell) in design.iter_alive_cells() {
            let mut clock_ports: FxHashSet<IdString> = FxHashSet::default();

            // Find clock ports.
            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::In {
                    continue;
                }
                let Some(net_idx) = port_info.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                let has_clock = net.clock_constraint > 0
                    || self.clock_constraints.get(&net.name).is_some_and(|&p| p > 0);
                if has_clock {
                    clock_ports.insert(*port_name);
                }
            }
            let is_sequential = !clock_ports.is_empty();

            for (port_name, port_info) in &cell.ports {
                let pin = CellPin::new(cell_idx, *port_name);
                let port_class = if clock_ports.contains(port_name) {
                    TimingPortClass::ClockInput
                } else if is_sequential && port_info.port_type() == PortType::In {
                    TimingPortClass::RegisterInput
                } else if is_sequential && port_info.port_type() == PortType::Out {
                    TimingPortClass::RegisterOutput
                } else {
                    TimingPortClass::Combinational
                };

                if let Some(pd) = self.port_data.get_mut(&pin) {
                    pd.port_class = port_class;
                }
            }
        }
    }

    // =====================================================================
    // Internal: Topological sort of ports (C++ topo_sort)
    // =====================================================================

    /// Topological sort at the port level (not cell level).
    fn topo_sort_ports(&mut self, design: &Design) {
        // Build a port-level DAG.
        let pins: Vec<CellPin> = self.port_data.keys().copied().collect();
        let mut in_degree: FxHashMap<CellPin, usize> = FxHashMap::default();
        let mut edges: FxHashMap<CellPin, Vec<CellPin>> = FxHashMap::default();

        for &pin in &pins {
            in_degree.entry(pin).or_insert(0);
        }

        for &pin in &pins {
            let pd = &self.port_data[&pin];
            if pd.port_type == PortType::In {
                // Input port: combinational arcs to output ports on same cell.
                for arc in &pd.cell_arcs {
                    if arc.arc_type != CellArcType::Combinational {
                        continue;
                    }
                    let target = CellPin::new(pin.cell, arc.other_port);
                    if self.port_data.contains_key(&target) {
                        edges.entry(pin).or_default().push(target);
                        *in_degree.entry(target).or_insert(0) += 1;
                    }
                }
            } else if pd.port_type == PortType::Out {
                // Output port: routing to net users.
                let cell = design.cell(pin.cell);
                if let Some(pi) = cell.ports.get(&pin.port) {
                    if let Some(net_idx) = pi.net() {
                        let net = design.net(net_idx);
                        for user in &net.users {
                            if !user.is_valid() {
                                continue;
                            }
                            let user_cell = user.cell;
                            let target = CellPin::new(user_cell, user.port);
                            if self.port_data.contains_key(&target) {
                                edges.entry(pin).or_default().push(target);
                                *in_degree.entry(target).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: Vec<CellPin> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&pin, _)| pin)
            .collect();
        let mut sorted = Vec::with_capacity(pins.len());

        while let Some(pin) = queue.pop() {
            sorted.push(pin);
            if let Some(targets) = edges.get(&pin) {
                for &target in targets {
                    let deg = in_degree.get_mut(&target).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(target);
                    }
                }
            }
        }

        self.topological_order = sorted;
    }

    // =====================================================================
    // Internal: Domain setup (C++ setup_port_domains)
    // =====================================================================

    /// Assign clock domains to ports via fixed-point iteration.
    ///
    /// Following the C++ `setup_port_domains()` pattern:
    /// 1. Forward pass: registered outputs are startpoints; propagate domains forward.
    /// 2. Backward pass: registered inputs are endpoints; propagate domains backward.
    /// 3. Compute domain pairs at each port.
    fn setup_port_domains(&mut self, ctx: &Context) {
        // Clear existing startpoints/endpoints.
        for pd in &mut self.per_domain {
            pd.startpoints.clear();
            pd.endpoints.clear();
        }

        // Per-port arrival and required domain sets.
        let mut port_arrival_domains: FxHashMap<CellPin, FxHashSet<ClockDomainId>> =
            FxHashMap::default();
        let mut port_required_domains: FxHashMap<CellPin, FxHashSet<ClockDomainId>> =
            FxHashMap::default();

        // Clone topological order to avoid borrow conflicts with self mutation.
        let topo_order = self.topological_order.clone();
        let mut first_iter = true;

        loop {
            let mut updated = false;

            // Forward pass: collect startpoint info first, then apply.
            if first_iter {
                let mut startpoint_info: Vec<(CellPin, IdString, ClockEdge)> = Vec::new();
                for &port in &topo_order {
                    let pd = match self.port_data.get(&port) {
                        Some(pd) => pd,
                        None => continue,
                    };
                    if pd.port_type == PortType::Out || pd.port_type == PortType::InOut {
                        for arc in &pd.cell_arcs {
                            if arc.arc_type == CellArcType::ClockToQ {
                                startpoint_info.push((port, arc.other_port, arc.edge));
                            }
                        }
                    }
                }
                for (port, clock_port, edge) in startpoint_info {
                    let dom = self.resolve_domain_id(ctx, port.cell, clock_port, edge);
                    port_arrival_domains.entry(port).or_default().insert(dom);
                    while self.per_domain.len() <= dom.0 as usize {
                        self.per_domain.push(PerDomain::new());
                    }
                    self.per_domain[dom.0 as usize]
                        .startpoints
                        .push((port, clock_port));
                }
            }

            // Forward pass: propagate domains.
            for &port in &topo_order {
                let pd = match self.port_data.get(&port) {
                    Some(pd) => pd,
                    None => continue,
                };

                if pd.port_type == PortType::Out || pd.port_type == PortType::InOut {
                    // Copy arrival domains through routing (output -> net users).
                    let cell = ctx.design.cell(port.cell);
                    if let Some(pi) = cell.ports.get(&port.port) {
                        if let Some(net_idx) = pi.net() {
                            let net = ctx.design.net(net_idx);
                            for user in &net.users {
                                if !user.is_valid() {
                                    continue;
                                }
                                let user_cell = user.cell;
                                let target = CellPin::new(user_cell, user.port);
                                if !self.port_data.contains_key(&target) {
                                    continue;
                                }
                                if let Some(src_domains) =
                                    port_arrival_domains.get(&port).cloned()
                                {
                                    let dst = port_arrival_domains.entry(target).or_default();
                                    for d in src_domains {
                                        if dst.insert(d) {
                                            updated = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Input port: copy arrival domains through combinational arcs.
                    let arcs: Vec<_> = pd
                        .cell_arcs
                        .iter()
                        .filter(|a| a.arc_type == CellArcType::Combinational)
                        .map(|a| a.other_port)
                        .collect();
                    if let Some(src_domains) = port_arrival_domains.get(&port).cloned() {
                        for other_port in arcs {
                            let target = CellPin::new(port.cell, other_port);
                            if !self.port_data.contains_key(&target) {
                                continue;
                            }
                            let dst = port_arrival_domains.entry(target).or_default();
                            for d in &src_domains {
                                if dst.insert(*d) {
                                    updated = true;
                                }
                            }
                        }
                    }
                }
            }

            // Backward pass: collect endpoint info first on first iter, then apply.
            if first_iter {
                let mut endpoint_info: Vec<(CellPin, IdString, ClockEdge)> = Vec::new();
                for &port in topo_order.iter().rev() {
                    let pd = match self.port_data.get(&port) {
                        Some(pd) => pd,
                        None => continue,
                    };
                    if pd.port_type == PortType::In {
                        for arc in &pd.cell_arcs {
                            if arc.arc_type == CellArcType::Setup {
                                endpoint_info.push((port, arc.other_port, arc.edge));
                            }
                        }
                    }
                }
                for (port, clock_port, edge) in endpoint_info {
                    let dom = self.resolve_domain_id(ctx, port.cell, clock_port, edge);
                    port_required_domains.entry(port).or_default().insert(dom);
                    while self.per_domain.len() <= dom.0 as usize {
                        self.per_domain.push(PerDomain::new());
                    }
                    self.per_domain[dom.0 as usize]
                        .endpoints
                        .push((port, clock_port));
                }
            }

            // Backward pass: propagate domains.
            for &port in topo_order.iter().rev() {
                let pd = match self.port_data.get(&port) {
                    Some(pd) => pd,
                    None => continue,
                };

                if pd.port_type == PortType::Out || pd.port_type == PortType::InOut {
                    // Copy required domains from output to input (through combinational arcs).
                    let arcs: Vec<_> = pd
                        .cell_arcs
                        .iter()
                        .filter(|a| a.arc_type == CellArcType::Combinational)
                        .map(|a| a.other_port)
                        .collect();
                    if let Some(src_domains) = port_required_domains.get(&port).cloned() {
                        for other_port in arcs {
                            let target = CellPin::new(port.cell, other_port);
                            if !self.port_data.contains_key(&target) {
                                continue;
                            }
                            let dst = port_required_domains.entry(target).or_default();
                            for d in &src_domains {
                                if dst.insert(*d) {
                                    updated = true;
                                }
                            }
                        }
                    }
                } else if pd.port_type == PortType::In {
                    // Copy required domains backward through routing.
                    let cell = ctx.design.cell(port.cell);
                    if let Some(pi) = cell.ports.get(&port.port) {
                        if let Some(net_idx) = pi.net() {
                            let net = ctx.design.net(net_idx);
                            if net.driver.is_valid() {
                                let driver_cell = net.driver.cell;
                                let target = CellPin::new(driver_cell, net.driver.port);
                                if self.port_data.contains_key(&target) {
                                    if let Some(src_domains) =
                                        port_required_domains.get(&port).cloned()
                                    {
                                        let dst =
                                            port_required_domains.entry(target).or_default();
                                        for d in src_domains {
                                            if dst.insert(d) {
                                                updated = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            first_iter = false;
            if !updated {
                break;
            }
        }

        // Compute domain pairs at each port.
        // Collect all pairs first to avoid borrow conflict.
        let mut pairs_to_create: Vec<(ClockDomainId, ClockDomainId)> = Vec::new();
        for &port in &topo_order {
            let arr_doms: Vec<_> = port_arrival_domains
                .get(&port)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            let req_doms: Vec<_> = port_required_domains
                .get(&port)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            for &arr in &arr_doms {
                for &req in &req_doms {
                    pairs_to_create.push((arr, req));
                }
            }
        }
        for (launch, capture) in pairs_to_create {
            self.domain_pair_id(launch, capture);
        }

        // Compute period for each domain pair.
        let default_period = self.get_default_period();
        for dp in &mut self.domain_pairs {
            let launch = self.domain_registry.get(dp.pair.launch);
            let capture = self.domain_registry.get(dp.pair.capture);

            if launch.clock_net != capture.clock_net {
                continue;
            }

            let clk = launch.clock_net;
            let mut period = self
                .clock_constraints
                .get(&clk)
                .copied()
                .unwrap_or(default_period);

            // Half period for opposite edges.
            if launch.edge != capture.edge {
                period /= 2;
            }

            dp.period = DelayPair::uniform(period);
        }
    }

    /// Resolve a domain ID from a cell's clock port.
    fn resolve_domain_id(
        &mut self,
        ctx: &Context,
        cell_idx: CellId,
        clock_port: IdString,
        edge: ClockEdge,
    ) -> ClockDomainId {
        let cell = ctx.design.cell(cell_idx);
        let net_idx = match cell.ports.get(&clock_port).and_then(|p| p.net()) {
            Some(n) => n,
            None => return self.domain_registry.async_domain,
        };
        let net = ctx.design.net(net_idx);

        let period = if net.clock_constraint > 0 {
            net.clock_constraint
        } else {
            self.clock_constraints
                .get(&net.name)
                .copied()
                .unwrap_or(0)
        };

        self.domain_registry.domain_id(net.name, edge, period)
    }

    // =====================================================================
    // Internal: Related domain identification (C++ identify_related_domains)
    // =====================================================================

    /// Identify related clock domains by tracing upstream through combinational logic.
    fn identify_related_domains(&mut self, ctx: &Context) {
        let clock_nets: FxHashSet<IdString> = self
            .domain_registry
            .iter()
            .filter(|(_, d)| d.is_clocked())
            .map(|(_, d)| d.clock_net)
            .collect();

        // For each clock net, find all upstream driver nets with cumulative delays.
        let mut clock_drivers: FxHashMap<IdString, FxHashMap<IdString, DelayT>> =
            FxHashMap::default();

        for &clk_net_name in &clock_nets {
            let net_idx = match ctx.design.net_by_name(clk_net_name) {
                Some(idx) => idx,
                None => continue,
            };
            let net = ctx.design.net(net_idx);
            if !net.driver.is_connected() {
                continue;
            }

            let mut drivers: FxHashMap<IdString, DelayT> = FxHashMap::default();
            let mut visited: FxHashSet<IdString> = FxHashSet::default();
            self.find_net_drivers(ctx, net_idx, &mut visited, &mut drivers, 0);
            clock_drivers.insert(clk_net_name, drivers);
        }

        // Find related clocks: two clocks sharing exactly one common upstream driver.
        let clk_names: Vec<IdString> = clock_drivers.keys().copied().collect();
        for &c1 in &clk_names {
            for &c2 in &clk_names {
                if c1 == c2 {
                    continue;
                }
                let d1 = &clock_drivers[&c1];
                let d2 = &clock_drivers[&c2];

                let common: Vec<IdString> = d1.keys().filter(|k| d2.contains_key(k)).copied().collect();
                if common.len() != 1 {
                    continue;
                }

                let driver = common[0];
                let delay = d2[&driver] - d1[&driver];
                self.clock_delays.insert((c1, c2), delay);
            }
        }
    }

    /// Recursively find upstream drivers of a net through combinational logic.
    fn find_net_drivers(
        &self,
        ctx: &Context,
        net_idx: NetId,
        visited: &mut FxHashSet<IdString>,
        drivers: &mut FxHashMap<IdString, DelayT>,
        delay_acc: DelayT,
    ) {
        let net = ctx.design.net(net_idx);
        if !net.driver.is_connected() {
            return;
        }
        let driver_cell_idx = net.driver.cell;

        // Cycle detection.
        if visited.contains(&net.name) {
            drivers.insert(net.name, delay_acc);
            return;
        }
        visited.insert(net.name);

        let cell = ctx.design.cell(driver_cell_idx);
        let driver_port = net.driver.port;

        // Single-port cell: this is a leaf driver.
        if cell.ports.len() == 1 {
            drivers.insert(net.name, delay_acc);
            return;
        }

        // Check if driver port is combinational output.
        let driver_pin = CellPin::new(driver_cell_idx, driver_port);
        let port_class = self
            .port_data
            .get(&driver_pin)
            .map(|pd| pd.port_class)
            .unwrap_or(TimingPortClass::Combinational);

        if port_class != TimingPortClass::Combinational {
            drivers.insert(net.name, delay_acc);
            return;
        }

        // Recurse upstream through combinational inputs.
        let mut went_upstream = false;
        for (input_name, input_port) in &cell.ports {
            if input_port.port_type() != PortType::In {
                continue;
            }
            let Some(input_net_idx) = input_port.net() else {
                continue;
            };

            let input_pin = CellPin::new(driver_cell_idx, *input_name);
            let Some(input_pd) = self.port_data.get(&input_pin) else {
                continue;
            };
            if input_pd.port_class != TimingPortClass::Combinational {
                continue;
            }

            // Find combinational arc from this input to driver_port.
            let Some(arc) = input_pd.cell_arcs.iter().find(|a| {
                a.arc_type == CellArcType::Combinational && a.other_port == driver_port
            }) else {
                continue;
            };
            let arc_delay = arc.value.max_delay();

            self.find_net_drivers(ctx, input_net_idx, visited, drivers, delay_acc + arc_delay);
            went_upstream = true;
        }

        if !went_upstream {
            drivers.insert(net.name, delay_acc);
        }
    }

    // =====================================================================
    // Internal: Forward propagation (new, uses CellArc cache)
    // =====================================================================

    fn forward_propagation(&mut self, design: &Design) {
        let with_skew = self.with_clock_skew;

        // Initialize arrival times for startpoints.
        for dom_idx in 0..self.per_domain.len() {
            for sp_idx in 0..self.per_domain[dom_idx].startpoints.len() {
                let (port, clock_port) = self.per_domain[dom_idx].startpoints[sp_idx];
                let mut init_arrival: DelayT = 0;

                // Add clock-to-Q delay.
                if let Some(pd) = self.port_data.get(&port) {
                    for arc in &pd.cell_arcs {
                        if arc.arc_type == CellArcType::ClockToQ
                            && arc.other_port == clock_port
                        {
                            init_arrival += arc.value.as_delay_pair().max_delay;
                            // Include clock routing delay for skew analysis.
                            if with_skew {
                                let clk_pin = CellPin::new(port.cell, arc.other_port);
                                if let Some(clk_pd) = self.port_data.get(&clk_pin) {
                                    init_arrival += clk_pd.route_delay.max_delay;
                                }
                            }
                            break;
                        }
                    }
                }

                self.arrival_times
                    .entry(port)
                    .and_modify(|t| *t = (*t).max(init_arrival))
                    .or_insert(init_arrival);
            }
        }

        // Walk forward in topological order.
        for i in 0..self.topological_order.len() {
            let port = self.topological_order[i];
            let Some(pd) = self.port_data.get(&port) else {
                continue;
            };

            if pd.port_type == PortType::Out || pd.port_type == PortType::InOut {
                // Output port: propagate through routing.
                let Some(arrival) = self.arrival_times.get(&port).copied() else {
                    continue;
                };
                let cell = design.cell(port.cell);
                let Some(pi) = cell.ports.get(&port.port) else {
                    continue;
                };
                let Some(net_idx) = pi.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                for user in &net.users {
                    if !user.is_valid() {
                        continue;
                    }
                    let user_cell = user.cell;
                    let target = CellPin::new(user_cell, user.port);
                    let route_delay = self
                        .port_data
                        .get(&target)
                        .map(|pd| pd.route_delay.max_delay)
                        .unwrap_or(0);
                    let next_arr = arrival + route_delay;
                    if self.arrival_times.get(&target).map_or(true, |&old| next_arr > old) {
                        self.arrival_times.insert(target, next_arr);
                        self.predecessors
                            .insert(target, (port, Some(net_idx), route_delay));
                    }
                }
            } else if pd.port_type == PortType::In {
                // Input port: propagate through combinational cell arcs.
                let Some(arrival) = self.arrival_times.get(&port).copied() else {
                    continue;
                };
                let arcs: Vec<_> = pd.cell_arcs
                    .iter()
                    .filter(|a| a.arc_type == CellArcType::Combinational)
                    .map(|a| (a.other_port, a.value.as_delay_pair().max_delay))
                    .collect();
                for (other_port, delay) in arcs {
                    let target = CellPin::new(port.cell, other_port);
                    let next_arr = arrival + delay;
                    if self.arrival_times.get(&target).map_or(true, |&old| next_arr > old) {
                        self.arrival_times.insert(target, next_arr);
                        self.predecessors.insert(target, (port, None, delay));
                    }
                }
            }
        }
    }

    // =====================================================================
    // Internal: Backward propagation (new, uses CellArc cache)
    // =====================================================================

    fn backward_propagation(&mut self, design: &Design) {
        let with_skew = self.with_clock_skew;

        // Initialize required times at endpoints.
        for dom_idx in 0..self.per_domain.len() {
            // Get the period for this domain.
            let domain_period = self.domain_registry.get(ClockDomainId(dom_idx as u32)).period;
            let period = if domain_period > 0 {
                domain_period
            } else {
                self.get_default_period()
            };

            for ep_idx in 0..self.per_domain[dom_idx].endpoints.len() {
                let (port, clock_port) = self.per_domain[dom_idx].endpoints[ep_idx];
                let mut init_required: DelayT = period;

                // Subtract setup time and add clock skew.
                if let Some(pd) = self.port_data.get(&port) {
                    for arc in &pd.cell_arcs {
                        if arc.arc_type == CellArcType::Setup && arc.other_port == clock_port {
                            init_required -= arc.value.max_delay();
                            // Include clock routing delay for skew analysis.
                            if with_skew {
                                let clk_pin = CellPin::new(port.cell, arc.other_port);
                                if let Some(clk_pd) = self.port_data.get(&clk_pin) {
                                    init_required += clk_pd.route_delay.max_delay;
                                }
                            }
                            break;
                        }
                    }
                }

                self.required_times
                    .entry(port)
                    .and_modify(|t| *t = (*t).min(init_required))
                    .or_insert(init_required);
            }
        }

        // Walk backward in reverse topological order.
        for i in (0..self.topological_order.len()).rev() {
            let port = self.topological_order[i];
            let Some(pd) = self.port_data.get(&port) else {
                continue;
            };

            if pd.port_type == PortType::In {
                // Input port: propagate backward through routing.
                let Some(required) = self.required_times.get(&port).copied() else {
                    continue;
                };
                let cell = design.cell(port.cell);
                let Some(pi) = cell.ports.get(&port.port) else {
                    continue;
                };
                let Some(net_idx) = pi.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                if net.driver.is_valid() {
                    let driver_cell = net.driver.cell;
                    let target = CellPin::new(driver_cell, net.driver.port);
                    let route_delay = pd.route_delay.max_delay;
                    let req_at_driver = required - route_delay;
                    self.required_times
                        .entry(target)
                        .and_modify(|t| *t = (*t).min(req_at_driver))
                        .or_insert(req_at_driver);
                }
            } else if pd.port_type == PortType::Out || pd.port_type == PortType::InOut {
                // Output port: propagate backward through combinational arcs.
                let Some(required) = self.required_times.get(&port).copied() else {
                    continue;
                };
                let arcs: Vec<_> = pd.cell_arcs
                    .iter()
                    .filter(|a| a.arc_type == CellArcType::Combinational)
                    .map(|a| (a.other_port, a.value.as_delay_pair().max_delay))
                    .collect();
                for (other_port, delay) in arcs {
                    let target = CellPin::new(port.cell, other_port);
                    let req = required - delay;
                    self.required_times
                        .entry(target)
                        .and_modify(|t| *t = (*t).min(req))
                        .or_insert(req);
                }
            }
        }
    }

    // =====================================================================
    // Internal: Legacy forward/backward (for analyse() without chipdb)
    // =====================================================================

    fn forward_propagation_legacy(&mut self, design: &Design, sorted_cells: &[CellId]) {
        // Initialize arrival times.
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);
            for (port_name, port_info) in &cell.ports {
                let pin = CellPin::new(cell_idx, *port_name);
                let port_class = self.port_class_or_comb(pin);
                match port_class {
                    TimingPortClass::RegisterOutput => {
                        self.arrival_times.insert(pin, DEFAULT_COMB_DELAY);
                    }
                    TimingPortClass::Combinational if port_info.port_type() == PortType::In => {
                        let is_primary = match port_info.net() {
                            None => true,
                            Some(net_idx) => !design.net(net_idx).driver.is_connected(),
                        };
                        if is_primary {
                            self.arrival_times.insert(pin, 0);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate through cells.
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);
            let mut max_input_arrival: DelayT = DelayT::MIN;
            let mut has_input = false;

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::In {
                    continue;
                }
                let pin = CellPin::new(cell_idx, *port_name);
                if self.port_class_or_comb(pin) == TimingPortClass::ClockInput {
                    continue;
                }
                let Some(net_idx) = port_info.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                if !net.driver.is_valid() {
                    continue;
                }
                let driver_cell = net.driver.cell;
                let driver_pin = CellPin::new(driver_cell, net.driver.port);
                if let Some(&driver_arrival) = self.arrival_times.get(&driver_pin) {
                    self.arrival_times
                        .entry(pin)
                        .and_modify(|t| *t = (*t).max(driver_arrival))
                        .or_insert(driver_arrival);
                    if driver_arrival > max_input_arrival {
                        max_input_arrival = driver_arrival;
                    }
                    has_input = true;
                }
            }

            let has_comb_input = cell.ports.iter().any(|(pn, pi)| {
                pi.port_type() == PortType::In
                    && self.port_class_or_comb(CellPin::new(cell_idx, *pn))
                        == TimingPortClass::Combinational
            });

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::Out {
                    continue;
                }
                let pin = CellPin::new(cell_idx, *port_name);
                if self.port_class_or_comb(pin) != TimingPortClass::Combinational {
                    continue;
                }
                let arrival = if has_input {
                    max_input_arrival + DEFAULT_COMB_DELAY
                } else if !has_comb_input {
                    0
                } else {
                    continue;
                };
                self.arrival_times
                    .entry(pin)
                    .and_modify(|t| *t = (*t).max(arrival))
                    .or_insert(arrival);
            }
        }
    }

    fn backward_propagation_legacy(&mut self, design: &Design, sorted_cells: &[CellId]) {
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);
            for (port_name, port_info) in &cell.ports {
                let pin = CellPin::new(cell_idx, *port_name);
                let port_class = self.port_class_or_comb(pin);
                match port_class {
                    TimingPortClass::RegisterInput => {
                        let period = self.port_domain_period(pin);
                        let setup_time = DEFAULT_COMB_DELAY / 2;
                        self.required_times.insert(pin, period - setup_time);
                    }
                    TimingPortClass::Combinational if port_info.port_type() == PortType::Out => {
                        let is_primary = match port_info.net() {
                            None => true,
                            Some(net_idx) => design.net(net_idx).users.is_empty(),
                        };
                        if is_primary {
                            self.required_times.insert(pin, self.get_default_period());
                        }
                    }
                    _ => {}
                }
            }
        }

        for &cell_idx in sorted_cells.iter().rev() {
            let cell = design.cell(cell_idx);
            let mut min_output_required: DelayT = DelayT::MAX;
            let mut has_output_required = false;

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::Out {
                    continue;
                }
                let pin = CellPin::new(cell_idx, *port_name);
                if self.port_class_or_comb(pin) != TimingPortClass::Combinational {
                    continue;
                }
                let Some(net_idx) = port_info.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                for user in &net.users {
                    if !user.is_valid() {
                        continue;
                    }
                    let user_cell = user.cell;
                    let user_pin = CellPin::new(user_cell, user.port);
                    if let Some(&user_required) = self.required_times.get(&user_pin) {
                        self.required_times
                            .entry(pin)
                            .and_modify(|t| *t = (*t).min(user_required))
                            .or_insert(user_required);
                        if user_required < min_output_required {
                            min_output_required = user_required;
                        }
                        has_output_required = true;
                    }
                }
            }

            if !has_output_required {
                continue;
            }
            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::In {
                    continue;
                }
                let pin = CellPin::new(cell_idx, *port_name);
                if self.port_class_or_comb(pin) == TimingPortClass::Combinational {
                    let required = min_output_required - DEFAULT_COMB_DELAY;
                    self.required_times
                        .entry(pin)
                        .and_modify(|t| *t = (*t).min(required))
                        .or_insert(required);
                }
            }
        }
    }

    // =====================================================================
    // Internal: Slack and path computation
    // =====================================================================

    fn compute_slack_and_paths(&mut self, design: &Design) {
        self.worst_slack = DelayT::MAX;
        self.worst_setup_slack = DelayT::MAX;
        self.worst_hold_slack = DelayT::MAX;

        for (cell_idx, cell) in design.iter_alive_cells() {
            for (&port_name, _) in &cell.ports {
                let pin = CellPin::new(cell_idx, port_name);
                if self.port_class_or_comb(pin) != TimingPortClass::RegisterInput {
                    continue;
                }

                let arrival = self.arrival_times.get(&pin).copied();
                let required = self.required_times.get(&pin).copied();

                if let (Some(arr), Some(req)) = (arrival, required) {
                    let setup_slack = req - arr;
                    self.worst_slack = self.worst_slack.min(setup_slack);
                    self.worst_setup_slack = self.worst_setup_slack.min(setup_slack);

                    let segments = self.reconstruct_path(pin);
                    let domain = self.port_domain_from_data(pin);

                    let from_endpoint = if let Some(first_seg) = segments.first() {
                        let from_domain = self.port_domain_from_data(
                            CellPin::new(first_seg.cell, first_seg.port),
                        );
                        TimingEndpoint {
                            cell: first_seg.cell,
                            port: first_seg.port,
                            domain: from_domain,
                        }
                    } else {
                        TimingEndpoint {
                            cell: cell_idx,
                            port: port_name,
                            domain: domain.clone(),
                        }
                    };

                    let to_endpoint = TimingEndpoint {
                        cell: cell_idx,
                        port: port_name,
                        domain,
                    };
                    self.paths.push(TimingPath {
                        from: from_endpoint,
                        to: to_endpoint,
                        delay: arr,
                        budget: req,
                        slack: setup_slack,
                        segments,
                    });
                }
            }
        }

        if self.worst_slack == DelayT::MAX {
            self.worst_slack = 0;
        }
        if self.worst_setup_slack == DelayT::MAX {
            self.worst_setup_slack = 0;
        }
        if self.worst_hold_slack == DelayT::MAX {
            self.worst_hold_slack = 0;
        }
    }

    /// Reconstruct the path from a given endpoint pin back to the startpoint.
    fn reconstruct_path(&self, endpoint: CellPin) -> Vec<PathSegment> {
        let mut segments = Vec::new();
        let mut current = endpoint;
        let mut visited = FxHashSet::default();

        while let Some(&(pred, net_id, delay)) = self.predecessors.get(&current) {
            if !visited.insert(current) {
                break; // Avoid infinite loops.
            }
            segments.push(PathSegment {
                net: net_id.unwrap_or(NetId::NONE),
                cell: current.cell,
                port: current.port,
                delay,
            });
            current = pred;
        }
        // Add the startpoint itself.
        if !segments.is_empty() {
            let start_delay = self.arrival_times.get(&current).copied().unwrap_or(0);
            segments.push(PathSegment {
                net: NetId::NONE,
                cell: current.cell,
                port: current.port,
                delay: start_delay,
            });
        }
        segments.reverse();
        segments
    }

    fn compute_criticality(&mut self, design: &Design) {
        if self.worst_slack >= 0 {
            return;
        }
        let neg_ws = -self.worst_slack as f64;

        for (net_idx, net) in design.iter_alive_nets() {
            let mut max_crit: f32 = 0.0;
            for user in &net.users {
                if !user.is_valid() {
                    continue;
                }
                let user_cell = user.cell;
                let user_pin = CellPin::new(user_cell, user.port);
                let arrival = self.arrival_times.get(&user_pin).copied().unwrap_or(0);
                let required = self.required_times.get(&user_pin).copied().unwrap_or(0);
                let slack = required - arrival;
                let crit =
                    (1.0 - ((slack - self.worst_slack) as f64 / neg_ws)).clamp(0.0, 1.0) as f32;
                if crit > max_crit {
                    max_crit = crit;
                }
            }
            self.net_criticality.insert(net_idx, max_crit);
        }
    }

    // =====================================================================
    // Internal: Helpers
    // =====================================================================

    /// Look up a port's timing class from cached port_data, defaulting to Combinational.
    fn port_class_or_comb(&self, pin: CellPin) -> TimingPortClass {
        self.port_data
            .get(&pin)
            .map(|pd| pd.port_class)
            .unwrap_or(TimingPortClass::Combinational)
    }

    /// Get clock domain for a pin from cached port data.
    fn port_domain_from_data(&self, pin: CellPin) -> ClockDomain {
        // Find the domain by checking if this pin is an endpoint in any domain.
        for (dom_id, dom) in self.domain_registry.iter() {
            if dom_id == self.domain_registry.async_domain {
                continue;
            }
            let per_dom = match self.per_domain.get(dom_id.0 as usize) {
                Some(pd) => pd,
                None => continue,
            };
            if per_dom.endpoints.iter().any(|(ep, _)| *ep == pin) {
                return dom.clone();
            }
        }
        ClockDomain::unclocked()
    }

    /// Get the period for a port's domain (legacy helper).
    fn port_domain_period(&self, pin: CellPin) -> DelayT {
        // Check legacy domains first.
        if let Some(d) = self.legacy_port_domains.get(&pin) {
            if d.period > 0 {
                return d.period;
            }
            if d.is_clocked() {
                return self
                    .clock_constraints
                    .get(&d.clock_net)
                    .copied()
                    .unwrap_or_else(|| self.get_default_period());
            }
        }
        let domain = self.port_domain_from_data(pin);
        if domain.period > 0 {
            domain.period
        } else if domain.is_clocked() {
            self.clock_constraints
                .get(&domain.clock_net)
                .copied()
                .unwrap_or_else(|| self.get_default_period())
        } else {
            self.get_default_period()
        }
    }

    /// Get the default clock period (smallest constrained period, or 10ns).
    fn get_default_period(&self) -> DelayT {
        self.clock_constraints
            .values()
            .copied()
            .min()
            .unwrap_or(10_000)
    }

    /// Heuristic port classification for legacy `analyse()` path.
    fn classify_ports_heuristic(&mut self, design: &Design) {
        self.init_ports(design);
        self.get_cell_delays_heuristic(design);

        // Also populate legacy_port_domains for test compatibility.
        for (cell_idx, cell) in design.iter_alive_cells() {
            let mut clock_domain_for_cell: Option<ClockDomain> = None;
            let mut clock_ports: FxHashSet<IdString> = FxHashSet::default();

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type() != PortType::In {
                    continue;
                }
                let Some(net_idx) = port_info.net() else {
                    continue;
                };
                let net = design.net(net_idx);
                let period = if net.clock_constraint > 0 {
                    Some(net.clock_constraint)
                } else {
                    self.clock_constraints.get(&net.name).copied().filter(|&p| p > 0)
                };
                if let Some(period) = period {
                    clock_ports.insert(*port_name);
                    clock_domain_for_cell = Some(ClockDomain {
                        clock_net: net.name,
                        edge: ClockEdge::Rising,
                        period,
                    });
                }
            }

            if let Some(ref domain) = clock_domain_for_cell {
                for (port_name, _) in &cell.ports {
                    let pin = CellPin::new(cell_idx, *port_name);
                    let port_class = self.port_class_or_comb(pin);
                    let is_clocked = matches!(
                        port_class,
                        TimingPortClass::RegisterInput
                            | TimingPortClass::RegisterOutput
                            | TimingPortClass::ClockInput
                    );
                    if is_clocked {
                        self.legacy_port_domains.insert(pin, domain.clone());
                    }
                }
            }
        }
    }
}

// =========================================================================
// Public test accessors
// =========================================================================

impl TimingAnalyser {
    /// Get the clock constraints map (for testing).
    pub fn clock_constraints(&self) -> &FxHashMap<IdString, DelayT> {
        &self.clock_constraints
    }

    /// Get arrival time for a cell pin (for testing).
    pub fn arrival_time(&self, cell: CellId, port: IdString) -> Option<DelayT> {
        self.arrival_times.get(&CellPin::new(cell, port)).copied()
    }

    /// Get required time for a cell pin (for testing).
    pub fn required_time(&self, cell: CellId, port: IdString) -> Option<DelayT> {
        self.required_times.get(&CellPin::new(cell, port)).copied()
    }

    /// Get port classification for a cell pin (for testing).
    pub fn port_class(&self, cell: CellId, port: IdString) -> Option<TimingPortClass> {
        self.port_data
            .get(&CellPin::new(cell, port))
            .map(|pd| pd.port_class)
    }

    /// Get clock domain for a cell pin (for testing).
    pub fn port_domain(&self, cell: CellId, port: IdString) -> Option<ClockDomain> {
        let pin = CellPin::new(cell, port);
        // Check legacy domains first (set by heuristic path).
        if let Some(d) = self.legacy_port_domains.get(&pin) {
            return Some(d.clone());
        }
        let domain = self.port_domain_from_data(pin);
        if domain.is_clocked() {
            Some(domain)
        } else {
            None
        }
    }

    /// Get all computed timing paths (for testing).
    pub fn paths(&self) -> &[TimingPath] {
        &self.paths
    }
}

impl Default for TimingAnalyser {
    fn default() -> Self {
        Self::new()
    }
}

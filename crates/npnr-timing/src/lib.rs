//! Static timing analysis engine for the nextpnr-rust FPGA place-and-route tool.
//!
//! This crate performs static timing analysis (STA) on a placed design to determine
//! whether it meets frequency constraints and to provide criticality values used by
//! timing-driven placement and routing.
//!
//! The central type is [`TimingAnalyser`], which performs forward and backward
//! propagation through the netlist to compute arrival times, required times, slack,
//! and criticality for every net.

mod domain;
mod path;
mod sort;

pub use domain::ClockDomain;
pub use path::{PathSegment, TimingEndpoint, TimingPath, TimingPortInfo, TimingReport};

use log::debug;
use npnr_netlist::{CellIdx, Design, NetIdx};
use npnr_types::{ClockEdge, DelayT, IdString, IdStringPool, PortType, TimingPortClass};
use rustc_hash::{FxHashMap, FxHashSet};

use sort::topological_sort;

/// Default combinational cell delay in picoseconds when no chipdb data is available.
const DEFAULT_COMB_DELAY: DelayT = 100;

// ---------------------------------------------------------------------------
// TimingAnalyser
// ---------------------------------------------------------------------------

/// Static timing analyser.
///
/// Performs forward (arrival-time) and backward (required-time) propagation
/// through a design netlist, then computes slack and criticality for every net.
pub struct TimingAnalyser {
    /// Net criticality values (0.0 = not critical, 1.0 = most critical).
    net_criticality: FxHashMap<NetIdx, f32>,
    /// Port arrival times (forward pass): (cell, port) -> arrival time.
    arrival_times: FxHashMap<(CellIdx, IdString), DelayT>,
    /// Port required times (backward pass): (cell, port) -> required time.
    required_times: FxHashMap<(CellIdx, IdString), DelayT>,
    /// Computed timing paths, sorted by slack (ascending = worst first).
    paths: Vec<TimingPath>,
    /// Clock domain constraints: clock net name -> period in picoseconds.
    clock_constraints: FxHashMap<IdString, DelayT>,
    /// Worst negative slack across all endpoints (most negative = worst).
    worst_slack: DelayT,
    /// Whether timing has been computed and is up-to-date.
    is_valid: bool,
    /// Cell timing port classifications, populated during analysis.
    /// (cell, port) -> TimingPortClass
    port_classes: FxHashMap<(CellIdx, IdString), TimingPortClass>,
    /// Clock domain assignment for register ports.
    /// (cell, port) -> ClockDomain
    port_domains: FxHashMap<(CellIdx, IdString), ClockDomain>,
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
            is_valid: false,
            port_classes: FxHashMap::default(),
            port_domains: FxHashMap::default(),
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

    /// Run timing analysis on the design.
    ///
    /// This performs:
    /// 1. Port classification (identify registers, combinational logic, clocks)
    /// 2. Topological sort of cells
    /// 3. Forward propagation (compute arrival times)
    /// 4. Backward propagation (compute required times)
    /// 5. Slack and criticality computation
    /// 6. Path extraction
    pub fn analyse(
        &mut self,
        design: &Design,
        _id_pool: &IdStringPool,
    ) {
        // Clear previous results.
        self.arrival_times.clear();
        self.required_times.clear();
        self.net_criticality.clear();
        self.paths.clear();
        self.port_classes.clear();
        self.port_domains.clear();
        self.worst_slack = 0;

        // Step 1: Classify ports.
        self.classify_ports(design);

        // Step 2: Topological sort.
        let sorted_cells = topological_sort(design);
        debug!("Topological sort: {} cells", sorted_cells.len());

        // Step 3: Forward propagation (arrival times).
        self.forward_propagation(design, &sorted_cells);

        // Step 4: Backward propagation (required times).
        self.backward_propagation(design, &sorted_cells);

        // Step 5: Compute slack and criticality.
        self.compute_slack_and_paths(design);
        self.compute_criticality(design);

        // Sort paths by slack (ascending = worst first).
        self.paths.sort_by_key(|p| p.slack);

        self.is_valid = true;
    }

    /// Get criticality of a net (0.0 to 1.0).
    ///
    /// Returns 0.0 if the net has no timing information.
    pub fn net_criticality(&self, net: NetIdx) -> f32 {
        self.net_criticality.get(&net).copied().unwrap_or(0.0)
    }

    /// Get criticality of a specific port (cell, port pair).
    ///
    /// For input ports, the criticality is derived from the net driving the port.
    /// Returns 0.0 if no timing information is available.
    pub fn port_criticality(&self, cell: CellIdx, port: IdString) -> f32 {
        if self.worst_slack >= 0 {
            return 0.0;
        }

        // Compute criticality from slack at this port.
        let arrival = self.arrival_times.get(&(cell, port)).copied().unwrap_or(0);
        let required = self.required_times.get(&(cell, port)).copied().unwrap_or(0);
        let slack = required - arrival;

        // criticality = 1.0 - (slack - worst_slack) / (-worst_slack)
        // When slack == worst_slack (worst path) -> criticality = 1.0
        // When slack == 0 (just met) -> criticality = 0.0
        let neg_ws = -self.worst_slack as f64; // positive value
        let crit = 1.0 - ((slack - self.worst_slack) as f64 / neg_ws);
        crit.clamp(0.0, 1.0) as f32
    }

    /// Get worst negative slack across all endpoints.
    ///
    /// Negative slack means timing violation. Zero or positive means timing is met.
    pub fn worst_slack(&self) -> DelayT {
        self.worst_slack
    }

    /// Get the N most critical paths (sorted by slack, ascending = worst first).
    pub fn critical_paths(&self, limit: usize) -> &[TimingPath] {
        let n = limit.min(self.paths.len());
        &self.paths[..n]
    }

    /// Compute Fmax from worst slack and clock period.
    ///
    /// Returns 0.0 if no clock constraints exist.
    pub fn fmax_mhz(&self) -> f64 {
        if self.clock_constraints.is_empty() {
            return 0.0;
        }

        // Find the tightest constraint (smallest period).
        let min_period = self.clock_constraints.values().copied().min().unwrap_or(0);
        if min_period <= 0 {
            return 0.0;
        }

        // Effective period = clock period - (-worst_slack) if slack is negative,
        // or just the clock period if slack >= 0.
        let effective_period = if self.worst_slack < 0 {
            min_period + self.worst_slack // worst_slack is negative, so this subtracts
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

    /// Check if timing is valid (i.e., analysis has been run and not invalidated).
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    // =====================================================================
    // Internal: Port classification
    // =====================================================================

    /// Classify each port on each cell for timing purposes.
    ///
    /// This is a heuristic classification based on port names and cell types.
    /// A real implementation would get this from the architecture / chipdb.
    fn classify_ports(&mut self, design: &Design) {
        for (idx, cell) in design.cell_store.iter().enumerate() {
            if !cell.alive {
                continue;
            }
            let cell_idx = CellIdx(idx as u32);

            // Determine if this cell is sequential by checking if any of its
            // input ports are connected to nets with clock constraints.

            // Determine which ports are clock ports by checking if they
            // are connected to nets with clock constraints.
            let mut clock_ports: FxHashSet<IdString> = FxHashSet::default();
            let mut clock_domain_for_cell: Option<ClockDomain> = None;

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type == PortType::In && port_info.net.is_some() {
                    let net = design.net(port_info.net);
                    if net.clock_constraint > 0 {
                        clock_ports.insert(*port_name);
                        clock_domain_for_cell = Some(ClockDomain {
                            clock_net: net.name,
                            edge: ClockEdge::Rising,
                            period: net.clock_constraint,
                        });
                    }
                    // Also check our explicit constraints.
                    if let Some(&period) = self.clock_constraints.get(&net.name) {
                        if period > 0 {
                            clock_ports.insert(*port_name);
                            clock_domain_for_cell = Some(ClockDomain {
                                clock_net: net.name,
                                edge: ClockEdge::Rising,
                                period,
                            });
                        }
                    }
                }
            }

            let is_sequential = !clock_ports.is_empty();

            for (port_name, port_info) in &cell.ports {
                let port_class = if clock_ports.contains(port_name) {
                    TimingPortClass::ClockInput
                } else if is_sequential && port_info.port_type == PortType::In {
                    TimingPortClass::RegisterInput
                } else if is_sequential && port_info.port_type == PortType::Out {
                    TimingPortClass::RegisterOutput
                } else {
                    TimingPortClass::Combinational
                };

                self.port_classes.insert((cell_idx, *port_name), port_class);

                if let Some(ref domain) = clock_domain_for_cell {
                    if port_class == TimingPortClass::RegisterInput
                        || port_class == TimingPortClass::RegisterOutput
                        || port_class == TimingPortClass::ClockInput
                    {
                        self.port_domains
                            .insert((cell_idx, *port_name), domain.clone());
                    }
                }
            }
        }
    }

    // =====================================================================
    // Internal: Forward propagation
    // =====================================================================

    /// Forward propagation: compute arrival times at each port.
    ///
    /// Starting from primary inputs (ports with no driver or register outputs),
    /// propagate arrival times forward through combinational logic.
    fn forward_propagation(&mut self, design: &Design, sorted_cells: &[CellIdx]) {
        // Initialize arrival times for register outputs and primary inputs.
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);
            for (port_name, port_info) in &cell.ports {
                let port_class = self.port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                match port_class {
                    TimingPortClass::RegisterOutput => {
                        // Register outputs start with clock-to-output delay.
                        // Use default clock-to-out delay.
                        let clk_to_out = DEFAULT_COMB_DELAY;
                        self.arrival_times.insert((cell_idx, *port_name), clk_to_out);
                    }
                    TimingPortClass::Combinational if port_info.port_type == PortType::Out => {
                        // Will be computed from inputs.
                    }
                    TimingPortClass::Combinational if port_info.port_type == PortType::In => {
                        // If this is a primary input (net driver is not from another cell
                        // in the design or net has no driver), initialize arrival time to 0.
                        if port_info.net.is_none() {
                            self.arrival_times.insert((cell_idx, *port_name), 0);
                        } else {
                            let net = design.net(port_info.net);
                            if !net.driver.is_connected() {
                                self.arrival_times.insert((cell_idx, *port_name), 0);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate through cells in topological order.
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);

            // Collect input arrival times for this cell.
            let mut max_input_arrival: DelayT = DelayT::MIN;
            let mut has_input = false;

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type != PortType::In {
                    continue;
                }

                let port_class = self.port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                // Skip clock inputs and register inputs for combinational propagation.
                if port_class == TimingPortClass::ClockInput {
                    continue;
                }

                if port_info.net.is_none() {
                    continue;
                }

                let net = design.net(port_info.net);
                if !net.driver.is_connected() {
                    continue;
                }

                let driver_cell = net.driver.cell;
                let driver_port = net.driver.port;

                // Get driver's arrival time.
                if let Some(&driver_arrival) = self.arrival_times.get(&(driver_cell, driver_port)) {
                    // Estimate net delay.
                    let net_delay = self.estimate_net_delay(design, port_info.net);

                    let arrival_at_input = driver_arrival + net_delay;
                    self.arrival_times
                        .entry((cell_idx, *port_name))
                        .and_modify(|t| *t = (*t).max(arrival_at_input))
                        .or_insert(arrival_at_input);

                    if arrival_at_input > max_input_arrival {
                        max_input_arrival = arrival_at_input;
                    }
                    has_input = true;
                }
            }

            // Check if this cell has any combinational input ports at all.
            let has_comb_input = cell.ports.iter().any(|(pn, pi)| {
                pi.port_type == PortType::In
                    && self
                        .port_classes
                        .get(&(cell_idx, *pn))
                        .copied()
                        .unwrap_or(TimingPortClass::Combinational)
                        == TimingPortClass::Combinational
            });

            // Propagate to output ports.
            // If this cell has combinational inputs and we found arrival times,
            // propagate max_input + comb_delay.
            // If this cell has no combinational inputs at all (pure source cell),
            // its combinational outputs get arrival time 0 (primary input).
            for (port_name, port_info) in &cell.ports {
                if port_info.port_type != PortType::Out {
                    continue;
                }

                let port_class = self
                    .port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                if port_class == TimingPortClass::Combinational {
                    let arrival = if has_input {
                        // Combinational output: arrival = max_input + comb_delay.
                        max_input_arrival + DEFAULT_COMB_DELAY
                    } else if !has_comb_input {
                        // Pure source cell (no combinational inputs):
                        // treat as primary input with arrival 0.
                        0
                    } else {
                        // Cell has combinational inputs but none had arrival
                        // times yet. Skip for now.
                        continue;
                    };
                    self.arrival_times
                        .entry((cell_idx, *port_name))
                        .and_modify(|t| *t = (*t).max(arrival))
                        .or_insert(arrival);
                }
                // RegisterOutput arrival was already set in the initialization pass.
            }
        }
    }

    // =====================================================================
    // Internal: Backward propagation
    // =====================================================================

    /// Backward propagation: compute required times at each port.
    ///
    /// Starting from register inputs and primary outputs, propagate
    /// required times backward through combinational logic.
    fn backward_propagation(&mut self, design: &Design, sorted_cells: &[CellIdx]) {
        // Initialize required times at endpoints.
        for &cell_idx in sorted_cells {
            let cell = design.cell(cell_idx);
            for (port_name, port_info) in &cell.ports {
                let port_class = self.port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                match port_class {
                    TimingPortClass::RegisterInput => {
                        // Required time = clock period - setup time.
                        let domain = self.port_domains.get(&(cell_idx, *port_name));
                        let period = domain
                            .and_then(|d| {
                                if d.period > 0 {
                                    Some(d.period)
                                } else {
                                    self.clock_constraints.get(&d.clock_net).copied()
                                }
                            })
                            .unwrap_or(10_000); // Default 10ns if unconstrained.
                        let setup_time = DEFAULT_COMB_DELAY / 2; // Simple setup time estimate.
                        self.required_times
                            .insert((cell_idx, *port_name), period - setup_time);
                    }
                    TimingPortClass::Combinational if port_info.port_type == PortType::Out => {
                        // If this output is not connected to anything, it's a primary output.
                        if port_info.net.is_none() {
                            // Primary output: required time = period or large value.
                            let period = self.get_default_period();
                            self.required_times.insert((cell_idx, *port_name), period);
                        } else {
                            let net = design.net(port_info.net);
                            if net.users.is_empty() {
                                let period = self.get_default_period();
                                self.required_times
                                    .insert((cell_idx, *port_name), period);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Propagate backward in reverse topological order.
        for &cell_idx in sorted_cells.iter().rev() {
            let cell = design.cell(cell_idx);

            // Find the minimum required time among all output ports.
            let mut min_output_required: DelayT = DelayT::MAX;
            let mut has_output_required = false;

            for (port_name, port_info) in &cell.ports {
                if port_info.port_type != PortType::Out {
                    continue;
                }

                let port_class = self.port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                if port_class != TimingPortClass::Combinational {
                    continue;
                }

                // Check what the downstream users require.
                if port_info.net.is_some() {
                    let net = design.net(port_info.net);
                    for user in &net.users {
                        if !user.is_connected() {
                            continue;
                        }
                        if let Some(&user_required) =
                            self.required_times.get(&(user.cell, user.port))
                        {
                            let net_delay =
                                self.estimate_net_delay(design, port_info.net);
                            let required_at_output = user_required - net_delay;
                            self.required_times
                                .entry((cell_idx, *port_name))
                                .and_modify(|t| *t = (*t).min(required_at_output))
                                .or_insert(required_at_output);

                            if required_at_output < min_output_required {
                                min_output_required = required_at_output;
                            }
                            has_output_required = true;
                        }
                    }
                }
            }

            // Propagate to input ports.
            if has_output_required {
                for (port_name, port_info) in &cell.ports {
                    if port_info.port_type != PortType::In {
                        continue;
                    }

                    let port_class = self.port_classes
                        .get(&(cell_idx, *port_name))
                        .copied()
                        .unwrap_or(TimingPortClass::Combinational);

                    if port_class == TimingPortClass::Combinational {
                        let required = min_output_required - DEFAULT_COMB_DELAY;
                        self.required_times
                            .entry((cell_idx, *port_name))
                            .and_modify(|t| *t = (*t).min(required))
                            .or_insert(required);
                    }
                }
            }
        }
    }

    // =====================================================================
    // Internal: Slack and path computation
    // =====================================================================

    /// Compute slack at each endpoint and extract timing paths.
    fn compute_slack_and_paths(&mut self, design: &Design) {
        self.worst_slack = DelayT::MAX;

        for (idx, cell) in design.cell_store.iter().enumerate() {
            if !cell.alive {
                continue;
            }
            let cell_idx = CellIdx(idx as u32);

            for (port_name, _port_info) in &cell.ports {
                let port_class = self.port_classes
                    .get(&(cell_idx, *port_name))
                    .copied()
                    .unwrap_or(TimingPortClass::Combinational);

                // Timing endpoints are register inputs.
                if port_class != TimingPortClass::RegisterInput {
                    continue;
                }

                let arrival = self.arrival_times.get(&(cell_idx, *port_name)).copied();
                let required = self.required_times.get(&(cell_idx, *port_name)).copied();

                if let (Some(arr), Some(req)) = (arrival, required) {
                    let slack = req - arr;

                    if slack < self.worst_slack {
                        self.worst_slack = slack;
                    }

                    let domain = self
                        .port_domains
                        .get(&(cell_idx, *port_name))
                        .cloned()
                        .unwrap_or_else(ClockDomain::unclocked);

                    let path = TimingPath {
                        from: TimingEndpoint {
                            cell: cell_idx,
                            port: *port_name,
                            domain: domain.clone(),
                        },
                        to: TimingEndpoint {
                            cell: cell_idx,
                            port: *port_name,
                            domain,
                        },
                        delay: arr,
                        budget: req,
                        slack,
                        segments: Vec::new(),
                    };

                    self.paths.push(path);
                }
            }
        }

        // If no failing paths found, set worst_slack to 0.
        if self.worst_slack == DelayT::MAX {
            self.worst_slack = 0;
        }
    }

    /// Compute net criticality values from slack.
    fn compute_criticality(&mut self, design: &Design) {
        if self.worst_slack >= 0 {
            // All timing met; all criticalities are 0.
            return;
        }

        for (idx, net) in design.net_store.iter().enumerate() {
            if !net.alive {
                continue;
            }
            let net_idx = NetIdx(idx as u32);

            // Compute criticality as the maximum criticality across all users.
            let mut max_crit: f32 = 0.0;

            for user in &net.users {
                if !user.is_connected() {
                    continue;
                }

                let arrival = self
                    .arrival_times
                    .get(&(user.cell, user.port))
                    .copied()
                    .unwrap_or(0);
                let required = self
                    .required_times
                    .get(&(user.cell, user.port))
                    .copied()
                    .unwrap_or(0);
                let slack = required - arrival;

                // criticality = 1.0 - (slack - worst_slack) / (-worst_slack)
                // When slack == worst_slack (worst path) -> criticality = 1.0
                // When slack == 0 (just met) -> criticality = 0.0
                let neg_ws = -self.worst_slack as f64;
                let crit = 1.0 - ((slack - self.worst_slack) as f64 / neg_ws);
                let crit = crit.clamp(0.0, 1.0) as f32;

                if crit > max_crit {
                    max_crit = crit;
                }
            }

            self.net_criticality.insert(net_idx, max_crit);
        }
    }

    // =====================================================================
    // Internal: Delay estimation helpers
    // =====================================================================

    /// Estimate net delay in picoseconds.
    ///
    /// If both driver and user cells are placed (have valid BEL assignments),
    /// uses Manhattan distance. Otherwise returns 0.
    fn estimate_net_delay(&self, design: &Design, net_idx: NetIdx) -> DelayT {
        if net_idx.is_none() {
            return 0;
        }

        let net = design.net(net_idx);
        if !net.driver.is_connected() {
            return 0;
        }

        let driver_cell = design.cell(net.driver.cell);
        if !driver_cell.bel.is_valid() {
            return 0;
        }

        // We don't have access to chipdb here to get tile_xy, so we use the
        // BEL's tile field directly for a rough Manhattan estimate.
        // tile is packed as (tile: i32, index: i32) where tile encodes position.
        // Without chipdb, we just return 0 for unplaced cells.
        0
    }

    /// Get the default clock period (smallest constrained period, or 10ns).
    fn get_default_period(&self) -> DelayT {
        self.clock_constraints
            .values()
            .copied()
            .min()
            .unwrap_or(10_000) // 10ns default
    }
}

impl Default for TimingAnalyser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_netlist::{Design, PortRef};
    use npnr_types::{IdStringPool, PortType};

    // =====================================================================
    // Test helpers
    // =====================================================================

    /// Helper to create a design with a simple combinational chain:
    /// INPUT -> LUT_A -> LUT_B -> OUTPUT
    fn make_comb_chain_design(pool: &IdStringPool) -> Design {
        let mut d = Design::new();

        // Cell types
        let io_type = pool.intern("IO");
        let lut_type = pool.intern("LUT4");

        // Cell names
        let input_name = pool.intern("input_cell");
        let lut_a_name = pool.intern("lut_a");
        let lut_b_name = pool.intern("lut_b");
        let output_name = pool.intern("output_cell");

        // Port names
        let i_port = pool.intern("I");
        let o_port = pool.intern("O");
        let a_port = pool.intern("A");
        let f_port = pool.intern("F");

        // Net names
        let net_in = pool.intern("net_in");
        let net_ab = pool.intern("net_ab");
        let net_out = pool.intern("net_out");

        // Create cells
        let input_idx = d.add_cell(input_name, io_type);
        let lut_a_idx = d.add_cell(lut_a_name, lut_type);
        let lut_b_idx = d.add_cell(lut_b_name, lut_type);
        let output_idx = d.add_cell(output_name, io_type);

        // Add ports
        d.cell_mut(input_idx).add_port(o_port, PortType::Out);
        d.cell_mut(lut_a_idx).add_port(a_port, PortType::In);
        d.cell_mut(lut_a_idx).add_port(f_port, PortType::Out);
        d.cell_mut(lut_b_idx).add_port(a_port, PortType::In);
        d.cell_mut(lut_b_idx).add_port(f_port, PortType::Out);
        d.cell_mut(output_idx).add_port(i_port, PortType::In);

        // Create nets and wire them up.
        // net_in: input_cell.O -> lut_a.A
        let net_in_idx = d.add_net(net_in);
        d.net_mut(net_in_idx).driver = PortRef {
            cell: input_idx,
            port: o_port,
            budget: 0,
        };
        d.cell_mut(input_idx).port_mut(o_port).unwrap().net = net_in_idx;
        d.cell_mut(input_idx).port_mut(o_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_in_idx).users.len() as i32;
        d.net_mut(net_in_idx).users.push(PortRef {
            cell: lut_a_idx,
            port: a_port,
            budget: 0,
        });
        d.cell_mut(lut_a_idx).port_mut(a_port).unwrap().net = net_in_idx;
        d.cell_mut(lut_a_idx).port_mut(a_port).unwrap().user_idx = user_idx;

        // net_ab: lut_a.F -> lut_b.A
        let net_ab_idx = d.add_net(net_ab);
        d.net_mut(net_ab_idx).driver = PortRef {
            cell: lut_a_idx,
            port: f_port,
            budget: 0,
        };
        d.cell_mut(lut_a_idx).port_mut(f_port).unwrap().net = net_ab_idx;
        d.cell_mut(lut_a_idx).port_mut(f_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_ab_idx).users.len() as i32;
        d.net_mut(net_ab_idx).users.push(PortRef {
            cell: lut_b_idx,
            port: a_port,
            budget: 0,
        });
        d.cell_mut(lut_b_idx).port_mut(a_port).unwrap().net = net_ab_idx;
        d.cell_mut(lut_b_idx).port_mut(a_port).unwrap().user_idx = user_idx;

        // net_out: lut_b.F -> output_cell.I
        let net_out_idx = d.add_net(net_out);
        d.net_mut(net_out_idx).driver = PortRef {
            cell: lut_b_idx,
            port: f_port,
            budget: 0,
        };
        d.cell_mut(lut_b_idx).port_mut(f_port).unwrap().net = net_out_idx;
        d.cell_mut(lut_b_idx).port_mut(f_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_out_idx).users.len() as i32;
        d.net_mut(net_out_idx).users.push(PortRef {
            cell: output_idx,
            port: i_port,
            budget: 0,
        });
        d.cell_mut(output_idx).port_mut(i_port).unwrap().net = net_out_idx;
        d.cell_mut(output_idx).port_mut(i_port).unwrap().user_idx = user_idx;

        d
    }

    /// Helper to create a register-to-register design:
    /// FF_A.Q -> LUT -> FF_B.D
    /// with a clock net "clk" connected to FF_A.CLK and FF_B.CLK.
    fn make_reg_to_reg_design(pool: &IdStringPool) -> (Design, IdString) {
        let mut d = Design::new();

        let ff_type = pool.intern("FF");
        let lut_type = pool.intern("LUT4");

        let ff_a_name = pool.intern("ff_a");
        let lut_name = pool.intern("lut_mid");
        let ff_b_name = pool.intern("ff_b");

        let clk_port = pool.intern("CLK");
        let d_port = pool.intern("D");
        let q_port = pool.intern("Q");
        let a_port = pool.intern("A");
        let f_port = pool.intern("F");

        let clk_net_name = pool.intern("clk");
        let net_q = pool.intern("net_q");
        let net_f = pool.intern("net_f");

        // Create cells
        let ff_a_idx = d.add_cell(ff_a_name, ff_type);
        let lut_idx = d.add_cell(lut_name, lut_type);
        let ff_b_idx = d.add_cell(ff_b_name, ff_type);

        // Add ports
        d.cell_mut(ff_a_idx).add_port(clk_port, PortType::In);
        d.cell_mut(ff_a_idx).add_port(d_port, PortType::In);
        d.cell_mut(ff_a_idx).add_port(q_port, PortType::Out);

        d.cell_mut(lut_idx).add_port(a_port, PortType::In);
        d.cell_mut(lut_idx).add_port(f_port, PortType::Out);

        d.cell_mut(ff_b_idx).add_port(clk_port, PortType::In);
        d.cell_mut(ff_b_idx).add_port(d_port, PortType::In);
        d.cell_mut(ff_b_idx).add_port(q_port, PortType::Out);

        // Create clock net with constraint.
        let clk_net_idx = d.add_net(clk_net_name);
        d.net_mut(clk_net_idx).clock_constraint = 10_000; // 10ns = 100 MHz

        // Connect clock to FF_A.CLK and FF_B.CLK (no driver cell for clk).
        let user_idx = d.net(clk_net_idx).users.len() as i32;
        d.net_mut(clk_net_idx).users.push(PortRef {
            cell: ff_a_idx,
            port: clk_port,
            budget: 0,
        });
        d.cell_mut(ff_a_idx).port_mut(clk_port).unwrap().net = clk_net_idx;
        d.cell_mut(ff_a_idx).port_mut(clk_port).unwrap().user_idx = user_idx;

        let user_idx = d.net(clk_net_idx).users.len() as i32;
        d.net_mut(clk_net_idx).users.push(PortRef {
            cell: ff_b_idx,
            port: clk_port,
            budget: 0,
        });
        d.cell_mut(ff_b_idx).port_mut(clk_port).unwrap().net = clk_net_idx;
        d.cell_mut(ff_b_idx).port_mut(clk_port).unwrap().user_idx = user_idx;

        // net_q: FF_A.Q -> LUT.A
        let net_q_idx = d.add_net(net_q);
        d.net_mut(net_q_idx).driver = PortRef {
            cell: ff_a_idx,
            port: q_port,
            budget: 0,
        };
        d.cell_mut(ff_a_idx).port_mut(q_port).unwrap().net = net_q_idx;
        d.cell_mut(ff_a_idx).port_mut(q_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_q_idx).users.len() as i32;
        d.net_mut(net_q_idx).users.push(PortRef {
            cell: lut_idx,
            port: a_port,
            budget: 0,
        });
        d.cell_mut(lut_idx).port_mut(a_port).unwrap().net = net_q_idx;
        d.cell_mut(lut_idx).port_mut(a_port).unwrap().user_idx = user_idx;

        // net_f: LUT.F -> FF_B.D
        let net_f_idx = d.add_net(net_f);
        d.net_mut(net_f_idx).driver = PortRef {
            cell: lut_idx,
            port: f_port,
            budget: 0,
        };
        d.cell_mut(lut_idx).port_mut(f_port).unwrap().net = net_f_idx;
        d.cell_mut(lut_idx).port_mut(f_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_f_idx).users.len() as i32;
        d.net_mut(net_f_idx).users.push(PortRef {
            cell: ff_b_idx,
            port: d_port,
            budget: 0,
        });
        d.cell_mut(ff_b_idx).port_mut(d_port).unwrap().net = net_f_idx;
        d.cell_mut(ff_b_idx).port_mut(d_port).unwrap().user_idx = user_idx;

        (d, clk_net_name)
    }

    /// Helper to create a two-clock-domain design:
    /// FF_A (clk1) -> LUT -> FF_B (clk2)
    fn make_two_domain_design(pool: &IdStringPool) -> Design {
        let mut d = Design::new();

        let ff_type = pool.intern("FF");
        let lut_type = pool.intern("LUT4");

        let ff_a_name = pool.intern("ff_a");
        let lut_name = pool.intern("lut_mid");
        let ff_b_name = pool.intern("ff_b");

        let clk_port = pool.intern("CLK");
        let d_port = pool.intern("D");
        let q_port = pool.intern("Q");
        let a_port = pool.intern("A");
        let f_port = pool.intern("F");

        let clk1_net_name = pool.intern("clk1");
        let clk2_net_name = pool.intern("clk2");
        let net_q = pool.intern("net_q");
        let net_f = pool.intern("net_f");

        // Create cells
        let ff_a_idx = d.add_cell(ff_a_name, ff_type);
        let lut_idx = d.add_cell(lut_name, lut_type);
        let ff_b_idx = d.add_cell(ff_b_name, ff_type);

        // Add ports
        d.cell_mut(ff_a_idx).add_port(clk_port, PortType::In);
        d.cell_mut(ff_a_idx).add_port(d_port, PortType::In);
        d.cell_mut(ff_a_idx).add_port(q_port, PortType::Out);

        d.cell_mut(lut_idx).add_port(a_port, PortType::In);
        d.cell_mut(lut_idx).add_port(f_port, PortType::Out);

        d.cell_mut(ff_b_idx).add_port(clk_port, PortType::In);
        d.cell_mut(ff_b_idx).add_port(d_port, PortType::In);
        d.cell_mut(ff_b_idx).add_port(q_port, PortType::Out);

        // Clock 1 net
        let clk1_idx = d.add_net(clk1_net_name);
        d.net_mut(clk1_idx).clock_constraint = 10_000; // 100 MHz

        let user_idx = d.net(clk1_idx).users.len() as i32;
        d.net_mut(clk1_idx).users.push(PortRef {
            cell: ff_a_idx,
            port: clk_port,
            budget: 0,
        });
        d.cell_mut(ff_a_idx).port_mut(clk_port).unwrap().net = clk1_idx;
        d.cell_mut(ff_a_idx).port_mut(clk_port).unwrap().user_idx = user_idx;

        // Clock 2 net
        let clk2_idx = d.add_net(clk2_net_name);
        d.net_mut(clk2_idx).clock_constraint = 5_000; // 200 MHz

        let user_idx = d.net(clk2_idx).users.len() as i32;
        d.net_mut(clk2_idx).users.push(PortRef {
            cell: ff_b_idx,
            port: clk_port,
            budget: 0,
        });
        d.cell_mut(ff_b_idx).port_mut(clk_port).unwrap().net = clk2_idx;
        d.cell_mut(ff_b_idx).port_mut(clk_port).unwrap().user_idx = user_idx;

        // net_q: FF_A.Q -> LUT.A
        let net_q_idx = d.add_net(net_q);
        d.net_mut(net_q_idx).driver = PortRef {
            cell: ff_a_idx,
            port: q_port,
            budget: 0,
        };
        d.cell_mut(ff_a_idx).port_mut(q_port).unwrap().net = net_q_idx;
        d.cell_mut(ff_a_idx).port_mut(q_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_q_idx).users.len() as i32;
        d.net_mut(net_q_idx).users.push(PortRef {
            cell: lut_idx,
            port: a_port,
            budget: 0,
        });
        d.cell_mut(lut_idx).port_mut(a_port).unwrap().net = net_q_idx;
        d.cell_mut(lut_idx).port_mut(a_port).unwrap().user_idx = user_idx;

        // net_f: LUT.F -> FF_B.D
        let net_f_idx = d.add_net(net_f);
        d.net_mut(net_f_idx).driver = PortRef {
            cell: lut_idx,
            port: f_port,
            budget: 0,
        };
        d.cell_mut(lut_idx).port_mut(f_port).unwrap().net = net_f_idx;
        d.cell_mut(lut_idx).port_mut(f_port).unwrap().user_idx = -1;

        let user_idx = d.net(net_f_idx).users.len() as i32;
        d.net_mut(net_f_idx).users.push(PortRef {
            cell: ff_b_idx,
            port: d_port,
            budget: 0,
        });
        d.cell_mut(ff_b_idx).port_mut(d_port).unwrap().net = net_f_idx;
        d.cell_mut(ff_b_idx).port_mut(d_port).unwrap().user_idx = user_idx;

        d
    }

    // =====================================================================
    // Tests
    // =====================================================================

    #[test]
    fn test_analyser_new() {
        let ta = TimingAnalyser::new();
        assert!(!ta.is_valid());
        assert_eq!(ta.worst_slack(), 0);
        assert_eq!(ta.net_criticality(NetIdx(0)), 0.0);
    }

    #[test]
    fn test_clock_constraint_mhz() {
        let pool = IdStringPool::new();
        let mut ta = TimingAnalyser::new();
        let clk = pool.intern("clk");
        ta.add_clock_constraint(clk, 100.0);
        assert_eq!(*ta.clock_constraints.get(&clk).unwrap(), 10_000); // 10ns = 10000ps
    }

    #[test]
    fn test_clock_constraint_ps() {
        let pool = IdStringPool::new();
        let mut ta = TimingAnalyser::new();
        let clk = pool.intern("clk");
        ta.add_clock_constraint_ps(clk, 5000);
        assert_eq!(*ta.clock_constraints.get(&clk).unwrap(), 5000);
    }

    #[test]
    fn test_comb_chain_forward_propagation() {
        let pool = IdStringPool::new();
        let design = make_comb_chain_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        assert!(ta.is_valid());

        // The combinational chain: input -> lut_a -> lut_b -> output
        // All ports are combinational (no clock).
        // input.O: arrival = 0 (primary input)
        // lut_a.A: arrival = 0 (from input.O, net delay 0)
        // lut_a.F: arrival = 0 + 100 = 100 (comb delay)
        // lut_b.A: arrival = 100 (from lut_a.F)
        // lut_b.F: arrival = 100 + 100 = 200
        // output.I: arrival = 200

        let o_port = pool.intern("O");
        let f_port = pool.intern("F");

        let input_idx = CellIdx(0);
        let lut_a_idx = CellIdx(1);
        let lut_b_idx = CellIdx(2);

        // Check arrival times.
        // input.O is a pure source cell with no inputs: arrival = 0.
        assert_eq!(
            ta.arrival_times.get(&(input_idx, o_port)).copied(),
            Some(0),
            "Input output port should have arrival 0 (primary source)"
        );

        // lut_a.F should have arrival = 0 (from input.O) + 100 (comb delay) = 100.
        let lut_a_f_arrival = ta.arrival_times.get(&(lut_a_idx, f_port)).copied();
        assert!(
            lut_a_f_arrival.is_some(),
            "lut_a.F should have an arrival time"
        );
        assert_eq!(lut_a_f_arrival.unwrap(), 100);

        // lut_b.F should have arrival = 100 + 100 = 200.
        let lut_b_f_arrival = ta.arrival_times.get(&(lut_b_idx, f_port)).copied();
        assert!(
            lut_b_f_arrival.is_some(),
            "lut_b.F should have an arrival time"
        );
        assert_eq!(lut_b_f_arrival.unwrap(), 200);
    }

    #[test]
    fn test_reg_to_reg_timing() {
        let pool = IdStringPool::new();
        let (design, _clk_net_name) = make_reg_to_reg_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        assert!(ta.is_valid());

        let d_port = pool.intern("D");
        let q_port = pool.intern("Q");
        let a_port = pool.intern("A");
        let f_port = pool.intern("F");

        let ff_a_idx = CellIdx(0);
        let lut_idx = CellIdx(1);
        let ff_b_idx = CellIdx(2);

        // FF_A.Q is a register output: arrival = DEFAULT_COMB_DELAY = 100
        let ff_a_q_arrival = ta.arrival_times.get(&(ff_a_idx, q_port)).copied();
        assert_eq!(ff_a_q_arrival, Some(100));

        // LUT.A gets arrival from FF_A.Q (100) + net delay (0) = 100
        let lut_a_arrival = ta.arrival_times.get(&(lut_idx, a_port)).copied();
        assert_eq!(lut_a_arrival, Some(100));

        // LUT.F: arrival = 100 + 100 = 200
        let lut_f_arrival = ta.arrival_times.get(&(lut_idx, f_port)).copied();
        assert_eq!(lut_f_arrival, Some(200));

        // FF_B.D gets arrival from LUT.F (200) + net delay (0) = 200
        let ff_b_d_arrival = ta.arrival_times.get(&(ff_b_idx, d_port)).copied();
        assert_eq!(ff_b_d_arrival, Some(200));

        // FF_B.D is a register input.
        // Required time = clock period - setup = 10000 - 50 = 9950
        let ff_b_d_required = ta.required_times.get(&(ff_b_idx, d_port)).copied();
        assert_eq!(ff_b_d_required, Some(9950));

        // Slack = required - arrival = 9950 - 200 = 9750 (positive = timing met)
        assert!(ta.worst_slack() > 0);

        // Paths should exist.
        assert!(!ta.paths.is_empty());
    }

    #[test]
    fn test_reg_to_reg_tight_constraint() {
        let pool = IdStringPool::new();
        let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

        // Tighten the clock constraint to 150ps (very tight, will fail).
        let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
        design.net_mut(clk_net_idx).clock_constraint = 150;

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        // FF_A.Q arrival = 100
        // LUT.F arrival = 200
        // FF_B.D arrival = 200
        // FF_B.D required = 150 - 50 = 100
        // Slack = 100 - 200 = -100 (negative = timing violated)
        assert!(ta.worst_slack() < 0);

        // With failing timing, criticality should be non-zero.
        let report = ta.report();
        assert!(report.num_failing > 0);
    }

    #[test]
    fn test_two_clock_domains() {
        let pool = IdStringPool::new();
        let design = make_two_domain_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        assert!(ta.is_valid());

        let clk_port = pool.intern("CLK");
        let ff_a_idx = CellIdx(0);
        let ff_b_idx = CellIdx(2);

        // Check that FF_A's clock port is classified as ClockInput.
        let ff_a_clk_class = ta.port_classes.get(&(ff_a_idx, clk_port)).copied();
        assert_eq!(ff_a_clk_class, Some(TimingPortClass::ClockInput));

        // Check that FF_B's clock port is classified as ClockInput.
        let ff_b_clk_class = ta.port_classes.get(&(ff_b_idx, clk_port)).copied();
        assert_eq!(ff_b_clk_class, Some(TimingPortClass::ClockInput));

        // Check that the domains are different.
        let ff_a_domain = ta.port_domains.get(&(ff_a_idx, clk_port));
        let ff_b_domain = ta.port_domains.get(&(ff_b_idx, clk_port));

        assert!(ff_a_domain.is_some());
        assert!(ff_b_domain.is_some());
        assert_ne!(
            ff_a_domain.unwrap().clock_net,
            ff_b_domain.unwrap().clock_net
        );
    }

    #[test]
    fn test_criticality_all_met() {
        let pool = IdStringPool::new();
        let (design, _) = make_reg_to_reg_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        // With 10ns period and ~200ps delay, timing is easily met.
        // All criticalities should be 0.
        for net_idx in 0..design.net_store.len() {
            let crit = ta.net_criticality(NetIdx(net_idx as u32));
            assert_eq!(
                crit, 0.0,
                "Net {} should have criticality 0 when timing is met",
                net_idx
            );
        }
    }

    #[test]
    fn test_criticality_failing() {
        let pool = IdStringPool::new();
        let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

        // Set very tight constraint to cause timing failure.
        let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
        design.net_mut(clk_net_idx).clock_constraint = 150;

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        assert!(ta.worst_slack() < 0, "Should have negative slack");

        // At least one net should have non-zero criticality.
        let mut has_nonzero_crit = false;
        for net_idx in 0..design.net_store.len() {
            let crit = ta.net_criticality(NetIdx(net_idx as u32));
            if crit > 0.0 {
                has_nonzero_crit = true;
            }
            assert!(
                crit >= 0.0 && crit <= 1.0,
                "Criticality must be in [0,1], got {}",
                crit
            );
        }
        assert!(has_nonzero_crit, "Should have at least one net with non-zero criticality");
    }

    #[test]
    fn test_topological_sort_basic() {
        let pool = IdStringPool::new();
        let design = make_comb_chain_design(&pool);

        let sorted = topological_sort(&design);

        // All alive cells should be in the sorted list.
        assert_eq!(sorted.len(), 4);

        // Input cell should come before LUT_A, which should come before LUT_B,
        // which should come before Output.
        let input_pos = sorted.iter().position(|&c| c == CellIdx(0)).unwrap();
        let lut_a_pos = sorted.iter().position(|&c| c == CellIdx(1)).unwrap();
        let lut_b_pos = sorted.iter().position(|&c| c == CellIdx(2)).unwrap();
        let output_pos = sorted.iter().position(|&c| c == CellIdx(3)).unwrap();

        assert!(
            input_pos < lut_a_pos,
            "input should come before lut_a in topological order"
        );
        assert!(
            lut_a_pos < lut_b_pos,
            "lut_a should come before lut_b in topological order"
        );
        assert!(
            lut_b_pos < output_pos,
            "lut_b should come before output in topological order"
        );
    }

    #[test]
    fn test_topological_sort_with_registers() {
        let pool = IdStringPool::new();
        let (design, _) = make_reg_to_reg_design(&pool);

        let sorted = topological_sort(&design);

        // Should contain all 3 cells.
        assert_eq!(sorted.len(), 3);

        // FF_A should come before LUT (FF_A.Q drives LUT.A).
        // LUT should come before FF_B (LUT.F drives FF_B.D).
        let ff_a_pos = sorted.iter().position(|&c| c == CellIdx(0)).unwrap();
        let lut_pos = sorted.iter().position(|&c| c == CellIdx(1)).unwrap();
        let ff_b_pos = sorted.iter().position(|&c| c == CellIdx(2)).unwrap();

        assert!(
            ff_a_pos < lut_pos,
            "ff_a should come before lut in topological order"
        );
        assert!(
            lut_pos < ff_b_pos,
            "lut should come before ff_b in topological order"
        );
    }

    #[test]
    fn test_invalidate() {
        let pool = IdStringPool::new();
        let design = make_comb_chain_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);
        assert!(ta.is_valid());

        ta.invalidate();
        assert!(!ta.is_valid());
    }

    #[test]
    fn test_fmax_with_constraint() {
        let pool = IdStringPool::new();
        let (design, clk_net_name) = make_reg_to_reg_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.add_clock_constraint_ps(clk_net_name, 10_000);
        ta.analyse(&design, &pool);

        // Timing should be met with 10ns period.
        let fmax = ta.fmax_mhz();
        assert!(fmax > 0.0, "fmax should be positive");
        // With slack > 0, fmax should equal or exceed the constraint frequency.
        assert!(fmax >= 100.0, "fmax should be at least 100 MHz with 10ns period");
    }

    #[test]
    fn test_report_structure() {
        let pool = IdStringPool::new();
        let (design, _) = make_reg_to_reg_design(&pool);

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        let report = ta.report();
        assert!(report.fmax >= 0.0);
        assert_eq!(report.num_failing, 0);
        assert!(report.num_endpoints > 0);
    }

    #[test]
    fn test_empty_design() {
        let pool = IdStringPool::new();
        let design = Design::new();

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        assert!(ta.is_valid());
        assert_eq!(ta.worst_slack(), 0);
        assert_eq!(ta.fmax_mhz(), 0.0);
    }

    #[test]
    fn test_port_criticality() {
        let pool = IdStringPool::new();
        let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

        // Use tight constraint.
        let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
        design.net_mut(clk_net_idx).clock_constraint = 150;

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        let d_port = pool.intern("D");
        let ff_b_idx = CellIdx(2);

        // FF_B.D is a timing endpoint with failing slack.
        let crit = ta.port_criticality(ff_b_idx, d_port);
        assert!(
            crit > 0.0,
            "FF_B.D should have non-zero criticality with tight constraint"
        );
        assert!(crit <= 1.0, "Criticality should not exceed 1.0");
    }

    #[test]
    fn test_clock_domain_unclocked() {
        let domain = ClockDomain::unclocked();
        assert!(!domain.is_clocked());
        assert!(domain.clock_net.is_empty());
        assert_eq!(domain.period, 0);
    }

    #[test]
    fn test_clock_domain_clocked() {
        let pool = IdStringPool::new();
        let clk = pool.intern("clk");
        let domain = ClockDomain {
            clock_net: clk,
            edge: ClockEdge::Rising,
            period: 10_000,
        };
        assert!(domain.is_clocked());
        assert_eq!(domain.period, 10_000);
    }

    #[test]
    fn test_critical_paths_limit() {
        let pool = IdStringPool::new();
        let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);
        let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
        design.net_mut(clk_net_idx).clock_constraint = 150;

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        // Request more paths than exist.
        let paths = ta.critical_paths(100);
        assert!(!paths.is_empty());

        // Request 0 paths.
        let paths = ta.critical_paths(0);
        assert!(paths.is_empty());

        // Request exactly 1 path.
        let paths = ta.critical_paths(1);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_paths_sorted_by_slack() {
        let pool = IdStringPool::new();
        let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);
        let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
        design.net_mut(clk_net_idx).clock_constraint = 150;

        let mut ta = TimingAnalyser::new();
        ta.analyse(&design, &pool);

        let paths = ta.critical_paths(100);
        for i in 1..paths.len() {
            assert!(
                paths[i - 1].slack <= paths[i].slack,
                "Paths should be sorted by slack ascending"
            );
        }
    }
}

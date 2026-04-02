//! Port initialization, topological sort, and classification.

use rustc_hash::FxHashMap;

use super::analyser::{PerPort, TimingAnalyser};
use super::delay::DelayT;
use super::domain::{CellArcType, ClockDomain};
use super::kinds::{ClockEdge, TimingPortClass};
use crate::common::IdString;
use crate::netlist::{CellPin, Design, PortType};

impl TimingAnalyser {
    /// Initialize per-port structures from the design netlist.
    pub(super) fn init_ports(&mut self, design: &Design) {
        for (cell_idx, cell) in design.iter_alive_cells() {
            for (port_name, port_info) in &cell.ports {
                let pin = CellPin::new(cell_idx, *port_name);
                self.port_data
                    .insert(pin, PerPort::new(port_info.port_type()));
            }
        }
    }

    /// Topological sort at the port level (not cell level).
    pub(super) fn topo_sort_ports(&mut self, design: &Design) {
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

    /// Heuristic port classification for legacy `analyse()` path.
    pub(super) fn classify_ports_heuristic(&mut self, design: &Design) {
        use rustc_hash::FxHashSet;

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
                    self.clock_constraints
                        .get(&net.name)
                        .copied()
                        .filter(|&p| p > 0)
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

    /// Look up a port's timing class from cached port_data, defaulting to Combinational.
    pub(super) fn port_class_or_comb(&self, pin: CellPin) -> TimingPortClass {
        self.port_data
            .get(&pin)
            .map(|pd| pd.port_class)
            .unwrap_or(TimingPortClass::Combinational)
    }

    /// Get clock domain for a pin from cached port data.
    pub(super) fn port_domain_from_data(&self, pin: CellPin) -> ClockDomain {
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
    pub(super) fn port_domain_period(&self, pin: CellPin) -> DelayT {
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
}

//! STA forward and backward propagation.

use super::analyser::{TimingAnalyser, DEFAULT_COMB_DELAY};
use super::delay::DelayT;
use super::domain::CellArcType;
use super::kinds::TimingPortClass;
use crate::netlist::{CellId, CellPin, Design, PortType};

impl TimingAnalyser {
    // =====================================================================
    // Forward propagation (uses CellArc cache)
    // =====================================================================

    pub(super) fn forward_propagation(&mut self, design: &Design) {
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
    // Backward propagation (uses CellArc cache)
    // =====================================================================

    pub(super) fn backward_propagation(&mut self, design: &Design) {
        let with_skew = self.with_clock_skew;

        // Initialize required times at endpoints.
        for dom_idx in 0..self.per_domain.len() {
            // Get the period for this domain.
            let domain_period = self.domain_registry.get(super::domain::ClockDomainId(dom_idx as u32)).period;
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
    // Legacy forward/backward (for analyse() without chipdb)
    // =====================================================================

    pub(super) fn forward_propagation_legacy(&mut self, design: &Design, sorted_cells: &[CellId]) {
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

    pub(super) fn backward_propagation_legacy(&mut self, design: &Design, sorted_cells: &[CellId]) {
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

    /// Recursively find upstream drivers of a net through combinational logic.
    pub(super) fn find_net_drivers(
        &self,
        ctx: &crate::context::Context,
        net_idx: crate::netlist::NetId,
        visited: &mut rustc_hash::FxHashSet<IdString>,
        drivers: &mut rustc_hash::FxHashMap<IdString, DelayT>,
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
                a.arc_type == super::domain::CellArcType::Combinational && a.other_port == driver_port
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
}

use crate::common::IdString;

//! Clock domain management: setup, resolution, and related domain identification.

use rustc_hash::{FxHashMap, FxHashSet};

use super::analyser::{PerDomain, PerDomainPair, TimingAnalyser};
use super::delay::{DelayPair, DelayT};
use super::domain::{CellArcType, ClockDomainId, ClockDomainPair};
use super::kinds::ClockEdge;
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellId, CellPin, PortType};

impl TimingAnalyser {
    /// Get or create a domain pair ID.
    pub(super) fn domain_pair_id(&mut self, launch: ClockDomainId, capture: ClockDomainId) -> usize {
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

    /// Assign clock domains to ports via fixed-point iteration.
    ///
    /// Following the C++ `setup_port_domains()` pattern:
    /// 1. Forward pass: registered outputs are startpoints; propagate domains forward.
    /// 2. Backward pass: registered inputs are endpoints; propagate domains backward.
    /// 3. Compute domain pairs at each port.
    pub(super) fn setup_port_domains(&mut self, ctx: &Context) {
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
    pub(super) fn resolve_domain_id(
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

    /// Identify related clock domains by tracing upstream through combinational logic.
    pub(super) fn identify_related_domains(&mut self, ctx: &Context) {
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

    /// Get the default clock period (smallest constrained period, or 10ns).
    pub(super) fn get_default_period(&self) -> DelayT {
        self.clock_constraints
            .values()
            .copied()
            .min()
            .unwrap_or(10_000)
    }
}

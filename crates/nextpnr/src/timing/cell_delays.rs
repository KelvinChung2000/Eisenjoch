//! Cell delay computation from chipdb timing data.

use super::analyser::TimingAnalyser;
use super::delay::{DelayPair, DelayQuad};
use super::domain::CellArc;
use super::kinds::TimingPortClass;
use crate::chipdb::ChipDb;
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellPin, PortType};

impl TimingAnalyser {
    /// Cache all cell timing arcs from chipdb, following the C++ pattern.
    pub(super) fn get_cell_delays(&mut self, ctx: &Context) {
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
            let type_idx = match cell.timing_index.map(|ti| ti.0 as usize).or_else(|| {
                ctx.chipdb()
                    .cell_timing_index(speed_grade, cell.cell_type.index())
            }) {
                Some(idx) => idx,
                None => continue,
            };

            let port_class =
                ctx.chipdb()
                    .port_timing_class(speed_grade, type_idx, pin.port.index(), port_type);

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
                                ctx.chipdb()
                                    .cell_reg_arcs(speed_grade, type_idx, pin.port.index())
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
                                ctx.chipdb()
                                    .cell_reg_arcs(speed_grade, type_idx, pin.port.index())
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
    pub(super) fn get_cell_delays_heuristic(&mut self, design: &crate::netlist::Design) {
        use rustc_hash::FxHashSet;

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
                    || self
                        .clock_constraints
                        .get(&net.name)
                        .is_some_and(|&p| p > 0);
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

    /// Update route delays for input ports from the routed design.
    pub(super) fn get_route_delays(&mut self, ctx: &Context) {
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
}

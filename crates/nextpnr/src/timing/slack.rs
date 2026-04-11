//! Slack computation and path reconstruction.

use rustc_hash::FxHashSet;

use super::analyser::TimingAnalyser;
use super::delay::DelayT;
use super::kinds::TimingPortClass;
use super::path::{PathSegment, TimingEndpoint, TimingPath};
use crate::netlist::{CellPin, Design, NetId};

impl TimingAnalyser {
    pub(super) fn compute_slack_and_paths(&mut self, design: &Design) {
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
                        let from_domain = self
                            .port_domain_from_data(CellPin::new(first_seg.cell, first_seg.port));
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
    pub(super) fn reconstruct_path(&self, endpoint: CellPin) -> Vec<PathSegment> {
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

    pub(super) fn compute_criticality(&mut self, design: &Design) {
        let mut min_slack = DelayT::MAX;
        let mut max_slack = DelayT::MIN;

        for (_, net) in design.iter_alive_nets() {
            for user in &net.users {
                if !user.is_valid() {
                    continue;
                }
                let user_pin = CellPin::new(user.cell, user.port);
                let arrival = self.arrival_times.get(&user_pin).copied().unwrap_or(0);
                let required = self.required_times.get(&user_pin).copied().unwrap_or(0);
                let slack = required - arrival;
                min_slack = min_slack.min(slack);
                max_slack = max_slack.max(slack);
            }
        }

        if min_slack == DelayT::MAX {
            return;
        }

        let slack_span = (max_slack - min_slack).max(1) as f64;

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
                let crit = (1.0 - ((slack - min_slack) as f64 / slack_span)).clamp(0.0, 1.0) as f32;
                if crit > max_crit {
                    max_crit = crit;
                }
            }
            self.net_criticality.insert(net_idx, max_crit);
        }
    }
}

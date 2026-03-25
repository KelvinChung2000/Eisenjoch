//! Public accessor/query methods and orchestration entry points.

use rustc_hash::FxHashMap;

use super::analyser::TimingAnalyser;
use super::delay::DelayT;
use super::domain::{ClockDomain, DomainRegistry};
use super::path::{TimingPath, TimingReport};
use super::sort::topological_sort;
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::{CellId, CellPin, Design, NetId};
use log::debug;

impl TimingAnalyser {
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
    pub fn analyse(&mut self, design: &Design, _id_pool: &crate::common::IdStringPool) {
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

    /// Access the domain registry.
    pub fn domain_registry(&self) -> &DomainRegistry {
        &self.domain_registry
    }

    /// Access clock-to-clock delays.
    pub fn clock_delays(&self) -> &FxHashMap<(IdString, IdString), DelayT> {
        &self.clock_delays
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
    pub fn port_class(&self, cell: CellId, port: IdString) -> Option<super::kinds::TimingPortClass> {
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

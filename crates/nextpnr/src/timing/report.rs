//! Vivado-style timing report generation.

use crate::common::{IdString, IdStringPool};
use crate::netlist::NetId;
use crate::timing::DelayT;

use super::path::TimingPath;
use super::TimingAnalyser;

/// Format a delay in picoseconds as nanoseconds with 3 decimal places.
fn ps_to_ns(ps: DelayT) -> f64 {
    ps as f64 / 1000.0
}

/// Per-clock summary row.
pub struct ClockSummary {
    pub name: IdString,
    pub period_ps: DelayT,
    pub wns: DelayT,
    pub tns: DelayT,
    pub failing: usize,
    pub total: usize,
}

/// Full timing summary matching Vivado format.
pub struct TimingSummary {
    pub wns: DelayT,
    pub tns: DelayT,
    pub failing: usize,
    pub total: usize,
    pub fmax_mhz: f64,
    pub clock_summaries: Vec<ClockSummary>,
}

impl TimingSummary {
    /// Generate a timing summary from an analyser.
    pub fn from_analyser(analyser: &TimingAnalyser) -> Self {
        let report = analyser.report();
        let tns: DelayT = report
            .critical_paths
            .iter()
            .filter(|p| p.slack < 0)
            .map(|p| p.slack)
            .sum();

        let mut clock_summaries = Vec::new();
        for (&clk_name, &period) in analyser.clock_constraints() {
            let paths_for_clk: Vec<&TimingPath> = report
                .critical_paths
                .iter()
                .filter(|p| p.to.domain.clock_net == clk_name)
                .collect();
            let clk_wns = paths_for_clk.iter().map(|p| p.slack).min().unwrap_or(0);
            let clk_tns: DelayT = paths_for_clk
                .iter()
                .filter(|p| p.slack < 0)
                .map(|p| p.slack)
                .sum();
            let clk_failing = paths_for_clk.iter().filter(|p| p.slack < 0).count();
            clock_summaries.push(ClockSummary {
                name: clk_name,
                period_ps: period,
                wns: clk_wns,
                tns: clk_tns,
                failing: clk_failing,
                total: paths_for_clk.len(),
            });
        }

        Self {
            wns: report.worst_slack,
            tns,
            failing: report.num_failing,
            total: report.num_endpoints,
            fmax_mhz: report.fmax,
            clock_summaries,
        }
    }

    /// Format as a Vivado-style timing summary string.
    pub fn format(&self, pool: &IdStringPool) -> String {
        let mut out = String::new();
        out.push_str("Timing Summary\n");
        out.push_str(&format!(
            "{:>10} {:>10} {:>10} {:>10}\n",
            "WNS(ns)", "TNS(ns)", "Failing", "Total"
        ));
        out.push_str(&format!(
            "{:>10.3} {:>10.3} {:>10} {:>10}\n",
            ps_to_ns(self.wns),
            ps_to_ns(self.tns),
            self.failing,
            self.total,
        ));
        if self.fmax_mhz > 0.0 {
            out.push_str(&format!("Fmax: {:.2} MHz\n", self.fmax_mhz));
        }

        if !self.clock_summaries.is_empty() {
            out.push_str("\nClock Summary\n");
            out.push_str(&format!(
                "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
                "Clock", "Freq(MHz)", "Period(ns)", "WNS(ns)", "TNS(ns)", "Failing"
            ));
            for cs in &self.clock_summaries {
                let freq = if cs.period_ps > 0 {
                    1_000_000.0 / cs.period_ps as f64
                } else {
                    0.0
                };
                out.push_str(&format!(
                    "{:<20} {:>10.2} {:>10.3} {:>10.3} {:>10.3} {:>10}\n",
                    pool.lookup(cs.name).unwrap_or("?"),
                    freq,
                    ps_to_ns(cs.period_ps),
                    ps_to_ns(cs.wns),
                    ps_to_ns(cs.tns),
                    cs.failing,
                ));
            }
        }

        out
    }
}

/// Format a detailed path report for a single timing path.
pub fn format_path_detail(path: &TimingPath, pool: &IdStringPool) -> String {
    let mut out = String::new();

    let from_cell = pool.lookup(path.from.port).unwrap_or("?");
    let to_cell = pool.lookup(path.to.port).unwrap_or("?");
    let from_domain = if path.from.domain.is_clocked() {
        format!(
            "rising edge of {}",
            pool.lookup(path.from.domain.clock_net).unwrap_or("?")
        )
    } else {
        "unclocked".to_string()
    };
    let to_domain = if path.to.domain.is_clocked() {
        format!(
            "rising edge of {}",
            pool.lookup(path.to.domain.clock_net).unwrap_or("?")
        )
    } else {
        "unclocked".to_string()
    };

    out.push_str(&format!("Startpoint: {} ({})\n", from_cell, from_domain));
    out.push_str(&format!("Endpoint:   {} ({})\n", to_cell, to_domain));

    out.push_str("Path Type:  Setup (max)\n\n");

    if !path.segments.is_empty() {
        out.push_str(&format!(
            "{:>10} {:>12}  {}\n",
            "Delay", "Cumulative", "Description"
        ));
        let mut cumulative: DelayT = 0;
        for seg in &path.segments {
            cumulative += seg.delay;
            let port_name = pool.lookup(seg.port).unwrap_or("?");
            let via = if seg.net != NetId::NONE {
                "net"
            } else {
                "cell"
            };
            out.push_str(&format!(
                "{:>10.3} {:>12.3}  {} ({})\n",
                ps_to_ns(seg.delay),
                ps_to_ns(cumulative),
                port_name,
                via,
            ));
        }
        out.push('\n');
    }

    out.push_str(&format!(
        "Data arrival time: {:.3} ns\n",
        ps_to_ns(path.delay)
    ));
    out.push_str(&format!(
        "Required time:     {:.3} ns\n",
        ps_to_ns(path.budget)
    ));
    let status = if path.slack >= 0 { "MET" } else { "VIOLATED" };
    out.push_str(&format!(
        "Slack:             {:.3} ns ({})\n",
        ps_to_ns(path.slack),
        status
    ));

    out
}

/// Format a cross-domain report showing all domain pair relationships.
pub fn format_cross_domain_report(analyser: &TimingAnalyser, pool: &IdStringPool) -> String {
    let mut out = String::new();
    out.push_str("Cross-Domain Paths\n");
    out.push_str(&format!(
        "{:<20} {:<20} {}\n",
        "From Clock", "To Clock", "Status"
    ));

    let registry = analyser.domain_registry();
    let clock_delays = analyser.clock_delays();

    for (launch_id, launch_dom) in registry.iter() {
        if !launch_dom.is_clocked() {
            continue;
        }
        for (capture_id, capture_dom) in registry.iter() {
            if !capture_dom.is_clocked() || launch_id == capture_id {
                continue;
            }
            let launch_name = pool.lookup(launch_dom.clock_net).unwrap_or("?");
            let capture_name = pool.lookup(capture_dom.clock_net).unwrap_or("?");

            let status =
                if clock_delays.contains_key(&(launch_dom.clock_net, capture_dom.clock_net)) {
                    "Constrained (related)"
                } else if launch_dom.clock_net == capture_dom.clock_net {
                    "Same clock"
                } else {
                    "Unconstrained (async)"
                };

            out.push_str(&format!(
                "{:<20} {:<20} {}\n",
                launch_name, capture_name, status
            ));
        }
    }

    out
}

/// Format a constraint coverage report.
pub fn format_constraint_coverage(analyser: &TimingAnalyser) -> String {
    let sdc = &analyser.sdc;
    format!(
        "Constraint Coverage\n\
         Clocks defined:     {}\n\
         Input delays:       {}\n\
         Output delays:      {}\n\
         False paths:        {}\n\
         Multicycle paths:   {}\n",
        analyser.clock_constraints().len(),
        sdc.input_delays.len(),
        sdc.output_delays.len(),
        sdc.false_paths.len(),
        sdc.multicycle_paths.len(),
    )
}

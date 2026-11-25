use std::collections::BTreeMap;
use std::fmt;

use ruststa::engine::incr::{PathPin, PathStart, ReportedPath};
use ruststa::engine::types::{An, NodeId, Tr};

use super::session::{DomainSummary, StaSession, StaSessionError};
use crate::common::ClockEdge;
use crate::context::Context;
use crate::netlist::CellPin;

const MHZ_NS: f64 = 1000.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PathPinReport {
    pub cell_pin: CellPin,
    pub pin: String,
    pub arrival_ns: f64,
    /// Delay of the arc into this pin. On the first pin, its seeded arrival or launch delay.
    pub delay_ns: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathReport {
    pub slack_ns: f64,
    pub launch_edge: ClockEdge,
    pub capture_edge: ClockEdge,
    /// The launching register's clock pin. `None` when an input delay or a clock source seeds the path.
    pub launch_clock_pin: Option<CellPin>,
    pub pins: Vec<PathPinReport>,
    /// Check relation less required time: the capture setup time plus clock uncertainty.
    pub setup_ns: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DomainReport {
    pub launch: String,
    pub capture: String,
    pub summary: DomainSummary,
    pub capture_period_ns: f64,
    pub fmax_mhz: Option<f64>,
    pub worst_paths: Vec<PathReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldReport {
    pub launch: String,
    pub capture: String,
    pub summary: DomainSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimingReport {
    pub domains: Vec<DomainReport>,
    pub hold: Vec<HoldReport>,
    pub black_boxes: BTreeMap<String, usize>,
}

impl TimingReport {
    pub fn wns_ns(&self) -> Option<f64> {
        self.domains
            .iter()
            .map(|domain| domain.summary.wns_ns)
            .reduce(f64::min)
    }

    pub fn tns_ns(&self) -> f64 {
        self.domains.iter().map(|domain| domain.summary.tns_ns).sum()
    }

    pub fn fmax_mhz(&self) -> Option<f64> {
        self.domains
            .iter()
            .filter_map(|domain| domain.fmax_mhz)
            .reduce(f64::min)
    }

    pub fn endpoints(&self) -> usize {
        self.domains.iter().map(|domain| domain.summary.endpoints).sum()
    }

    pub fn failing_endpoints(&self) -> usize {
        self.domains
            .iter()
            .map(|domain| domain.summary.failing_endpoints)
            .sum()
    }

    pub fn hold_wns_ns(&self) -> Option<f64> {
        self.hold.iter().map(|pair| pair.summary.wns_ns).reduce(f64::min)
    }

    pub fn hold_tns_ns(&self) -> f64 {
        self.hold.iter().map(|pair| pair.summary.tns_ns).sum()
    }

    pub fn hold_endpoints(&self) -> usize {
        self.hold.iter().map(|pair| pair.summary.endpoints).sum()
    }

    pub fn hold_failing_endpoints(&self) -> usize {
        self.hold.iter().map(|pair| pair.summary.failing_endpoints).sum()
    }
}

impl fmt::Display for TimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for domain in &self.domains {
            let summary = &domain.summary;
            write!(
                f,
                "{} -> {}: WNS {:.3} ns, TNS {:.3} ns, {}/{} endpoints failing",
                domain.launch,
                domain.capture,
                summary.wns_ns,
                summary.tns_ns,
                summary.failing_endpoints,
                summary.endpoints,
            )?;
            match domain.fmax_mhz {
                Some(fmax) => writeln!(f, ", Fmax {fmax:.2} MHz")?,
                None => writeln!(f)?,
            }
            for path in &domain.worst_paths {
                writeln!(f, "  slack {:.3} ns", path.slack_ns)?;
                for pin in &path.pins {
                    writeln!(f, "    {:>10.3} ns  {}", pin.arrival_ns, pin.pin)?;
                }
            }
        }
        for pair in &self.hold {
            let summary = &pair.summary;
            writeln!(
                f,
                "{} -> {} hold: WNS {:.3} ns, TNS {:.3} ns, {}/{} endpoints failing",
                pair.launch,
                pair.capture,
                summary.wns_ns,
                summary.tns_ns,
                summary.failing_endpoints,
                summary.endpoints,
            )?;
        }
        for (cell_type, cells) in &self.black_boxes {
            writeln!(f, "untimed black box: {cells} cells of type {cell_type}")?;
        }
        Ok(())
    }
}

impl StaSession {
    pub fn report(
        &self,
        ctx: &Context,
        paths_per_pair: usize,
    ) -> Result<TimingReport, StaSessionError> {
        let session = self.sta().session();
        let clocks = &session.state().constraints.clocks;
        let corner = &session.default_corner().corner_name;
        let pair_paths = session.worst_paths(corner, An::Late, paths_per_pair)?;
        let domains = self
            .domains()
            .iter()
            .map(|summary| {
                let paths = pair_paths
                    .iter()
                    .find(|pair| (pair.launch, pair.capture) == (summary.launch, summary.capture))
                    .map(|pair| &pair.paths[..])
                    .expect("RustSTA reports every domain pair with setup checks");
                let fmax_mhz = (summary.launch == summary.capture).then(|| {
                    MHZ_NS / (clocks[summary.launch.index()].period_ns - summary.wns_ns)
                });
                DomainReport {
                    launch: clocks[summary.launch.index()].name.clone(),
                    capture: clocks[summary.capture.index()].name.clone(),
                    summary: *summary,
                    capture_period_ns: clocks[summary.capture.index()].period_ns,
                    fmax_mhz,
                    worst_paths: paths.iter().map(|path| self.path_report(ctx, path)).collect(),
                }
            })
            .collect();
        let hold = self
            .hold_domains()
            .iter()
            .map(|summary| HoldReport {
                launch: clocks[summary.launch.index()].name.clone(),
                capture: clocks[summary.capture.index()].name.clone(),
                summary: *summary,
            })
            .collect();
        Ok(TimingReport {
            domains,
            hold,
            black_boxes: self.sta().black_boxes().clone(),
        })
    }

    fn path_report(&self, ctx: &Context, path: &ReportedPath) -> PathReport {
        let (launch_clock_pin, start_delay_ns) = match path.launch {
            PathStart::Seed => (None, path.start.arrival_ns),
            PathStart::Launch {
                clock_pin,
                delay_ns,
                ..
            } => (Some(self.cell_pin(clock_pin)), delay_ns),
        };
        let pins = std::iter::once((&path.start, start_delay_ns))
            .chain(path.steps.iter().map(|step| (&step.pin, step.delay_ns)))
            .map(|(pin, delay_ns)| self.pin_report(ctx, pin, delay_ns))
            .collect();
        PathReport {
            slack_ns: path.record.slack,
            launch_edge: clock_edge(path.record.launch_edge),
            capture_edge: clock_edge(path.record.capture_edge),
            launch_clock_pin,
            pins,
            setup_ns: path.record.relation - path.record.required,
        }
    }

    fn pin_report(&self, ctx: &Context, pin: &PathPin, delay_ns: f64) -> PathPinReport {
        let cell_pin = self.cell_pin(pin.node);
        PathPinReport {
            cell_pin,
            pin: format!(
                "{}/{}",
                ctx.name_of(ctx.design.cell(cell_pin.cell).name),
                ctx.name_of(cell_pin.port),
            ),
            arrival_ns: pin.arrival_ns,
            delay_ns,
        }
    }

    fn cell_pin(&self, node: NodeId) -> CellPin {
        self.sta().ids().node_pins[node.index()]
            .expect("every timing graph node is a design pin or port")
    }
}

fn clock_edge(tr: Tr) -> ClockEdge {
    match tr {
        Tr::Rise => ClockEdge::Rising,
        Tr::Fall => ClockEdge::Falling,
    }
}

use std::collections::BTreeMap;
use std::fmt;

use ruststa::engine::incr::ReportedPath;
use ruststa::engine::types::An;

use super::session::{DomainSummary, StaSession, StaSessionError};
use crate::context::Context;

const MHZ_NS: f64 = 1000.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PathPinReport {
    pub pin: String,
    pub arrival_ns: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathReport {
    pub slack_ns: f64,
    pub pins: Vec<PathPinReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DomainReport {
    pub launch: String,
    pub capture: String,
    pub summary: DomainSummary,
    pub fmax_mhz: Option<f64>,
    pub worst_paths: Vec<PathReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimingReport {
    pub domains: Vec<DomainReport>,
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
                    fmax_mhz,
                    worst_paths: paths.iter().map(|path| self.path_report(ctx, path)).collect(),
                }
            })
            .collect();
        Ok(TimingReport {
            domains,
            black_boxes: self.sta().black_boxes().clone(),
        })
    }

    fn path_report(&self, ctx: &Context, path: &ReportedPath) -> PathReport {
        let node_pins = &self.sta().ids().node_pins;
        let pins = std::iter::once(&path.start)
            .chain(path.steps.iter().map(|step| &step.pin))
            .map(|pin| {
                let cell_pin = node_pins[pin.node.index()]
                    .expect("every timing graph node is a design pin or port");
                PathPinReport {
                    pin: format!(
                        "{}/{}",
                        ctx.name_of(ctx.design.cell(cell_pin.cell).name),
                        ctx.name_of(cell_pin.port),
                    ),
                    arrival_ns: pin.arrival_ns,
                }
            })
            .collect();
        PathReport {
            slack_ns: path.record.slack,
            pins,
        }
    }
}

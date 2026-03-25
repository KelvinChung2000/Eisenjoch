//! Static timing analysis engine for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module performs static timing analysis (STA) on a placed design to determine
//! whether it meets frequency constraints and to provide criticality values used by
//! timing-driven placement and routing.
//!
//! The central type is [`TimingAnalyser`], which performs forward and backward
//! propagation through the netlist to compute arrival times, required times, slack,
//! and criticality for every net.

mod analyser;
mod cell_delays;
mod domain_setup;
pub mod oracle;
mod ports;
mod propagation;
mod queries;
mod slack;

pub mod constraints;
pub mod delay;
pub mod domain;
pub mod kinds;
pub mod path;
pub mod report;
pub mod sort;

pub use analyser::TimingAnalyser;
pub use oracle::TimingOracle;
pub use constraints::SdcConstraints;
pub use delay::{DelayPair, DelayQuad, DelayT};
pub use domain::{
    CellArc, CellArcType, ClockDomain, ClockDomainId, ClockDomainPair, DomainRegistry,
};
pub use kinds::{ClockEdge, TimingPortClass};
pub use path::{PathSegment, TimingEndpoint, TimingPath, TimingPortInfo, TimingReport};
pub use report::{
    format_constraint_coverage, format_cross_domain_report, format_path_detail, TimingSummary,
};
pub use sort::topological_sort;

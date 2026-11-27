//! The default place-and-route flow of `nextpnr-himbaechel-rust`, as one call.
//!
//! A program that links Eisenjoch runs [`run`] on a thread of its own. It packs
//! the netlist, constrains every register clock, places with HeAP, refines,
//! routes with Router2, runs routed timing analysis, writes the outputs that
//! [`FlowOptions`] names, and checks timing. A [`Live`] it passes watches the
//! state as it changes and can cancel the flow between and within steps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::chipdb::ChipDb;
use crate::common::DelayT;
use crate::context::Context;
use crate::fasm::{fasm_features, fasm_text, FasmError};
use crate::live::{Cancelled, Live, LiveProgress};
use crate::netlist::NetId;
use crate::placer::heap::{PlacerHeap, PlacerHeapCfg};
use crate::placer::{refine_placement, Placer, PlacerError, PlacerPipeline, RefineCfg};
use crate::report_json::report_json;
use crate::routed_json::{routed_json, RoutedJsonError};
use crate::router::common::collect_routable_nets;
use crate::router::router2::{Router2Cfg, TimingDrivenCfg};
use crate::router::{Router, Router2, RouterError};
use crate::sta::{
    register_clock_nets, CritNorm, InterconnectModel, InterconnectStage, PlacedEstimate,
    StaSession, StaSessionError, TimingReport, NOTIONAL_CLASSLESS_PIPS,
};

/// The clock target, in MHz, of a flow given no frequency, as nextpnr's default.
pub const DEFAULT_TARGET_MHZ: f64 = 12.0;
const PS_PER_US: f64 = 1_000_000.0;
const HEAP_CONGESTION_WEIGHT: f64 = 1.0;

/// What the flow reads, how it times the design, and what it writes.
#[derive(Debug, Clone)]
pub struct FlowOptions {
    pub chipdb: PathBuf,
    /// The synthesised Yosys JSON netlist.
    pub netlist: PathBuf,
    /// The clock target for every register clock; [`DEFAULT_TARGET_MHZ`] when `None`.
    pub freq_mhz: Option<f64>,
    pub seed: u64,
    /// Refine and route by STA criticality.
    pub timing_driven: bool,
    /// Succeed even when setup or hold slack is negative.
    pub allow_timing_failure: bool,
    /// Where to write the routed netlist as nextpnr JSON.
    pub routed: Option<PathBuf>,
    pub fasm: Option<PathBuf>,
    /// Where to write the timing and utilisation report in nextpnr's `--report` schema.
    pub report: Option<PathBuf>,
}

impl FlowOptions {
    pub fn new(chipdb: impl Into<PathBuf>, netlist: impl Into<PathBuf>) -> Self {
        Self {
            chipdb: chipdb.into(),
            netlist: netlist.into(),
            freq_mhz: None,
            seed: 1,
            timing_driven: true,
            allow_timing_failure: false,
            routed: None,
            fasm: None,
            report: None,
        }
    }
}

/// What a flow that finished found.
#[derive(Debug, Clone)]
pub struct FlowSummary {
    /// The nets the flow constrained as clocks.
    pub clocks: Vec<String>,
    /// The routed timing report as text.
    pub timing: String,
    /// One message per failed timing check, accepted because the options allow it.
    pub timing_failures: Vec<String>,
}

#[derive(Debug, Error)]
pub enum FlowError {
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
    #[error("Load the chipdb {path}: {message}")]
    Chipdb { path: PathBuf, message: String },
    #[error("Read the netlist {path}: {message}")]
    Netlist { path: PathBuf, message: String },
    #[error("Pack the netlist: {0}")]
    Pack(String),
    #[error("Clock frequency must be positive and finite, got {0} MHz")]
    Frequency(f64),
    #[error("No net drives a register clock pin, so timing-driven place and route has no clock. Check that the chipdb has cell timing, or turn timing-driven off.")]
    NoClock,
    #[error("Place: {0}")]
    Place(PlacerError),
    #[error("Refine the placement: {0}")]
    Refine(PlacerError),
    #[error("Route: {0}")]
    Route(RouterError),
    #[error("Timing analysis: {0}")]
    Timing(#[from] StaSessionError),
    #[error("Write the routed netlist: {0}")]
    RoutedJson(#[from] RoutedJsonError),
    #[error("Write FASM: {0}")]
    Fasm(#[from] FasmError),
    #[error("Write {path}: {error}")]
    Write {
        path: PathBuf,
        error: std::io::Error,
    },
    #[error("Timing constraints not met: {}", .0.join("; "))]
    TimingFailed(Vec<String>),
}

impl FlowError {
    fn placer(error: PlacerError, wrap: fn(PlacerError) -> Self) -> Self {
        match error {
            PlacerError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            error => wrap(error),
        }
    }
}

/// Run the flow; `log` receives one line per step and the timing report.
pub fn run(
    options: &FlowOptions,
    live: Option<Arc<Live>>,
    log: &mut dyn FnMut(&str),
) -> Result<FlowSummary, FlowError> {
    let cancelled = || match live.as_ref().is_some_and(|live| live.is_cancelled()) {
        true => Err(Cancelled),
        false => Ok(()),
    };
    let chipdb = ChipDb::load(&options.chipdb).map_err(|error| FlowError::Chipdb {
        path: options.chipdb.clone(),
        message: error.to_string(),
    })?;
    let mut ctx = Context::new(chipdb);
    if let Some(live) = &live {
        ctx.attach_live(Arc::clone(live));
    }
    let netlist_error = |message: String| FlowError::Netlist {
        path: options.netlist.clone(),
        message,
    };
    let text = std::fs::read_to_string(&options.netlist)
        .map_err(|error| netlist_error(error.to_string()))?;
    ctx.design = crate::frontend::parse_json(&text, &ctx.id_pool)
        .map_err(|error| netlist_error(format!("{error:#}")))?;
    cancelled()?;

    log("Packing");
    crate::packer::pack(&mut ctx, None).map_err(|error| FlowError::Pack(error.to_string()))?;
    ctx.publish_live(LiveProgress::bound("pack", None));
    let target_mhz = options.freq_mhz.unwrap_or(DEFAULT_TARGET_MHZ);
    let clocks = constrain_clocks(&mut ctx, target_mhz)?;
    if options.timing_driven && clocks.is_empty() {
        return Err(FlowError::NoClock);
    }
    log(&format!(
        "Constrained {} to {target_mhz} MHz",
        if clocks.is_empty() {
            "no net".to_owned()
        } else {
            clocks.join(", ")
        }
    ));
    cancelled()?;

    log("Placing with HeAP");
    place(&mut ctx, options)?;
    log("Refining the placement");
    refine(&mut ctx, options)?;
    log("Routing with Router2");
    let nets = collect_routable_nets(&ctx);
    route(&mut ctx, options, &nets)?;
    cancelled()?;

    let report = routed_timing(&ctx)?;
    let timing = report.to_string();
    log(&timing);
    write_outputs(&ctx, options, &report)?;
    let timing_failures = timing_failures(&report);
    for failure in &timing_failures {
        log(&format!("Timing check failed: {failure}"));
    }
    if !timing_failures.is_empty() && !options.allow_timing_failure {
        return Err(FlowError::TimingFailed(timing_failures));
    }
    Ok(FlowSummary {
        clocks,
        timing,
        timing_failures,
    })
}

/// Constrain every unconstrained net that drives a register clock pin; returns their names.
pub fn constrain_clocks(ctx: &mut Context, freq_mhz: f64) -> Result<Vec<String>, FlowError> {
    if !(freq_mhz.is_finite() && freq_mhz > 0.0) {
        return Err(FlowError::Frequency(freq_mhz));
    }
    let period_ps = (PS_PER_US / freq_mhz).round() as DelayT;
    let nets: Vec<NetId> = register_clock_nets(ctx)
        .map_err(StaSessionError::from)?
        .into_iter()
        .filter(|&net| ctx.design.net(net).clock_constraint == 0)
        .collect();
    for &net in &nets {
        ctx.design.net_edit(net).set_clock_constraint(period_ps);
    }
    Ok(nets
        .iter()
        .map(|&net| ctx.name_of(ctx.design.net(net).name).to_owned())
        .collect())
}

/// Place the design with HeAP, seeded from `options.seed`.
pub fn place(ctx: &mut Context, options: &FlowOptions) -> Result<(), FlowError> {
    let heap = PlacerHeapCfg {
        seed: options.seed,
        congestion_weight: HEAP_CONGESTION_WEIGHT,
        ..PlacerHeapCfg::default()
    };
    PlacerHeap
        .place(ctx, &heap)
        .map_err(|error| FlowError::placer(error, FlowError::Place))?;
    ctx.publish_live(LiveProgress::bound("placed", None));
    Ok(())
}

/// Refine the placement, by STA criticality when `options.timing_driven`, then validate it.
pub fn refine(ctx: &mut Context, options: &FlowOptions) -> Result<(), FlowError> {
    let refine = RefineCfg {
        timing_driven: options.timing_driven,
        crit_norm: CritNorm::WnsRelaxed,
        ..RefineCfg::default()
    };
    refine_placement(ctx, &refine)
        .and_then(|_| PlacerPipeline::validate(ctx))
        .map_err(|error| FlowError::placer(error, FlowError::Refine))?;
    ctx.publish_live(LiveProgress::bound("refined", None));
    Ok(())
}

/// Route `nets` with Router2, timing-driven when `options.timing_driven`.
pub fn route(ctx: &mut Context, options: &FlowOptions, nets: &[NetId]) -> Result<(), FlowError> {
    let router = Router2Cfg {
        timing: options.timing_driven.then(TimingDrivenCfg::default),
        ..Router2Cfg::default()
    };
    Router2
        .route_nets(ctx, &router, nets)
        .map_err(|error| match error {
            RouterError::Cancelled(cancelled) => FlowError::Cancelled(cancelled),
            error => FlowError::Route(error),
        })?;
    ctx.publish_live(LiveProgress::bound("routed", None));
    Ok(())
}

/// Write the report, routed JSON and FASM that `options` names, overwriting each file.
pub fn write_outputs(
    ctx: &Context,
    options: &FlowOptions,
    report: &TimingReport,
) -> Result<(), FlowError> {
    if let Some(path) = &options.report {
        write(path, pretty(&report_json(ctx, report)))?;
    }
    if let Some(path) = &options.routed {
        write(path, pretty(&routed_json(ctx)?))?;
    }
    if let Some(path) = &options.fasm {
        write(path, fasm_text(&fasm_features(ctx)?)?)?;
    }
    Ok(())
}

/// Time the routed design with the interconnect model the flow signs off with.
pub fn routed_timing(ctx: &Context) -> Result<TimingReport, StaSessionError> {
    let model = InterconnectModel::new(ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)?;
    StaSession::build(ctx, model, InterconnectStage::Routed)?.report(ctx, 1)
}

/// One message per failed sign-off check; none means timing is met.
pub fn timing_failures(report: &TimingReport) -> Vec<String> {
    let ns = |value: Option<f64>| value.map_or_else(|| "?".to_owned(), |ns| format!("{ns:.3}"));
    let mut failures = Vec::new();
    if report.endpoints() == 0 {
        failures.push("no setup endpoint is constrained; check the clock definitions".to_owned());
    }
    if report.failing_endpoints() > 0 {
        failures.push(format!(
            "setup WNS {} ns, {}/{} endpoints failing",
            ns(report.wns_ns()),
            report.failing_endpoints(),
            report.endpoints()
        ));
    }
    if report.hold_failing_endpoints() > 0 {
        failures.push(format!(
            "hold WNS {} ns, {}/{} endpoints failing",
            ns(report.hold_wns_ns()),
            report.hold_failing_endpoints(),
            report.hold_endpoints()
        ));
    }
    failures
}

fn pretty(json: &serde_json::Value) -> String {
    serde_json::to_string_pretty(json).expect("a JSON value serialises")
}

fn write(path: &Path, contents: String) -> Result<(), FlowError> {
    std::fs::write(path, contents).map_err(|error| FlowError::Write {
        path: path.to_owned(),
        error,
    })
}

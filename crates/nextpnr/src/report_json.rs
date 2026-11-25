//! The utilization and timing report in the JSON schema of nextpnr's `--report`, as its
//! `Context::writeJsonReport` writes it.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::common::ClockEdge;
use crate::context::Context;
use crate::netlist::CellPin;
use crate::sta::{PathPinReport, PathReport, TimingReport};

const MHZ_NS: f64 = 1000.0;

#[derive(Serialize)]
struct Report {
    utilization: BTreeMap<String, Utilization>,
    fmax: BTreeMap<String, Fmax>,
    critical_paths: Vec<CriticalPath>,
}

#[derive(Serialize)]
struct Utilization {
    used: usize,
    available: usize,
}

#[derive(Serialize)]
struct Fmax {
    achieved: f64,
    constraint: f64,
}

#[derive(Serialize)]
struct CriticalPath {
    from: String,
    to: String,
    path: Vec<Segment>,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
enum SegmentType {
    ClkToQ,
    Source,
    Logic,
    Routing,
    Setup,
}

#[derive(Serialize)]
struct Segment {
    #[serde(rename = "type")]
    segment_type: SegmentType,
    delay: f64,
    from: Endpoint,
    to: Endpoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    net: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sources: Option<Vec<String>>,
}

#[derive(Serialize)]
struct Endpoint {
    cell: String,
    port: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    loc: Option<[i32; 2]>,
}

/// Utilization per BEL type, Fmax per clock and the worst path of each clock domain pair.
pub fn report_json(ctx: &Context, timing: &TimingReport) -> serde_json::Value {
    let utilization = ctx
        .utilization_report()
        .rows
        .into_iter()
        .map(|row| {
            let usage = Utilization {
                used: row.used,
                available: row.available,
            };
            (row.resource, usage)
        })
        .collect();
    let fmax = timing
        .domains
        .iter()
        .filter_map(|domain| {
            let achieved = domain.fmax_mhz?;
            let constraint = MHZ_NS / domain.capture_period_ns;
            Some((
                domain.capture.clone(),
                Fmax {
                    achieved,
                    constraint,
                },
            ))
        })
        .collect();
    let critical_paths = timing
        .domains
        .iter()
        .flat_map(|domain| {
            domain.worst_paths.iter().map(|path| CriticalPath {
                from: clock_event(path.launch_edge, &domain.launch),
                to: clock_event(path.capture_edge, &domain.capture),
                path: segments(ctx, path),
            })
        })
        .collect();
    let report = Report {
        utilization,
        fmax,
        critical_paths,
    };
    serde_json::to_value(report).expect("string-keyed maps of numbers serialise")
}

fn clock_event(edge: ClockEdge, clock: &str) -> String {
    let edge = match edge {
        ClockEdge::Rising => "posedge",
        ClockEdge::Falling => "negedge",
    };
    format!("{edge} {clock}")
}

fn segments(ctx: &Context, path: &PathReport) -> Vec<Segment> {
    let first = path
        .pins
        .first()
        .expect("RustSTA reports a path's start pin");
    let last = path
        .pins
        .last()
        .expect("RustSTA reports a path's start pin");
    let start = match path.launch_clock_pin {
        Some(clock_pin) => segment(
            ctx,
            SegmentType::ClkToQ,
            first.delay_ns,
            clock_pin,
            first.cell_pin,
        ),
        None => segment(
            ctx,
            SegmentType::Source,
            first.delay_ns,
            first.cell_pin,
            first.cell_pin,
        ),
    };
    let arcs = path
        .pins
        .windows(2)
        .map(|arc| arc_segment(ctx, &arc[0], &arc[1]));
    let setup = segment(
        ctx,
        SegmentType::Setup,
        path.setup_ns,
        last.cell_pin,
        last.cell_pin,
    );
    std::iter::once(start)
        .chain(arcs)
        .chain(std::iter::once(setup))
        .collect()
}

fn arc_segment(ctx: &Context, from: &PathPinReport, to: &PathPinReport) -> Segment {
    let driven_net = ctx
        .design
        .cell(from.cell_pin.cell)
        .port_net(from.cell_pin.port)
        .map(|net| ctx.design.net(net))
        .filter(|net| net.driver() == Some(from.cell_pin));
    let Some(net) = driven_net else {
        return segment(
            ctx,
            SegmentType::Logic,
            to.delay_ns,
            from.cell_pin,
            to.cell_pin,
        );
    };
    let sources = net
        .attrs
        .get(&ctx.id("src"))
        .map(|src| src.as_str().split('|').map(str::to_owned).collect())
        .unwrap_or_default();
    Segment {
        net: Some(ctx.name_of(net.name).to_owned()),
        sources: Some(sources),
        ..segment(
            ctx,
            SegmentType::Routing,
            to.delay_ns,
            from.cell_pin,
            to.cell_pin,
        )
    }
}

fn segment(
    ctx: &Context,
    segment_type: SegmentType,
    delay: f64,
    from: CellPin,
    to: CellPin,
) -> Segment {
    Segment {
        segment_type,
        delay,
        from: endpoint(ctx, from),
        to: endpoint(ctx, to),
        net: None,
        sources: None,
    }
}

fn endpoint(ctx: &Context, pin: CellPin) -> Endpoint {
    let cell = ctx.design.cell(pin.cell);
    Endpoint {
        cell: ctx.name_of(cell.name).to_owned(),
        port: ctx.name_of(pin.port).to_owned(),
        loc: cell.bel.map(|bel| {
            let loc = ctx.chipdb().bel_loc(bel);
            [loc.x, loc.y]
        }),
    }
}

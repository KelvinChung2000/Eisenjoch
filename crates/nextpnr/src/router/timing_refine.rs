//! Post-route timing refinement for the combinational LUT6/IOB example fabric.

use crate::{
    chipdb::{PipId, WireId},
    common::DelayT,
    context::Context,
    netlist::{CellId, NetId, PortType},
    router::{
        astar::{astar_search, AStarOptions, PathCostModel, VisitBudget},
        common::{
            apply_route_plan, collect_sink_wires, resolve_source_wire, unroute_net,
            validate_all_routed, RoutePlan, SinkRoute,
        },
    },
};
use anyhow::{bail, ensure, Context as _, Result};
use std::collections::{HashMap, VecDeque};

fn topological_cells(ctx: &Context) -> Result<Vec<CellId>> {
    let cells: Vec<_> = ctx.design.iter_alive_cells().map(|(id, _)| id).collect();
    let mut incoming: HashMap<_, usize> = cells.iter().map(|&id| (id, 0)).collect();
    let mut dependents = HashMap::<_, Vec<_>>::new();
    for &id in &cells {
        let cell = ctx.cell(id);
        for pin in cell
            .ports()
            .filter(|p| cell.port_type(p.port) == Some(PortType::In))
        {
            if let Some(driver) = cell
                .port_net(pin.port)
                .and_then(|n| ctx.net(n).driver_cell_port())
            {
                *incoming.get_mut(&id).unwrap() += 1;
                dependents.entry(driver.cell).or_default().push(id);
            }
        }
    }
    let mut ready: VecDeque<_> = cells
        .iter()
        .copied()
        .filter(|id| incoming[id] == 0)
        .collect();
    let mut sorted = Vec::with_capacity(cells.len());
    while let Some(id) = ready.pop_front() {
        sorted.push(id);
        for sink in dependents.remove(&id).unwrap_or_default() {
            let remaining = incoming.get_mut(&sink).unwrap();
            *remaining -= 1;
            if *remaining == 0 {
                ready.push_back(sink);
            }
        }
    }
    ensure!(
        sorted.len() == cells.len(),
        "combinational loop in timing refinement"
    );
    Ok(sorted)
}

#[derive(Clone, Copy, Default)]
struct Arrival {
    logic: i64,
    route: i64,
}

impl Arrival {
    fn total(self) -> i64 {
        self.logic + self.route
    }
}

struct Arc {
    driver: CellId,
    sink: CellId,
    net: NetId,
    delay: i64,
}

fn analyse(ctx: &Context) -> Result<(Arrival, Vec<NetId>)> {
    let db = ctx.chipdb();
    let speed = ctx.speed_grade().context("missing chipdb speed grade")?;
    let mut arrivals = HashMap::<CellId, Option<Arrival>>::new();
    let mut arcs = Vec::new();
    let mut outputs = Vec::new();

    for id in topological_cells(ctx)? {
        let cell = ctx.cell(id);
        let lut = match cell.cell_type() {
            "LUT6" => true,
            "IOB" | "GND_DRV" | "VCC_DRV" => false,
            other => bail!("unsupported combinational timing cell {other}"),
        };
        let mut best = (cell.cell_type() == "IOB"
            && cell.ports().any(|pin| {
                cell.port_type(pin.port) == Some(PortType::Out) && cell.port_net(pin.port).is_some()
            }))
        .then(Arrival::default);
        for pin in cell.ports() {
            let port = pin.port;
            if cell.port_type(port) != Some(PortType::In) {
                continue;
            }
            let Some(net) = cell.port_net(port) else {
                continue;
            };
            let driver = ctx
                .net(net)
                .driver_cell_port()
                .context("input net without driver")?;
            // Constant drivers do not launch input-to-output timing paths.
            if matches!(ctx.cell(driver.cell).cell_type(), "GND_DRV" | "VCC_DRV") {
                continue;
            }
            let Some(mut at) = arrivals
                .get(&driver.cell)
                .copied()
                .context("combinational loop")?
            else {
                continue;
            };
            let base = at.total();
            let src = resolve_source_wire(ctx, net)?.context("missing driver wire")?;
            let mut wire = cell
                .bel()
                .unwrap()
                .pin_wire(port)
                .context("missing sink wire")?
                .id();
            let model = DelayModel(net);
            while wire != src {
                let pip = ctx.net(net).wires()[&wire]
                    .pip
                    .context("missing route parent")?;
                at.route += i64::from(model.pip_cost(ctx, pip));
                wire = db.pip_src_wire(pip);
            }
            if lut {
                let canon = |name: &str| {
                    db.constid_for_name(name)
                        .context("unknown timing pin or cell")
                };
                let index = db
                    .cell_timing_index(speed, canon("LUT6")?)
                    .context("missing LUT timing")?;
                let delay = db
                    .cell_delay(speed, index, canon(ctx.name_of(port))?, canon("F")?)
                    .context("missing LUT combinational arc")?;
                at.logic += i64::from(delay.max_delay());
            } else if cell.cell_type() == "IOB" && port == ctx.id("I") {
                outputs.push(id);
            }
            arcs.push(Arc {
                driver: driver.cell,
                sink: id,
                net,
                delay: at.total() - base,
            });
            if best.is_none_or(|b| at.total() > b.total()) {
                best = Some(at);
            }
        }
        arrivals.insert(id, best);
    }

    let worst = outputs
        .iter()
        .filter_map(|id| arrivals[id])
        .max_by_key(|a| a.total())
        .context("no timed output pads")?;
    let mut remaining: HashMap<_, _> = outputs.into_iter().map(|id| (id, 0)).collect();
    let mut scores = HashMap::new();
    for arc in arcs.into_iter().rev() {
        let Some(&tail) = remaining.get(&arc.sink) else {
            continue;
        };
        let tail = tail + arc.delay;
        let through = arrivals[&arc.driver].unwrap().total() + tail;
        remaining
            .entry(arc.driver)
            .and_modify(|v| *v = (*v).max(tail))
            .or_insert(tail);
        scores
            .entry(arc.net)
            .and_modify(|v: &mut i64| *v = (*v).max(through))
            .or_insert(through);
    }
    let mut nets: Vec<_> = scores.keys().copied().collect();
    nets.sort_by(|&a, &b| {
        scores[&b]
            .cmp(&scores[&a])
            .then_with(|| ctx.net(a).name().cmp(ctx.net(b).name()))
    });
    Ok((worst, nets))
}

struct DelayModel(NetId);

impl PathCostModel for DelayModel {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        ctx.pip(pip).delay().max_delay()
            + ctx.wire(ctx.chipdb().pip_dst_wire(pip)).delay().max_delay()
    }

    fn heuristic(&self, _: &Context, _: WireId, _: WireId) -> DelayT {
        0
    }

    fn is_blocked(&self, ctx: &Context, wire: WireId) -> bool {
        ctx.wire_binding(wire)
            .is_some_and(|(owner, _)| owner != self.0)
    }
}

fn shorten(ctx: &mut Context, net: NetId) -> bool {
    let source_wire = resolve_source_wire(ctx, net).unwrap().unwrap();
    let sinks = collect_sink_wires(ctx, net);
    let sink_count = sinks.len();
    let old = RoutePlan {
        net,
        source_wire,
        sink_routes: ctx
            .net(net)
            .wires()
            .values()
            .filter_map(|w| w.pip)
            .map(|pip| SinkRoute {
                sink_wire: ctx.chipdb().pip_dst_wire(pip),
                pips: vec![pip],
            })
            .collect(),
    };
    let model = DelayModel(net);
    let old_delays: Vec<_> = sinks
        .iter()
        .map(|&sink| {
            let mut wire = sink;
            let mut delay = 0;
            while wire != source_wire {
                let pip = ctx.net(net).wires()[&wire].pip.unwrap();
                delay += model.pip_cost(ctx, pip);
                wire = ctx.chipdb().pip_src_wire(pip);
            }
            delay
        })
        .collect();
    // Search the occupancy graph with only this net removed. Preserve its
    // existing tree if any sink cannot be reached or its delay increases.
    unroute_net(ctx, net);
    let sources = [source_wire].into_iter().collect();
    let options = AStarOptions {
        visit_limit: VisitBudget::Limit(20_000),
        retain_trace: false,
        ..AStarOptions::default()
    };
    let mut improved = false;
    let mut parents = HashMap::new();
    let mut candidate = Vec::new();
    for (sink, before) in sinks.into_iter().zip(old_delays) {
        let Some(pips) = astar_search(ctx, &model, &sources, sink, &options).path else {
            break;
        };
        let after: DelayT = pips.iter().map(|&p| model.pip_cost(ctx, p)).sum();
        if after > before
            || pips.iter().any(|&p| {
                parents
                    .insert(ctx.chipdb().pip_dst_wire(p), p)
                    .is_some_and(|old| old != p)
            })
        {
            break;
        }
        improved |= after < before;
        candidate.push(SinkRoute {
            sink_wire: sink,
            pips,
        });
    }
    if candidate.len() == sink_count && improved {
        let plan = RoutePlan {
            net,
            source_wire,
            sink_routes: candidate,
        };
        if apply_route_plan(ctx, &plan).is_ok() {
            return true;
        }
    }
    apply_route_plan(ctx, &old).unwrap();
    false
}

/// Modeled timing after a route-refinement pass. Pass zero describes the input.
#[derive(Debug, Clone, Copy)]
pub struct TimingPass {
    pub shortened_nets: usize,
    pub logic_ps: i64,
    pub route_ps: i64,
}

impl TimingPass {
    pub fn total_ps(self) -> i64 {
        self.logic_ps + self.route_ps
    }
}

/// Shorten routed net trees in descending path criticality without moving any BELs.
/// This model covers the example fabric's combinational LUT6/IOB data paths.
pub fn refine(ctx: &mut Context, passes: usize) -> Result<Vec<TimingPass>> {
    validate_all_routed(ctx)?;
    let (mut worst, mut nets) = analyse(ctx)?;
    let report = |shortened_nets, at: Arrival| TimingPass {
        shortened_nets,
        logic_ps: at.logic,
        route_ps: at.route,
    };
    let mut results = vec![report(0, worst)];
    for _ in 0..passes {
        let changed = nets.into_iter().filter(|&net| shorten(ctx, net)).count();
        validate_all_routed(ctx)?;
        let (revised, candidates) = analyse(ctx)?;
        ensure!(revised.total() <= worst.total(), "modeled timing worsened");
        (worst, nets) = (revised, candidates);
        results.push(report(changed, worst));
        if changed == 0 {
            break;
        }
    }
    Ok(results)
}

use std::collections::BTreeMap;
use std::ops::Range;

use rustc_hash::{FxHashMap, FxHashSet};
use ruststa::engine::delay::annotate::DelayAnnotations;
use ruststa::engine::incr::{
    ArcTiming, ArcTimingTable, Criticality, IncrementalStats, SessionError, SessionNodeKey,
    TimingSession,
};
use ruststa::engine::prop::EndpointCheckKind;
use ruststa::engine::types::{ClockId, NodeId};
use thiserror::Error;

use super::criticality::{CritNorm, PairBudget};
use super::interconnect::{DelayBounds, InterconnectError, InterconnectModel, InterconnectStage};
use super::{NetSink, RustSta, StaBuildError};
use crate::context::Context;
use crate::netlist::{CellId, NetId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Untimed {
    VacantUser,
    OffChip,
    Undriven,
    ConstantDriver,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SinkTiming {
    Untimed(Untimed),
    Arc(ArcTiming),
}

impl SinkTiming {
    pub fn setup_slack_ns(self) -> Option<f64> {
        match self {
            SinkTiming::Arc(ArcTiming::Propagated {
                setup: Some(setup), ..
            }) => Some(setup.slack_ns),
            _ => None,
        }
    }

    pub fn hold_slack_ns(self) -> Option<f64> {
        match self {
            SinkTiming::Arc(ArcTiming::Propagated { hold_slack_ns, .. }) => hold_slack_ns,
            _ => None,
        }
    }

    pub fn criticality(self) -> Option<Criticality> {
        match self {
            SinkTiming::Arc(ArcTiming::Propagated {
                setup: Some(setup), ..
            }) => Some(setup.criticality),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainSummary {
    pub launch: ClockId,
    pub capture: ClockId,
    pub wns_ns: f64,
    pub tns_ns: f64,
    pub endpoints: usize,
    pub failing_endpoints: usize,
}

#[derive(Debug, Error)]
pub enum StaSessionError {
    #[error(transparent)]
    Build(#[from] StaBuildError),
    #[error(transparent)]
    Interconnect(#[from] InterconnectError),
    #[error("RustSTA session: {0}")]
    Session(#[from] SessionError),
    #[error("the timing graph has {graph} net arcs, {mapped} design connections map to them")]
    UnmappedNetArcs { graph: usize, mapped: usize },
    #[error("net {0:?} is not a net of the timed design")]
    UnknownNet(NetId),
    #[error("{0:?} is not a sink of the timed design")]
    UnknownSink(NetSink),
    #[error("{sink:?} has no timing arc: {reason:?}")]
    UntimedSink { sink: NetSink, reason: Untimed },
    #[error("{0:?} is a clock pin of an ideal clock and carries no interconnect delay")]
    IdealClockSink(NetSink),
    #[error("{0:?} is a register clock pin that no defined clock reaches, so RustSTA leaves the register unconstrained; define a clock on its net with add_clock or an SDC create_clock")]
    UnclockedClockSink(NetSink),
    #[error("clock '{clock}' is propagated, but clock pins carry no interconnect delay, so its skew would time as zero")]
    PropagatedClock { clock: String },
    #[error("the check records of the corner were released")]
    CheckRecordsReleased,
}

#[derive(Debug, Clone, Copy)]
struct ArcLink {
    from: SessionNodeKey,
    to: SessionNodeKey,
    arc: usize,
    delay: DelayBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockPin {
    Ideal,
    Unclocked,
}

impl ClockPin {
    fn sink_error(self, sink: NetSink) -> StaSessionError {
        match self {
            ClockPin::Ideal => StaSessionError::IdealClockSink(sink),
            ClockPin::Unclocked => StaSessionError::UnclockedClockSink(sink),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SinkLink {
    Untimed(Untimed),
    ClockPin { arc: usize, pin: ClockPin },
    Arc(ArcLink),
}

#[derive(Debug)]
struct SinkIndex {
    nets: Vec<Option<NetId>>,
    first: Vec<usize>,
}

impl SinkIndex {
    fn new(ctx: &Context) -> Self {
        let slots = ctx
            .design
            .iter_alive_nets()
            .map(|(net, _)| net.slot() as usize + 1)
            .max()
            .unwrap_or(0);
        let mut nets = vec![None; slots];
        let mut users = vec![0; slots];
        for (net, info) in ctx.design.iter_alive_nets() {
            nets[net.slot() as usize] = Some(net);
            users[net.slot() as usize] = info.users().len();
        }
        let first = std::iter::once(0)
            .chain(users.iter().scan(0, |total, &count| {
                *total += count;
                Some(*total)
            }))
            .collect();
        Self { nets, first }
    }

    fn len(&self) -> usize {
        *self.first.last().expect("the index has a leading offset")
    }

    fn range(&self, net: NetId) -> Result<Range<usize>, StaSessionError> {
        let slot = net.slot() as usize;
        match self.nets.get(slot) {
            Some(&Some(known)) if known == net => Ok(self.first[slot]..self.first[slot + 1]),
            _ => Err(StaSessionError::UnknownNet(net)),
        }
    }

    fn entry(&self, sink: NetSink) -> Result<usize, StaSessionError> {
        let range = self.range(sink.net)?;
        let entry = range.start + sink.user as usize;
        match range.contains(&entry) {
            true => Ok(entry),
            false => Err(StaSessionError::UnknownSink(sink)),
        }
    }
}

#[derive(Debug)]
pub struct StaSession {
    sta: RustSta,
    model: InterconnectModel,
    corner: String,
    index: SinkIndex,
    links: Vec<SinkLink>,
    timing: Vec<SinkTiming>,
    domains: Vec<DomainSummary>,
    hold_domains: Vec<DomainSummary>,
    table: ArcTimingTable,
    budgets: Vec<PairBudget>,
}

impl StaSession {
    pub fn build(
        ctx: &Context,
        model: InterconnectModel,
        stage: InterconnectStage,
    ) -> Result<Self, StaSessionError> {
        Self::assemble(ctx, model, Some(stage))
    }

    pub fn build_ideal(ctx: &Context, model: InterconnectModel) -> Result<Self, StaSessionError> {
        Self::assemble(ctx, model, None)
    }

    fn assemble(
        ctx: &Context,
        model: InterconnectModel,
        stage: Option<InterconnectStage>,
    ) -> Result<Self, StaSessionError> {
        let mut sta = RustSta::build(ctx)?;
        let index = SinkIndex::new(ctx);
        let mut links = sink_links(ctx, &sta, &index)?;
        let graph_net_arcs = sta
            .session
            .graph()
            .arcs()
            .iter()
            .filter(|arc| arc.kind.is_net())
            .count();
        let mapped = links
            .iter()
            .filter(|link| matches!(link, SinkLink::Arc(_) | SinkLink::ClockPin { .. }))
            .count();
        if graph_net_arcs != mapped {
            return Err(StaSessionError::UnmappedNetArcs {
                graph: graph_net_arcs,
                mapped,
            });
        }
        let annotated_nets = stage
            .into_iter()
            .flat_map(|stage| index.nets.iter().flatten().map(move |&net| (net, stage)));
        for (net, stage) in annotated_nets {
            let range = index.range(net)?;
            let users = timed_users(&links[range.clone()]);
            if users.is_empty() {
                continue;
            }
            for (user, delay) in model.net_delays(ctx, stage, net, &users)? {
                let SinkLink::Arc(link) = &mut links[range.start + user as usize] else {
                    unreachable!("every placed user of a timed net has a timing arc");
                };
                link.delay = delay;
            }
        }
        let mut annotations = DelayAnnotations::new();
        for link in &links {
            if let SinkLink::Arc(link) = link {
                annotations.insert(link.from, link.to, link.delay.arc_delays());
            }
        }
        sta.set_net_delays(annotations)?;
        let corner = sta.session.default_corner().corner_name.clone();
        let table = sta.session.arc_timing(&corner)?;
        let mut session = Self {
            sta,
            model,
            corner,
            index,
            timing: vec![SinkTiming::Untimed(Untimed::VacantUser); links.len()],
            links,
            domains: Vec::new(),
            hold_domains: Vec::new(),
            table,
            budgets: Vec::new(),
        };
        session.sync_table()?;
        Ok(session)
    }

    pub fn update_cells(
        &mut self,
        ctx: &Context,
        stage: InterconnectStage,
        cells: &[CellId],
    ) -> Result<IncrementalStats, StaSessionError> {
        let mut seen = FxHashSet::default();
        let nets: Vec<NetId> = cells
            .iter()
            .flat_map(|&cell| ctx.design.cell(cell).ports().values())
            .filter_map(|port| port.net())
            .filter(|&net| seen.insert(net))
            .collect();
        self.update_nets(ctx, stage, &nets)
    }

    pub fn update_nets(
        &mut self,
        ctx: &Context,
        stage: InterconnectStage,
        nets: &[NetId],
    ) -> Result<IncrementalStats, StaSessionError> {
        let mut delays = Vec::new();
        for &net in nets {
            let users = timed_users(&self.links[self.index.range(net)?]);
            if users.is_empty() {
                continue;
            }
            for (user, delay) in self.model.net_delays(ctx, stage, net, &users)? {
                delays.push((NetSink { net, user }, delay));
            }
        }
        self.set_connection_delays(&delays)
    }

    pub fn set_connection_delays(
        &mut self,
        delays: &[(NetSink, DelayBounds)],
    ) -> Result<IncrementalStats, StaSessionError> {
        let mut annotations = DelayAnnotations::new();
        let mut changed = Vec::new();
        for &(sink, delay) in delays {
            let entry = self.index.entry(sink)?;
            let link = match self.links[entry] {
                SinkLink::Arc(link) => link,
                SinkLink::ClockPin { pin, .. } => return Err(pin.sink_error(sink)),
                SinkLink::Untimed(reason) => {
                    return Err(StaSessionError::UntimedSink { sink, reason })
                }
            };
            if link.delay != delay {
                annotations.insert(link.from, link.to, delay.arc_delays());
                changed.push((entry, delay));
            }
        }
        if changed.is_empty() {
            return Ok(IncrementalStats::default());
        }
        let batch = BTreeMap::from([(self.corner.clone(), annotations)]);
        let stats = self.sta.session.merge_delay_annotations(&batch)?;
        for (entry, delay) in changed {
            let SinkLink::Arc(link) = &mut self.links[entry] else {
                unreachable!("only timed sinks were changed");
            };
            link.delay = delay;
        }
        self.refresh()?;
        Ok(stats)
    }

    pub fn sink_timing(&self, sink: NetSink) -> Result<SinkTiming, StaSessionError> {
        Ok(self.timing[self.index.entry(sink)?])
    }

    pub fn criticality(&self, sink: NetSink) -> Result<Option<Criticality>, StaSessionError> {
        Ok(self.sink_timing(sink)?.criticality())
    }

    pub fn normalised_criticality(
        &self,
        sink: NetSink,
        norm: CritNorm,
    ) -> Result<Option<f64>, StaSessionError> {
        let arc = match self.links[self.index.entry(sink)?] {
            SinkLink::Untimed(_) => return Ok(None),
            SinkLink::ClockPin { arc, .. } | SinkLink::Arc(ArcLink { arc, .. }) => arc,
        };
        Ok(self
            .budgets
            .iter()
            .enumerate()
            .filter_map(|(pair, &budget)| {
                self.table
                    .pair_slack(arc, pair)
                    .map(|slack_ns| norm.pair_criticality(slack_ns, budget))
            })
            .reduce(f64::max))
    }

    pub fn pair_budgets(&self) -> &[PairBudget] {
        &self.budgets
    }

    pub fn net_sinks(&self, net: NetId) -> Result<&[SinkTiming], StaSessionError> {
        Ok(&self.timing[self.index.range(net)?])
    }

    pub fn connection_delay(&self, sink: NetSink) -> Result<DelayBounds, StaSessionError> {
        match self.links[self.index.entry(sink)?] {
            SinkLink::Arc(link) => Ok(link.delay),
            SinkLink::ClockPin { pin, .. } => Err(pin.sink_error(sink)),
            SinkLink::Untimed(reason) => Err(StaSessionError::UntimedSink { sink, reason }),
        }
    }

    pub fn domains(&self) -> &[DomainSummary] {
        &self.domains
    }

    pub fn hold_domains(&self) -> &[DomainSummary] {
        &self.hold_domains
    }

    pub fn sta(&self) -> &RustSta {
        &self.sta
    }

    fn refresh(&mut self) -> Result<(), StaSessionError> {
        self.table = self.sta.session.arc_timing(&self.corner)?;
        self.sync_table()
    }

    fn sync_table(&mut self) -> Result<(), StaSessionError> {
        let rows = self.table.rows();
        for (timing, link) in self.timing.iter_mut().zip(&self.links) {
            *timing = match link {
                SinkLink::Untimed(reason) => SinkTiming::Untimed(*reason),
                SinkLink::ClockPin { arc, .. } => SinkTiming::Arc(rows[*arc].timing),
                SinkLink::Arc(link) => SinkTiming::Arc(rows[link.arc].timing),
            };
        }
        self.domains = domain_summaries(&self.sta.session, EndpointCheckKind::is_setup_family)?;
        self.hold_domains = domain_summaries(&self.sta.session, EndpointCheckKind::is_hold_family)?;
        self.budgets = pair_budgets(&self.table);
        Ok(())
    }
}

fn pair_budgets(table: &ArcTimingTable) -> Vec<PairBudget> {
    table
        .pairs()
        .iter()
        .enumerate()
        .map(|(pair, domain)| PairBudget {
            launch: domain.launch,
            capture: domain.capture,
            d_req_ns: domain.d_req_ns,
            wns_ns: (0..table.rows().len())
                .filter_map(|arc| table.pair_slack(arc, pair))
                .fold(f64::INFINITY, f64::min),
        })
        .collect()
}

fn timed_users(links: &[SinkLink]) -> Vec<u32> {
    links
        .iter()
        .enumerate()
        .filter(|(_, link)| matches!(link, SinkLink::Arc(_)))
        .map(|(user, _)| u32::try_from(user).expect("net user index fits in u32"))
        .collect()
}

fn clock_pins(session: &TimingSession) -> Result<FxHashMap<NodeId, ClockPin>, StaSessionError> {
    let graph = session.graph();
    let clock_model = &session.default_corner().clock_model;
    let pins: Vec<NodeId> = graph
        .launches()
        .iter()
        .map(|launch| launch.clock_pin)
        .chain(graph.checks().iter().map(|check| check.clock_pin))
        .collect();
    let clocks = &session.state().constraints.clocks;
    if let Some(clock) = pins
        .iter()
        .flat_map(|&pin| clock_model.node_clocks(pin))
        .map(|&(clock, _)| &clocks[clock.index()])
        .find(|clock| clock.propagated)
    {
        return Err(StaSessionError::PropagatedClock {
            clock: clock.name.clone(),
        });
    }
    Ok(pins
        .into_iter()
        .map(|pin| match clock_model.node_clocks(pin).is_empty() {
            true => (pin, ClockPin::Unclocked),
            false => (pin, ClockPin::Ideal),
        })
        .collect())
}

fn sink_links(
    ctx: &Context,
    sta: &RustSta,
    index: &SinkIndex,
) -> Result<Vec<SinkLink>, StaSessionError> {
    let ids = sta.ids();
    let clock_pins = clock_pins(&sta.session)?;
    let mut links = Vec::with_capacity(index.len());
    for net in index.nets.iter().flatten().copied() {
        let info = ctx.design.net(net);
        let off_chip = ids.off_chip_nets.contains(&net);
        for (user, &sink) in info.users().iter().enumerate() {
            let user = u32::try_from(user).expect("net user index fits in u32");
            let link = match (ids.net_arcs.get(&NetSink { net, user }), info.driver()) {
                (Some(&arc), Some(driver)) => match clock_pins.get(&ids.pins[&sink].node) {
                    Some(&pin) => SinkLink::ClockPin { arc, pin },
                    None => SinkLink::Arc(ArcLink {
                        from: ids.pins[&driver].key,
                        to: ids.pins[&sink].key,
                        arc,
                        delay: DelayBounds::ZERO,
                    }),
                },
                _ if !sink.is_valid() => SinkLink::Untimed(Untimed::VacantUser),
                _ if off_chip => SinkLink::Untimed(Untimed::OffChip),
                (_, None) => SinkLink::Untimed(Untimed::Undriven),
                (None, Some(_)) => SinkLink::Untimed(Untimed::ConstantDriver),
            };
            links.push(link);
        }
    }
    Ok(links)
}

fn domain_summaries(
    session: &TimingSession,
    in_family: fn(EndpointCheckKind) -> bool,
) -> Result<Vec<DomainSummary>, StaSessionError> {
    let endpoints = &session.default_corner().endpoints;
    if endpoints.records_released() {
        return Err(StaSessionError::CheckRecordsReleased);
    }
    let mut worst: BTreeMap<(ClockId, ClockId), BTreeMap<NodeId, f64>> = BTreeMap::new();
    for record in endpoints
        .records
        .iter()
        .filter(|record| in_family(record.kind))
    {
        worst
            .entry((record.launch_clock, record.capture_clock))
            .or_default()
            .entry(record.endpoint)
            .and_modify(|slack| *slack = slack.min(record.slack))
            .or_insert(record.slack);
    }
    Ok(worst
        .into_iter()
        .map(|((launch, capture), slacks)| DomainSummary {
            launch,
            capture,
            wns_ns: slacks.values().copied().fold(f64::INFINITY, f64::min),
            tns_ns: slacks.values().map(|&slack| slack.min(0.0)).sum(),
            endpoints: slacks.len(),
            failing_endpoints: slacks.values().filter(|&&slack| slack < 0.0).count(),
        })
        .collect())
}

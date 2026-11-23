use std::collections::BTreeMap;
use std::ops::Range;

use rustc_hash::FxHashSet;
use ruststa::engine::delay::annotate::DelayAnnotations;
use ruststa::engine::incr::{
    ArcTiming, Criticality, IncrementalStats, SessionError, SessionNodeKey, TimingSession,
};
use ruststa::engine::types::{ClockId, NodeId};
use thiserror::Error;

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

#[derive(Debug, Clone, Copy)]
enum SinkLink {
    Untimed(Untimed),
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
        let mut links = sink_links(ctx, &sta, &index);
        let graph_net_arcs = sta
            .session
            .graph()
            .arcs()
            .iter()
            .filter(|arc| arc.kind.is_net())
            .count();
        let mapped = links
            .iter()
            .filter(|link| matches!(link, SinkLink::Arc(_)))
            .count();
        if graph_net_arcs != mapped {
            return Err(StaSessionError::UnmappedNetArcs {
                graph: graph_net_arcs,
                mapped,
            });
        }
        let annotated_nets = stage.into_iter().flat_map(|stage| {
            index.nets.iter().flatten().map(move |&net| (net, stage))
        });
        for (net, stage) in annotated_nets {
            let range = index.range(net)?;
            if !links[range.clone()]
                .iter()
                .any(|link| matches!(link, SinkLink::Arc(_)))
            {
                continue;
            }
            for (user, delay) in model.net_delays(ctx, stage, net)? {
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
        let corner = sta.session.default_corner().corner_name.clone();
        sta.session.set_delay_annotations(&corner, annotations)?;
        let mut session = Self {
            sta,
            model,
            corner,
            index,
            timing: vec![SinkTiming::Untimed(Untimed::VacantUser); links.len()],
            links,
            domains: Vec::new(),
        };
        session.refresh()?;
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
            .flat_map(|&cell| ctx.design.cell(cell).ports.values())
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
            let range = self.index.range(net)?;
            if !self.links[range]
                .iter()
                .any(|link| matches!(link, SinkLink::Arc(_)))
            {
                continue;
            }
            for (user, delay) in self.model.net_delays(ctx, stage, net)? {
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

    pub fn net_sinks(&self, net: NetId) -> Result<&[SinkTiming], StaSessionError> {
        Ok(&self.timing[self.index.range(net)?])
    }

    pub fn connection_delay(&self, sink: NetSink) -> Result<DelayBounds, StaSessionError> {
        match self.links[self.index.entry(sink)?] {
            SinkLink::Arc(link) => Ok(link.delay),
            SinkLink::Untimed(reason) => Err(StaSessionError::UntimedSink { sink, reason }),
        }
    }

    pub fn domains(&self) -> &[DomainSummary] {
        &self.domains
    }

    pub fn sta(&self) -> &RustSta {
        &self.sta
    }

    fn refresh(&mut self) -> Result<(), StaSessionError> {
        let table = self.sta.session.arc_timing(&self.corner)?;
        let rows = table.rows();
        for (timing, link) in self.timing.iter_mut().zip(&self.links) {
            *timing = match link {
                SinkLink::Untimed(reason) => SinkTiming::Untimed(*reason),
                SinkLink::Arc(link) => SinkTiming::Arc(rows[link.arc].timing),
            };
        }
        self.domains = domain_summaries(&self.sta.session)?;
        Ok(())
    }
}

fn sink_links(ctx: &Context, sta: &RustSta, index: &SinkIndex) -> Vec<SinkLink> {
    let ids = sta.ids();
    let mut links = Vec::with_capacity(index.len());
    for net in index.nets.iter().flatten().copied() {
        let info = ctx.design.net(net);
        let off_chip = ids.off_chip_nets.contains(&net);
        for (user, &sink) in info.users().iter().enumerate() {
            let user = u32::try_from(user).expect("net user index fits in u32");
            let link = match (ids.net_arcs.get(&NetSink { net, user }), info.driver()) {
                (Some(&arc), Some(driver)) => SinkLink::Arc(ArcLink {
                    from: ids.pins[&driver].key,
                    to: ids.pins[&sink].key,
                    arc,
                    delay: DelayBounds::ZERO,
                }),
                _ if !sink.is_valid() => SinkLink::Untimed(Untimed::VacantUser),
                _ if off_chip => SinkLink::Untimed(Untimed::OffChip),
                (_, None) => SinkLink::Untimed(Untimed::Undriven),
                (None, Some(_)) => SinkLink::Untimed(Untimed::ConstantDriver),
            };
            links.push(link);
        }
    }
    links
}

fn domain_summaries(session: &TimingSession) -> Result<Vec<DomainSummary>, StaSessionError> {
    let endpoints = &session.default_corner().endpoints;
    if endpoints.records_released() {
        return Err(StaSessionError::CheckRecordsReleased);
    }
    let mut worst: BTreeMap<(ClockId, ClockId), BTreeMap<NodeId, f64>> = BTreeMap::new();
    for record in endpoints
        .records
        .iter()
        .filter(|record| record.kind.is_setup_family())
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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Add;

use rustc_hash::{FxHashMap, FxHashSet};
use ruststa::engine::delay::annotate::ArcDelays;
use thiserror::Error;

use super::delay_table::{DelayTable, Reach, TableEntry, TableKey, TileOffset};
use super::{is_clock_net, PS_IN_NS};
use crate::chipdb::{ChipDb, PipId, TimingValue, WireId};
use crate::context::{BelPin, Context};
use crate::netlist::{CellPin, NetId, NetInfo, PipMap};
use crate::common::DelayT;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DelayBounds {
    pub early_ps: DelayT,
    pub late_ps: DelayT,
}

impl DelayBounds {
    pub const ZERO: Self = Self {
        early_ps: 0,
        late_ps: 0,
    };

    fn of(value: TimingValue) -> Self {
        Self {
            early_ps: value.fast_min,
            late_ps: value.slow_max,
        }
    }

    pub fn arc_delays(self) -> ArcDelays {
        ArcDelays::uniform(
            f64::from(self.early_ps) * PS_IN_NS,
            f64::from(self.late_ps) * PS_IN_NS,
        )
        .expect("a delay in whole ps is finite")
    }
}

impl Add for DelayBounds {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            early_ps: self.early_ps + other.early_ps,
            late_ps: self.late_ps + other.late_ps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClasslessPips {
    Reject,
    Notional(DelayBounds),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterconnectStage {
    Placed,
    Routed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Untabulated {
    NoSample,
    VariesWithSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacedEstimate {
    DelayTable,
    ExactSearch,
}

#[derive(Debug, Error)]
pub enum InterconnectError {
    #[error("chipdb has no speed grade {index}")]
    MissingSpeedGrade { index: usize },
    #[error(
        "speed grade has {count} node timing classes; the interconnect model has no wire RC term"
    )]
    NodeClassesUnsupported { count: usize },
    #[error("pip {pip} names timing class {class}, the speed grade has {classes}")]
    PipClassOutOfRange {
        pip: PipId,
        class: i32,
        classes: usize,
    },
    #[error("pip {pip} has no timing class and classless pips are rejected")]
    ClasslessPip { pip: PipId },
    #[error("net '{net}' has no driver")]
    Undriven { net: String },
    #[error("cell '{cell}' is not placed")]
    UnplacedCell { cell: String },
    #[error("BEL of cell '{cell}' has no wire for pin '{port}'")]
    NoPinWire { cell: String, port: String },
    #[error("net '{net}': no timed path reaches pin '{port}' of cell '{cell}'")]
    Unreachable {
        net: String,
        cell: String,
        port: String,
    },
    #[error(
        "net '{net}': the delay table holds no delay for pin '{port}' of cell '{cell}' at offset {offset:?}: {reason:?}"
    )]
    Untabulated {
        net: String,
        cell: String,
        port: String,
        offset: TileOffset,
        reason: Untabulated,
    },
    #[error("net '{net}' is a clock net; clocks are ideal and the delay table does not hold them")]
    ClockNotTabulated { net: String },
    #[error("net '{net}': pin '{port}' of cell '{cell}' is not routed")]
    Unrouted {
        net: String,
        cell: String,
        port: String,
    },
    #[error(
        "net '{net}': the route to pin '{port}' of cell '{cell}' does not lead back to the driver"
    )]
    BrokenRoute {
        net: String,
        cell: String,
        port: String,
    },
}

#[derive(Debug, Clone)]
pub(super) struct PipDelays {
    classes: Vec<DelayBounds>,
    classless: ClasslessPips,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SearchGoal {
    AllWires,
    Sinks(FxHashSet<WireId>),
}

impl PipDelays {
    fn delay(&self, chipdb: &ChipDb, pip: PipId) -> Option<DelayBounds> {
        match usize::try_from(chipdb.pip_timing_index(pip)) {
            Ok(class) => Some(self.classes[class]),
            Err(_) => match self.classless {
                ClasslessPips::Notional(delay) => Some(delay),
                ClasslessPips::Reject => None,
            },
        }
    }

    pub(super) fn shortest_delays(
        &self,
        chipdb: &ChipDb,
        source: WireId,
        mut goal: SearchGoal,
    ) -> FxHashMap<WireId, DelayBounds> {
        let mut settled: FxHashMap<WireId, DelayBounds> = FxHashMap::default();
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(SearchEntry::new(DelayBounds::ZERO, source)));
        while let Some(Reverse(entry)) = frontier.pop() {
            let (wire, delay) = (entry.wire(), entry.delay());
            if settled.contains_key(&wire) {
                continue;
            }
            settled.insert(wire, delay);
            if let SearchGoal::Sinks(pending) = &mut goal {
                pending.remove(&wire);
                if pending.is_empty() {
                    break;
                }
            }
            chipdb.node_wires_cb(wire, |peer| {
                frontier.push(Reverse(SearchEntry::new(delay, peer)));
            });
            for &pip_index in chipdb.wire_info(wire).pips_downhill.get() {
                let pip = PipId::new(wire.tile(), pip_index);
                let Some(pip_delay) = self.delay(chipdb, pip) else {
                    continue;
                };
                frontier.push(Reverse(SearchEntry::new(
                    delay + pip_delay,
                    chipdb.pip_dst_wire(pip),
                )));
            }
        }
        settled
    }
}

#[derive(Debug, Clone)]
enum PlacedEstimator {
    Table(DelayTable),
    Search,
}

#[derive(Debug, Clone)]
pub struct InterconnectModel {
    pips: PipDelays,
    placed: PlacedEstimator,
}

impl InterconnectModel {
    pub fn new(
        ctx: &Context,
        classless_pips: ClasslessPips,
        placed: PlacedEstimate,
    ) -> Result<Self, InterconnectError> {
        let speed_grade =
            ctx.speed_grade()
                .ok_or_else(|| InterconnectError::MissingSpeedGrade {
                    index: ctx.speed_grade_idx(),
                })?;
        let node_classes = speed_grade.node_classes.get().len();
        if node_classes > 0 {
            return Err(InterconnectError::NodeClassesUnsupported {
                count: node_classes,
            });
        }
        let classes: Vec<DelayBounds> = speed_grade
            .pip_classes
            .get()
            .iter()
            .map(|class| DelayBounds::of(class.int_delay))
            .collect();
        let chipdb = ctx.chipdb();
        if let Some(pip) = chipdb
            .pips()
            .find(|&pip| chipdb.pip_timing_index(pip) >= classes.len() as i32)
        {
            return Err(InterconnectError::PipClassOutOfRange {
                pip,
                class: chipdb.pip_timing_index(pip),
                classes: classes.len(),
            });
        }
        let pips = PipDelays {
            classes,
            classless: classless_pips,
        };
        let placed = match placed {
            PlacedEstimate::DelayTable => PlacedEstimator::Table(DelayTable::build(chipdb, &pips)),
            PlacedEstimate::ExactSearch => PlacedEstimator::Search,
        };
        Ok(Self { pips, placed })
    }

    pub fn pip_delay(&self, chipdb: &ChipDb, pip: PipId) -> Option<DelayBounds> {
        self.pips.delay(chipdb, pip)
    }

    pub fn delay_table(&self) -> Option<&DelayTable> {
        match &self.placed {
            PlacedEstimator::Table(table) => Some(table),
            PlacedEstimator::Search => None,
        }
    }

    pub fn exact_delays_from(
        &self,
        chipdb: &ChipDb,
        source: WireId,
    ) -> FxHashMap<WireId, DelayBounds> {
        self.pips
            .shortest_delays(chipdb, source, SearchGoal::AllWires)
    }

    pub fn net_delays(
        &self,
        ctx: &Context,
        stage: InterconnectStage,
        net: NetId,
    ) -> Result<Vec<(u32, DelayBounds)>, InterconnectError> {
        let info = ctx.design.net(net);
        let driver = info.driver().ok_or_else(|| InterconnectError::Undriven {
            net: ctx.name_of(info.name).to_owned(),
        })?;
        let source = pin_wire(ctx, driver)?;
        let sinks = info
            .users()
            .iter()
            .enumerate()
            .filter(|(_, sink)| sink.is_valid())
            .map(|(user, &sink)| {
                let user = u32::try_from(user).expect("net user index fits in u32");
                Ok((user, sink, pin_wire(ctx, sink)?))
            })
            .collect::<Result<Vec<_>, InterconnectError>>()?;
        match (stage, &self.placed) {
            (InterconnectStage::Placed, PlacedEstimator::Search) => {
                self.estimate(ctx, info, source, &sinks)
            }
            (InterconnectStage::Placed, PlacedEstimator::Table(table)) => {
                look_up(table, ctx, info, source, &sinks)
            }
            (InterconnectStage::Routed, _) => sinks
                .iter()
                .map(|&(user, sink, wire)| Ok((user, self.routed(ctx, info, source, sink, wire)?)))
                .collect(),
        }
    }

    fn estimate(
        &self,
        ctx: &Context,
        net: &NetInfo,
        source: WireId,
        sinks: &[(u32, CellPin, WireId)],
    ) -> Result<Vec<(u32, DelayBounds)>, InterconnectError> {
        let pending = sinks.iter().map(|&(_, _, wire)| wire).collect();
        let settled = self
            .pips
            .shortest_delays(ctx.chipdb(), source, SearchGoal::Sinks(pending));
        sinks
            .iter()
            .map(|&(user, sink, wire)| match settled.get(&wire) {
                Some(&delay) => Ok((user, delay)),
                None => Err(unreachable(ctx, net, sink)),
            })
            .collect()
    }

    fn routed(
        &self,
        ctx: &Context,
        net: &NetInfo,
        source: WireId,
        sink: CellPin,
        sink_wire: WireId,
    ) -> Result<DelayBounds, InterconnectError> {
        let chipdb = ctx.chipdb();
        let source_node = chipdb.node_id(source);
        let at_source = |wire: WireId| {
            wire == source || (source_node.is_some() && chipdb.node_id(wire) == source_node)
        };
        let error = |unrouted: bool| {
            let (net, cell, port) = (
                ctx.name_of(net.name).to_owned(),
                cell_name(ctx, sink),
                ctx.name_of(sink.port).to_owned(),
            );
            match unrouted {
                true => InterconnectError::Unrouted { net, cell, port },
                false => InterconnectError::BrokenRoute { net, cell, port },
            }
        };
        let mut delay = DelayBounds::ZERO;
        let mut wire = sink_wire;
        for hop in 0..=net.wires.len() {
            if at_source(wire) {
                return Ok(delay);
            }
            let pip = tree_entry(chipdb, net, wire)
                .ok_or_else(|| error(hop == 0))?
                .pip
                .ok_or_else(|| error(false))?;
            delay = delay
                + self
                    .pips
                    .delay(chipdb, pip)
                    .ok_or(InterconnectError::ClasslessPip { pip })?;
            wire = chipdb.pip_src_wire(pip);
        }
        Err(error(false))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SearchEntry {
    late_ps: DelayT,
    early_ps: DelayT,
    tile: i32,
    index: i32,
}

impl SearchEntry {
    fn new(delay: DelayBounds, wire: WireId) -> Self {
        Self {
            late_ps: delay.late_ps,
            early_ps: delay.early_ps,
            tile: wire.tile(),
            index: wire.index(),
        }
    }

    fn wire(self) -> WireId {
        WireId::new(self.tile, self.index)
    }

    fn delay(self) -> DelayBounds {
        DelayBounds {
            early_ps: self.early_ps,
            late_ps: self.late_ps,
        }
    }
}

fn look_up(
    table: &DelayTable,
    ctx: &Context,
    net: &NetInfo,
    source: WireId,
    sinks: &[(u32, CellPin, WireId)],
) -> Result<Vec<(u32, DelayBounds)>, InterconnectError> {
    if is_clock_net(net) {
        return Err(InterconnectError::ClockNotTabulated {
            net: ctx.name_of(net.name).to_owned(),
        });
    }
    let chipdb = ctx.chipdb();
    sinks
        .iter()
        .map(|&(user, sink, wire)| {
            let key = TableKey::between(chipdb, source, wire);
            let untabulated = |reason| InterconnectError::Untabulated {
                net: ctx.name_of(net.name).to_owned(),
                cell: cell_name(ctx, sink),
                port: ctx.name_of(sink.port).to_owned(),
                offset: key.offset,
                reason,
            };
            match table.get(&key) {
                Some(TableEntry::Sampled {
                    reach: Reach::Delay(delay),
                    ..
                }) => Ok((user, delay)),
                Some(TableEntry::Sampled {
                    reach: Reach::Unreachable,
                    ..
                }) => Err(unreachable(ctx, net, sink)),
                Some(TableEntry::VariesWithSource) => {
                    Err(untabulated(Untabulated::VariesWithSource))
                }
                None => Err(untabulated(Untabulated::NoSample)),
            }
        })
        .collect()
}

fn unreachable(ctx: &Context, net: &NetInfo, sink: CellPin) -> InterconnectError {
    InterconnectError::Unreachable {
        net: ctx.name_of(net.name).to_owned(),
        cell: cell_name(ctx, sink),
        port: ctx.name_of(sink.port).to_owned(),
    }
}

fn tree_entry<'a>(chipdb: &ChipDb, net: &'a NetInfo, wire: WireId) -> Option<&'a PipMap> {
    if let Some(entry) = net.wires.get(&wire) {
        return Some(entry);
    }
    let mut found = None;
    chipdb.node_wires_cb(wire, |peer| {
        if found.is_none() {
            found = net.wires.get(&peer);
        }
    });
    found
}

fn pin_wire(ctx: &Context, pin: CellPin) -> Result<WireId, InterconnectError> {
    let bel = ctx
        .design
        .cell(pin.cell)
        .bel
        .ok_or_else(|| InterconnectError::UnplacedCell {
            cell: cell_name(ctx, pin),
        })?;
    ctx.bel_pin_wire(BelPin::new(bel, pin.port))
        .map(|wire| wire.id())
        .ok_or_else(|| InterconnectError::NoPinWire {
            cell: cell_name(ctx, pin),
            port: ctx.name_of(pin.port).to_owned(),
        })
}

fn cell_name(ctx: &Context, pin: CellPin) -> String {
    ctx.name_of(ctx.design.cell(pin.cell).name).to_owned()
}

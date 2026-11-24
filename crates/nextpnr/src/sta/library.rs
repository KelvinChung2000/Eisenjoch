use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::FxHashMap;
use rusteda_core::liberty::{Cell, Direction, Library, Pin, Table, TimingArc, Units};
use ruststa::engine::delay::annotate::DelayAnnotations;
use ruststa::engine::incr::SessionNodeKey;

use super::interconnect::DelayBounds;
use super::role::{CellRole, CellTiming, Roles};
use super::{StaBuildError, PS_IN_NS};
use crate::chipdb::{ChipDb, SpeedGradePod};
use crate::common::IdString;
use crate::common::{ClockEdge, DelayT};
use crate::context::Context;
use crate::netlist::{CellPin, NetId, PortType};

#[derive(Debug, Clone, Copy)]
pub(super) struct TypePort {
    pub id: IdString,
    pub dir: PortType,
}

#[derive(Debug, Clone, Copy)]
enum DelayArcKind {
    Combinational,
    Launch(ClockEdge),
}

#[derive(Debug, Clone, Copy)]
struct DelayArc {
    from: TypePort,
    to: TypePort,
    kind: DelayArcKind,
    bounds: DelayBounds,
}

#[derive(Debug)]
pub(super) struct CellType {
    pub timing: CellTiming,
    pub ports: BTreeMap<String, TypePort>,
    delay_arcs: Vec<DelayArc>,
}

pub(super) fn cell_types(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    roles: &Roles,
) -> Result<BTreeMap<String, CellType>, StaBuildError> {
    let mut types: BTreeMap<String, CellType> = BTreeMap::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let CellRole::Instance(timing) = roles.cells[&cell_id] else {
            continue;
        };
        let type_name = ctx.name_of(cell.cell_type).to_owned();
        let cell_type = types.entry(type_name.clone()).or_insert_with(|| CellType {
            timing,
            ports: BTreeMap::new(),
            delay_arcs: Vec::new(),
        });
        for (&port, data) in &cell.ports {
            let port_name = ctx.name_of(port);
            let declared = *cell_type
                .ports
                .entry(port_name.to_owned())
                .or_insert(TypePort {
                    id: port,
                    dir: data.port_type(),
                });
            if declared.dir != data.port_type() {
                return Err(StaBuildError::PortTypeConflict {
                    cell_type: type_name,
                    port: port_name.to_owned(),
                    first: declared.dir,
                    second: data.port_type(),
                });
            }
        }
    }
    for cell_type in types.values_mut() {
        declare_clock_pins(ctx, speed_grade, cell_type);
    }
    for cell_type in types.values_mut() {
        cell_type.delay_arcs = delay_arcs(ctx, speed_grade, cell_type);
    }
    Ok(types)
}

pub(super) fn register_clock_nets(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    roles: &Roles,
) -> BTreeSet<NetId> {
    let mut nets = BTreeSet::new();
    for (&cell_id, role) in &roles.cells {
        let CellRole::Instance(CellTiming::SpeedGrade { type_idx }) = role else {
            continue;
        };
        let cell = ctx.design.cell(cell_id);
        for (&id, port) in &cell.ports {
            let port = TypePort {
                id,
                dir: port.port_type(),
            };
            let clocks = register_arcs(ctx, speed_grade, *type_idx, &port)
                .map(|(_, _, pod)| IdString(ChipDb::reg_arc_info(pod).clock_port));
            nets.extend(clocks.filter_map(|clock| cell.port_net(clock)));
        }
    }
    nets
}

fn declare_clock_pins(ctx: &Context, speed_grade: &SpeedGradePod, cell_type: &mut CellType) {
    let CellTiming::SpeedGrade { type_idx } = cell_type.timing else {
        return;
    };
    let clocks: Vec<IdString> = cell_type
        .ports
        .values()
        .flat_map(|port| register_arcs(ctx, speed_grade, type_idx, port))
        .map(|(_, _, pod)| IdString(ChipDb::reg_arc_info(pod).clock_port))
        .collect();
    for clock in clocks {
        cell_type
            .ports
            .entry(ctx.name_of(clock).to_owned())
            .or_insert(TypePort {
                id: clock,
                dir: PortType::In,
            });
    }
}

fn delay_arcs(ctx: &Context, speed_grade: &SpeedGradePod, cell_type: &CellType) -> Vec<DelayArc> {
    let CellTiming::SpeedGrade { type_idx } = cell_type.timing else {
        return Vec::new();
    };
    let chipdb = ctx.chipdb();
    let outputs = cell_type
        .ports
        .values()
        .filter(|port| matches!(port.dir, PortType::Out | PortType::InOut));
    let mut arcs = Vec::new();
    for &output in outputs {
        for input in cell_type
            .ports
            .values()
            .filter(|port| port.dir == PortType::In)
        {
            let timing = chipdb.cell_comb_timing(
                speed_grade,
                type_idx,
                input.id.index(),
                output.id.index(),
            );
            if let Some(timing) = timing {
                arcs.push(DelayArc {
                    from: *input,
                    to: output,
                    kind: DelayArcKind::Combinational,
                    bounds: DelayBounds::of(timing),
                });
            }
        }
        for (clock, edge, pod) in register_arcs(ctx, speed_grade, type_idx, &output) {
            arcs.push(DelayArc {
                from: cell_type.ports[clock],
                to: output,
                kind: DelayArcKind::Launch(edge),
                bounds: DelayBounds::of(ChipDb::reg_arc_clock_to_q(pod)),
            });
        }
    }
    arcs
}

pub(super) fn cell_delay_annotations(
    ctx: &Context,
    roles: &Roles,
    types: &BTreeMap<String, CellType>,
    pins: &FxHashMap<CellPin, SessionNodeKey>,
) -> DelayAnnotations<SessionNodeKey> {
    let mut annotations = DelayAnnotations::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let CellRole::Instance(_) = roles.cells[&cell_id] else {
            continue;
        };
        for arc in &types[ctx.name_of(cell.cell_type)].delay_arcs {
            let ends = (
                pins.get(&CellPin::new(cell_id, arc.from.id)),
                pins.get(&CellPin::new(cell_id, arc.to.id)),
            );
            if let (Some(&from), Some(&to)) = ends {
                annotations.insert(from, to, arc.bounds.arc_delays());
            }
        }
    }
    annotations
}

pub(super) fn library(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    types: &BTreeMap<String, CellType>,
) -> Library {
    let cells = types
        .iter()
        .map(|(name, cell_type)| liberty_cell(ctx, speed_grade, name, cell_type))
        .collect();
    let mut units = Units::default();
    units.time.scale = PS_IN_NS;
    Library {
        name: String::from("eisenjoch"),
        time_unit_ns: PS_IN_NS,
        cap_unit_pf: 1.0,
        units,
        cells,
        ..Library::default()
    }
}

fn liberty_cell(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    name: &str,
    cell_type: &CellType,
) -> Cell {
    let pins = cell_type
        .ports
        .iter()
        .map(|(pin_name, port)| {
            let arcs = match port.dir {
                PortType::In => check_arcs(ctx, speed_grade, cell_type, port),
                PortType::Out | PortType::InOut => cell_type
                    .delay_arcs
                    .iter()
                    .filter(|arc| arc.to.id == port.id)
                    .map(|arc| delay_arc(ctx, arc))
                    .collect(),
            };
            Pin {
                name: pin_name.clone(),
                direction: direction(port.dir),
                arcs,
                ..Pin::default()
            }
        })
        .collect();
    Cell {
        name: name.to_owned(),
        pins,
        ..Cell::default()
    }
}

fn delay_arc(ctx: &Context, arc: &DelayArc) -> TimingArc {
    let (timing_sense, timing_type) = match arc.kind {
        DelayArcKind::Combinational => (
            Some(String::from("non_unate")),
            String::from("combinational"),
        ),
        DelayArcKind::Launch(edge) => (None, format!("{}_edge", edge_name(edge))),
    };
    TimingArc {
        related_pin: ctx.name_of(arc.from.id).to_owned(),
        timing_sense,
        timing_type,
        ..delay_tables(arc.bounds.late_ps)
    }
}

fn check_arcs(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    cell_type: &CellType,
    input: &TypePort,
) -> Vec<TimingArc> {
    let CellTiming::SpeedGrade { type_idx } = cell_type.timing else {
        return Vec::new();
    };
    let mut arcs = Vec::new();
    for (clock, edge, pod) in register_arcs(ctx, speed_grade, type_idx, input) {
        let info = ChipDb::reg_arc_info(pod);
        let related_pin = clock.to_owned();
        for (check, value) in [
            ("setup", info.setup.max_delay),
            ("hold", info.hold.max_delay),
        ] {
            arcs.push(TimingArc {
                related_pin: related_pin.clone(),
                timing_type: format!("{check}_{}", edge_name(edge)),
                rise_constraint: scalar(value),
                fall_constraint: scalar(value),
                ..TimingArc::default()
            });
        }
    }
    arcs
}

fn register_arcs<'a>(
    ctx: &'a Context,
    speed_grade: &'a SpeedGradePod,
    type_idx: usize,
    port: &TypePort,
) -> impl Iterator<Item = (&'a str, ClockEdge, &'a crate::chipdb::CellPinRegArcPod)> + 'a {
    ctx.chipdb()
        .cell_reg_arcs(speed_grade, type_idx, port.id.index())
        .unwrap_or_default()
        .iter()
        .map(|pod| {
            let info = ChipDb::reg_arc_info(pod);
            (ctx.name_of(IdString(info.clock_port)), info.edge, pod)
        })
}

fn delay_tables(delay: DelayT) -> TimingArc {
    TimingArc {
        cell_rise: scalar(delay),
        cell_fall: scalar(delay),
        rise_transition: scalar(0),
        fall_transition: scalar(0),
        ..TimingArc::default()
    }
}

fn scalar(value: DelayT) -> Option<Table> {
    Some(Table {
        values: vec![f64::from(value)],
        ..Table::default()
    })
}

fn edge_name(edge: ClockEdge) -> &'static str {
    match edge {
        ClockEdge::Rising => "rising",
        ClockEdge::Falling => "falling",
    }
}

fn direction(dir: PortType) -> Direction {
    match dir {
        PortType::In => Direction::Input,
        PortType::Out => Direction::Output,
        PortType::InOut => Direction::Inout,
    }
}

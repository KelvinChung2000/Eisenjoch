use std::collections::BTreeMap;

use rusteda_core::liberty::{Cell, Direction, Library, Pin, Table, TimingArc, Units};

use super::role::{CellRole, Roles};
use super::{StaBuildError, PS_IN_NS};
use crate::chipdb::{ChipDb, SpeedGradePod};
use crate::common::IdString;
use crate::context::Context;
use crate::netlist::PortType;
use crate::common::{ClockEdge, DelayT};

#[derive(Debug, Clone, Copy)]
pub(super) struct TypePort {
    pub id: IdString,
    pub dir: PortType,
}

#[derive(Debug)]
pub(super) struct CellType {
    pub type_idx: usize,
    pub ports: BTreeMap<String, TypePort>,
}

pub(super) fn cell_types(
    ctx: &Context,
    roles: &Roles,
) -> Result<BTreeMap<String, CellType>, StaBuildError> {
    let mut types: BTreeMap<String, CellType> = BTreeMap::new();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let CellRole::Timed { type_idx } = roles.cells[&cell_id] else {
            continue;
        };
        let type_name = ctx.name_of(cell.cell_type).to_owned();
        let cell_type = types.entry(type_name.clone()).or_insert_with(|| CellType {
            type_idx,
            ports: BTreeMap::new(),
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
    Ok(types)
}

pub(super) fn library(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    types: &BTreeMap<String, CellType>,
) -> Result<Library, StaBuildError> {
    let cells = types
        .iter()
        .map(|(name, cell_type)| liberty_cell(ctx, speed_grade, name, cell_type))
        .collect::<Result<Vec<_>, _>>()?;
    let mut units = Units::default();
    units.time.scale = PS_IN_NS;
    Ok(Library {
        name: String::from("eisenjoch"),
        time_unit_ns: PS_IN_NS,
        cap_unit_pf: 1.0,
        units,
        cells,
        ..Library::default()
    })
}

fn liberty_cell(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    name: &str,
    cell_type: &CellType,
) -> Result<Cell, StaBuildError> {
    let chipdb = ctx.chipdb();
    let type_idx = cell_type.type_idx;
    let pins = cell_type
        .ports
        .iter()
        .map(|(pin_name, port)| {
            let arcs = match port.dir {
                PortType::In => check_arcs(ctx, speed_grade, type_idx, name, cell_type, port)?,
                PortType::Out | PortType::InOut => {
                    let mut arcs =
                        combinational_arcs(chipdb, speed_grade, type_idx, cell_type, port);
                    arcs.extend(launch_arcs(
                        ctx,
                        speed_grade,
                        type_idx,
                        name,
                        cell_type,
                        port,
                    )?);
                    arcs
                }
            };
            Ok(Pin {
                name: pin_name.clone(),
                direction: direction(port.dir),
                arcs,
                ..Pin::default()
            })
        })
        .collect::<Result<Vec<_>, StaBuildError>>()?;
    Ok(Cell {
        name: name.to_owned(),
        pins,
        ..Cell::default()
    })
}

fn combinational_arcs(
    chipdb: &ChipDb,
    speed_grade: &SpeedGradePod,
    type_idx: usize,
    cell_type: &CellType,
    output: &TypePort,
) -> Vec<TimingArc> {
    cell_type
        .ports
        .iter()
        .filter(|(_, input)| input.dir == PortType::In)
        .filter_map(|(input_name, input)| {
            let delay =
                chipdb.cell_delay(speed_grade, type_idx, input.id.index(), output.id.index())?;
            Some(TimingArc {
                related_pin: input_name.clone(),
                timing_sense: Some(String::from("non_unate")),
                timing_type: String::from("combinational"),
                ..delay_tables(delay.max_delay())
            })
        })
        .collect()
}

fn launch_arcs(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    type_idx: usize,
    type_name: &str,
    cell_type: &CellType,
    output: &TypePort,
) -> Result<Vec<TimingArc>, StaBuildError> {
    register_arcs(ctx, speed_grade, type_idx, output)
        .map(|(clock, edge, info)| {
            Ok(TimingArc {
                related_pin: clock_pin(type_name, cell_type, clock)?,
                timing_type: format!("{}_edge", edge_name(edge)),
                ..delay_tables(info.clock_to_q.max_delay())
            })
        })
        .collect()
}

fn check_arcs(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    type_idx: usize,
    type_name: &str,
    cell_type: &CellType,
    input: &TypePort,
) -> Result<Vec<TimingArc>, StaBuildError> {
    let mut arcs = Vec::new();
    for (clock, edge, info) in register_arcs(ctx, speed_grade, type_idx, input) {
        let related_pin = clock_pin(type_name, cell_type, clock)?;
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
    Ok(arcs)
}

fn register_arcs<'a>(
    ctx: &'a Context,
    speed_grade: &'a SpeedGradePod,
    type_idx: usize,
    port: &TypePort,
) -> impl Iterator<Item = (&'a str, ClockEdge, crate::chipdb::RegArcInfo)> + 'a {
    ctx.chipdb()
        .cell_reg_arcs(speed_grade, type_idx, port.id.index())
        .unwrap_or_default()
        .iter()
        .map(|pod| {
            let info = ChipDb::reg_arc_info(pod);
            (ctx.name_of(IdString(info.clock_port)), info.edge, info)
        })
}

fn clock_pin(type_name: &str, cell_type: &CellType, clock: &str) -> Result<String, StaBuildError> {
    match cell_type.ports.get(clock) {
        Some(_) => Ok(clock.to_owned()),
        None => Err(StaBuildError::UnknownPin {
            cell_type: type_name.to_owned(),
            pin: clock.to_owned(),
        }),
    }
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

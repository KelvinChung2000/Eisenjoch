use std::collections::BTreeMap;

use super::interconnect::{DelayBounds, InterconnectModel, InterconnectStage};
use super::library::{self, CellType};
use super::role::{self, CellRole, CellTiming, Roles};
use super::session::StaSessionError;
use crate::chipdb::SpeedGradePod;
use crate::common::{ClockEdge, DelayT, IdString};
use crate::context::Context;
use crate::netlist::CellId;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PinRef {
    pub instance: String,
    pub port: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterconnectDelay {
    pub source: PinRef,
    pub load: PinRef,
    pub delay: DelayBounds,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IopathDelay {
    pub from: String,
    pub to: String,
    pub launch_edge: Option<ClockEdge>,
    pub delay: DelayBounds,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimingCheck {
    pub data: String,
    pub clock: String,
    pub edge: ClockEdge,
    pub setup_ps: DelayT,
    pub hold_ps: DelayT,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellDelays {
    pub instance: String,
    pub cell_type: String,
    pub iopaths: Vec<IopathDelay>,
    pub checks: Vec<TimingCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackAnnotation {
    pub design: String,
    pub cells: Vec<CellDelays>,
    pub interconnects: Vec<InterconnectDelay>,
}

pub fn routed_back_annotation(
    ctx: &Context,
    model: &InterconnectModel,
) -> Result<BackAnnotation, StaSessionError> {
    let speed_grade = super::speed_grade(ctx)?;
    let roles = role::roles(ctx, speed_grade)?;
    let types = library::cell_types(ctx, speed_grade, &roles)?;
    Ok(BackAnnotation {
        design: ctx.name_of(ctx.design.top_module).to_owned(),
        cells: cell_delays(ctx, speed_grade, &roles, &types),
        interconnects: interconnects(ctx, model, &roles)?,
    })
}

fn cell_delays(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    roles: &Roles,
    types: &BTreeMap<String, CellType>,
) -> Vec<CellDelays> {
    let mut cells: Vec<CellDelays> = ctx
        .design
        .iter_alive_cells()
        .filter(|(cell_id, _)| {
            matches!(
                roles.cells[cell_id],
                CellRole::Instance(CellTiming::SpeedGrade { .. })
            )
        })
        .map(|(_, cell)| {
            let cell_type = &types[ctx.name_of(cell.cell_type)];
            CellDelays {
                instance: ctx.name_of(cell.name).to_owned(),
                cell_type: ctx.name_of(cell.cell_type).to_owned(),
                iopaths: library::iopaths(ctx, cell_type),
                checks: library::timing_checks(ctx, speed_grade, cell_type),
            }
        })
        .collect();
    cells.sort_by(|a, b| a.instance.cmp(&b.instance));
    cells
}

fn interconnects(
    ctx: &Context,
    model: &InterconnectModel,
    roles: &Roles,
) -> Result<Vec<InterconnectDelay>, StaSessionError> {
    let mut delays = Vec::new();
    for (net_id, net) in ctx.design.iter_alive_nets() {
        let Some(driver) = net.driver() else {
            continue;
        };
        if roles.off_chip_nets.contains(&net_id)
            || matches!(roles.cells[&driver.cell], CellRole::Constant(_))
        {
            continue;
        }
        let users: Vec<u32> = net
            .users()
            .iter()
            .enumerate()
            .filter(|(_, user)| user.is_valid())
            .map(|(user, _)| u32::try_from(user).expect("net user index fits in u32"))
            .collect();
        let source = pin_ref(ctx, driver.cell, driver.port);
        for (user, delay) in model.net_delays(ctx, InterconnectStage::Routed, net_id, &users)? {
            let load = net.users()[user as usize];
            delays.push(InterconnectDelay {
                source: source.clone(),
                load: pin_ref(ctx, load.cell, load.port),
                delay,
            });
        }
    }
    delays.sort_by(|a, b| (&a.source, &a.load).cmp(&(&b.source, &b.load)));
    Ok(delays)
}

fn pin_ref(ctx: &Context, cell: CellId, port: IdString) -> PinRef {
    PinRef {
        instance: ctx.name_of(ctx.design.cell(cell).name).to_owned(),
        port: ctx.name_of(port).to_owned(),
    }
}

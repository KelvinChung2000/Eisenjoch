use rustc_hash::FxHashMap;
use rusteda_core::netlist::PortDir;

use super::StaBuildError;
use crate::chipdb::SpeedGradePod;
use crate::common::IdString;
use crate::context::Context;
use crate::frontend::IO_CELL_PREFIX;
use crate::netlist::{CellId, CellInfo, NetId, PortType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Level {
    Low,
    High,
}

impl Level {
    pub fn is_high(self) -> bool {
        self == Level::High
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Pad {
    pub name: String,
    pub dir: PortDir,
    pub port: IdString,
    pub net: Option<NetId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CellRole {
    Timed { type_idx: usize },
    Pad(Pad),
    Constant(Level),
}

pub(super) type Roles = FxHashMap<CellId, CellRole>;

pub(super) fn roles(ctx: &Context, speed_grade: &SpeedGradePod) -> Result<Roles, StaBuildError> {
    ctx.design
        .iter_alive_cells()
        .map(|(cell_id, cell)| Ok((cell_id, cell_role(ctx, speed_grade, cell)?)))
        .collect()
}

fn cell_role(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    cell: &CellInfo,
) -> Result<CellRole, StaBuildError> {
    match ctx.name_of(cell.cell_type) {
        "$nextpnr_IBUF" | "$nextpnr_OBUF" | "$nextpnr_IOBUF" | "IOB" => {
            pad(ctx, cell).map(CellRole::Pad)
        }
        "GND" | "GND_DRV" => Ok(CellRole::Constant(Level::Low)),
        "VCC" | "VCC_DRV" => Ok(CellRole::Constant(Level::High)),
        type_name => ctx
            .chipdb()
            .cell_timing_index(speed_grade, cell.cell_type.index())
            .map(|type_idx| CellRole::Timed { type_idx })
            .ok_or_else(|| StaBuildError::CellTypeWithoutTiming {
                cell_type: type_name.to_owned(),
            }),
    }
}

fn pad(ctx: &Context, cell: &CellInfo) -> Result<Pad, StaBuildError> {
    let cell_name = ctx.name_of(cell.name);
    let name = cell_name.strip_prefix(IO_CELL_PREFIX).ok_or_else(|| {
        StaBuildError::PadWithoutPortName {
            cell: cell_name.to_owned(),
        }
    })?;
    let [(&port, data)] = cell.ports.iter().collect::<Vec<_>>()[..] else {
        return Err(StaBuildError::PadPortCount {
            cell: cell_name.to_owned(),
            ports: cell.ports.len(),
        });
    };
    Ok(Pad {
        name: name.to_owned(),
        dir: match data.port_type() {
            PortType::Out => PortDir::Input,
            PortType::In => PortDir::Output,
            PortType::InOut => PortDir::Inout,
        },
        port,
        net: data.net(),
    })
}

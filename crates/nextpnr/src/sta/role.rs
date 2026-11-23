use rustc_hash::{FxHashMap, FxHashSet};
use rusteda_core::netlist::PortDir;

use super::StaBuildError;
use crate::chipdb::SpeedGradePod;
use crate::common::IdString;
use crate::context::Context;
use crate::frontend::IO_CELL_PREFIX;
use crate::netlist::{CellId, CellInfo, NetId, PortType};

const TOP_PORT_TYPES: [&str; 3] = ["$nextpnr_IBUF", "$nextpnr_OBUF", "$nextpnr_IOBUF"];
const IO_BUFFER_TYPES: [&str; 5] = ["IOB", "INBUF", "OUTBUF", "IBUF", "OBUF"];

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
    OffChip,
}

#[derive(Debug)]
pub(super) struct Roles {
    pub cells: FxHashMap<CellId, CellRole>,
    pub off_chip_nets: FxHashSet<NetId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IoKind {
    TopPort,
    Buffer,
}

fn io_kind(ctx: &Context, cell: &CellInfo) -> Option<IoKind> {
    let type_name = ctx.name_of(cell.cell_type);
    if TOP_PORT_TYPES.contains(&type_name) {
        Some(IoKind::TopPort)
    } else if IO_BUFFER_TYPES.contains(&type_name) {
        Some(IoKind::Buffer)
    } else {
        None
    }
}

pub(super) fn roles(ctx: &Context, speed_grade: &SpeedGradePod) -> Result<Roles, StaBuildError> {
    let off_chip_nets = off_chip_nets(ctx);
    let cells = ctx
        .design
        .iter_alive_cells()
        .map(|(cell_id, cell)| Ok((cell_id, cell_role(ctx, speed_grade, &off_chip_nets, cell)?)))
        .collect::<Result<_, StaBuildError>>()?;
    Ok(Roles {
        cells,
        off_chip_nets,
    })
}

fn off_chip_nets(ctx: &Context) -> FxHashSet<NetId> {
    ctx.design
        .iter_alive_nets()
        .filter(|(_, info)| {
            let kinds: Vec<Option<IoKind>> = info
                .driver()
                .into_iter()
                .chain(info.users().iter().copied().filter(|pin| pin.is_valid()))
                .map(|pin| io_kind(ctx, ctx.design.cell(pin.cell)))
                .collect();
            let buffers = kinds.iter().filter(|&&kind| kind == Some(IoKind::Buffer)).count();
            let top_ports = kinds.iter().filter(|&&kind| kind == Some(IoKind::TopPort)).count();
            buffers == 1 && buffers + top_ports == kinds.len()
        })
        .map(|(net, _)| net)
        .collect()
}

fn cell_role(
    ctx: &Context,
    speed_grade: &SpeedGradePod,
    off_chip_nets: &FxHashSet<NetId>,
    cell: &CellInfo,
) -> Result<CellRole, StaBuildError> {
    match io_kind(ctx, cell) {
        Some(IoKind::TopPort) => top_port_role(ctx, off_chip_nets, cell),
        Some(IoKind::Buffer) => buffer_role(ctx, off_chip_nets, cell),
        None => match ctx.name_of(cell.cell_type) {
            "GND" | "GND_DRV" => Ok(CellRole::Constant(Level::Low)),
            "VCC" | "VCC_DRV" => Ok(CellRole::Constant(Level::High)),
            type_name => ctx
                .chipdb()
                .cell_timing_index(speed_grade, cell.cell_type.index())
                .map(|type_idx| CellRole::Timed { type_idx })
                .ok_or_else(|| StaBuildError::CellTypeWithoutTiming {
                    cell_type: type_name.to_owned(),
                }),
        },
    }
}

fn top_port_role(
    ctx: &Context,
    off_chip_nets: &FxHashSet<NetId>,
    cell: &CellInfo,
) -> Result<CellRole, StaBuildError> {
    let pad = pad(ctx, cell)?;
    match pad.net.is_some_and(|net| off_chip_nets.contains(&net)) {
        true => Ok(CellRole::OffChip),
        false => Ok(CellRole::Pad(pad)),
    }
}

fn buffer_role(
    ctx: &Context,
    off_chip_nets: &FxHashSet<NetId>,
    cell: &CellInfo,
) -> Result<CellRole, StaBuildError> {
    let (pad_side, fabric_side): (Vec<_>, Vec<_>) = cell
        .ports
        .iter()
        .filter_map(|(&port, data)| Some((port, data.port_type(), data.net()?)))
        .partition(|&(_, _, net)| off_chip_nets.contains(&net));
    match (&pad_side[..], &fabric_side[..]) {
        ([], _) => pad(ctx, cell).map(CellRole::Pad),
        ([_], []) => Ok(CellRole::OffChip),
        ([(_, _, pad_net)], [(port, port_type, net)]) => Ok(CellRole::Pad(Pad {
            name: ctx.name_of(ctx.design.net(*pad_net).name).to_owned(),
            dir: pad_dir(*port_type),
            port: *port,
            net: Some(*net),
        })),
        _ => Err(StaBuildError::IoBufferSides {
            cell: ctx.name_of(cell.name).to_owned(),
            pad_side: pad_side.len(),
            fabric_side: fabric_side.len(),
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
        dir: pad_dir(data.port_type()),
        port,
        net: data.net(),
    })
}

fn pad_dir(fabric_port: PortType) -> PortDir {
    match fabric_port {
        PortType::Out => PortDir::Input,
        PortType::In => PortDir::Output,
        PortType::InOut => PortDir::Inout,
    }
}

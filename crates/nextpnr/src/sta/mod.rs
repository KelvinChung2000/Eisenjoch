mod interconnect;
mod library;
mod report;
mod role;
mod session;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rustc_hash::{FxHashMap, FxHashSet};
use rusteda_core::liberty::{Direction, Library};
use rusteda_core::netlist::{self as gate, Conn, GateNetlist, NetlistError, PinDir, PortDir};
use ruststa::engine::analysis::{AnalysisOptions, WireModelOption};
use ruststa::engine::incr::{SessionError, SessionNodeKey, TimingSession};
use ruststa::engine::session_api::EndpointSlack;
use ruststa::engine::types::NodeId;
use thiserror::Error;

use crate::context::Context;
use crate::netlist::{CellId, CellPin, NetId, PortType};
use role::{CellRole, Roles};

pub use interconnect::{
    ClasslessPips, DelayBounds, InterconnectError, InterconnectModel, InterconnectStage,
};

pub const NOTIONAL_CLASSLESS_PIPS: ClasslessPips = ClasslessPips::Notional(DelayBounds {
    early_ps: 100,
    late_ps: 100,
});
pub use ruststa::engine::incr::{ArcTiming, Criticality};
pub use report::{DomainReport, PathPinReport, PathReport, TimingReport};
pub use ruststa::engine::types::ClockId;
pub use session::{DomainSummary, SinkTiming, StaSession, StaSessionError, Untimed};

const PS_IN_NS: f64 = 0.001;
const PS_PER_NS: f64 = 1000.0;

#[derive(Debug, Error)]
pub enum StaBuildError {
    #[error("chipdb has no speed grade {index}")]
    MissingSpeedGrade { index: usize },
    #[error("cell type '{cell_type}' has no timing entry in the chipdb speed grade")]
    CellTypeWithoutTiming { cell_type: String },
    #[error("cell type '{cell_type}' has port '{port}' as both {first:?} and {second:?}")]
    PortTypeConflict {
        cell_type: String,
        port: String,
        first: PortType,
        second: PortType,
    },
    #[error("chipdb register arc of cell type '{cell_type}' names clock pin '{pin}', which the cell type does not have")]
    UnknownPin { cell_type: String, pin: String },
    #[error("IO cell '{cell}' is not named after a top-level port")]
    PadWithoutPortName { cell: String },
    #[error("IO cell '{cell}' has {ports} ports, a pad has exactly one")]
    PadPortCount { cell: String, ports: usize },
    #[error("IO buffer '{cell}' has {pad_side} pad-side and {fabric_side} fabric-side connections, a pad has one of each")]
    IoBufferSides {
        cell: String,
        pad_side: usize,
        fabric_side: usize,
    },
    #[error("clock net '{net}' has no driver")]
    ClockNetWithoutDriver { net: String },
    #[error("clock net '{net}' is driven by a constant")]
    ClockNetDrivenByConstant { net: String },
    #[error("clock net '{net}' is off chip, between a top-level port and its IO buffer")]
    ClockNetOffChip { net: String },
    #[error("RustSTA netlist: {0}")]
    Netlist(#[from] NetlistError),
    #[error("RustSTA session: {0}")]
    Session(#[from] SessionError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetSink {
    pub net: NetId,
    pub user: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinLink {
    pub key: SessionNodeKey,
    pub node: NodeId,
}

#[derive(Debug)]
pub struct StaIds {
    pub pins: FxHashMap<CellPin, PinLink>,
    pub node_pins: Vec<Option<CellPin>>,
    pub nets: FxHashMap<NetId, gate::NetId>,
    pub net_arcs: FxHashMap<NetSink, usize>,
    pub off_chip_nets: FxHashSet<NetId>,
}

#[derive(Debug)]
pub struct RustSta {
    session: TimingSession,
    ids: StaIds,
}

impl RustSta {
    pub fn build(ctx: &Context) -> Result<Self, StaBuildError> {
        let speed_grade = ctx
            .speed_grade()
            .ok_or_else(|| StaBuildError::MissingSpeedGrade {
                index: ctx.speed_grade_idx(),
            })?;
        let roles = role::roles(ctx, speed_grade)?;
        let library = library::library(ctx, speed_grade, &library::cell_types(ctx, &roles)?)?;
        let GateDesign {
            netlist,
            pins,
            nets,
        } = gate_design(ctx, &library, &roles)?;
        let sdc = format!("{}{}", clock_sdc(ctx, &roles)?, ctx.design.sdc);
        let options = AnalysisOptions {
            wire_model: WireModelOption::Ideal,
            ..AnalysisOptions::default()
        };
        let session = TimingSession::new(netlist, vec![library], &sdc, options)?;
        let ids = link_ids(ctx, &session, roles, pins, nets);
        Ok(Self { session, ids })
    }

    pub fn session(&self) -> &TimingSession {
        &self.session
    }

    pub fn ids(&self) -> &StaIds {
        &self.ids
    }

    pub fn endpoint_slack(&self, pin: CellPin) -> Option<EndpointSlack> {
        let node = self.ids.pins.get(&pin)?.node;
        let record = self
            .session
            .default_corner()
            .endpoints
            .endpoints
            .get(&node)?;
        Some(EndpointSlack {
            setup_ns: record.setup.as_ref().map(|check| check.slack),
            hold_ns: record.hold.as_ref().map(|check| check.slack),
        })
    }
}

struct GateDesign {
    netlist: GateNetlist,
    pins: FxHashMap<CellPin, SessionNodeKey>,
    nets: FxHashMap<NetId, gate::NetId>,
}

fn gate_design(
    ctx: &Context,
    library: &Library,
    roles: &Roles,
) -> Result<GateDesign, StaBuildError> {
    let mut netlist = GateNetlist::new("top")?;
    let mut cell_types = BTreeMap::new();
    for cell in &library.cells {
        let pins: Vec<(&str, PinDir)> = cell
            .pins
            .iter()
            .map(|pin| (pin.name.as_str(), pin_dir(pin.direction)))
            .collect();
        cell_types.insert(
            cell.name.as_str(),
            netlist.add_cell_type(&cell.name, &pins)?,
        );
    }
    let mut nets = FxHashMap::default();
    for (net, _) in ctx.design.iter_alive_nets() {
        if roles.off_chip_nets.contains(&net) {
            continue;
        }
        nets.insert(net, netlist.add_net(Some(&net_name(net)), None)?);
    }
    let mut pins = FxHashMap::default();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        match &roles.cells[&cell_id] {
            CellRole::Timed { .. } => {
                let inst = netlist.add_instance(
                    &instance_name(cell_id),
                    cell_types[ctx.name_of(cell.cell_type)],
                )?;
                for (&port, data) in &cell.ports {
                    let Some(net) = data.net() else { continue };
                    let pin = netlist.connect(inst, ctx.name_of(port), Conn::Net(nets[&net]))?;
                    pins.insert(CellPin::new(cell_id, port), SessionNodeKey::Pin(pin));
                }
            }
            CellRole::Pad(pad) => {
                let port = netlist.add_port(&pad.name, pad.dir, None)?;
                let Some(net) = pad.net else { continue };
                let (design_net, port_net) = (nets[&net], netlist.ports()[port].bits[0]);
                match pad.dir {
                    PortDir::Input | PortDir::Inout => {
                        netlist.add_assign(design_net, Conn::Net(port_net))?
                    }
                    PortDir::Output => netlist.add_assign(port_net, Conn::Net(design_net))?,
                }
                let bit = u32::try_from(port).expect("scalar port index fits in u32");
                pins.insert(
                    CellPin::new(cell_id, pad.port),
                    SessionNodeKey::PortBit(bit),
                );
            }
            CellRole::Constant(level) => {
                for data in cell.ports.values() {
                    let Some(net) = data.net() else { continue };
                    netlist.add_assign(nets[&net], Conn::Const(level.is_high()))?;
                }
            }
            CellRole::OffChip => {}
        }
    }
    Ok(GateDesign {
        netlist,
        pins,
        nets,
    })
}

fn clock_sdc(ctx: &Context, roles: &Roles) -> Result<String, StaBuildError> {
    let mut sdc = String::from("set_units -time ns\n");
    for (net_id, net) in ctx.design.iter_alive_nets() {
        if net.clock_constraint <= 0 {
            continue;
        }
        let driver = net
            .driver()
            .ok_or_else(|| StaBuildError::ClockNetWithoutDriver {
                net: ctx.name_of(net.name).to_owned(),
            })?;
        if roles.off_chip_nets.contains(&net_id) {
            return Err(StaBuildError::ClockNetOffChip {
                net: ctx.name_of(net.name).to_owned(),
            });
        }
        let source = match &roles.cells[&driver.cell] {
            CellRole::Timed { .. } => format!(
                "get_pins {{{}/{}}}",
                instance_name(driver.cell),
                glob_escape(ctx.name_of(driver.port)),
            ),
            CellRole::Pad(pad) => format!("get_ports {{{}}}", glob_escape(&pad.name)),
            CellRole::Constant(_) => {
                return Err(StaBuildError::ClockNetDrivenByConstant {
                    net: ctx.name_of(net.name).to_owned(),
                });
            }
            CellRole::OffChip => unreachable!("an off-chip cell drives only off-chip nets"),
        };
        writeln!(
            sdc,
            "create_clock -name {{{}}} -period {} [{source}]",
            ctx.name_of(net.name),
            f64::from(net.clock_constraint) / PS_PER_NS,
        )
        .expect("writing to a String cannot fail");
    }
    Ok(sdc)
}

fn link_ids(
    ctx: &Context,
    session: &TimingSession,
    roles: Roles,
    gate_pins: FxHashMap<CellPin, SessionNodeKey>,
    nets: FxHashMap<NetId, gate::NetId>,
) -> StaIds {
    let mut node_pins = vec![None; session.graph().num_nodes()];
    let pins: FxHashMap<CellPin, PinLink> = gate_pins
        .into_iter()
        .map(|(cell_pin, key)| {
            let node = session
                .node_of_key(key)
                .expect("every connected pin and port is a timing graph node");
            node_pins[node.index()] = Some(cell_pin);
            (cell_pin, PinLink { key, node })
        })
        .collect();
    let arc_by_nodes: FxHashMap<(NodeId, NodeId), usize> = session
        .graph()
        .arcs()
        .iter()
        .enumerate()
        .filter(|(_, arc)| arc.kind.is_net())
        .map(|(index, arc)| ((arc.src, arc.dst), index))
        .collect();
    let mut net_arcs = FxHashMap::default();
    for (net_id, net) in ctx.design.iter_alive_nets() {
        let Some(driver) = net.driver() else { continue };
        if roles.off_chip_nets.contains(&net_id) {
            continue;
        }
        if let CellRole::Constant(_) = roles.cells[&driver.cell] {
            continue;
        }
        let driver_node = pins[&driver].node;
        for (user, &sink) in net.users().iter().enumerate() {
            if !sink.is_valid() {
                continue;
            }
            let arc = arc_by_nodes[&(driver_node, pins[&sink].node)];
            let user = u32::try_from(user).expect("net user index fits in u32");
            net_arcs.insert(NetSink { net: net_id, user }, arc);
        }
    }
    StaIds {
        pins,
        node_pins,
        nets,
        net_arcs,
        off_chip_nets: roles.off_chip_nets,
    }
}

fn instance_name(cell: CellId) -> String {
    format!("c{}", cell.slot())
}

fn net_name(net: NetId) -> String {
    format!("n{}", net.slot())
}

fn glob_escape(name: &str) -> String {
    name.chars().fold(String::new(), |mut escaped, ch| {
        if matches!(ch, '\\' | '*' | '?' | '[' | ']' | '{' | '}') {
            escaped.push('\\');
        }
        escaped.push(ch);
        escaped
    })
}

fn pin_dir(direction: Direction) -> PinDir {
    match direction {
        Direction::Input => PinDir::Input,
        Direction::Output => PinDir::Output,
        Direction::Inout => PinDir::Inout,
        Direction::Internal => PinDir::Other,
    }
}

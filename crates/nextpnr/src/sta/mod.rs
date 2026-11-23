mod library;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rustc_hash::FxHashMap;
use rusteda_core::liberty::{Direction, Library};
use rusteda_core::netlist::{self as gate, Conn, GateNetlist, NetlistError, PinDir};
use ruststa::engine::analysis::{AnalysisOptions, WireModelOption};
use ruststa::engine::incr::{SessionError, SessionNodeKey, TimingSession};
use ruststa::engine::session_api::EndpointSlack;
use ruststa::engine::types::NodeId;
use thiserror::Error;

use crate::context::Context;
use crate::netlist::{CellId, CellPin, NetId, PortType};

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
    #[error("clock net '{net}' has no driver")]
    ClockNetWithoutDriver { net: String },
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
    pub gate: gate::PinId,
    pub node: NodeId,
}

#[derive(Debug)]
pub struct StaIds {
    pub pins: FxHashMap<CellPin, PinLink>,
    pub node_pins: Vec<Option<CellPin>>,
    pub nets: FxHashMap<NetId, gate::NetId>,
    pub net_arcs: FxHashMap<NetSink, usize>,
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
        let library = library::library(ctx, speed_grade, &library::cell_types(ctx)?)?;
        let GateDesign {
            netlist,
            pins,
            nets,
        } = gate_design(ctx, &library)?;
        let sdc = clock_sdc(ctx)?;
        let options = AnalysisOptions {
            wire_model: WireModelOption::Ideal,
            ..AnalysisOptions::default()
        };
        let session = TimingSession::new(netlist, vec![library], &sdc, options)?;
        let ids = link_ids(ctx, &session, pins, nets);
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
    pins: FxHashMap<CellPin, gate::PinId>,
    nets: FxHashMap<NetId, gate::NetId>,
}

fn gate_design(ctx: &Context, library: &Library) -> Result<GateDesign, StaBuildError> {
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
        nets.insert(net, netlist.add_net(Some(&net_name(net)), None)?);
    }
    let mut pins = FxHashMap::default();
    for (cell_id, cell) in ctx.design.iter_alive_cells() {
        let inst = netlist.add_instance(
            &instance_name(cell_id),
            cell_types[ctx.name_of(cell.cell_type)],
        )?;
        for (&port, data) in &cell.ports {
            let Some(net) = data.net() else { continue };
            let pin = netlist.connect(inst, ctx.name_of(port), Conn::Net(nets[&net]))?;
            pins.insert(CellPin::new(cell_id, port), pin);
        }
    }
    Ok(GateDesign {
        netlist,
        pins,
        nets,
    })
}

fn clock_sdc(ctx: &Context) -> Result<String, StaBuildError> {
    let mut sdc = String::new();
    for (net_id, net) in ctx.design.iter_alive_nets() {
        if net.clock_constraint <= 0 {
            continue;
        }
        let driver = net
            .driver()
            .ok_or_else(|| StaBuildError::ClockNetWithoutDriver {
                net: ctx.name_of(net.name).to_owned(),
            })?;
        writeln!(
            sdc,
            "create_clock -name {} -period {} [get_pins {{{}/{}}}]",
            net_name(net_id),
            net.clock_constraint,
            instance_name(driver.cell),
            glob_escape(ctx.name_of(driver.port)),
        )
        .expect("writing to a String cannot fail");
    }
    Ok(sdc)
}

fn link_ids(
    ctx: &Context,
    session: &TimingSession,
    gate_pins: FxHashMap<CellPin, gate::PinId>,
    nets: FxHashMap<NetId, gate::NetId>,
) -> StaIds {
    let mut node_pins = vec![None; session.graph().num_nodes()];
    let pins: FxHashMap<CellPin, PinLink> = gate_pins
        .into_iter()
        .map(|(cell_pin, gate_pin)| {
            let node = session
                .node_of_key(SessionNodeKey::Pin(gate_pin))
                .expect("every connected pin is a timing graph node");
            node_pins[node.index()] = Some(cell_pin);
            (
                cell_pin,
                PinLink {
                    gate: gate_pin,
                    node,
                },
            )
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

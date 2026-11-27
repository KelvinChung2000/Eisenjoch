//! Arc-level route edits on a live context.
//!
//! An arc is the branch of a net's PIP tree from the source wire to one sink
//! wire. Each edit validates against the net's wire map before it binds or
//! unbinds anything, so a rejected edit leaves the context unchanged.

use std::iter;

use rustc_hash::{FxHashMap, FxHashSet};

use super::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use crate::chipdb::{PipId, WireId};
use crate::common::{DelayT, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellPin, NetId};
use crate::routed_json::{pip_name, wire_name};

/// Why an arc edit was rejected. Names follow upstream nextpnr's wire and pip names.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ArcEditError {
    #[error("net {net} has no sink {sink}")]
    NotASink { net: String, sink: String },
    #[error("net {net} has no driver; connect a driver before routing it")]
    NoDriver { net: String },
    #[error("pin {pin} of net {net} is on an unplaced cell; place the cell first")]
    Unplaced { net: String, pin: String },
    #[error("the BEL under pin {pin} of net {net} has no wire for that port")]
    NoPinWire { net: String, pin: String },
    #[error("arc {net} -> {sink} is already routed; unroute it first")]
    AlreadyRouted { net: String, sink: String },
    #[error("arc {net} -> {sink} is not routed")]
    NotRouted { net: String, sink: String },
    #[error("arc {net} -> {sink} was given no pips")]
    EmptyChain { net: String, sink: String },
    #[error("arc {net} -> {sink} names pip {pip}, which the chipdb does not have")]
    UnknownPip {
        net: String,
        sink: String,
        pip: PipId,
    },
    #[error("arc {net} -> {sink} names wire {wire}, which the chipdb does not have")]
    UnknownWire {
        net: String,
        sink: String,
        wire: WireId,
    },
    #[error("arc {net} -> {sink} starts at pip {pip}, whose source wire {wire} is not in the net's tree")]
    DetachedStart {
        net: String,
        sink: String,
        pip: String,
        wire: String,
    },
    #[error("arc {net} -> {sink} breaks at pip {pip}: it starts on {wire}, but the previous pip ends on {expected}")]
    Discontinuous {
        net: String,
        sink: String,
        pip: String,
        wire: String,
        expected: String,
    },
    #[error("arc {net} -> {sink} ends on {wire}, not on the sink wire {sink_wire}")]
    WrongEnd {
        net: String,
        sink: String,
        wire: String,
        sink_wire: String,
    },
    #[error(
        "arc {net} -> {sink} needs wire {wire}, which net {owner} holds; unroute that net first"
    )]
    WireTaken {
        net: String,
        sink: String,
        wire: String,
        owner: String,
    },
    #[error("arc {net} -> {sink} reaches wire {wire} twice or re-enters the net's tree there")]
    Loop {
        net: String,
        sink: String,
        wire: String,
    },
    #[error("arc {net} -> {sink} has no path over free wires to wire {wire}")]
    NoPath {
        net: String,
        sink: String,
        wire: String,
    },
}

struct ArcEnds {
    net: String,
    sink: String,
    source_wire: WireId,
    sink_wire: WireId,
}

impl ArcEnds {
    fn resolve(ctx: &Context, net: NetId, sink: CellPin) -> Result<Self, ArcEditError> {
        let view = ctx.net(net);
        let net_name = view.name().to_owned();
        let sink_name = pin_name(ctx, sink);
        if !view.users().contains(&sink) {
            return Err(ArcEditError::NotASink {
                net: net_name,
                sink: sink_name,
            });
        }
        let driver = view.driver().ok_or_else(|| ArcEditError::NoDriver {
            net: net_name.clone(),
        })?;
        Ok(Self {
            source_wire: pin_wire(ctx, &net_name, driver)?,
            sink_wire: pin_wire(ctx, &net_name, sink)?,
            net: net_name,
            sink: sink_name,
        })
    }

    fn in_tree(&self, ctx: &Context, net: NetId, wire: WireId) -> bool {
        wire == self.source_wire || ctx.net(net).wires().contains_key(&wire)
    }
}

fn pin_name(ctx: &Context, pin: CellPin) -> String {
    format!("{}.{}", ctx.cell(pin.cell).name(), ctx.name_of(pin.port))
}

fn pin_wire(ctx: &Context, net: &str, pin: CellPin) -> Result<WireId, ArcEditError> {
    let bel = ctx
        .cell(pin.cell)
        .bel()
        .ok_or_else(|| ArcEditError::Unplaced {
            net: net.to_owned(),
            pin: pin_name(ctx, pin),
        })?;
    bel.pin_wire(pin.port)
        .map(|wire| wire.id())
        .ok_or_else(|| ArcEditError::NoPinWire {
            net: net.to_owned(),
            pin: pin_name(ctx, pin),
        })
}

/// Unbind the wires and PIPs of `sink`'s arc that no other part of `net`'s tree needs.
///
/// Walks up from the sink wire and stops at the first wire that still drives
/// another branch or is another sink's wire. Fails if the arc is not routed.
pub fn unroute_arc(ctx: &mut Context, net: NetId, sink: CellPin) -> Result<(), ArcEditError> {
    let arc = ArcEnds::resolve(ctx, net, sink)?;
    let exclusive = exclusive_entries(ctx, net, sink, &arc)?;
    for (wire, pip) in exclusive {
        ctx.unbind_wire(wire);
        if let Some(pip) = pip {
            ctx.unbind_pip(pip);
        }
        ctx.design.net_edit(net).remove_wire(wire);
    }
    Ok(())
}

fn exclusive_entries(
    ctx: &Context,
    net: NetId,
    sink: CellPin,
    arc: &ArcEnds,
) -> Result<Vec<(WireId, Option<PipId>)>, ArcEditError> {
    let chipdb = ctx.chipdb();
    let tree = ctx.net(net).wires();
    if !tree.contains_key(&arc.sink_wire) {
        return Err(ArcEditError::NotRouted {
            net: arc.net.clone(),
            sink: arc.sink.clone(),
        });
    }
    let other_sink_wires: FxHashSet<WireId> = ctx
        .net(net)
        .users()
        .iter()
        .filter(|&&user| user.is_valid() && user != sink)
        .filter_map(|&user| ctx.cell(user.cell).bel()?.pin_wire(user.port))
        .map(|wire| wire.id())
        .collect();
    let mut fanout: FxHashMap<WireId, usize> = FxHashMap::default();
    for pip in tree.values().filter_map(|entry| entry.pip) {
        *fanout.entry(chipdb.pip_src_wire(pip)).or_default() += 1;
    }
    let mut exclusive = Vec::new();
    let mut wire = arc.sink_wire;
    while fanout.get(&wire).is_none_or(|&count| count == 0) && !other_sink_wires.contains(&wire) {
        let pip = tree
            .get(&wire)
            .expect("a net's tree holds every wire its PIPs are driven from")
            .pip;
        exclusive.push((wire, pip));
        let Some(pip) = pip else {
            break;
        };
        wire = chipdb.pip_src_wire(pip);
        *fanout
            .get_mut(&wire)
            .expect("the fanout map counts every PIP source wire") -= 1;
    }
    Ok(exclusive)
}

/// Bind `pips` as the arc from `net`'s tree to `sink`'s wire.
///
/// The chain must start on a tree wire or the source wire, be continuous, end
/// on the sink wire and use only wires that are free and new to the tree.
/// Every check runs before the first bind, so an error changes nothing.
pub fn bind_arc(
    ctx: &mut Context,
    net: NetId,
    sink: CellPin,
    pips: &[PipId],
    strength: PlaceStrength,
) -> Result<(), ArcEditError> {
    let arc = ArcEnds::resolve(ctx, net, sink)?;
    check_chain(ctx, net, &arc, pips)?;
    if !ctx.net(net).wires().contains_key(&arc.source_wire) {
        ctx.bind_wire(arc.source_wire, net, strength);
        ctx.design
            .net_edit(net)
            .add_wire(arc.source_wire, None, strength);
    }
    for &pip in pips {
        let wire = ctx.chipdb().pip_dst_wire(pip);
        ctx.bind_pip(pip, net, strength);
        ctx.bind_wire(wire, net, strength);
        ctx.design.net_edit(net).add_wire(wire, Some(pip), strength);
    }
    Ok(())
}

fn check_chain(
    ctx: &Context,
    net: NetId,
    arc: &ArcEnds,
    pips: &[PipId],
) -> Result<(), ArcEditError> {
    let chipdb = ctx.chipdb();
    let arc_net = || arc.net.clone();
    let arc_sink = || arc.sink.clone();
    if arc.in_tree(ctx, net, arc.sink_wire) {
        return Err(ArcEditError::AlreadyRouted {
            net: arc_net(),
            sink: arc_sink(),
        });
    }
    let Some(&first) = pips.first() else {
        return Err(ArcEditError::EmptyChain {
            net: arc_net(),
            sink: arc_sink(),
        });
    };
    if let Some(&pip) = pips.iter().find(|&&pip| !chipdb.has_pip(pip)) {
        return Err(ArcEditError::UnknownPip {
            net: arc_net(),
            sink: arc_sink(),
            pip,
        });
    }
    let start = chipdb.pip_src_wire(first);
    if !arc.in_tree(ctx, net, start) {
        return Err(ArcEditError::DetachedStart {
            net: arc_net(),
            sink: arc_sink(),
            pip: pip_name(ctx, first),
            wire: wire_name(ctx, start),
        });
    }
    let taken = |wire: WireId| {
        ctx.wire_binding(wire)
            .filter(|&(owner, _)| owner != net)
            .map(|(owner, _)| ArcEditError::WireTaken {
                net: arc_net(),
                sink: arc_sink(),
                wire: wire_name(ctx, wire),
                owner: ctx.net(owner).name().to_owned(),
            })
    };
    if let Some(error) = taken(arc.source_wire) {
        return Err(error);
    }
    let mut reached = FxHashSet::default();
    let mut at = start;
    for &pip in pips {
        let src = chipdb.pip_src_wire(pip);
        if src != at {
            return Err(ArcEditError::Discontinuous {
                net: arc_net(),
                sink: arc_sink(),
                pip: pip_name(ctx, pip),
                wire: wire_name(ctx, src),
                expected: wire_name(ctx, at),
            });
        }
        at = chipdb.pip_dst_wire(pip);
        if arc.in_tree(ctx, net, at) || !reached.insert(at) {
            return Err(ArcEditError::Loop {
                net: arc_net(),
                sink: arc_sink(),
                wire: wire_name(ctx, at),
            });
        }
        if let Some(error) = taken(at) {
            return Err(error);
        }
    }
    if at != arc.sink_wire {
        return Err(ArcEditError::WrongEnd {
            net: arc_net(),
            sink: arc_sink(),
            wire: wire_name(ctx, at),
            sink_wire: wire_name(ctx, arc.sink_wire),
        });
    }
    Ok(())
}

struct FreeWireCost<'a> {
    net: NetId,
    blocked: &'a FxHashSet<WireId>,
}

impl PathCostModel for FreeWireCost<'_> {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }

    fn heuristic(&self, ctx: &Context, wire: WireId, dst: WireId) -> DelayT {
        ctx.estimate_delay(wire, dst)
    }

    fn is_blocked(&self, ctx: &Context, wire: WireId) -> bool {
        self.blocked.contains(&wire)
            || ctx
                .wire_binding(wire)
                .is_some_and(|(owner, _)| owner != self.net)
    }
}

/// Route `sink`'s arc from `net`'s tree through `waypoints` in order, bind it
/// through [`bind_arc`] and return its PIPs.
///
/// A waypoint may be any tile wire of its node. Each [`astar_search`] leg uses only
/// free wires new to the tree; [`ArcEditError::NoPath`] names the wire it missed.
pub fn route_arc_via(
    ctx: &mut Context,
    net: NetId,
    sink: CellPin,
    waypoints: &[WireId],
    strength: PlaceStrength,
) -> Result<Vec<PipId>, ArcEditError> {
    let pips = search_arc(ctx, net, sink, waypoints)?;
    bind_arc(ctx, net, sink, &pips, strength)?;
    Ok(pips)
}

fn search_arc(
    ctx: &Context,
    net: NetId,
    sink: CellPin,
    waypoints: &[WireId],
) -> Result<Vec<PipId>, ArcEditError> {
    let arc = ArcEnds::resolve(ctx, net, sink)?;
    if arc.in_tree(ctx, net, arc.sink_wire) {
        return Err(ArcEditError::AlreadyRouted {
            net: arc.net,
            sink: arc.sink,
        });
    }
    if let Some(&wire) = waypoints.iter().find(|&&wire| !ctx.chipdb().has_wire(wire)) {
        return Err(ArcEditError::UnknownWire {
            net: arc.net,
            sink: arc.sink,
            wire,
        });
    }
    let mut blocked: FxHashSet<WireId> = ctx
        .net(net)
        .wire_ids()
        .chain(iter::once(arc.source_wire))
        .collect();
    let mut starts = blocked.clone();
    let mut pips = Vec::new();
    let opts = AStarOptions {
        retain_trace: false,
        ..AStarOptions::default()
    };
    let targets = waypoints
        .iter()
        .map(|&wire| ctx.chipdb().canonical_wire(wire))
        .chain(iter::once(arc.sink_wire));
    for target in targets {
        let model = FreeWireCost {
            net,
            blocked: &blocked,
        };
        let leg = astar_search(ctx, &model, &starts, target, &opts)
            .path
            .ok_or_else(|| ArcEditError::NoPath {
                net: arc.net.clone(),
                sink: arc.sink.clone(),
                wire: wire_name(ctx, target),
            })?;
        blocked.extend(leg.iter().map(|&pip| ctx.chipdb().pip_dst_wire(pip)));
        pips.extend(leg);
        starts = FxHashSet::from_iter([target]);
    }
    Ok(pips)
}

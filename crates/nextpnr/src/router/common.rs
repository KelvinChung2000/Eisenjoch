//! Shared helper functions used by both Router1 and Router2.

use crate::chipdb::{PipId, WireId};
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::NetId;
use rustc_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Route plan types (pure computation output, no Context mutation)
// ---------------------------------------------------------------------------

/// Computed route for a single net, before binding.
pub struct RoutePlan {
    pub net: NetId,
    pub source_wire: WireId,
    pub sink_routes: Vec<SinkRoute>,
}

/// Computed path from the routing tree to a single sink wire.
pub struct SinkRoute {
    pub sink_wire: WireId,
    pub pips: Vec<PipId>,
}

/// Read-only source wire resolution (no binding).
///
/// Returns the driver wire for a net, or `Ok(None)` if it has no connected driver.
pub fn resolve_source_wire(
    ctx: &Context,
    net_idx: NetId,
) -> Result<Option<WireId>, super::RouterError> {
    let net = ctx.net(net_idx);
    let net_name = net.name_id();

    let Some(driver_pin) = net.driver_cell_port() else {
        return Ok(None);
    };

    let driver_cell = ctx.cell(driver_pin.cell);
    let driver_bel = match driver_cell.bel() {
        Some(bel) => bel,
        None => {
            return Err(super::RouterError::Generic(format!(
                "Driver cell for net {} is not placed",
                ctx.name_of(net_name)
            )));
        }
    };

    let src_wire = driver_bel
        .pin_wire(driver_pin.port)
        .map(|w| w.id())
        .ok_or_else(|| {
            super::RouterError::Generic(format!(
                "Cannot find driver wire for net {}",
                ctx.name_of(net_name)
            ))
        })?;

    Ok(Some(src_wire))
}

/// Apply a computed RoutePlan by binding source wire, PIPs, and dest wires.
///
/// Binds atomically: if any wire (including any of its node-equivalents) is
/// already claimed by a different net, all bindings applied by this call are
/// rolled back and `Err(conflicting_wire)` is returned. No partial state is
/// left behind in the context or in the net's wire map.
pub fn apply_route_plan(ctx: &mut Context, plan: &RoutePlan) -> Result<(), WireId> {
    // Track wires we've bound so we can unwind on conflict. Source wire is
    // first; PIP destinations follow in the order they were bound. PIPs get
    // their own tracking vec.
    let mut bound_wires: Vec<WireId> = Vec::new();
    let mut bound_pips: Vec<PipId> = Vec::new();
    let mut net_wire_entries: Vec<WireId> = Vec::new();

    // Helper closure: unwind on conflict.
    // We cannot easily capture &mut ctx in a closure here, so inline the
    // rollback at each failure site.

    // Bind source wire via node-aware atomic bind.
    match ctx.try_bind_wire_node(plan.source_wire, plan.net, PlaceStrength::Strong) {
        Ok(()) => {
            bound_wires.push(plan.source_wire);
            ctx.design
                .net_edit(plan.net)
                .add_wire(plan.source_wire, None, PlaceStrength::Strong);
            net_wire_entries.push(plan.source_wire);
        }
        Err(conflict) => {
            return Err(conflict);
        }
    }

    // Bind each sink route's PIPs and destination wires.
    for sink in &plan.sink_routes {
        for &pip in &sink.pips {
            let dst_wire = ctx.pip(pip).dst_wire().id();

            // Try to bind the destination wire (and its node-equivalents).
            if let Err(conflict) = ctx.try_bind_wire_node(dst_wire, plan.net, PlaceStrength::Strong)
            {
                // Unwind: release PIPs, wires, and the net's wire map entries.
                for &p in &bound_pips {
                    ctx.unbind_pip(p);
                }
                // Re-read node equivalents for each previously-bound wire and
                // unbind every one — mirrors what try_bind_wire_node did.
                for &w in &bound_wires {
                    ctx.unbind_wire(w);
                    let mut equivs: Vec<WireId> = Vec::new();
                    ctx.chipdb().node_wires_cb(w, |nw| equivs.push(nw));
                    for nw in equivs {
                        ctx.unbind_wire(nw);
                    }
                }
                for w in net_wire_entries.iter().rev() {
                    ctx.design.net_edit(plan.net).remove_wire(*w);
                }
                return Err(conflict);
            }

            // Bind the PIP itself.
            ctx.bind_pip(pip, plan.net, PlaceStrength::Strong);
            bound_pips.push(pip);
            bound_wires.push(dst_wire);

            ctx.design
                .net_edit(plan.net)
                .add_wire(dst_wire, Some(pip), PlaceStrength::Strong);
            net_wire_entries.push(dst_wire);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Net collection helpers
// ---------------------------------------------------------------------------

/// Collect all net indices that need routing.
///
/// A net needs routing if it has a connected driver and at least one user.
pub fn collect_routable_nets(ctx: &Context) -> Vec<NetId> {
    ctx.design
        .iter_net_indices()
        .filter(|&net_idx| {
            let net = ctx.net(net_idx);
            net.is_alive() && net.has_driver() && net.num_users() > 0
        })
        .collect()
}

/// Bind a sequence of PIPs as the route for a net.
///
/// For each PIP in the path, binds the PIP and its destination wire to the
/// given net, and records the routing in the net's wire map.
pub fn bind_route(ctx: &mut Context, net_idx: NetId, path: &[PipId]) {
    for &pip in path {
        let dst_wire = ctx.pip(pip).dst_wire().id();
        ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
        ctx.bind_wire(dst_wire, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(dst_wire, Some(pip), PlaceStrength::Strong);
    }
}

/// Rip up (unroute) a net by unbinding all its wires and PIPs.
pub fn unroute_net(ctx: &mut Context, net_idx: NetId) {
    let net = ctx.net(net_idx);
    let entries: Vec<(WireId, Option<PipId>)> = net
        .wires()
        .iter()
        .map(|(&wire, pm)| (wire, pm.pip))
        .collect();

    for (wire, pip) in entries {
        ctx.unbind_wire(wire);
        if let Some(pip) = pip {
            ctx.unbind_pip(pip);
        }
    }

    ctx.design.net_edit(net_idx).clear_wires();
}

/// Collect the sink wires for all users of a net.
///
/// Resolves each user's BEL pin to a wire via the view API.
/// Skips unconnected or unplaced users.
pub fn collect_sink_wires(ctx: &Context, net_idx: NetId) -> Vec<WireId> {
    let net = ctx.net(net_idx);
    let mut sink_wires = Vec::with_capacity(net.num_users());
    for user in net.users() {
        if !user.is_valid() {
            continue;
        }
        let user_cell_idx = user.cell;
        let user_cell = ctx.cell(user_cell_idx);
        let user_bel = match user_cell.bel() {
            Some(bel) => bel,
            None => continue,
        };
        if let Some(sink_wire) = user_bel.pin_wire(user.port) {
            sink_wires.push(sink_wire.id());
        }
    }
    sink_wires
}

/// Collect all wires across the chip that have a given nonzero `const_value`.
///
/// These serve as additional routing sources for constant nets (GND/VCC),
/// since each tile has its own constant wire connected to the local switch matrix.
pub fn collect_constant_source_wires(ctx: &Context, const_value: i32) -> Vec<WireId> {
    ctx.chipdb()
        .wires()
        .filter(|&wire| ctx.chipdb().wire_info(wire).const_value == const_value)
        .collect()
}

/// Get the `const_value` of a net's driver wire, or 0 if not a constant net.
///
/// A nonzero return means the net is driven by a constant wire (GND/VCC).
pub fn source_wire_const_value(ctx: &Context, source_wire: WireId) -> i32 {
    ctx.chipdb().wire_info(source_wire).const_value
}

/// Find the single uphill PIP from `sink` whose source wire has the matching
/// `const_value` (typically the tile's tied GND/VCC in the local switch matrix).
///
/// Real FPGA switch matrices expose GND/VCC as locally-tied wires, so every
/// const-net sink can be reached via exactly one PIP from a neighbour wire.
/// Callers use this to bypass multi-source A* for constant nets entirely:
/// the path is deterministic, O(|pips_uphill|), and always succeeds when the
/// chipdb encodes the switch-matrix const wires (verified on xc7_large:
/// 320/320 const sinks reach a matching const wire in exactly one uphill hop).
///
/// Returns `None` only when the sink's switch matrix genuinely has no tied
/// wire — callers should fall back to generic routing in that case.
pub fn find_local_const_pip(ctx: &Context, sink: WireId, const_value: i32) -> Option<PipId> {
    let chipdb = ctx.chipdb();
    for &pip_idx in chipdb.wire_info(sink).pips_uphill.get() {
        let pip = PipId::new(sink.tile(), pip_idx);
        if chipdb.wire_info(chipdb.pip_src_wire(pip)).const_value == const_value {
            return Some(pip);
        }
    }
    let mut result: Option<PipId> = None;
    chipdb.node_wires_cb(sink, |peer| {
        if result.is_some() {
            return;
        }
        for &pip_idx in chipdb.wire_info(peer).pips_uphill.get() {
            let pip = PipId::new(peer.tile(), pip_idx);
            if chipdb.wire_info(chipdb.pip_src_wire(pip)).const_value == const_value {
                result = Some(pip);
                return;
            }
        }
    });
    result
}

/// Return true for global-clock wires in XC7-style chipdbs.
pub fn is_global_clock_wire(ctx: &Context, wire: WireId) -> bool {
    let wire_type = ctx.chipdb().wire_type(wire);
    matches!(
        wire_type,
        "GCLK_SPINE"
            | "CLK_IN"
            | "GCLK"
            | "GLOBAL"
            | "BUFGROUT"
            | "NODE_GLOBAL_BUFG"
            | "NODE_GLOBAL_VROUTE"
            | "NODE_GLOBAL_HROUTE"
            | "NODE_GLOBAL_HDISTR"
            | "NODE_GLOBAL_VDISTR"
            | "NODE_GLOBAL_LEAF"
    ) || wire_type.contains("GCLK")
        || wire_type.contains("HCLK")
        || wire_type.contains("BUFG")
}

/// Return true when a PIP touches a global-clock wire.
pub fn is_global_clock_pip(ctx: &Context, pip: PipId) -> bool {
    let chipdb = ctx.chipdb();
    is_global_clock_wire(ctx, chipdb.pip_src_wire(pip))
        || is_global_clock_wire(ctx, chipdb.pip_dst_wire(pip))
}

/// Validate the routed design against the cell netlist.
///
/// Parallels `placer::common::validate_all_placed`: walks every alive net that
/// has a connected driver and at least one user, and verifies that routing
/// actually connects each sink back to the source. Collects every failure it
/// finds (up to a small number of examples per category) and returns them as a
/// single [`RouterError::ValidationFailed`] so the caller sees a complete
/// diagnostic rather than whichever problem happened to be hit first.
///
/// Checks performed:
/// 1. Driver and user cells are placed.
/// 2. The driver BEL pin resolves to a wire, and that wire is present in the
///    net's routing tree.
/// 3. Every valid user's BEL pin wire is present in the routing tree.
/// 4. Each sink wire traces back to the source through parent PIPs
///    (no broken chains, no cycles).
/// 5. Every wire listed in `net.wires` is bound to this net in the context's
///    wire table, and likewise for every PIP.
/// 6. No wire or PIP is claimed by more than one net.
pub fn validate_all_routed(ctx: &Context) -> Result<(), super::RouterError> {
    const MAX_EXAMPLES: usize = 10;

    struct Bucket {
        total: usize,
        examples: Vec<String>,
    }

    impl Bucket {
        fn new() -> Self {
            Self { total: 0, examples: Vec::new() }
        }

        fn add(&mut self, msg: impl FnOnce() -> String) {
            self.total += 1;
            if self.examples.len() < MAX_EXAMPLES {
                self.examples.push(msg());
            }
        }

        fn append(&self, out: &mut String, label: &str) {
            if self.total == 0 {
                return;
            }
            out.push_str(&format!("  {} ({} total", label, self.total));
            if self.total > self.examples.len() {
                out.push_str(&format!(", showing first {})\n", self.examples.len()));
            } else {
                out.push_str(")\n");
            }
            for ex in &self.examples {
                out.push_str("    - ");
                out.push_str(ex);
                out.push('\n');
            }
        }
    }

    let mut source_not_in_tree = Bucket::new();
    let mut unplaced_driver = Bucket::new();
    let mut unplaced_user = Bucket::new();
    let mut missing_source_wire = Bucket::new();
    let mut missing_sink_wire = Bucket::new();
    let mut sink_not_in_tree = Bucket::new();
    let mut disconnected_sink = Bucket::new();
    let mut wire_conflict = Bucket::new();
    let mut pip_conflict = Bucket::new();
    let mut wire_binding_mismatch = Bucket::new();
    let mut pip_binding_mismatch = Bucket::new();

    let mut wire_claims: FxHashMap<WireId, Vec<NetId>> = FxHashMap::default();
    let mut pip_claims: FxHashMap<PipId, Vec<NetId>> = FxHashMap::default();

    let chipdb = ctx.chipdb();

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
            continue;
        }

        let net_name = ctx.name_of(net.name_id()).to_owned();

        // Record claims from the wire map and check binding consistency.
        for (&wire, pm) in net.wires() {
            wire_claims.entry(wire).or_default().push(net_idx);
            match ctx.wire_binding(wire) {
                Some((owner, _)) if owner == net_idx => {}
                Some((owner, _)) => {
                    let other = ctx.name_of(ctx.net(owner).name_id()).to_owned();
                    let wn = chipdb.wire_name(wire).to_owned();
                    wire_binding_mismatch.add(|| format!(
                        "net '{}' lists wire {} ({}) but wire is bound to net '{}'",
                        net_name, wn, wire, other,
                    ));
                }
                None => {
                    let wn = chipdb.wire_name(wire).to_owned();
                    wire_binding_mismatch.add(|| format!(
                        "net '{}' lists wire {} ({}) but wire is unbound",
                        net_name, wn, wire,
                    ));
                }
            }
            if let Some(pip) = pm.pip {
                pip_claims.entry(pip).or_default().push(net_idx);
                match ctx.pip_binding(pip) {
                    Some((owner, _)) if owner == net_idx => {}
                    Some((owner, _)) => {
                        let other = ctx.name_of(ctx.net(owner).name_id()).to_owned();
                        pip_binding_mismatch.add(|| format!(
                            "net '{}' lists pip {} but pip is bound to net '{}'",
                            net_name, pip, other,
                        ));
                    }
                    None => {
                        pip_binding_mismatch.add(|| format!(
                            "net '{}' lists pip {} but pip is unbound",
                            net_name, pip,
                        ));
                    }
                }
            }
        }

        // Resolve driver.
        let driver = net
            .driver_cell_port()
            .expect("has_driver already checked above");
        let driver_cell = ctx.cell(driver.cell);
        let driver_bel = match driver_cell.bel() {
            Some(b) => b,
            None => {
                let cell_name = driver_cell.name().to_owned();
                let port_name = ctx.name_of(driver.port).to_owned();
                unplaced_driver.add(|| format!(
                    "net '{}': driver cell '{}' pin '{}' is not placed",
                    net_name, cell_name, port_name,
                ));
                continue;
            }
        };
        let source_wire = match driver_bel.pin_wire(driver.port) {
            Some(w) => w.id(),
            None => {
                let bel_name = driver_bel.name().to_owned();
                let port_name = ctx.name_of(driver.port).to_owned();
                missing_source_wire.add(|| format!(
                    "net '{}': driver BEL '{}' has no wire for pin '{}'",
                    net_name, bel_name, port_name,
                ));
                continue;
            }
        };
        // Same node-peer tolerance as the sink check below: the router can
        // bind a node-peer of the source instead of the source wire itself.
        let source_in_tree = net.wires().contains_key(&source_wire) || {
            let mut hit = false;
            chipdb.node_wires_cb(source_wire, |peer| {
                if !hit && net.wires().contains_key(&peer) {
                    hit = true;
                }
            });
            hit
        };
        if !source_in_tree {
            let sn = chipdb.wire_name(source_wire).to_owned();
            let wires_len = net.wires().len();
            source_not_in_tree.add(|| format!(
                "net '{}': source wire {} ({}) not present in routing tree (tree has {} wires)",
                net_name, sn, source_wire, wires_len,
            ));
        }

        // Check every user.
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let user_cell = ctx.cell(user.cell);
            let user_bel = match user_cell.bel() {
                Some(b) => b,
                None => {
                    let cell_name = user_cell.name().to_owned();
                    let port_name = ctx.name_of(user.port).to_owned();
                    unplaced_user.add(|| format!(
                        "net '{}': user cell '{}' pin '{}' is not placed",
                        net_name, cell_name, port_name,
                    ));
                    continue;
                }
            };
            let sink_wire = match user_bel.pin_wire(user.port) {
                Some(w) => w.id(),
                None => {
                    let bel_name = user_bel.name().to_owned();
                    let port_name = ctx.name_of(user.port).to_owned();
                    missing_sink_wire.add(|| format!(
                        "net '{}': user BEL '{}' has no wire for pin '{}'",
                        net_name, bel_name, port_name,
                    ));
                    continue;
                }
            };
            // A sink reached via a free node hop (one electrical wire
            // spanning multiple tiles) ends its route at a PIP whose dst
            // wire is a NODE PEER of the sink rather than the sink wire
            // itself; `apply_route_plan` binds the PIP dst and only
            // implicitly claims the sink via `try_bind_wire_node` (which
            // locks all peers at the ctx level but does not expand the
            // net's wire map). Treat the sink as routed whenever it or
            // any of its node peers landed in the net's tree.
            let sink_in_tree = net.wires().contains_key(&sink_wire) || {
                let mut hit = false;
                chipdb.node_wires_cb(sink_wire, |peer| {
                    if !hit && net.wires().contains_key(&peer) {
                        hit = true;
                    }
                });
                hit
            };
            if !sink_in_tree {
                let sn = chipdb.wire_name(sink_wire).to_owned();
                let cell_name = user_cell.name().to_owned();
                let port_name = ctx.name_of(user.port).to_owned();
                sink_not_in_tree.add(|| format!(
                    "net '{}': sink wire {} ({}) for cell '{}' pin '{}' not in routing tree",
                    net_name, sn, sink_wire, cell_name, port_name,
                ));
                continue;
            }

            // Walk parent PIPs back to the source. For constant nets (GND/VCC)
            // the chain may terminate at any wire with matching `const_value`
            // rather than at the driver's wire: each tile's switch matrix has
            // its own tied const wire, so different sinks legitimately anchor
            // on different local const wires. See `find_local_const_pip`.
            //
            // A* expansion treats all node peers as one electrical wire and may
            // record a PIP whose src_wire is a peer of a tree-bound wire rather
            // than the tree-bound wire itself. When `cur` isn't in the tree,
            // consult `node_wires_cb` for a peer that is — a free node hop is
            // not a break in the chain.
            let source_const_value = chipdb.wire_info(source_wire).const_value;
            let source_node_id = chipdb.node_id(source_wire);
            let mut cur = sink_wire;
            let mut seen: FxHashSet<WireId> = FxHashSet::default();
            let mut reached_source = false;
            let mut cause = "reached a wire with no parent pip";
            loop {
                if cur == source_wire {
                    reached_source = true;
                    break;
                }
                if source_node_id.is_some() && chipdb.node_id(cur) == source_node_id {
                    reached_source = true;
                    break;
                }
                if source_const_value != 0
                    && chipdb.wire_info(cur).const_value == source_const_value
                    && net.wires().contains_key(&cur)
                {
                    reached_source = true;
                    break;
                }
                if !seen.insert(cur) {
                    cause = "cycle detected in parent-pip chain";
                    break;
                }
                let pm = match net.wires().get(&cur) {
                    Some(pm) => pm,
                    None => {
                        // Free node hop: find a peer of `cur` that is in the
                        // tree and continue from there. Peers are electrically
                        // the same wire; binding any one binds the whole node.
                        let mut peer_in_tree: Option<WireId> = None;
                        chipdb.node_wires_cb(cur, |peer| {
                            if peer_in_tree.is_none() && net.wires().contains_key(&peer) {
                                peer_in_tree = Some(peer);
                            }
                        });
                        match peer_in_tree {
                            Some(peer) => {
                                cur = peer;
                                continue;
                            }
                            None => {
                                cause = "parent chain left the routing tree";
                                break;
                            }
                        }
                    }
                };
                let Some(parent_pip) = pm.pip else {
                    cause = "reached a wire with no parent pip before hitting the source";
                    break;
                };
                cur = chipdb.pip_src_wire(parent_pip);
            }
            if !reached_source {
                let sn = chipdb.wire_name(sink_wire).to_owned();
                let cell_name = user_cell.name().to_owned();
                let port_name = ctx.name_of(user.port).to_owned();
                let cause = cause.to_owned();
                disconnected_sink.add(|| format!(
                    "net '{}': sink wire {} ({}) for cell '{}' pin '{}' not connected to source ({})",
                    net_name, sn, sink_wire, cell_name, port_name, cause,
                ));
            }
        }
    }

    // Cross-net conflicts from the accumulated claim maps.
    for (&wire, owners) in &wire_claims {
        if owners.len() <= 1 {
            continue;
        }
        let wn = chipdb.wire_name(wire).to_owned();
        let sample: Vec<String> = owners
            .iter()
            .take(3)
            .map(|&n| ctx.name_of(ctx.net(n).name_id()).to_owned())
            .collect();
        let more = if owners.len() > sample.len() { ", ..." } else { "" };
        wire_conflict.add(|| format!(
            "wire {} ({}) claimed by {} nets: {}{}",
            wn, wire, owners.len(), sample.join(", "), more,
        ));
    }
    for (&pip, owners) in &pip_claims {
        if owners.len() <= 1 {
            continue;
        }
        let sample: Vec<String> = owners
            .iter()
            .take(3)
            .map(|&n| ctx.name_of(ctx.net(n).name_id()).to_owned())
            .collect();
        let more = if owners.len() > sample.len() { ", ..." } else { "" };
        pip_conflict.add(|| format!(
            "pip {} claimed by {} nets: {}{}",
            pip, owners.len(), sample.join(", "), more,
        ));
    }

    let total = source_not_in_tree.total
        + unplaced_driver.total
        + unplaced_user.total
        + missing_source_wire.total
        + missing_sink_wire.total
        + sink_not_in_tree.total
        + disconnected_sink.total
        + wire_conflict.total
        + pip_conflict.total
        + wire_binding_mismatch.total
        + pip_binding_mismatch.total;
    if total == 0 {
        return Ok(());
    }

    let mut out = format!("Routing validation failed ({} issues):\n", total);
    unplaced_driver.append(&mut out, "nets with unplaced driver cell");
    unplaced_user.append(&mut out, "nets with unplaced user cell");
    missing_source_wire.append(&mut out, "driver BEL pins with no wire");
    missing_sink_wire.append(&mut out, "user BEL pins with no wire");
    source_not_in_tree.append(&mut out, "nets whose source wire is not in the routing tree");
    sink_not_in_tree.append(&mut out, "sink wires missing from the routing tree");
    disconnected_sink.append(&mut out, "sinks not connected back to the source");
    wire_conflict.append(&mut out, "wires claimed by more than one net");
    pip_conflict.append(&mut out, "pips claimed by more than one net");
    wire_binding_mismatch
        .append(&mut out, "wires listed in a net but not bound to it in the context");
    pip_binding_mismatch
        .append(&mut out, "pips listed in a net but not bound to it in the context");

    Err(super::RouterError::ValidationFailed(out))
}

/// Diagnose why Router1 fails on a specific net.
///
/// For every (driver, sink) pair: runs A* on an empty context with
/// `wire_penalty` empty and no bounding box, and prints:
///
/// 1. Driver / sink wire names, tile coordinates, BEL pin context
/// 2. `pips_downhill` count on the driver wire and `pips_uphill` on the sink
///    (sanity check that the chipdb actually gives each endpoint some edges)
/// 3. A* outcome: path length, visit count, exit reason
/// 4. When no path is found: exploration bbox, reachable-wire count, and the
///    closest wire to the sink that A* did visit (by estimate_delay)
///
/// Output goes to stderr via `eprintln!`. Returns the number of sinks that
/// failed so callers can decide whether to continue.
pub fn diagnose_unroutable_net(
    ctx: &Context,
    net_idx: NetId,
    visit_budget: Option<usize>,
) -> usize {
    use super::maze::{astar_route_with_trace, AStarExit};

    let net = ctx.net(net_idx);
    let net_name = ctx.name_of(net.name_id()).to_owned();
    eprintln!("=== diagnose_unroutable_net('{}') ===", net_name);

    let Some(driver_pin) = net.driver_cell_port() else {
        eprintln!("  no driver; aborting");
        return 0;
    };
    let driver_cell = ctx.cell(driver_pin.cell);
    let Some(driver_bel) = driver_cell.bel() else {
        eprintln!("  driver cell '{}' unplaced; aborting", driver_cell.name());
        return 0;
    };
    let Some(src_wire_v) = driver_bel.pin_wire(driver_pin.port) else {
        eprintln!(
            "  driver BEL '{}' has no wire for pin '{}'; aborting",
            driver_bel.name(),
            ctx.name_of(driver_pin.port)
        );
        return 0;
    };
    let src_wire = src_wire_v.id();
    let chipdb = ctx.chipdb();
    let (sx, sy) = chipdb.tile_xy(src_wire.tile());
    let src_downhill = chipdb.wire_info(src_wire).pips_downhill.get().len();
    eprintln!(
        "  driver: cell='{}' port='{}' bel='{}' wire='{}' ({}) at ({},{}); pips_downhill={}",
        driver_cell.name(),
        ctx.name_of(driver_pin.port),
        driver_bel.name(),
        chipdb.wire_name(src_wire),
        src_wire,
        sx,
        sy,
        src_downhill,
    );

    let mut src_set = rustc_hash::FxHashSet::default();
    src_set.insert(src_wire);
    let empty_penalty: FxHashMap<WireId, crate::timing::DelayT> = FxHashMap::default();

    let mut failed = 0usize;
    for (ui, user) in net.users().iter().enumerate() {
        if !user.is_valid() {
            continue;
        }
        let user_cell = ctx.cell(user.cell);
        let Some(user_bel) = user_cell.bel() else {
            eprintln!("  user[{}]: cell '{}' unplaced; skipping", ui, user_cell.name());
            failed += 1;
            continue;
        };
        let Some(dst_wire_v) = user_bel.pin_wire(user.port) else {
            eprintln!(
                "  user[{}]: BEL '{}' pin '{}' has no wire; skipping",
                ui,
                user_bel.name(),
                ctx.name_of(user.port)
            );
            failed += 1;
            continue;
        };
        let dst_wire = dst_wire_v.id();
        let (dx, dy) = chipdb.tile_xy(dst_wire.tile());
        let manhattan = (dx - sx).abs() + (dy - sy).abs();
        let dst_uphill = chipdb.wire_info(dst_wire).pips_uphill.get().len();
        eprintln!(
            "  user[{}]: cell='{}' port='{}' bel='{}' wire='{}' ({}) at ({},{}); manhattan={}; pips_uphill={}",
            ui,
            user_cell.name(),
            ctx.name_of(user.port),
            user_bel.name(),
            chipdb.wire_name(dst_wire),
            dst_wire,
            dx,
            dy,
            manhattan,
            dst_uphill,
        );

        let (path, trace) = astar_route_with_trace(
            ctx,
            net_idx,
            &src_set,
            dst_wire,
            &empty_penalty,
            None,
            50,
            None,
            visit_budget,
            false,
        );
        let outcome = match (path.as_ref(), trace.exit) {
            (Some(p), AStarExit::Reached) => format!(
                "REACHED ({} pips, cost={}, visits={}/{})",
                p.len(),
                trace.best_score,
                trace.visit_count,
                trace.max_visits
            ),
            (None, AStarExit::HeapDrained) => format!(
                "UNREACHABLE (heap drained, visited {} wires after {} visits)",
                trace.visited.len(),
                trace.visit_count
            ),
            (None, AStarExit::VisitLimit) => format!(
                "VISIT_LIMIT (budget {} hit, visited {} wires)",
                trace.max_visits,
                trace.visited.len()
            ),
            (_, e) => format!("UNEXPECTED exit={:?}", e),
        };
        eprintln!("    A*: {}", outcome);

        if path.is_none() {
            failed += 1;
            // Exploration bbox + closest-reached-wire diagnostics.
            let mut min_x = i32::MAX;
            let mut max_x = i32::MIN;
            let mut min_y = i32::MAX;
            let mut max_y = i32::MIN;
            let mut closest_dist = i32::MAX;
            let mut closest_wire: Option<WireId> = None;
            for &w in trace.visited.keys() {
                let (wx, wy) = chipdb.tile_xy(w.tile());
                min_x = min_x.min(wx);
                max_x = max_x.max(wx);
                min_y = min_y.min(wy);
                max_y = max_y.max(wy);
                let d = (wx - dx).abs() + (wy - dy).abs();
                if d < closest_dist {
                    closest_dist = d;
                    closest_wire = Some(w);
                }
            }
            if trace.visited.is_empty() {
                eprintln!("    explored: 0 wires (seed already visited set only)");
            } else {
                eprintln!(
                    "    explored bbox: x=[{}..{}] y=[{}..{}] ({} wires)",
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    trace.visited.len()
                );
                if let Some(cw) = closest_wire {
                    let (cx, cy) = chipdb.tile_xy(cw.tile());
                    eprintln!(
                        "    closest reached: '{}' ({}) at ({},{}), manhattan={} from dst",
                        chipdb.wire_name(cw),
                        cw,
                        cx,
                        cy,
                        closest_dist,
                    );
                }
            }
        }
    }
    failed
}

/// Find all wires that are used by more than one net (congested).
pub fn find_congested_wires(ctx: &Context) -> Vec<WireId> {
    let mut wire_usage: FxHashMap<WireId, u32> = FxHashMap::default();

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        for &wire in net.wires().keys() {
            *wire_usage.entry(wire).or_default() += 1;
        }
    }

    wire_usage
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(wire, _)| wire)
        .collect()
}

// ---------------------------------------------------------------------------
// Shared negotiation cost model (PathFinder-style)
// ---------------------------------------------------------------------------

/// Cost-model parameters for PathFinder-style negotiation routing.
///
/// These fields control the negotiation cost function that encourages nets
/// to find alternative paths when wires are congested.
#[derive(Clone)]
pub struct NegotiationCfg {
    /// Base cost added to every wire traversal.
    pub base_cost: f64,
    /// Multiplier applied to present-congestion penalty.
    pub present_cost_multiplier: f64,
    /// Multiplier applied to historical congestion penalty.
    pub history_cost_multiplier: f64,
    /// Initial value of the present-congestion cost factor.
    pub initial_present_cost: f64,
    /// Growth factor applied to the present-congestion cost each iteration.
    pub present_cost_growth: f64,
    /// A net that has been ripped up at least this many times becomes
    /// "sticky": the pass-level rip-up loop preserves its routing tree and
    /// instead forces the conflict partners to find paths around it. This
    /// breaks the rip-up thrash where the same net keeps getting torn down
    /// and rebuilt across passes.
    pub sticky_ripup_threshold: u32,
}

impl Default for NegotiationCfg {
    fn default() -> Self {
        Self {
            base_cost: 1.0,
            present_cost_multiplier: 2.0,
            history_cost_multiplier: 1.0,
            initial_present_cost: 1.0,
            present_cost_growth: 1.5,
            sticky_ripup_threshold: 1,
        }
    }
}

/// Mutable state for the PathFinder-style negotiation cost model.
///
/// Tracks per-wire usage, ownership, history costs, and present-congestion
/// costs. This is generic and can be used by any router that needs
/// negotiation-based congestion resolution.
pub struct NegotiationState {
    /// Cost-model configuration.
    pub cfg: NegotiationCfg,
    /// Current present-congestion cost factor (grows each iteration).
    pub present_cost: f64,
    /// Per-wire historical congestion cost, accumulated over iterations.
    pub wire_history: FxHashMap<WireId, f64>,
    /// Per-wire current usage count (how many nets use each wire).
    pub wire_usage: FxHashMap<WireId, u32>,
    /// Per-wire owner: last net that claimed the wire. When exactly one net
    /// uses a wire, this identifies the owner (no present-cost penalty for
    /// the owner).
    pub wire_owner: FxHashMap<WireId, NetId>,
    /// External congestion hints (e.g. from placer density map).
    /// Wire -> additional cost added to wire_cost().
    pub external_hints: FxHashMap<WireId, f64>,
    /// Per-net rip-up count: how many times a net has been torn down across
    /// passes. Used to detect rip-up thrash and switch to sticky behavior.
    pub rip_up_count: FxHashMap<NetId, u32>,
    /// Nets that have ever had a partial tree at gate-entry time. Once a
    /// net hits this set, the gate refuses to rip it up again — its partial
    /// is its best bet, and rebuilding from scratch only repeats whatever
    /// failure produced the partial in the first place.
    pub ever_seen_partial: FxHashSet<NetId>,
}

impl NegotiationState {
    /// Create a new negotiation state from the given configuration.
    pub fn new(cfg: NegotiationCfg) -> Self {
        let present_cost = cfg.initial_present_cost;
        Self {
            cfg,
            present_cost,
            wire_history: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
            wire_owner: FxHashMap::default(),
            external_hints: FxHashMap::default(),
            rip_up_count: FxHashMap::default(),
            ever_seen_partial: FxHashSet::default(),
        }
    }

    /// Record that a net has been ripped up.
    pub fn bump_rip_up(&mut self, net: NetId) {
        *self.rip_up_count.entry(net).or_default() += 1;
    }

    /// Returns true once a net has been ripped up enough times to be
    /// preserved instead of torn down again.
    pub fn is_sticky(&self, net: NetId) -> bool {
        self.rip_up_count.get(&net).copied().unwrap_or(0) >= self.cfg.sticky_ripup_threshold
    }

    /// Compute the negotiation-based cost of using a wire for a given net.
    ///
    /// The cost has four components:
    /// 1. Base cost (constant per wire).
    /// 2. Present-congestion penalty: proportional to the number of other nets
    ///    currently using the wire, scaled by the present cost factor.
    /// 3. Historical penalty: accumulated from prior iterations where the wire
    ///    was congested.
    /// 4. External hint cost: additional cost injected by external systems
    ///    (e.g. placer density maps).
    pub fn wire_cost(&self, wire: WireId, net_idx: NetId) -> f64 {
        let usage = self.wire_usage.get(&wire).copied().unwrap_or(0);
        let is_own = self.wire_owner.get(&wire) == Some(&net_idx);
        let present_penalty = if is_own { 0.0 } else { usage as f64 };
        let history = self.wire_history.get(&wire).copied().unwrap_or(0.0);
        let hint = self.external_hints.get(&wire).copied().unwrap_or(0.0);
        self.cfg.base_cost
            + present_penalty * self.present_cost * self.cfg.present_cost_multiplier
            + history * self.cfg.history_cost_multiplier
            + hint
    }

    /// Update the historical congestion costs.
    ///
    /// For every wire that is currently used by more than one net, the excess
    /// usage (usage - 1) is added to the wire's history cost.
    pub fn update_history(&mut self) {
        // Exponentially decay all existing history so that stale congestion
        // fades out over iterations, then add fresh excess usage.
        for v in self.wire_history.values_mut() {
            *v *= 0.9;
        }
        for (&wire, &usage) in &self.wire_usage {
            if usage > 1 {
                *self.wire_history.entry(wire).or_default() += (usage - 1) as f64;
            }
        }
    }

    /// Recompute wire usage and ownership from the current design state.
    pub fn update_usage(&mut self, design: &crate::netlist::Design) {
        self.wire_usage.clear();
        self.wire_owner.clear();
        for net_idx in design.iter_net_indices() {
            let net = design.net(net_idx);
            if !net.alive {
                continue;
            }
            for &wire in net.wires.keys() {
                *self.wire_usage.entry(wire).or_default() += 1;
                self.wire_owner.insert(wire, net_idx);
            }
        }
    }

    /// Increment usage/owner state from one net's currently routed wires.
    pub fn add_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetId) {
        let net = design.net(net_idx);
        if !net.alive {
            return;
        }

        for &wire in net.wires.keys() {
            *self.wire_usage.entry(wire).or_default() += 1;
            self.wire_owner.insert(wire, net_idx);
        }
    }

    /// Decrement usage/owner state for one net's currently routed wires.
    pub fn remove_net_usage(&mut self, design: &crate::netlist::Design, net_idx: NetId) {
        let net = design.net(net_idx);
        if !net.alive {
            return;
        }

        for &wire in net.wires.keys() {
            if let Some(count) = self.wire_usage.get_mut(&wire) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.wire_usage.remove(&wire);
                    self.wire_owner.remove(&wire);
                }
            }
        }
    }

    /// Find all nets that touch at least one congested wire (usage > 1).
    pub fn find_congested_nets(&self, design: &crate::netlist::Design) -> Vec<NetId> {
        let congested_wires: rustc_hash::FxHashSet<WireId> = self
            .wire_usage
            .iter()
            .filter(|(_, &u)| u > 1)
            .map(|(&w, _)| w)
            .collect();

        if congested_wires.is_empty() {
            return Vec::new();
        }

        let mut nets = rustc_hash::FxHashSet::default();
        for net_idx in design.iter_net_indices() {
            let net = design.net(net_idx);
            if !net.alive {
                continue;
            }
            if net.wires.keys().any(|w| congested_wires.contains(w)) {
                nets.insert(net_idx);
            }
        }
        nets.into_iter().collect()
    }

    /// Count the number of wires with usage > 1 (congested wires).
    pub fn count_congested_wires(&self) -> usize {
        self.wire_usage.values().filter(|&&u| u > 1).count()
    }
}

//! Router1: A* rip-up and reroute router.
//!
//! This module implements an iterative A* routing algorithm that routes each net
//! independently, then detects congestion (wires used by multiple nets) and
//! rips up congested nets for rerouting with increased penalties. The process
//! repeats until all congestion is resolved or the iteration limit is reached.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use log::info;
use npnr_chipdb::read_packed;
use npnr_context::Context;
use npnr_netlist::NetIdx;
use npnr_types::{DelayT, IdString, PipId, PlaceStrength, WireId};
use rustc_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the Router1 algorithm.
pub struct Router1Cfg {
    /// Maximum number of rip-up-and-reroute iterations.
    pub max_iterations: usize,
    /// Penalty added to a wire each time it is involved in congestion.
    pub rip_up_penalty: DelayT,
    /// Weight multiplier for congestion cost.
    pub congestion_weight: f64,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for Router1Cfg {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            rip_up_penalty: 10,
            congestion_weight: 1.0,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during routing.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// A* search could not find any path for the named net.
    #[error("Failed to route net {0}: no path found")]
    NoPath(String),
    /// Routing did not converge within the iteration limit.
    #[error("Routing failed after {0} iterations, {1} nets still congested")]
    Congestion(usize, usize),
    /// Generic router error.
    #[error("Router error: {0}")]
    Generic(String),
}

// ---------------------------------------------------------------------------
// A* priority queue entry
// ---------------------------------------------------------------------------

/// An entry in the A* search priority queue.
#[derive(Clone)]
struct QueueEntry {
    /// The wire this entry represents.
    wire: WireId,
    /// g(n): accumulated cost from the source to this wire.
    cost: DelayT,
    /// f(n) = g(n) + h(n): total estimated cost through this wire.
    estimate: DelayT,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap (max-heap) behaves as a min-heap
        // by estimate. Break ties by preferring lower g-cost.
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.cost.cmp(&self.cost))
    }
}

// ---------------------------------------------------------------------------
// Router state
// ---------------------------------------------------------------------------

/// Internal mutable state for the Router1 algorithm.
struct Router1State {
    /// Per-wire penalty that increases when a wire is involved in congestion.
    wire_penalty: FxHashMap<WireId, DelayT>,
    /// Set of net indices ripped up in the current iteration.
    ripped_nets: FxHashSet<NetIdx>,
}

impl Router1State {
    fn new() -> Self {
        Self {
            wire_penalty: FxHashMap::default(),
            ripped_nets: FxHashSet::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Route all nets in the design using A* rip-up and reroute.
///
/// The algorithm:
/// 1. Collect all nets that need routing (have a driver and at least one user).
/// 2. Attempt an initial route for every net.
/// 3. Iteratively detect congested wires (used by >1 net), rip up the involved
///    nets, increase wire penalties, and reroute them.
/// 4. Repeat until no congestion remains or `max_iterations` is reached.
pub fn route_router1(ctx: &mut Context, cfg: &Router1Cfg) -> Result<(), RouterError> {
    let mut state = Router1State::new();

    // 1. Collect nets that need routing.
    let nets_to_route = collect_routable_nets(ctx);
    if cfg.verbose {
        info!("Router1: {} nets to route", nets_to_route.len());
    }

    // 2. Initial route attempt.
    for &net_idx in &nets_to_route {
        route_net(ctx, net_idx, &state.wire_penalty)?;
    }

    // 3. Rip-up-and-reroute loop.
    for iteration in 0..cfg.max_iterations {
        let congested = find_congested_nets(ctx);
        if congested.is_empty() {
            if cfg.verbose {
                info!("Router1: converged after {} iterations", iteration);
            }
            return Ok(());
        }

        if cfg.verbose {
            info!(
                "Router1: iteration {}, {} congested nets",
                iteration,
                congested.len()
            );
        }

        // Increase penalties for congested wires.
        let congested_wires = find_congested_wires(ctx);
        for wire in &congested_wires {
            let penalty = state.wire_penalty.entry(*wire).or_insert(0);
            *penalty += cfg.rip_up_penalty;
        }

        // Rip up all congested nets.
        state.ripped_nets.clear();
        for &net_idx in &congested {
            unroute_net(ctx, net_idx);
            state.ripped_nets.insert(net_idx);
        }

        // Reroute them with updated penalties.
        for &net_idx in &congested {
            route_net(ctx, net_idx, &state.wire_penalty)?;
        }
    }

    // Check if we still have congestion after exhausting iterations.
    let remaining = find_congested_nets(ctx);
    if remaining.is_empty() {
        Ok(())
    } else {
        Err(RouterError::Congestion(cfg.max_iterations, remaining.len()))
    }
}

// ---------------------------------------------------------------------------
// Net collection
// ---------------------------------------------------------------------------

/// Collect all net indices that need routing.
///
/// A net needs routing if it has a connected driver and at least one user.
pub fn collect_routable_nets(ctx: &Context) -> Vec<NetIdx> {
    let mut result = Vec::new();
    for (_, &net_idx) in &ctx.design.nets {
        let net = ctx.design.net(net_idx);
        if net.alive && net.has_driver() && net.num_users() > 0 {
            result.push(net_idx);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Single-net routing
// ---------------------------------------------------------------------------

/// Route a single net from its driver to all of its sinks using A* search.
///
/// For each user (sink) of the net, we find the sink cell's BEL pin wire and
/// run A* from the current routing tree to that sink wire. The resulting path
/// of PIPs is then bound in the context.
fn route_net(
    ctx: &mut Context,
    net_idx: NetIdx,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    let net = ctx.design.net(net_idx);
    let net_name = net.name;

    // Determine the driver wire.
    let driver = &net.driver;
    if !driver.is_connected() {
        return Ok(());
    }
    let driver_cell_idx = driver.cell;
    let driver_port = driver.port;

    let driver_cell = ctx.design.cell(driver_cell_idx);
    let driver_bel = driver_cell.bel;
    if !driver_bel.is_valid() {
        return Err(RouterError::Generic(format!(
            "Driver cell for net {} is not placed",
            ctx.name_of(net_name)
        )));
    }

    let src_wire = find_bel_pin_wire(ctx, driver_bel, driver_port);
    if !src_wire.is_valid() {
        return Err(RouterError::Generic(format!(
            "Cannot find driver wire for net {}",
            ctx.name_of(net_name)
        )));
    }

    // Bind the source wire to this net if not already bound.
    if ctx.is_wire_available(src_wire) {
        ctx.bind_wire(src_wire, net_name, PlaceStrength::Strong);
        let net_mut = ctx.design.net_mut(net_idx);
        net_mut.wires.insert(
            src_wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );
    }

    // Collect sink wires. We gather them before mutating ctx to avoid borrow issues.
    let num_users = ctx.design.net(net_idx).users.len();
    let mut sink_wires = Vec::with_capacity(num_users);
    for user_idx in 0..num_users {
        let user = &ctx.design.net(net_idx).users[user_idx];
        if !user.is_connected() {
            continue;
        }
        let user_cell_idx = user.cell;
        let user_port = user.port;
        let user_cell = ctx.design.cell(user_cell_idx);
        let user_bel = user_cell.bel;
        if !user_bel.is_valid() {
            continue;
        }
        let sink_wire = find_bel_pin_wire(ctx, user_bel, user_port);
        if sink_wire.is_valid() {
            sink_wires.push(sink_wire);
        }
    }

    // Route to each sink.
    for sink_wire in sink_wires {
        // Check if this sink is already routed.
        if ctx.design.net(net_idx).wires.contains_key(&sink_wire) {
            continue;
        }

        // Collect current routing tree wires as potential A* start points.
        let existing_wires: Vec<WireId> =
            ctx.design.net(net_idx).wires.keys().cloned().collect();

        let path = astar_route(ctx, &existing_wires, sink_wire, wire_penalty);

        match path {
            Some(pips) => {
                bind_route(ctx, net_idx, net_name, &pips);
            }
            None => {
                return Err(RouterError::NoPath(ctx.name_of(net_name)));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// BEL pin wire lookup
// ---------------------------------------------------------------------------

/// Find the wire connected to a specific BEL pin.
///
/// Iterates the BEL's pins in the chipdb to find one matching `port_name`,
/// then returns the corresponding tile wire as a WireId.
pub fn find_bel_pin_wire(ctx: &Context, bel: npnr_types::BelId, port_name: IdString) -> WireId {
    let bel_info = ctx.chipdb.bel_info(bel);
    let pins = bel_info.pins.get();
    let port_str = ctx.name_of(port_name);

    for pin in pins {
        let pin_name_ptr = pin.name.get();
        let pin_name = unsafe {
            let cstr = std::ffi::CStr::from_ptr(pin_name_ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("")
        };
        if pin_name == port_str {
            let wire_index: i32 = unsafe { read_packed!(*pin, wire_index) };
            return WireId::new(bel.tile(), wire_index);
        }
    }

    WireId::INVALID
}

// ---------------------------------------------------------------------------
// A* search
// ---------------------------------------------------------------------------

/// Run A* search from multiple source wires to a single destination wire.
///
/// Searches from all `src_wires` simultaneously, which allows the algorithm
/// to find the shortest path from any wire already in the routing tree.
/// Uses `estimate_delay` as the heuristic function.
///
/// Returns a sequence of PIPs forming the path in forward order (source to
/// destination), or `None` if no path exists.
fn astar_route(
    ctx: &Context,
    src_wires: &[WireId],
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Option<Vec<PipId>> {
    let src_set: FxHashSet<WireId> = src_wires.iter().cloned().collect();

    // Trivial case: destination is already in the source set.
    if src_set.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let mut heap = BinaryHeap::new();
    // visited: wire -> (best cost so far, pip used to reach it)
    let mut visited: FxHashMap<WireId, (DelayT, PipId)> = FxHashMap::default();

    // Seed the search with all source wires.
    for &src in src_wires {
        let h = ctx.estimate_delay(src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            estimate: h,
        });
        visited.insert(src, (0, PipId::INVALID));
    }

    while let Some(entry) = heap.pop() {
        // Skip if we already found a cheaper path to this wire.
        if let Some(&(prev_cost, _)) = visited.get(&entry.wire) {
            if entry.cost > prev_cost {
                continue;
            }
        }

        // Check if we reached the destination.
        if entry.wire == dst_wire {
            // Trace back the path through visited.
            let mut pips = Vec::new();
            let mut current = dst_wire;
            loop {
                if let Some(&(_, pip)) = visited.get(&current) {
                    if !pip.is_valid() {
                        // Reached a source wire.
                        break;
                    }
                    pips.push(pip);
                    current = ctx.get_pip_src_wire(pip);
                } else {
                    break;
                }
            }
            pips.reverse();
            return Some(pips);
        }

        // Expand: iterate all downhill pips from this wire.
        let wire_info = ctx.chipdb.wire_info(entry.wire);
        let downhill_refs = wire_info.pips_downhill.get();

        for pip_ref in downhill_refs {
            let tile_delta: i32 = unsafe { read_packed!(*pip_ref, tile_delta) };
            let pip_index: i32 = unsafe { read_packed!(*pip_ref, index) };
            let pip_tile = ctx.chipdb.rel_tile(entry.wire.tile(), tile_delta, 0);
            let pip = PipId::new(pip_tile, pip_index);

            let next_wire = ctx.get_pip_dst_wire(pip);

            // Cost of traversing this pip: pip delay + wire penalty + 1 base cost.
            let pip_delay = ctx.get_pip_delay(pip).max_delay();
            let penalty = wire_penalty.get(&next_wire).copied().unwrap_or(0);
            let new_cost = entry.cost + pip_delay + penalty + 1;

            // Skip if we already have a cheaper or equal path to next_wire.
            if let Some(&(prev_cost, _)) = visited.get(&next_wire) {
                if new_cost >= prev_cost {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, pip));

            let h = ctx.estimate_delay(next_wire, dst_wire);
            heap.push(QueueEntry {
                wire: next_wire,
                cost: new_cost,
                estimate: new_cost + h,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Route binding / unbinding
// ---------------------------------------------------------------------------

/// Bind a sequence of PIPs as the route for a net.
///
/// For each PIP in the path, binds the PIP and its destination wire to the
/// given net, and records the routing in the net's wire map.
pub fn bind_route(ctx: &mut Context, net_idx: NetIdx, net_name: IdString, path: &[PipId]) {
    for &pip in path {
        let dst_wire = ctx.get_pip_dst_wire(pip);
        ctx.bind_pip(pip, net_name, PlaceStrength::Strong);
        ctx.bind_wire(dst_wire, net_name, PlaceStrength::Strong);
        let net = ctx.design.net_mut(net_idx);
        net.wires.insert(
            dst_wire,
            npnr_netlist::PipMap {
                pip,
                strength: PlaceStrength::Strong,
            },
        );
    }
}

/// Rip up (unroute) a net by unbinding all its wires and PIPs.
pub fn unroute_net(ctx: &mut Context, net_idx: NetIdx) {
    let net = ctx.design.net(net_idx);
    let wires: Vec<WireId> = net.wires.keys().cloned().collect();
    let pips: Vec<PipId> = net.wires.values().map(|pm| pm.pip).collect();

    for wire in &wires {
        ctx.unbind_wire(*wire);
    }
    for pip in &pips {
        if pip.is_valid() {
            ctx.unbind_pip(*pip);
        }
    }

    ctx.design.net_mut(net_idx).wires.clear();
}

// ---------------------------------------------------------------------------
// Congestion detection
// ---------------------------------------------------------------------------

/// Find all wires that are used by more than one net (congested).
fn find_congested_wires(ctx: &Context) -> Vec<WireId> {
    let mut wire_usage: FxHashMap<WireId, u32> = FxHashMap::default();

    for (_, &net_idx) in &ctx.design.nets {
        let net = ctx.design.net(net_idx);
        if !net.alive {
            continue;
        }
        for wire in net.wires.keys() {
            *wire_usage.entry(*wire).or_insert(0) += 1;
        }
    }

    wire_usage
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(wire, _)| wire)
        .collect()
}

/// Find all nets that use at least one congested wire.
///
/// Returns a deduplicated list of net indices.
fn find_congested_nets(ctx: &Context) -> Vec<NetIdx> {
    let congested_wires: FxHashSet<WireId> = find_congested_wires(ctx).into_iter().collect();

    if congested_wires.is_empty() {
        return Vec::new();
    }

    let mut congested_nets = FxHashSet::default();
    for (_, &net_idx) in &ctx.design.nets {
        let net = ctx.design.net(net_idx);
        if !net.alive {
            continue;
        }
        for wire in net.wires.keys() {
            if congested_wires.contains(wire) {
                congested_nets.insert(net_idx);
                break;
            }
        }
    }

    congested_nets.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_chipdb::testutil::make_test_chipdb;
    use npnr_context::Context;
    use npnr_netlist::NetIdx;
    use npnr_types::{BelId, PipId, PlaceStrength, PortType, WireId};

    /// Create a fresh Context backed by the synthetic 2x2 chipdb.
    ///
    /// The chipdb has:
    /// - 4 tiles at (0,0), (1,0), (0,1), (1,1)
    /// - Each tile has 1 bel (LUT0), 2 wires (W0, W1), 1 pip (W0 -> W1)
    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    // =====================================================================
    // QueueEntry ordering tests
    // =====================================================================

    #[test]
    fn queue_entry_min_heap_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(QueueEntry {
            wire: WireId::new(0, 0),
            cost: 10,
            estimate: 50,
        });
        heap.push(QueueEntry {
            wire: WireId::new(0, 1),
            cost: 5,
            estimate: 20,
        });
        heap.push(QueueEntry {
            wire: WireId::new(1, 0),
            cost: 8,
            estimate: 35,
        });

        // Min-heap: smallest estimate should come out first.
        let first = heap.pop().unwrap();
        assert_eq!(first.estimate, 20);
        let second = heap.pop().unwrap();
        assert_eq!(second.estimate, 35);
        let third = heap.pop().unwrap();
        assert_eq!(third.estimate, 50);
    }

    #[test]
    fn queue_entry_tiebreak_by_cost() {
        let mut heap = BinaryHeap::new();

        heap.push(QueueEntry {
            wire: WireId::new(0, 0),
            cost: 30,
            estimate: 50,
        });
        heap.push(QueueEntry {
            wire: WireId::new(0, 1),
            cost: 10,
            estimate: 50,
        });

        // Same estimate, lower cost should come first.
        let first = heap.pop().unwrap();
        assert_eq!(first.cost, 10);
    }

    // =====================================================================
    // A* pathfinding on the synthetic chipdb
    // =====================================================================

    #[test]
    fn astar_same_wire_returns_empty_path() {
        let ctx = make_context();
        let wire = WireId::new(0, 0);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[wire], wire, &penalty);
        assert!(path.is_some());
        assert!(path.unwrap().is_empty());
    }

    #[test]
    fn astar_single_pip_path() {
        let ctx = make_context();
        // In the synthetic chipdb, tile 0 has pip 0: W0 -> W1.
        let src = WireId::new(0, 0);
        let dst = WireId::new(0, 1);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_some());
        let pips = path.unwrap();
        assert_eq!(pips.len(), 1);
        assert_eq!(pips[0], PipId::new(0, 0));
    }

    #[test]
    fn astar_verifies_pip_connectivity() {
        let ctx = make_context();
        let src = WireId::new(0, 0);
        let dst = WireId::new(0, 1);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[src], dst, &penalty).unwrap();
        let pip = path[0];
        assert_eq!(ctx.get_pip_src_wire(pip), src);
        assert_eq!(ctx.get_pip_dst_wire(pip), dst);
    }

    #[test]
    fn astar_no_path_returns_none() {
        let ctx = make_context();
        // W1 has no downhill pips, so from W1 we cannot reach W0.
        let src = WireId::new(0, 1);
        let dst = WireId::new(0, 0);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_none());
    }

    #[test]
    fn astar_cross_tile_no_path() {
        let ctx = make_context();
        // In the synthetic chipdb there are no cross-tile pips.
        let src = WireId::new(0, 0);
        let dst = WireId::new(1, 0);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_none());
    }

    #[test]
    fn astar_with_penalty_still_finds_path() {
        let ctx = make_context();
        let src = WireId::new(0, 0);
        let dst = WireId::new(0, 1);
        let mut penalty = FxHashMap::default();
        penalty.insert(dst, 1000);

        // Should still find the path (only one route available).
        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1);
    }

    #[test]
    fn astar_multi_source_picks_closest() {
        let ctx = make_context();
        // Two sources: W0 in tile 0 and W0 in tile 1.
        // Destination: W1 in tile 0.
        // Only the pip in tile 0 connects W0(tile 0) -> W1(tile 0).
        let src_wires = vec![WireId::new(0, 0), WireId::new(1, 0)];
        let dst = WireId::new(0, 1);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &src_wires, dst, &penalty);
        assert!(path.is_some());
        let pips = path.unwrap();
        assert_eq!(pips.len(), 1);
        assert_eq!(pips[0], PipId::new(0, 0));
    }

    #[test]
    fn astar_empty_sources_returns_none() {
        let ctx = make_context();
        let dst = WireId::new(0, 1);
        let penalty = FxHashMap::default();

        let path = astar_route(&ctx, &[], dst, &penalty);
        assert!(path.is_none());
    }

    // =====================================================================
    // Bind route tests
    // =====================================================================

    #[test]
    fn bind_route_records_wires_and_pips() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_bind");
        let net_idx = ctx.design.add_net(net_name);

        let pip = PipId::new(0, 0);
        let dst_wire = ctx.get_pip_dst_wire(pip);

        bind_route(&mut ctx, net_idx, net_name, &[pip]);

        // Wire should be bound in the context.
        assert!(!ctx.is_wire_available(dst_wire));
        assert_eq!(ctx.get_bound_wire_net(dst_wire), Some(net_name));

        // PIP should be bound in the context.
        assert!(!ctx.is_pip_available(pip));

        // Net should record the wire in its routing tree.
        let net = ctx.design.net(net_idx);
        assert!(net.wires.contains_key(&dst_wire));
        assert_eq!(net.wires[&dst_wire].pip, pip);
    }

    #[test]
    fn bind_empty_route() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_empty");
        let net_idx = ctx.design.add_net(net_name);

        bind_route(&mut ctx, net_idx, net_name, &[]);

        // No wires or pips should be bound.
        let net = ctx.design.net(net_idx);
        assert!(net.wires.is_empty());
    }

    // =====================================================================
    // Unroute (rip-up) mechanics
    // =====================================================================

    #[test]
    fn unroute_clears_net_wires() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_rip");
        let net_idx = ctx.design.add_net(net_name);

        let wire = WireId::new(0, 1);
        let pip = PipId::new(0, 0);

        ctx.bind_wire(wire, net_name, PlaceStrength::Strong);
        ctx.bind_pip(pip, net_name, PlaceStrength::Strong);
        ctx.design.net_mut(net_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip,
                strength: PlaceStrength::Strong,
            },
        );

        assert!(!ctx.is_wire_available(wire));
        assert!(!ctx.is_pip_available(pip));

        unroute_net(&mut ctx, net_idx);

        assert!(ctx.is_wire_available(wire));
        assert!(ctx.is_pip_available(pip));
        assert!(ctx.design.net(net_idx).wires.is_empty());
    }

    #[test]
    fn unroute_handles_invalid_pip() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_src");
        let net_idx = ctx.design.add_net(net_name);

        // Source wire has PipId::INVALID (no driving pip).
        let wire = WireId::new(0, 0);
        ctx.bind_wire(wire, net_name, PlaceStrength::Strong);
        ctx.design.net_mut(net_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        // Should not panic when encountering INVALID pip.
        unroute_net(&mut ctx, net_idx);
        assert!(ctx.is_wire_available(wire));
        assert!(ctx.design.net(net_idx).wires.is_empty());
    }

    #[test]
    fn unroute_multiple_wires() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_multi");
        let net_idx = ctx.design.add_net(net_name);

        let wire0 = WireId::new(0, 0);
        let wire1 = WireId::new(0, 1);
        let pip = PipId::new(0, 0);

        ctx.bind_wire(wire0, net_name, PlaceStrength::Strong);
        ctx.design.net_mut(net_idx).wires.insert(
            wire0,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        ctx.bind_wire(wire1, net_name, PlaceStrength::Strong);
        ctx.bind_pip(pip, net_name, PlaceStrength::Strong);
        ctx.design.net_mut(net_idx).wires.insert(
            wire1,
            npnr_netlist::PipMap {
                pip,
                strength: PlaceStrength::Strong,
            },
        );

        unroute_net(&mut ctx, net_idx);

        assert!(ctx.is_wire_available(wire0));
        assert!(ctx.is_wire_available(wire1));
        assert!(ctx.is_pip_available(pip));
        assert!(ctx.design.net(net_idx).wires.is_empty());
    }

    // =====================================================================
    // Congestion detection
    // =====================================================================

    #[test]
    fn no_congestion_with_no_nets() {
        let ctx = make_context();
        assert!(find_congested_wires(&ctx).is_empty());
        assert!(find_congested_nets(&ctx).is_empty());
    }

    #[test]
    fn no_congestion_with_single_net() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_a");
        let net_idx = ctx.design.add_net(net_name);

        let wire = WireId::new(0, 0);
        ctx.design.net_mut(net_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        assert!(find_congested_wires(&ctx).is_empty());
    }

    #[test]
    fn congestion_detected_with_shared_wire() {
        let mut ctx = make_context();
        let wire = WireId::new(0, 0);

        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design.add_net(net_a_name);
        ctx.design.net_mut(net_a_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design.net_mut(net_b_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        let congested = find_congested_wires(&ctx);
        assert_eq!(congested.len(), 1);
        assert_eq!(congested[0], wire);
    }

    #[test]
    fn congested_nets_identified() {
        let mut ctx = make_context();
        let wire = WireId::new(0, 0);

        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design.add_net(net_a_name);
        ctx.design.net_mut(net_a_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design.net_mut(net_b_idx).wires.insert(
            wire,
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        let congested_nets = find_congested_nets(&ctx);
        assert_eq!(congested_nets.len(), 2);
        let net_set: FxHashSet<NetIdx> = congested_nets.into_iter().collect();
        assert!(net_set.contains(&net_a_idx));
        assert!(net_set.contains(&net_b_idx));
    }

    #[test]
    fn non_shared_wires_not_congested() {
        let mut ctx = make_context();

        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design.add_net(net_a_name);
        ctx.design.net_mut(net_a_idx).wires.insert(
            WireId::new(0, 0),
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design.net_mut(net_b_idx).wires.insert(
            WireId::new(1, 0),
            npnr_netlist::PipMap {
                pip: PipId::INVALID,
                strength: PlaceStrength::Strong,
            },
        );

        assert!(find_congested_wires(&ctx).is_empty());
        assert!(find_congested_nets(&ctx).is_empty());
    }

    // =====================================================================
    // Net routing
    // =====================================================================

    #[test]
    fn route_net_same_pin_driver_and_sink() {
        // Route a net where driver and sink are on the same bel pin wire.
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port_name = ctx.id("I0");

        let cell_name = ctx.id("cell_a");
        let cell_idx = ctx.design.add_cell(cell_name, lut_type);
        ctx.design.cell_mut(cell_idx).add_port(port_name, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_name, PlaceStrength::Placer);

        let net_name = ctx.id("net_self");
        let net_idx = ctx.design.add_net(net_name);

        ctx.design.net_mut(net_idx).driver = npnr_netlist::PortRef {
            cell: cell_idx,
            port: port_name,
            budget: 0,
        };
        ctx.design.net_mut(net_idx).users.push(npnr_netlist::PortRef {
            cell: cell_idx,
            port: port_name,
            budget: 0,
        });

        let penalty = FxHashMap::default();
        let result = route_net(&mut ctx, net_idx, &penalty);
        assert!(result.is_ok());
    }

    #[test]
    fn route_net_cross_tile_fails_in_minimal_chipdb() {
        // The synthetic chipdb has no cross-tile pips.
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");

        let driver_name = ctx.id("driver");
        let driver_idx = ctx.design.add_cell(driver_name, lut_type);
        ctx.design.cell_mut(driver_idx).add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), driver_name, PlaceStrength::Placer);

        let sink_name = ctx.id("sink");
        let sink_idx = ctx.design.add_cell(sink_name, lut_type);
        ctx.design.cell_mut(sink_idx).add_port(port, PortType::In);
        ctx.bind_bel(BelId::new(1, 0), sink_name, PlaceStrength::Placer);

        let net_name = ctx.id("test_net");
        let net_idx = ctx.design.add_net(net_name);

        ctx.design.net_mut(net_idx).driver = npnr_netlist::PortRef {
            cell: driver_idx,
            port,
            budget: 0,
        };
        ctx.design.net_mut(net_idx).users.push(npnr_netlist::PortRef {
            cell: sink_idx,
            port,
            budget: 0,
        });

        let penalty = FxHashMap::default();
        let result = route_net(&mut ctx, net_idx, &penalty);
        assert!(result.is_err());
    }

    // =====================================================================
    // find_bel_pin_wire
    // =====================================================================

    #[test]
    fn find_bel_pin_wire_valid() {
        let ctx = make_context();
        let bel = BelId::new(0, 0);
        let port = ctx.id("I0");
        let wire = find_bel_pin_wire(&ctx, bel, port);
        assert_eq!(wire, WireId::new(0, 0));
    }

    #[test]
    fn find_bel_pin_wire_different_tiles() {
        let ctx = make_context();
        for tile in 0..4 {
            let bel = BelId::new(tile, 0);
            let port = ctx.id("I0");
            let wire = find_bel_pin_wire(&ctx, bel, port);
            assert_eq!(wire, WireId::new(tile, 0));
        }
    }

    #[test]
    fn find_bel_pin_wire_invalid_port() {
        let ctx = make_context();
        let bel = BelId::new(0, 0);
        let port = ctx.id("NONEXISTENT");
        let wire = find_bel_pin_wire(&ctx, bel, port);
        assert!(!wire.is_valid());
    }

    // =====================================================================
    // Router1Cfg defaults
    // =====================================================================

    #[test]
    fn default_config() {
        let cfg = Router1Cfg::default();
        assert_eq!(cfg.max_iterations, 500);
        assert_eq!(cfg.rip_up_penalty, 10);
        assert!((cfg.congestion_weight - 1.0).abs() < f64::EPSILON);
        assert!(!cfg.verbose);
    }

    // =====================================================================
    // RouterError display
    // =====================================================================

    #[test]
    fn router_error_no_path() {
        let err = RouterError::NoPath("my_net".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("my_net"));
        assert!(msg.contains("no path"));
    }

    #[test]
    fn router_error_congestion() {
        let err = RouterError::Congestion(100, 5);
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn router_error_generic() {
        let err = RouterError::Generic("oops".to_string());
        assert!(format!("{}", err).contains("oops"));
    }

    // =====================================================================
    // Integration: route_router1
    // =====================================================================

    #[test]
    fn route_empty_design() {
        let mut ctx = make_context();
        let cfg = Router1Cfg::default();
        let result = route_router1(&mut ctx, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn route_design_with_no_routable_nets() {
        let mut ctx = make_context();
        // Add a net with no driver.
        let net_name = ctx.id("no_driver");
        ctx.design.add_net(net_name);

        let cfg = Router1Cfg::default();
        let result = route_router1(&mut ctx, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn route_design_with_no_users() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");

        let cell_name = ctx.id("driver");
        let cell_idx = ctx.design.add_cell(cell_name, lut_type);
        ctx.design.cell_mut(cell_idx).add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_name, PlaceStrength::Placer);

        let net_name = ctx.id("driveronly");
        let net_idx = ctx.design.add_net(net_name);
        ctx.design.net_mut(net_idx).driver = npnr_netlist::PortRef {
            cell: cell_idx,
            port,
            budget: 0,
        };
        // No users.

        let cfg = Router1Cfg::default();
        let result = route_router1(&mut ctx, &cfg);
        assert!(result.is_ok());
    }

    // =====================================================================
    // Wire penalty accumulation
    // =====================================================================

    #[test]
    fn wire_penalty_accumulates() {
        let cfg = Router1Cfg::default();
        let mut state = Router1State::new();
        let wire = WireId::new(0, 0);

        *state.wire_penalty.entry(wire).or_insert(0) += cfg.rip_up_penalty;
        assert_eq!(state.wire_penalty[&wire], 10);

        *state.wire_penalty.entry(wire).or_insert(0) += cfg.rip_up_penalty;
        assert_eq!(state.wire_penalty[&wire], 20);
    }

    // =====================================================================
    // collect_routable_nets
    // =====================================================================

    #[test]
    fn collect_routable_nets_empty_design() {
        let ctx = make_context();
        assert!(collect_routable_nets(&ctx).is_empty());
    }

    #[test]
    fn collect_routable_nets_skips_no_driver() {
        let mut ctx = make_context();
        let net_name = ctx.id("no_driver");
        ctx.design.add_net(net_name);
        assert!(collect_routable_nets(&ctx).is_empty());
    }

    #[test]
    fn collect_routable_nets_finds_valid_net() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");

        let cell_name = ctx.id("cell");
        let cell_idx = ctx.design.add_cell(cell_name, lut_type);
        ctx.design.cell_mut(cell_idx).add_port(port, PortType::Out);

        let net_name = ctx.id("routable");
        let net_idx = ctx.design.add_net(net_name);
        ctx.design.net_mut(net_idx).driver = npnr_netlist::PortRef {
            cell: cell_idx,
            port,
            budget: 0,
        };
        ctx.design.net_mut(net_idx).users.push(npnr_netlist::PortRef {
            cell: cell_idx,
            port,
            budget: 0,
        });

        let nets = collect_routable_nets(&ctx);
        assert_eq!(nets.len(), 1);
        assert_eq!(nets[0], net_idx);
    }
}

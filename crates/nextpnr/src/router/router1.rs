//! Router1: A* rip-up and reroute router.
//!
//! This module implements an iterative A* routing algorithm that routes each net
//! independently, then detects congestion (wires used by multiple nets) and
//! rips up congested nets for rerouting with increased penalties. The process
//! repeats until all congestion is resolved or the iteration limit is reached.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::context::Context;
use crate::netlist::NetId;
use crate::types::{DelayT, PipId, WireId};
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    bind_route, collect_routable_nets, collect_sink_wires, find_congested_wires, setup_net_source,
    unroute_net,
};
use super::RouterError;

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
// A* priority queue entry
// ---------------------------------------------------------------------------

/// An entry in the A* search priority queue.
#[derive(Clone)]
pub(crate) struct QueueEntry {
    /// The wire this entry represents.
    pub wire: WireId,
    /// g(n): accumulated cost from the source to this wire.
    pub cost: DelayT,
    /// f(n) = g(n) + h(n): total estimated cost through this wire.
    pub estimate: DelayT,
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
pub(crate) struct Router1State {
    /// Per-wire penalty that increases when a wire is involved in congestion.
    pub wire_penalty: FxHashMap<WireId, DelayT>,
    /// Per-wire usage count, updated incrementally as nets are ripped up/rerouted.
    pub wire_usage: FxHashMap<WireId, u32>,
    /// Set of net indices ripped up in the current iteration.
    pub ripped_nets: FxHashSet<NetId>,
}

impl Router1State {
    pub(crate) fn new() -> Self {
        Self {
            wire_penalty: FxHashMap::default(),
            wire_usage: FxHashMap::default(),
            ripped_nets: FxHashSet::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Router1: A* rip-up and reroute.
pub struct Router1;

impl super::Router for Router1 {
    type Config = Router1Cfg;

    fn route(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), super::RouterError> {
        route_router1(ctx, cfg)
    }
}

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
        route_net_impl(ctx, net_idx, &state.wire_penalty)?;
        update_wire_usage_for_net(ctx, &mut state.wire_usage, net_idx, true);
    }

    // 3. Rip-up-and-reroute loop.
    for iteration in 0..cfg.max_iterations {
        let congested_wires: Vec<WireId> = state
            .wire_usage
            .iter()
            .filter_map(|(&wire, &count)| (count > 1).then_some(wire))
            .collect();
        let congested = find_nets_touching_wires(ctx, &congested_wires);
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
        for wire in &congested_wires {
            let penalty = state.wire_penalty.entry(*wire).or_insert(0);
            *penalty += cfg.rip_up_penalty;
        }

        // Rip up all congested nets.
        state.ripped_nets.clear();
        for &net_idx in &congested {
            update_wire_usage_for_net(ctx, &mut state.wire_usage, net_idx, false);
            unroute_net(ctx, net_idx);
            state.ripped_nets.insert(net_idx);
        }

        // Reroute them with updated penalties.
        for &net_idx in &congested {
            route_net_impl(ctx, net_idx, &state.wire_penalty)?;
            update_wire_usage_for_net(ctx, &mut state.wire_usage, net_idx, true);
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
// Single-net routing
// ---------------------------------------------------------------------------

/// Route a single net from its driver to all of its sinks using A* search.
///
/// For each user (sink) of the net, we find the sink cell's BEL pin wire and
/// run A* from the current routing tree to that sink wire. The resulting path
/// of PIPs is then bound in the context.
#[cfg(test)]
pub(crate) fn route_net(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    route_net_impl(ctx, net, wire_penalty)
}

fn route_net_impl(
    ctx: &mut Context,
    net: NetId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Result<(), RouterError> {
    if setup_net_source(ctx, net)?.is_none() {
        return Ok(());
    }

    let net_name = ctx.net(net).name_id();
    let sink_wires = collect_sink_wires(ctx, net);

    // Route to each sink.
    for sink_wire in sink_wires {
        // Check if this sink is already routed.
        if ctx.net(net).wires().contains_key(&sink_wire) {
            continue;
        }

        // Collect current routing tree wires as potential A* start points.
        let existing_wires: Vec<WireId> = ctx.net(net).wire_ids().collect();

        let path = astar_route(ctx, &existing_wires, sink_wire, wire_penalty);

        match path {
            Some(pips) => {
                bind_route(ctx, net, &pips);
            }
            None => {
                return Err(RouterError::NoPath(ctx.name_of(net_name).to_owned()));
            }
        }
    }

    Ok(())
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
pub(crate) fn astar_route(
    ctx: &Context,
    src_wires: &[WireId],
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
) -> Option<Vec<PipId>> {
    let src_set: FxHashSet<WireId> = src_wires.iter().copied().collect();

    // Trivial case: destination is already in the source set.
    if src_set.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let init_capacity = src_wires.len().saturating_mul(8).max(16);
    let mut heap = BinaryHeap::with_capacity(init_capacity);
    // visited: wire -> (best cost so far, pip used to reach it)
    let mut visited: FxHashMap<WireId, (DelayT, Option<PipId>)> =
        FxHashMap::with_capacity_and_hasher(init_capacity, Default::default());

    // Seed the search with all source wires.
    for &src in src_wires {
        let h = ctx.estimate_delay(src, dst_wire);
        heap.push(QueueEntry {
            wire: src,
            cost: 0,
            estimate: h,
        });
        visited.insert(src, (0, None));
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
            while let Some(&(_, pip)) = visited.get(&current) {
                let pip = match pip {
                    Some(pip) => pip,
                    None => break,
                };
                if !pip.is_valid() {
                    // Reached a source wire.
                    break;
                }
                pips.push(pip);
                current = ctx.pip(pip).src_wire().id();
            }
            pips.reverse();
            return Some(pips);
        }

        // Expand: iterate all downhill pips from this wire.
        let wire_info = ctx.chipdb().wire_info(entry.wire);
        let downhill_indices = wire_info.pips_downhill.get();

        for &pip_index in downhill_indices {
            // PIPs are tile-local: same tile as the wire.
            let pip = PipId::new(entry.wire.tile(), pip_index);

            let next_wire = ctx.pip(pip).dst_wire().id();

            // Cost of traversing this pip: pip delay + wire penalty + 1 base cost.
            let pip_delay = ctx.pip(pip).delay().max_delay();
            let penalty = wire_penalty.get(&next_wire).copied().unwrap_or(0);
            let new_cost = entry.cost + pip_delay + penalty + 1;

            // Skip if we already have a cheaper or equal path to next_wire.
            if let Some(&(prev_cost, _)) = visited.get(&next_wire) {
                if new_cost >= prev_cost {
                    continue;
                }
            }

            visited.insert(next_wire, (new_cost, Some(pip)));

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
// Congestion detection
// ---------------------------------------------------------------------------

fn update_wire_usage_for_net(
    ctx: &Context,
    wire_usage: &mut FxHashMap<WireId, u32>,
    net_idx: NetId,
    add: bool,
) {
    let net = ctx.net(net_idx);
    if !net.is_alive() {
        return;
    }

    for &wire in net.wires().keys() {
        if add {
            *wire_usage.entry(wire).or_default() += 1;
        } else if let Some(count) = wire_usage.get_mut(&wire) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                wire_usage.remove(&wire);
            }
        }
    }
}

fn find_nets_touching_wires(ctx: &Context, congested_wires: &[WireId]) -> Vec<NetId> {
    if congested_wires.is_empty() {
        return Vec::new();
    }

    let congested: FxHashSet<WireId> = congested_wires.iter().copied().collect();
    let mut nets = FxHashSet::default();

    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        if net.wires().keys().any(|wire| congested.contains(wire)) {
            nets.insert(net_idx);
        }
    }

    nets.into_iter().collect()
}

/// Find all nets that use at least one congested wire.
///
/// Returns a deduplicated list of net indices.
pub(crate) fn find_congested_nets(ctx: &Context) -> Vec<NetId> {
    let congested_wires = find_congested_wires(ctx);
    find_nets_touching_wires(ctx, &congested_wires)
}

#[cfg(test)]
#[cfg(feature = "test-utils")]
mod tests {
    use super::*;
    use crate::chipdb::testutil::make_test_chipdb;
    use crate::context::Context;
    use crate::netlist::PortRef;
    use crate::types::{BelId, PipId, PlaceStrength, PortType, WireId};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::collections::BinaryHeap;

    fn make_context() -> Context {
        let chipdb = make_test_chipdb();
        Context::new(chipdb)
    }

    // QueueEntry ordering tests

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
        let first = heap.pop().unwrap();
        assert_eq!(first.cost, 10);
    }

    // A* pathfinding tests

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
        assert_eq!(ctx.pip(pip).src_wire().id(), src);
        assert_eq!(ctx.pip(pip).dst_wire().id(), dst);
    }

    #[test]
    fn astar_no_path_returns_none() {
        let ctx = make_context();
        let src = WireId::new(0, 1);
        let dst = WireId::new(0, 0);
        let penalty = FxHashMap::default();
        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_none());
    }

    #[test]
    fn astar_cross_tile_no_path() {
        let ctx = make_context();
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
        let path = astar_route(&ctx, &[src], dst, &penalty);
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1);
    }

    #[test]
    fn astar_multi_source_picks_closest() {
        let ctx = make_context();
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

    // Bind route tests

    #[test]
    fn bind_route_records_wires_and_pips() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_bind");
        let net_idx = ctx.design.add_net(net_name);
        let pip = PipId::new(0, 0);
        let dst_wire = ctx.pip(pip).dst_wire().id();
        bind_route(&mut ctx, net_idx, &[pip]);
        assert!(!ctx.wire(dst_wire).is_available());
        assert_eq!(
            ctx.wire(dst_wire).bound_net().map(|n| n.name_id()),
            Some(net_name)
        );
        assert!(!ctx.pip(pip).is_available());
        let net = ctx.net(net_idx);
        assert!(net.wires().contains_key(&dst_wire));
        assert_eq!(net.wires()[&dst_wire].pip, Some(pip));
    }

    #[test]
    fn bind_empty_route() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_empty");
        let net_idx = ctx.design.add_net(net_name);
        bind_route(&mut ctx, net_idx, &[]);
        let net = ctx.net(net_idx);
        assert!(net.wires().is_empty());
    }

    // Unroute tests

    #[test]
    fn unroute_clears_net_wires() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_rip");
        let net_idx = ctx.design.add_net(net_name);
        let wire = WireId::new(0, 1);
        let pip = PipId::new(0, 0);
        ctx.bind_wire(wire, net_idx, PlaceStrength::Strong);
        ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(wire, Some(pip), PlaceStrength::Strong);
        assert!(!ctx.wire(wire).is_available());
        assert!(!ctx.pip(pip).is_available());
        unroute_net(&mut ctx, net_idx);
        assert!(ctx.wire(wire).is_available());
        assert!(ctx.pip(pip).is_available());
        assert!(ctx.net(net_idx).wires().is_empty());
    }

    #[test]
    fn unroute_handles_invalid_pip() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_src");
        let net_idx = ctx.design.add_net(net_name);
        let wire = WireId::new(0, 0);
        ctx.bind_wire(wire, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
        unroute_net(&mut ctx, net_idx);
        assert!(ctx.wire(wire).is_available());
        assert!(ctx.net(net_idx).wires().is_empty());
    }

    #[test]
    fn unroute_multiple_wires() {
        let mut ctx = make_context();
        let net_name = ctx.id("net_multi");
        let net_idx = ctx.design.add_net(net_name);
        let wire0 = WireId::new(0, 0);
        let wire1 = WireId::new(0, 1);
        let pip = PipId::new(0, 0);
        ctx.bind_wire(wire0, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(wire0, None, PlaceStrength::Strong);
        ctx.bind_wire(wire1, net_idx, PlaceStrength::Strong);
        ctx.bind_pip(pip, net_idx, PlaceStrength::Strong);
        ctx.design
            .net_edit(net_idx)
            .add_wire(wire1, Some(pip), PlaceStrength::Strong);
        unroute_net(&mut ctx, net_idx);
        assert!(ctx.wire(wire0).is_available());
        assert!(ctx.wire(wire1).is_available());
        assert!(ctx.pip(pip).is_available());
        assert!(ctx.net(net_idx).wires().is_empty());
    }

    // Congestion detection tests

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
        ctx.design
            .net_edit(net_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
        assert!(find_congested_wires(&ctx).is_empty());
    }

    #[test]
    fn congestion_detected_with_shared_wire() {
        let mut ctx = make_context();
        let wire = WireId::new(0, 0);
        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design.add_net(net_a_name);
        ctx.design
            .net_edit(net_a_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design
            .net_edit(net_b_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
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
        ctx.design
            .net_edit(net_a_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design
            .net_edit(net_b_idx)
            .add_wire(wire, None, PlaceStrength::Strong);
        let congested_nets_result = find_congested_nets(&ctx);
        assert_eq!(congested_nets_result.len(), 2);
        let net_set: FxHashSet<NetId> = congested_nets_result.into_iter().collect();
        assert!(net_set.contains(&net_a_idx));
        assert!(net_set.contains(&net_b_idx));
    }

    #[test]
    fn non_shared_wires_not_congested() {
        let mut ctx = make_context();
        let net_a_name = ctx.id("net_a");
        let net_a_idx = ctx.design.add_net(net_a_name);
        ctx.design
            .net_edit(net_a_idx)
            .add_wire(WireId::new(0, 0), None, PlaceStrength::Strong);
        let net_b_name = ctx.id("net_b");
        let net_b_idx = ctx.design.add_net(net_b_name);
        ctx.design
            .net_edit(net_b_idx)
            .add_wire(WireId::new(1, 0), None, PlaceStrength::Strong);
        assert!(find_congested_wires(&ctx).is_empty());
        assert!(find_congested_nets(&ctx).is_empty());
    }

    // Net routing tests

    #[test]
    fn route_net_same_pin_driver_and_sink() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port_name = ctx.id("I0");
        let cell_name = ctx.id("cell_a");
        let cell_idx = ctx.design.add_cell(cell_name, lut_type);
        ctx.design
            .cell_edit(cell_idx)
            .add_port(port_name, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), cell_idx, PlaceStrength::Placer);
        let net_name = ctx.id("net_self");
        let net_idx = ctx.design.add_net(net_name);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(cell_idx),
            port: port_name,
            budget: 0,
        });
        ctx.design.net_edit(net_idx).add_user_raw(PortRef {
            cell: Some(cell_idx),
            port: port_name,
            budget: 0,
        });
        let penalty = FxHashMap::default();
        let result = route_net(&mut ctx, net_idx, &penalty);
        assert!(result.is_ok());
    }

    #[test]
    fn route_net_cross_tile_fails_in_minimal_chipdb() {
        let mut ctx = make_context();
        let lut_type = ctx.id("LUT4");
        let port = ctx.id("I0");
        let driver_name = ctx.id("driver");
        let driver_idx = ctx.design.add_cell(driver_name, lut_type);
        ctx.design
            .cell_edit(driver_idx)
            .add_port(port, PortType::Out);
        ctx.bind_bel(BelId::new(0, 0), driver_idx, PlaceStrength::Placer);
        let sink_name = ctx.id("sink");
        let sink_idx = ctx.design.add_cell(sink_name, lut_type);
        ctx.design.cell_edit(sink_idx).add_port(port, PortType::In);
        ctx.bind_bel(BelId::new(1, 0), sink_idx, PlaceStrength::Placer);
        let net_name = ctx.id("test_net");
        let net_idx = ctx.design.add_net(net_name);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(driver_idx),
            port,
            budget: 0,
        });
        ctx.design.net_edit(net_idx).add_user_raw(PortRef {
            cell: Some(sink_idx),
            port,
            budget: 0,
        });
        let penalty = FxHashMap::default();
        let result = route_net(&mut ctx, net_idx, &penalty);
        assert!(result.is_err());
    }

    // BEL pin wire lookup tests (via view API)

    #[test]
    fn bel_pin_wire_valid() {
        let ctx = make_context();
        let bel = BelId::new(0, 0);
        let port = ctx.id("I0");
        let wire = ctx.bel(bel).pin_wire(port).map(|w| w.id());
        assert_eq!(wire, Some(WireId::new(0, 0)));
    }

    #[test]
    fn bel_pin_wire_different_tiles() {
        let ctx = make_context();
        for tile in 0..4 {
            let bel = BelId::new(tile, 0);
            let port = ctx.id("I0");
            let wire = ctx.bel(bel).pin_wire(port).map(|w| w.id());
            assert_eq!(wire, Some(WireId::new(tile, 0)));
        }
    }

    #[test]
    fn bel_pin_wire_invalid_port() {
        let ctx = make_context();
        let bel = BelId::new(0, 0);
        let port = ctx.id("NONEXISTENT");
        let wire = ctx.bel(bel).pin_wire(port).map(|w| w.id());
        assert!(wire.is_none());
    }

    // Wire penalty tests

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

    // collect_routable_nets tests

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
        ctx.design.cell_edit(cell_idx).add_port(port, PortType::Out);
        let net_name = ctx.id("routable");
        let net_idx = ctx.design.add_net(net_name);
        ctx.design.net_edit(net_idx).set_driver_raw(PortRef {
            cell: Some(cell_idx),
            port,
            budget: 0,
        });
        ctx.design.net_edit(net_idx).add_user_raw(PortRef {
            cell: Some(cell_idx),
            port,
            budget: 0,
        });
        let nets = collect_routable_nets(&ctx);
        assert_eq!(nets.len(), 1);
        assert_eq!(nets[0], net_idx);
    }
}

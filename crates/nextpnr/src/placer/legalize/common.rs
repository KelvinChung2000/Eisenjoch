//! Shared legalization helpers: unbinding movable cells and placing cluster children.

use crate::chipdb::{BelId, WireId};
use crate::common::{IdString, PlaceStrength};
use crate::context::Context;
use crate::netlist::{CellId, NetId, PortType};
use crate::placer::PlacerError;

use rustc_hash::FxHashMap;

/// Unbind all movable cells and their cluster children.
pub(crate) fn unbind_movable_cells(ctx: &mut Context, idx_to_cell: &[CellId]) {
    for &cell_id in idx_to_cell {
        let cell = ctx.design.cell(cell_id);
        if let Some(bel) = cell.bel {
            if !cell.bel_strength.is_locked() {
                ctx.unbind_bel(bel);
            }
        }
        if let Some(cluster) = ctx.design.clusters.get(&cell_id) {
            let children: Vec<_> = cluster.constr_children.clone();
            for child_id in children {
                let child = ctx.design.cell(child_id);
                if let Some(bel) = child.bel {
                    if !child.bel_strength.is_locked() {
                        ctx.unbind_bel(bel);
                    }
                }
            }
        }
    }
}

/// Place cluster children relative to the root BEL location.
///
/// Tries exact constraint position first, then any available BEL of matching type.
pub(crate) fn place_cluster_children(
    ctx: &mut Context,
    cell_id: CellId,
    root_bel: BelId,
) -> Result<(), PlacerError> {
    let cluster = match ctx.design.clusters.get(&cell_id) {
        Some(c) => c,
        None => return Ok(()),
    };
    let children: Vec<_> = cluster.constr_children.clone();
    let root_loc = ctx.bel(root_bel).loc();

    for child_id in children {
        let child = ctx.design.cell(child_id);
        let child_type = child.cell_type;
        let child_x = root_loc.x + child.constr_x;
        let child_y = root_loc.y + child.constr_y;

        let mut placed = false;

        let exact_candidates: Vec<_> = ctx
            .bels_for_bucket(child_type)
            .filter(|b| b.is_available() && b.loc().x == child_x && b.loc().y == child_y)
            .map(|b| b.id())
            .collect();
        for bel_id in exact_candidates {
            if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                placed = true;
                break;
            }
        }

        if !placed {
            let fallback_candidates: Vec<_> = ctx
                .bels_for_bucket(child_type)
                .filter(|b| b.is_available())
                .map(|b| b.id())
                .collect();
            for bel_id in fallback_candidates {
                if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                    placed = true;
                    break;
                }
            }
        }

        if !placed {
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to place cluster child {}",
                ctx.name_of(ctx.design.cell(child_id).name)
            )));
        }
    }

    Ok(())
}

/// Collect `(driver_wire, net)` pairs for every output port of `cell_id` that
/// drives a live net, assuming the cell is bound to `bel`. Returns empty if
/// the cell has no externally-driven output.
pub(crate) fn cell_driver_wires(
    ctx: &Context,
    cell_id: CellId,
    bel: BelId,
) -> Vec<(WireId, NetId)> {
    let cell = ctx.design.cell(cell_id);
    let ports: Vec<(IdString, PortType, Option<NetId>)> = cell
        .ports
        .iter()
        .map(|(name, data)| (*name, data.port_type(), data.net()))
        .collect();

    let bel_view = ctx.bel(bel);
    let mut out = Vec::with_capacity(2);
    for (name, ptype, net_opt) in ports {
        if !matches!(ptype, PortType::Out | PortType::InOut) {
            continue;
        }
        let Some(net_id) = net_opt else { continue };
        let net = ctx.net(net_id);
        if !net.is_alive() || net.num_users() == 0 {
            continue;
        }
        if net.driver_cell_port().map(|p| p.cell) != Some(cell_id) {
            continue;
        }
        if let Some(w) = bel_view.pin_wire(name) {
            out.push((w.id(), net_id));
        }
    }
    out
}

/// Collect `(sink_wire, net)` pairs for every input port of `cell_id` that
/// consumes a live net. Mirrors `cell_driver_wires` on the user side so that
/// the placer can reject a placement whose BEL input-pin wire lands on the
/// same routing node as another net's already-claimed pin wire — the
/// shared-input-mux case in xc7 slices that the driver-only check misses.
pub(crate) fn cell_user_wires(
    ctx: &Context,
    cell_id: CellId,
    bel: BelId,
) -> Vec<(WireId, NetId)> {
    let cell = ctx.design.cell(cell_id);
    let ports: Vec<(IdString, PortType, Option<NetId>)> = cell
        .ports
        .iter()
        .map(|(name, data)| (*name, data.port_type(), data.net()))
        .collect();

    let bel_view = ctx.bel(bel);
    let mut out = Vec::with_capacity(8);
    for (name, ptype, net_opt) in ports {
        if !matches!(ptype, PortType::In | PortType::InOut) {
            continue;
        }
        let Some(net_id) = net_opt else { continue };
        let net = ctx.net(net_id);
        if !net.is_alive() {
            continue;
        }
        if matches!(ptype, PortType::InOut)
            && net.driver_cell_port().map(|p| p.cell) == Some(cell_id)
        {
            continue;
        }
        if let Some(w) = bel_view.pin_wire(name) {
            out.push((w.id(), net_id));
        }
    }
    out
}

/// Union of `cell_driver_wires` and `cell_user_wires`: every BEL-pin wire
/// that `cell_id` would claim if bound to `bel`, paired with the net each
/// pin belongs to.
pub(crate) fn cell_pin_wires(
    ctx: &Context,
    cell_id: CellId,
    bel: BelId,
) -> Vec<(WireId, NetId)> {
    let mut out = cell_driver_wires(ctx, cell_id, bel);
    out.extend(cell_user_wires(ctx, cell_id, bel));
    out
}

/// Registry of routing-node ownership claimed by already-placed cells.
///
/// Covers both driver output wires *and* user input wires: two cells whose
/// BEL pin-wires belong to the same routing node (output mux or shared
/// input mux, as in xc7 slices) can only coexist if the shared node carries
/// a single net. This is the placer-side gate that prevents the router's
/// later `try_bind_wire_node` from rejecting a pre-reservation as a conflict.
pub(crate) struct DriverNodeRegistry {
    /// Wire -> owning NetId. Populated with every BEL-pin wire (driver +
    /// user) of placed cells plus all of their node-equivalent wires so
    /// membership lookup is O(1).
    claimed: FxHashMap<WireId, NetId>,
}

impl DriverNodeRegistry {
    pub fn new() -> Self {
        Self {
            claimed: FxHashMap::default(),
        }
    }

    /// Seed the registry with every cell already bound to a BEL. Captures
    /// packer-placed fixed cells (BUFG, IO) that legalize must respect.
    pub fn seed_from_bound(ctx: &Context) -> Self {
        let mut reg = Self::new();
        let bound: Vec<(CellId, BelId)> = ctx
            .design
            .iter_alive_cells()
            .filter_map(|(cid, cell)| cell.bel.map(|b| (cid, b)))
            .collect();
        for (cid, bel) in bound {
            reg.record(ctx, cid, bel);
        }
        reg
    }

    /// True iff placing `cell_id` at `bel` would not collide with an existing
    /// claim in the same routing node — either a shared output mux or a
    /// shared input mux from another placed cell.
    pub fn is_legal(&self, ctx: &Context, cell_id: CellId, bel: BelId) -> bool {
        for (w, net) in cell_pin_wires(ctx, cell_id, bel) {
            if let Some(&existing) = self.claimed.get(&w) {
                if existing != net {
                    return false;
                }
            }
            let mut conflict = false;
            ctx.chipdb().node_wires_cb(w, |nw| {
                if conflict {
                    return;
                }
                if let Some(&existing) = self.claimed.get(&nw) {
                    if existing != net {
                        conflict = true;
                    }
                }
            });
            if conflict {
                return false;
            }
        }
        true
    }

    /// Record `cell_id`'s driver and user pin wires (plus node-equivalents)
    /// as claimed. Call this after a successful `bind_bel`.
    pub fn record(&mut self, ctx: &Context, cell_id: CellId, bel: BelId) {
        for (w, net) in cell_pin_wires(ctx, cell_id, bel) {
            self.claimed.insert(w, net);
            ctx.chipdb().node_wires_cb(w, |nw| {
                self.claimed.insert(nw, net);
            });
        }
    }
}

/// After legalization, verify no two placed cells claim the same routing
/// node via their BEL pin wires (driver *or* user). Returns the first
/// offending cell pair as a `PlacerError`.
pub(crate) fn verify_shared_mux_legality(
    ctx: &Context,
) -> Result<(), PlacerError> {
    let mut seen: FxHashMap<WireId, (CellId, NetId)> = FxHashMap::default();
    let bound: Vec<(CellId, BelId)> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(cid, cell)| cell.bel.map(|b| (cid, b)))
        .collect();

    for (cid, bel) in bound {
        for (w, net) in cell_pin_wires(ctx, cid, bel) {
            let mut node_wires: Vec<WireId> = Vec::with_capacity(8);
            node_wires.push(w);
            ctx.chipdb().node_wires_cb(w, |nw| node_wires.push(nw));
            for nw in node_wires {
                if let Some(&(other_cid, other_net)) = seen.get(&nw) {
                    if other_net != net {
                        let a = ctx.name_of(ctx.design.cell(cid).name);
                        let b = ctx.name_of(ctx.design.cell(other_cid).name);
                        return Err(PlacerError::PlacementFailed(format!(
                            "Shared-mux conflict: cells {} and {} both claim node containing wire {:?}",
                            a, b, nw
                        )));
                    }
                } else {
                    seen.insert(nw, (cid, net));
                }
            }
        }
    }
    Ok(())
}

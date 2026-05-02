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

/// Place cluster children at exactly `root_loc + (constr_x, constr_y, constr_z)`,
/// honoring `constr_abs_z`. There is no fallback to a different BEL — if the
/// required slot is unavailable, that's an error the caller must surface so
/// the legalizer can try a different root position.
///
/// (The previous implementation fell back to "any available BEL of matching
/// type", which silently nuked the cluster contract: e.g. the LUT half of a
/// LUT-FF pair landed in the right SLICE but its FF half was rebound to a
/// random distant SLICE, which then both wrecked routability and surfaced as
/// shared-mux conflicts on whichever neighboring net's input mux that BEL
/// stomped on.)
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
        let (child_type, constr_x, constr_y, constr_z, constr_abs_z) = {
            let child = ctx.design.cell(child_id);
            (
                child.cell_type,
                child.constr_x,
                child.constr_y,
                child.constr_z,
                child.constr_abs_z,
            )
        };
        let want_x = root_loc.x + constr_x;
        let want_y = root_loc.y + constr_y;
        let want_z = if constr_abs_z {
            constr_z
        } else {
            root_loc.z + constr_z
        };

        let exact_candidates: Vec<_> = ctx
            .bels_for_bucket(child_type)
            .filter(|b| {
                b.is_available()
                    && b.loc().x == want_x
                    && b.loc().y == want_y
                    && b.loc().z == want_z
            })
            .map(|b| b.id())
            .collect();

        let mut placed = false;
        for bel_id in exact_candidates {
            if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                placed = true;
                break;
            }
        }

        if !placed {
            let child_name = ctx.name_of(ctx.design.cell(child_id).name).to_owned();
            let root_name = ctx.name_of(ctx.design.cell(cell_id).name).to_owned();
            let child_type_name = ctx.name_of(child_type).to_owned();
            return Err(PlacerError::PlacementFailed(format!(
                "Cannot place cluster child {} of root {}: required BEL slot \
                 ({},{},{}) of type {} is unavailable (root at ({},{},{}), \
                 constr_offset=({},{},{}){})",
                child_name,
                root_name,
                want_x,
                want_y,
                want_z,
                child_type_name,
                root_loc.x,
                root_loc.y,
                root_loc.z,
                constr_x,
                constr_y,
                constr_z,
                if constr_abs_z { ", abs_z" } else { "" },
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

/// Public re-export of `cell_pin_wires` for the routability module.
pub fn cell_pin_wires_pub(
    ctx: &Context,
    cell_id: CellId,
    bel: BelId,
) -> Vec<(WireId, NetId)> {
    cell_pin_wires(ctx, cell_id, bel)
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

/// Maximum peers walked per pin wire. Mega-nodes (xc7 BUFG global clock
/// distribution has ~65k peers) would dominate runtime if walked in full.
/// Set high enough to cover SLICE shared input mux topology (observed up
/// to several hundred peers when long routing wires fan into shared mux
/// inputs across multiple tiles).
const NODE_PEER_CAP: usize = 4096;

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
            let mut peers = 0usize;
            ctx.chipdb().node_wires_cb(w, |nw| {
                if conflict || peers >= NODE_PEER_CAP {
                    peers += 1;
                    return;
                }
                peers += 1;
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
            let mut peers = 0usize;
            ctx.chipdb().node_wires_cb(w, |nw| {
                if peers >= NODE_PEER_CAP {
                    peers += 1;
                    return;
                }
                peers += 1;
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
                        return Err(PlacerError::PlacementFailed(format!(
                            "Shared-mux conflict: {} and {} both claim node containing wire {:?}",
                            describe_cell(ctx, cid),
                            describe_cell(ctx, other_cid),
                            nw,
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

/// Format a cell as `name (cluster child of root, offset (dx,dy,dz))` when it
/// participates in a cluster, or just `name` otherwise. Used by the checkers
/// so error messages tell the operator whether a conflict involves a packed
/// cluster member.
fn describe_cell(ctx: &Context, cid: CellId) -> String {
    let cell = ctx.design.cell(cid);
    let name = ctx.name_of(cell.name).to_owned();
    let Some(root_id) = cell.cluster else {
        return format!("cell {}", name);
    };
    if root_id == cid {
        return format!("cell {} (cluster root)", name);
    }
    let root_name = ctx.name_of(ctx.design.cell(root_id).name);
    format!(
        "cell {} (cluster child of {}, constr_offset=({},{},{}){})",
        name,
        root_name,
        cell.constr_x,
        cell.constr_y,
        cell.constr_z,
        if cell.constr_abs_z { ", abs_z" } else { "" },
    )
}

/// Verify every cluster child is bound to a BEL at exactly
/// `root_loc + (constr_x, constr_y, constr_z)`. Catches the silent
/// fallback in `place_cluster_children` that picks any available BEL of
/// matching type when the exact target is full — that fallback breaks the
/// cluster contract and is a frequent root cause of routing-time failures
/// (LUT and FF land in unrelated tiles, no local interconnect path exists).
///
/// Run after `verify_shared_mux_legality` so cluster errors don't get
/// shadowed by node-conflict errors caused by the same misplacement.
pub(crate) fn verify_cluster_placement(ctx: &Context) -> Result<(), PlacerError> {
    let cluster_roots: Vec<CellId> = ctx.design.clusters.keys().copied().collect();

    for root_id in cluster_roots {
        let cluster = ctx.design.clusters.get(&root_id).expect("root just listed");
        let root_cell = ctx.design.cell(root_id);
        let root_name = ctx.name_of(root_cell.name).to_owned();

        let Some(root_bel) = root_cell.bel else {
            return Err(PlacerError::PlacementFailed(format!(
                "Cluster root {} is not bound to any BEL",
                root_name,
            )));
        };
        let root_loc = ctx.bel(root_bel).loc();

        let children: Vec<CellId> = cluster.constr_children.clone();
        for child_id in children {
            let child = ctx.design.cell(child_id);
            let child_name = ctx.name_of(child.name).to_owned();

            let Some(child_bel) = child.bel else {
                return Err(PlacerError::PlacementFailed(format!(
                    "Cluster child {} (root {}) is not bound to any BEL",
                    child_name, root_name,
                )));
            };
            let child_loc = ctx.bel(child_bel).loc();

            let want_x = root_loc.x + child.constr_x;
            let want_y = root_loc.y + child.constr_y;
            let want_z = if child.constr_abs_z {
                child.constr_z
            } else {
                root_loc.z + child.constr_z
            };

            if child_loc.x != want_x || child_loc.y != want_y || child_loc.z != want_z {
                return Err(PlacerError::PlacementFailed(format!(
                    "Cluster constraint violated: child {} of root {} is at \
                     ({},{},{}) but should be at ({},{},{}) \
                     (root at ({},{},{}), constr_offset=({},{},{}){})",
                    child_name,
                    root_name,
                    child_loc.x,
                    child_loc.y,
                    child_loc.z,
                    want_x,
                    want_y,
                    want_z,
                    root_loc.x,
                    root_loc.y,
                    root_loc.z,
                    child.constr_x,
                    child.constr_y,
                    child.constr_z,
                    if child.constr_abs_z { ", abs_z" } else { "" },
                )));
            }
        }
    }
    Ok(())
}

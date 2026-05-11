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
    bel_by_loc: &FxHashMap<(IdString, i32, i32, i32), BelId>,
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

        // O(1) lookup against the caller-built (bucket, x, y, z) -> BelId
        // index. The previous code did `ctx.bels_for_bucket(child_type).collect()`
        // matching the slot, which on FPGA01 (~1 M LUT BELs) was ~1 M ops per
        // child × thousands of cluster roots, dominating the post-rejection
        // bind cost.
        let key = (ctx.resolve_bucket(child_type), want_x, want_y, want_z);
        let candidate = bel_by_loc
            .get(&key)
            .copied()
            .filter(|&b| ctx.bel(b).is_available());

        let mut placed = false;
        if let Some(bel_id) = candidate {
            if ctx.bind_bel(bel_id, child_id, PlaceStrength::Placer) {
                placed = true;
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
///
/// Storage is per-node, not per-wire. The previous implementation expanded
/// every claimed wire's node-equivalence class into `claimed` at record
/// time and re-walked the class at is_legal time. Both walks were O(peers),
/// and mega-nodes (BUFG ≈ 65 k peers) were silently truncated at
/// `NODE_PEER_CAP`. Keying directly on `chipdb.node_id(w)` makes record and
/// is_legal both O(1) per pin, and the mega-node truncation goes away.
pub(crate) struct DriverNodeRegistry {
    /// Multi-tile node id -> owning NetId. Populated for every BEL-pin wire
    /// of a placed cell whose `chipdb.node_id(...)` returns `Some(_)`.
    claimed_nodes: FxHashMap<u64, NetId>,
    /// Tile-local wires (no multi-tile node) keyed directly.
    claimed_local: FxHashMap<WireId, NetId>,
}

impl DriverNodeRegistry {
    pub fn new() -> Self {
        Self {
            claimed_nodes: FxHashMap::default(),
            claimed_local: FxHashMap::default(),
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

    #[inline]
    fn lookup(&self, ctx: &Context, wire: WireId) -> Option<NetId> {
        match ctx.chipdb().node_id(wire) {
            Some(node) => self.claimed_nodes.get(&node).copied(),
            None => self.claimed_local.get(&wire).copied(),
        }
    }

    #[inline]
    fn insert(&mut self, ctx: &Context, wire: WireId, net: NetId) {
        match ctx.chipdb().node_id(wire) {
            Some(node) => {
                self.claimed_nodes.insert(node, net);
            }
            None => {
                self.claimed_local.insert(wire, net);
            }
        }
    }

    /// True iff placing `cell_id` at `bel` would not collide with an existing
    /// claim in the same routing node — either a shared output mux or a
    /// shared input mux from another placed cell.
    pub fn is_legal(&self, ctx: &Context, cell_id: CellId, bel: BelId) -> bool {
        for (w, net) in cell_pin_wires(ctx, cell_id, bel) {
            if let Some(existing) = self.lookup(ctx, w) {
                if existing != net {
                    return false;
                }
            }
        }
        true
    }

    /// Record `cell_id`'s driver and user pin wires as claimed. Call this
    /// after a successful `bind_bel`.
    pub fn record(&mut self, ctx: &Context, cell_id: CellId, bel: BelId) {
        for (w, net) in cell_pin_wires(ctx, cell_id, bel) {
            self.insert(ctx, w, net);
        }
    }

    /// Same as `is_legal`, but takes a precomputed `CellPinTemplate` so the
    /// per-call cost is only the `bel.pin_wire(name)` lookups + a single
    /// `node_id` resolution per pin, skipping per-call cell-port walks.
    pub fn is_legal_template(
        &self,
        ctx: &Context,
        template: &CellPinTemplate,
        bel: BelId,
    ) -> bool {
        let bel_view = ctx.bel(bel);
        for (port_name, net) in &template.pin_ports {
            let net = *net;
            let Some(w) = bel_view.pin_wire(*port_name) else {
                continue;
            };
            let w = w.id();
            if let Some(existing) = self.lookup(ctx, w) {
                if existing != net {
                    return false;
                }
            }
        }
        true
    }
}

/// Cached cell-side pin info: filtered list of `(port_name, net_id)` for all
/// driver/user ports of a cell that contribute to shared-mux legality.
/// Build once per cell via `build_cell_pin_template`; reuse for every BEL
/// candidate the cell is tested against.
pub(crate) struct CellPinTemplate {
    pub pin_ports: Vec<(IdString, NetId)>,
}

pub(crate) fn build_cell_pin_template(ctx: &Context, cell_id: CellId) -> CellPinTemplate {
    let cell = ctx.design.cell(cell_id);
    let mut pin_ports: Vec<(IdString, NetId)> = Vec::with_capacity(8);
    for (name, data) in cell.ports.iter() {
        let port_type = data.port_type();
        let Some(net_id) = data.net() else { continue };
        let net = ctx.net(net_id);
        if !net.is_alive() {
            continue;
        }
        match port_type {
            PortType::Out => {
                if net.num_users() == 0 {
                    continue;
                }
                if net.driver_cell_port().map(|p| p.cell) != Some(cell_id) {
                    continue;
                }
                pin_ports.push((*name, net_id));
            }
            PortType::InOut => {
                let is_driver =
                    net.driver_cell_port().map(|p| p.cell) == Some(cell_id);
                if is_driver && net.num_users() == 0 {
                    continue;
                }
                pin_ports.push((*name, net_id));
            }
            PortType::In => {
                pin_ports.push((*name, net_id));
            }
            _ => {}
        }
    }
    CellPinTemplate { pin_ports }
}

/// Dry-run check: would binding `root_id` at `root_bel` plus every cluster
/// child at its `root_loc + constr_offset` slot keep the shared-mux registry
/// consistent? Resolves child BELs the same way `place_cluster_children` will
/// later, so a `true` here means `place_cluster_children` won't bind anything
/// that conflicts with previously-placed cells.
///
/// Caller responsibility: only call when `root_bel` is itself available. The
/// registry is **not** mutated; on success the caller still needs to bind and
/// then `register.record(...)` for root and each child.
///
/// Background: `ring.rs` previously only checked `is_legal` for the cluster
/// root, leaving cluster children to be bound blindly by `place_cluster_children`
/// at their constraint-offset BELs. When two clusters' z=1 children landed on
/// BELs whose pin wires shared a routing node, the runtime check passed but
/// the post-legalization `verify_shared_mux_legality` fired. This helper
/// closes that gap so the ring search keeps looking when a cluster's
/// children would conflict.
/// Cached per-cell legality data: pin-port templates for the root and every
/// constrained cluster child, plus child constraint offsets resolved once.
/// One cache per movable cell, built before the ring search begins.
pub(crate) struct CellLegalityCache {
    pub root_template: CellPinTemplate,
    /// `(child_id, child_cell_type, constr_x, constr_y, constr_z, constr_abs_z, template)`
    pub children: Vec<(CellId, IdString, i32, i32, i32, bool, CellPinTemplate)>,
}

pub(crate) fn build_cell_legality_cache(ctx: &Context, root_id: CellId) -> CellLegalityCache {
    let root_template = build_cell_pin_template(ctx, root_id);
    let mut children: Vec<(CellId, IdString, i32, i32, i32, bool, CellPinTemplate)> = Vec::new();
    if let Some(cluster) = ctx.design.clusters.get(&root_id) {
        for &child_id in &cluster.constr_children {
            let child = ctx.design.cell(child_id);
            children.push((
                child_id,
                child.cell_type,
                child.constr_x,
                child.constr_y,
                child.constr_z,
                child.constr_abs_z,
                build_cell_pin_template(ctx, child_id),
            ));
        }
    }
    CellLegalityCache {
        root_template,
        children,
    }
}

/// Templated variant of `cluster_is_legal`: uses `CellLegalityCache` to skip
/// per-call cell-port enumeration. Logically identical; ~2-3× faster on
/// FPGA01 because is_legal is the hot loop and 30-40% of its cost was the
/// repeated `cell_pin_wires` builds.
pub(crate) fn cluster_is_legal_cached(
    ctx: &Context,
    registry: &DriverNodeRegistry,
    bel_by_loc: &FxHashMap<(IdString, i32, i32, i32), BelId>,
    cache: &CellLegalityCache,
    root_bel: BelId,
) -> bool {
    if !registry.is_legal_template(ctx, &cache.root_template, root_bel) {
        return false;
    }
    if cache.children.is_empty() {
        return true;
    }
    let root_loc = ctx.bel(root_bel).loc();
    for (_, child_type, cx, cy, cz, abs_z, child_template) in &cache.children {
        let want_x = root_loc.x + cx;
        let want_y = root_loc.y + cy;
        let want_z = if *abs_z { *cz } else { root_loc.z + cz };
        let key = (ctx.resolve_bucket(*child_type), want_x, want_y, want_z);
        let Some(&child_bel) = bel_by_loc.get(&key) else {
            return false;
        };
        if !ctx.bel(child_bel).is_available() {
            return false;
        }
        if !registry.is_legal_template(ctx, child_template, child_bel) {
            return false;
        }
    }
    true
}

pub(crate) fn cluster_is_legal(
    ctx: &Context,
    registry: &DriverNodeRegistry,
    bel_by_loc: &FxHashMap<(IdString, i32, i32, i32), BelId>,
    root_id: CellId,
    root_bel: BelId,
) -> bool {
    if !registry.is_legal(ctx, root_id, root_bel) {
        return false;
    }
    let Some(cluster) = ctx.design.clusters.get(&root_id) else {
        return true;
    };
    let root_loc = ctx.bel(root_bel).loc();
    let children: Vec<CellId> = cluster.constr_children.clone();
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
        // O(1) lookup against the caller-built (cell_type, x, y, z) -> BelId
        // index. Replaces the per-call full `bels_for_bucket(child_type).find(...)`
        // scan, which on FPGA01 was O(n_bels_for_bucket) ≈ 1M per cluster_is_legal
        // call, dominating ring-legalize wall time at ~1 trillion ops total.
        let key = (ctx.resolve_bucket(child_type), want_x, want_y, want_z);
        let Some(&child_bel) = bel_by_loc.get(&key) else {
            return false;
        };
        if !ctx.bel(child_bel).is_available() {
            return false;
        }
        if !registry.is_legal(ctx, child_id, child_bel) {
            return false;
        }
    }
    true
}

/// Build a `(bucket, x, y, z) -> BelId` index over every bucket+location.
/// Lets `cluster_is_legal` resolve a constraint-offset child slot in O(1)
/// instead of O(n_bels_for_bucket). The caller passes `cell_type`; the lookup
/// resolves to bucket via `ctx.resolve_bucket` so cell_type aliases work.
pub(crate) fn build_bel_by_loc(
    ctx: &Context,
) -> FxHashMap<(IdString, i32, i32, i32), BelId> {
    let mut idx: FxHashMap<(IdString, i32, i32, i32), BelId> = FxHashMap::default();
    for bucket_id in ctx.bel_buckets() {
        for bel in ctx.bels_for_bucket(bucket_id) {
            let loc = bel.loc();
            idx.insert((bucket_id, loc.x, loc.y, loc.z), bel.id());
        }
    }
    idx
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

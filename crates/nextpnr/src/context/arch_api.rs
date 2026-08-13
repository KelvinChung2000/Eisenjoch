//! The slice of nextpnr's Arch API that the ported placers and routers call,
//! implemented on top of eisenjoch's `Context`.
//!
//! Source: upstream YosysHQ nextpnr `main` @ `4d235150`, `common/kernel/arch_api.h`
//! and `common/kernel/base_arch.h`.
//!
//! eisenjoch's `Context` already covers most of the Arch API, but under its own
//! names and through view types (`bel.loc()`, `bel.bound_cell()`). The ported
//! algorithms are transcribed from C++ that calls `ctx->getBelLocation(bel)`
//! and friends, so this module supplies the missing calls under nextpnr's
//! spelling. Keeping the names means the ports stay diffable against the
//! originals.
//!
//! Where nextpnr's behaviour is arch-specific, this follows `BaseArch`'s
//! default -- that is the behaviour a himbaechel arch without a custom uarch
//! hook gets, and eisenjoch's chipdb is himbaechel-derived.

use crate::chipdb::{BelId, Loc, WireId};
use crate::common::IdString;
use crate::netlist::{CellId, CellPin, NetId};
use crate::timing::{DelayT, TimingPortClass};

use super::Context;

impl Context {
    // -----------------------------------------------------------------------
    // Grid geometry
    // -----------------------------------------------------------------------

    /// `getGridDimX`.
    #[inline]
    pub fn grid_dim_x(&self) -> i32 {
        self.chipdb().width()
    }

    /// `getGridDimY`.
    #[inline]
    pub fn grid_dim_y(&self) -> i32 {
        self.chipdb().height()
    }

    /// `getBelsByTile` -- every BEL in the tile at `(x, y)`.
    ///
    /// Returns an empty vector for out-of-range coordinates rather than
    /// panicking: the placers scan outwards from a cell's current location and
    /// routinely probe past the edge of the grid.
    pub fn bels_by_tile(&self, x: i32, y: i32) -> Vec<BelId> {
        if x < 0 || y < 0 || x >= self.grid_dim_x() || y >= self.grid_dim_y() {
            return Vec::new();
        }
        let tile = self.chipdb().tile_by_xy(x, y);
        if tile < 0 {
            return Vec::new();
        }
        let count = self.chipdb().tile_type(tile).bels.get().len();
        (0..count).map(|i| BelId::new(tile, i as i32)).collect()
    }

    /// `getTileBelDimZ` -- the number of BEL slots in the tile at `(x, y)`.
    #[inline]
    pub fn tile_bel_dim_z(&self, x: i32, y: i32) -> i32 {
        self.bels_by_tile(x, y).len() as i32
    }

    /// `getBelByLocation` -- the BEL at an exact `(x, y, z)`, if one exists.
    pub fn bel_by_location(&self, loc: Loc) -> Option<BelId> {
        self.bels_by_tile(loc.x, loc.y)
            .into_iter()
            .find(|&bel| self.chipdb().bel_loc(bel).z == loc.z)
    }

    // -----------------------------------------------------------------------
    // BEL properties and binding
    // -----------------------------------------------------------------------

    /// `getBelGlobalBuf` -- whether a BEL drives a global buffer network.
    ///
    /// `BaseArch` returns false unconditionally and himbaechel does not
    /// override it, so this is faithful rather than a stub. It matters because
    /// `get_net_metric` skips global-buffer nets entirely when scoring
    /// wirelength: an arch that grows a real notion of global buffers must
    /// override this or those nets will be costed as ordinary routing.
    #[inline]
    pub fn bel_global_buf(&self, _bel: BelId) -> bool {
        false
    }

    /// `getConflictingBelCell` -- the cell currently occupying `bel`.
    #[inline]
    pub fn conflicting_bel_cell(&self, bel: BelId) -> Option<CellId> {
        self.bel(bel).bound_cell().map(|c| c.id())
    }

    /// `isBelLocationValid` -- whether the tile containing `bel` is in a legal
    /// configuration given everything currently bound to it.
    ///
    /// nextpnr's `BaseArch` default is `true`; himbaechel delegates to the
    /// uarch. eisenjoch's equivalent of the uarch hook is
    /// `PlacerPlugin::check_placement_validity`, but the plugin manager is not
    /// owned by `Context`, so the plugin-aware form is
    /// [`Self::is_bel_location_valid_with`]. This bare version is the
    /// no-uarch default that plain himbaechel archs get.
    #[inline]
    pub fn is_bel_location_valid(&self, _bel: BelId) -> bool {
        true
    }

    /// `isBelLocationValid`, consulting a placer plugin as himbaechel consults
    /// its uarch.
    #[inline]
    pub fn is_bel_location_valid_with(
        &self,
        bel: BelId,
        check: &dyn Fn(&Context, BelId) -> bool,
    ) -> bool {
        check(self, bel)
    }

    // -----------------------------------------------------------------------
    // Delays
    // -----------------------------------------------------------------------

    /// `getDelayNS` -- picoseconds to nanoseconds.
    ///
    /// nextpnr returns a `double` here and the placers feed it straight into
    /// `std::exp`, so the width matters for the cost function.
    #[inline]
    pub fn delay_ns(&self, delay: DelayT) -> f64 {
        delay as f64 / 1000.0
    }

    /// `predictArcDelay` -- the estimated delay of one net arc, from the
    /// driver's BEL to this user's BEL, before routing exists.
    ///
    /// Returns 0 when either end is unplaced, matching nextpnr's behaviour of
    /// treating an unplaced arc as costless rather than infinite.
    pub fn predict_arc_delay(&self, net: NetId, user: CellPin) -> DelayT {
        let net_info = self.design.net(net);
        if !net_info.driver.is_valid() {
            return 0;
        }
        let Some(driver_bel) = self.design.cell(net_info.driver.cell).bel else {
            return 0;
        };
        let Some(user_bel) = self.design.cell(user.cell).bel else {
            return 0;
        };
        self.predict_bel_delay(driver_bel, user_bel)
    }

    /// `predictDelay` between two BELs -- the same Manhattan estimate
    /// [`Context::estimate_delay`] applies to wires, lifted to BEL endpoints.
    #[inline]
    pub fn predict_bel_delay(&self, src: BelId, dst: BelId) -> DelayT {
        let (sx, sy) = self.chipdb().tile_xy(src.tile());
        let (dx, dy) = self.chipdb().tile_xy(dst.tile());
        // Kept in step with `context::timing`'s DELAY_SCALE so placement-time
        // and routing-time estimates agree.
        ((sx - dx).abs() + (sy - dy).abs()) * 10
    }

    /// `getPortTimingClass` -- the timing role of a cell port.
    ///
    /// nextpnr also hands back a clock-argument count; no caller in the ported
    /// placers uses it, so it is omitted. Ports with no timing data, or on a
    /// chipdb with no speed grade loaded, come back `Ignore` -- which is what
    /// `get_net_metric` needs to skip untimed nets rather than cost them at
    /// zero slack.
    pub fn port_timing_class(&self, cell: CellId, port: IdString) -> TimingPortClass {
        let Some(speed_grade) = self.speed_grade() else {
            return TimingPortClass::Ignore;
        };
        let info = self.design.cell(cell);
        let Some(port_info) = info.ports.get(&port) else {
            return TimingPortClass::Ignore;
        };
        let type_idx = info
            .timing_index
            .map(|ti| ti.0 as usize)
            .or_else(|| {
                self.chipdb()
                    .cell_timing_index(speed_grade, info.cell_type.index())
            });
        let Some(type_idx) = type_idx else {
            return TimingPortClass::Ignore;
        };
        self.chipdb()
            .port_timing_class(speed_grade, type_idx, port.index(), port_info.port_type)
    }

    /// `setting<bool>` -- read a boolean run setting, defaulting to false.
    ///
    /// Settings arrive as `Property`, which has no native boolean, so an
    /// integer is truthy when non-zero and a string when it reads "1", "true"
    /// or "yes".
    pub fn setting_bool(&self, name: &str) -> bool {
        self.setting_bool_or(name, false)
    }

    /// `setting<bool>` with an explicit default for an absent key.
    ///
    /// nextpnr seeds several settings in `command.cc` before the placers run,
    /// so "absent" there does not mean "false". Callers that mirror one of
    /// those settings must pass nextpnr's seeded value as the default -- see
    /// [`Self::timing_driven`].
    pub fn setting_bool_or(&self, name: &str, default: bool) -> bool {
        let key = self.id(name);
        let Some(prop) = self.settings().get(&key) else {
            return default;
        };
        if let Some(v) = prop.as_int() {
            return v != 0;
        }
        matches!(
            prop.as_str().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    }

    /// The `timing_driven` setting, defaulting to **true**.
    ///
    /// nextpnr's `command.cc` seeds `timing_driven = true` unless `--no-tmdriv`
    /// was passed (upstream `4d235150`, `command.cc:537`), so a real nextpnr run
    /// is timing-driven by default. Reading the raw setting and defaulting to
    /// false would quietly make the baseline pure-HPWL and bias every
    /// comparison against it.
    #[inline]
    pub fn timing_driven(&self) -> bool {
        self.setting_bool_or("timing_driven", true)
    }

    // -----------------------------------------------------------------------
    // Routing geometry
    // -----------------------------------------------------------------------

    /// `getRouteBoundingBox` -- the box a net's routing is allowed to occupy,
    /// as the tight bounding box of the two endpoints.
    ///
    /// Routers widen this themselves; nextpnr's base implementation does not.
    pub fn route_bounding_box(&self, src: WireId, dst: WireId) -> BoundingBox {
        let (sx, sy) = self.chipdb().tile_xy(src.tile());
        let (dx, dy) = self.chipdb().tile_xy(dst.tile());
        BoundingBox {
            x0: sx.min(dx),
            y0: sy.min(dy),
            x1: sx.max(dx),
            y1: sy.max(dy),
        }
    }

    // -----------------------------------------------------------------------
    // Clusters
    // -----------------------------------------------------------------------

    /// `getClusterOffset` -- a constrained child's placement offset from its
    /// cluster root.
    ///
    /// Read from the cell's own `constr_*` fields rather than derived from
    /// current BEL positions, so it is meaningful before placement and while a
    /// cluster is mid-move. A cell that is not a constrained child has a zero
    /// offset.
    pub fn cluster_offset(&self, cell: CellId) -> Loc {
        let info = self.design.cell(cell);
        match info.cluster {
            Some(root) if root != cell => Loc::new(info.constr_x, info.constr_y, info.constr_z),
            _ => Loc::new(0, 0, 0),
        }
    }

    /// `getClusterRootCell`.
    #[inline]
    pub fn cluster_root_cell(&self, cluster: CellId) -> CellId {
        self.design.cell(cluster).cluster.unwrap_or(cluster)
    }

    /// `BaseArch::getClusterPlacement` -- where every member of `cluster` would
    /// land if its root sat on `root_bel`.
    ///
    /// Returns `None` if any member has no BEL at its constrained offset *or*
    /// that BEL cannot host the member's cell type. Callers rely on this: it is
    /// how the constraint legaliser rejects a candidate root position without
    /// binding anything. Checking only for a BEL's existence would be too
    /// permissive and would surface as mysterious legaliser behaviour later.
    ///
    /// A root with `constr_abs_z` is first coerced onto its absolute z and
    /// re-resolved, so the whole cluster hangs off the corrected root. Children
    /// with `constr_abs_z` take their z absolutely rather than relative to the
    /// root -- that is how a cluster spans fixed BEL slots within each tile.
    pub fn cluster_placement(
        &self,
        cluster: CellId,
        root_bel: BelId,
    ) -> Option<Vec<(CellId, BelId)>> {
        let root = self.cluster_root_cell(cluster);
        let root_info = self.design.cell(root);
        let mut root_bel = root_bel;
        let mut root_loc = self.chipdb().bel_loc(root_bel);

        if root_info.constr_abs_z {
            root_loc.z = root_info.constr_z;
            root_bel = self.bel_by_location(root_loc)?;
            if !self.bel(root_bel).is_valid_for_cell_type(root_info.cell_type) {
                return None;
            }
        }

        let mut placement = vec![(root, root_bel)];

        let Some(cluster_data) = self.design.clusters.get(&root) else {
            return Some(placement);
        };

        for &child in &cluster_data.constr_children {
            let info = self.design.cell(child);
            let want = Loc::new(
                root_loc.x + info.constr_x,
                root_loc.y + info.constr_y,
                if info.constr_abs_z {
                    info.constr_z
                } else {
                    root_loc.z + info.constr_z
                },
            );
            let child_bel = self.bel_by_location(want)?;
            if !self.bel(child_bel).is_valid_for_cell_type(info.cell_type) {
                return None;
            }
            placement.push((child, child_bel));
        }

        Some(placement)
    }
}

/// nextpnr's `BoundingBox` -- inclusive on all four edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl BoundingBox {
    /// Whether `(x, y)` falls inside the box.
    #[inline]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    /// Grow the box by `d` tiles on every side.
    #[inline]
    pub fn expand(&self, d: i32) -> Self {
        Self {
            x0: self.x0 - d,
            y0: self.y0 - d,
            x1: self.x1 + d,
            y1: self.y1 + d,
        }
    }
}

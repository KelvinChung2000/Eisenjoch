//! Netlist data structures for the nextpnr-rust FPGA place-and-route tool.
//!
//! This crate provides arena-indexed storage for cells (logic elements) and nets
//! (wires connecting them). Instead of pointer-based storage, cells and nets are
//! stored in `Vec`-based arenas indexed by `CellIdx` and `NetIdx` newtypes. This
//! provides memory safety and cache-friendliness.
//!
//! The central type is [`Design`], which owns all cells, nets, and hierarchy info.

use npnr_types::{BelId, DelayT, IdString, PipId, PlaceStrength, PortType, Property, WireId};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Index types
// ---------------------------------------------------------------------------

/// Index into the cell arena (`Design::cell_store`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellIdx(pub u32);

impl CellIdx {
    /// Sentinel value meaning "no cell" / unconnected.
    pub const NONE: Self = Self(u32::MAX);

    /// Returns `true` if this index is the NONE sentinel.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns `true` if this index refers to a valid slot (not NONE).
    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Index into the net arena (`Design::net_store`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NetIdx(pub u32);

impl NetIdx {
    /// Sentinel value meaning "no net" / unconnected.
    pub const NONE: Self = Self(u32::MAX);

    /// Returns `true` if this index is the NONE sentinel.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns `true` if this index refers to a valid slot (not NONE).
    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// PortRef — reference to a port on a cell (used in net driver / users)
// ---------------------------------------------------------------------------

/// Reference to a port on a cell, used as the driver or a user of a net.
#[derive(Clone, Debug)]
pub struct PortRef {
    /// Index of the cell that owns the port.
    /// `CellIdx::NONE` means unconnected.
    pub cell: CellIdx,
    /// Port name on the cell.
    pub port: IdString,
    /// Timing budget in picoseconds (0 = unconstrained).
    pub budget: DelayT,
}

impl PortRef {
    /// Create an unconnected port reference.
    pub fn unconnected() -> Self {
        Self {
            cell: CellIdx::NONE,
            port: IdString::EMPTY,
            budget: 0,
        }
    }

    /// Returns `true` if this port reference points to a valid cell.
    #[inline]
    pub fn is_connected(&self) -> bool {
        self.cell.is_some()
    }
}

// ---------------------------------------------------------------------------
// PortInfo — port definition on a cell
// ---------------------------------------------------------------------------

/// Definition of a port on a cell.
#[derive(Clone, Debug)]
pub struct PortInfo {
    /// Name of this port.
    pub name: IdString,
    /// Direction of this port.
    pub port_type: PortType,
    /// Net connected to this port (`NetIdx::NONE` if unconnected).
    pub net: NetIdx,
    /// Index into the connected net's `users` list.
    /// `-1` if this port is the driver of the net.
    pub user_idx: i32,
}

impl PortInfo {
    /// Create a new port with the given name and type, initially unconnected.
    pub fn new(name: IdString, port_type: PortType) -> Self {
        Self {
            name,
            port_type,
            net: NetIdx::NONE,
            user_idx: -1,
        }
    }
}

// ---------------------------------------------------------------------------
// PipMap — routing tree entry
// ---------------------------------------------------------------------------

/// Entry in a net's routing tree: records which PIP drives a particular wire.
#[derive(Clone, Debug)]
pub struct PipMap {
    /// The PIP used to reach this wire.
    pub pip: PipId,
    /// Strength of this routing assignment.
    pub strength: PlaceStrength,
}

// ---------------------------------------------------------------------------
// CellInfo
// ---------------------------------------------------------------------------

/// Complete information about a single cell (logic element) in the design.
pub struct CellInfo {
    /// The cell's name (unique within the design).
    pub name: IdString,
    /// The cell's type (e.g. "LUT4", "FDRE").
    pub cell_type: IdString,
    /// Ports on this cell, keyed by port name.
    pub ports: FxHashMap<IdString, PortInfo>,
    /// Attributes (string key-value pairs from the netlist).
    pub attrs: FxHashMap<IdString, Property>,
    /// Parameters.
    pub params: FxHashMap<IdString, Property>,

    // -- Placement --
    /// The BEL this cell is placed on.
    pub bel: BelId,
    /// Strength of the placement.
    pub bel_strength: PlaceStrength,

    // -- Clustering --
    /// Root cell of the cluster this cell belongs to.
    /// Equal to the cell's own index if it is the root, or `CellIdx::NONE` if
    /// it is not part of any cluster.
    pub cluster: CellIdx,
    /// Next cell in the cluster linked list.
    pub cluster_next: CellIdx,
    /// Cluster port mapping: (this_port, cluster_port, offset).
    pub cluster_ports: Vec<(IdString, IdString, i32)>,

    // -- Region --
    /// Region constraint index, if any.
    pub region: Option<u32>,

    // -- Misc indices --
    /// Index used for flat-array iteration over live cells.
    pub flat_index: i32,
    /// Index into the timing data structures.
    pub timing_index: i32,

    /// Whether this cell is alive (has not been logically removed).
    pub alive: bool,
}

impl CellInfo {
    /// Create a new cell with the given name and type.
    /// All other fields are initialised to sensible defaults.
    pub fn new(name: IdString, cell_type: IdString) -> Self {
        Self {
            name,
            cell_type,
            ports: FxHashMap::default(),
            attrs: FxHashMap::default(),
            params: FxHashMap::default(),
            bel: BelId::INVALID,
            bel_strength: PlaceStrength::None,
            cluster: CellIdx::NONE,
            cluster_next: CellIdx::NONE,
            cluster_ports: Vec::new(),
            region: None,
            flat_index: -1,
            timing_index: -1,
            alive: true,
        }
    }

    /// Add a port to this cell. Returns `&mut PortInfo` for the newly added port.
    pub fn add_port(&mut self, name: IdString, port_type: PortType) -> &mut PortInfo {
        self.ports
            .entry(name)
            .or_insert_with(|| PortInfo::new(name, port_type))
    }

    /// Look up a port by name.
    pub fn port(&self, name: IdString) -> Option<&PortInfo> {
        self.ports.get(&name)
    }

    /// Look up a port by name (mutable).
    pub fn port_mut(&mut self, name: IdString) -> Option<&mut PortInfo> {
        self.ports.get_mut(&name)
    }
}

// ---------------------------------------------------------------------------
// NetInfo
// ---------------------------------------------------------------------------

/// Complete information about a single net in the design.
pub struct NetInfo {
    /// The net's name (unique within the design).
    pub name: IdString,
    /// The driver of this net.
    pub driver: PortRef,
    /// The users (sinks) of this net.
    pub users: Vec<PortRef>,
    /// Attributes.
    pub attrs: FxHashMap<IdString, Property>,
    /// Routing tree: maps each routed wire to the PIP that drives it.
    pub wires: FxHashMap<WireId, PipMap>,
    /// Clock constraint period in picoseconds (0 = unconstrained).
    pub clock_constraint: DelayT,
    /// Region constraint index, if any.
    pub region: Option<u32>,
    /// Whether this net is alive (has not been logically removed).
    pub alive: bool,
}

impl NetInfo {
    /// Create a new net with the given name.
    pub fn new(name: IdString) -> Self {
        Self {
            name,
            driver: PortRef::unconnected(),
            users: Vec::new(),
            attrs: FxHashMap::default(),
            wires: FxHashMap::default(),
            clock_constraint: 0,
            region: None,
            alive: true,
        }
    }

    /// Returns `true` if the net has a connected driver.
    #[inline]
    pub fn has_driver(&self) -> bool {
        self.driver.is_connected()
    }

    /// Returns the number of users (sinks) on this net.
    #[inline]
    pub fn num_users(&self) -> usize {
        self.users.len()
    }
}

// ---------------------------------------------------------------------------
// HierarchicalNet / HierarchicalCell
// ---------------------------------------------------------------------------

/// A net within the design hierarchy.
#[derive(Clone, Debug)]
pub struct HierarchicalNet {
    /// Name of this net in the hierarchical context.
    pub name: IdString,
    /// Corresponding flat net name.
    pub flat_net: IdString,
}

/// A cell within the design hierarchy.
pub struct HierarchicalCell {
    /// Name of this hierarchical cell.
    pub name: IdString,
    /// Type of this hierarchical cell.
    pub cell_type: IdString,
    /// Parent hierarchical cell name.
    pub parent: IdString,
    /// Full hierarchical path.
    pub fullpath: IdString,
    /// Sub-cell name -> hierarchical cell name.
    pub hier_cells: FxHashMap<IdString, IdString>,
    /// Sub-cell name -> flat cell name.
    pub leaf_cells: FxHashMap<IdString, IdString>,
    /// Nets in this hierarchical cell.
    pub nets: FxHashMap<IdString, HierarchicalNet>,
}

impl HierarchicalCell {
    /// Create a new hierarchical cell with the given name and type.
    pub fn new(name: IdString, cell_type: IdString) -> Self {
        Self {
            name,
            cell_type,
            parent: IdString::EMPTY,
            fullpath: IdString::EMPTY,
            hier_cells: FxHashMap::default(),
            leaf_cells: FxHashMap::default(),
            nets: FxHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Design — the top-level container
// ---------------------------------------------------------------------------

/// The top-level design, owning all cells, nets, and hierarchy information.
///
/// Cells and nets are stored in `Vec`-based arenas for cache-friendliness.
/// Name-to-index lookups use `FxHashMap` for fast access with integer-like keys.
pub struct Design {
    /// Cell name -> arena index.
    pub cells: FxHashMap<IdString, CellIdx>,
    /// Cell arena storage.
    pub cell_store: Vec<CellInfo>,

    /// Net name -> arena index.
    pub nets: FxHashMap<IdString, NetIdx>,
    /// Net arena storage.
    pub net_store: Vec<NetInfo>,

    /// Hierarchical cells keyed by name.
    pub hierarchy: FxHashMap<IdString, HierarchicalCell>,

    /// Name of the top module.
    pub top_module: IdString,
}

impl Design {
    /// Create a new, empty design.
    pub fn new() -> Self {
        Self {
            cells: FxHashMap::default(),
            cell_store: Vec::new(),
            nets: FxHashMap::default(),
            net_store: Vec::new(),
            hierarchy: FxHashMap::default(),
            top_module: IdString::EMPTY,
        }
    }

    // -- Cell operations ---------------------------------------------------

    /// Add a new cell with the given name and type.
    ///
    /// Returns the index of the newly created cell.
    ///
    /// # Panics
    ///
    /// Panics if a cell with the same name already exists.
    pub fn add_cell(&mut self, name: IdString, cell_type: IdString) -> CellIdx {
        assert!(
            !self.cells.contains_key(&name),
            "cell already exists in design"
        );
        let idx = CellIdx(self.cell_store.len() as u32);
        self.cell_store.push(CellInfo::new(name, cell_type));
        self.cells.insert(name, idx);
        idx
    }

    /// Get a shared reference to a cell by its arena index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    #[inline]
    pub fn cell(&self, idx: CellIdx) -> &CellInfo {
        &self.cell_store[idx.0 as usize]
    }

    /// Get a mutable reference to a cell by its arena index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    #[inline]
    pub fn cell_mut(&mut self, idx: CellIdx) -> &mut CellInfo {
        &mut self.cell_store[idx.0 as usize]
    }

    /// Look up a cell index by name.
    pub fn cell_by_name(&self, name: IdString) -> Option<CellIdx> {
        self.cells.get(&name).copied()
    }

    /// Mark a cell as removed.
    ///
    /// The cell is **not** physically removed from the arena (that would
    /// invalidate indices). Instead it is marked as dead and its name is
    /// removed from the lookup map.
    pub fn remove_cell(&mut self, name: IdString) {
        if let Some(idx) = self.cells.remove(&name) {
            self.cell_store[idx.0 as usize].alive = false;
        }
    }

    // -- Net operations ----------------------------------------------------

    /// Add a new net with the given name.
    ///
    /// Returns the index of the newly created net.
    ///
    /// # Panics
    ///
    /// Panics if a net with the same name already exists.
    pub fn add_net(&mut self, name: IdString) -> NetIdx {
        assert!(
            !self.nets.contains_key(&name),
            "net already exists in design"
        );
        let idx = NetIdx(self.net_store.len() as u32);
        self.net_store.push(NetInfo::new(name));
        self.nets.insert(name, idx);
        idx
    }

    /// Get a shared reference to a net by its arena index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    #[inline]
    pub fn net(&self, idx: NetIdx) -> &NetInfo {
        &self.net_store[idx.0 as usize]
    }

    /// Get a mutable reference to a net by its arena index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    #[inline]
    pub fn net_mut(&mut self, idx: NetIdx) -> &mut NetInfo {
        &mut self.net_store[idx.0 as usize]
    }

    /// Look up a net index by name.
    pub fn net_by_name(&self, name: IdString) -> Option<NetIdx> {
        self.nets.get(&name).copied()
    }

    /// Mark a net as removed.
    ///
    /// The net is **not** physically removed from the arena. Instead it is
    /// marked as dead and its name is removed from the lookup map.
    pub fn remove_net(&mut self, name: IdString) {
        if let Some(idx) = self.nets.remove(&name) {
            self.net_store[idx.0 as usize].alive = false;
        }
    }
}

impl Default for Design {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use npnr_types::IdStringPool;

    /// Helper: create a pool and intern some names.
    fn make_pool() -> IdStringPool {
        IdStringPool::new()
    }

    // =====================================================================
    // CellIdx / NetIdx constants and basic properties
    // =====================================================================

    #[test]
    fn cell_idx_none_is_max() {
        assert_eq!(CellIdx::NONE.0, u32::MAX);
        assert!(CellIdx::NONE.is_none());
        assert!(!CellIdx::NONE.is_some());
    }

    #[test]
    fn net_idx_none_is_max() {
        assert_eq!(NetIdx::NONE.0, u32::MAX);
        assert!(NetIdx::NONE.is_none());
        assert!(!NetIdx::NONE.is_some());
    }

    #[test]
    fn cell_idx_zero_is_some() {
        let idx = CellIdx(0);
        assert!(idx.is_some());
        assert!(!idx.is_none());
    }

    #[test]
    fn net_idx_zero_is_some() {
        let idx = NetIdx(0);
        assert!(idx.is_some());
        assert!(!idx.is_none());
    }

    #[test]
    fn cell_idx_equality_and_hashing() {
        use std::collections::HashSet;
        let a = CellIdx(1);
        let b = CellIdx(1);
        let c = CellIdx(2);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn net_idx_equality_and_hashing() {
        use std::collections::HashSet;
        let a = NetIdx(10);
        let b = NetIdx(10);
        let c = NetIdx(20);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn cell_idx_copy_semantics() {
        let a = CellIdx(5);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn net_idx_copy_semantics() {
        let a = NetIdx(5);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn cell_idx_debug() {
        let idx = CellIdx(42);
        let s = format!("{:?}", idx);
        assert!(s.contains("CellIdx"));
        assert!(s.contains("42"));
    }

    #[test]
    fn net_idx_debug() {
        let idx = NetIdx(99);
        let s = format!("{:?}", idx);
        assert!(s.contains("NetIdx"));
        assert!(s.contains("99"));
    }

    // =====================================================================
    // PortRef
    // =====================================================================

    #[test]
    fn port_ref_unconnected() {
        let pr = PortRef::unconnected();
        assert!(!pr.is_connected());
        assert!(pr.cell.is_none());
        assert_eq!(pr.budget, 0);
    }

    #[test]
    fn port_ref_connected() {
        let pool = make_pool();
        let port_name = pool.intern("A");
        let pr = PortRef {
            cell: CellIdx(0),
            port: port_name,
            budget: 100,
        };
        assert!(pr.is_connected());
        assert_eq!(pr.budget, 100);
    }

    // =====================================================================
    // PortInfo
    // =====================================================================

    #[test]
    fn port_info_new_defaults() {
        let pool = make_pool();
        let name = pool.intern("clk");
        let pi = PortInfo::new(name, PortType::In);
        assert_eq!(pi.name, name);
        assert_eq!(pi.port_type, PortType::In);
        assert!(pi.net.is_none());
        assert_eq!(pi.user_idx, -1);
    }

    // =====================================================================
    // CellInfo
    // =====================================================================

    #[test]
    fn cell_info_new_defaults() {
        let pool = make_pool();
        let name = pool.intern("lut0");
        let ctype = pool.intern("LUT4");
        let ci = CellInfo::new(name, ctype);

        assert_eq!(ci.name, name);
        assert_eq!(ci.cell_type, ctype);
        assert!(ci.ports.is_empty());
        assert!(ci.attrs.is_empty());
        assert!(ci.params.is_empty());
        assert!(!ci.bel.is_valid());
        assert_eq!(ci.bel_strength, PlaceStrength::None);
        assert!(ci.cluster.is_none());
        assert!(ci.cluster_next.is_none());
        assert!(ci.cluster_ports.is_empty());
        assert_eq!(ci.region, None);
        assert_eq!(ci.flat_index, -1);
        assert_eq!(ci.timing_index, -1);
        assert!(ci.alive);
    }

    #[test]
    fn cell_info_add_port() {
        let pool = make_pool();
        let name = pool.intern("ff0");
        let ctype = pool.intern("FDRE");
        let mut ci = CellInfo::new(name, ctype);

        let d_name = pool.intern("D");
        let q_name = pool.intern("Q");

        ci.add_port(d_name, PortType::In);
        ci.add_port(q_name, PortType::Out);

        assert_eq!(ci.ports.len(), 2);
        assert_eq!(ci.port(d_name).unwrap().port_type, PortType::In);
        assert_eq!(ci.port(q_name).unwrap().port_type, PortType::Out);
    }

    #[test]
    fn cell_info_add_port_idempotent() {
        let pool = make_pool();
        let name = pool.intern("cell");
        let ctype = pool.intern("TYPE");
        let mut ci = CellInfo::new(name, ctype);

        let port_name = pool.intern("A");
        ci.add_port(port_name, PortType::In);
        // Adding same port again should not overwrite
        ci.add_port(port_name, PortType::Out);
        // Should still be In (first insert wins with or_insert_with)
        assert_eq!(ci.port(port_name).unwrap().port_type, PortType::In);
        assert_eq!(ci.ports.len(), 1);
    }

    #[test]
    fn cell_info_port_mut() {
        let pool = make_pool();
        let name = pool.intern("cell");
        let ctype = pool.intern("TYPE");
        let mut ci = CellInfo::new(name, ctype);

        let port_name = pool.intern("A");
        ci.add_port(port_name, PortType::In);

        // Mutate via port_mut
        let pi = ci.port_mut(port_name).unwrap();
        pi.net = NetIdx(7);
        pi.user_idx = 3;

        assert_eq!(ci.port(port_name).unwrap().net, NetIdx(7));
        assert_eq!(ci.port(port_name).unwrap().user_idx, 3);
    }

    #[test]
    fn cell_info_nonexistent_port() {
        let pool = make_pool();
        let ci = CellInfo::new(pool.intern("x"), pool.intern("Y"));
        assert!(ci.port(pool.intern("Z")).is_none());
    }

    #[test]
    fn cell_info_attrs_and_params() {
        let pool = make_pool();
        let mut ci = CellInfo::new(pool.intern("c"), pool.intern("T"));

        let key = pool.intern("INIT");
        ci.params.insert(key, Property::bit_vector("1010"));
        assert_eq!(ci.params.get(&key).unwrap().as_int(), Some(0b1010));

        let attr_key = pool.intern("LOC");
        ci.attrs.insert(attr_key, Property::string("SLICE_X0Y0"));
        assert_eq!(ci.attrs.get(&attr_key).unwrap().as_str(), "SLICE_X0Y0");
    }

    // =====================================================================
    // NetInfo
    // =====================================================================

    #[test]
    fn net_info_new_defaults() {
        let pool = make_pool();
        let name = pool.intern("net0");
        let ni = NetInfo::new(name);

        assert_eq!(ni.name, name);
        assert!(!ni.has_driver());
        assert_eq!(ni.num_users(), 0);
        assert!(ni.attrs.is_empty());
        assert!(ni.wires.is_empty());
        assert_eq!(ni.clock_constraint, 0);
        assert_eq!(ni.region, None);
        assert!(ni.alive);
    }

    #[test]
    fn net_info_set_driver() {
        let pool = make_pool();
        let mut ni = NetInfo::new(pool.intern("n"));

        let port_name = pool.intern("Q");
        ni.driver = PortRef {
            cell: CellIdx(0),
            port: port_name,
            budget: 0,
        };
        assert!(ni.has_driver());
        assert_eq!(ni.driver.cell, CellIdx(0));
        assert_eq!(ni.driver.port, port_name);
    }

    #[test]
    fn net_info_add_users() {
        let pool = make_pool();
        let mut ni = NetInfo::new(pool.intern("n"));

        let port_a = pool.intern("A");
        let port_b = pool.intern("B");

        ni.users.push(PortRef {
            cell: CellIdx(1),
            port: port_a,
            budget: 50,
        });
        ni.users.push(PortRef {
            cell: CellIdx(2),
            port: port_b,
            budget: 75,
        });

        assert_eq!(ni.num_users(), 2);
        assert_eq!(ni.users[0].cell, CellIdx(1));
        assert_eq!(ni.users[0].budget, 50);
        assert_eq!(ni.users[1].cell, CellIdx(2));
        assert_eq!(ni.users[1].budget, 75);
    }

    #[test]
    fn net_info_routing_tree() {
        let pool = make_pool();
        let mut ni = NetInfo::new(pool.intern("n"));

        let wire = WireId::new(0, 5);
        let pip = PipId::new(0, 10);

        ni.wires.insert(
            wire,
            PipMap {
                pip,
                strength: PlaceStrength::Placer,
            },
        );

        assert_eq!(ni.wires.len(), 1);
        let pm = ni.wires.get(&wire).unwrap();
        assert_eq!(pm.pip, pip);
        assert_eq!(pm.strength, PlaceStrength::Placer);
    }

    #[test]
    fn net_info_clock_constraint() {
        let pool = make_pool();
        let mut ni = NetInfo::new(pool.intern("clk_net"));
        ni.clock_constraint = 10000; // 10 ns
        assert_eq!(ni.clock_constraint, 10000);
    }

    // =====================================================================
    // PipMap
    // =====================================================================

    #[test]
    fn pip_map_clone() {
        let pm = PipMap {
            pip: PipId::new(1, 2),
            strength: PlaceStrength::Fixed,
        };
        let pm2 = pm.clone();
        assert_eq!(pm2.pip, PipId::new(1, 2));
        assert_eq!(pm2.strength, PlaceStrength::Fixed);
    }

    // =====================================================================
    // HierarchicalNet / HierarchicalCell
    // =====================================================================

    #[test]
    fn hierarchical_net_basic() {
        let pool = make_pool();
        let hn = HierarchicalNet {
            name: pool.intern("sub/clk"),
            flat_net: pool.intern("clk"),
        };
        assert_eq!(hn.name, pool.intern("sub/clk"));
        assert_eq!(hn.flat_net, pool.intern("clk"));
    }

    #[test]
    fn hierarchical_cell_new() {
        let pool = make_pool();
        let name = pool.intern("inst0");
        let ctype = pool.intern("MOD");
        let hc = HierarchicalCell::new(name, ctype);

        assert_eq!(hc.name, name);
        assert_eq!(hc.cell_type, ctype);
        assert!(hc.parent.is_empty());
        assert!(hc.fullpath.is_empty());
        assert!(hc.hier_cells.is_empty());
        assert!(hc.leaf_cells.is_empty());
        assert!(hc.nets.is_empty());
    }

    #[test]
    fn hierarchical_cell_populate() {
        let pool = make_pool();
        let mut hc = HierarchicalCell::new(pool.intern("top"), pool.intern("TOP"));

        hc.parent = pool.intern("");
        hc.fullpath = pool.intern("top");

        let sub_name = pool.intern("sub_inst");
        let sub_hier = pool.intern("top/sub_inst");
        hc.hier_cells.insert(sub_name, sub_hier);

        let leaf_name = pool.intern("lut0");
        let flat_name = pool.intern("top/lut0");
        hc.leaf_cells.insert(leaf_name, flat_name);

        let net_name = pool.intern("n0");
        hc.nets.insert(
            net_name,
            HierarchicalNet {
                name: net_name,
                flat_net: pool.intern("top/n0"),
            },
        );

        assert_eq!(hc.hier_cells.len(), 1);
        assert_eq!(hc.leaf_cells.len(), 1);
        assert_eq!(hc.nets.len(), 1);
    }

    // =====================================================================
    // Design — cell operations
    // =====================================================================

    #[test]
    fn design_new_is_empty() {
        let d = Design::new();
        assert!(d.cells.is_empty());
        assert!(d.cell_store.is_empty());
        assert!(d.nets.is_empty());
        assert!(d.net_store.is_empty());
        assert!(d.hierarchy.is_empty());
        assert!(d.top_module.is_empty());
    }

    #[test]
    fn design_default_is_empty() {
        let d = Design::default();
        assert!(d.cells.is_empty());
    }

    #[test]
    fn design_add_cell() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("lut0");
        let ctype = pool.intern("LUT4");
        let idx = d.add_cell(name, ctype);

        assert_eq!(idx, CellIdx(0));
        assert_eq!(d.cell_store.len(), 1);
        assert_eq!(d.cells.len(), 1);
        assert_eq!(d.cell(idx).name, name);
        assert_eq!(d.cell(idx).cell_type, ctype);
        assert!(d.cell(idx).alive);
    }

    #[test]
    fn design_add_multiple_cells() {
        let pool = make_pool();
        let mut d = Design::new();

        let idx0 = d.add_cell(pool.intern("a"), pool.intern("LUT4"));
        let idx1 = d.add_cell(pool.intern("b"), pool.intern("FDRE"));
        let idx2 = d.add_cell(pool.intern("c"), pool.intern("IBUF"));

        assert_eq!(idx0, CellIdx(0));
        assert_eq!(idx1, CellIdx(1));
        assert_eq!(idx2, CellIdx(2));
        assert_eq!(d.cell_store.len(), 3);
    }

    #[test]
    #[should_panic(expected = "cell already exists")]
    fn design_add_duplicate_cell_panics() {
        let pool = make_pool();
        let mut d = Design::new();
        let name = pool.intern("dup");
        d.add_cell(name, pool.intern("T"));
        d.add_cell(name, pool.intern("T"));
    }

    #[test]
    fn design_cell_by_name() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("cell0");
        let idx = d.add_cell(name, pool.intern("LUT4"));
        assert_eq!(d.cell_by_name(name), Some(idx));

        let missing = pool.intern("nonexistent");
        assert_eq!(d.cell_by_name(missing), None);
    }

    #[test]
    fn design_cell_mut() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("cell0");
        let idx = d.add_cell(name, pool.intern("LUT4"));

        // Mutate cell
        d.cell_mut(idx).bel = BelId::new(1, 2);
        d.cell_mut(idx).bel_strength = PlaceStrength::Fixed;

        assert_eq!(d.cell(idx).bel, BelId::new(1, 2));
        assert_eq!(d.cell(idx).bel_strength, PlaceStrength::Fixed);
    }

    #[test]
    fn design_remove_cell() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("to_remove");
        let idx = d.add_cell(name, pool.intern("LUT4"));

        assert!(d.cell(idx).alive);
        assert_eq!(d.cell_by_name(name), Some(idx));

        d.remove_cell(name);

        // Cell is still in the arena but marked dead
        assert!(!d.cell(idx).alive);
        // Name lookup no longer finds it
        assert_eq!(d.cell_by_name(name), None);
        // Arena size unchanged
        assert_eq!(d.cell_store.len(), 1);
    }

    #[test]
    fn design_remove_nonexistent_cell_is_noop() {
        let pool = make_pool();
        let mut d = Design::new();
        // Should not panic
        d.remove_cell(pool.intern("ghost"));
    }

    #[test]
    fn design_indices_stable_after_remove() {
        let pool = make_pool();
        let mut d = Design::new();

        let name_a = pool.intern("a");
        let name_b = pool.intern("b");
        let name_c = pool.intern("c");

        let idx_a = d.add_cell(name_a, pool.intern("T"));
        let idx_b = d.add_cell(name_b, pool.intern("T"));
        let idx_c = d.add_cell(name_c, pool.intern("T"));

        // Remove the middle cell
        d.remove_cell(name_b);

        // Indices for a and c should still work
        assert_eq!(d.cell(idx_a).name, name_a);
        assert!(d.cell(idx_a).alive);
        assert_eq!(d.cell(idx_b).name, name_b);
        assert!(!d.cell(idx_b).alive);
        assert_eq!(d.cell(idx_c).name, name_c);
        assert!(d.cell(idx_c).alive);
    }

    // =====================================================================
    // Design — net operations
    // =====================================================================

    #[test]
    fn design_add_net() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("net0");
        let idx = d.add_net(name);

        assert_eq!(idx, NetIdx(0));
        assert_eq!(d.net_store.len(), 1);
        assert_eq!(d.nets.len(), 1);
        assert_eq!(d.net(idx).name, name);
        assert!(d.net(idx).alive);
    }

    #[test]
    fn design_add_multiple_nets() {
        let pool = make_pool();
        let mut d = Design::new();

        let idx0 = d.add_net(pool.intern("n0"));
        let idx1 = d.add_net(pool.intern("n1"));
        let idx2 = d.add_net(pool.intern("n2"));

        assert_eq!(idx0, NetIdx(0));
        assert_eq!(idx1, NetIdx(1));
        assert_eq!(idx2, NetIdx(2));
        assert_eq!(d.net_store.len(), 3);
    }

    #[test]
    #[should_panic(expected = "net already exists")]
    fn design_add_duplicate_net_panics() {
        let pool = make_pool();
        let mut d = Design::new();
        let name = pool.intern("dup");
        d.add_net(name);
        d.add_net(name);
    }

    #[test]
    fn design_net_by_name() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("net0");
        let idx = d.add_net(name);
        assert_eq!(d.net_by_name(name), Some(idx));

        assert_eq!(d.net_by_name(pool.intern("missing")), None);
    }

    #[test]
    fn design_net_mut() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("net0");
        let idx = d.add_net(name);

        d.net_mut(idx).clock_constraint = 5000;
        d.net_mut(idx).region = Some(3);

        assert_eq!(d.net(idx).clock_constraint, 5000);
        assert_eq!(d.net(idx).region, Some(3));
    }

    #[test]
    fn design_remove_net() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("to_remove");
        let idx = d.add_net(name);

        assert!(d.net(idx).alive);
        d.remove_net(name);

        assert!(!d.net(idx).alive);
        assert_eq!(d.net_by_name(name), None);
        assert_eq!(d.net_store.len(), 1);
    }

    #[test]
    fn design_remove_nonexistent_net_is_noop() {
        let pool = make_pool();
        let mut d = Design::new();
        d.remove_net(pool.intern("ghost"));
    }

    #[test]
    fn design_net_indices_stable_after_remove() {
        let pool = make_pool();
        let mut d = Design::new();

        let name_a = pool.intern("na");
        let name_b = pool.intern("nb");
        let name_c = pool.intern("nc");

        let idx_a = d.add_net(name_a);
        let idx_b = d.add_net(name_b);
        let idx_c = d.add_net(name_c);

        d.remove_net(name_b);

        assert!(d.net(idx_a).alive);
        assert!(!d.net(idx_b).alive);
        assert!(d.net(idx_c).alive);
        assert_eq!(d.net(idx_a).name, name_a);
        assert_eq!(d.net(idx_c).name, name_c);
    }

    // =====================================================================
    // Design — integrated cell + net wiring
    // =====================================================================

    #[test]
    fn design_wire_driver_and_user() {
        let pool = make_pool();
        let mut d = Design::new();

        // Create two cells
        let drv_name = pool.intern("driver_cell");
        let usr_name = pool.intern("user_cell");
        let drv_idx = d.add_cell(drv_name, pool.intern("OBUF"));
        let usr_idx = d.add_cell(usr_name, pool.intern("IBUF"));

        // Add ports
        let q_port = pool.intern("Q");
        let a_port = pool.intern("A");
        d.cell_mut(drv_idx).add_port(q_port, PortType::Out);
        d.cell_mut(usr_idx).add_port(a_port, PortType::In);

        // Create net
        let net_name = pool.intern("wire0");
        let net_idx = d.add_net(net_name);

        // Wire driver
        d.net_mut(net_idx).driver = PortRef {
            cell: drv_idx,
            port: q_port,
            budget: 0,
        };
        d.cell_mut(drv_idx).port_mut(q_port).unwrap().net = net_idx;
        d.cell_mut(drv_idx).port_mut(q_port).unwrap().user_idx = -1;

        // Wire user
        let user_idx_in_net = d.net(net_idx).users.len() as i32;
        d.net_mut(net_idx).users.push(PortRef {
            cell: usr_idx,
            port: a_port,
            budget: 200,
        });
        d.cell_mut(usr_idx).port_mut(a_port).unwrap().net = net_idx;
        d.cell_mut(usr_idx).port_mut(a_port).unwrap().user_idx = user_idx_in_net;

        // Verify
        assert!(d.net(net_idx).has_driver());
        assert_eq!(d.net(net_idx).driver.cell, drv_idx);
        assert_eq!(d.net(net_idx).num_users(), 1);
        assert_eq!(d.net(net_idx).users[0].cell, usr_idx);
        assert_eq!(d.net(net_idx).users[0].budget, 200);
        assert_eq!(d.cell(drv_idx).port(q_port).unwrap().net, net_idx);
        assert_eq!(d.cell(usr_idx).port(a_port).unwrap().net, net_idx);
        assert_eq!(d.cell(usr_idx).port(a_port).unwrap().user_idx, 0);
    }

    #[test]
    fn design_routing_tree_multiple_wires() {
        let pool = make_pool();
        let mut d = Design::new();

        let net_idx = d.add_net(pool.intern("routed_net"));

        // Add several wires to the routing tree
        for i in 0..5 {
            let wire = WireId::new(0, i);
            let pip = PipId::new(0, i + 100);
            d.net_mut(net_idx).wires.insert(
                wire,
                PipMap {
                    pip,
                    strength: PlaceStrength::Placer,
                },
            );
        }

        assert_eq!(d.net(net_idx).wires.len(), 5);

        let wire3 = WireId::new(0, 3);
        let pm = d.net(net_idx).wires.get(&wire3).unwrap();
        assert_eq!(pm.pip, PipId::new(0, 103));
    }

    // =====================================================================
    // Design — hierarchy
    // =====================================================================

    #[test]
    fn design_hierarchy() {
        let pool = make_pool();
        let mut d = Design::new();

        let top_name = pool.intern("top");
        d.top_module = top_name;

        let hc = HierarchicalCell::new(top_name, pool.intern("TOP_MODULE"));
        d.hierarchy.insert(top_name, hc);

        assert_eq!(d.hierarchy.len(), 1);
        assert_eq!(d.hierarchy.get(&top_name).unwrap().name, top_name);
    }

    // =====================================================================
    // Design — clustering
    // =====================================================================

    #[test]
    fn design_cluster_linked_list() {
        let pool = make_pool();
        let mut d = Design::new();

        let root_idx = d.add_cell(pool.intern("root"), pool.intern("CARRY"));
        let child1_idx = d.add_cell(pool.intern("child1"), pool.intern("CARRY"));
        let child2_idx = d.add_cell(pool.intern("child2"), pool.intern("CARRY"));

        // Form cluster: root -> child1 -> child2
        d.cell_mut(root_idx).cluster = root_idx; // root points to itself
        d.cell_mut(root_idx).cluster_next = child1_idx;

        d.cell_mut(child1_idx).cluster = root_idx;
        d.cell_mut(child1_idx).cluster_next = child2_idx;

        d.cell_mut(child2_idx).cluster = root_idx;
        // child2 has no next (NONE by default)

        // Add cluster port mappings
        let ci_port = pool.intern("CI");
        let co_port = pool.intern("CO");
        d.cell_mut(child1_idx)
            .cluster_ports
            .push((ci_port, co_port, 1));

        // Verify cluster chain
        assert_eq!(d.cell(root_idx).cluster, root_idx);
        assert_eq!(d.cell(root_idx).cluster_next, child1_idx);
        assert_eq!(d.cell(child1_idx).cluster, root_idx);
        assert_eq!(d.cell(child1_idx).cluster_next, child2_idx);
        assert_eq!(d.cell(child2_idx).cluster, root_idx);
        assert!(d.cell(child2_idx).cluster_next.is_none());

        // Verify port mapping
        assert_eq!(d.cell(child1_idx).cluster_ports.len(), 1);
        assert_eq!(d.cell(child1_idx).cluster_ports[0].2, 1);
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn none_indices_are_not_confused_with_valid() {
        let pool = make_pool();
        let mut d = Design::new();

        // Add enough cells so that index 0 is valid
        let idx = d.add_cell(pool.intern("c0"), pool.intern("T"));
        assert_eq!(idx, CellIdx(0));
        assert!(idx.is_some());

        // NONE should never equal a valid index
        assert_ne!(CellIdx::NONE, idx);
        assert_ne!(NetIdx::NONE, NetIdx(0));
    }

    #[test]
    fn net_no_driver() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_net(pool.intern("floating"));
        assert!(!d.net(idx).has_driver());
    }

    #[test]
    fn net_unconnect_driver() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_net(pool.intern("n"));

        // Connect a driver
        d.net_mut(idx).driver = PortRef {
            cell: CellIdx(0),
            port: pool.intern("Q"),
            budget: 0,
        };
        assert!(d.net(idx).has_driver());

        // Disconnect it
        d.net_mut(idx).driver = PortRef::unconnected();
        assert!(!d.net(idx).has_driver());
    }

    #[test]
    fn cell_region_constraint() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_cell(pool.intern("c"), pool.intern("T"));

        assert_eq!(d.cell(idx).region, None);
        d.cell_mut(idx).region = Some(42);
        assert_eq!(d.cell(idx).region, Some(42));
    }

    #[test]
    fn net_region_constraint() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_net(pool.intern("n"));

        assert_eq!(d.net(idx).region, None);
        d.net_mut(idx).region = Some(7);
        assert_eq!(d.net(idx).region, Some(7));
    }

    #[test]
    fn cell_placement() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_cell(pool.intern("placed"), pool.intern("LUT4"));

        // Initially unplaced
        assert!(!d.cell(idx).bel.is_valid());
        assert_eq!(d.cell(idx).bel_strength, PlaceStrength::None);

        // Place it
        d.cell_mut(idx).bel = BelId::new(3, 7);
        d.cell_mut(idx).bel_strength = PlaceStrength::User;

        assert!(d.cell(idx).bel.is_valid());
        assert_eq!(d.cell(idx).bel.tile(), 3);
        assert_eq!(d.cell(idx).bel.index(), 7);
        assert_eq!(d.cell(idx).bel_strength, PlaceStrength::User);
    }

    #[test]
    fn design_large_scale_add() {
        let pool = make_pool();
        let mut d = Design::new();

        // Add many cells and nets
        let n = 1000;
        for i in 0..n {
            let cname = pool.intern(&format!("cell_{}", i));
            let nname = pool.intern(&format!("net_{}", i));
            let cidx = d.add_cell(cname, pool.intern("LUT4"));
            let nidx = d.add_net(nname);
            assert_eq!(cidx, CellIdx(i as u32));
            assert_eq!(nidx, NetIdx(i as u32));
        }

        assert_eq!(d.cell_store.len(), n);
        assert_eq!(d.net_store.len(), n);

        // Spot-check a few
        let check_name = pool.intern("cell_500");
        let check_idx = d.cell_by_name(check_name).unwrap();
        assert_eq!(d.cell(check_idx).name, check_name);

        let net_check = pool.intern("net_999");
        let net_check_idx = d.net_by_name(net_check).unwrap();
        assert_eq!(d.net(net_check_idx).name, net_check);
    }

    #[test]
    fn net_attrs() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_net(pool.intern("n"));

        let key = pool.intern("SRC");
        d.net_mut(idx)
            .attrs
            .insert(key, Property::string("module.v:42"));

        assert_eq!(
            d.net(idx).attrs.get(&key).unwrap().as_str(),
            "module.v:42"
        );
    }

    #[test]
    fn cell_flat_and_timing_index() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_cell(pool.intern("c"), pool.intern("T"));

        // Defaults
        assert_eq!(d.cell(idx).flat_index, -1);
        assert_eq!(d.cell(idx).timing_index, -1);

        d.cell_mut(idx).flat_index = 42;
        d.cell_mut(idx).timing_index = 7;

        assert_eq!(d.cell(idx).flat_index, 42);
        assert_eq!(d.cell(idx).timing_index, 7);
    }

    #[test]
    fn net_multiple_wire_routing() {
        let pool = make_pool();
        let mut d = Design::new();
        let idx = d.add_net(pool.intern("big_net"));

        // Simulate a routing tree with wires from different tiles
        let entries = vec![
            (WireId::new(0, 0), PipId::new(0, 10)),
            (WireId::new(0, 1), PipId::new(0, 11)),
            (WireId::new(1, 0), PipId::new(1, 20)),
            (WireId::new(1, 1), PipId::new(1, 21)),
            (WireId::new(2, 5), PipId::new(2, 50)),
        ];

        for (wire, pip) in &entries {
            d.net_mut(idx).wires.insert(
                *wire,
                PipMap {
                    pip: *pip,
                    strength: PlaceStrength::Placer,
                },
            );
        }

        assert_eq!(d.net(idx).wires.len(), 5);

        // Verify each wire maps to the correct pip
        for (wire, pip) in &entries {
            let pm = d.net(idx).wires.get(wire).unwrap();
            assert_eq!(pm.pip, *pip);
        }
    }

    #[test]
    fn remove_cell_then_add_with_same_name() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("reused");
        let _idx1 = d.add_cell(name, pool.intern("LUT4"));
        d.remove_cell(name);

        // Should be able to add a cell with the same name again
        // (it gets a new index)
        let idx2 = d.add_cell(name, pool.intern("FDRE"));
        assert_eq!(idx2, CellIdx(1)); // new slot in arena
        assert!(d.cell(idx2).alive);
        assert_eq!(d.cell(idx2).cell_type, pool.intern("FDRE"));
    }

    #[test]
    fn remove_net_then_add_with_same_name() {
        let pool = make_pool();
        let mut d = Design::new();

        let name = pool.intern("reused_net");
        let _idx1 = d.add_net(name);
        d.remove_net(name);

        let idx2 = d.add_net(name);
        assert_eq!(idx2, NetIdx(1));
        assert!(d.net(idx2).alive);
    }

    #[test]
    fn port_ref_clone() {
        let pool = make_pool();
        let pr = PortRef {
            cell: CellIdx(5),
            port: pool.intern("D"),
            budget: 123,
        };
        let pr2 = pr.clone();
        assert_eq!(pr2.cell, CellIdx(5));
        assert_eq!(pr2.budget, 123);
    }

    #[test]
    fn port_info_clone() {
        let pool = make_pool();
        let pi = PortInfo {
            name: pool.intern("CLK"),
            port_type: PortType::In,
            net: NetIdx(3),
            user_idx: 1,
        };
        let pi2 = pi.clone();
        assert_eq!(pi2.name, pool.intern("CLK"));
        assert_eq!(pi2.port_type, PortType::In);
        assert_eq!(pi2.net, NetIdx(3));
        assert_eq!(pi2.user_idx, 1);
    }
}

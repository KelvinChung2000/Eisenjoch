//! Integration tests for the netlist module.
//!
//! Tests cover CellIdx/NetIdx sentinels, CellPin/CellInfo/NetInfo
//! construction and mutation, Design arena operations (add/remove/lookup),
//! wiring, hierarchy, clustering, and edge cases.

use nextpnr::netlist::*;
use nextpnr::types::*;

#[allow(non_snake_case)]
fn CellIdx(raw: u32) -> nextpnr::netlist::CellId {
    nextpnr::netlist::CellId::from_raw(raw)
}

#[allow(non_snake_case)]
fn NetIdx(raw: u32) -> nextpnr::netlist::NetId {
    nextpnr::netlist::NetId::from_raw(raw)
}

/// Helper: create a pool and intern some names.
fn make_pool() -> IdStringPool {
    IdStringPool::new()
}

// =====================================================================
// CellIdx / NetIdx constants and basic properties
// =====================================================================

#[test]
fn cell_idx_none_is_max() {
    assert_eq!(CellId::NONE.raw(), u32::MAX);
    assert!(CellId::NONE.is_none());
    assert!(!CellId::NONE.is_some());
}

#[test]
fn net_idx_none_is_max() {
    assert_eq!(NetId::NONE.raw(), u32::MAX);
    assert!(NetId::NONE.is_none());
    assert!(!NetId::NONE.is_some());
}

#[test]
fn cell_idx_zero_is_some() {
    let idx = CellId::from_raw(0);
    assert!(idx.is_some());
    assert!(!idx.is_none());
}

#[test]
fn net_idx_zero_is_some() {
    let idx = NetId::from_raw(0);
    assert!(idx.is_some());
    assert!(!idx.is_none());
}

#[test]
fn cell_idx_equality_and_hashing() {
    use std::collections::HashSet;
    let a = CellId::from_raw(1);
    let b = CellId::from_raw(1);
    let c = CellId::from_raw(2);
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
    let a = NetId::from_raw(10);
    let b = NetId::from_raw(10);
    let c = NetId::from_raw(20);
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
    let a = CellId::from_raw(5);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn net_idx_copy_semantics() {
    let a = NetId::from_raw(5);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn cell_idx_debug() {
    let idx = CellId::from_raw(42);
    let s = format!("{:?}", idx);
    assert!(s.contains("CellId"));
    assert!(s.contains("42"));
}

#[test]
fn net_idx_debug() {
    let idx = NetId::from_raw(99);
    let s = format!("{:?}", idx);
    assert!(s.contains("NetId"));
    assert!(s.contains("99"));
}

// =====================================================================
// CellPin
// =====================================================================

#[test]
fn cell_pin_invalid_defaults() {
    let pin = CellPin::INVALID;
    assert!(!pin.is_connected());
    assert!(pin.cell.is_none());
    assert!(pin.port.is_empty());
}

#[test]
fn cell_pin_connected() {
    let pool = make_pool();
    let pin = CellPin::new(CellId::from_raw(0), pool.intern("A"));
    assert!(pin.is_connected());
    assert_eq!(pin.cell, CellIdx(0));
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
    assert_eq!(ci.num_ports(), 0);
    assert!(ci.attrs.is_empty());
    assert!(ci.params.is_empty());
    assert!(ci.bel.is_none());
    assert_eq!(ci.bel_strength, PlaceStrength::None);
    assert!(ci.cluster.is_none());
    assert_eq!(ci.region, None);
    assert_eq!(ci.flat_index, None);
    assert_eq!(ci.timing_index, None);
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

    assert_eq!(ci.num_ports(), 2);
    assert_eq!(ci.port_type(d_name), Some(PortType::In));
    assert_eq!(ci.port_type(q_name), Some(PortType::Out));
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
    assert_eq!(ci.port_type(port_name), Some(PortType::In));
    assert_eq!(ci.num_ports(), 1);
}

#[test]
fn cell_info_port_mut() {
    let pool = make_pool();
    let name = pool.intern("cell");
    let ctype = pool.intern("TYPE");
    let mut ci = CellInfo::new(name, ctype);

    let port_name = pool.intern("A");
    ci.add_port(port_name, PortType::In);

    ci.set_port_net(port_name, Some(NetId::from_raw(7)));
    ci.set_port_user_idx(port_name, Some(3));

    assert_eq!(ci.port_net(port_name), Some(NetIdx(7)));
    assert_eq!(ci.port_user_idx(port_name), Some(3));
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
    ni.set_driver_raw(CellPin::new(CellId::from_raw(0), port_name));
    assert!(ni.has_driver());
    assert_eq!(ni.driver().unwrap().cell, CellIdx(0));
    assert_eq!(ni.driver().unwrap().port, port_name);
}

#[test]
fn net_info_add_users() {
    let pool = make_pool();
    let mut ni = NetInfo::new(pool.intern("n"));

    let port_a = pool.intern("A");
    let port_b = pool.intern("B");

    ni.add_user_raw(CellPin::new(CellId::from_raw(1), port_a));
    ni.add_user_raw(CellPin::new(CellId::from_raw(2), port_b));

    assert_eq!(ni.num_users(), 2);
    assert_eq!(ni.users()[0].cell, CellIdx(1));
    assert_eq!(ni.users()[1].cell, CellIdx(2));
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
            pip: Some(pip),
            strength: PlaceStrength::Placer,
        },
    );

    assert_eq!(ni.wires.len(), 1);
    let pm = ni.wires.get(&wire).unwrap();
    assert_eq!(pm.pip, Some(pip));
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
        pip: Some(PipId::new(1, 2)),
        strength: PlaceStrength::Fixed,
    };
    let pm2 = pm.clone();
    assert_eq!(pm2.pip, Some(PipId::new(1, 2)));
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
// Design -- cell operations
// =====================================================================

#[test]
fn design_new_is_empty() {
    let d = Design::new();
    assert!(d.is_empty());
    assert_eq!(d.cell_slots_len(), 0);
    assert_eq!(d.net_slots_len(), 0);
    assert!(d.hierarchy.is_empty());
    assert!(d.top_module.is_empty());
}

#[test]
fn design_default_is_empty() {
    let d = Design::default();
    assert!(d.is_empty());
}

#[test]
fn design_add_cell() {
    let pool = make_pool();
    let mut d = Design::new();

    let name = pool.intern("lut0");
    let ctype = pool.intern("LUT4");
    let idx = d.add_cell(name, ctype);

    assert_eq!(idx, CellIdx(0));
    assert_eq!(d.cell_slots_len(), 1);
    assert_eq!(d.num_cells(), 1);
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
    assert_eq!(d.cell_slots_len(), 3);
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
    d.cell_edit(idx)
        .set_bel(Some(BelId::new(1, 2)), PlaceStrength::Fixed);

    assert_eq!(d.cell(idx).bel, Some(BelId::new(1, 2)));
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

    // Removed slot is now empty and stale handle must not be used.
    assert!(d.cell_idx_at_slot(0).is_none());
    // Name lookup no longer finds it
    assert_eq!(d.cell_by_name(name), None);
    // Arena size unchanged
    assert_eq!(d.cell_slots_len(), 1);
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
    let _idx_b = d.add_cell(name_b, pool.intern("T"));
    let idx_c = d.add_cell(name_c, pool.intern("T"));

    // Remove the middle cell
    d.remove_cell(name_b);

    // Indices for a and c should still work
    assert_eq!(d.cell(idx_a).name, name_a);
    assert!(d.cell(idx_a).alive);
    assert!(d.cell_idx_at_slot(1).is_none());
    assert_eq!(d.cell(idx_c).name, name_c);
    assert!(d.cell(idx_c).alive);
}

// =====================================================================
// Design -- net operations
// =====================================================================

#[test]
fn design_add_net() {
    let pool = make_pool();
    let mut d = Design::new();

    let name = pool.intern("net0");
    let idx = d.add_net(name);

    assert_eq!(idx, NetIdx(0));
    assert_eq!(d.net_slots_len(), 1);
    assert_eq!(d.num_nets(), 1);
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
    assert_eq!(d.net_slots_len(), 3);
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

    d.net_edit(idx).set_clock_constraint(5000);
    d.net_edit(idx).set_region(Some(3));

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

    assert!(d.net_idx_at_slot(0).is_none());
    assert_eq!(d.net_by_name(name), None);
    assert_eq!(d.net_slots_len(), 1);
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
    let _idx_b = d.add_net(name_b);
    let idx_c = d.add_net(name_c);

    d.remove_net(name_b);

    assert!(d.net(idx_a).alive);
    assert!(d.net_idx_at_slot(1).is_none());
    assert!(d.net(idx_c).alive);
    assert_eq!(d.net(idx_a).name, name_a);
    assert_eq!(d.net(idx_c).name, name_c);
}

// =====================================================================
// Design -- integrated cell + net wiring
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
    d.cell_edit(drv_idx).add_port(q_port, PortType::Out);
    d.cell_edit(usr_idx).add_port(a_port, PortType::In);

    // Create net
    let net_name = pool.intern("wire0");
    let net_idx = d.add_net(net_name);

    // Wire driver
    d.net_edit(net_idx).set_driver(drv_idx, q_port);
    d.cell_edit(drv_idx)
        .set_port_net(q_port, Some(net_idx), None);

    // Wire user
    let user_idx_in_net = d
        .net_edit(net_idx)
        .add_user_raw(CellPin::new(usr_idx, a_port));
    d.cell_edit(usr_idx)
        .set_port_net(a_port, Some(net_idx), Some(user_idx_in_net))
        .set_port_budget(a_port, 200);

    // Verify
    assert!(d.net(net_idx).has_driver());
    assert_eq!(d.net(net_idx).driver().unwrap().cell, drv_idx);
    assert_eq!(d.net(net_idx).num_users(), 1);
    assert_eq!(d.net(net_idx).users()[0].cell, usr_idx);
    assert_eq!(d.cell(drv_idx).port_net(q_port), Some(net_idx));
    assert_eq!(d.cell(usr_idx).port_net(a_port), Some(net_idx));
    assert_eq!(d.cell(usr_idx).port_user_idx(a_port), Some(0));
    assert_eq!(d.cell(usr_idx).port_budget(a_port), Some(200));
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
        d.net_edit(net_idx)
            .add_wire(wire, Some(pip), PlaceStrength::Placer);
    }

    assert_eq!(d.net(net_idx).wires.len(), 5);

    let wire3 = WireId::new(0, 3);
    let pm = d.net(net_idx).wires.get(&wire3).unwrap();
    assert_eq!(pm.pip, Some(PipId::new(0, 103)));
}

// =====================================================================
// Design -- hierarchy
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
// Design -- clustering
// =====================================================================

#[test]
fn design_cluster_aggregate() {
    let pool = make_pool();
    let mut d = Design::new();

    let root_idx = d.add_cell(pool.intern("root"), pool.intern("CARRY"));
    let child1_idx = d.add_cell(pool.intern("child1"), pool.intern("CARRY"));
    let child2_idx = d.add_cell(pool.intern("child2"), pool.intern("CARRY"));

    // Form cluster membership.
    d.cell_edit(root_idx).set_cluster(Some(root_idx));
    d.cell_edit(child1_idx).set_cluster(Some(root_idx));
    d.cell_edit(child2_idx).set_cluster(Some(root_idx));
    let cluster = d
        .clusters
        .entry(root_idx)
        .or_insert_with(|| nextpnr::netlist::Cluster::new(root_idx));
    cluster.add_member(child1_idx);
    cluster.add_member(child2_idx);

    let ci_port = pool.intern("CI");
    let co_port = pool.intern("CO");
    cluster.ports.push((ci_port, co_port, 1));

    assert_eq!(d.cell(root_idx).cluster, Some(root_idx));
    assert_eq!(d.cell(child1_idx).cluster, Some(root_idx));
    assert_eq!(d.cell(child2_idx).cluster, Some(root_idx));

    let cluster = d.clusters.get(&root_idx).unwrap();
    assert_eq!(cluster.members.len(), 3);
    assert_eq!(cluster.ports.len(), 1);
    assert_eq!(cluster.ports[0].2, 1);
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
    assert_ne!(CellId::NONE, idx);
    assert_ne!(NetId::NONE, NetIdx(0));
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
    d.net_edit(idx)
        .set_driver(CellId::from_raw(0), pool.intern("Q"));
    assert!(d.net(idx).has_driver());

    // Disconnect it
    d.net_edit(idx).clear_driver();
    assert!(!d.net(idx).has_driver());
}

#[test]
fn cell_region_constraint() {
    let pool = make_pool();
    let mut d = Design::new();
    let idx = d.add_cell(pool.intern("c"), pool.intern("T"));

    assert_eq!(d.cell(idx).region, None);
    d.cell_edit(idx).set_region(Some(42));
    assert_eq!(d.cell(idx).region, Some(42));
}

#[test]
fn net_region_constraint() {
    let pool = make_pool();
    let mut d = Design::new();
    let idx = d.add_net(pool.intern("n"));

    assert_eq!(d.net(idx).region, None);
    d.net_edit(idx).set_region(Some(7));
    assert_eq!(d.net(idx).region, Some(7));
}

#[test]
fn cell_placement() {
    let pool = make_pool();
    let mut d = Design::new();
    let idx = d.add_cell(pool.intern("placed"), pool.intern("LUT4"));

    // Initially unplaced
    assert!(d.cell(idx).bel.is_none());
    assert_eq!(d.cell(idx).bel_strength, PlaceStrength::None);

    // Place it
    d.cell_edit(idx)
        .set_bel(Some(BelId::new(3, 7)), PlaceStrength::User);

    assert!(d.cell(idx).bel.is_some());
    assert_eq!(d.cell(idx).bel.unwrap().tile(), 3);
    assert_eq!(d.cell(idx).bel.unwrap().index(), 7);
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

    assert_eq!(d.cell_slots_len(), n);
    assert_eq!(d.net_slots_len(), n);

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
    d.net_edit(idx)
        .set_attr(key, Property::string("module.v:42"));

    assert_eq!(d.net(idx).attrs.get(&key).unwrap().as_str(), "module.v:42");
}

#[test]
fn cell_flat_and_timing_index() {
    let pool = make_pool();
    let mut d = Design::new();
    let idx = d.add_cell(pool.intern("c"), pool.intern("T"));

    // Defaults
    assert_eq!(d.cell(idx).flat_index, None);
    assert_eq!(d.cell(idx).timing_index, None);

    d.cell_edit(idx).set_flat_index(Some(FlatIndex(42)));
    d.cell_edit(idx).set_timing_index(Some(TimingIndex(7)));

    assert_eq!(d.cell(idx).flat_index, Some(FlatIndex(42)));
    assert_eq!(d.cell(idx).timing_index, Some(TimingIndex(7)));
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
        d.net_edit(idx)
            .add_wire(*wire, Some(*pip), PlaceStrength::Placer);
    }

    assert_eq!(d.net(idx).wires.len(), 5);

    // Verify each wire maps to the correct pip
    for (wire, pip) in &entries {
        let pm = d.net(idx).wires.get(wire).unwrap();
        assert_eq!(pm.pip, Some(*pip));
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
    // (it reuses the removed slot)
    let idx2 = d.add_cell(name, pool.intern("FDRE"));
    assert_ne!(idx2, CellIdx(0));
    assert!(d.cell(idx2).alive);
    assert_eq!(d.cell(idx2).cell_type, pool.intern("FDRE"));
    assert_eq!(d.cell_slots_len(), 1);
}

#[test]
fn remove_net_then_add_with_same_name() {
    let pool = make_pool();
    let mut d = Design::new();

    let name = pool.intern("reused_net");
    let _idx1 = d.add_net(name);
    d.remove_net(name);

    let idx2 = d.add_net(name);
    assert_ne!(idx2, NetIdx(0));
    assert!(d.net(idx2).alive);
    assert_eq!(d.net_slots_len(), 1);
}

#[test]
fn cell_pin_clone() {
    let pool = make_pool();
    let pin = CellPin::new(CellId::from_raw(5), pool.intern("D"));
    let pin2 = pin;
    assert_eq!(pin2.cell, CellIdx(5));
    assert_eq!(pin2.port, pool.intern("D"));
}

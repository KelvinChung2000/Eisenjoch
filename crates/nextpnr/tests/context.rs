//! Integration tests for the context module.
//!
//! Uses the synthetic chipdb builder from `nextpnr::chipdb::testutil` to create
//! a minimal 2x2 chip database for testing Context operations.

use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::context::Context;
use nextpnr::types::{BelId, DelayQuad, PlaceStrength, PipId, WireId};

/// Create a fresh Context backed by the synthetic 2x2 chipdb.
fn make_context() -> Context {
    let chipdb = make_test_chipdb();
    Context::new(chipdb)
}

// =========================================================================
// Construction and basic queries
// =========================================================================

#[test]
fn context_creation() {
    let ctx = make_context();
    assert_eq!(ctx.width(), 2);
    assert_eq!(ctx.height(), 2);
    assert!(!ctx.verbose());
    assert!(!ctx.debug());
    assert!(!ctx.force());
}

#[test]
fn string_interning() {
    let ctx = make_context();
    let id = ctx.id("hello");
    assert!(!id.is_empty());
    assert_eq!(ctx.name_of(id), "hello");
}

#[test]
fn string_interning_dedup() {
    let ctx = make_context();
    let a = ctx.id("test");
    let b = ctx.id("test");
    assert_eq!(a, b);
}

#[test]
fn name_of_unknown_id() {
    let ctx = make_context();
    let bad = nextpnr::types::IdString(9999);
    assert_eq!(ctx.name_of(bad), "<unknown>");
}

// =========================================================================
// BEL operations
// =========================================================================

#[test]
fn get_bels_count() {
    let ctx = make_context();
    let bels: Vec<_> = ctx.bels().collect();
    assert_eq!(bels.len(), 4);
}

#[test]
fn get_bel_name() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).name(), "LUT0");
}

#[test]
fn get_bel_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).bel_type(), "LUT4");
}

#[test]
fn get_bel_bucket() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).bucket(), "LUT4");
}

#[test]
fn get_bel_location() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    let loc = ctx.bel(bel).loc();
    assert_eq!(loc.x, 0);
    assert_eq!(loc.y, 0);
    assert_eq!(loc.z, 0);

    let bel3 = BelId::new(3, 0);
    let loc3 = ctx.bel(bel3).loc();
    assert_eq!(loc3.x, 1);
    assert_eq!(loc3.y, 1);
    assert_eq!(loc3.z, 0);
}

#[test]
fn bel_initially_available() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert!(ctx.bel(bel).is_available());
    assert!(ctx.bel(bel).bound_cell().is_none());
}

#[test]
fn bind_bel_success() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_name = ctx.id("my_lut");
    let cell_type = ctx.id("LUT4");
    let cell_idx = ctx.add_cell(cell_name, cell_type);
    assert!(ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer));
    assert!(!ctx.bel(bel).is_available());
    assert_eq!(ctx.bel(bel).bound_cell().map(|c| c.name_id()), Some(cell_name));
}

#[test]
fn bind_bel_updates_cell_info() {
    let mut ctx = make_context();
    let bel = BelId::new(1, 0);
    let cell_type = ctx.id("LUT4");
    let cell_name = ctx.id("my_lut");
    let cell_idx = ctx.add_cell(cell_name, cell_type);

    assert!(ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer));

    let cell = ctx.cell(cell_idx);
    assert_eq!(cell.bel_id(), Some(bel));
    assert_eq!(cell.bel_strength(), PlaceStrength::Placer);
}

#[test]
fn bind_bel_duplicate_fails() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT4");
    let name1 = ctx.id("cell1");
    let name2 = ctx.id("cell2");
    let idx1 = ctx.add_cell(name1, cell_type);
    let idx2 = ctx.add_cell(name2, cell_type);
    assert!(ctx.bind_bel(bel, idx1, PlaceStrength::Placer));
    assert!(!ctx.bind_bel(bel, idx2, PlaceStrength::Placer));
    assert_eq!(ctx.bel(bel).bound_cell().map(|c| c.name_id()), Some(name1));
}

#[test]
fn unbind_bel() {
    let mut ctx = make_context();
    let bel = BelId::new(2, 0);
    let cell_type = ctx.id("LUT4");
    let cell_name = ctx.id("my_lut");
    let cell_idx = ctx.add_cell(cell_name, cell_type);

    ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer);
    assert!(!ctx.bel(bel).is_available());

    ctx.unbind_bel(bel);
    assert!(ctx.bel(bel).is_available());
    assert!(ctx.bel(bel).bound_cell().is_none());

    let cell = ctx.cell(cell_idx);
    assert!(cell.bel_id().is_none());
    assert_eq!(cell.bel_strength(), PlaceStrength::None);
}

#[test]
fn unbind_bel_not_bound_is_noop() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    ctx.unbind_bel(bel);
    assert!(ctx.bel(bel).is_available());
}

#[test]
fn bind_rebind_bel() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT4");
    let name1 = ctx.id("cell_a");
    let name2 = ctx.id("cell_b");
    let idx1 = ctx.add_cell(name1, cell_type);
    let idx2 = ctx.add_cell(name2, cell_type);

    ctx.bind_bel(bel, idx1, PlaceStrength::Placer);
    ctx.unbind_bel(bel);
    assert!(ctx.bind_bel(bel, idx2, PlaceStrength::Strong));
    assert_eq!(ctx.bel(bel).bound_cell().map(|c| c.name_id()), Some(name2));
}

// =========================================================================
// Wire operations
// =========================================================================

#[test]
fn wire_initially_available() {
    let ctx = make_context();
    let wire = WireId::new(0, 0);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.wire(wire).bound_net().is_none());
}

#[test]
fn bind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(0, 0);
    let net_name = ctx.id("net_a");
    let net_idx = ctx.add_net(net_name);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Placer);
    assert!(!ctx.wire(wire).is_available());
    assert_eq!(ctx.wire(wire).bound_net().map(|n| n.name_id()), Some(net_name));
}

#[test]
fn unbind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(0, 1);
    let net_name = ctx.id("net_b");
    let net_idx = ctx.add_net(net_name);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Placer);
    ctx.unbind_wire(wire);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.wire(wire).bound_net().is_none());
}

#[test]
fn rebind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(1, 0);
    let net1 = ctx.id("net_1");
    let net2 = ctx.id("net_2");
    let idx1 = ctx.add_net(net1);
    let idx2 = ctx.add_net(net2);
    ctx.bind_wire(wire, idx1, PlaceStrength::Placer);
    ctx.bind_wire(wire, idx2, PlaceStrength::Strong);
    assert_eq!(ctx.wire(wire).bound_net().map(|n| n.name_id()), Some(net2));
}

// =========================================================================
// PIP operations
// =========================================================================

#[test]
fn pip_initially_available() {
    let ctx = make_context();
    let pip = PipId::new(0, 0);
    assert!(ctx.pip(pip).is_available());
}

#[test]
fn bind_pip() {
    let mut ctx = make_context();
    let pip = PipId::new(0, 0);
    let net_name = ctx.id("net_x");
    let net_idx = ctx.add_net(net_name);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);
    assert!(!ctx.pip(pip).is_available());
}

#[test]
fn unbind_pip() {
    let mut ctx = make_context();
    let pip = PipId::new(1, 0);
    let net_name = ctx.id("net_y");
    let net_idx = ctx.add_net(net_name);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);
    ctx.unbind_pip(pip);
    assert!(ctx.pip(pip).is_available());
}

#[test]
fn pip_src_dst_wires() {
    let ctx = make_context();
    let pip = PipId::new(0, 0);
    let src = ctx.pip(pip).src_wire().id();
    let dst = ctx.pip(pip).dst_wire().id();
    assert_eq!(src, WireId::new(0, 0));
    assert_eq!(dst, WireId::new(0, 1));
}

// =========================================================================
// Delay estimation
// =========================================================================

#[test]
fn estimate_delay_same_tile() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0);
    let w2 = WireId::new(0, 1);
    assert_eq!(ctx.estimate_delay(w1, w2), 0);
}

#[test]
fn estimate_delay_adjacent() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0);
    let w2 = WireId::new(1, 0);
    assert_eq!(ctx.estimate_delay(w1, w2), 100);
}

#[test]
fn estimate_delay_diagonal() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0);
    let w2 = WireId::new(3, 0);
    assert_eq!(ctx.estimate_delay(w1, w2), 200);
}

#[test]
fn estimate_delay_symmetric() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0);
    let w2 = WireId::new(3, 0);
    assert_eq!(ctx.estimate_delay(w1, w2), ctx.estimate_delay(w2, w1));
}

#[test]
fn pip_delay_returns_default() {
    let ctx = make_context();
    let pip = PipId::new(0, 0);
    assert_eq!(ctx.pip(pip).delay(), DelayQuad::default());
}

#[test]
fn wire_delay_returns_default() {
    let ctx = make_context();
    let wire = WireId::new(0, 0);
    assert_eq!(ctx.wire(wire).delay(), DelayQuad::default());
}

// =========================================================================
// Placement validity
// =========================================================================

#[test]
fn valid_bel_for_matching_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT4");
    assert!(ctx.bel(bel).is_valid_for_cell_type(cell_type));
}

#[test]
fn invalid_bel_for_mismatched_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("FF");
    assert!(!ctx.bel(bel).is_valid_for_cell_type(cell_type));
}

// =========================================================================
// BEL bucket operations
// =========================================================================

#[test]
fn bel_buckets_empty_before_populate() {
    let ctx = make_context();
    assert!(ctx.bel_buckets().is_empty());
}

#[test]
fn populate_bel_buckets() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let buckets = ctx.bel_buckets();
    assert_eq!(buckets.len(), 1);

    let bucket_name = ctx.name_of(buckets[0]);
    assert_eq!(bucket_name, "LUT4");
}

#[test]
fn get_bels_for_bucket() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let lut_bels: Vec<BelId> = ctx.bels_for_bucket("LUT4").map(|b| b.id()).collect();
    assert_eq!(lut_bels.len(), 4);

    let mut unique = std::collections::HashSet::new();
    for bel in &lut_bels {
        assert!(unique.insert(*bel));
    }
}

#[test]
fn get_bels_for_unknown_bucket() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    assert_eq!(ctx.bels_for_bucket("NONEXISTENT").count(), 0);
}

#[test]
fn populate_bel_buckets_idempotent() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    let first = ctx.bel_buckets().len();
    ctx.populate_bel_buckets();
    let second = ctx.bel_buckets().len();
    assert_eq!(first, second);
}

// =========================================================================
// Integration: multiple operations together
// =========================================================================

#[test]
fn full_placement_flow() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let cell_type_id = ctx.id("LUT4");
    let cell_name = ctx.id("top/lut_0");
    let cell_idx = ctx.add_cell(cell_name, cell_type_id);

    let lut_bels: Vec<BelId> = ctx.bels_for_bucket("LUT4").map(|b| b.id()).collect();
    assert!(!lut_bels.is_empty());
    let target_bel = lut_bels[0];

    assert!(ctx.bel(target_bel).is_valid_for_cell_type(cell_type_id));

    assert!(ctx.bel(target_bel).is_available());
    assert!(ctx.bind_bel(target_bel, cell_idx, PlaceStrength::Placer));

    assert!(!ctx.bel(target_bel).is_available());
    assert_eq!(ctx.bel(target_bel).bound_cell().map(|c| c.name_id()), Some(cell_name));

    let cell = ctx.cell(cell_idx);
    assert_eq!(cell.bel_id(), Some(target_bel));
}

#[test]
fn full_routing_flow() {
    let mut ctx = make_context();

    let net_name = ctx.id("net_clk");
    let net_idx = ctx.add_net(net_name);

    let pip = PipId::new(0, 0);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);

    let dst_wire = ctx.pip(pip).dst_wire().id();
    ctx.bind_wire(dst_wire, net_idx, PlaceStrength::Placer);

    assert!(!ctx.pip(pip).is_available());
    assert!(!ctx.wire(dst_wire).is_available());
    assert_eq!(ctx.wire(dst_wire).bound_net().map(|n| n.name_id()), Some(net_name));

    ctx.unbind_pip(pip);
    ctx.unbind_wire(dst_wire);
    assert!(ctx.pip(pip).is_available());
    assert!(ctx.wire(dst_wire).is_available());
}

// =========================================================================
// Settings and flags
// =========================================================================

#[test]
fn settings_operations() {
    let mut ctx = make_context();
    let key = ctx.id("opt_level");
    ctx.settings_mut()
        .insert(key, nextpnr::types::Property::int(2));
    assert_eq!(
        ctx.settings().get(&key).and_then(|p| p.as_int()),
        Some(2)
    );
}

#[test]
fn rng_deterministic_from_context() {
    let mut ctx1 = make_context();
    let mut ctx2 = make_context();
    let v1 = ctx1.rng_mut().next_u64();
    let v2 = ctx2.rng_mut().next_u64();
    assert_eq!(v1, v2);
}

// =========================================================================
// DeterministicRng unit tests (extracted from rng.rs)
// =========================================================================

use nextpnr::context::DeterministicRng;

#[test]
fn zero_seed_is_adjusted() {
    let mut rng = DeterministicRng::new(0);
    let v = rng.next_u64();
    assert_ne!(v, 0);
}

#[test]
fn deterministic_output() {
    let mut a = DeterministicRng::new(42);
    let mut b = DeterministicRng::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn different_seeds_different_output() {
    let mut a = DeterministicRng::new(1);
    let mut b = DeterministicRng::new(2);
    assert_ne!(a.next_u64(), b.next_u64());
}

#[test]
fn next_u32_truncates() {
    let mut rng = DeterministicRng::new(123);
    let v64 = {
        let mut r2 = DeterministicRng::new(123);
        r2.next_u64()
    };
    let v32 = rng.next_u32();
    assert_eq!(v32, v64 as u32);
}

#[test]
fn next_range_bounded() {
    let mut rng = DeterministicRng::new(99);
    for _ in 0..1000 {
        let v = rng.next_range(10);
        assert!(v < 10);
    }
}

#[test]
#[should_panic]
fn next_range_zero_panics() {
    let mut rng = DeterministicRng::new(1);
    rng.next_range(0);
}

#[test]
fn shuffle_empty() {
    let mut rng = DeterministicRng::new(1);
    let mut data: Vec<i32> = vec![];
    rng.shuffle(&mut data);
    assert!(data.is_empty());
}

#[test]
fn shuffle_single() {
    let mut rng = DeterministicRng::new(1);
    let mut data = vec![42];
    rng.shuffle(&mut data);
    assert_eq!(data, vec![42]);
}

#[test]
fn shuffle_preserves_elements() {
    let mut rng = DeterministicRng::new(1);
    let mut data: Vec<i32> = (0..20).collect();
    rng.shuffle(&mut data);
    data.sort();
    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(data, expected);
}

#[test]
fn shuffle_deterministic() {
    let mut rng1 = DeterministicRng::new(42);
    let mut rng2 = DeterministicRng::new(42);
    let mut data1: Vec<i32> = (0..50).collect();
    let mut data2: Vec<i32> = (0..50).collect();
    rng1.shuffle(&mut data1);
    rng2.shuffle(&mut data2);
    assert_eq!(data1, data2);
}

#[test]
fn shuffle_actually_shuffles() {
    let mut rng = DeterministicRng::new(12345);
    let original: Vec<i32> = (0..20).collect();
    let mut data = original.clone();
    rng.shuffle(&mut data);
    assert_ne!(data, original);
}

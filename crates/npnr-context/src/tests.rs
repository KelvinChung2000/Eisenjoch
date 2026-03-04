//! Unit tests for the npnr-context crate.
//!
//! Uses the synthetic chipdb builder from `npnr_chipdb::testutil` to create
//! a minimal 2x2 chip database for testing Context operations.

use npnr_chipdb::testutil::make_test_chipdb;
use npnr_types::{BelId, DelayQuad, PlaceStrength, PipId, WireId};

use crate::Context;

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
    assert!(!ctx.verbose);
    assert!(!ctx.debug);
    assert!(!ctx.force);
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
    let bad = npnr_types::IdString(9999);
    assert_eq!(ctx.name_of(bad), "<unknown>");
}

// =========================================================================
// BEL operations
// =========================================================================

#[test]
fn get_bels_count() {
    let ctx = make_context();
    let bels: Vec<_> = ctx.get_bels().collect();
    // 2x2 grid, 1 bel per tile = 4 bels
    assert_eq!(bels.len(), 4);
}

#[test]
fn get_bel_name() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.get_bel_name(bel), "LUT0");
}

#[test]
fn get_bel_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.get_bel_type(bel), "LUT4");
}

#[test]
fn get_bel_bucket() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.get_bel_bucket(bel), "LUT");
}

#[test]
fn get_bel_location() {
    let ctx = make_context();
    // Tile 0 is at (0,0), bel z=0
    let bel = BelId::new(0, 0);
    let loc = ctx.get_bel_location(bel);
    assert_eq!(loc.x, 0);
    assert_eq!(loc.y, 0);
    assert_eq!(loc.z, 0);

    // Tile 3 is at (1,1), bel z=0
    let bel3 = BelId::new(3, 0);
    let loc3 = ctx.get_bel_location(bel3);
    assert_eq!(loc3.x, 1);
    assert_eq!(loc3.y, 1);
    assert_eq!(loc3.z, 0);
}

#[test]
fn bel_initially_available() {
    let ctx = make_context();
    let bel = BelId::new(0, 0);
    assert!(ctx.is_bel_available(bel));
    assert!(ctx.get_bound_bel_cell(bel).is_none());
}

#[test]
fn bind_bel_success() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_name = ctx.id("my_lut");
    assert!(ctx.bind_bel(bel, cell_name, PlaceStrength::Placer));
    assert!(!ctx.is_bel_available(bel));
    assert_eq!(ctx.get_bound_bel_cell(bel), Some(cell_name));
}

#[test]
fn bind_bel_updates_cell_info() {
    let mut ctx = make_context();
    let bel = BelId::new(1, 0);
    let cell_type = ctx.id("LUT4");
    let cell_name = ctx.id("my_lut");
    ctx.design.add_cell(cell_name, cell_type);

    assert!(ctx.bind_bel(bel, cell_name, PlaceStrength::Placer));

    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    let cell = ctx.design.cell(cell_idx);
    assert_eq!(cell.bel, bel);
    assert_eq!(cell.bel_strength, PlaceStrength::Placer);
}

#[test]
fn bind_bel_duplicate_fails() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let name1 = ctx.id("cell1");
    let name2 = ctx.id("cell2");
    assert!(ctx.bind_bel(bel, name1, PlaceStrength::Placer));
    assert!(!ctx.bind_bel(bel, name2, PlaceStrength::Placer));
    // Original binding should be unchanged
    assert_eq!(ctx.get_bound_bel_cell(bel), Some(name1));
}

#[test]
fn unbind_bel() {
    let mut ctx = make_context();
    let bel = BelId::new(2, 0);
    let cell_type = ctx.id("LUT4");
    let cell_name = ctx.id("my_lut");
    ctx.design.add_cell(cell_name, cell_type);

    ctx.bind_bel(bel, cell_name, PlaceStrength::Placer);
    assert!(!ctx.is_bel_available(bel));

    ctx.unbind_bel(bel);
    assert!(ctx.is_bel_available(bel));
    assert!(ctx.get_bound_bel_cell(bel).is_none());

    // Cell should have its placement cleared
    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    let cell = ctx.design.cell(cell_idx);
    assert_eq!(cell.bel, BelId::INVALID);
    assert_eq!(cell.bel_strength, PlaceStrength::None);
}

#[test]
fn unbind_bel_not_bound_is_noop() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    ctx.unbind_bel(bel); // should not panic
    assert!(ctx.is_bel_available(bel));
}

#[test]
fn bind_rebind_bel() {
    let mut ctx = make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT4");
    let name1 = ctx.id("cell_a");
    let name2 = ctx.id("cell_b");
    ctx.design.add_cell(name1, cell_type);
    ctx.design.add_cell(name2, cell_type);

    ctx.bind_bel(bel, name1, PlaceStrength::Placer);
    ctx.unbind_bel(bel);
    assert!(ctx.bind_bel(bel, name2, PlaceStrength::Strong));
    assert_eq!(ctx.get_bound_bel_cell(bel), Some(name2));
}

// =========================================================================
// Wire operations
// =========================================================================

#[test]
fn wire_initially_available() {
    let ctx = make_context();
    let wire = WireId::new(0, 0);
    assert!(ctx.is_wire_available(wire));
    assert!(ctx.get_bound_wire_net(wire).is_none());
}

#[test]
fn bind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(0, 0);
    let net_name = ctx.id("net_a");
    ctx.bind_wire(wire, net_name, PlaceStrength::Placer);
    assert!(!ctx.is_wire_available(wire));
    assert_eq!(ctx.get_bound_wire_net(wire), Some(net_name));
}

#[test]
fn unbind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(0, 1);
    let net_name = ctx.id("net_b");
    ctx.bind_wire(wire, net_name, PlaceStrength::Placer);
    ctx.unbind_wire(wire);
    assert!(ctx.is_wire_available(wire));
    assert!(ctx.get_bound_wire_net(wire).is_none());
}

#[test]
fn rebind_wire() {
    let mut ctx = make_context();
    let wire = WireId::new(1, 0);
    let net1 = ctx.id("net_1");
    let net2 = ctx.id("net_2");
    ctx.bind_wire(wire, net1, PlaceStrength::Placer);
    // Overwriting is allowed (the map replaces the old entry)
    ctx.bind_wire(wire, net2, PlaceStrength::Strong);
    assert_eq!(ctx.get_bound_wire_net(wire), Some(net2));
}

// =========================================================================
// PIP operations
// =========================================================================

#[test]
fn pip_initially_available() {
    let ctx = make_context();
    let pip = PipId::new(0, 0);
    assert!(ctx.is_pip_available(pip));
}

#[test]
fn bind_pip() {
    let mut ctx = make_context();
    let pip = PipId::new(0, 0);
    let net_name = ctx.id("net_x");
    ctx.bind_pip(pip, net_name, PlaceStrength::Placer);
    assert!(!ctx.is_pip_available(pip));
}

#[test]
fn unbind_pip() {
    let mut ctx = make_context();
    let pip = PipId::new(1, 0);
    let net_name = ctx.id("net_y");
    ctx.bind_pip(pip, net_name, PlaceStrength::Placer);
    ctx.unbind_pip(pip);
    assert!(ctx.is_pip_available(pip));
}

#[test]
fn pip_src_dst_wires() {
    let ctx = make_context();
    // In the synthetic chipdb, pip(tile=0, idx=0) goes from wire 0 to wire 1
    // with zero tile deltas
    let pip = PipId::new(0, 0);
    let src = ctx.get_pip_src_wire(pip);
    let dst = ctx.get_pip_dst_wire(pip);
    assert_eq!(src, WireId::new(0, 0));
    assert_eq!(dst, WireId::new(0, 1));
}

// =========================================================================
// Delay estimation
// =========================================================================

#[test]
fn estimate_delay_same_tile() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0); // tile 0 = (0,0)
    let w2 = WireId::new(0, 1); // tile 0 = (0,0)
    assert_eq!(ctx.estimate_delay(w1, w2), 0);
}

#[test]
fn estimate_delay_adjacent() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0); // tile 0 = (0,0)
    let w2 = WireId::new(1, 0); // tile 1 = (1,0)
    // dx=1, dy=0 => 1 * 100 = 100
    assert_eq!(ctx.estimate_delay(w1, w2), 100);
}

#[test]
fn estimate_delay_diagonal() {
    let ctx = make_context();
    let w1 = WireId::new(0, 0); // tile 0 = (0,0)
    let w2 = WireId::new(3, 0); // tile 3 = (1,1)
    // dx=1, dy=1 => 2 * 100 = 200
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
    assert_eq!(ctx.get_pip_delay(pip), DelayQuad::default());
}

#[test]
fn wire_delay_returns_default() {
    let ctx = make_context();
    let wire = WireId::new(0, 0);
    assert_eq!(ctx.get_wire_delay(wire), DelayQuad::default());
}

// =========================================================================
// Placement validity
// =========================================================================

#[test]
fn valid_bel_for_matching_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0); // bucket = "LUT"
    let cell_type = ctx.id("LUT");
    assert!(ctx.is_valid_bel_for_cell(bel, cell_type));
}

#[test]
fn invalid_bel_for_mismatched_type() {
    let ctx = make_context();
    let bel = BelId::new(0, 0); // bucket = "LUT"
    let cell_type = ctx.id("FF");
    assert!(!ctx.is_valid_bel_for_cell(bel, cell_type));
}

// =========================================================================
// BEL bucket operations
// =========================================================================

#[test]
fn bel_buckets_empty_before_populate() {
    let ctx = make_context();
    assert!(ctx.get_bel_buckets().is_empty());
}

#[test]
fn populate_bel_buckets() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let buckets = ctx.get_bel_buckets();
    assert_eq!(buckets.len(), 1); // only "LUT" bucket in the synthetic chipdb

    let bucket_name = ctx.name_of(buckets[0]);
    assert_eq!(bucket_name, "LUT");
}

#[test]
fn get_bels_for_bucket() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let lut_bels = ctx.get_bels_for_bucket("LUT");
    assert_eq!(lut_bels.len(), 4); // 4 tiles, each with 1 LUT

    // All bels should be distinct
    let mut unique = std::collections::HashSet::new();
    for bel in lut_bels {
        assert!(unique.insert(*bel));
    }
}

#[test]
fn get_bels_for_unknown_bucket() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    assert!(ctx.get_bels_for_bucket("NONEXISTENT").is_empty());
}

#[test]
fn populate_bel_buckets_idempotent() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();
    let first = ctx.get_bel_buckets().len();
    ctx.populate_bel_buckets();
    let second = ctx.get_bel_buckets().len();
    assert_eq!(first, second);
}

// =========================================================================
// Integration: multiple operations together
// =========================================================================

#[test]
fn full_placement_flow() {
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    // Create a cell in the design
    let cell_type_id = ctx.id("LUT");
    let cell_name = ctx.id("top/lut_0");
    ctx.design.add_cell(cell_name, cell_type_id);

    // Find a valid bel
    let lut_bels = ctx.get_bels_for_bucket("LUT");
    assert!(!lut_bels.is_empty());
    let target_bel = lut_bels[0];

    // Validate placement
    assert!(ctx.is_valid_bel_for_cell(target_bel, cell_type_id));

    // Place the cell
    assert!(ctx.is_bel_available(target_bel));
    assert!(ctx.bind_bel(target_bel, cell_name, PlaceStrength::Placer));

    // Verify placement
    assert!(!ctx.is_bel_available(target_bel));
    assert_eq!(ctx.get_bound_bel_cell(target_bel), Some(cell_name));

    let cell_idx = ctx.design.cell_by_name(cell_name).unwrap();
    let cell = ctx.design.cell(cell_idx);
    assert_eq!(cell.bel, target_bel);
}

#[test]
fn full_routing_flow() {
    let mut ctx = make_context();

    // Create a net in the design
    let net_name = ctx.id("net_clk");
    ctx.design.add_net(net_name);

    // Bind a PIP
    let pip = PipId::new(0, 0);
    ctx.bind_pip(pip, net_name, PlaceStrength::Placer);

    // Bind the destination wire
    let dst_wire = ctx.get_pip_dst_wire(pip);
    ctx.bind_wire(dst_wire, net_name, PlaceStrength::Placer);

    // Verify state
    assert!(!ctx.is_pip_available(pip));
    assert!(!ctx.is_wire_available(dst_wire));
    assert_eq!(ctx.get_bound_wire_net(dst_wire), Some(net_name));

    // Unbind everything
    ctx.unbind_pip(pip);
    ctx.unbind_wire(dst_wire);
    assert!(ctx.is_pip_available(pip));
    assert!(ctx.is_wire_available(dst_wire));
}

// =========================================================================
// Settings and flags
// =========================================================================

#[test]
fn settings_operations() {
    let mut ctx = make_context();
    let key = ctx.id("opt_level");
    ctx.settings
        .insert(key, npnr_types::Property::int(2));
    assert_eq!(
        ctx.settings.get(&key).and_then(|p| p.as_int()),
        Some(2)
    );
}

#[test]
fn flags_default_values() {
    let ctx = make_context();
    assert!(!ctx.verbose);
    assert!(!ctx.debug);
    assert!(!ctx.force);
}

#[test]
fn rng_deterministic_from_context() {
    let mut ctx1 = make_context();
    let mut ctx2 = make_context();
    // Both contexts start with the same seed
    let v1 = ctx1.rng.next_u64();
    let v2 = ctx2.rng.next_u64();
    assert_eq!(v1, v2);
}

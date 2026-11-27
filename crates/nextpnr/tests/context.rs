//! Integration tests for the context module.
//!
//! Most tests use the minimal 2x2 chipdb from `nextpnr::chipdb::testutil`. The
//! typed bind, swap and unbind tests use the example arch chipdb in
//! `tests/fixtures`.

mod common;

use nextpnr::chipdb::{BelId, Loc, PipId, WireId};
use nextpnr::common::{DelayQuad, DelayT};
use nextpnr::common::{IdString, PlaceStrength};
use nextpnr::context::{BelBindError, BelLocation, BelPin, Context};
use nextpnr::netlist::{CellId, PortType, Property};
use rstest::rstest;
use std::sync::Arc;

// =========================================================================
// Construction and basic queries
// =========================================================================

#[test]
fn context_creation() {
    let ctx = common::make_context();
    assert_eq!(ctx.chipdb().width(), 2);
    assert_eq!(ctx.chipdb().height(), 2);
    assert!(!ctx.verbose());
    assert!(!ctx.debug());
    assert!(!ctx.force());
}

#[test]
fn string_interning() {
    let ctx = common::make_context();
    let id = ctx.id("hello");
    assert!(!id.is_empty());
    assert_eq!(ctx.name_of(id), "hello");
}

#[test]
fn string_interning_dedup() {
    let ctx = common::make_context();
    let a = ctx.id("test");
    let b = ctx.id("test");
    assert_eq!(a, b);
}

#[test]
fn name_of_unknown_id() {
    let ctx = common::make_context();
    let bad = IdString(9999);
    assert_eq!(ctx.name_of(bad), "<unknown>");
}

// =========================================================================
// BEL operations
// =========================================================================

#[test]
fn get_bels_count() {
    let ctx = common::make_context();
    let bels: Vec<_> = ctx.bels().collect();
    assert_eq!(bels.len(), 4);
}

#[test]
fn get_bel_name() {
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).name(), "LUT0");
}

#[test]
fn get_bel_type() {
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).bel_type(), "LUT6");
}

#[test]
fn get_bel_bucket() {
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    assert_eq!(ctx.bel(bel).bucket(), "LUT6");
}

#[test]
fn get_bel_location() {
    let ctx = common::make_context();
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
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    assert!(ctx.bel(bel).is_available());
    assert!(ctx.bel(bel).bound_cell().is_none());
}

#[test]
fn bind_bel_success() {
    let mut ctx = common::make_context();
    let bel = BelId::new(0, 0);
    let cell_name = ctx.id("my_lut");
    let cell_type = ctx.id("LUT6");
    let cell_idx = ctx.design.add_cell(cell_name, cell_type);
    assert!(ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer));
    assert!(!ctx.bel(bel).is_available());
    assert_eq!(
        ctx.bel(bel).bound_cell().map(|c| c.name_id()),
        Some(cell_name)
    );
}

#[test]
fn bind_bel_updates_cell_info() {
    let mut ctx = common::make_context();
    let bel = BelId::new(1, 0);
    let cell_type = ctx.id("LUT6");
    let cell_name = ctx.id("my_lut");
    let cell_idx = ctx.design.add_cell(cell_name, cell_type);

    assert!(ctx.bind_bel(bel, cell_idx, PlaceStrength::Placer));

    let cell = ctx.cell(cell_idx);
    assert_eq!(cell.bel_id(), Some(bel));
    assert_eq!(cell.bel_strength(), PlaceStrength::Placer);
}

#[test]
fn bind_bel_duplicate_fails() {
    let mut ctx = common::make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT6");
    let name1 = ctx.id("cell1");
    let name2 = ctx.id("cell2");
    let idx1 = ctx.design.add_cell(name1, cell_type);
    let idx2 = ctx.design.add_cell(name2, cell_type);
    assert!(ctx.bind_bel(bel, idx1, PlaceStrength::Placer));
    assert!(!ctx.bind_bel(bel, idx2, PlaceStrength::Placer));
    assert_eq!(ctx.bel(bel).bound_cell().map(|c| c.name_id()), Some(name1));
}

#[test]
fn unbind_bel() {
    let mut ctx = common::make_context();
    let bel = BelId::new(2, 0);
    let cell_type = ctx.id("LUT6");
    let cell_name = ctx.id("my_lut");
    let cell_idx = ctx.design.add_cell(cell_name, cell_type);

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
    let mut ctx = common::make_context();
    let bel = BelId::new(0, 0);
    ctx.unbind_bel(bel);
    assert!(ctx.bel(bel).is_available());
}

#[test]
fn bind_rebind_bel() {
    let mut ctx = common::make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT6");
    let name1 = ctx.id("cell_a");
    let name2 = ctx.id("cell_b");
    let idx1 = ctx.design.add_cell(name1, cell_type);
    let idx2 = ctx.design.add_cell(name2, cell_type);

    ctx.bind_bel(bel, idx1, PlaceStrength::Placer);
    ctx.unbind_bel(bel);
    assert!(ctx.bind_bel(bel, idx2, PlaceStrength::Strong));
    assert_eq!(ctx.bel(bel).bound_cell().map(|c| c.name_id()), Some(name2));
}

#[test]
fn bel_pin_view_exposes_associated_wire() {
    let ctx = common::make_context();
    let pin = BelPin::new(BelId::new(0, 0), ctx.id("I0"));

    assert_eq!(
        pin.view(&ctx).wire().map(|wire| wire.id()),
        Some(WireId::new(0, 0))
    );
}

#[test]
fn cell_pin_view_exposes_port_properties() {
    let mut ctx = common::make_context();
    let cell_name = ctx.id("sink");
    let cell_type = ctx.id("LUT6");
    let port_name = ctx.id("A");
    let net_name = ctx.id("net_a");

    let cell_idx = ctx.design.add_cell(cell_name, cell_type);
    let net_idx = ctx.design.add_net(net_name);

    ctx.design
        .cell_edit(cell_idx)
        .add_port(port_name, PortType::In);
    ctx.design
        .cell_edit(cell_idx)
        .set_port_net(port_name, Some(net_idx), Some(3));

    let pin = ctx.cell(cell_idx).port(port_name).unwrap();
    let pin_view = ctx.cell_pin(pin);

    assert_eq!(pin.cell, cell_idx);
    assert_eq!(pin.port, port_name);
    assert_eq!(pin_view.port_type(), PortType::In);
    assert_eq!(pin_view.net().map(|net| net.id()), Some(net_idx));
    assert_eq!(pin_view.user_idx(), Some(3));
    assert!(pin_view.is_connected());
}

// =========================================================================
// Wire operations
// =========================================================================

#[test]
fn wire_initially_available() {
    let ctx = common::make_context();
    let wire = WireId::new(0, 0);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.wire(wire).bound_net().is_none());
}

#[test]
fn bind_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let net_name = ctx.id("net_a");
    let net_idx = ctx.design.add_net(net_name);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Placer);
    assert!(!ctx.wire(wire).is_available());
    assert_eq!(
        ctx.wire(wire).bound_net().map(|n| n.name_id()),
        Some(net_name)
    );
}

#[test]
fn unbind_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(0, 1);
    let net_name = ctx.id("net_b");
    let net_idx = ctx.design.add_net(net_name);
    ctx.bind_wire(wire, net_idx, PlaceStrength::Placer);
    ctx.unbind_wire(wire);
    assert!(ctx.wire(wire).is_available());
    assert!(ctx.wire(wire).bound_net().is_none());
}

#[test]
fn rebind_wire() {
    let mut ctx = common::make_context();
    let wire = WireId::new(1, 0);
    let net1 = ctx.id("net_1");
    let net2 = ctx.id("net_2");
    let idx1 = ctx.design.add_net(net1);
    let idx2 = ctx.design.add_net(net2);
    ctx.bind_wire(wire, idx1, PlaceStrength::Placer);
    ctx.bind_wire(wire, idx2, PlaceStrength::Strong);
    assert_eq!(ctx.wire(wire).bound_net().map(|n| n.name_id()), Some(net2));
}

// =========================================================================
// PIP operations
// =========================================================================

#[test]
fn pip_initially_available() {
    let ctx = common::make_context();
    let pip = PipId::new(0, 0);
    assert!(ctx.pip(pip).is_available());
}

#[test]
fn bind_pip() {
    let mut ctx = common::make_context();
    let pip = PipId::new(0, 0);
    let net_name = ctx.id("net_x");
    let net_idx = ctx.design.add_net(net_name);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);
    assert!(!ctx.pip(pip).is_available());
}

#[test]
fn unbind_pip() {
    let mut ctx = common::make_context();
    let pip = PipId::new(1, 0);
    let net_name = ctx.id("net_y");
    let net_idx = ctx.design.add_net(net_name);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);
    ctx.unbind_pip(pip);
    assert!(ctx.pip(pip).is_available());
}

#[test]
fn pip_src_dst_wires() {
    let ctx = common::make_context();
    let pip = PipId::new(0, 0);
    let src = ctx.pip(pip).src_wire().id();
    let dst = ctx.pip(pip).dst_wire().id();
    assert_eq!(src, WireId::new(0, 0));
    assert_eq!(dst, WireId::new(0, 1));
}

// =========================================================================
// Delay estimation
// =========================================================================

#[rstest]
#[case::same_tile(WireId::new(0, 1), 0)]
#[case::adjacent(WireId::new(1, 0), 10)]
#[case::diagonal(WireId::new(3, 0), 20)]
fn estimate_delay_is_ten_per_manhattan_tile(#[case] dst: WireId, #[case] expected: DelayT) {
    let ctx = common::make_context();
    assert_eq!(ctx.estimate_delay(WireId::new(0, 0), dst), expected);
}

#[test]
fn estimate_delay_symmetric() {
    let ctx = common::make_context();
    let w1 = WireId::new(0, 0);
    let w2 = WireId::new(3, 0);
    assert_eq!(ctx.estimate_delay(w1, w2), ctx.estimate_delay(w2, w1));
}

#[test]
fn pip_delay_from_timing_data() {
    let ctx = common::make_context();
    let pip = PipId::new(0, 0);
    let delay = ctx.pip(pip).delay();
    // Synthetic chipdb has uniform pip timing: 100ps fast, 150ps slow
    assert_ne!(
        delay,
        DelayQuad::default(),
        "pip should have non-zero timing"
    );
}

#[test]
fn wire_delay_from_timing_data() {
    let ctx = common::make_context();
    let wire = WireId::new(0, 0);
    let delay = ctx.wire(wire).delay();
    // Synthetic chipdb has uniform node timing: 50ps fast, 75ps slow
    assert_ne!(
        delay,
        DelayQuad::default(),
        "wire should have non-zero timing"
    );
}

// =========================================================================
// Placement validity
// =========================================================================

#[test]
fn valid_bel_for_matching_type() {
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("LUT6");
    assert!(ctx.bel(bel).is_valid_for_cell_type(cell_type));
}

#[test]
fn invalid_bel_for_mismatched_type() {
    let ctx = common::make_context();
    let bel = BelId::new(0, 0);
    let cell_type = ctx.id("FF");
    assert!(!ctx.bel(bel).is_valid_for_cell_type(cell_type));
}

// =========================================================================
// BEL bucket operations
// =========================================================================

#[test]
fn bel_buckets_empty_before_populate() {
    let ctx = common::make_context();
    assert!(ctx.bel_buckets().is_empty());
}

#[test]
fn populate_bel_buckets() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();

    let buckets = ctx.bel_buckets();
    assert_eq!(buckets.len(), 1);

    let bucket_name = ctx.name_of(buckets[0]);
    assert_eq!(bucket_name, "LUT6");
}

#[test]
fn get_bels_for_bucket() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();

    let lut_bucket = ctx.id("LUT6");
    let lut_bels: Vec<BelId> = ctx.bels_for_bucket(lut_bucket).map(|b| b.id()).collect();
    assert_eq!(lut_bels.len(), 4);

    let mut unique = std::collections::HashSet::new();
    for bel in &lut_bels {
        assert!(unique.insert(*bel));
    }
}

#[test]
fn get_bels_for_unknown_bucket() {
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();
    let bogus = ctx.id("NONEXISTENT");
    assert_eq!(ctx.bels_for_bucket(bogus).count(), 0);
}

#[test]
fn populate_bel_buckets_idempotent() {
    let mut ctx = common::make_context();
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
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();

    let cell_type_id = ctx.id("LUT6");
    let cell_name = ctx.id("top/lut_0");
    let cell_idx = ctx.design.add_cell(cell_name, cell_type_id);

    let lut_bels: Vec<BelId> = ctx.bels_for_bucket(cell_type_id).map(|b| b.id()).collect();
    assert!(!lut_bels.is_empty());
    let target_bel = lut_bels[0];

    assert!(ctx.bel(target_bel).is_valid_for_cell_type(cell_type_id));

    assert!(ctx.bel(target_bel).is_available());
    assert!(ctx.bind_bel(target_bel, cell_idx, PlaceStrength::Placer));

    assert!(!ctx.bel(target_bel).is_available());
    assert_eq!(
        ctx.bel(target_bel).bound_cell().map(|c| c.name_id()),
        Some(cell_name)
    );

    let cell = ctx.cell(cell_idx);
    assert_eq!(cell.bel_id(), Some(target_bel));
}

#[test]
fn full_routing_flow() {
    let mut ctx = common::make_context();

    let net_name = ctx.id("net_clk");
    let net_idx = ctx.design.add_net(net_name);

    let pip = PipId::new(0, 0);
    ctx.bind_pip(pip, net_idx, PlaceStrength::Placer);

    let dst_wire = ctx.pip(pip).dst_wire().id();
    ctx.bind_wire(dst_wire, net_idx, PlaceStrength::Placer);

    assert!(!ctx.pip(pip).is_available());
    assert!(!ctx.wire(dst_wire).is_available());
    assert_eq!(
        ctx.wire(dst_wire).bound_net().map(|n| n.name_id()),
        Some(net_name)
    );

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
    let mut ctx = common::make_context();
    let key = ctx.id("opt_level");
    ctx.settings_mut().insert(key, Property::int(2));
    assert_eq!(ctx.settings().get(&key).and_then(|p| p.as_int()), Some(2));
}

#[test]
fn rng_deterministic_from_context() {
    let mut ctx1 = common::make_context();
    let mut ctx2 = common::make_context();
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

fn example_bels(ctx: &Context, bel_type: &str) -> Vec<BelId> {
    ctx.chipdb()
        .bels()
        .filter(|&bel| ctx.chipdb().bel_type(bel) == bel_type)
        .collect()
}

fn location(ctx: &Context, bel: BelId) -> BelLocation {
    BelLocation {
        tile: ctx.chipdb().tile_name(bel.tile()).into(),
        bel: ctx.chipdb().bel_name(bel).into(),
    }
}

type Placement = (
    Vec<Option<CellId>>,
    Vec<(CellId, Option<BelId>, PlaceStrength)>,
);

fn placement(ctx: &Context) -> Placement {
    let bels = ctx
        .chipdb()
        .bels()
        .map(|bel| ctx.conflicting_bel_cell(bel))
        .collect();
    let cells = ctx
        .design
        .iter_alive_cells()
        .map(|(cell, info)| (cell, info.bel, info.bel_strength))
        .collect();
    (bels, cells)
}

enum Edit {
    Bind { cell: CellId, bel: BelId },
    Swap { first: CellId, second: CellId },
    Unbind { cell: CellId },
}

impl Edit {
    fn apply(&self, ctx: &mut Context) -> Result<(), BelBindError> {
        match *self {
            Edit::Bind { cell, bel } => ctx.try_bind_bel(cell, bel, PlaceStrength::User),
            Edit::Swap { first, second } => ctx.try_swap_cells(first, second),
            Edit::Unbind { cell } => ctx.try_unbind_cell(cell),
        }
    }
}

struct Refusal {
    ctx: Context,
    edit: Edit,
    expected: BelBindError,
}

fn bind_to_missing_bel() -> Refusal {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let bel = BelId::new(ctx.chipdb().num_tiles(), 0);
    Refusal {
        ctx,
        edit: Edit::Bind { cell, bel },
        expected: BelBindError::BelNotFound {
            cell: "lut".into(),
            bel,
        },
    }
}

fn bind_to_other_type() -> Refusal {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let bel = example_bels(&ctx, "DFF")[0];
    let expected = BelBindError::TypeMismatch {
        cell: "lut".into(),
        bel: location(&ctx, bel),
        cell_type: "LUT6".into(),
        bel_type: "DFF".into(),
    };
    Refusal {
        ctx,
        edit: Edit::Bind { cell, bel },
        expected,
    }
}

fn bind_bound_cell() -> Refusal {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let luts = example_bels(&ctx, "LUT6");
    ctx.try_bind_bel(cell, luts[0], PlaceStrength::Placer)
        .unwrap();
    let expected = BelBindError::CellAlreadyBound {
        cell: "lut".into(),
        bel: location(&ctx, luts[1]),
        bound_to: location(&ctx, luts[0]),
    };
    Refusal {
        ctx,
        edit: Edit::Bind { cell, bel: luts[1] },
        expected,
    }
}

fn bind_to_occupied_bel() -> Refusal {
    let mut ctx = common::make_example_context();
    let occupant = common::add_cell(&mut ctx, "occupant", "LUT6");
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let bel = example_bels(&ctx, "LUT6")[0];
    ctx.try_bind_bel(occupant, bel, PlaceStrength::Placer)
        .unwrap();
    let expected = BelBindError::BelOccupied {
        cell: "lut".into(),
        bel: location(&ctx, bel),
        occupant: "occupant".into(),
    };
    Refusal {
        ctx,
        edit: Edit::Bind { cell, bel },
        expected,
    }
}

fn bind_to_invalid_location() -> Refusal {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let bel = example_bels(&ctx, "LUT6")[0];
    ctx.set_validity_check(Arc::new(|_, _| false));
    let expected = BelBindError::InvalidLocation {
        cell: "lut".into(),
        bel: location(&ctx, bel),
    };
    Refusal {
        ctx,
        edit: Edit::Bind { cell, bel },
        expected,
    }
}

fn placed_luts() -> (Context, CellId, CellId, BelId, BelId) {
    let mut ctx = common::make_example_context();
    let first = common::add_cell(&mut ctx, "lut_a", "LUT6");
    let second = common::add_cell(&mut ctx, "lut_b", "LUT6");
    let luts = example_bels(&ctx, "LUT6");
    ctx.try_bind_bel(first, luts[0], PlaceStrength::Placer)
        .unwrap();
    ctx.try_bind_bel(second, luts[1], PlaceStrength::User)
        .unwrap();
    (ctx, first, second, luts[0], luts[1])
}

fn swap_with_unplaced_cell() -> Refusal {
    let (mut ctx, first, _, _, _) = placed_luts();
    let unplaced = common::add_cell(&mut ctx, "unplaced", "LUT6");
    Refusal {
        ctx,
        edit: Edit::Swap {
            first,
            second: unplaced,
        },
        expected: BelBindError::CellNotPlaced {
            cell: "unplaced".into(),
        },
    }
}

fn swap_across_types() -> Refusal {
    let (mut ctx, first, _, _, _) = placed_luts();
    let dff = common::add_cell(&mut ctx, "dff", "DFF");
    let dff_bel = example_bels(&ctx, "DFF")[0];
    ctx.try_bind_bel(dff, dff_bel, PlaceStrength::Placer)
        .unwrap();
    let expected = BelBindError::TypeMismatch {
        cell: "lut_a".into(),
        bel: location(&ctx, dff_bel),
        cell_type: "LUT6".into(),
        bel_type: "DFF".into(),
    };
    Refusal {
        ctx,
        edit: Edit::Swap { first, second: dff },
        expected,
    }
}

fn swap_into_invalid_location() -> Refusal {
    let (mut ctx, first, second, first_bel, _) = placed_luts();
    ctx.set_validity_check(Arc::new(move |_, bel| bel != first_bel));
    let expected = BelBindError::InvalidLocation {
        cell: "lut_b".into(),
        bel: location(&ctx, first_bel),
    };
    Refusal {
        ctx,
        edit: Edit::Swap { first, second },
        expected,
    }
}

struct ClusterFixture {
    ctx: Context,
    root: CellId,
    ff: CellId,
    lut_b: CellId,
    root_bel: BelId,
}

impl ClusterFixture {
    fn bel_at(&self, dx: i32, z: i32) -> BelId {
        let loc = self.ctx.chipdb().bel_loc(self.root_bel);
        self.ctx
            .bel_by_location(Loc::new(loc.x + dx, loc.y, z))
            .expect("the fixture tile row has this BEL")
    }

    fn expected_members(&self) -> [(CellId, BelLocation); 3] {
        let root_tile = self.ctx.chipdb().tile_name(self.root_bel.tile());
        let east_tile = self.ctx.chipdb().tile_name(self.bel_at(1, 0).tile());
        [
            (self.root, location(&self.ctx, self.root_bel)),
            (
                self.ff,
                BelLocation {
                    tile: root_tile.into(),
                    bel: "L0_FF".into(),
                },
            ),
            (
                self.lut_b,
                BelLocation {
                    tile: east_tile.into(),
                    bel: "L0_LUT".into(),
                },
            ),
        ]
    }
}

fn cluster() -> ClusterFixture {
    let mut ctx = common::make_example_context();
    let (root, ff) = common::add_lut_into_ff(&mut ctx, "lut", "ff");
    assert_eq!(common::cluster_lut_ff_pairs(&mut ctx), 1);
    let lut_b = common::add_cell(&mut ctx, "lut_b", "LUT6");
    ctx.design
        .cell_edit(lut_b)
        .set_cluster(Some(root))
        .set_constraints(1, 0, 0, false);
    let members = ctx
        .design
        .clusters
        .get_mut(&root)
        .expect("lut roots a cluster");
    members.members.push(lut_b);
    members.constr_children.push(lut_b);
    let root_bel = example_bels(&ctx, "LUT6")
        .into_iter()
        .find(|&bel| {
            let loc = ctx.chipdb().bel_loc(bel);
            ctx.chipdb().bel_name(bel) == "L0_LUT"
                && ctx
                    .bel_by_location(Loc::new(loc.x + 1, loc.y, 0))
                    .is_some_and(|east| ctx.chipdb().bel_name(east) == "L0_LUT")
        })
        .expect("a logic tile has a logic tile to its east");
    ClusterFixture {
        ctx,
        root,
        ff,
        lut_b,
        root_bel,
    }
}

fn placed_cluster() -> ClusterFixture {
    let mut fixture = cluster();
    fixture
        .ctx
        .try_bind_bel(fixture.root, fixture.root_bel, PlaceStrength::Placer)
        .unwrap();
    fixture
}

fn bind_cluster_child() -> Refusal {
    let fixture = cluster();
    let bel = fixture.bel_at(0, 1);
    Refusal {
        edit: Edit::Bind {
            cell: fixture.ff,
            bel,
        },
        expected: BelBindError::ClusterChild {
            cell: "ff".into(),
            root: "lut".into(),
        },
        ctx: fixture.ctx,
    }
}

fn bind_cluster_with_member_off_chip() -> Refusal {
    let mut fixture = cluster();
    fixture
        .ctx
        .design
        .cell_edit(fixture.lut_b)
        .set_constraints(0, 0, 16, false);
    let root_loc = fixture.ctx.chipdb().bel_loc(fixture.root_bel);
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel: fixture.root_bel,
        },
        expected: BelBindError::NoBelAtClusterOffset {
            cell: "lut_b".into(),
            root: "lut".into(),
            location: Loc::new(root_loc.x, root_loc.y, 16),
        },
        ctx: fixture.ctx,
    }
}

fn bind_cluster_with_member_on_other_type() -> Refusal {
    let mut fixture = cluster();
    fixture
        .ctx
        .design
        .cell_edit(fixture.ff)
        .set_constraints(0, 0, 2, false);
    let expected = BelBindError::TypeMismatch {
        cell: "ff".into(),
        bel: location(&fixture.ctx, fixture.bel_at(0, 2)),
        cell_type: "DFF".into(),
        bel_type: "LUT6".into(),
    };
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel: fixture.root_bel,
        },
        expected,
        ctx: fixture.ctx,
    }
}

fn bind_cluster_with_bound_member() -> Refusal {
    let mut fixture = cluster();
    let elsewhere = fixture.bel_at(0, 4);
    assert!(fixture
        .ctx
        .bind_bel(elsewhere, fixture.lut_b, PlaceStrength::Placer));
    let expected = BelBindError::CellAlreadyBound {
        cell: "lut_b".into(),
        bel: location(&fixture.ctx, fixture.bel_at(1, 0)),
        bound_to: location(&fixture.ctx, elsewhere),
    };
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel: fixture.root_bel,
        },
        expected,
        ctx: fixture.ctx,
    }
}

fn bind_cluster_onto_occupied_member_bel() -> Refusal {
    let mut fixture = cluster();
    let occupant = common::add_cell(&mut fixture.ctx, "occupant", "DFF");
    let ff_bel = fixture.bel_at(0, 1);
    fixture
        .ctx
        .try_bind_bel(occupant, ff_bel, PlaceStrength::Placer)
        .unwrap();
    let expected = BelBindError::BelOccupied {
        cell: "ff".into(),
        bel: location(&fixture.ctx, ff_bel),
        occupant: "occupant".into(),
    };
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel: fixture.root_bel,
        },
        expected,
        ctx: fixture.ctx,
    }
}

fn bind_cluster_into_invalid_tile() -> Refusal {
    let mut fixture = cluster();
    let east = fixture.bel_at(1, 0);
    fixture
        .ctx
        .set_validity_check(Arc::new(move |_, bel| bel != east));
    let expected = BelBindError::InvalidLocation {
        cell: "lut_b".into(),
        bel: location(&fixture.ctx, east),
    };
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel: fixture.root_bel,
        },
        expected,
        ctx: fixture.ctx,
    }
}

fn bind_cluster_root_off_its_absolute_z() -> Refusal {
    let mut fixture = cluster();
    fixture
        .ctx
        .design
        .cell_edit(fixture.root)
        .set_constraints(0, 0, 0, true);
    let bel = fixture.bel_at(0, 2);
    let expected = BelBindError::ClusterRootZ {
        cell: "lut".into(),
        bel: location(&fixture.ctx, bel),
        required: location(&fixture.ctx, fixture.root_bel),
    };
    Refusal {
        edit: Edit::Bind {
            cell: fixture.root,
            bel,
        },
        expected,
        ctx: fixture.ctx,
    }
}

fn placed_cluster_and_lone_lut() -> (ClusterFixture, CellId) {
    let mut fixture = placed_cluster();
    let lone = common::add_cell(&mut fixture.ctx, "lone", "LUT6");
    let bel = fixture.bel_at(0, 4);
    fixture
        .ctx
        .try_bind_bel(lone, bel, PlaceStrength::Placer)
        .unwrap();
    (fixture, lone)
}

fn swap_cluster_root() -> Refusal {
    let (fixture, lone) = placed_cluster_and_lone_lut();
    Refusal {
        edit: Edit::Swap {
            first: fixture.root,
            second: lone,
        },
        expected: BelBindError::ClusterSwap {
            cell: "lut".into(),
            root: "lut".into(),
        },
        ctx: fixture.ctx,
    }
}

fn swap_cluster_child() -> Refusal {
    let (fixture, lone) = placed_cluster_and_lone_lut();
    Refusal {
        edit: Edit::Swap {
            first: lone,
            second: fixture.ff,
        },
        expected: BelBindError::ClusterSwap {
            cell: "ff".into(),
            root: "lut".into(),
        },
        ctx: fixture.ctx,
    }
}

fn unbind_cluster_child() -> Refusal {
    let fixture = placed_cluster();
    Refusal {
        edit: Edit::Unbind { cell: fixture.ff },
        expected: BelBindError::ClusterChild {
            cell: "ff".into(),
            root: "lut".into(),
        },
        ctx: fixture.ctx,
    }
}

fn unbind_unplaced_cell() -> Refusal {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    Refusal {
        ctx,
        edit: Edit::Unbind { cell },
        expected: BelBindError::CellNotPlaced { cell: "lut".into() },
    }
}

#[test]
fn try_bind_bel_places_cluster_through_root() {
    let fixture = placed_cluster();
    for (cell, expected) in fixture.expected_members() {
        let info = fixture.ctx.design.cell(cell);
        let bel = info.bel.expect("every cluster member is placed");
        assert_eq!(location(&fixture.ctx, bel), expected);
        assert_eq!(info.bel_strength, PlaceStrength::Placer);
        assert_eq!(fixture.ctx.conflicting_bel_cell(bel), Some(cell));
    }
}

type PlacedMembers = (Context, CellId, Vec<(CellId, BelId)>);

fn unbind_lone_cell() -> PlacedMembers {
    let (ctx, first, _, first_bel, _) = placed_luts();
    (ctx, first, vec![(first, first_bel)])
}

fn unbind_cluster_root() -> PlacedMembers {
    let fixture = placed_cluster();
    let members = [fixture.root, fixture.ff, fixture.lut_b]
        .into_iter()
        .map(|cell| (cell, fixture.ctx.design.cell(cell).bel.unwrap()))
        .collect();
    (fixture.ctx, fixture.root, members)
}

#[rstest]
#[case::lone_cell(unbind_lone_cell)]
#[case::cluster_root(unbind_cluster_root)]
fn try_unbind_cell_frees_every_member(#[case] setup: fn() -> PlacedMembers) {
    let (mut ctx, cell, members) = setup();
    ctx.try_unbind_cell(cell).unwrap();
    for (member, bel) in members {
        let info = ctx.design.cell(member);
        assert_eq!((info.bel, info.bel_strength), (None, PlaceStrength::None));
        assert_eq!(ctx.conflicting_bel_cell(bel), None);
    }
}

#[test]
fn try_bind_bel_binds_cell_with_strength() {
    let mut ctx = common::make_example_context();
    let cell = common::add_cell(&mut ctx, "lut", "LUT6");
    let bel = example_bels(&ctx, "LUT6")[0];
    ctx.try_bind_bel(cell, bel, PlaceStrength::User).unwrap();
    assert_eq!(ctx.conflicting_bel_cell(bel), Some(cell));
    assert_eq!(ctx.design.cell(cell).bel, Some(bel));
    assert_eq!(ctx.design.cell(cell).bel_strength, PlaceStrength::User);
}

#[test]
fn try_swap_cells_exchanges_bels_and_keeps_strengths() {
    let (mut ctx, first, second, first_bel, second_bel) = placed_luts();
    ctx.try_swap_cells(first, second).unwrap();
    assert_eq!(ctx.conflicting_bel_cell(first_bel), Some(second));
    assert_eq!(ctx.conflicting_bel_cell(second_bel), Some(first));
    assert_eq!(ctx.design.cell(first).bel, Some(second_bel));
    assert_eq!(ctx.design.cell(second).bel, Some(first_bel));
    assert_eq!(ctx.design.cell(first).bel_strength, PlaceStrength::Placer);
    assert_eq!(ctx.design.cell(second).bel_strength, PlaceStrength::User);
}

#[rstest]
#[case::bind_to_missing_bel(bind_to_missing_bel)]
#[case::bind_to_other_type(bind_to_other_type)]
#[case::bind_bound_cell(bind_bound_cell)]
#[case::bind_to_occupied_bel(bind_to_occupied_bel)]
#[case::bind_to_invalid_location(bind_to_invalid_location)]
#[case::swap_with_unplaced_cell(swap_with_unplaced_cell)]
#[case::swap_across_types(swap_across_types)]
#[case::swap_into_invalid_location(swap_into_invalid_location)]
#[case::bind_cluster_child(bind_cluster_child)]
#[case::bind_cluster_with_member_off_chip(bind_cluster_with_member_off_chip)]
#[case::bind_cluster_with_member_on_other_type(bind_cluster_with_member_on_other_type)]
#[case::bind_cluster_with_bound_member(bind_cluster_with_bound_member)]
#[case::bind_cluster_onto_occupied_member_bel(bind_cluster_onto_occupied_member_bel)]
#[case::bind_cluster_into_invalid_tile(bind_cluster_into_invalid_tile)]
#[case::bind_cluster_root_off_its_absolute_z(bind_cluster_root_off_its_absolute_z)]
#[case::swap_cluster_root(swap_cluster_root)]
#[case::swap_cluster_child(swap_cluster_child)]
#[case::unbind_cluster_child(unbind_cluster_child)]
#[case::unbind_unplaced_cell(unbind_unplaced_cell)]
fn refused_edit_returns_its_error(#[case] setup: fn() -> Refusal) {
    let Refusal {
        mut ctx,
        edit,
        expected,
    } = setup();
    assert_eq!(edit.apply(&mut ctx), Err(expected));
}

#[rstest]
#[case::bind_to_missing_bel(bind_to_missing_bel)]
#[case::bind_to_other_type(bind_to_other_type)]
#[case::bind_bound_cell(bind_bound_cell)]
#[case::bind_to_occupied_bel(bind_to_occupied_bel)]
#[case::bind_to_invalid_location(bind_to_invalid_location)]
#[case::swap_with_unplaced_cell(swap_with_unplaced_cell)]
#[case::swap_across_types(swap_across_types)]
#[case::swap_into_invalid_location(swap_into_invalid_location)]
#[case::bind_cluster_child(bind_cluster_child)]
#[case::bind_cluster_with_member_off_chip(bind_cluster_with_member_off_chip)]
#[case::bind_cluster_with_member_on_other_type(bind_cluster_with_member_on_other_type)]
#[case::bind_cluster_with_bound_member(bind_cluster_with_bound_member)]
#[case::bind_cluster_onto_occupied_member_bel(bind_cluster_onto_occupied_member_bel)]
#[case::bind_cluster_into_invalid_tile(bind_cluster_into_invalid_tile)]
#[case::bind_cluster_root_off_its_absolute_z(bind_cluster_root_off_its_absolute_z)]
#[case::swap_cluster_root(swap_cluster_root)]
#[case::swap_cluster_child(swap_cluster_child)]
#[case::unbind_cluster_child(unbind_cluster_child)]
#[case::unbind_unplaced_cell(unbind_unplaced_cell)]
fn refused_edit_leaves_placement_unchanged(#[case] setup: fn() -> Refusal) {
    let Refusal { mut ctx, edit, .. } = setup();
    let before = placement(&ctx);
    edit.apply(&mut ctx).unwrap_err();
    assert_eq!(placement(&ctx), before);
}

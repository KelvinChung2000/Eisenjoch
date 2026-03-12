mod common;

use nextpnr::netlist::Rect;
use nextpnr::placer::common::initial_placement;
use nextpnr::placer::heap::PlacerHeapCfg;
use nextpnr::placer::sa::PlacerSaCfg;
use nextpnr::placer::{Placer, PlacerHeap, PlacerSa};

// =============================================================
// Rect tests
// =============================================================

#[test]
fn rect_contains_inside() {
    let r = Rect::new(1, 2, 5, 6);
    assert!(r.contains(1, 2));
    assert!(r.contains(5, 6));
    assert!(r.contains(3, 4));
}

#[test]
fn rect_contains_outside() {
    let r = Rect::new(1, 2, 5, 6);
    assert!(!r.contains(0, 4));
    assert!(!r.contains(6, 4));
    assert!(!r.contains(3, 1));
    assert!(!r.contains(3, 7));
}

#[test]
fn rect_area() {
    let r = Rect::new(0, 0, 3, 3);
    assert_eq!(r.area(), 16);
}

// =============================================================
// RegionConstraint tests
// =============================================================

#[test]
fn region_contains_multi_rect() {
    let ctx = common::make_context();
    let name = ctx.id("test_region");
    let mut region = nextpnr::netlist::RegionConstraint::new(name);
    region.rects.push(Rect::new(0, 0, 0, 0));
    region.rects.push(Rect::new(1, 1, 1, 1));

    assert!(region.contains(0, 0));
    assert!(region.contains(1, 1));
    assert!(!region.contains(1, 0));
    assert!(!region.contains(0, 1));
}

#[test]
fn region_bounding_box() {
    let ctx = common::make_context();
    let name = ctx.id("test_region");
    let mut region = nextpnr::netlist::RegionConstraint::new(name);
    region.rects.push(Rect::new(0, 0, 0, 0));
    region.rects.push(Rect::new(1, 1, 1, 1));

    let bbox = region.bounding_box();
    assert!(bbox.is_some());
    let bbox = bbox.unwrap();
    assert_eq!(bbox.x0, 0);
    assert_eq!(bbox.y0, 0);
    assert_eq!(bbox.x1, 1);
    assert_eq!(bbox.y1, 1);
}

#[test]
fn region_bounding_box_empty() {
    let ctx = common::make_context();
    let name = ctx.id("empty_region");
    let region = nextpnr::netlist::RegionConstraint::new(name);
    // Empty region returns zero-area bbox at origin (not None, from the source)
    let _bbox = region.bounding_box();
}

// =============================================================
// Design region management tests
// =============================================================

#[test]
fn design_add_and_lookup_region() {
    let mut ctx = common::make_context();
    let name = ctx.id("pblock_0");
    let idx = ctx.design.add_region(name);
    assert_eq!(idx, 0);

    ctx.design.region_mut(idx).rects.push(Rect::new(0, 0, 1, 1));
    assert_eq!(ctx.design.region(idx).rects.len(), 1);
    assert_eq!(ctx.design.region_by_name(name), Some(0));
}

// =============================================================
// Region-aware initial placement
// =============================================================

#[test]
fn initial_placement_respects_region() {
    let mut ctx = common::make_context_with_cells(2);

    // Constrain cell_0 to tile (0,0) only.
    let region_name = ctx.id("bottom_left");
    let region_idx = ctx.design.add_region(region_name);
    ctx.design
        .region_mut(region_idx)
        .rects
        .push(Rect::new(0, 0, 0, 0));

    let cell0_name = ctx.id("cell_0");
    let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
    ctx.design.cell_edit(cell0_idx).set_region(Some(region_idx));

    initial_placement(&mut ctx).expect("initial placement should succeed");

    // Verify cell_0 is placed at (0,0).
    let cell0 = ctx.cell(cell0_idx);
    let bel = cell0.bel().expect("cell_0 should be placed");
    let loc = bel.loc();
    assert_eq!((loc.x, loc.y), (0, 0), "cell_0 should be in region (0,0)");
}

// =============================================================
// HeAP respects region
// =============================================================

#[test]
fn heap_respects_region_constraint() {
    let mut ctx = common::make_context_with_cells(2);

    // Constrain cell_0 to top-right quadrant.
    let region_name = ctx.id("top_right");
    let region_idx = ctx.design.add_region(region_name);
    ctx.design
        .region_mut(region_idx)
        .rects
        .push(Rect::new(1, 1, 1, 1));

    let cell0_name = ctx.id("cell_0");
    let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
    ctx.design.cell_edit(cell0_idx).set_region(Some(region_idx));

    let cfg = PlacerHeapCfg::default();
    PlacerHeap
        .place(&mut ctx, &cfg)
        .expect("HeAP should succeed");

    let cell0 = ctx.cell(cell0_idx);
    let bel = cell0.bel().expect("cell_0 should be placed");
    let loc = bel.loc();
    assert_eq!(
        (loc.x, loc.y),
        (1, 1),
        "cell_0 should be placed in region (1,1)"
    );
}

// =============================================================
// SA respects region
// =============================================================

#[test]
fn sa_respects_region_constraint() {
    let mut ctx = common::make_context_with_cells(2);

    // Constrain cell_0 to tile (0,0).
    let region_name = ctx.id("bottom_left");
    let region_idx = ctx.design.add_region(region_name);
    ctx.design
        .region_mut(region_idx)
        .rects
        .push(Rect::new(0, 0, 0, 0));

    let cell0_name = ctx.id("cell_0");
    let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
    ctx.design.cell_edit(cell0_idx).set_region(Some(region_idx));

    let cfg = PlacerSaCfg::default();
    PlacerSa
        .place(&mut ctx, &cfg)
        .expect("SA should succeed");

    let cell0 = ctx.cell(cell0_idx);
    let bel = cell0.bel().expect("cell_0 should be placed");
    let loc = bel.loc();
    assert_eq!(
        (loc.x, loc.y),
        (0, 0),
        "cell_0 should be placed in region (0,0)"
    );
}

// =============================================================
// Context region query tests
// =============================================================

#[test]
fn is_bel_in_region() {
    let mut ctx = common::make_context_with_cells(1);

    let region_name = ctx.id("corner");
    let region_idx = ctx.design.add_region(region_name);
    ctx.design
        .region_mut(region_idx)
        .rects
        .push(Rect::new(0, 0, 0, 0));

    // Iterate all bels and check region membership.
    let mut in_region = 0;
    let mut out_region = 0;
    for bel in ctx.bels() {
        if ctx.is_bel_in_region(bel.id(), region_idx) {
            in_region += 1;
        } else {
            out_region += 1;
        }
    }
    assert!(in_region > 0, "some bels should be in region");
    assert!(out_region > 0, "some bels should be outside region");
}

mod common;

use nextpnr::checkpoint;
use nextpnr::placer::heap::PlacerHeapCfg;
use nextpnr::placer::sa::PlacerSaCfg;
use nextpnr::placer::{Placer, PlacerHeap, PlacerSa};
use nextpnr::router::router1::{Router1, Router1Cfg};
use nextpnr::router::Router;

// =============================================================
// place_cells: HeAP incremental placement
// =============================================================

#[test]
fn heap_place_cells_places_only_target() {
    let mut ctx = common::make_context_with_cells(3);
    let cfg = PlacerHeapCfg::default();

    // Place all cells first.
    PlacerHeap
        .place(&mut ctx, &cfg)
        .expect("full placement should succeed");

    // Record cell_0's placement.
    let cell0_name = ctx.id("cell_0");
    let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
    let cell0_bel = ctx.cell(cell0_idx).bel_id().unwrap();

    // Unbind cell_1 and re-place only cell_1.
    let cell1_name = ctx.id("cell_1");
    let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();
    let cell1_bel = ctx.cell(cell1_idx).bel_id().unwrap();
    ctx.unbind_bel(cell1_bel);

    PlacerHeap
        .place_cells(&mut ctx, &cfg, &[cell1_idx])
        .expect("incremental placement should succeed");

    // cell_0 should still be at its original BEL.
    assert_eq!(ctx.cell(cell0_idx).bel_id(), Some(cell0_bel));
    // cell_1 should be placed (somewhere).
    assert!(ctx.cell(cell1_idx).bel_id().is_some());
}

// =============================================================
// place_cells: SA incremental placement
// =============================================================

#[test]
fn sa_place_cells_places_only_target() {
    let mut ctx = common::make_context_with_cells(3);
    let cfg = PlacerSaCfg::default();

    PlacerSa
        .place(&mut ctx, &cfg)
        .expect("full placement should succeed");

    let cell0_name = ctx.id("cell_0");
    let cell0_idx = ctx.design.cell_by_name(cell0_name).unwrap();
    let cell0_bel = ctx.cell(cell0_idx).bel_id().unwrap();

    let cell1_name = ctx.id("cell_1");
    let cell1_idx = ctx.design.cell_by_name(cell1_name).unwrap();
    let cell1_bel = ctx.cell(cell1_idx).bel_id().unwrap();
    ctx.unbind_bel(cell1_bel);

    PlacerSa
        .place_cells(&mut ctx, &cfg, &[cell1_idx])
        .expect("incremental placement should succeed");

    assert_eq!(ctx.cell(cell0_idx).bel_id(), Some(cell0_bel));
    assert!(ctx.cell(cell1_idx).bel_id().is_some());
}

// =============================================================
// route_net: Router1 on empty design (no routable nets)
// =============================================================

#[test]
fn router1_route_net_on_empty_design() {
    // Synthetic chipdb doesn't have BEL pin wires, so we test with
    // an empty design (no nets to route) which exercises the trait
    // method dispatch without hitting the missing-wire limitation.
    let mut ctx = common::make_context();
    let cfg = Router1Cfg::default();
    Router1
        .route(&mut ctx, &cfg)
        .expect("routing empty design should succeed");
}

// =============================================================
// route_nets: default delegates to route_net
// =============================================================

#[test]
fn router1_route_nets_empty() {
    let mut ctx = common::make_context();
    let cfg = Router1Cfg::default();
    Router1
        .route_nets(&mut ctx, &cfg, &[])
        .expect("route_nets with empty list should succeed");
}

// =============================================================
// Checkpoint save/restore roundtrip (placement only)
// =============================================================

#[test]
fn save_restore_roundtrip() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerHeapCfg::default();
    PlacerHeap.place(&mut ctx, &cfg).expect("place");

    // Save checkpoint.
    let tmp_dir = std::env::temp_dir().join("nextpnr_test_incr");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let cp_path = tmp_dir.join("test.json");
    checkpoint::save(&ctx, &cp_path).expect("save");

    // Create a fresh context with the same design.
    let mut ctx2 = common::make_context_with_cells(2);

    // Load and restore.
    let cp = checkpoint::Checkpoint::load_from_file(&cp_path).expect("load");
    let report = checkpoint::restore(&mut ctx2, &cp).expect("restore");

    // All cells should be restored (same design).
    assert_eq!(report.cells_restored, 2);
    assert_eq!(report.cells_skipped, 0);
    assert!(report.cells_to_place.is_empty());

    // Verify cells are actually placed.
    for (_ci, cell) in ctx2.design.iter_alive_cells() {
        assert!(cell.bel.is_some(), "restored cells should be placed");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// =============================================================
// Normal placement works after restore (restored cells are Fixed)
// =============================================================

#[test]
fn place_after_restore_skips_fixed_cells() {
    let mut ctx = common::make_context_with_cells(2);
    let cfg = PlacerHeapCfg::default();
    PlacerHeap.place(&mut ctx, &cfg).expect("place");

    let tmp_dir = std::env::temp_dir().join("nextpnr_test_place_after_restore");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let cp_path = tmp_dir.join("test.json");
    checkpoint::save(&ctx, &cp_path).expect("save");

    // Fresh context, restore, then run normal placement.
    let mut ctx2 = common::make_context_with_cells(2);
    let cp = checkpoint::Checkpoint::load_from_file(&cp_path).expect("load");
    checkpoint::restore(&mut ctx2, &cp).expect("restore");

    // Normal place() should succeed (all cells already Fixed).
    PlacerHeap
        .place(&mut ctx2, &cfg)
        .expect("placement after restore should succeed");

    for (_ci, cell) in ctx2.design.iter_alive_cells() {
        assert!(cell.bel.is_some());
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// =============================================================
// Partial restore: checkpoint has more cells than current design
// =============================================================

#[test]
fn partial_restore_extra_checkpoint_cells() {
    // Place 3 cells.
    let mut ctx = common::make_context_with_cells(3);
    let cfg = PlacerHeapCfg::default();
    PlacerHeap.place(&mut ctx, &cfg).expect("place");

    // Save checkpoint.
    let tmp_dir = std::env::temp_dir().join("nextpnr_test_incr_partial");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let cp_path = tmp_dir.join("test.json");
    checkpoint::save(&ctx, &cp_path).expect("save");

    // Create fresh context with only 2 of the 3 original cells.
    let mut ctx2 = common::make_context_with_cells(2);

    let cp = checkpoint::Checkpoint::load_from_file(&cp_path).expect("load");
    let report = checkpoint::restore(&mut ctx2, &cp).expect("restore");

    // 2 cells restored (cell_0, cell_1), cell_2 was in checkpoint but not in new design.
    assert_eq!(report.cells_restored, 2);
    assert!(report.cells_to_place.is_empty());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

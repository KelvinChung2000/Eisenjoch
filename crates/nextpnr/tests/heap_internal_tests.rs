mod common;

use nextpnr::placer::heap::{count_bels_in_region, place_heap, HeapState, PlacerHeapCfg};
use nextpnr::placer::solver::{conjugate_gradient, spmv, Solver, SparseSystem};

#[test]
fn sparse_system_new() {
    let sys = SparseSystem::new(3);
    assert_eq!(sys.n, 3);
    assert_eq!(sys.diag.len(), 3);
    assert_eq!(sys.rhs.len(), 3);
    assert!(sys.off_diag.is_empty());
}

#[test]
fn sparse_system_add_connection() {
    let mut sys = SparseSystem::new(3);
    sys.add_connection(0, 2, 5.0);
    assert_eq!(sys.diag[0], 5.0);
    assert_eq!(sys.diag[2], 5.0);
    assert_eq!(sys.off_diag[0], (0, 2, -5.0));
}

#[test]
fn sparse_system_add_connection_self_is_noop() {
    let mut sys = SparseSystem::new(2);
    sys.add_connection(1, 1, 3.0);
    assert_eq!(sys.diag, vec![0.0, 0.0]);
    assert!(sys.off_diag.is_empty());
}

#[test]
fn sparse_system_add_anchor() {
    let mut sys = SparseSystem::new(2);
    sys.add_anchor(0, 3.0, 2.0);
    assert_eq!(sys.diag[0], 2.0);
    assert_eq!(sys.rhs[0], 6.0);
}

#[test]
fn sparse_system_solve_identity() {
    let mut sys = SparseSystem::new(3);
    sys.diag = vec![1.0, 1.0, 1.0];
    sys.rhs = vec![2.0, 5.0, -1.0];
    let mut x = vec![0.0; 3];
    let iters = sys.solve(&mut x, 1e-10, 100);
    assert!((x[0] - 2.0).abs() < 1e-6);
    assert!((x[1] - 5.0).abs() < 1e-6);
    assert!((x[2] + 1.0).abs() < 1e-6);
    assert!(iters <= 3);
}

#[test]
fn sparse_system_solve_with_connections() {
    let mut sys = SparseSystem::new(2);
    sys.add_connection(0, 1, 1.0);
    sys.add_anchor(0, 0.0, 1.0);
    sys.add_anchor(1, 4.0, 1.0);
    let mut x = vec![0.0; 2];
    sys.solve(&mut x, 1e-10, 100);
    assert!((x[0] - 4.0 / 3.0).abs() < 1e-6);
    assert!((x[1] - 8.0 / 3.0).abs() < 1e-6);
}

#[test]
fn cg_identity_system() {
    let mut x = vec![0.0, 0.0, 0.0];
    let iters = conjugate_gradient(&[1.0, 1.0, 1.0], &[], &[3.0, 7.0, -2.0], &mut x, 1e-10, 100);
    assert!((x[0] - 3.0).abs() < 1e-6);
    assert!((x[1] - 7.0).abs() < 1e-6);
    assert!((x[2] + 2.0).abs() < 1e-6);
    assert!(iters <= 3);
}

#[test]
fn cg_diagonal_system() {
    let mut x = vec![0.0, 0.0, 0.0];
    conjugate_gradient(&[2.0, 3.0, 5.0], &[], &[4.0, 9.0, 25.0], &mut x, 1e-10, 100);
    assert!((x[0] - 2.0).abs() < 1e-6);
    assert!((x[1] - 3.0).abs() < 1e-6);
    assert!((x[2] - 5.0).abs() < 1e-6);
}

#[test]
fn cg_empty_system() {
    let mut x: Vec<f64> = vec![];
    assert_eq!(conjugate_gradient(&[], &[], &[], &mut x, 1e-10, 100), 0);
}

#[test]
fn cg_single_variable() {
    let mut x = vec![0.0];
    conjugate_gradient(&[4.0], &[], &[12.0], &mut x, 1e-10, 100);
    assert!((x[0] - 3.0).abs() < 1e-6);
}

#[test]
fn cg_with_off_diagonal() {
    let mut x = vec![0.0, 0.0];
    conjugate_gradient(
        &[4.0, 4.0],
        &[(0, 1, -1.0)],
        &[3.0, 3.0],
        &mut x,
        1e-10,
        100,
    );
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] - 1.0).abs() < 1e-6);
}

#[test]
fn spmv_identity() {
    let mut result = vec![0.0; 3];
    spmv(&[1.0, 1.0, 1.0], &[], &[3.0, 5.0, 7.0], &mut result);
    assert_eq!(result, vec![3.0, 5.0, 7.0]);
}

#[test]
fn spmv_diagonal() {
    let mut result = vec![0.0; 3];
    spmv(&[2.0, 3.0, 4.0], &[], &[1.0, 2.0, 3.0], &mut result);
    assert_eq!(result, vec![2.0, 6.0, 12.0]);
}

#[test]
fn spmv_with_off_diagonal() {
    let mut result = vec![0.0; 2];
    spmv(&[2.0, 3.0], &[(0, 1, -1.0)], &[1.0, 2.0], &mut result);
    assert_eq!(result, vec![0.0, 5.0]);
}

#[test]
fn spmv_symmetric() {
    let mut result = vec![0.0; 3];
    spmv(
        &[4.0, 4.0, 4.0],
        &[(0, 1, -1.0), (1, 2, -1.0)],
        &[1.0, 2.0, 3.0],
        &mut result,
    );
    assert_eq!(result, vec![2.0, 4.0, 10.0]);
}

#[test]
fn spreading_no_cells() {
    let ctx = common::make_context();
    let mut state = HeapState::new(&ctx, &PlacerHeapCfg::default()).unwrap();
    assert_eq!(state.spread(&ctx).unwrap(), 1.0);
}

#[test]
fn spreading_cells_fit() {
    let ctx = common::make_context_with_cells(4);
    let mut state = HeapState::new(&ctx, &PlacerHeapCfg::default()).unwrap();
    state.cell_x = vec![0.0, 1.0, 0.0, 1.0];
    state.cell_y = vec![0.0, 0.0, 1.0, 1.0];
    assert!(state.spread(&ctx).unwrap() >= 0.9);
}

#[test]
fn spreading_clustered_cells() {
    let ctx = common::make_context_with_cells(3);
    let mut state = HeapState::new(&ctx, &PlacerHeapCfg::default()).unwrap();
    state.cell_x = vec![0.0, 0.0, 0.0];
    state.cell_y = vec![0.0, 0.0, 0.0];
    assert!(state.spread(&ctx).unwrap() > 0.0);
}

#[test]
fn count_bels_full_grid() {
    let ctx = common::make_context();
    assert_eq!(count_bels_in_region(&ctx, 0, 0, 1, 1), 4);
}

#[test]
fn count_bels_single_tile() {
    let ctx = common::make_context();
    assert_eq!(count_bels_in_region(&ctx, 0, 0, 0, 0), 1);
}

#[test]
fn count_bels_row() {
    let ctx = common::make_context();
    assert_eq!(count_bels_in_region(&ctx, 0, 0, 1, 0), 2);
}

#[test]
fn count_bels_empty_region() {
    let ctx = common::make_context();
    assert_eq!(count_bels_in_region(&ctx, 5, 5, 10, 10), 0);
}

#[test]
fn full_heap_run_smoke() {
    let mut ctx = common::make_context_with_cells(4);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 5,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).unwrap();
    for cell in ctx.cells() {
        if cell.is_alive() {
            assert!(cell.bel_id().is_some());
        }
    }
}

#[test]
fn cg_jacobi_preconditioned_fewer_iters() {
    // A diagonally-dominant system where Jacobi preconditioning should help.
    // A = [[10, -1], [-1, 10]], b = [9, 9] => x = [1, 1]
    let mut x_precond = vec![0.0, 0.0];
    let iters = conjugate_gradient(
        &[10.0, 10.0],
        &[(0, 1, -1.0)],
        &[9.0, 9.0],
        &mut x_precond,
        1e-10,
        100,
    );
    assert!((x_precond[0] - 1.0).abs() < 1e-6);
    assert!((x_precond[1] - 1.0).abs() < 1e-6);
    // Preconditioned CG on a 2x2 system should converge very quickly.
    assert!(iters <= 2, "expected <= 2 iters, got {}", iters);
}

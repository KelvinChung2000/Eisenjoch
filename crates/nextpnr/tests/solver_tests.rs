//! Tests for the sparse linear solver infrastructure:
//! CSC format, nested dissection, Cholesky factorization, direct solver, AMG preconditioner.

use nextpnr::placer::solver::amg::{self, AmgPreconditioner, AmgStructure};
use nextpnr::placer::solver::cholesky::{numeric_ldlt, symbolic_cholesky};
use nextpnr::placer::solver::cg::{conjugate_gradient, spmv};
use nextpnr::placer::solver::direct::DirectSolver;
use nextpnr::placer::solver::ordering::{expand_to_ports, inverse_perm, nested_dissection_2d};
use nextpnr::placer::solver::sparse::from_diag_and_upper_coo;

// ============================================================
// Shared test utilities
// ============================================================

/// Verify ||Ax - b|| / ||b|| < tol using the original diag+off_diag format.
fn check_residual(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    x: &[f64],
    rhs: &[f64],
    tol: f64,
    label: &str,
) {
    let n = diag.len();
    let mut ax = vec![0.0; n];
    spmv(diag, off_diag, x, &mut ax);
    let res_norm: f64 = rhs
        .iter()
        .zip(ax.iter())
        .map(|(b, a)| (b - a) * (b - a))
        .sum::<f64>()
        .sqrt();
    let rhs_norm: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-30);
    let rel = res_norm / rhs_norm;
    assert!(
        rel < tol,
        "{}: relative residual {:.2e} exceeds {:.2e}",
        label,
        rel,
        tol
    );
}

/// Build a 2D grid Laplacian with 5-point stencil, anchored at node 0.
fn make_grid_laplacian(w: usize, h: usize) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let n = w * h;
    let mut diag = vec![0.0f64; n];
    let mut off_diag = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                let j = y * w + x + 1;
                diag[i] += 1.0;
                diag[j] += 1.0;
                off_diag.push((i, j, -1.0));
            }
            if y + 1 < h {
                let j = (y + 1) * w + x;
                diag[i] += 1.0;
                diag[j] += 1.0;
                off_diag.push((i, j, -1.0));
            }
        }
    }
    diag[0] += 1e6;
    (diag, off_diag)
}

/// Cholesky solve pipeline: CSC -> symbolic -> numeric -> solve.
fn cholesky_solve(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
) -> Vec<f64> {
    let n = diag.len();
    let (csc, _) = from_diag_and_upper_coo(n, diag, off_diag);
    let sym = symbolic_cholesky(&csc);
    let num = numeric_ldlt(&sym, &csc);
    let mut x = vec![0.0; n];
    num.solve(rhs, &mut x);
    x
}

/// Build a 4-port-per-tile grid system mimicking the Kirchhoff pipe network.
fn make_4port_grid(
    w: usize,
    h: usize,
) -> (usize, Vec<f64>, Vec<(usize, usize, f64)>, Vec<(usize, usize)>) {
    let n = w * h * 4;
    let mut diag = vec![0.0f64; n];
    let mut off_diag = Vec::new();
    let mut endpoints = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            for p1 in 0..4 {
                for p2 in (p1 + 1)..4 {
                    let i = base + p1;
                    let j = base + p2;
                    diag[i] += 0.5;
                    diag[j] += 0.5;
                    off_diag.push((i, j, -0.5));
                    endpoints.push((i, j));
                }
            }
            if x + 1 < w {
                let i = base + 1;
                let j = (y * w + x + 1) * 4 + 3;
                diag[i] += 1.0;
                diag[j] += 1.0;
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                off_diag.push((lo, hi, -1.0));
                endpoints.push((lo, hi));
            }
            if y + 1 < h {
                let i = base + 2;
                let j = ((y + 1) * w + x) * 4;
                diag[i] += 1.0;
                diag[j] += 1.0;
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                off_diag.push((lo, hi, -1.0));
                endpoints.push((lo, hi));
            }
        }
    }
    diag[0] += 1e10;
    (n, diag, off_diag, endpoints)
}

// ============================================================
// CSC sparse matrix tests
// ============================================================

#[test]
fn csc_from_3x3_laplacian() {
    let diag = vec![1.0, 2.0, 1.0];
    let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
    let (csc, pos_map) = from_diag_and_upper_coo(3, &diag, &off_diag);
    assert_eq!(csc.n, 3);
    assert_eq!(csc.col_ptrs, vec![0, 2, 4, 5]);
    assert_eq!(csc.row_indices, vec![0, 1, 1, 2, 2]);
    assert_eq!(csc.values, vec![1.0, -1.0, 2.0, -1.0, 1.0]);
    assert_eq!(pos_map.diag_positions.len(), 3);
    assert_eq!(pos_map.offdiag_positions.len(), 2);
}

#[test]
fn csc_4node_complete() {
    let diag = vec![3.0, 3.0, 3.0, 3.0];
    let off_diag = vec![
        (0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0),
        (1, 2, -1.0), (1, 3, -1.0), (2, 3, -1.0),
    ];
    let (csc, _) = from_diag_and_upper_coo(4, &diag, &off_diag);
    assert_eq!(csc.col_ptrs, vec![0, 4, 7, 9, 10]);
    assert_eq!(csc.nnz(), 10);
}

#[test]
fn csc_update_values() {
    use nextpnr::placer::solver::sparse::update_values;
    let diag = vec![1.0, 2.0, 1.0];
    let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
    let (mut csc, pos_map) = from_diag_and_upper_coo(3, &diag, &off_diag);
    let new_diag = vec![3.0, 5.0, 3.0];
    let new_off_diag = vec![(0, 1, -2.0), (1, 2, -3.0)];
    update_values(&mut csc, &pos_map, &new_diag, &new_off_diag);
    assert_eq!(csc.values, vec![3.0, -2.0, 5.0, -3.0, 3.0]);
}

// ============================================================
// Nested dissection ordering tests
// ============================================================

#[test]
fn nd_valid_permutation_8x8() {
    let perm = nested_dissection_2d(8, 8);
    assert_eq!(perm.len(), 64);
    let mut sorted = perm.clone();
    sorted.sort();
    assert_eq!(sorted, (0..64).collect::<Vec<_>>());
}

#[test]
fn nd_valid_permutation_81x81() {
    let perm = nested_dissection_2d(81, 81);
    assert_eq!(perm.len(), 81 * 81);
    let mut sorted = perm.clone();
    sorted.sort();
    assert_eq!(sorted, (0..81 * 81).collect::<Vec<_>>());
}

#[test]
fn nd_separator_last() {
    let perm = nested_dissection_2d(9, 1);
    assert_eq!(*perm.last().unwrap(), 4);
}

#[test]
fn nd_rectangular_grid() {
    let perm = nested_dissection_2d(20, 5);
    assert_eq!(perm.len(), 100);
    let mut sorted = perm.clone();
    sorted.sort();
    assert_eq!(sorted, (0..100).collect::<Vec<_>>());
}

#[test]
fn expand_ports_consistency() {
    let tile_perm = vec![2, 0, 1];
    let port_perm = expand_to_ports(&tile_perm);
    assert_eq!(&port_perm[0..4], &[8, 9, 10, 11]);
    assert_eq!(&port_perm[4..8], &[0, 1, 2, 3]);
    assert_eq!(&port_perm[8..12], &[4, 5, 6, 7]);
}

#[test]
fn inverse_perm_roundtrip() {
    let perm = nested_dissection_2d(8, 8);
    let inv = inverse_perm(&perm);
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        assert_eq!(inv[old_idx], new_idx);
    }
}

// ============================================================
// Cholesky factorization tests
// ============================================================

#[test]
fn cholesky_symbolic_3x3_tridiagonal() {
    let diag = vec![1.0 + 1e6, 2.0, 1.0];
    let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
    let (csc, _) = from_diag_and_upper_coo(3, &diag, &off_diag);
    let sym = symbolic_cholesky(&csc);
    assert_eq!(sym.l_nnz(), 2);
    assert_eq!(sym.etree[0], Some(1));
    assert_eq!(sym.etree[1], Some(2));
    assert_eq!(sym.etree[2], None);
}

#[test]
fn cholesky_symbolic_arrowhead() {
    let diag = vec![3.0, 1.0, 1.0, 1.0];
    let off_diag = vec![(0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0)];
    let (csc, _) = from_diag_and_upper_coo(4, &diag, &off_diag);
    let sym = symbolic_cholesky(&csc);
    assert_eq!(sym.l_nnz(), 6); // dense below column 0
}

#[test]
fn cholesky_solve_1x1() {
    let x = cholesky_solve(&[5.0], &[], &[10.0]);
    assert!((x[0] - 2.0).abs() < 1e-12);
}

#[test]
fn cholesky_solve_2x2() {
    let x = cholesky_solve(&[4.0, 4.0], &[(0, 1, -1.0)], &[3.0, 3.0]);
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 1.0).abs() < 1e-12);
}

#[test]
fn cholesky_solve_8x8_grid() {
    let (diag, off_diag) = make_grid_laplacian(8, 8);
    let rhs: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin()).collect();
    let x = cholesky_solve(&diag, &off_diag, &rhs);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "cholesky 8x8");
}

#[test]
fn cholesky_solve_32x32_grid() {
    let (diag, off_diag) = make_grid_laplacian(32, 32);
    let rhs: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.13).cos()).collect();
    let x = cholesky_solve(&diag, &off_diag, &rhs);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "cholesky 32x32");
}

#[test]
fn cholesky_matches_cg() {
    let (diag, off_diag) = make_grid_laplacian(8, 8);
    let rhs: Vec<f64> = (0..64).map(|i| (i as f64 * 0.5).sin()).collect();
    let x_direct = cholesky_solve(&diag, &off_diag, &rhs);
    let mut x_cg = vec![0.0; 64];
    conjugate_gradient(&diag, &off_diag, &rhs, &mut x_cg, 1e-12, 5000);
    let max_diff: f64 = x_direct.iter().zip(x_cg.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    assert!(max_diff < 1e-4, "direct vs CG max diff = {:.2e}", max_diff);
}

#[test]
fn cholesky_high_anchor_weight() {
    let (mut diag, off_diag) = make_grid_laplacian(8, 8);
    diag[0] = 1e10;
    let rhs: Vec<f64> = (0..64).map(|i| if i == 0 { 0.0 } else { (i as f64).sin() }).collect();
    let x = cholesky_solve(&diag, &off_diag, &rhs);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-4, "high anchor");
    assert!(x[0].abs() < 1e-4);
}

#[test]
fn cholesky_refactorize_different_values() {
    let (diag1, off_diag1) = make_grid_laplacian(8, 8);
    let n = 64;
    let (csc1, _) = from_diag_and_upper_coo(n, &diag1, &off_diag1);
    let sym = symbolic_cholesky(&csc1);

    let num1 = numeric_ldlt(&sym, &csc1);
    let rhs = vec![1.0; n];
    let mut x1 = vec![0.0; n];
    num1.solve(&rhs, &mut x1);
    check_residual(&diag1, &off_diag1, &x1, &rhs, 1e-6, "refactor 1");

    let mut diag2 = diag1.clone();
    for i in 0..n { diag2[i] += 0.5 * (i as f64 + 1.0); }
    let (csc2, _) = from_diag_and_upper_coo(n, &diag2, &off_diag1);
    let num2 = numeric_ldlt(&sym, &csc2);
    let mut x2 = vec![0.0; n];
    num2.solve(&rhs, &mut x2);
    check_residual(&diag2, &off_diag1, &x2, &rhs, 1e-6, "refactor 2");

    let diff: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6);
}

// ============================================================
// Direct solver (Cholesky + permutation) tests
// ============================================================

#[test]
fn direct_solver_4x4_grid() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(4, 4);
    let mut solver = DirectSolver::new(n, &endpoints, 4, 4);
    let mut rhs = vec![0.0; n]; rhs[0] = 1.0;
    let mut x = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-8, "direct 4x4");
}

#[test]
fn direct_solver_8x8_grid() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(8, 8);
    let mut solver = DirectSolver::new(n, &endpoints, 8, 8);
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut x = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "direct 8x8");
}

#[test]
fn direct_solver_reuse_values() {
    let (n, mut diag, off_diag, endpoints) = make_4port_grid(4, 4);
    let mut solver = DirectSolver::new(n, &endpoints, 4, 4);
    let rhs = vec![1.0; n];
    let mut x1 = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x1);
    check_residual(&diag, &off_diag, &x1, &rhs, 1e-8, "reuse 1");

    diag[1] += 10.0;
    let mut x2 = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x2);
    check_residual(&diag, &off_diag, &x2, &rhs, 1e-8, "reuse 2");

    let diff: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-10);
}

#[test]
fn direct_solver_matches_cg() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(5, 5);
    let mut solver = DirectSolver::new(n, &endpoints, 5, 5);
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
    let mut x_direct = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x_direct);
    let mut x_cg = vec![0.0; n];
    conjugate_gradient(&diag, &off_diag, &rhs, &mut x_cg, 1e-12, 10000);
    let max_diff: f64 = x_direct.iter().zip(x_cg.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    assert!(max_diff < 1e-4, "direct vs CG max diff = {:.2e}", max_diff);
}

// ============================================================
// AMG preconditioner tests
// ============================================================

#[test]
fn amg_hierarchy_builds() {
    let (_diag, off_diag) = make_grid_laplacian(16, 16);
    let pairs: Vec<_> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
    let structure = AmgStructure::new(256, &pairs);
    assert!(structure.num_levels() >= 2);
}

#[test]
fn amg_cg_8x8_grid() {
    let (diag, off_diag) = make_grid_laplacian(8, 8);
    let rhs: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut x = vec![0.0; 64];
    let iters = amg::amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
    assert!(iters < 500, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg 8x8");
}

#[test]
fn amg_cg_32x32_grid() {
    let (diag, off_diag) = make_grid_laplacian(32, 32);
    let rhs: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.13).cos()).collect();
    let mut x = vec![0.0; 1024];
    let iters = amg::amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
    assert!(iters < 200, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg 32x32");
}

#[test]
fn amg_cached_reuse() {
    let (diag, off_diag) = make_grid_laplacian(16, 16);
    let n = 256;
    let pairs: Vec<_> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
    let structure = AmgStructure::new(n, &pairs);
    let mut amg_pc = AmgPreconditioner::new(structure, &diag, &off_diag);

    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut x = vec![0.0; n];
    let iters1 = amg::amg_preconditioned_cg_cached(&mut amg_pc, &diag, &off_diag, &rhs, &mut x, 1e-3, 500);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "cached 1");

    let mut diag2 = diag.clone();
    for d in &mut diag2 { *d += 0.5; }
    let rhs2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
    let mut x2 = vec![0.0; n];
    let iters2 = amg::amg_preconditioned_cg_cached(&mut amg_pc, &diag2, &off_diag, &rhs2, &mut x2, 1e-3, 500);
    check_residual(&diag2, &off_diag, &x2, &rhs2, 1e-3, "cached 2");
    assert!(iters1 < 500 && iters2 < 500);
}

#[test]
fn amg_cg_nonuniform_conductance() {
    let (n, w, h) = (100, 10, 10);
    let mut diag = vec![0.0f64; n];
    let mut off_diag = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                let j = y * w + x + 1;
                let g = if (x + y) % 3 == 0 { 0.001 } else { 10.0 };
                diag[i] += g; diag[j] += g;
                off_diag.push((i, j, -g));
            }
            if y + 1 < h {
                let j = (y + 1) * w + x;
                let g = if (x * y) % 5 == 0 { 0.01 } else { 5.0 };
                diag[i] += g; diag[j] += g;
                off_diag.push((i, j, -g));
            }
        }
    }
    diag[0] += 1e6;
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let mut x = vec![0.0; n];
    let iters = amg::amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
    assert!(iters < 500, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg nonuniform");
}

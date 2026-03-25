//! Tests for the solver infrastructure:
//! faer-backed direct Cholesky solver, CG solver, system builder.

use nextpnr::placer::solver::faer_backend::{faer_cg, FaerDirectSolver};
use nextpnr::placer::solver::system::SparseSystemBuilder;
use nextpnr::placer::solver::Solver;

// ============================================================
// Shared test utilities
// ============================================================

/// Symmetric sparse matrix-vector product: result = A * x.
fn spmv(diag: &[f64], off_diag: &[(usize, usize, f64)], x: &[f64], result: &mut [f64]) {
    let n = diag.len();
    for i in 0..n {
        result[i] = diag[i] * x[i];
    }
    for &(i, j, w) in off_diag {
        result[i] += w * x[j];
        result[j] += w * x[i];
    }
}

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
// CG solver tests
// ============================================================

#[test]
fn cg_diagonal_system() {
    let diag = vec![2.0, 3.0];
    let off_diag = vec![];
    let rhs = vec![4.0, 9.0];
    let mut x = vec![0.0, 0.0];
    let iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
    assert!(iters <= 2);
    assert!((x[0] - 2.0).abs() < 1e-8);
    assert!((x[1] - 3.0).abs() < 1e-8);
}

#[test]
fn cg_8x8_grid() {
    let (diag, off_diag) = make_grid_laplacian(8, 8);
    let rhs: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut x = vec![0.0; 64];
    let iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-6, 5000);
    assert!(iters < 5000, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-5, "cg 8x8");
}

#[test]
fn cg_32x32_grid() {
    let (diag, off_diag) = make_grid_laplacian(32, 32);
    let rhs: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.13).cos()).collect();
    let mut x = vec![0.0; 1024];
    let iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 5000);
    assert!(iters < 5000, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "cg 32x32");
}

// ============================================================
// Direct solver (faer Cholesky) tests
// ============================================================

#[test]
fn direct_solver_4x4_grid() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(4, 4);
    let mut solver = FaerDirectSolver::new(n, &endpoints);
    let mut rhs = vec![0.0; n]; rhs[0] = 1.0;
    let mut x = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "direct 4x4");
}

#[test]
fn direct_solver_8x8_grid() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(8, 8);
    let mut solver = FaerDirectSolver::new(n, &endpoints);
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut x = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "direct 8x8");
}

#[test]
fn direct_solver_reuse_values() {
    let (n, mut diag, off_diag, endpoints) = make_4port_grid(4, 4);
    let mut solver = FaerDirectSolver::new(n, &endpoints);
    let rhs = vec![1.0; n];
    let mut x1 = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x1);
    check_residual(&diag, &off_diag, &x1, &rhs, 1e-6, "reuse 1");

    diag[1] += 10.0;
    let mut x2 = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x2);
    check_residual(&diag, &off_diag, &x2, &rhs, 1e-6, "reuse 2");

    let diff: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-10);
}

#[test]
fn direct_solver_matches_cg() {
    let (n, diag, off_diag, endpoints) = make_4port_grid(5, 5);
    let mut solver = FaerDirectSolver::new(n, &endpoints);
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
    let mut x_direct = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x_direct);
    let mut x_cg = vec![0.0; n];
    faer_cg(&diag, &off_diag, &rhs, &mut x_cg, 1e-12, 10000);
    let max_diff: f64 = x_direct.iter().zip(x_cg.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-4, "direct vs CG max diff = {:.2e}", max_diff);
}

#[test]
fn direct_solver_high_anchor_weight() {
    let (diag, off_diag) = make_grid_laplacian(8, 8);
    let endpoints: Vec<_> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
    let n = diag.len();
    let mut solver = FaerDirectSolver::new(n, &endpoints);
    let rhs: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { (i as f64).sin() }).collect();
    let mut x = vec![0.0; n];
    solver.solve(&diag, &off_diag, &rhs, &mut x);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-4, "high anchor");
    assert!(x[0].abs() < 1e-4);
}

// ============================================================
// SparseSystemBuilder tests
// ============================================================

#[test]
fn system_builder_basic() {
    let mut sys = SparseSystemBuilder::new(3);
    sys.add_connection(0, 1, 1.0);
    sys.add_connection(1, 2, 1.0);
    sys.add_anchor(0, 0.0, 1e6);
    sys.add_anchor(2, 10.0, 1.0);

    let mut x = vec![5.0; 3];
    let iters = sys.solve_cg(&mut x, 1e-10, 1000);
    assert!(iters < 1000);
    // Node 0 should be near 0 (strong anchor), node 2 near 10 (weak anchor)
    assert!(x[0].abs() < 0.01, "x[0] = {}", x[0]);
}

#[test]
fn system_builder_solver_trait() {
    let mut sys = SparseSystemBuilder::new(2);
    sys.add_anchor(0, 5.0, 1.0);
    sys.add_anchor(1, 10.0, 1.0);

    let mut x = vec![0.0; 2];
    let iters = sys.solve(&mut x, 1e-10, 100);
    assert!(iters <= 2);
    assert!((x[0] - 5.0).abs() < 1e-6);
    assert!((x[1] - 10.0).abs() < 1e-6);
}

#[test]
fn system_builder_clear_reuse() {
    let mut sys = SparseSystemBuilder::new(2);
    sys.add_anchor(0, 5.0, 1.0);
    sys.add_anchor(1, 10.0, 1.0);
    let mut x = vec![0.0; 2];
    sys.solve_cg(&mut x, 1e-10, 100);

    sys.clear();
    sys.add_anchor(0, 100.0, 1.0);
    sys.add_anchor(1, 200.0, 1.0);
    let mut x2 = vec![0.0; 2];
    sys.solve_cg(&mut x2, 1e-10, 100);
    assert!((x2[0] - 100.0).abs() < 1e-6);
    assert!((x2[1] - 200.0).abs() < 1e-6);
}

// ============================================================
// CG nonuniform conductance test
// ============================================================

#[test]
fn cg_nonuniform_conductance() {
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
    let iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
    assert!(iters < 500, "got {} iters", iters);
    check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "cg nonuniform");
}

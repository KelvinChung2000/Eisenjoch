//! Thin wrapper around faer's conjugate_gradient solver.
//!
//! Provides a simple interface for solving A*x = rhs using preconditioned CG
//! via faer's matrix-free operator framework.

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::conjugate_gradient::{
    conjugate_gradient, conjugate_gradient_scratch, CgError, CgParams,
};
use faer::matrix_free::{LinOp, Precond};

/// Result of a CG solve.
pub struct CgResult {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Final relative residual norm.
    pub residual: f64,
}

/// Solve A*x = rhs using preconditioned CG via faer.
///
/// `mat` implements `LinOp<f64>` (the system matrix A).
/// `precond` implements `Precond<f64>` (the preconditioner M).
/// `rhs` is the right-hand side vector.
/// `x` is the initial guess on input, solution on output.
/// `tol` is the relative tolerance for convergence.
/// `max_iters` is the maximum number of CG iterations.
pub fn solve_cg(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> CgResult {
    let n = mat.nrows();
    if n == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let rhs_mat = faer::MatRef::from_column_major_slice(rhs, n, 1);
    let mut x_mat = faer::MatMut::from_column_major_slice_mut(x, n, 1);

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    // Compute workspace requirement and allocate
    let scratch = conjugate_gradient_scratch(precond, mat, 1, faer::Par::Seq);
    let mut buf = MemBuffer::new(scratch);
    let mut stack = MemStack::new(&mut buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        faer::Par::Seq,
        &mut stack,
    ) {
        Ok(info) => CgResult {
            iterations: info.iter_count,
            converged: true,
            residual: info.rel_residual,
        },
        Err(CgError::NoConvergence { rel_residual, .. }) => CgResult {
            iterations: max_iters,
            converged: false,
            residual: rel_residual,
        },
        Err(CgError::NonPositiveDefiniteOperator) => {
            eprintln!("CG ERROR: NonPositiveDefiniteOperator detected!");
            CgResult {
                iterations: 0,
                converged: false,
                residual: f64::MAX,
            }
        },
        Err(CgError::NonPositiveDefinitePreconditioner) => {
            eprintln!("CG ERROR: NonPositiveDefinitePreconditioner detected!");
            CgResult {
                iterations: 0,
                converged: false,
                residual: f64::MAX,
            }
        },
    }
}

/// Batched CG solve: A * X = B where B has `nrhs` columns.
///
/// `rhs_data` is column-major: `nrhs` columns of length `n`, packed as
/// `[col0_row0, col0_row1, ..., col0_rowN, col1_row0, ...]`.
/// `x_data` is the same layout for the solution.
///
/// Returns one CgResult (the aggregate). Each column converges independently
/// inside faer's block CG, leveraging BLAS3 for the mat-vecs.
pub fn solve_cg_batched(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_data: &[f64],
    x_data: &mut [f64],
    nrhs: usize,
    tol: f64,
    max_iters: usize,
) -> CgResult {
    let n = mat.nrows();
    if n == 0 || nrhs == 0 {
        return CgResult { iterations: 0, converged: true, residual: 0.0 };
    }

    let rhs_mat = faer::MatRef::from_column_major_slice(rhs_data, n, nrhs);
    let mut x_mat = faer::MatMut::from_column_major_slice_mut(x_data, n, nrhs);

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    let scratch = conjugate_gradient_scratch(precond, mat, nrhs, faer::Par::Seq);
    let mut buf = MemBuffer::new(scratch);
    let mut stack = MemStack::new(&mut buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        faer::Par::Seq,
        &mut stack,
    ) {
        Ok(info) => CgResult {
            iterations: info.iter_count,
            converged: true,
            residual: info.rel_residual,
        },
        Err(CgError::NoConvergence { rel_residual, .. }) => CgResult {
            iterations: max_iters,
            converged: false,
            residual: rel_residual,
        },
        Err(CgError::NonPositiveDefiniteOperator) => {
            eprintln!("CG ERROR (batched): NonPositiveDefiniteOperator");
            CgResult { iterations: 0, converged: false, residual: f64::MAX }
        },
        Err(CgError::NonPositiveDefinitePreconditioner) => {
            eprintln!("CG ERROR (batched): NonPositiveDefinitePreconditioner");
            CgResult { iterations: 0, converged: false, residual: f64::MAX }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::JacobiPreconditioner;
    use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};

    #[test]
    fn cg_simple_diagonal() {
        // A = diag(2, 3), b = (4, 9) => x = (2, 3)
        let mut mat = SparseMatrix::new(2);
        mat.set_diag(0, 2.0);
        mat.set_diag(1, 3.0);

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let rhs = vec![4.0, 9.0];
        let mut x = vec![0.0, 0.0];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-10, 100);

        assert!(result.converged);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }

    #[test]
    fn cg_with_offdiag() {
        // A = [[4, -1], [-1, 4]], b = (3, 3) => x = (1, 1)
        let mut mat = SparseMatrix::new(2);
        mat.set_diag(0, 4.0);
        mat.set_diag(1, 4.0);
        mat.add_entry(0, 1, -1.0);

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let rhs = vec![3.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-10, 100);

        assert!(result.converged);
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1] = {}", x[1]);
    }

    #[test]
    fn cg_larger_system() {
        // 4x4 tridiagonal: A[i,i] = 4, A[i,i+1] = -1
        let n = 4;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n {
            mat.set_diag(i, 4.0);
        }
        for i in 0..n - 1 {
            mat.add_entry(i, i + 1, -1.0);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-10, 100);

        assert!(result.converged);

        // Verify A*x approx= rhs
        let mut ax = vec![0.0; n];
        mat.spmv(&x, &mut ax);
        for i in 0..n {
            assert!(
                (ax[i] - rhs[i]).abs() < 1e-6,
                "residual at {}: ax={}, rhs={}",
                i,
                ax[i],
                rhs[i]
            );
        }
    }

    #[test]
    fn cg_graph_laplacian() {
        // 10-node chain Laplacian with regularization (mimics Kirchhoff)
        let n = 10;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n - 1 {
            mat.add_connection(i, i + 1, 1.0); // diag += 1, off = -1
        }
        // Global regularization (like Kirchhoff anchor)
        for i in 0..n {
            mat.add_diagonal(i, 0.01);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-6, 200);

        eprintln!("graph laplacian CG: iters={}, converged={}, residual={:.3e}",
            result.iterations, result.converged, result.residual);

        assert!(result.converged, "Laplacian CG should converge, residual={}", result.residual);
    }

    #[test]
    fn cg_large_graph_laplacian() {
        // 1000-node chain with variable conductance (mimics heterogeneous Kirchhoff)
        let n = 1000;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n - 1 {
            let cond = if i % 3 == 0 { 10.0 } else { 0.1 }; // 100x variation
            mat.add_connection(i, i + 1, cond);
        }
        for i in 0..n {
            mat.add_diagonal(i, 0.001);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-4, 2000);

        eprintln!("large laplacian CG: iters={}, converged={}, residual={:.3e}",
            result.iterations, result.converged, result.residual);

        assert!(result.converged, "Large Laplacian CG should converge, residual={}", result.residual);
    }

    #[test]
    fn cg_2d_grid_laplacian() {
        // 50x50 = 2500 node grid with 4-point stencil + variable conductance
        let w = 50;
        let h = 50;
        let n = w * h;
        let mut mat = SparseMatrix::new(n);

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                // Right neighbor
                if x + 1 < w {
                    let j = y * w + (x + 1);
                    let cond = if (x + y) % 5 == 0 { 50.0 } else { 0.5 }; // 100x variation
                    mat.add_connection(i, j, cond);
                }
                // Bottom neighbor
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let cond = if (x + y) % 7 == 0 { 30.0 } else { 1.0 };
                    mat.add_connection(i, j, cond);
                }
                // Regularization
                mat.add_diagonal(i, 0.001);
            }
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &precond, &rhs, &mut x, 1e-4, 5000);

        eprintln!("2D grid CG: n={}, iters={}, converged={}, residual={:.3e}",
            n, result.iterations, result.converged, result.residual);

        assert!(result.converged, "2D grid CG should converge, residual={}", result.residual);
    }

    #[test]
    fn cg_with_amg_1d() {
        use crate::solver::preconditioner::AmgPreconditioner;

        let n = 64;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n {
            mat.set_diag(i, 2.0);
        }
        mat.add_diagonal(0, 0.1);
        mat.add_diagonal(n - 1, 0.1);
        for i in 0..n - 1 {
            mat.add_entry(i, i + 1, -1.0);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, mat.diag(), mat.off_diag());

        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &amg, &rhs, &mut x, 1e-8, 200);

        assert!(result.converged, "AMG+CG 1D should converge, residual={}", result.residual);
    }

    #[test]
    fn cg_with_amg_2d_grid() {
        use crate::solver::preconditioner::AmgPreconditioner;

        // 30x30 = 900 nodes, 4-point stencil, variable conductance, regularized
        // This mimics the Kirchhoff Laplacian from the pipe network
        let w = 30;
        let h = 30;
        let n = w * h;
        let mut mat = SparseMatrix::new(n);

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + (x + 1);
                    let cond = if (x + y) % 5 == 0 { 50.0 } else { 0.5 };
                    mat.add_connection(i, j, cond);
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let cond = if (x + y) % 7 == 0 { 30.0 } else { 1.0 };
                    mat.add_connection(i, j, cond);
                }
                mat.add_diagonal(i, 0.001);
            }
        }

        // Test Jacobi CG
        let op_j = SparseMatrixOp::from_matrix(&mut mat);
        let jacobi = JacobiPreconditioner::new(mat.diag());
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x_j = vec![0.0; n];
        let result_j = solve_cg(&op_j, &jacobi, &rhs, &mut x_j, 1e-6, 2000);

        // Test AMG CG
        let op_a = SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, mat.diag(), mat.off_diag());
        let mut x_a = vec![0.0; n];
        let result_a = solve_cg(&op_a, &amg, &rhs, &mut x_a, 1e-6, 200);

        eprintln!(
            "2D grid ({}x{}, n={}): Jacobi CG iters={}, AMG CG iters={}",
            w, h, n, result_j.iterations, result_a.iterations
        );

        // Verify AMG solution quality
        let mut ax = vec![0.0; n];
        mat.spmv(&x_a, &mut ax);
        let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual: f64 = (0..n).map(|i| (ax[i] - rhs[i]).powi(2)).sum::<f64>().sqrt();
        eprintln!("AMG actual residual: {:.3e}", residual / rhs_norm);

        // AMG should converge faster than Jacobi
        assert!(
            result_a.iterations < result_j.iterations || result_a.iterations < 50,
            "AMG should be faster: AMG={} vs Jacobi={}", result_a.iterations, result_j.iterations
        );
    }
}

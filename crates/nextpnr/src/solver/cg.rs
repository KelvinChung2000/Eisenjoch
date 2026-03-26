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

    let iter_count = std::cell::Cell::new(0usize);
    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |x_current| {
            let i = iter_count.get();
            if i < 5 || i % 100 == 0 {
                let x_norm: f64 = (0..n).map(|j| x_current[(j, 0)] * x_current[(j, 0)]).sum::<f64>().sqrt();
                eprintln!("  CG iter {}: ||x|| = {:.6e}", i, x_norm);
            }
            iter_count.set(i + 1);
        },
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
    fn cg_with_amg_preconditioner() {
        use crate::solver::preconditioner::AmgPreconditioner;

        let n = 64;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n {
            mat.set_diag(i, 2.0);
        }
        // Add small anchor to ensure SPD
        mat.add_diagonal(0, 0.1);
        mat.add_diagonal(n - 1, 0.1);
        for i in 0..n - 1 {
            mat.add_entry(i, i + 1, -1.0);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(
            n,
            mat.diag(),
            mat.off_diag(),
        );

        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];
        let result = solve_cg(&op, &amg, &rhs, &mut x, 1e-8, 200);

        assert!(result.converged, "AMG+CG should converge, residual={}", result.residual);

        // Verify solution
        let mut ax = vec![0.0; n];
        mat.spmv(&x, &mut ax);
        let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual: f64 = (0..n)
            .map(|i| (ax[i] - rhs[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual / rhs_norm < 1e-6,
            "Relative residual {} too large",
            residual / rhs_norm
        );
    }
}

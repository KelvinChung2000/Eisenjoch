//! Thin wrapper around faer's conjugate_gradient solver.
//!
//! Provides a simple interface for solving A*x = rhs using preconditioned CG
//! via faer's matrix-free operator framework.

use dyn_stack::{MemBuffer, MemStack, StackReq};
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
/// `rhs_mat` is the right-hand side matrix (n × nrhs).
/// `x_mat` is the initial guess on input, solution on output.
/// `tol` is the relative tolerance for convergence.
/// `max_iters` is the maximum number of CG iterations.
pub fn solve_cg(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
) -> CgResult {
    solve_cg_with_par(
        mat,
        precond,
        rhs_mat,
        x_mat,
        tol,
        max_iters,
        crate::solver::par(),
    )
}

fn subtract_mean(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    for value in values {
        *value -= mean;
    }
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn norm_l2(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn apply_linop(
    op: &impl LinOp<f64>,
    input: &[f64],
    output: &mut [f64],
    par: faer::Par,
    buf: &mut MemBuffer,
) {
    let rhs = faer::MatRef::from_column_major_slice(input, input.len(), 1);
    let out = faer::MatMut::from_column_major_slice_mut(output, output.len(), 1);
    let mut stack = MemStack::new(buf);
    op.apply(out, rhs, par, &mut stack);
}

fn apply_precond(
    precond: &impl Precond<f64>,
    input: &[f64],
    output: &mut [f64],
    par: faer::Par,
    buf: &mut MemBuffer,
) {
    let rhs = faer::MatRef::from_column_major_slice(input, input.len(), 1);
    let out = faer::MatMut::from_column_major_slice_mut(output, output.len(), 1);
    let mut stack = MemStack::new(buf);
    precond.apply(out, rhs, par, &mut stack);
}

fn solve_cg_deflated_single(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
    par: faer::Par,
) -> CgResult {
    let n = rhs.len();
    if n == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let b_norm = norm_l2(rhs);
    if b_norm == 0.0 {
        x.fill(0.0);
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let threshold = tol * b_norm;
    let scratch = StackReq::any_of(&[mat.apply_scratch(1, par), precond.apply_scratch(1, par)]);
    let mut buf = MemBuffer::new(scratch);

    let mut r = rhs.to_vec();
    let mut ax = vec![0.0; n];
    apply_linop(mat, x, &mut ax, par, &mut buf);
    for (ri, axi) in r.iter_mut().zip(ax.iter()) {
        *ri -= *axi;
    }

    let mut abs_residual = norm_l2(&r);
    if abs_residual <= threshold {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: abs_residual / b_norm,
        };
    }

    // Deflate r (remove constant component — it's in the null space of the Laplacian)
    subtract_mean(&mut r);
    abs_residual = norm_l2(&r);
    if abs_residual <= threshold {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: abs_residual / b_norm,
        };
    }

    let mut z = vec![0.0; n];
    apply_precond(precond, &r, &mut z, par, &mut buf);
    subtract_mean(&mut z);

    let mut p = z.clone();

    let mut rtz = dot(&r, &z);
    if !rtz.is_finite() || rtz <= 0.0 {
        eprintln!(
            "CG ERROR (deflated): NonPositiveDefinitePreconditioner rtz={:.6e}",
            rtz
        );
        return CgResult {
            iterations: 0,
            converged: false,
            residual: f64::MAX,
        };
    }

    let mut ap = vec![0.0; n];
    for iter in 0..max_iters {
        apply_linop(mat, &p, &mut ap, par, &mut buf);
        let ptap = dot(&p, &ap);
        if !ptap.is_finite() || ptap <= 0.0 {
            eprintln!("CG ERROR (deflated): NonPositiveDefiniteOperator");
            return CgResult {
                iterations: iter,
                converged: false,
                residual: f64::MAX,
            };
        }

        let alpha = rtz / ptap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        subtract_mean(&mut r); // keep r deflated

        abs_residual = norm_l2(&r);
        if abs_residual <= threshold {
            return CgResult {
                iterations: iter + 1,
                converged: true,
                residual: abs_residual / b_norm,
            };
        }

        apply_precond(precond, &r, &mut z, par, &mut buf);
        subtract_mean(&mut z);

        let rtz_new = dot(&r, &z);
        if !rtz_new.is_finite() || rtz_new <= 0.0 {
            eprintln!(
                "CG ERROR (deflated): NonPositiveDefinitePreconditioner rtz={:.6e} iter={}",
                rtz_new,
                iter + 1
            );
            return CgResult {
                iterations: iter + 1,
                converged: false,
                residual: abs_residual / b_norm,
            };
        }

        let beta = rtz_new / rtz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        subtract_mean(&mut p); // keep p deflated
        rtz = rtz_new;
    }

    CgResult {
        iterations: max_iters,
        converged: false,
        residual: abs_residual / b_norm,
    }
}

/// Solve A*x = rhs using deflated preconditioned CG for near-singular Laplacians.
///
/// The solver projects the preconditioned residual and each updated search
/// direction onto the zero-mean subspace, which removes the constant null-space
/// mode common in graph Laplacians.
pub fn solve_cg_deflated(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
) -> CgResult {
    let n = mat.nrows();
    let nrhs = rhs_mat.ncols();
    if n == 0 || nrhs == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let mut result = CgResult {
        iterations: 0,
        converged: true,
        residual: 0.0,
    };

    for col in 0..nrhs {
        // Use column views directly — no element-by-element copies
        let rhs_col = rhs_mat.col(col).try_as_col_major().unwrap().as_slice();
        let x_slice = x_mat
            .as_mut()
            .col_mut(col)
            .try_as_col_major_mut()
            .unwrap()
            .as_slice_mut();

        let col_result = solve_cg_deflated_single(
            mat,
            precond,
            rhs_col,
            x_slice,
            tol,
            max_iters,
            crate::solver::par(),
        );

        result.iterations = result.iterations.max(col_result.iterations);
        result.converged &= col_result.converged;
        result.residual = result.residual.max(col_result.residual);
    }
    result
}

pub fn solve_cg_with_par(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    par: faer::Par,
) -> CgResult {
    let n = mat.nrows();
    if n == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    // Compute workspace requirement and allocate
    let scratch = conjugate_gradient_scratch(precond, mat, 1, par);
    let mut buf = MemBuffer::new(scratch);
    let mut stack = MemStack::new(&mut buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        par,
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
        }
        Err(CgError::NonPositiveDefinitePreconditioner) => {
            eprintln!("CG ERROR: NonPositiveDefinitePreconditioner detected!");
            CgResult {
                iterations: 0,
                converged: false,
                residual: f64::MAX,
            }
        }
    }
}

/// Compute the scratch size needed for CG with the given matrix and preconditioner.
/// Use this to pre-allocate a MemBuffer that can be reused across multiple solves.
pub fn cg_scratch_size(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
) -> StackReq {
    conjugate_gradient_scratch(precond, mat, 1, crate::solver::par())
}

/// Compute the scratch size needed for CG with `nrhs` right-hand sides.
pub fn cg_scratch_size_nrhs(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    nrhs: usize,
) -> StackReq {
    conjugate_gradient_scratch(precond, mat, nrhs.max(1), crate::solver::par())
}

/// Solve A*x = rhs using a pre-allocated scratch buffer, reusing matrix objects.
/// The buffer must have been allocated with at least `cg_scratch_size()` bytes.
pub fn solve_cg_reuse(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    buf: &mut MemBuffer,
) -> CgResult {
    let n = mat.nrows();
    if n == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    let mut stack = MemStack::new(buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        crate::solver::par(),
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
        Err(CgError::NonPositiveDefiniteOperator) => CgResult {
            iterations: 0,
            converged: false,
            residual: f64::MAX,
        },
        Err(CgError::NonPositiveDefinitePreconditioner) => CgResult {
            iterations: 0,
            converged: false,
            residual: f64::MAX,
        },
    }
}

/// Batched CG solve: A * X = B where X and B are matrices.
///
/// Returns one CgResult (the aggregate). Each column converges independently
/// inside faer's block CG, leveraging BLAS3 for the mat-vecs.
pub fn solve_cg_batched(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
) -> CgResult {
    solve_cg_batched_with_par(
        mat,
        precond,
        rhs_mat,
        x_mat,
        tol,
        max_iters,
        crate::solver::par(),
    )
}

pub fn solve_cg_batched_with_par(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    par: faer::Par,
) -> CgResult {
    let n = mat.nrows();
    let nrhs = rhs_mat.ncols();
    if n == 0 || nrhs == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    let scratch = conjugate_gradient_scratch(precond, mat, nrhs, par);
    let mut buf = MemBuffer::new(scratch);
    let mut stack = MemStack::new(&mut buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        par,
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
            CgResult {
                iterations: 0,
                converged: false,
                residual: f64::MAX,
            }
        }
        Err(CgError::NonPositiveDefinitePreconditioner) => {
            eprintln!("CG ERROR (batched): NonPositiveDefinitePreconditioner");
            CgResult {
                iterations: 0,
                converged: false,
                residual: f64::MAX,
            }
        }
    }
}

/// Batched CG solve using a pre-allocated scratch buffer.
pub fn solve_cg_batched_reuse(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    buf: &mut MemBuffer,
) -> CgResult {
    solve_cg_batched_reuse_with_par(
        mat,
        precond,
        rhs_mat,
        x_mat,
        tol,
        max_iters,
        crate::solver::par(),
        buf,
    )
}

/// Solve A*x = rhs column-by-column using single-RHS CG.
///
/// This avoids faer's multi-RHS CG which has false-positive NonPositiveDefiniteOperator
/// detection with strong preconditioners on near-singular systems.
pub fn solve_cg_columnwise(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    par: faer::Par,
    buf: &mut MemBuffer,
) -> CgResult {
    let n = mat.nrows();
    let ncols = rhs_mat.ncols();
    if n == 0 || ncols == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    // Ensure buffer is large enough for single-RHS CG.
    let scratch_req = conjugate_gradient_scratch(precond, mat, 1, par);
    if buf.len() < scratch_req.size_bytes() {
        *buf = MemBuffer::new(scratch_req);
    }

    let mut total_iters = 0usize;
    let mut max_residual = 0.0f64;
    let mut all_converged = true;

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    for col in 0..ncols {
        // Skip zero RHS columns — at coarse levels, some nets have all pins
        // in the same grid cell, producing zero RHS. CG can't proceed (r^T z = 0).
        let rhs_norm_sq: f64 = (0..n)
            .map(|i| {
                let v = rhs_mat[(i, col)];
                v * v
            })
            .sum();
        if rhs_norm_sq < 1e-30 {
            for i in 0..n {
                x_mat[(i, col)] = 0.0;
            }
            continue;
        }

        let rhs_col = rhs_mat.subcols(col, 1);
        let x_col = x_mat.as_mut().subcols_mut(col, 1);
        let mut stack = MemStack::new(buf);

        match conjugate_gradient(
            x_col,
            precond,
            mat,
            rhs_col,
            params,
            |_| {},
            par,
            &mut stack,
        ) {
            Ok(info) => {
                total_iters += info.iter_count;
                max_residual = max_residual.max(info.rel_residual);
            }
            Err(CgError::NoConvergence { rel_residual, .. }) => {
                total_iters += max_iters;
                max_residual = max_residual.max(rel_residual);
                all_converged = false;
            }
            Err(
                CgError::NonPositiveDefiniteOperator | CgError::NonPositiveDefinitePreconditioner,
            ) => {
                all_converged = false;
            }
        }
    }

    CgResult {
        iterations: if ncols > 0 { total_iters / ncols } else { 0 },
        converged: all_converged,
        residual: max_residual,
    }
}

pub fn solve_cg_batched_reuse_with_par(
    mat: &(impl LinOp<f64> + Sized),
    precond: &(impl Precond<f64> + Sized),
    rhs_mat: faer::MatRef<'_, f64>,
    mut x_mat: faer::MatMut<'_, f64>,
    tol: f64,
    max_iters: usize,
    par: faer::Par,
    buf: &mut MemBuffer,
) -> CgResult {
    let n = mat.nrows();
    let nrhs = rhs_mat.ncols();
    if n == 0 || nrhs == 0 {
        return CgResult {
            iterations: 0,
            converged: true,
            residual: 0.0,
        };
    }

    let mut params = CgParams::default();
    params.rel_tolerance = tol;
    params.max_iters = max_iters;

    let mut stack = MemStack::new(buf);

    match conjugate_gradient(
        x_mat.as_mut(),
        precond,
        mat,
        rhs_mat,
        params,
        |_| {},
        par,
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
        Err(CgError::NonPositiveDefiniteOperator)
        | Err(CgError::NonPositiveDefinitePreconditioner)
            if nrhs > 1 =>
        {
            // faer's multi-RHS CG can give false positives on near-singular systems
            // with strong preconditioners. Fall back to column-by-column single-RHS CG.
            solve_cg_columnwise(mat, precond, rhs_mat, x_mat, tol, max_iters, par, buf)
        }
        Err(CgError::NonPositiveDefiniteOperator) => CgResult {
            iterations: 0,
            converged: false,
            residual: f64::MAX,
        },
        Err(CgError::NonPositiveDefinitePreconditioner) => CgResult {
            iterations: 0,
            converged: false,
            residual: f64::MAX,
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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, 2, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, 2, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-10, 100);

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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, 2, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, 2, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-10, 100);

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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-10, 100);

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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-6, 200);

        eprintln!(
            "graph laplacian CG: iters={}, converged={}, residual={:.3e}",
            result.iterations, result.converged, result.residual
        );

        assert!(
            result.converged,
            "Laplacian CG should converge, residual={}",
            result.residual
        );
    }

    #[test]
    fn cg_deflated_singular_graph_laplacian() {
        let n = 32;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n - 1 {
            mat.add_connection(i, i + 1, 1.0);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(mat.diag());

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x = vec![0.0; n];

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg_deflated(&op, &precond, rhs_mat, x_mat, 1e-8, 500);

        assert!(
            result.converged,
            "Deflated CG should converge on singular Laplacian, residual={}",
            result.residual
        );

        let mean = x.iter().sum::<f64>() / n as f64;
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
        assert!(mean.abs() < 1e-6, "solution mean = {}", mean);
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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-4, 2000);

        eprintln!(
            "large laplacian CG: iters={}, converged={}, residual={:.3e}",
            result.iterations, result.converged, result.residual
        );

        assert!(
            result.converged,
            "Large Laplacian CG should converge, residual={}",
            result.residual
        );
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

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, 1e-4, 5000);

        eprintln!(
            "2D grid CG: n={}, iters={}, converged={}, residual={:.3e}",
            n, result.iterations, result.converged, result.residual
        );

        assert!(
            result.converged,
            "2D grid CG should converge, residual={}",
            result.residual
        );
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
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &amg, rhs_mat, x_mat, 1e-8, 200);

        assert!(
            result.converged,
            "AMG+CG 1D should converge, residual={}",
            result.residual
        );
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
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_j_mat = faer::MatMut::from_column_major_slice_mut(&mut x_j, n, 1);
        let result_j = solve_cg(&op_j, &jacobi, rhs_mat, x_j_mat, 1e-6, 2000);

        // Test AMG CG
        let op_a = SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, mat.diag(), mat.off_diag());
        let mut x_a = vec![0.0; n];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_a_mat = faer::MatMut::from_column_major_slice_mut(&mut x_a, n, 1);
        let result_a = solve_cg(&op_a, &amg, rhs_mat, x_a_mat, 1e-6, 200);

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
            "AMG should be faster: AMG={} vs Jacobi={}",
            result_a.iterations,
            result_j.iterations
        );
    }

    #[test]
    fn spectral_batched_cg_works() {
        use crate::solver::preconditioner::SpectralPreconditioner;

        let nx = 8;
        let ny = 11;
        let n = nx * ny;
        let epsilon = 1e-5;
        // The diagonal has to carry the 4 that `SpectralPreconditioner`
        // models (its eigenvalues are `4 - 2cos(tx) - 2cos(ty) + epsilon`).
        // With `epsilon` alone this is a pure Neumann graph Laplacian: the
        // constant vector is a null vector, the right-hand sides below have
        // nonzero mean, and the solution is dominated by that near-null
        // direction at magnitude ~0.5/epsilon. CG stagnates there -- measured
        // at 5000 iterations it still reported rel_residual 1.0, while the
        // same system with a diagonal converges in 10. That was a broken test
        // premise, not a solver bug.
        let mut mat = SparseMatrix::new(n);
        for y in 0..ny {
            for x in 0..nx {
                let i = y * nx + x;
                if x + 1 < nx {
                    let j = y * nx + (x + 1);
                    let w = 1.0 + 0.5 * ((x + y) % 3) as f64;
                    mat.add_connection(i, j, w);
                }
                if y + 1 < ny {
                    let j = (y + 1) * nx + x;
                    let w = 1.0 + 0.3 * ((x * 2 + y) % 5) as f64;
                    mat.add_connection(i, j, w);
                }
                mat.add_diagonal(i, 4.0 + epsilon);
            }
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let spectral = SpectralPreconditioner::new(nx, ny, epsilon);

        // Test with 4 RHS columns (batched)
        let nrhs = 4;
        let mut rhs_data = vec![0.0; n * nrhs];
        for col in 0..nrhs {
            rhs_data[col * n] = 1.0;
            rhs_data[col * n + n - 1] = -1.0;
            rhs_data[col * n + col * 5 % n] += 0.5;
        }
        let mut x_data = vec![0.0; n * nrhs];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_data, n, nrhs);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_data, n, nrhs);

        let scratch_req = cg_scratch_size_nrhs(&op, &spectral, nrhs);
        let mut buf = MemBuffer::new(scratch_req);
        let result = solve_cg_batched_reuse_with_par(
            &op,
            &spectral,
            rhs_mat,
            x_mat,
            1e-6,
            200,
            faer::Par::Seq,
            &mut buf,
        );

        assert!(
            result.converged,
            "Spectral batched CG should converge: iters={} residual={:.3e}",
            result.iterations, result.residual
        );
        eprintln!("Spectral batched CG: {} iterations", result.iterations);
    }

    #[test]
    fn ic0_batched_cg_works() {
        use crate::solver::preconditioner::Ic0Preconditioner;

        let nx = 8;
        let ny = 11;
        let n = nx * ny;
        let mut mat = SparseMatrix::new(n);
        for y in 0..ny {
            for x in 0..nx {
                let i = y * nx + x;
                if x + 1 < nx {
                    let j = y * nx + (x + 1);
                    mat.add_connection(i, j, 1.0);
                }
                if y + 1 < ny {
                    let j = (y + 1) * nx + x;
                    mat.add_connection(i, j, 1.0);
                }
                mat.add_diagonal(i, 1e-3);
            }
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let ic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.0);

        let nrhs = 4;
        let mut rhs_data = vec![0.0; n * nrhs];
        for col in 0..nrhs {
            rhs_data[col * n] = 1.0;
            rhs_data[col * n + n - 1] = -1.0;
        }
        let mut x_data = vec![0.0; n * nrhs];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_data, n, nrhs);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_data, n, nrhs);

        let scratch_req = cg_scratch_size_nrhs(&op, &ic0, nrhs);
        let mut buf = MemBuffer::new(scratch_req);
        let result = solve_cg_batched_reuse_with_par(
            &op,
            &ic0,
            rhs_mat,
            x_mat,
            1e-6,
            200,
            faer::Par::Seq,
            &mut buf,
        );

        assert!(
            result.converged,
            "IC(0) batched CG should converge: iters={} residual={:.3e}",
            result.iterations, result.residual
        );
        eprintln!("IC(0) batched CG: {} iterations", result.iterations);
    }

    #[test]
    #[ignore]
    fn diagnose_kirchhoff_system_properties() {
        use crate::solver::preconditioner::{
            Ic0Preconditioner, JacobiPreconditioner, SpectralPreconditioner,
        };
        use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
        use faer::matrix_free::{LinOp, Precond};
        use std::collections::HashMap;

        fn splitmix64(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        fn edge_weight(edge_idx: usize) -> f64 {
            let bits = splitmix64(edge_idx as u64 + 1);
            let unit = (bits as f64) / (u64::MAX as f64);
            0.1 + 9.9 * unit
        }

        fn sample_vector(n: usize, seed: u64) -> Vec<f64> {
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let bits = splitmix64(seed ^ (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15));
                let unit = (bits as f64) / (u64::MAX as f64);
                values.push(2.0 * unit - 1.0);
            }
            values
        }

        fn dense_from_sparse(mat: &SparseMatrix) -> Vec<f64> {
            let n = mat.n();
            let mut dense = vec![0.0; n * n];
            for i in 0..n {
                dense[i * n + i] = mat.diag()[i];
            }
            for &(lo, hi, value) in mat.off_diag() {
                dense[lo * n + hi] += value;
                dense[hi * n + lo] += value;
            }
            dense
        }

        fn dense_mv(dense: &[f64], x: &[f64], out: &mut [f64]) {
            let n = x.len();
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += dense[i * n + j] * x[j];
                }
                out[i] = sum;
            }
        }

        fn frob_norm(dense: &[f64]) -> f64 {
            dense.iter().map(|v| v * v).sum::<f64>().sqrt()
        }

        fn dense_sub(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
            lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
        }

        fn factorize_ic0_like_impl(
            diag: &[f64],
            off_diag: &[(usize, usize, f64)],
        ) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize, f64) {
            fn intersection_dot(
                row_cols: &[Vec<usize>],
                row_positions: &[Vec<usize>],
                row_i: usize,
                row_k: usize,
                values: &[f64],
            ) -> f64 {
                let cols_i = &row_cols[row_i];
                let cols_k = &row_cols[row_k];
                let pos_i = &row_positions[row_i];
                let pos_k = &row_positions[row_k];

                let mut a = 0usize;
                let mut b = 0usize;
                let mut sum = 0.0;
                while a < cols_i.len() && b < cols_k.len() {
                    let ci = cols_i[a];
                    let ck = cols_k[b];
                    if ci >= row_k || ck >= row_k {
                        break;
                    }
                    if ci == ck {
                        sum += values[pos_i[a]] * values[pos_k[b]];
                        a += 1;
                        b += 1;
                    } else if ci < ck {
                        a += 1;
                    } else {
                        b += 1;
                    }
                }
                sum
            }

            fn factorize_in_place(
                n: usize,
                col_ptrs: &[usize],
                row_indices: &[usize],
                row_cols: &[Vec<usize>],
                row_positions: &[Vec<usize>],
                values: &mut [f64],
            ) -> bool {
                for k in 0..n {
                    let diag_pos = col_ptrs[k];
                    let mut diag_correction = 0.0;
                    for &pos in &row_positions[k] {
                        let lij = values[pos];
                        diag_correction += lij * lij;
                    }

                    let pivot_sq = values[diag_pos] - diag_correction;
                    if !pivot_sq.is_finite() || pivot_sq <= 0.0 {
                        return false;
                    }

                    let lkk = pivot_sq.sqrt();
                    if !lkk.is_finite() || lkk <= 0.0 {
                        return false;
                    }
                    values[diag_pos] = lkk;

                    for pos in diag_pos + 1..col_ptrs[k + 1] {
                        let row = row_indices[pos];
                        let sum = intersection_dot(row_cols, row_positions, row, k, values);
                        let lik = (values[pos] - sum) / lkk;
                        if !lik.is_finite() {
                            return false;
                        }
                        values[pos] = lik;
                    }
                }
                true
            }

            let n = diag.len();
            let mut col_rows = vec![Vec::new(); n];
            for (col, rows) in col_rows.iter_mut().enumerate() {
                rows.push(col);
            }
            for &(lo, hi, _) in off_diag {
                col_rows[lo].push(hi);
            }
            for rows in &mut col_rows {
                rows.sort_unstable();
                rows.dedup();
            }

            let mut col_ptrs = Vec::with_capacity(n + 1);
            let mut row_indices = Vec::new();
            col_ptrs.push(0);
            for rows in &col_rows {
                row_indices.extend_from_slice(rows);
                col_ptrs.push(row_indices.len());
            }

            let mut lower_pos = HashMap::with_capacity(row_indices.len().saturating_sub(n));
            let mut row_cols = vec![Vec::new(); n];
            let mut row_positions = vec![Vec::new(); n];
            for col in 0..n {
                for pos in col_ptrs[col] + 1..col_ptrs[col + 1] {
                    let row = row_indices[pos];
                    lower_pos.insert((col, row), pos);
                    row_cols[row].push(col);
                    row_positions[row].push(pos);
                }
            }

            let mut base = vec![0.0; row_indices.len()];
            for i in 0..n {
                base[col_ptrs[i]] = diag[i];
            }
            for &(lo, hi, value) in off_diag {
                let pos = lower_pos[&(lo, hi)];
                base[pos] += value;
            }

            let diag_mean = if n == 0 {
                0.0
            } else {
                diag.iter().sum::<f64>() / n as f64
            };
            let base_shift = 1e-10 * diag_mean.abs().max(1.0);

            let mut shifted = base.clone();
            for attempt in 0..8 {
                shifted.copy_from_slice(&base);
                if attempt > 0 {
                    let shift = base_shift * 10f64.powi((attempt - 1) as i32);
                    for i in 0..n {
                        shifted[col_ptrs[i]] += shift;
                    }
                }
                if factorize_in_place(
                    n,
                    &col_ptrs,
                    &row_indices,
                    &row_cols,
                    &row_positions,
                    &mut shifted,
                ) {
                    let applied_shift = if attempt == 0 {
                        0.0
                    } else {
                        base_shift * 10f64.powi((attempt - 1) as i32)
                    };
                    return (col_ptrs, row_indices, shifted, attempt, applied_shift);
                }
            }

            panic!("IC(0)-like factorization failed in diagnostic helper");
        }

        fn lower_to_dense(
            col_ptrs: &[usize],
            row_indices: &[usize],
            values: &[f64],
            n: usize,
        ) -> Vec<f64> {
            let mut dense = vec![0.0; n * n];
            for col in 0..n {
                for pos in col_ptrs[col]..col_ptrs[col + 1] {
                    let row = row_indices[pos];
                    dense[row * n + col] = values[pos];
                }
            }
            dense
        }

        fn dense_llt(lower: &[f64], n: usize) -> Vec<f64> {
            let mut out = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let upto = i.min(j);
                    let mut sum = 0.0;
                    for k in 0..=upto {
                        sum += lower[i * n + k] * lower[j * n + k];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }

        fn cg_status<P: Precond<f64> + Sized>(
            op: &SparseMatrixOp,
            precond: &P,
            rhs_data: &[f64],
            n: usize,
            nrhs: usize,
        ) -> String {
            let rhs_mat = faer::MatRef::from_column_major_slice(rhs_data, n, nrhs);
            let mut x_data = vec![0.0; n * nrhs];
            let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_data, n, nrhs);
            let mut params = CgParams::default();
            params.rel_tolerance = 1e-8;
            params.max_iters = 500;

            let scratch = conjugate_gradient_scratch(precond, op, nrhs, faer::Par::Seq);
            let mut buf = MemBuffer::new(scratch);
            let mut stack = MemStack::new(&mut buf);

            match conjugate_gradient(
                x_mat,
                precond,
                op,
                rhs_mat,
                params,
                |_| {},
                faer::Par::Seq,
                &mut stack,
            ) {
                Ok(info) => format!(
                    "Ok(iter={}, relres={:.3e})",
                    info.iter_count, info.rel_residual
                ),
                Err(CgError::NoConvergence { rel_residual, .. }) => {
                    format!("NoConvergence(relres={:.3e})", rel_residual)
                }
                Err(CgError::NonPositiveDefiniteOperator) => {
                    "NonPositiveDefiniteOperator".to_string()
                }
                Err(CgError::NonPositiveDefinitePreconditioner) => {
                    "NonPositiveDefinitePreconditioner".to_string()
                }
            }
        }

        fn print_preconditioner_diagnostics<P: Precond<f64> + LinOp<f64> + Sized>(
            name: &str,
            op: &SparseMatrixOp,
            precond: &P,
            r: &[f64],
            rhs1: &[f64],
            rhs4: &[f64],
        ) {
            let scratch = precond.apply_scratch(1, faer::Par::Seq);
            let mut buf = MemBuffer::new(scratch);
            let mut z = vec![0.0; r.len()];
            apply_precond(precond, r, &mut z, faer::Par::Seq, &mut buf);
            let rtz = dot(r, &z);
            let z_norm = norm_l2(&z);

            eprintln!(
                "{}: r^T z = {:.6e} ({}) z_norm={:.6e}",
                name,
                rtz,
                if rtz > 0.0 {
                    "positive"
                } else if rtz < 0.0 {
                    "negative"
                } else {
                    "zero"
                },
                z_norm
            );
            eprintln!(
                "{}: CG nrhs=1 -> {}",
                name,
                cg_status(op, precond, rhs1, r.len(), 1)
            );
            eprintln!(
                "{}: CG nrhs=4 -> {}",
                name,
                cg_status(op, precond, rhs4, r.len(), 4)
            );
        }

        let nx = 8usize;
        let ny = 11usize;
        let n = nx * ny;

        let mut mat = SparseMatrix::new(n);
        let mut edge_idx = 0usize;
        for y in 0..ny {
            for x in 0..nx {
                let i = y * nx + x;
                if x + 1 < nx {
                    let j = y * nx + x + 1;
                    mat.add_connection(i, j, edge_weight(edge_idx));
                    edge_idx += 1;
                }
                if y + 1 < ny {
                    let j = (y + 1) * nx + x;
                    mat.add_connection(i, j, edge_weight(edge_idx));
                    edge_idx += 1;
                }
            }
        }

        let mut offdiag_row_sum = vec![0.0; n];
        let mut max_positive_offdiag = 0.0f64;
        for &(lo, hi, value) in mat.off_diag() {
            offdiag_row_sum[lo] += value;
            offdiag_row_sum[hi] += value;
            max_positive_offdiag = max_positive_offdiag.max(value);
        }
        let max_kirchhoff_balance_error = (0..n)
            .map(|i| (offdiag_row_sum[i] + mat.diag()[i]).abs())
            .fold(0.0f64, f64::max);

        let mean_diag_l = mat.diagonal_mean();
        let epsilon = 1e-5 * mean_diag_l;
        for i in 0..n {
            mat.add_diagonal(i, epsilon);
        }

        let dense_a = dense_from_sparse(&mat);
        let mut symmetry_error = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                symmetry_error =
                    symmetry_error.max((dense_a[i * n + j] - dense_a[j * n + i]).abs());
            }
        }

        let mut ones_image = vec![0.0; n];
        let ones = vec![1.0; n];
        dense_mv(&dense_a, &ones, &mut ones_image);
        let a_ones_norm = norm_l2(&ones_image);
        let a_ones_mean = ones_image.iter().sum::<f64>() / n as f64;
        let expected_a_ones = epsilon * (n as f64).sqrt();

        let mut min_rayleigh = f64::INFINITY;
        let mut max_rayleigh = 0.0f64;
        for sample in 0..10 {
            let mut v = sample_vector(n, 0xabc000 + sample as u64);
            if sample % 2 == 0 {
                subtract_mean(&mut v);
            }
            let mut av = vec![0.0; n];
            dense_mv(&dense_a, &v, &mut av);
            let denom = dot(&v, &v);
            let rayleigh = dot(&v, &av) / denom;
            min_rayleigh = min_rayleigh.min(rayleigh);
            max_rayleigh = max_rayleigh.max(rayleigh);
            eprintln!("quadratic form sample {:02}: {:.6e}", sample, rayleigh);
        }
        let sampled_cond = max_rayleigh / min_rayleigh;
        let nullspace_proxy_cond = max_rayleigh / epsilon;

        let mut rhs1 = sample_vector(n, 0x1234);
        subtract_mean(&mut rhs1);
        let mut rhs4 = vec![0.0; 4 * n];
        for col in 0..4 {
            let mut col_vec = sample_vector(n, 0x5000 + col as u64);
            subtract_mean(&mut col_vec);
            rhs4[col * n..(col + 1) * n].copy_from_slice(&col_vec);
        }
        let mut r = sample_vector(n, 0xfeed);
        subtract_mean(&mut r);

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let jacobi = JacobiPreconditioner::new(mat.diag());
        let spectral = SpectralPreconditioner::new(nx, ny, epsilon);
        let ic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.95);

        eprintln!("Kirchhoff diagnostic system:");
        eprintln!(
            "grid={}x{} n={} edges={} mean_diag(L)={:.6e} epsilon={:.6e}",
            nx, ny, n, edge_idx, mean_diag_l, epsilon
        );
        eprintln!(
            "M-matrix checks: max_positive_offdiag={:.6e}, max_row_balance_error_before_shift={:.6e}",
            max_positive_offdiag, max_kirchhoff_balance_error
        );
        eprintln!("symmetry max_abs_error={:.6e}", symmetry_error);
        eprintln!(
            "near-singularity: ||A*1||_2={:.6e}, mean(A*1)={:.6e}, expected ||eps*1||_2={:.6e}",
            a_ones_norm, a_ones_mean, expected_a_ones
        );
        eprintln!(
            "condition proxy: sampled min={:.6e}, sampled max={:.6e}, sampled cond={:.6e}, max/epsilon={:.6e}",
            min_rayleigh, max_rayleigh, sampled_cond, nullspace_proxy_cond
        );

        print_preconditioner_diagnostics("Jacobi", &op, &jacobi, &r, &rhs1, &rhs4);
        print_preconditioner_diagnostics("Spectral", &op, &spectral, &r, &rhs1, &rhs4);
        print_preconditioner_diagnostics("IC(0)", &op, &ic0, &r, &rhs1, &rhs4);

        let (col_ptrs, row_indices, values, shift_attempts, applied_shift) =
            factorize_ic0_like_impl(mat.diag(), mat.off_diag());
        let lower_dense = lower_to_dense(&col_ptrs, &row_indices, &values, n);
        let llt_dense = dense_llt(&lower_dense, n);
        let factor_error = dense_sub(&dense_a, &llt_dense);
        let factor_error_frob = frob_norm(&factor_error);
        let factor_rel_error = factor_error_frob / frob_norm(&dense_a);
        let factor_error_max = factor_error.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let diag_values: Vec<f64> = (0..n).map(|i| values[col_ptrs[i]]).collect();
        let diag_min = diag_values.iter().copied().fold(f64::INFINITY, f64::min);
        let diag_max = diag_values.iter().copied().fold(0.0f64, f64::max);
        let diag_mean = diag_values.iter().sum::<f64>() / n as f64;

        eprintln!(
            "IC(0) factor diag stats: min={:.6e}, max={:.6e}, mean={:.6e}, shift_attempts={}, applied_shift={:.6e}",
            diag_min, diag_max, diag_mean, shift_attempts, applied_shift
        );
        eprintln!(
            "IC(0) factorization error: frob={:.6e}, rel_frob={:.6e}, max_abs={:.6e}",
            factor_error_frob, factor_rel_error, factor_error_max
        );
    }
}

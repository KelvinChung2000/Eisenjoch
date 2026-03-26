//! Jacobi-preconditioned Conjugate Gradient solver.
//!
//! Direct implementation using SparseMatrix::spmv(). No external dependencies.
//! This works correctly for near-singular graph Laplacians where faer's
//! block CG has convergence issues.

use super::sparse_matrix::SparseMatrix;

/// Result of a PCG solve.
pub struct PcgResult {
    pub iterations: usize,
    pub converged: bool,
    pub rel_residual: f64,
}

/// Solve A*x = rhs using Jacobi-preconditioned CG.
///
/// `mat` is the symmetric positive-definite matrix.
/// `rhs` is the right-hand side vector.
/// `x` is the initial guess on input, solution on output.
/// `tol` is the relative tolerance: stop when ||r|| / ||b|| < tol.
/// `max_iters` is the maximum number of CG iterations.
pub fn pcg_solve(
    mat: &mut SparseMatrix,
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> PcgResult {
    let n = mat.n();
    if n == 0 {
        return PcgResult { iterations: 0, converged: true, rel_residual: 0.0 };
    }

    // Jacobi preconditioner: M^{-1} = diag(1/A[i,i])
    let inv_diag: Vec<f64> = mat.diag().iter()
        .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
        .collect();

    // r = b - A*x
    let mut r = vec![0.0; n];
    let mut ap = vec![0.0; n];
    mat.spmv(x, &mut ap);
    for i in 0..n {
        r[i] = rhs[i] - ap[i];
    }

    let b_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if b_norm < 1e-30 {
        return PcgResult { iterations: 0, converged: true, rel_residual: 0.0 };
    }

    // z = M^{-1} r
    let mut z: Vec<f64> = r.iter().zip(&inv_diag).map(|(&ri, &id)| ri * id).collect();
    // p = z
    let mut p = z.clone();
    // rz = r^T z
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

    for k in 0..max_iters {
        // ap = A * p
        mat.spmv(&p, &mut ap);

        // alpha = rz / (p^T A p)
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-30 {
            return PcgResult { iterations: k, converged: false, rel_residual: f64::MAX };
        }
        let alpha = rz / pap;

        // x += alpha * p, r -= alpha * Ap
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        // Check convergence
        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm / b_norm < tol {
            return PcgResult { iterations: k + 1, converged: true, rel_residual: r_norm / b_norm };
        }

        // z = M^{-1} r
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }

        // beta = (r^T z_new) / (r^T z_old)
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz.max(1e-30);

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    PcgResult { iterations: max_iters, converged: false, rel_residual: r_norm / b_norm }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_diagonal() {
        let mut mat = SparseMatrix::new(2);
        mat.set_diag(0, 2.0);
        mat.set_diag(1, 3.0);
        let rhs = vec![4.0, 9.0];
        let mut x = vec![0.0, 0.0];
        let result = pcg_solve(&mut mat, &rhs, &mut x, 1e-10, 100);
        assert!(result.converged);
        assert!((x[0] - 2.0).abs() < 1e-8);
        assert!((x[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn pcg_laplacian_2x2() {
        // Graph Laplacian + regularization
        let mut mat = SparseMatrix::new(2);
        mat.add_connection(0, 1, 1.0); // diag += 1, off = -1
        mat.add_diagonal(0, 0.01); // regularize
        mat.add_diagonal(1, 0.01);
        let rhs = vec![1.0, -1.0];
        let mut x = vec![0.0, 0.0];
        let result = pcg_solve(&mut mat, &rhs, &mut x, 1e-8, 100);
        assert!(result.converged, "rel_residual={}", result.rel_residual);
    }

    #[test]
    fn pcg_large_laplacian() {
        // 1D chain Laplacian with regularization
        let n = 1000;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n - 1 {
            mat.add_connection(i, i + 1, 1.0);
        }
        // Global regularization
        for i in 0..n {
            mat.add_diagonal(i, 0.001);
        }
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
        let mut x = vec![0.0; n];
        let result = pcg_solve(&mut mat, &rhs, &mut x, 1e-6, 2000);
        assert!(result.converged, "iters={}, rel_residual={}", result.iterations, result.rel_residual);
    }
}

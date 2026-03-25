//! CPU backend using faer for sparse Cholesky and CG.
//!
//! `FaerDirectSolver` wraps faer's sparse LLT factorization with AMD ordering.
//! `faer_cg` provides Jacobi-preconditioned Conjugate Gradient using manual spmv.

use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;

use super::backend::LinearSolver;

/// Direct sparse solver for symmetric positive-definite systems using faer's
/// sparse Cholesky factorization with AMD fill-reducing ordering.
///
/// Symbolic factorization is performed once; numeric factorization and solve
/// are performed for each new set of values.
pub struct FaerDirectSolver {
    /// Matrix dimension.
    n: usize,
    /// Number of off-diagonal entries expected per solve.
    expected_offdiag_len: usize,
    /// Cached symbolic factorization (includes AMD ordering).
    symbolic: SymbolicLlt<usize>,
}

impl FaerDirectSolver {
    /// Create a new direct solver for a system of size `n` with the given
    /// sparsity pattern (pipe endpoints as (from, to) pairs).
    ///
    /// Performs symbolic factorization (AMD ordering + elimination tree).
    /// This is the expensive one-time cost.
    pub fn new(
        n: usize,
        pipe_endpoints: &[(usize, usize)],
        _grid_width: usize,
        _grid_height: usize,
    ) -> Self {
        // Build a dummy CSC matrix to get the symbolic structure.
        let triplets = build_triplets_from_pipes(n, pipe_endpoints);
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
            .expect("failed to build sparse matrix from pipe endpoints");

        let symbolic = SymbolicLlt::try_new(mat.symbolic(), Side::Upper)
            .expect("symbolic Cholesky factorization failed");

        Self {
            n,
            expected_offdiag_len: pipe_endpoints.len(),
            symbolic,
        }
    }

    /// Solve A*x = rhs where A is defined by diagonal and off-diagonal entries.
    ///
    /// This performs:
    /// 1. Build CSC from diag + off_diag
    /// 2. Numeric LLT factorization using cached symbolic structure
    /// 3. Solve using forward/backward substitution
    pub fn solve(
        &mut self,
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
        rhs: &[f64],
        x: &mut [f64],
    ) {
        debug_assert_eq!(diag.len(), self.n);
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(
            off_diag.len(),
            self.expected_offdiag_len,
            "sparsity pattern changed: expected {} off-diag entries, got {}",
            self.expected_offdiag_len,
            off_diag.len(),
        );

        // Build upper-triangle CSC matrix from diag + off_diag.
        let triplets = build_triplets_from_system(self.n, diag, off_diag);
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(self.n, self.n, &triplets)
            .expect("failed to build sparse matrix");

        // Numeric factorization using cached symbolic structure.
        let llt = Llt::try_new_with_symbolic(self.symbolic.clone(), mat.as_ref(), Side::Upper)
            .expect("numeric Cholesky factorization failed");

        // Solve: copy rhs into a dense column vector, solve in place.
        x.copy_from_slice(rhs);
        let mut x_mat = faer::MatMut::from_column_major_slice_mut(x, self.n, 1);
        llt.solve_in_place_with_conj(faer::Conj::No, x_mat.as_mut());
    }
}

impl LinearSolver for FaerDirectSolver {
    fn solve(
        &mut self,
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
        rhs: &[f64],
        x: &mut [f64],
    ) {
        FaerDirectSolver::solve(self, diag, off_diag, rhs, x);
    }
}

/// Symmetric sparse matrix-vector product: result = A * x.
///
/// A is represented by its diagonal and a list of upper-triangle off-diagonal
/// entries (i, j, weight) where i < j. The matrix is symmetric, so each
/// off-diagonal entry contributes to both (i,j) and (j,i).
#[inline]
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

/// Dot product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Jacobi-preconditioned Conjugate Gradient solver for A*x = b.
///
/// Uses M^{-1} = diag(1/A[i,i]) as preconditioner. Uses relative residual
/// norm convergence criterion (||r|| / ||b|| < tol).
///
/// Returns the number of iterations performed.
pub fn faer_cg(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }

    // Jacobi preconditioner: inv_diag[i] = 1 / diag[i]
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
        .collect();

    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];

    // Initial residual: r = b - A * x
    spmv(diag, off_diag, x, &mut ap);
    for i in 0..n {
        r[i] = rhs[i] - ap[i];
        z[i] = r[i] * inv_diag[i];
        p[i] = z[i];
    }

    let mut rz_old = dot(&r, &z);
    let rhs_norm = dot(rhs, rhs).sqrt().max(1e-12);

    for iter in 0..max_iters {
        spmv(diag, off_diag, &p, &mut ap);

        let p_ap = dot(&p, &ap);
        let alpha = rz_old / p_ap.max(1e-16);

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        if dot(&r, &r).sqrt() / rhs_norm < tol {
            return iter + 1;
        }

        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    max_iters
}

/// Build faer Triplets for an upper-triangle symmetric matrix from pipe endpoints.
/// Used for symbolic factorization (dummy values).
fn build_triplets_from_pipes(
    n: usize,
    pipe_endpoints: &[(usize, usize)],
) -> Vec<Triplet<usize, usize, f64>> {
    let mut triplets = Vec::with_capacity(n + pipe_endpoints.len());

    // Diagonal entries.
    for i in 0..n {
        triplets.push(Triplet::new(i, i, 1.0));
    }

    // Upper-triangle off-diagonal entries.
    for &(from, to) in pipe_endpoints {
        let (lo, hi) = if from < to { (from, to) } else { (to, from) };
        triplets.push(Triplet::new(lo, hi, -1.0));
    }

    triplets
}

/// Build faer Triplets for a full upper-triangle symmetric matrix from
/// diagonal values and off-diagonal entries.
fn build_triplets_from_system(
    n: usize,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
) -> Vec<Triplet<usize, usize, f64>> {
    let mut triplets = Vec::with_capacity(n + off_diag.len());

    // Diagonal entries.
    for i in 0..n {
        triplets.push(Triplet::new(i, i, diag[i]));
    }

    // Upper-triangle off-diagonal entries (lo < hi).
    for &(lo, hi, val) in off_diag {
        debug_assert!(lo < hi, "off_diag entries must have lo < hi");
        triplets.push(Triplet::new(lo, hi, val));
    }

    triplets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_simple_diagonal() {
        // A = diag(2, 3), b = (4, 9) => x = (2, 3)
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
    fn cg_with_offdiag() {
        // A = [[4, -1], [-1, 4]], b = (3, 3) => x = (1, 1)
        let diag = vec![4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0)];
        let rhs = vec![3.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!(iters <= 10);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn direct_solver_basic() {
        // A = [[4, -1], [-1, 4]], b = (3, 3) => x = (1, 1)
        let mut solver = FaerDirectSolver::new(2, &[(0, 1)], 1, 1);
        let diag = vec![4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0)];
        let rhs = vec![3.0, 3.0];
        let mut x = vec![0.0, 0.0];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        assert!((x[0] - 1.0).abs() < 1e-8, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-8, "x[1] = {}", x[1]);
    }

    #[test]
    fn direct_solver_larger() {
        // 4x4 tridiagonal: A[i,i] = 4, A[i,i+1] = -1
        let n = 4;
        let pipes: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let mut solver = FaerDirectSolver::new(n, &pipes, 2, 2);

        let diag = vec![4.0; n];
        let off_diag: Vec<(usize, usize, f64)> =
            (0..n - 1).map(|i| (i, i + 1, -1.0)).collect();
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        solver.solve(&diag, &off_diag, &rhs, &mut x);

        // Verify A*x = rhs
        let mut ax = vec![0.0; n];
        for i in 0..n {
            ax[i] = diag[i] * x[i];
        }
        for &(lo, hi, w) in &off_diag {
            ax[lo] += w * x[hi];
            ax[hi] += w * x[lo];
        }
        for i in 0..n {
            assert!(
                (ax[i] - rhs[i]).abs() < 1e-8,
                "residual at {}: ax={}, rhs={}",
                i,
                ax[i],
                rhs[i]
            );
        }
    }
}

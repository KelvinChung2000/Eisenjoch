//! Direct sparse Cholesky solver using faer's LLT factorization.
//!
//! `FaerDirectSolver` wraps faer's sparse LLT factorization with AMD ordering.
//! Symbolic factorization is performed once; numeric factorization and solve
//! are performed for each new set of values.

use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;

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
    pub fn new(n: usize, pipe_endpoints: &[(usize, usize)]) -> Self {
        // Build a dummy CSC matrix to get the symbolic structure.
        let dummy_off_diag: Vec<(usize, usize, f64)> = pipe_endpoints
            .iter()
            .map(|&(from, to)| {
                let (lo, hi) = if from < to { (from, to) } else { (to, from) };
                (lo, hi, -1.0)
            })
            .collect();
        let triplets = build_upper_triplets(n, None, &dummy_off_diag);
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
        let triplets = build_upper_triplets(self.n, Some(diag), off_diag);
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

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Build faer Triplets for an upper-triangle symmetric matrix.
///
/// If `diag` is `None`, uses 1.0 for all diagonal entries and -1.0 for
/// off-diagonal (suitable for symbolic factorization with dummy values).
/// If `diag` is `Some`, uses actual diagonal values and the f64 in each
/// off-diagonal entry.
fn build_upper_triplets(
    n: usize,
    diag: Option<&[f64]>,
    off_diag: &[(usize, usize, f64)],
) -> Vec<Triplet<usize, usize, f64>> {
    let mut triplets = Vec::with_capacity(n + off_diag.len());

    // Diagonal entries.
    for i in 0..n {
        let d = diag.map_or(1.0, |v| v[i]);
        triplets.push(Triplet::new(i, i, d));
    }

    // Upper-triangle off-diagonal entries.
    for &(lo, hi, val) in off_diag {
        let v = if diag.is_some() { val } else { -1.0 };
        debug_assert!(lo < hi, "off_diag entries must have lo < hi");
        triplets.push(Triplet::new(lo, hi, v));
    }

    triplets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_solver_basic() {
        // A = [[4, -1], [-1, 4]], b = (3, 3) => x = (1, 1)
        let mut solver = FaerDirectSolver::new(2, &[(0, 1)]);
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
        let mut solver = FaerDirectSolver::new(n, &pipes);

        let diag = vec![4.0; n];
        let off_diag: Vec<(usize, usize, f64)> =
            (0..n - 1).map(|i| (i, i + 1, -1.0)).collect();
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        solver.solve(&diag, &off_diag, &rhs, &mut x);

        // Verify A*x = rhs manually
        for i in 0..n {
            let mut ax_i = diag[i] * x[i];
            for &(lo, hi, val) in &off_diag {
                if lo == i {
                    ax_i += val * x[hi];
                }
                if hi == i {
                    ax_i += val * x[lo];
                }
            }
            assert!(
                (ax_i - rhs[i]).abs() < 1e-8,
                "residual at {}: ax={}, rhs={}",
                i,
                ax_i,
                rhs[i]
            );
        }
    }
}

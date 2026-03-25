//! CPU backend using faer for sparse linear algebra and argmin for CG.
//!
//! `FaerDirectSolver` wraps faer's sparse LLT factorization with AMD ordering.
//! `faer_cg` uses argmin's Conjugate Gradient with a faer sparse matrix operator.

use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;

use super::backend::LinearSolver;

// ---------------------------------------------------------------------------
// Direct Solver (faer Cholesky)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Sparse matrix wrapper for argmin CG
// ---------------------------------------------------------------------------

/// Symmetric sparse matrix stored as faer CSC, implementing argmin's Operator
/// trait for use with `argmin::solver::conjugategradient::ConjugateGradient`.
struct FaerSparseOperator {
    /// Full symmetric CSC matrix (both triangles stored).
    mat: SparseColMat<usize, f64>,
}

impl FaerSparseOperator {
    /// Build a full symmetric CSC matrix from diag + upper-triangle off_diag.
    fn from_symmetric(n: usize, diag: &[f64], off_diag: &[(usize, usize, f64)]) -> Self {
        // Build full symmetric matrix (both triangles) for spmv.
        let mut triplets = Vec::with_capacity(n + 2 * off_diag.len());
        for i in 0..n {
            triplets.push(Triplet::new(i, i, diag[i]));
        }
        for &(lo, hi, val) in off_diag {
            triplets.push(Triplet::new(lo, hi, val)); // upper
            triplets.push(Triplet::new(hi, lo, val)); // lower (symmetric)
        }
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
            .expect("failed to build full symmetric sparse matrix");
        Self { mat }
    }

    /// Sparse matrix-vector product: result = A * x, using faer.
    fn spmv(&self, x: &[f64], result: &mut [f64]) {
        let n = self.mat.nrows();
        let x_col = faer::MatRef::from_column_major_slice(x, n, 1);
        let mut r_col = faer::MatMut::from_column_major_slice_mut(result, n, 1);
        use faer::sparse::linalg::matmul::sparse_dense_matmul;
        sparse_dense_matmul(
            r_col.as_mut(),
            faer::Accum::Replace,
            self.mat.as_ref(),
            x_col,
            1.0,
            faer::Par::Seq,
        );
    }
}

/// Implement argmin's `Operator` trait so CG can use our sparse matrix for A*x.
impl argmin::core::Operator for FaerSparseOperator {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut result = vec![0.0; param.len()];
        self.spmv(param, &mut result);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CG solver using argmin + faer
// ---------------------------------------------------------------------------

/// Jacobi-preconditioned Conjugate Gradient solver using argmin's CG engine
/// with faer sparse matrix for spmv.
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

    // Build faer sparse operator for A*x.
    let operator = FaerSparseOperator::from_symmetric(n, diag, off_diag);

    // Apply Jacobi preconditioning: scale the system by M^{-1/2} where M = diag(A).
    // Transform: A' = M^{-1/2} A M^{-1/2}, b' = M^{-1/2} b, x' = M^{1/2} x
    // This is equivalent to left-preconditioning in unpreconditioned CG.
    //
    // argmin's CG is unpreconditioned, so we apply Jacobi preconditioning
    // externally by transforming the system: solve M^{-1}Ax = M^{-1}b using
    // a wrapper operator that computes M^{-1}(A*x).
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
        .collect();

    let precond_operator = JacobiPreconditionedOperator {
        inner: operator,
        inv_diag: inv_diag.clone(),
    };

    // Preconditioned RHS: b' = M^{-1} * b
    let precond_rhs: Vec<f64> = rhs
        .iter()
        .zip(inv_diag.iter())
        .map(|(&b, &inv_d)| b * inv_d)
        .collect();

    // Run argmin CG.
    use argmin::core::{Executor, State};
    use argmin::solver::conjugategradient::ConjugateGradient;

    let cg: ConjugateGradient<Vec<f64>, f64> = ConjugateGradient::new(precond_rhs);
    let init_x: Vec<f64> = x.to_vec();

    let result = Executor::new(precond_operator, cg)
        .configure(|state| state.param(init_x).max_iters(max_iters as u64))
        .ctrlc(false)
        .run();

    match result {
        Ok(res) => {
            if let Some(best) = res.state().get_best_param() {
                x.copy_from_slice(best);
            }
            res.state().get_iter() as usize
        }
        Err(_) => {
            // If argmin fails (shouldn't happen for SPD), fall back to current x
            max_iters
        }
    }
}

/// Wrapper that applies Jacobi preconditioning: computes M^{-1} * A * x.
struct JacobiPreconditionedOperator {
    inner: FaerSparseOperator,
    inv_diag: Vec<f64>,
}

impl argmin::core::Operator for JacobiPreconditionedOperator {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut ax = self.inner.apply(param)?;
        for (v, &inv_d) in ax.iter_mut().zip(self.inv_diag.iter()) {
            *v *= inv_d;
        }
        Ok(ax)
    }
}

// ---------------------------------------------------------------------------
// Triplet builders
// ---------------------------------------------------------------------------

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
        let _iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "x[1] = {}", x[1]);
    }

    #[test]
    fn cg_with_offdiag() {
        // A = [[4, -1], [-1, 4]], b = (3, 3) => x = (1, 1)
        let diag = vec![4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0)];
        let rhs = vec![3.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let _iters = faer_cg(&diag, &off_diag, &rhs, &mut x, 1e-10, 100);
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1] = {}", x[1]);
    }

    #[test]
    fn faer_spmv_matches_manual() {
        // A = [[4, -1], [-1, 3]], x = [1, 2] => Ax = [2, 5]
        let diag = vec![4.0, 3.0];
        let off_diag = vec![(0usize, 1usize, -1.0f64)];
        let op = FaerSparseOperator::from_symmetric(2, &diag, &off_diag);
        let mut result = vec![0.0; 2];
        op.spmv(&[1.0, 2.0], &mut result);
        assert!((result[0] - 2.0).abs() < 1e-12, "result[0] = {}", result[0]);
        assert!((result[1] - 5.0).abs() < 1e-12, "result[1] = {}", result[1]);
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

        // Verify A*x = rhs using faer spmv
        let op = FaerSparseOperator::from_symmetric(n, &diag, &off_diag);
        let mut ax = vec![0.0; n];
        op.spmv(&x, &mut ax);
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

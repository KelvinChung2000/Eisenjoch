//! User-friendly sparse matrix with lazy faer CSC conversion.
//!
//! Stores a symmetric positive-definite matrix as diagonal + upper-triangle
//! off-diagonal entries. Converts to faer CSC lazily on first use and caches
//! the result. Modifications invalidate the cache.

use dyn_stack::{MemStack, StackReq};
use faer::sparse::{SparseColMat, Triplet};

/// Symmetric sparse matrix stored as diagonal + upper-triangle off-diagonal.
///
/// Provides a simple builder interface for constructing the matrix, and
/// lazy conversion to faer CSC for use with faer solvers.
pub struct SparseMatrix {
    n: usize,
    diag: Vec<f64>,
    off_diag: Vec<(usize, usize, f64)>, // upper triangle (lo < hi)
    /// Cached faer CSC (built lazily on first use, invalidated on modification)
    cached_csc: Option<SparseColMat<usize, f64>>,
}

impl SparseMatrix {
    /// Create a new empty matrix of size `n x n`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            diag: vec![0.0; n],
            off_diag: Vec::new(),
            cached_csc: None,
        }
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Diagonal elements.
    pub fn diag(&self) -> &[f64] {
        &self.diag
    }

    /// Mutable access to diagonal elements.
    pub fn diag_mut(&mut self) -> &mut [f64] {
        self.cached_csc = None;
        &mut self.diag
    }

    /// Off-diagonal entries (upper triangle only, lo < hi).
    pub fn off_diag(&self) -> &[(usize, usize, f64)] {
        &self.off_diag
    }

    /// Mutable access to off-diagonal entries.
    pub fn off_diag_mut(&mut self) -> &mut [(usize, usize, f64)] {
        self.cached_csc = None;
        &mut self.off_diag
    }

    /// Reserve capacity for off-diagonal entries.
    pub fn reserve_off_diag(&mut self, additional: usize) {
        self.off_diag.reserve(additional);
    }

    /// Add symmetric connection: A[i,i] += w, A[j,j] += w, A[i,j] = A[j,i] += -w
    pub fn add_connection(&mut self, i: usize, j: usize, weight: f64) {
        if i == j {
            return;
        }
        self.diag[i] += weight;
        self.diag[j] += weight;
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.off_diag.push((lo, hi, -weight));
        self.cached_csc = None;
    }

    /// Add anchor: A[i,i] += weight, rhs handled separately by caller.
    pub fn add_diagonal(&mut self, i: usize, weight: f64) {
        self.diag[i] += weight;
        self.cached_csc = None;
    }

    /// Raw entry (upper triangle only, lo < hi).
    pub fn add_entry(&mut self, lo: usize, hi: usize, val: f64) {
        debug_assert!(lo < hi);
        self.off_diag.push((lo, hi, val));
        self.cached_csc = None;
    }

    /// Set diagonal directly.
    pub fn set_diag(&mut self, i: usize, val: f64) {
        self.diag[i] = val;
        self.cached_csc = None;
    }

    /// Clear for reuse (keeps capacity).
    pub fn clear(&mut self) {
        self.diag.fill(0.0);
        self.off_diag.clear();
        self.cached_csc = None;
    }

    /// Add symmetric connections from an iterator of (i, j, weight) tuples.
    pub fn add_connections_from_iter<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (usize, usize, f64)>,
    {
        for (i, j, w) in iter {
            self.add_connection(i, j, w);
        }
    }

    /// Mean of diagonal entries.
    pub fn diagonal_mean(&self) -> f64 {
        let sum: f64 = self.diag.iter().sum();
        sum / self.n.max(1) as f64
    }

    /// Add a uniform shift to all diagonal entries: A[i,i] += shift for all i.
    pub fn add_uniform_diagonal_shift(&mut self, shift: f64) {
        for d in self.diag.iter_mut() {
            *d += shift;
        }
        self.cached_csc = None;
    }

    /// Get or build the full symmetric faer CSC matrix.
    pub fn to_faer_symmetric(&mut self) -> &SparseColMat<usize, f64> {
        if self.cached_csc.is_none() {
            let mut triplets = Vec::with_capacity(self.n + 2 * self.off_diag.len());
            for i in 0..self.n {
                triplets.push(Triplet::new(i, i, self.diag[i]));
            }
            for &(lo, hi, val) in &self.off_diag {
                triplets.push(Triplet::new(lo, hi, val));
                triplets.push(Triplet::new(hi, lo, val));
            }
            self.cached_csc = Some(
                SparseColMat::try_new_from_triplets(self.n, self.n, &triplets)
                    .expect("failed to build faer CSC matrix"),
            );
        }
        self.cached_csc.as_ref().unwrap()
    }

    /// Sparse matrix-vector product: result = A * x (using full symmetric matrix).
    /// Accepts pre-constructed faer matrices to enable reuse and avoid conversions.
    pub fn spmv_mat(&mut self, x: faer::MatRef<'_, f64>, result: faer::MatMut<'_, f64>) {
        let csc = self.to_faer_symmetric().clone();
        faer::sparse::linalg::matmul::sparse_dense_matmul(
            result,
            faer::Accum::Replace,
            csc.as_ref(),
            x,
            1.0,
            crate::solver::par(),
        );
    }

    /// Sparse matrix-vector product: result = A * x (slice-based convenience wrapper).
    /// NOTE: Creates temporary matrix objects on each call. Consider using spmv_mat() for
    /// repeated calls to avoid allocation overhead.
    pub fn spmv(&mut self, x: &[f64], result: &mut [f64]) {
        let x_col = faer::MatRef::from_column_major_slice(x, self.n, 1);
        let r_col = faer::MatMut::from_column_major_slice_mut(result, self.n, 1);
        self.spmv_mat(x_col, r_col);
    }
}

/// Wrapper for use with faer's CG. Holds a pre-built faer CSC.
///
/// This is needed because `LinOp` takes `&self` but `SparseMatrix::spmv`
/// needs `&mut self` for lazy caching. This wrapper clones the CSC once
/// at construction time.
pub struct SparseMatrixOp {
    csc: SparseColMat<usize, f64>,
    n: usize,
}

/// Borrowed sparse matrix operator over diagonal + upper-triangle entries.
///
/// This avoids rebuilding or cloning a faer CSC matrix when the numeric values
/// are updated frequently but the sparsity pattern stays fixed.
pub struct SparseMatrixOpRef<'a> {
    diag: &'a [f64],
    off_diag: &'a [(usize, usize, f64)],
    n: usize,
}

impl<'a> SparseMatrixOpRef<'a> {
    pub fn from_matrix(mat: &'a SparseMatrix) -> Self {
        Self {
            diag: mat.diag(),
            off_diag: mat.off_diag(),
            n: mat.n(),
        }
    }
}

impl std::fmt::Debug for SparseMatrixOpRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SparseMatrixOpRef(n={})", self.n)
    }
}

unsafe impl Sync for SparseMatrixOpRef<'_> {}

impl faer::matrix_free::LinOp<f64> for SparseMatrixOpRef<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let ncols = rhs.ncols();
        for col in 0..ncols {
            for i in 0..self.n {
                out[(i, col)] = self.diag[i] * rhs[(i, col)];
            }
            for &(lo, hi, val) in self.off_diag {
                let rhs_lo = rhs[(lo, col)];
                let rhs_hi = rhs[(hi, col)];
                out[(lo, col)] += val * rhs_hi;
                out[(hi, col)] += val * rhs_lo;
            }
        }
    }

    fn conj_apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

impl SparseMatrixOp {
    /// Build from a SparseMatrix, triggering CSC construction if needed.
    pub fn from_matrix(mat: &mut SparseMatrix) -> Self {
        let csc = mat.to_faer_symmetric().clone();
        Self { csc, n: mat.n() }
    }
}

impl std::fmt::Debug for SparseMatrixOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SparseMatrixOp(n={})", self.n)
    }
}

// SAFETY: SparseColMat<usize, f64> is Send+Sync (immutable data), and we only
// read from it in apply/conj_apply.
unsafe impl Sync for SparseMatrixOp {}

impl faer::matrix_free::LinOp<f64> for SparseMatrixOp {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        _stack: &mut MemStack,
    ) {
        faer::sparse::linalg::matmul::sparse_dense_matmul(
            out,
            faer::Accum::Replace,
            self.csc.as_ref(),
            rhs,
            1.0,
            par,
        );
    }

    fn conj_apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        _stack: &mut MemStack,
    ) {
        faer::sparse::linalg::matmul::sparse_dense_matmul(
            out,
            faer::Accum::Replace,
            self.csc.as_ref(),
            rhs,
            1.0,
            par,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_construction_and_spmv() {
        // A = [[4, -1], [-1, 3]], x = [1, 2] => Ax = [2, 5]
        let mut mat = SparseMatrix::new(2);
        mat.set_diag(0, 4.0);
        mat.set_diag(1, 3.0);
        mat.add_entry(0, 1, -1.0);

        let mut result = vec![0.0; 2];
        mat.spmv(&[1.0, 2.0], &mut result);
        assert!((result[0] - 2.0).abs() < 1e-12, "result[0] = {}", result[0]);
        assert!((result[1] - 5.0).abs() < 1e-12, "result[1] = {}", result[1]);
    }

    #[test]
    fn add_connection_symmetric() {
        // add_connection(0, 1, 2.0) should create:
        // A = [[2, -2], [-2, 2]]
        let mut mat = SparseMatrix::new(2);
        mat.add_connection(0, 1, 2.0);

        assert!((mat.diag()[0] - 2.0).abs() < 1e-12);
        assert!((mat.diag()[1] - 2.0).abs() < 1e-12);
        assert_eq!(mat.off_diag().len(), 1);
        assert!((mat.off_diag()[0].2 - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn clear_resets() {
        let mut mat = SparseMatrix::new(3);
        mat.add_connection(0, 1, 5.0);
        mat.add_diagonal(2, 3.0);
        mat.clear();

        assert!(mat.diag().iter().all(|&d| d == 0.0));
        assert!(mat.off_diag().is_empty());
    }

    #[test]
    fn sparse_matrix_op_linop() {
        use dyn_stack::MemBuffer;

        let mut mat = SparseMatrix::new(2);
        mat.set_diag(0, 4.0);
        mat.set_diag(1, 3.0);
        mat.add_entry(0, 1, -1.0);

        let op = SparseMatrixOp::from_matrix(&mut mat);

        let x = faer::MatRef::from_column_major_slice(&[1.0, 2.0], 2, 1);
        let mut result = vec![0.0; 2];
        let out = faer::MatMut::from_column_major_slice_mut(&mut result, 2, 1);

        let scratch = op.apply_scratch(1, crate::solver::par());
        let mut buf = MemBuffer::new(scratch);
        let mut stack = MemStack::new(&mut buf);

        use faer::matrix_free::LinOp;
        op.apply(out, x, crate::solver::par(), &mut stack);

        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - 5.0).abs() < 1e-12);
    }
}

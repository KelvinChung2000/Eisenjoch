//! Backend-swappable linear solver interface.
//!
//! CPU backend uses faer for sparse Cholesky and CG.
//! Future GPU backend would implement the same trait.

/// A precomputed sparse factorization that can solve Ax=b efficiently
/// for changing values but fixed sparsity pattern.
pub trait LinearSolver: Send {
    /// Solve A*x = rhs, where A has the same sparsity pattern as at construction
    /// but potentially different numeric values.
    ///
    /// `diag`: diagonal elements of A
    /// `off_diag`: upper-triangle entries (row, col, value) where row < col
    /// `rhs`: right-hand side vector
    /// `x`: solution vector (output)
    fn solve(
        &mut self,
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
        rhs: &[f64],
        x: &mut [f64],
    );
}

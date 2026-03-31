//! High-level sparse system builder for placement problems.
//!
//! Callers add connections and anchors, then solve.
//! This is the main interface used by HeAP and other placers.

use super::cg::solve_cg;
use super::preconditioner::JacobiPreconditioner;
use super::sparse_matrix::{SparseMatrix, SparseMatrixOp};

/// Builder for Ax=b systems in analytical placement.
///
/// Callers add connections and anchors, then solve.
/// The system represents a symmetric positive-definite matrix A stored as
/// diagonal elements plus upper-triangle off-diagonal entries.
pub struct SparseSystemBuilder {
    /// Number of variables.
    pub n: usize,
    /// Diagonal elements of A.
    pub diag: Vec<f64>,
    /// Off-diagonal entries: (row, col, weight). Only upper triangle stored
    /// (row < col), but the matrix is treated as symmetric.
    pub off_diag: Vec<(usize, usize, f64)>,
    /// Right-hand side vector b.
    pub rhs: Vec<f64>,
}

impl SparseSystemBuilder {
    /// Create a new empty system of size n.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            diag: vec![0.0; n],
            off_diag: Vec::new(),
            rhs: vec![0.0; n],
        }
    }

    /// Add a connection between movable cells i and j with the given weight.
    ///
    /// This adds weight to A[i,i] and A[j,j], and -weight to A[i,j] and A[j,i].
    pub fn add_connection(&mut self, i: usize, j: usize, weight: f64) {
        debug_assert!(i < self.n && j < self.n);
        if i == j {
            return;
        }
        self.diag[i] += weight;
        self.diag[j] += weight;
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.off_diag.push((lo, hi, -weight));
    }

    /// Add an anchor force pulling cell i toward position pos with the given weight.
    ///
    /// Adds weight to A[i,i] and weight*pos to rhs[i].
    pub fn add_anchor(&mut self, i: usize, pos: f64, weight: f64) {
        debug_assert!(i < self.n);
        self.diag[i] += weight;
        self.rhs[i] += weight * pos;
    }

    /// Solve using Jacobi-preconditioned CG via faer.
    ///
    /// Returns the number of iterations performed.
    pub fn solve_cg(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize {
        debug_assert_eq!(x.len(), self.n);
        if self.n == 0 {
            return 0;
        }

        // Build SparseMatrix from our data
        let mut mat = SparseMatrix::new(self.n);
        for (i, &d) in self.diag.iter().enumerate() {
            mat.set_diag(i, d);
        }
        for &(lo, hi, val) in &self.off_diag {
            mat.add_entry(lo, hi, val);
        }

        let op = SparseMatrixOp::from_matrix(&mut mat);
        let precond = JacobiPreconditioner::new(&self.diag);

        let rhs_mat = faer::MatRef::from_column_major_slice(&self.rhs, self.n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(x, self.n, 1);
        let result = solve_cg(&op, &precond, rhs_mat, x_mat, tol, max_iters);
        result.iterations
    }

    /// Clear all values for reuse (keeps allocated capacity).
    pub fn clear(&mut self) {
        self.diag.fill(0.0);
        self.off_diag.clear();
        self.rhs.fill(0.0);
    }
}

/// Trait for linear system solvers used in analytical placement.
///
/// Implementors solve A*x = b where A is a symmetric positive-definite matrix.
pub trait Solver {
    /// Solve the system, writing the solution into `x`.
    ///
    /// Returns the number of iterations used.
    fn solve(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize;
}

impl Solver for SparseSystemBuilder {
    fn solve(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize {
        self.solve_cg(x, tol, max_iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_builder_basic() {
        let mut sys = SparseSystemBuilder::new(2);
        sys.add_connection(0, 1, 1.0);
        sys.add_anchor(0, 0.0, 1.0);
        sys.add_anchor(1, 10.0, 1.0);

        // A = [[2, -1], [-1, 2]], rhs = [0, 10]
        // Solution: x = [10/3, 20/3] approx [3.33, 6.67]
        let mut x = vec![5.0, 5.0];
        let iters = sys.solve_cg(&mut x, 1e-10, 100);
        assert!(iters <= 10);
        assert!((x[0] - 10.0 / 3.0).abs() < 1e-4, "x[0] = {}", x[0]);
        assert!((x[1] - 20.0 / 3.0).abs() < 1e-4, "x[1] = {}", x[1]);
    }

    #[test]
    fn system_builder_trait() {
        let mut sys = SparseSystemBuilder::new(2);
        sys.add_connection(0, 1, 1.0);
        sys.add_anchor(0, 0.0, 1.0);
        sys.add_anchor(1, 10.0, 1.0);

        let mut x = vec![5.0, 5.0];
        let iters = Solver::solve(&sys, &mut x, 1e-10, 100);
        assert!(iters <= 10);
    }

    #[test]
    fn system_builder_clear() {
        let mut sys = SparseSystemBuilder::new(3);
        sys.add_connection(0, 1, 5.0);
        sys.add_anchor(2, 3.0, 1.0);
        sys.clear();
        assert!(sys.diag.iter().all(|&d| d == 0.0));
        assert!(sys.off_diag.is_empty());
        assert!(sys.rhs.iter().all(|&r| r == 0.0));
    }
}

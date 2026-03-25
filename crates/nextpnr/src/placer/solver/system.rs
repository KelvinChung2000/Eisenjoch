//! High-level sparse system builder for placement problems.
//!
//! Callers add connections and anchors, then solve.
//! This is the main interface used by HeAP and other placers.

use super::faer_backend::faer_cg;

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

    /// Solve using Jacobi-preconditioned CG.
    ///
    /// Returns the number of iterations performed.
    pub fn solve_cg(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize {
        debug_assert_eq!(x.len(), self.n);
        faer_cg(&self.diag, &self.off_diag, &self.rhs, x, tol, max_iters)
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

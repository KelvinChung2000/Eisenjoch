//! Conjugate Gradient solver for symmetric positive-definite sparse systems.
//!
//! Implements Jacobi-preconditioned CG. The system A*x = b uses a symmetric
//! matrix stored as diagonal elements plus upper-triangle off-diagonal triples.

use super::Solver;

/// Sparse linear system for analytical placement.
///
/// Represents the system A*x = b where A is a symmetric positive-definite
/// matrix stored as diagonal elements plus off-diagonal (i, j, weight) triples.
pub struct SparseSystem {
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

impl SparseSystem {
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
}

impl Solver for SparseSystem {
    fn solve(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize {
        debug_assert_eq!(x.len(), self.n);
        conjugate_gradient(&self.diag, &self.off_diag, &self.rhs, x, tol, max_iters)
    }
}

/// Symmetric sparse matrix-vector product: result = A * x.
///
/// A is represented by its diagonal and a list of upper-triangle off-diagonal
/// entries (i, j, weight) where i < j. The matrix is symmetric, so each
/// off-diagonal entry contributes to both (i,j) and (j,i).
pub fn spmv(diag: &[f64], off_diag: &[(usize, usize, f64)], x: &[f64], result: &mut [f64]) {
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
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Jacobi-preconditioned Conjugate Gradient solver for A*x = b.
///
/// Uses M^{-1} = diag(1/A[i,i]) as preconditioner, which halves iteration
/// count for diagonally-dominant systems typical of placement.
///
/// Returns the number of iterations performed.
pub fn conjugate_gradient(
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

    // Jacobi preconditioner: inv_diag[i] = 1 / A[i,i]
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    // r = b - A*x
    let mut ax = vec![0.0; n];
    spmv(diag, off_diag, x, &mut ax);
    let mut r: Vec<f64> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(bi, axi)| bi - axi)
        .collect();

    // z = M^{-1} * r
    let mut z: Vec<f64> = r.iter().zip(inv_diag.iter()).map(|(ri, mi)| ri * mi).collect();

    // p = z
    let mut p = z.clone();

    let mut rz_old = dot(&r, &z);

    // Convergence check on unpreconditioned residual.
    let rhs_norm_sq = dot(rhs, rhs);
    let tol_sq = tol * tol * rhs_norm_sq.max(1e-30);
    let rs = dot(&r, &r);

    if rs < tol_sq {
        return 0;
    }

    let mut ap = vec![0.0; n];

    for iter in 0..max_iters {
        // ap = A*p
        spmv(diag, off_diag, &p, &mut ap);

        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            return iter + 1;
        }

        let alpha = rz_old / p_ap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - alpha * A*p
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);

        if rs_new < tol_sq {
            return iter + 1;
        }

        // z = M^{-1} * r
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    max_iters
}

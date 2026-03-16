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
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Compute Jacobi preconditioner: inv_diag[i] = 1 / diag[i], with safeguard for near-zero.
fn jacobi_preconditioner(diag: &[f64]) -> Vec<f64> {
    diag.iter()
        .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
        .collect()
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

    let inv_diag = jacobi_preconditioner(diag);

    // r = b - A*x
    let mut ax = vec![0.0; n];
    spmv(diag, off_diag, x, &mut ax);
    let mut r: Vec<f64> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(bi, axi)| bi - axi)
        .collect();

    // z = M^{-1} * r
    let mut z: Vec<f64> = r
        .iter()
        .zip(inv_diag.iter())
        .map(|(ri, mi)| ri * mi)
        .collect();

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

/// Multigrid-preconditioned Conjugate Gradient solver.
///
/// Uses a multigrid V-cycle as preconditioner when the system matches a 4-port-per-tile
/// grid layout. Falls back to Jacobi PCG for small systems or non-grid layouts.
///
/// For 4-port-per-tile systems:
/// 1. Projects the residual to a scalar tile grid (average 4 ports)
/// 2. Applies one multigrid V-cycle on the scalar grid
/// 3. Broadcasts the tile solution back to 4 ports
pub fn multigrid_preconditioned_cg(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
    grid_width: usize,
    grid_height: usize,
) -> usize {
    let n = diag.len();
    let n_tiles = grid_width * grid_height;

    // Fall back to Jacobi PCG if system doesn't match 4-port layout or is small.
    if n != n_tiles * 4 || n_tiles <= 16 {
        return preconditioned_conjugate_gradient(diag, off_diag, rhs, x, tol, max_iters);
    }

    let mg = super::MultigridSolver::new(grid_width, grid_height);
    let inv_diag = jacobi_preconditioner(diag);

    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut tile_rhs = vec![0.0; n_tiles];
    let mut tile_z = vec![0.0; n_tiles];

    // Initial residual: r = b - A*x
    spmv(diag, off_diag, x, &mut ap);
    for i in 0..n {
        r[i] = rhs[i] - ap[i];
    }
    let rhs_norm = dot(rhs, rhs).sqrt().max(1e-12);

    // Apply multigrid preconditioner: z = M^{-1} * r
    mg_precondition(&mg, &r, &mut z, n_tiles, &inv_diag, &mut tile_rhs, &mut tile_z);
    p.copy_from_slice(&z);
    let mut rz_old = dot(&r, &z);

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

        mg_precondition(&mg, &r, &mut z, n_tiles, &inv_diag, &mut tile_rhs, &mut tile_z);

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    max_iters
}

/// Apply multigrid V-cycle as preconditioner.
///
/// Projects the 4-port-per-tile residual to a scalar tile grid (average 4 ports),
/// applies a V-cycle, and broadcasts back to 4 ports. Falls back to Jacobi for
/// ports that can't be mapped.
fn mg_precondition(
    mg: &super::MultigridSolver,
    r: &[f64],
    z: &mut [f64],
    n_tiles: usize,
    inv_diag: &[f64],
    tile_rhs: &mut [f64],
    tile_z: &mut [f64],
) {
    // Project: average 4 ports per tile to get scalar tile residual.
    for t in 0..n_tiles {
        let base = t * 4;
        tile_rhs[t] = (r[base] + r[base + 1] + r[base + 2] + r[base + 3]) / 4.0;
    }

    // Solve one V-cycle on the scalar tile grid.
    tile_z.fill(0.0);
    mg.solve(tile_rhs, tile_z, 1);

    // Broadcast tile solution back to 4 ports, blended with Jacobi.
    for t in 0..n_tiles {
        let base = t * 4;
        for port in 0..4 {
            let i = base + port;
            let jacobi = r[i] * inv_diag[i];
            // Blend: 50% multigrid + 50% Jacobi for smooth + high-frequency components.
            z[i] = 0.5 * tile_z[t] + 0.5 * jacobi;
        }
    }
}

/// Preconditioned Conjugate Gradient solver using a Jacobi (diagonal) preconditioner.
///
/// This variant uses a relative residual norm convergence criterion
/// (||r|| / ||b|| < tol) and is called by the Kirchhoff solver.
pub fn preconditioned_conjugate_gradient(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    let inv_diag = jacobi_preconditioner(diag);

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

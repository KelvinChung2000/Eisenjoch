//! Direct sparse solver facade.
//!
//! Wraps the sparse matrix, ordering, and Cholesky modules into a single
//! reusable solver object. Symbolic factorization is performed once;
//! numeric factorization and solve are performed for each new set of values.

use super::cholesky::{numeric_ldlt, symbolic_cholesky, SymbolicCholesky};
use super::ordering::{expand_to_ports, inverse_perm, nested_dissection_2d};
use super::sparse::{from_diag_and_upper_coo, CscMatrix};

/// Maps values from unpermuted diag+off_diag directly into a permuted CSC matrix.
/// Precomputed once during `DirectSolver::new()`, used every solve to avoid
/// allocating a fresh permuted CSC.
struct PermutedScatterMap {
    /// For each diagonal element i: position in permuted CSC values array.
    diag_targets: Vec<usize>,
    /// For each off-diagonal entry idx: position in permuted CSC values array.
    offdiag_targets: Vec<usize>,
}

/// Direct sparse solver for symmetric positive-definite systems.
///
/// Performs symbolic analysis once (ordering + elimination tree), then
/// supports repeated numeric factorization and solve with different values
/// but the same sparsity pattern.
pub struct DirectSolver {
    /// Matrix dimension.
    n: usize,
    /// Fill-reducing permutation: perm[new] = old.
    perm: Vec<usize>,
    /// Inverse permutation: inv_perm[old] = new.
    inv_perm: Vec<usize>,
    /// Symbolic Cholesky of the permuted matrix.
    symbolic: SymbolicCholesky,
    /// Preallocated permuted CSC matrix (reused every solve).
    permuted_csc: CscMatrix,
    /// Precomputed scatter map: diag+off_diag -> permuted CSC positions.
    scatter_map: PermutedScatterMap,
    /// Number of off-diagonal entries expected per solve.
    expected_offdiag_len: usize,
    /// Workspace for permuted RHS / solution.
    work_rhs: Vec<f64>,
    work_x: Vec<f64>,
}

impl DirectSolver {
    /// Create a new direct solver for a system of size `n` with the given
    /// sparsity pattern (pipe endpoints as (from, to) pairs).
    ///
    /// `grid_width` and `grid_height` are the tile grid dimensions for nested
    /// dissection ordering. The system has 4 ports per tile (n = 4 * W * H).
    ///
    /// Performs symbolic factorization (ordering + elimination tree). This is
    /// the expensive one-time cost.
    pub fn new(
        n: usize,
        pipe_endpoints: &[(usize, usize)],
        grid_width: usize,
        grid_height: usize,
    ) -> Self {
        // Build initial CSC from a dummy system to get the structure.
        let diag = vec![1.0; n];
        let off_diag: Vec<(usize, usize, f64)> = pipe_endpoints
            .iter()
            .map(|&(from, to)| {
                let (lo, hi) = if from < to { (from, to) } else { (to, from) };
                (lo, hi, -1.0)
            })
            .collect();

        let (work_csc, _) = from_diag_and_upper_coo(n, &diag, &off_diag);

        // Compute fill-reducing ordering via nested dissection on the tile grid.
        let n_tiles = grid_width * grid_height;
        let (perm, inv_perm) = if n == n_tiles * 4 && n_tiles > 16 {
            let tile_perm = nested_dissection_2d(grid_width, grid_height);
            let port_perm = expand_to_ports(&tile_perm);
            let port_inv = inverse_perm(&port_perm);
            (port_perm, port_inv)
        } else {
            let perm: Vec<usize> = (0..n).collect();
            let inv_perm = perm.clone();
            (perm, inv_perm)
        };

        // Apply permutation to get the permuted CSC structure.
        let permuted_csc = work_csc.permute(&perm, &inv_perm);

        // Symbolic factorization on the permuted matrix.
        let symbolic = symbolic_cholesky(&permuted_csc);

        eprintln!(
            "DirectSolver: n={}, A_nnz={}, L_nnz={}, fill_ratio={:.1}x, L_mem={:.1}MB",
            n, n + 2 * off_diag.len(), symbolic.l_nnz(),
            symbolic.l_nnz() as f64 / off_diag.len().max(1) as f64,
            (symbolic.l_nnz() * 16) as f64 / 1024.0 / 1024.0, // row_idx + value per entry
        );

        // Build scatter map: for each diag[i] and off_diag[idx], find the
        // corresponding position in the permuted CSC values array.
        let scatter_map = build_scatter_map(
            n,
            &off_diag,
            &inv_perm,
            &permuted_csc,
        );

        let expected_offdiag_len = off_diag.len();

        Self {
            n,
            perm,
            inv_perm,
            symbolic,
            permuted_csc,
            scatter_map,
            expected_offdiag_len,
            work_rhs: vec![0.0; n],
            work_x: vec![0.0; n],
        }
    }

    /// Solve A*x = rhs where A is defined by diagonal and off-diagonal entries.
    ///
    /// This performs:
    /// 1. Scatter diag/off_diag values into permuted CSC (O(nnz), no allocation)
    /// 2. Numeric LDL^T factorization
    /// 3. Forward/backward solve
    /// 4. Unpermute solution
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

        // Step 1: Scatter values into preallocated permuted CSC.
        self.permuted_csc.values.fill(0.0);

        // Scatter diagonal: each diag[old_i] goes to the permuted diagonal position.
        for (old_i, &d) in diag.iter().enumerate() {
            self.permuted_csc.values[self.scatter_map.diag_targets[old_i]] += d;
        }

        // Scatter off-diagonal: each entry maps to a permuted off-diag position
        // AND contributes to two diagonal positions.
        for (idx, &(_lo, _hi, w)) in off_diag.iter().enumerate() {
            self.permuted_csc.values[self.scatter_map.offdiag_targets[idx]] += w;
            // Off-diagonal entries also contribute to the diagonal via add_connection:
            // diag[lo] += weight, diag[hi] += weight. But that's already in the
            // diag[] array passed by the caller. So we only scatter the off-diag value.
        }

        // Step 2: Numeric factorization.
        let numeric = numeric_ldlt(&self.symbolic, &self.permuted_csc);

        // Step 3: Permute RHS: y[new] = b[perm[new]].
        for new_idx in 0..self.n {
            self.work_rhs[new_idx] = rhs[self.perm[new_idx]];
        }

        // Step 4: Solve in permuted space.
        numeric.solve(&self.work_rhs, &mut self.work_x);

        // Step 5: Unpermute solution: x[old] = work_x[inv_perm[old]].
        for old_idx in 0..self.n {
            x[old_idx] = self.work_x[self.inv_perm[old_idx]];
        }
    }
}

/// Build the scatter map that maps unpermuted diag/off_diag positions
/// to positions in the permuted CSC values array.
fn build_scatter_map(
    n: usize,
    off_diag: &[(usize, usize, f64)],
    inv_perm: &[usize],
    permuted_csc: &CscMatrix,
) -> PermutedScatterMap {
    // For each original diagonal element diag[old_i]:
    // In permuted space, this becomes diagonal element at new_i = inv_perm[old_i].
    // Find the position of (new_i, new_i) in permuted_csc.
    let mut diag_targets = vec![0usize; n];
    for old_i in 0..n {
        let new_i = inv_perm[old_i];
        // Diagonal is the first entry in column new_i (row == col).
        let col_start = permuted_csc.col_ptrs[new_i];
        debug_assert_eq!(permuted_csc.row_indices[col_start], new_i);
        diag_targets[old_i] = col_start;
    }

    // For each original off_diag entry (lo, hi, w):
    // In permuted space: new_lo = inv_perm[lo], new_hi = inv_perm[hi].
    // The lower-triangle entry is at (max(new_lo, new_hi), min(new_lo, new_hi)).
    let mut offdiag_targets = vec![0usize; off_diag.len()];
    for (idx, &(lo, hi, _)) in off_diag.iter().enumerate() {
        let new_lo = inv_perm[lo];
        let new_hi = inv_perm[hi];
        let (col, row) = if new_lo < new_hi {
            (new_lo, new_hi)
        } else {
            (new_hi, new_lo)
        };

        // Find position of (row, col) in permuted_csc column `col`.
        let col_start = permuted_csc.col_ptrs[col];
        let col_end = permuted_csc.col_ptrs[col + 1];
        let pos = permuted_csc.row_indices[col_start..col_end]
            .binary_search(&row)
            .unwrap_or_else(|_| {
                panic!(
                    "permuted CSC missing entry ({}, {}) from original ({}, {})",
                    row, col, lo, hi
                )
            });
        offdiag_targets[idx] = col_start + pos;

    }

    PermutedScatterMap {
        diag_targets,
        offdiag_targets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_residual(
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
        x: &[f64],
        rhs: &[f64],
        tol: f64,
        label: &str,
    ) {
        let n = diag.len();
        let mut ax = vec![0.0; n];
        crate::placer::solver::cg::spmv(diag, off_diag, x, &mut ax);
        let res_norm: f64 = rhs.iter().zip(ax.iter())
            .map(|(b, a)| (b - a) * (b - a)).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-30);
        let rel = res_norm / rhs_norm;
        assert!(rel < tol, "{}: relative residual {:.2e} exceeds {:.2e}", label, rel, tol);
    }

    /// Build a 4-port-per-tile grid system mimicking the Kirchhoff pipe network.
    fn make_grid_system(
        w: usize,
        h: usize,
    ) -> (usize, Vec<f64>, Vec<(usize, usize, f64)>, Vec<(usize, usize)>) {
        let n = w * h * 4;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        let mut endpoints = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 4;
                // Intra-tile: connect all 4 ports.
                for p1 in 0..4 {
                    for p2 in (p1 + 1)..4 {
                        let i = base + p1;
                        let j = base + p2;
                        diag[i] += 0.5;
                        diag[j] += 0.5;
                        off_diag.push((i, j, -0.5));
                        endpoints.push((i, j));
                    }
                }
                if x + 1 < w {
                    let i = base + 1;
                    let j = (y * w + x + 1) * 4 + 3;
                    diag[i] += 1.0;
                    diag[j] += 1.0;
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    off_diag.push((lo, hi, -1.0));
                    endpoints.push((lo, hi));
                }
                if y + 1 < h {
                    let i = base + 2;
                    let j = ((y + 1) * w + x) * 4;
                    diag[i] += 1.0;
                    diag[j] += 1.0;
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    off_diag.push((lo, hi, -1.0));
                    endpoints.push((lo, hi));
                }
            }
        }
        diag[0] += 1e10;
        (n, diag, off_diag, endpoints)
    }

    /// Build a 4-port grid with non-uniform conductances (closer to real Kirchhoff).
    fn make_nonuniform_grid_system(
        w: usize,
        h: usize,
    ) -> (usize, Vec<f64>, Vec<(usize, usize, f64)>, Vec<(usize, usize)>) {
        let n = w * h * 4;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        let mut endpoints = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 4;
                let tile_idx = y * w + x;
                // Intra-tile with variable conductance.
                let g_intra = 0.2 + 0.8 * ((tile_idx as f64) / (w * h) as f64);
                for p1 in 0..4 {
                    for p2 in (p1 + 1)..4 {
                        let i = base + p1;
                        let j = base + p2;
                        diag[i] += g_intra;
                        diag[j] += g_intra;
                        off_diag.push((i, j, -g_intra));
                        endpoints.push((i, j));
                    }
                }
                if x + 1 < w {
                    let i = base + 1;
                    let j = (y * w + x + 1) * 4 + 3;
                    let g = 0.5 + 4.5 * ((x as f64 + 1.0) / w as f64);
                    diag[i] += g;
                    diag[j] += g;
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    off_diag.push((lo, hi, -g));
                    endpoints.push((lo, hi));
                }
                if y + 1 < h {
                    let i = base + 2;
                    let j = ((y + 1) * w + x) * 4;
                    let g = 0.3 + 3.0 * ((y as f64 + 1.0) / h as f64);
                    diag[i] += g;
                    diag[j] += g;
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    off_diag.push((lo, hi, -g));
                    endpoints.push((lo, hi));
                }
            }
        }
        diag[0] += 1e10;
        (n, diag, off_diag, endpoints)
    }

    // ---- Basic correctness ----

    #[test]
    fn direct_solver_4x4_grid() {
        let (n, diag, off_diag, endpoints) = make_grid_system(4, 4);
        let mut solver = DirectSolver::new(n, &endpoints, 4, 4);
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-8, "4x4 unit rhs");
    }

    #[test]
    fn direct_solver_8x8_grid() {
        let (n, diag, off_diag, endpoints) = make_grid_system(8, 8);
        let mut solver = DirectSolver::new(n, &endpoints, 8, 8);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "8x8 oscillatory");
    }

    #[test]
    fn direct_solver_16x16_grid() {
        let (n, diag, off_diag, endpoints) = make_grid_system(16, 16);
        let mut solver = DirectSolver::new(n, &endpoints, 16, 16);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).cos()).collect();
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "16x16 grid");
    }

    // ---- Non-uniform conductances (real Kirchhoff-like) ----

    #[test]
    fn direct_solver_nonuniform_8x8() {
        let (n, diag, off_diag, endpoints) = make_nonuniform_grid_system(8, 8);
        let mut solver = DirectSolver::new(n, &endpoints, 8, 8);
        let rhs: Vec<f64> = (0..n).map(|i| if i % 4 == 0 { 1.0 } else { -0.3 }).collect();
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "nonuniform 8x8");
    }

    // ---- Reuse: same solver, multiple solves with different values ----

    #[test]
    fn direct_solver_reuse_values() {
        let (n, mut diag, off_diag, endpoints) = make_grid_system(4, 4);
        let mut solver = DirectSolver::new(n, &endpoints, 4, 4);

        // First solve.
        let rhs = vec![1.0; n];
        let mut x1 = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x1);
        check_residual(&diag, &off_diag, &x1, &rhs, 1e-8, "reuse solve 1");

        // Change diagonal and solve again.
        diag[1] += 10.0;
        let mut x2 = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x2);
        check_residual(&diag, &off_diag, &x2, &rhs, 1e-8, "reuse solve 2");

        // Solutions must differ.
        let diff: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-10, "solutions should differ after value change");
    }

    #[test]
    fn direct_solver_reuse_10_solves() {
        let (n, diag, off_diag, endpoints) = make_grid_system(6, 6);
        let mut solver = DirectSolver::new(n, &endpoints, 6, 6);

        for k in 0..10 {
            // Vary the diagonal to simulate Newton iterations.
            let mut d = diag.clone();
            for i in 0..n {
                d[i] += 0.1 * k as f64 * (i as f64 * 0.1).sin().abs();
            }
            let rhs: Vec<f64> = (0..n).map(|i| ((i + k * 7) as f64 * 0.2).sin()).collect();
            let mut x = vec![0.0; n];
            solver.solve(&d, &off_diag, &rhs, &mut x);
            check_residual(&d, &off_diag, &x, &rhs, 1e-6, &format!("10-solve #{}", k));
        }
    }

    // ---- Matches CG reference ----

    #[test]
    fn direct_solver_matches_cg() {
        let (n, diag, off_diag, endpoints) = make_grid_system(5, 5);
        let mut solver = DirectSolver::new(n, &endpoints, 5, 5);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();

        let mut x_direct = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x_direct);

        let mut x_cg = vec![0.0; n];
        crate::placer::solver::cg::conjugate_gradient(
            &diag, &off_diag, &rhs, &mut x_cg, 1e-12, 10000,
        );

        let max_diff: f64 = x_direct.iter().zip(x_cg.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-4, "direct vs CG max diff = {:.2e}", max_diff);
    }

    // ---- Anchored node (pressure reference) ----

    #[test]
    fn direct_solver_anchor_node_near_zero() {
        let (n, diag, off_diag, endpoints) = make_grid_system(8, 8);
        let mut solver = DirectSolver::new(n, &endpoints, 8, 8);
        let mut rhs = vec![0.0; n];
        rhs[0] = 0.0; // Anchored node gets zero RHS
        for i in 1..n {
            rhs[i] = (i as f64 * 0.3).sin();
        }
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "anchor near zero");
        // Anchored node should have near-zero pressure.
        assert!(x[0].abs() < 1e-3, "anchored node x[0] = {:.2e} (should be ~0)", x[0]);
    }

    // ---- Small system (no nested dissection, natural ordering) ----

    #[test]
    fn direct_solver_2x2_no_nd() {
        // 2x2 grid = 16 junctions. n_tiles=4 <= 16, so natural ordering.
        let (n, diag, off_diag, endpoints) = make_grid_system(2, 2);
        let mut solver = DirectSolver::new(n, &endpoints, 2, 2);
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-8, "2x2 no ND");
    }

    // ---- Rectangular grid ----

    #[test]
    fn direct_solver_rectangular_10x3() {
        let (n, diag, off_diag, endpoints) = make_grid_system(10, 3);
        let mut solver = DirectSolver::new(n, &endpoints, 10, 3);
        let rhs: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { 2.0 } else { -0.1 }).collect();
        let mut x = vec![0.0; n];
        solver.solve(&diag, &off_diag, &rhs, &mut x);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "10x3 rectangular");
    }
}

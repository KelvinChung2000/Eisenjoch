//! Sparse LDL^T Cholesky factorization for symmetric positive-definite systems.
//!
//! Two-phase approach:
//! 1. **Symbolic**: compute elimination tree and nonzero pattern of L from sparsity
//!    structure alone (no numeric values). Called once per grid geometry.
//! 2. **Numeric**: compute L and D values using the precomputed structure. Called
//!    each time the matrix values change (e.g., each Newton step).
//!
//! Uses left-looking LDL^T (no square roots). Data layouts use contiguous arrays
//! for future GPU compute shader portability.

use super::sparse::CscMatrix;

/// Result of symbolic Cholesky analysis.
///
/// Contains the sparsity pattern of L and the elimination tree.
/// Reused across multiple numeric factorizations with different values.
pub struct SymbolicCholesky {
    /// Matrix dimension.
    pub n: usize,
    /// CSC column pointers for L (length n+1).
    pub l_col_ptrs: Vec<usize>,
    /// CSC row indices for L (length l_nnz). Sorted ascending per column.
    /// Each column j has entries for rows > j (strictly lower triangle).
    pub l_row_indices: Vec<usize>,
    /// Elimination tree: `etree[j] = Some(parent)` where parent is the
    /// minimum row > j in column j of L, or `None` for root nodes.
    pub etree: Vec<Option<usize>>,
}

/// Result of numeric LDL^T factorization.
///
/// Stores L (unit lower triangular) and D (diagonal) such that A = L * D * L^T.
/// L is stored in CSC format with the same structure as `SymbolicCholesky`.
pub struct NumericCholesky<'a> {
    /// Reference to the symbolic structure.
    pub symbolic: &'a SymbolicCholesky,
    /// Numeric values of L in CSC layout (same positions as `l_row_indices`).
    /// L has implicit 1s on the diagonal (unit lower triangular).
    pub l_values: Vec<f64>,
    /// Diagonal entries D[j] for j = 0..n.
    pub d: Vec<f64>,
}

impl SymbolicCholesky {
    /// Number of nonzeros in L (excluding the implicit diagonal of 1s).
    pub fn l_nnz(&self) -> usize {
        self.l_col_ptrs[self.n]
    }
}

/// Compute the symbolic Cholesky factorization of a lower-triangular CSC matrix.
///
/// The input `csc` must store the lower triangle (including diagonal) of an SPD matrix.
/// This function computes the elimination tree and the sparsity pattern of L.
///
/// The matrix should already be permuted if a fill-reducing ordering is desired.
pub fn symbolic_cholesky(csc: &CscMatrix) -> SymbolicCholesky {
    let n = csc.n;

    // Phase 1: Build row lists for efficient row-access of lower-triangle CSC.
    // row_lists[k] = columns j < k that have A[k,j] != 0.
    let mut row_lists: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        for pos in csc.col_ptrs[j]..csc.col_ptrs[j + 1] {
            let i = csc.row_indices[pos];
            if i > j {
                row_lists[i].push(j);
            }
        }
    }

    // Phase 2: Compute elimination tree using standard row-based algorithm.
    // Reference: Gilbert, Ng, Peyton 1994; Tim Davis "Direct Methods" Algorithm 4.1.
    //
    // etree[j] = min{ i > j : L[i,j] != 0 } = parent of j.
    // Process rows k = 0..n-1. For each column j < k with A[k,j] != 0,
    // walk from j up the partial forest to root r, set parent[r] = k.
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut anc: Vec<usize> = (0..n).collect(); // union-find with path compression

    for k in 0..n {
        for &j in &row_lists[k] {
            let mut r = j;
            while anc[r] != r && anc[r] != k {
                let next = anc[r];
                anc[r] = k;
                r = next;
            }
            if r != k {
                if parent[r].is_none() {
                    parent[r] = Some(k);
                }
                anc[r] = k;
            }
        }
    }

    // Phase 3: Compute column patterns of L using etree-based fill propagation.
    // For column j of L, the row set includes:
    //   - Original rows from A[:,j] with i > j
    //   - Fill inherited from children: if parent[c] = j, rows of L[:,c] that are > j
    //
    // Process j = 0..n-1 (children before parents since parent[j] > j).
    let mut col_sets: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Initialize with structural nonzeros from A.
    for j in 0..n {
        for pos in csc.col_ptrs[j]..csc.col_ptrs[j + 1] {
            let i = csc.row_indices[pos];
            if i > j {
                col_sets[j].push(i);
            }
        }
    }

    // Propagate fill through the etree.
    for j in 0..n {
        if let Some(p) = parent[j] {
            let rows: Vec<usize> = col_sets[j].iter().copied().filter(|&i| i > p).collect();
            for i in rows {
                col_sets[p].push(i);
            }
        }
    }

    // Deduplicate and sort each column's row set.
    for j in 0..n {
        col_sets[j].sort_unstable();
        col_sets[j].dedup();
    }

    // Phase 4: Build CSC structure for L.
    let mut l_col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        l_col_ptrs[j + 1] = l_col_ptrs[j] + col_sets[j].len();
    }

    let l_nnz = l_col_ptrs[n];
    let mut l_row_indices = vec![0usize; l_nnz];

    for j in 0..n {
        let start = l_col_ptrs[j];
        for (k, &row) in col_sets[j].iter().enumerate() {
            l_row_indices[start + k] = row;
        }
    }

    SymbolicCholesky {
        n,
        l_col_ptrs,
        l_row_indices,
        etree: parent,
    }
}

/// Compute the numeric LDL^T factorization using a precomputed symbolic structure.
///
/// The input `csc` must be the same (permuted) matrix whose sparsity was used for
/// `symbolic_cholesky`. Only the numeric values may differ.
///
/// Returns L (unit lower triangular) and D (diagonal) such that A = L * D * L^T.
pub fn numeric_ldlt<'a>(
    sym: &'a SymbolicCholesky,
    csc: &CscMatrix,
) -> NumericCholesky<'a> {
    let n = sym.n;
    let mut l_values = vec![0.0f64; sym.l_nnz()];
    let mut d = vec![0.0f64; n];

    // Workspace: dense column accumulator.
    let mut work = vec![0.0f64; n];

    // Linked-list for left-looking factorization.
    // head[j] = first column whose next update targets row j.
    let mut head: Vec<Option<usize>> = vec![None; n];
    let mut next: Vec<Option<usize>> = vec![None; n];
    let mut col_cursor: Vec<usize> = sym.l_col_ptrs[..n].to_vec();

    for j in 0..n {
        // Scatter column j of A into work.
        for pos in csc.col_ptrs[j]..csc.col_ptrs[j + 1] {
            let i = csc.row_indices[pos];
            work[i] = csc.values[pos];
        }

        // Process all columns k linked to row j (left-looking updates).
        let mut k_opt = head[j];
        while let Some(k) = k_opt {
            let next_k = next[k];

            let ljk_pos = col_cursor[k];
            let ljk = l_values[ljk_pos];
            let dk = d[k];
            let ljk_dk = ljk * dk;

            // Update diagonal: work[j] -= L[j,k]^2 * D[k]
            work[j] -= ljk * ljk_dk;

            // Update off-diagonal: work[i] -= L[i,k] * L[j,k] * D[k]
            let k_end = sym.l_col_ptrs[k + 1];
            for pos in (ljk_pos + 1)..k_end {
                let i = sym.l_row_indices[pos];
                work[i] -= l_values[pos] * ljk_dk;
            }

            // Advance cursor and re-link.
            col_cursor[k] = ljk_pos + 1;
            if col_cursor[k] < k_end {
                let next_row = sym.l_row_indices[col_cursor[k]];
                next[k] = head[next_row];
                head[next_row] = Some(k);
            }

            k_opt = next_k;
        }

        // Store D[j].
        d[j] = work[j];
        work[j] = 0.0;

        if d[j].abs() < 1e-30 {
            debug_assert!(false, "near-singular pivot d[{}] = {:.2e}", j, d[j]);
            d[j] = 1e-30;
        }

        // Compute L[:,j] = work[:] / D[j].
        let d_inv = 1.0 / d[j];
        let j_start = sym.l_col_ptrs[j];
        let j_end = sym.l_col_ptrs[j + 1];

        for pos in j_start..j_end {
            let i = sym.l_row_indices[pos];
            l_values[pos] = work[i] * d_inv;
            work[i] = 0.0;
        }

        // Link column j to its first target row.
        if j_start < j_end {
            let first_row = sym.l_row_indices[j_start];
            next[j] = head[first_row];
            head[first_row] = Some(j);
            col_cursor[j] = j_start;
        }
    }

    NumericCholesky {
        symbolic: sym,
        l_values,
        d,
    }
}

impl<'a> NumericCholesky<'a> {
    /// Solve A*x = b via L*D*L^T * x = b.
    ///
    /// Steps:
    /// 1. Forward substitution: L * y = b
    /// 2. Diagonal solve: D * z = y
    /// 3. Backward substitution: L^T * x = z
    pub fn solve(&self, rhs: &[f64], x: &mut [f64]) {
        let n = self.symbolic.n;
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(x.len(), n);

        x.copy_from_slice(rhs);

        // Forward substitution: L * y = b (L is unit lower triangular).
        for j in 0..n {
            let xj = x[j];
            for pos in self.symbolic.l_col_ptrs[j]..self.symbolic.l_col_ptrs[j + 1] {
                let i = self.symbolic.l_row_indices[pos];
                x[i] -= self.l_values[pos] * xj;
            }
        }

        // Diagonal solve: D * z = y.
        for j in 0..n {
            x[j] /= self.d[j];
        }

        // Backward substitution: L^T * x = z (L^T is unit upper triangular).
        for j in (0..n).rev() {
            let mut sum = 0.0;
            for pos in self.symbolic.l_col_ptrs[j]..self.symbolic.l_col_ptrs[j + 1] {
                let i = self.symbolic.l_row_indices[pos];
                sum += self.l_values[pos] * x[i];
            }
            x[j] -= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::solver::sparse::from_diag_and_upper_coo;

    /// Verify A*x = b residual using the original diag+off_diag representation.
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
        let res_norm: f64 = rhs
            .iter()
            .zip(ax.iter())
            .map(|(b, a)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-30);
        let rel = res_norm / rhs_norm;
        assert!(rel < tol, "{}: relative residual {:.2e} exceeds {:.2e}", label, rel, tol);
    }

    /// Build the full solve pipeline from diag+off_diag: CSC -> symbolic -> numeric -> solve.
    fn solve_system(
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
        rhs: &[f64],
    ) -> Vec<f64> {
        let n = diag.len();
        let (csc, _) = from_diag_and_upper_coo(n, diag, off_diag);
        let sym = symbolic_cholesky(&csc);
        let num = numeric_ldlt(&sym, &csc);
        let mut x = vec![0.0; n];
        num.solve(rhs, &mut x);
        x
    }

    fn make_grid_laplacian(w: usize, h: usize) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let n = w * h;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    diag[i] += 1.0;
                    diag[j] += 1.0;
                    off_diag.push((i, j, -1.0));
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    diag[i] += 1.0;
                    diag[j] += 1.0;
                    off_diag.push((i, j, -1.0));
                }
            }
        }
        diag[0] += 1e6;
        (diag, off_diag)
    }

    // ---- Symbolic analysis tests ----

    #[test]
    fn symbolic_3x3_tridiagonal() {
        let diag = vec![1.0 + 1e6, 2.0, 1.0];
        let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
        let (csc, _) = from_diag_and_upper_coo(3, &diag, &off_diag);
        let sym = symbolic_cholesky(&csc);

        assert_eq!(sym.n, 3);
        // Tridiagonal: no fill. L has col 0 -> row 1, col 1 -> row 2.
        assert_eq!(sym.l_nnz(), 2);
        assert_eq!(sym.etree[0], Some(1));
        assert_eq!(sym.etree[1], Some(2));
        assert_eq!(sym.etree[2], None);
    }

    #[test]
    fn symbolic_4x4_arrowhead() {
        // Arrowhead matrix: node 0 connected to all others.
        // A = [3 -1 -1 -1; -1 1 0 0; -1 0 1 0; -1 0 0 1]
        // L should be dense below diagonal of column 0, then no fill.
        let diag = vec![3.0, 1.0, 1.0, 1.0];
        let off_diag = vec![(0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0)];
        let (csc, _) = from_diag_and_upper_coo(4, &diag, &off_diag);
        let sym = symbolic_cholesky(&csc);

        // Column 0: rows 1,2,3. Column 1: rows 2,3 (fill!). Column 2: row 3 (fill!).
        // etree: 0->1->2->3
        assert_eq!(sym.etree[0], Some(1));
        // Total off-diag: 3 + 2 + 1 = 6
        assert_eq!(sym.l_nnz(), 6);
    }

    #[test]
    fn symbolic_5x5_grid_fill_count() {
        // 5-node path: 0-1-2-3-4. Tridiagonal => no fill.
        let n = 5;
        let mut diag = vec![0.0; n];
        let mut off_diag = Vec::new();
        for i in 0..n - 1 {
            diag[i] += 1.0;
            diag[i + 1] += 1.0;
            off_diag.push((i, i + 1, -1.0));
        }
        diag[0] += 1e6;
        let (csc, _) = from_diag_and_upper_coo(n, &diag, &off_diag);
        let sym = symbolic_cholesky(&csc);
        // Tridiagonal: exactly n-1 off-diagonal entries in L, zero fill.
        assert_eq!(sym.l_nnz(), n - 1);
    }

    // ---- Numeric solve correctness tests ----

    #[test]
    fn solve_1x1_trivial() {
        let diag = vec![5.0];
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs = vec![10.0];
        let x = solve_system(&diag, &off_diag, &rhs);
        assert!((x[0] - 2.0).abs() < 1e-12, "expected 2.0, got {}", x[0]);
    }

    #[test]
    fn solve_2x2_known_answer() {
        // A = [4 -1; -1 4], b = [3, 3]. Solution: x = [1, 1].
        let diag = vec![4.0, 4.0];
        let off_diag = vec![(0, 1, -1.0)];
        let rhs = vec![3.0, 3.0];
        let x = solve_system(&diag, &off_diag, &rhs);
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1]={}", x[1]);
    }

    #[test]
    fn solve_3x3_tridiagonal() {
        let diag = vec![1.0 + 1e6, 2.0, 1.0];
        let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
        let rhs = vec![1.0, 0.0, 0.0];
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-10, "3x3 tridiag");
    }

    #[test]
    fn solve_8x8_grid_unit_rhs() {
        let (diag, off_diag) = make_grid_laplacian(8, 8);
        let mut rhs = vec![0.0; 64];
        rhs[0] = 1.0;
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "8x8 unit rhs");
    }

    #[test]
    fn solve_8x8_grid_oscillatory_rhs() {
        let (diag, off_diag) = make_grid_laplacian(8, 8);
        let n = 64;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "8x8 oscillatory rhs");
    }

    #[test]
    fn solve_16x16_grid() {
        let (diag, off_diag) = make_grid_laplacian(16, 16);
        let n = 256;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "16x16 grid");
    }

    #[test]
    fn solve_32x32_grid() {
        let (diag, off_diag) = make_grid_laplacian(32, 32);
        let n = 1024;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).cos()).collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "32x32 grid");
    }

    #[test]
    fn solve_rectangular_grid_20x5() {
        let (diag, off_diag) = make_grid_laplacian(20, 5);
        let n = 100;
        let rhs: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { -0.5 }).collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "20x5 rect grid");
    }

    // ---- Tests against CG reference ----

    #[test]
    fn solve_matches_cg_8x8() {
        let (diag, off_diag) = make_grid_laplacian(8, 8);
        let n = 64;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();

        // Direct solve.
        let x_direct = solve_system(&diag, &off_diag, &rhs);

        // CG solve (high iteration limit to converge).
        let mut x_cg = vec![0.0; n];
        crate::placer::solver::cg::conjugate_gradient(
            &diag, &off_diag, &rhs, &mut x_cg, 1e-12, 5000,
        );

        // Both solutions should agree.
        let max_diff: f64 = x_direct
            .iter()
            .zip(x_cg.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-4,
            "direct vs CG max diff = {:.2e} (should be small)",
            max_diff,
        );
    }

    // ---- Numeric stability tests ----

    #[test]
    fn solve_well_conditioned_diagonal() {
        // Pure diagonal: condition number = max/min diagonal.
        let n = 100;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let off_diag: Vec<(usize, usize, f64)> = vec![];
        let rhs: Vec<f64> = diag.iter().map(|d| d * 2.0).collect(); // x should be all 2s
        let x = solve_system(&diag, &off_diag, &rhs);
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                (xi - 2.0).abs() < 1e-10,
                "x[{}] = {} (expected 2.0)",
                i,
                xi,
            );
        }
    }

    #[test]
    fn solve_high_anchor_weight() {
        // Kirchhoff-like: diag[0] = 1e10 (anchor), rest normal.
        // This is the actual pattern used in the placer.
        let (mut diag, off_diag) = make_grid_laplacian(8, 8);
        diag[0] = 1e10; // override to exact anchor
        let n = 64;
        let rhs: Vec<f64> = (0..n)
            .map(|i| if i == 0 { 0.0 } else { (i as f64).sin() })
            .collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-4, "high anchor weight");
        // Anchored node should be near zero pressure.
        assert!(x[0].abs() < 1e-4, "anchored node x[0] = {} (should be ~0)", x[0]);
    }

    #[test]
    fn solve_varying_conductances() {
        // Non-uniform conductances like the real Kirchhoff system.
        let n = 25; // 5x5 grid
        let w = 5;
        let h = 5;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    // Conductance varies: 0.1 to 10.0
                    let g = 0.1 + 9.9 * ((i + j) as f64 / (2 * n) as f64);
                    diag[i] += g;
                    diag[j] += g;
                    off_diag.push((i, j, -g));
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let g = 0.5 + 5.0 * ((i * j) as f64 / (n * n) as f64);
                    diag[i] += g;
                    diag[j] += g;
                    off_diag.push((i, j, -g));
                }
            }
        }
        diag[0] += 1e8;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let x = solve_system(&diag, &off_diag, &rhs);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-6, "varying conductances");
    }

    // ---- Multiple RHS with same symbolic structure ----

    #[test]
    fn solve_multiple_rhs_reuse_symbolic() {
        let (diag, off_diag) = make_grid_laplacian(8, 8);
        let n = 64;
        let (csc, _) = from_diag_and_upper_coo(n, &diag, &off_diag);
        let sym = symbolic_cholesky(&csc);
        let num = numeric_ldlt(&sym, &csc);

        // Solve 10 different RHS vectors with the same factorization.
        for k in 0..10 {
            let rhs: Vec<f64> = (0..n)
                .map(|i| ((i + k * 7) as f64 * 0.3).sin())
                .collect();
            let mut x = vec![0.0; n];
            num.solve(&rhs, &mut x);
            check_residual(&diag, &off_diag, &x, &rhs, 1e-6, &format!("multi-rhs #{}", k));
        }
    }

    // ---- Refactorization with different values ----

    #[test]
    fn refactorize_different_values() {
        let (diag1, off_diag1) = make_grid_laplacian(8, 8);
        let n = 64;
        let (csc1, _) = from_diag_and_upper_coo(n, &diag1, &off_diag1);
        let sym = symbolic_cholesky(&csc1);

        // First factorization + solve.
        let num1 = numeric_ldlt(&sym, &csc1);
        let rhs = vec![1.0; n];
        let mut x1 = vec![0.0; n];
        num1.solve(&rhs, &mut x1);
        check_residual(&diag1, &off_diag1, &x1, &rhs, 1e-6, "refactor solve 1");

        // Modify diagonal (simulates conductance change) and refactorize.
        let mut diag2 = diag1.clone();
        for i in 0..n {
            diag2[i] += 0.5 * (i as f64 + 1.0);
        }
        let (csc2, _) = from_diag_and_upper_coo(n, &diag2, &off_diag1);
        let num2 = numeric_ldlt(&sym, &csc2);
        let mut x2 = vec![0.0; n];
        num2.solve(&rhs, &mut x2);
        check_residual(&diag2, &off_diag1, &x2, &rhs, 1e-6, "refactor solve 2");

        // Solutions must differ.
        let diff: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "solutions should differ after value change");
    }
}

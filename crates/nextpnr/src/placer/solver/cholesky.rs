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

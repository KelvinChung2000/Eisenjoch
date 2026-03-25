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
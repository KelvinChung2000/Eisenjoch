//! Compressed Sparse Column (CSC) matrix format for SPD systems.
//!
//! Stores the lower triangle of a symmetric matrix in CSC layout.
//! Contiguous arrays are GPU-friendly (single buffer upload for future Vulkan backend).

/// CSC sparse matrix storing the lower triangle of a symmetric positive-definite matrix.
///
/// For column j, the nonzero row indices are `row_indices[col_ptrs[j]..col_ptrs[j+1]]`
/// and corresponding values are `values[col_ptrs[j]..col_ptrs[j+1]]`.
/// Row indices within each column are sorted in ascending order.
/// The diagonal entry (j, j) is always the first entry in column j.
#[derive(Clone, Debug)]
pub struct CscMatrix {
    /// Matrix dimension (n×n).
    pub n: usize,
    /// Column pointers, length n+1. `col_ptrs[j]..col_ptrs[j+1]` indexes into
    /// `row_indices` and `values` for column j.
    pub col_ptrs: Vec<usize>,
    /// Row indices for each nonzero, length nnz. Sorted ascending within each column.
    pub row_indices: Vec<usize>,
    /// Numeric values for each nonzero, length nnz.
    pub values: Vec<f64>,
}

/// Maps a pipe (or off-diagonal entry) to positions in the CSC values array.
/// Used for O(1) numeric updates without rebuilding structure.
#[derive(Clone, Debug)]
pub struct CscPositionMap {
    /// For each diagonal element i: position in `values` where diag[i] lives.
    pub diag_positions: Vec<usize>,
    /// For each off-diagonal entry (lo, hi): (position of (hi,lo) in lower triangle,
    /// index into diag_positions for lo, index into diag_positions for hi).
    /// The diagonal contributions are accumulated separately.
    pub offdiag_positions: Vec<usize>,
}

impl CscMatrix {
    /// Number of structural nonzeros.
    pub fn nnz(&self) -> usize {
        self.col_ptrs[self.n]
    }

    /// Create an empty CSC matrix of dimension n with preallocated structure.
    pub fn with_structure(n: usize, col_ptrs: Vec<usize>, row_indices: Vec<usize>) -> Self {
        let nnz = col_ptrs[n];
        Self {
            n,
            col_ptrs,
            row_indices,
            values: vec![0.0; nnz],
        }
    }

    /// Zero all numeric values (keeps structure).
    pub fn zero_values(&mut self) {
        self.values.fill(0.0);
    }

    /// Apply a symmetric permutation: B = P * A * P^T.
    ///
    /// `perm[new] = old` maps new indices to old indices.
    /// `inv_perm[old] = new` maps old indices to new indices.
    ///
    /// Returns a new CscMatrix with the permuted structure and values.
    pub fn permute(&self, perm: &[usize], inv_perm: &[usize]) -> CscMatrix {
        let n = self.n;
        debug_assert_eq!(perm.len(), n);
        debug_assert_eq!(inv_perm.len(), n);

        // Count entries per new column.
        let mut col_counts = vec![0usize; n];
        for new_col in 0..n {
            let old_col = perm[new_col];
            for pos in self.col_ptrs[old_col]..self.col_ptrs[old_col + 1] {
                let old_row = self.row_indices[pos];
                let new_row = inv_perm[old_row];
                // In lower triangle of permuted matrix: max(new_row, new_col) is row,
                // min is column.
                let col = new_row.min(new_col);
                col_counts[col] += 1;
            }
        }

        let mut new_col_ptrs = vec![0usize; n + 1];
        for j in 0..n {
            new_col_ptrs[j + 1] = new_col_ptrs[j] + col_counts[j];
        }
        let nnz = new_col_ptrs[n];
        let mut new_row_indices = vec![0usize; nnz];
        let mut new_values = vec![0.0f64; nnz];
        let mut write_pos = new_col_ptrs.clone();

        for new_col in 0..n {
            let old_col = perm[new_col];
            for pos in self.col_ptrs[old_col]..self.col_ptrs[old_col + 1] {
                let old_row = self.row_indices[pos];
                let new_row = inv_perm[old_row];
                let val = self.values[pos];

                let (r, c) = if new_row >= new_col {
                    (new_row, new_col)
                } else {
                    (new_col, new_row)
                };
                let wp = write_pos[c];
                new_row_indices[wp] = r;
                new_values[wp] = val;
                write_pos[c] += 1;
            }
        }

        // Sort each column by row index.
        for j in 0..n {
            let start = new_col_ptrs[j];
            let end = new_col_ptrs[j + 1];
            if end - start <= 1 {
                continue;
            }
            // Simple insertion sort (columns are typically short).
            for i in (start + 1)..end {
                let key_row = new_row_indices[i];
                let key_val = new_values[i];
                let mut k = i;
                while k > start && new_row_indices[k - 1] > key_row {
                    new_row_indices[k] = new_row_indices[k - 1];
                    new_values[k] = new_values[k - 1];
                    k -= 1;
                }
                new_row_indices[k] = key_row;
                new_values[k] = key_val;
            }
        }

        CscMatrix {
            n,
            col_ptrs: new_col_ptrs,
            row_indices: new_row_indices,
            values: new_values,
        }
    }
}

/// Convert from the existing diagonal + upper-triangle COO format to lower-triangle CSC.
///
/// The existing format stores:
/// - `diag[i]`: diagonal element A[i,i]
/// - `off_diag`: upper-triangle entries (lo, hi, weight) where lo < hi, representing A[lo,hi] = weight
///
/// For the lower triangle: diagonal stays at (i,i), each upper entry (lo,hi) maps to (hi,lo).
/// Each (lo, hi) pair must be unique; duplicate pairs get separate positions.
///
/// Returns the CSC matrix and a position map for fast numeric updates.
pub fn from_diag_and_upper_coo(
    n: usize,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
) -> (CscMatrix, CscPositionMap) {
    debug_assert_eq!(diag.len(), n);

    // Count entries per column in lower triangle.
    // Each diagonal entry contributes to its column.
    // Each off-diag (lo, hi, w) contributes to column lo (row hi, since hi > lo).
    let mut col_counts = vec![0usize; n];
    for i in 0..n {
        col_counts[i] += 1; // diagonal
    }
    for &(lo, _hi, _) in off_diag {
        col_counts[lo] += 1; // lower triangle: row=hi, col=lo
    }

    // Build column pointers.
    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
    }

    let nnz = col_ptrs[n];
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];

    // Position map for fast numeric updates.
    let mut diag_positions = vec![0usize; n];
    let mut offdiag_positions = vec![0usize; off_diag.len()];

    // Write position tracker per column.
    let mut write_pos: Vec<usize> = col_ptrs[..n].to_vec();

    // Fill diagonal entries first (they go at position (j, j) in column j).
    for i in 0..n {
        let pos = write_pos[i];
        row_indices[pos] = i;
        values[pos] = diag[i];
        diag_positions[i] = pos;
        write_pos[i] += 1;
    }

    // Fill off-diagonal entries: (lo, hi, w) -> lower triangle (hi, lo).
    for (idx, &(lo, hi, w)) in off_diag.iter().enumerate() {
        debug_assert!(lo < hi, "off_diag entries must have lo < hi");
        let pos = write_pos[lo];
        row_indices[pos] = hi;
        values[pos] = w;
        offdiag_positions[idx] = pos;
        write_pos[lo] += 1;
    }

    // Sort each column by row index (diagonal is already first since j <= all off-diag rows).
    // But we need to sort the off-diagonal part within each column.
    for j in 0..n {
        let start = col_ptrs[j] + 1; // skip diagonal
        let end = col_ptrs[j + 1];
        if end - start <= 1 {
            continue;
        }
        // Insertion sort on (row_indices, values) pairs.
        for i in (start + 1)..end {
            let key_row = row_indices[i];
            let key_val = values[i];
            let mut k = i;
            while k > start && row_indices[k - 1] > key_row {
                row_indices[k] = row_indices[k - 1];
                values[k] = values[k - 1];
                k -= 1;
            }
            row_indices[k] = key_row;
            values[k] = key_val;
        }
    }

    // After sorting, update offdiag_positions to reflect sorted positions.
    // We need to find where each (lo, hi) entry ended up.
    // Rebuild by scanning columns.
    let mut offdiag_map: Vec<(usize, usize, usize)> = off_diag
        .iter()
        .enumerate()
        .map(|(idx, &(lo, hi, _))| (lo, hi, idx))
        .collect();
    offdiag_map.sort_unstable_by_key(|&(lo, hi, _)| (lo, hi));

    let mut map_cursor = 0;
    for j in 0..n {
        let start = col_ptrs[j] + 1; // skip diagonal
        let end = col_ptrs[j + 1];
        for pos in start..end {
            let row = row_indices[pos];
            // Find matching off_diag entry.
            while map_cursor < offdiag_map.len() {
                let (lo, hi, idx) = offdiag_map[map_cursor];
                if lo == j && hi == row {
                    offdiag_positions[idx] = pos;
                    map_cursor += 1;
                    break;
                }
                map_cursor += 1;
            }
        }
    }

    let csc = CscMatrix {
        n,
        col_ptrs,
        row_indices,
        values,
    };

    let pos_map = CscPositionMap {
        diag_positions,
        offdiag_positions,
    };

    (csc, pos_map)
}

/// Fast numeric update: fill CSC values from new diagonal and off-diagonal data
/// using a precomputed position map. O(n + nnz_offdiag).
///
/// The sparsity structure must match what was used to create the position map.
pub fn update_values(
    csc: &mut CscMatrix,
    pos_map: &CscPositionMap,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
) {
    // Zero all values first.
    csc.values.fill(0.0);

    // Fill diagonal.
    for (i, &d) in diag.iter().enumerate() {
        csc.values[pos_map.diag_positions[i]] = d;
    }

    // Fill off-diagonal.
    for (idx, &(_lo, _hi, w)) in off_diag.iter().enumerate() {
        csc.values[pos_map.offdiag_positions[idx]] = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_from_3x3_laplacian() {
        // 3-node path graph Laplacian:
        // A = [ 1 -1  0]
        //     [-1  2 -1]
        //     [ 0 -1  1]
        let n = 3;
        let diag = vec![1.0, 2.0, 1.0];
        let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];

        let (csc, pos_map) = from_diag_and_upper_coo(n, &diag, &off_diag);

        assert_eq!(csc.n, 3);
        // Column 0: diag (0,0)=1, lower (1,0)=-1
        // Column 1: diag (1,1)=2, lower (2,1)=-1
        // Column 2: diag (2,2)=1
        assert_eq!(csc.col_ptrs, vec![0, 2, 4, 5]);
        assert_eq!(csc.row_indices, vec![0, 1, 1, 2, 2]);
        assert_eq!(csc.values, vec![1.0, -1.0, 2.0, -1.0, 1.0]);

        // Verify position map works for updates.
        assert_eq!(pos_map.diag_positions.len(), 3);
        assert_eq!(pos_map.offdiag_positions.len(), 2);
    }

    #[test]
    fn csc_update_values() {
        let n = 3;
        let diag = vec![1.0, 2.0, 1.0];
        let off_diag = vec![(0, 1, -1.0), (1, 2, -1.0)];
        let (mut csc, pos_map) = from_diag_and_upper_coo(n, &diag, &off_diag);

        // Update with new values.
        let new_diag = vec![3.0, 5.0, 3.0];
        let new_off_diag = vec![(0, 1, -2.0), (1, 2, -3.0)];
        update_values(&mut csc, &pos_map, &new_diag, &new_off_diag);

        assert_eq!(csc.values, vec![3.0, -2.0, 5.0, -3.0, 3.0]);
    }

    #[test]
    fn csc_4node_complete() {
        // 4-node with edges 0-1, 0-2, 0-3, 1-2, 1-3, 2-3
        let n = 4;
        let diag = vec![3.0, 3.0, 3.0, 3.0];
        let off_diag = vec![
            (0, 1, -1.0),
            (0, 2, -1.0),
            (0, 3, -1.0),
            (1, 2, -1.0),
            (1, 3, -1.0),
            (2, 3, -1.0),
        ];

        let (csc, _) = from_diag_and_upper_coo(n, &diag, &off_diag);

        assert_eq!(csc.n, 4);
        // Col 0: (0,0), (1,0), (2,0), (3,0) = 4 entries
        // Col 1: (1,1), (2,1), (3,1) = 3 entries
        // Col 2: (2,2), (3,2) = 2 entries
        // Col 3: (3,3) = 1 entry
        assert_eq!(csc.col_ptrs, vec![0, 4, 7, 9, 10]);
        assert_eq!(csc.nnz(), 10);
    }
}

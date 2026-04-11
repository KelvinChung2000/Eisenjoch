//! Incomplete Cholesky (IC(0)) preconditioner for SPD Kirchhoff Laplacians.

use std::collections::HashMap;

use dyn_stack::{MemStack, StackReq};

/// IC(0) / MIC(0) preconditioner storing the lower-triangular factor in CSC format.
///
/// When `mic_alpha > 0`, uses Modified Incomplete Cholesky (Gustafsson 1978):
/// dropped fill-in is compensated on the diagonal, preserving the row-sum
/// property `L*L^T * 1 ≈ A * 1`. This is guaranteed SPD for M-matrices
/// regardless of conductance variation.
pub struct Ic0Preconditioner {
    n: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<f64>,
    row_cols: Vec<Vec<usize>>,
    row_positions: Vec<Vec<usize>>,
    lower_pos: HashMap<(usize, usize), usize>,
    /// MIC(0) relaxation parameter. 0.0 = standard IC(0), 0.95 = recommended MIC(0).
    mic_alpha: f64,
}

impl Ic0Preconditioner {
    /// Build the IC(0) factorization from diagonal + upper-triangle entries.
    ///
    /// `mic_alpha`: MIC(0) relaxation. 0.0 = standard IC(0), 0.95 = recommended MIC(0).
    pub fn new(diag: &[f64], off_diag: &[(usize, usize, f64)], mic_alpha: f64) -> Self {
        let n = diag.len();
        let mut col_rows = vec![Vec::new(); n];
        for (col, rows) in col_rows.iter_mut().enumerate() {
            rows.push(col);
        }
        for &(lo, hi, _) in off_diag {
            debug_assert!(lo < hi);
            col_rows[lo].push(hi);
        }
        for rows in &mut col_rows {
            rows.sort_unstable();
            rows.dedup();
        }

        let mut col_ptrs = Vec::with_capacity(n + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for rows in &col_rows {
            row_indices.extend_from_slice(rows);
            col_ptrs.push(row_indices.len());
        }

        let mut lower_pos = HashMap::with_capacity(row_indices.len().saturating_sub(n));
        let mut row_cols = vec![Vec::new(); n];
        let mut row_positions = vec![Vec::new(); n];
        for col in 0..n {
            for pos in col_ptrs[col] + 1..col_ptrs[col + 1] {
                let row = row_indices[pos];
                lower_pos.insert((col, row), pos);
                row_cols[row].push(col);
                row_positions[row].push(pos);
            }
        }

        let nnz = row_indices.len();
        let mut this = Self {
            n,
            col_ptrs,
            row_indices,
            values: vec![0.0; nnz],
            row_cols,
            row_positions,
            lower_pos,
            mic_alpha,
        };
        this.update(diag, off_diag);
        this
    }

    /// Refresh the numeric factorization for the same sparsity pattern.
    pub fn update(&mut self, diag: &[f64], off_diag: &[(usize, usize, f64)]) {
        assert_eq!(diag.len(), self.n, "IC(0) update size mismatch");

        let mut base = vec![0.0; self.values.len()];
        for i in 0..self.n {
            base[self.col_ptrs[i]] = diag[i];
        }
        for &(lo, hi, value) in off_diag {
            let pos = *self
                .lower_pos
                .get(&(lo, hi))
                .expect("IC(0) update requires identical sparsity pattern");
            base[pos] += value;
        }

        let diag_mean = if self.n == 0 {
            0.0
        } else {
            diag.iter().sum::<f64>() / self.n as f64
        };
        let base_shift = 1e-10 * diag_mean.abs().max(1.0);

        let mut shifted = base.clone();
        let mut factorized = false;
        for attempt in 0..8 {
            shifted.copy_from_slice(&base);
            if attempt > 0 {
                let shift = base_shift * 10f64.powi((attempt - 1) as i32);
                for i in 0..self.n {
                    shifted[self.col_ptrs[i]] += shift;
                }
            }
            if self.factorize_in_place(&mut shifted) {
                factorized = true;
                break;
            }
        }

        assert!(
            factorized,
            "IC(0) factorization failed after diagonal shifting"
        );
        self.values.copy_from_slice(&shifted);
    }

    fn factorize_in_place(&self, values: &mut [f64]) -> bool {
        let alpha = self.mic_alpha;

        for k in 0..self.n {
            let diag_pos = self.col_ptrs[k];
            let mut diag_correction = 0.0;
            for &pos in &self.row_positions[k] {
                let lij = values[pos];
                diag_correction += lij * lij;
            }

            let pivot_sq = values[diag_pos] - diag_correction;
            if !pivot_sq.is_finite() || pivot_sq <= 0.0 {
                return false;
            }

            let lkk = pivot_sq.sqrt();
            if !lkk.is_finite() || lkk <= 0.0 {
                return false;
            }
            values[diag_pos] = lkk;

            let col_start = diag_pos + 1;
            let col_end = self.col_ptrs[k + 1];

            // Compute L[i,k] for all i > k in the sparsity pattern.
            for pos in col_start..col_end {
                let row = self.row_indices[pos];
                let sum = self.intersection_dot(row, k, values);
                let lik = (values[pos] - sum) / lkk;
                if !lik.is_finite() {
                    return false;
                }
                values[pos] = lik;
            }

            // MIC(0) compensation (Gustafsson 1978): for each pair (i,j) both > k
            // in column k's pattern, if (i,j) is NOT in the sparsity pattern,
            // add alpha * L[i,k] * L[j,k] to diag(i). This preserves row sums
            // and keeps the preconditioner SPD for M-matrices.
            if alpha > 0.0 {
                for pos_i in col_start..col_end {
                    let i = self.row_indices[pos_i];
                    let lik = values[pos_i];
                    let mut dropped_sum = 0.0;
                    for pos_j in col_start..col_end {
                        let j = self.row_indices[pos_j];
                        if i == j {
                            continue;
                        }
                        let lo = i.min(j);
                        let hi = i.max(j);
                        if !self.lower_pos.contains_key(&(lo, hi)) {
                            dropped_sum += values[pos_j];
                        }
                    }
                    if dropped_sum != 0.0 {
                        values[self.col_ptrs[i]] += alpha * lik * dropped_sum;
                    }
                }
            }
        }
        true
    }

    fn intersection_dot(&self, row_i: usize, row_k: usize, values: &[f64]) -> f64 {
        let cols_i = &self.row_cols[row_i];
        let cols_k = &self.row_cols[row_k];
        let pos_i = &self.row_positions[row_i];
        let pos_k = &self.row_positions[row_k];

        let mut a = 0usize;
        let mut b = 0usize;
        let mut sum = 0.0;

        while a < cols_i.len() && b < cols_k.len() {
            let ci = cols_i[a];
            let ck = cols_k[b];
            if ci >= row_k || ck >= row_k {
                break;
            }
            if ci == ck {
                sum += values[pos_i[a]] * values[pos_k[b]];
                a += 1;
                b += 1;
            } else if ci < ck {
                a += 1;
            } else {
                b += 1;
            }
        }

        sum
    }

    fn solve_in_place(&self, x: &mut [f64]) {
        for col in 0..self.n {
            let diag_pos = self.col_ptrs[col];
            let y = x[col] / self.values[diag_pos];
            x[col] = y;
            for pos in diag_pos + 1..self.col_ptrs[col + 1] {
                let row = self.row_indices[pos];
                x[row] -= self.values[pos] * y;
            }
        }

        for col in (0..self.n).rev() {
            let diag_pos = self.col_ptrs[col];
            let mut z = x[col];
            for pos in diag_pos + 1..self.col_ptrs[col + 1] {
                let row = self.row_indices[pos];
                z -= self.values[pos] * x[row];
            }
            x[col] = z / self.values[diag_pos];
        }
    }
}

impl std::fmt::Debug for Ic0Preconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ic0Preconditioner(n={}, nnz_l={})",
            self.n,
            self.values.len()
        )
    }
}

// IC(0) is immutable after factorization — solve_in_place only reads self.values.
// Apply methods allocate per-call scratch, so no shared mutable state.
unsafe impl Sync for Ic0Preconditioner {}

impl faer::matrix_free::LinOp<f64> for Ic0Preconditioner {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let mut scratch = vec![0.0; self.n];
        for col in 0..rhs.ncols() {
            for i in 0..self.n {
                scratch[i] = rhs[(i, col)];
            }
            self.solve_in_place(&mut scratch);
            for i in 0..self.n {
                out[(i, col)] = scratch[i];
            }
        }
    }

    fn conj_apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

impl faer::matrix_free::Precond<f64> for Ic0Preconditioner {
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(
        &self,
        mut rhs: faer::MatMut<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        for col in 0..rhs.ncols() {
            let rhs_col = rhs
                .as_mut()
                .col_mut(col)
                .try_as_col_major_mut()
                .unwrap()
                .as_slice_mut();
            self.solve_in_place(rhs_col);
        }
    }

    fn conj_apply_in_place(
        &self,
        rhs: faer::MatMut<'_, f64>,
        par: faer::Par,
        stack: &mut MemStack,
    ) {
        self.apply_in_place(rhs, par, stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::JacobiPreconditioner;
    use crate::solver::solve_cg;
    use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp};
    use faer::matrix_free::LinOp;

    #[test]
    fn ic0_1d_chain_converges_faster_than_jacobi() {
        let n = 100;
        let mut mat = SparseMatrix::new(n);
        for i in 0..n - 1 {
            mat.add_connection(i, i + 1, 1.0);
        }
        mat.add_diagonal(0, 1e-3);
        mat.add_diagonal(n - 1, 1e-3);

        let op_j = SparseMatrixOp::from_matrix(&mut mat);
        let jacobi = JacobiPreconditioner::new(mat.diag());
        let mut x_j = vec![0.0; n];

        let op_i = SparseMatrixOp::from_matrix(&mut mat);
        let ic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.0);
        let mut x_i = vec![0.0; n];

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_j_mat = faer::MatMut::from_column_major_slice_mut(&mut x_j, n, 1);
        let result_j = solve_cg(&op_j, &jacobi, rhs_mat, x_j_mat, 1e-8, 1000);

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_i_mat = faer::MatMut::from_column_major_slice_mut(&mut x_i, n, 1);
        let result_i = solve_cg(&op_i, &ic0, rhs_mat, x_i_mat, 1e-8, 1000);

        assert!(
            result_j.converged,
            "Jacobi CG failed: {}",
            result_j.residual
        );
        assert!(result_i.converged, "IC(0) CG failed: {}", result_i.residual);
        assert!(
            result_i.iterations < result_j.iterations,
            "IC(0) should be faster on the chain: ic0={} jacobi={}",
            result_i.iterations,
            result_j.iterations
        );
    }

    #[test]
    fn ic0_2d_grid_is_spd_and_converges() {
        let w = 20;
        let h = 20;
        let n = w * h;
        let mut mat = SparseMatrix::new(n);

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    let cond = if (x + 2 * y) % 5 == 0 { 10.0 } else { 0.4 };
                    mat.add_connection(i, j, cond);
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let cond = if (3 * x + y) % 7 == 0 { 7.5 } else { 1.2 };
                    mat.add_connection(i, j, cond);
                }
                mat.add_diagonal(i, 1e-3);
            }
        }

        let ic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.0);

        let mut rhs_two = vec![0.0; n * 2];
        for i in 0..n {
            rhs_two[i] = if i % 2 == 0 { 1.0 } else { -0.5 };
            rhs_two[n + i] = ((i % w) as f64 + 1.0) / (w as f64);
        }
        let rhs_ref = faer::MatRef::from_column_major_slice(&rhs_two, n, 2);
        let mut out_two = vec![0.0; n * 2];
        let out_mut = faer::MatMut::from_column_major_slice_mut(&mut out_two, n, 2);
        let mut buf = dyn_stack::MemBuffer::new(StackReq::EMPTY);
        let mut stack = MemStack::new(&mut buf);
        ic0.apply(out_mut, rhs_ref, faer::Par::Seq, &mut stack);

        let dot0: f64 = rhs_two[..n]
            .iter()
            .zip(out_two[..n].iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot1: f64 = rhs_two[n..]
            .iter()
            .zip(out_two[n..].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot0 > 0.0,
            "first RHS did not produce a positive quadratic form"
        );
        assert!(
            dot1 > 0.0,
            "second RHS did not produce a positive quadratic form"
        );
        assert!(out_two.iter().all(|v| v.is_finite()));

        let op_j = SparseMatrixOp::from_matrix(&mut mat);
        let jacobi = JacobiPreconditioner::new(mat.diag());
        let mut x_j = vec![0.0; n];

        let op_i = SparseMatrixOp::from_matrix(&mut mat);
        let mut x_i = vec![0.0; n];

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n / 3] = -0.5;
        rhs[n - 1] = -0.5;

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_j_mat = faer::MatMut::from_column_major_slice_mut(&mut x_j, n, 1);
        let result_j = solve_cg(&op_j, &jacobi, rhs_mat, x_j_mat, 1e-6, 3000);

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_i_mat = faer::MatMut::from_column_major_slice_mut(&mut x_i, n, 1);
        let result_i = solve_cg(&op_i, &ic0, rhs_mat, x_i_mat, 1e-6, 1000);

        assert!(result_i.converged, "IC(0) CG failed: {}", result_i.residual);
        assert!(
            result_i.iterations < result_j.iterations || result_i.iterations < 100,
            "IC(0) should improve convergence on the 2D grid: ic0={} jacobi={}",
            result_i.iterations,
            result_j.iterations
        );
    }

    #[test]
    fn mic0_high_contrast_grid_is_spd() {
        // 2D grid with 100x conductance variation — this is what breaks standard IC(0).
        // MIC(0) with alpha=0.95 should produce SPD preconditioner.
        let w = 10;
        let h = 10;
        let n = w * h;
        let mut mat = SparseMatrix::new(n);

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    // 100x variation: conductances range from 0.1 to 10.0
                    let cond = if (x + y) % 3 == 0 { 10.0 } else { 0.1 };
                    mat.add_connection(i, j, cond);
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let cond = if (x * 3 + y) % 5 == 0 { 8.0 } else { 0.15 };
                    mat.add_connection(i, j, cond);
                }
                mat.add_diagonal(i, 1e-2);
            }
        }

        let mic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.95);

        // Verify SPD: r^T M^{-1} r > 0 for random RHS vectors.
        let mut rhs_data = vec![0.0; n * 4];
        for col in 0..4 {
            for i in 0..n {
                rhs_data[col * n + i] = ((i * 7 + col * 13) % 37) as f64 / 37.0 - 0.5;
            }
        }
        let rhs_ref = faer::MatRef::from_column_major_slice(&rhs_data, n, 4);
        let mut out_data = vec![0.0; n * 4];
        let out_mut = faer::MatMut::from_column_major_slice_mut(&mut out_data, n, 4);
        let mut buf = dyn_stack::MemBuffer::new(StackReq::EMPTY);
        let mut stack = MemStack::new(&mut buf);
        mic0.apply(out_mut, rhs_ref, faer::Par::Seq, &mut stack);

        for col in 0..4 {
            let dot: f64 = (0..n)
                .map(|i| rhs_data[col * n + i] * out_data[col * n + i])
                .sum();
            assert!(
                dot > 0.0,
                "MIC(0) col {} produced non-positive r^T z = {:.6e}",
                col,
                dot
            );
        }

        // Verify CG convergence.
        let op = SparseMatrixOp::from_matrix(&mut mat);
        let mut x = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &mic0, rhs_mat, x_mat, 1e-6, 500);
        assert!(
            result.converged,
            "MIC(0) CG failed: residual={:.6e}",
            result.residual
        );
    }

    #[test]
    fn mic0_near_singular_sparse_rhs() {
        // Mimics real Kirchhoff system: 8x11 grid, variable conductances,
        // very small epsilon (near-singular), and SPARSE RHS vectors (per-net).
        let w = 8;
        let h = 11;
        let n = w * h;
        let mut mat = SparseMatrix::new(n);

        // Conductances from real system range: off=[-9.261, -2.196], so
        // conductance range is about 2.2 to 9.3 (4.2x variation).
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    let cond = 2.2 + 7.1 * ((x * 3 + y * 7) % 11) as f64 / 10.0;
                    mat.add_connection(i, j, cond);
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let cond = 2.2 + 7.1 * ((x * 5 + y * 3) % 13) as f64 / 12.0;
                    mat.add_connection(i, j, cond);
                }
            }
        }
        // Small epsilon like real system (diag_shift_scale=1e-2 * mean_diag)
        let diag_mean = mat.diag().iter().sum::<f64>() / n as f64;
        let epsilon = 1e-2 * diag_mean;
        for i in 0..n {
            mat.add_diagonal(i, epsilon);
        }

        let mic0 = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.95);

        // Test with SPARSE RHS vectors (like per-net injection):
        // only 2-4 nonzero entries per column.
        let ncols = 20;
        let mut rhs_data = vec![0.0; n * ncols];
        for col in 0..ncols {
            // Inject at 2-3 nodes with signed weights (sum ≈ 0, like net demand).
            let src = (col * 7 + 3) % n;
            let snk1 = (col * 13 + 17) % n;
            let snk2 = (col * 11 + 5) % n;
            rhs_data[col * n + src] = 1.0;
            rhs_data[col * n + snk1] = -0.6;
            if snk2 != src && snk2 != snk1 {
                rhs_data[col * n + snk2] = -0.4;
            } else {
                rhs_data[col * n + snk1] = -1.0;
            }
        }

        let rhs_ref = faer::MatRef::from_column_major_slice(&rhs_data, n, ncols);
        let mut out_data = vec![0.0; n * ncols];
        let out_mut = faer::MatMut::from_column_major_slice_mut(&mut out_data, n, ncols);
        let mut buf = dyn_stack::MemBuffer::new(StackReq::EMPTY);
        let mut stack = MemStack::new(&mut buf);
        mic0.apply(out_mut, rhs_ref, faer::Par::Seq, &mut stack);

        let mut neg_count = 0;
        for col in 0..ncols {
            let dot: f64 = (0..n)
                .map(|i| rhs_data[col * n + i] * out_data[col * n + i])
                .sum();
            if dot <= 0.0 {
                eprintln!(
                    "MIC(0) sparse RHS col {} produced r^T z = {:.6e} (rhs_norm={:.6e})",
                    col,
                    dot,
                    rhs_data[col * n..(col + 1) * n]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                );
                neg_count += 1;
            }
        }
        assert_eq!(
            neg_count, 0,
            "MIC(0) produced negative r^T z for {}/{} sparse RHS vectors",
            neg_count, ncols
        );
    }
}

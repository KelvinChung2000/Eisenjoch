//! DST-I spectral preconditioner for rectangular 2D grid Laplacians.

use dyn_stack::{MemStack, StackReq};
use rustdct::{DctPlanner, Dst1};
use std::cell::RefCell;
use std::sync::Arc;

/// Spectral preconditioner for a constant-coefficient 2D grid Laplacian.
///
/// All apply methods allocate workspace per-call for thread safety under
/// Rayon parallel CG solves.
pub struct SpectralPreconditioner {
    nx: usize,
    ny: usize,
    inv_eigenvalues: Vec<f64>,
    row_dst: Arc<dyn Dst1<f64>>,
    col_dst: Arc<dyn Dst1<f64>>,
    round_trip_scale: f64,
    row_scratch_len: usize,
    col_scratch_len: usize,
}

impl SpectralPreconditioner {
    /// Build the spectral inverse for an `nx x ny` grid with diagonal regularization `epsilon`.
    pub fn new(nx: usize, ny: usize, epsilon: f64) -> Self {
        assert!(nx > 0, "SpectralPreconditioner requires nx > 0");
        assert!(ny > 0, "SpectralPreconditioner requires ny > 0");

        let mut planner = DctPlanner::new();
        let row_dst = planner.plan_dst1(nx);
        let col_dst = planner.plan_dst1(ny);

        let mut inv_eigenvalues = vec![0.0; nx * ny];
        for j in 0..ny {
            let theta_y = std::f64::consts::PI * (j + 1) as f64 / (ny + 1) as f64;
            let cos_y = theta_y.cos();
            for i in 0..nx {
                let theta_x = std::f64::consts::PI * (i + 1) as f64 / (nx + 1) as f64;
                let lambda = 4.0 - 2.0 * theta_x.cos() - 2.0 * cos_y;
                inv_eigenvalues[j * nx + i] = 1.0 / (lambda + epsilon);
            }
        }

        let row_scratch_len = row_dst.get_scratch_len();
        let col_scratch_len = col_dst.get_scratch_len();
        let round_trip_scale = 4.0 / ((nx + 1) * (ny + 1)) as f64;

        Self {
            nx,
            ny,
            inv_eigenvalues,
            row_dst,
            col_dst,
            round_trip_scale,
            row_scratch_len,
            col_scratch_len,
        }
    }

    fn apply_dst_rows(&self, field: &mut [f64], row_buf: &mut [f64], row_scratch: &mut [f64]) {
        for row in field.chunks_exact_mut(self.nx) {
            row_buf.copy_from_slice(row);
            self.row_dst.process_dst1_with_scratch(row_buf, row_scratch);
            row.copy_from_slice(row_buf);
        }
    }

    fn apply_dst_cols(&self, field: &mut [f64], col_buf: &mut [f64], col_scratch: &mut [f64]) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                col_buf[y] = field[y * self.nx + x];
            }
            self.col_dst.process_dst1_with_scratch(col_buf, col_scratch);
            for y in 0..self.ny {
                field[y * self.nx + x] = col_buf[y];
            }
        }
    }

    /// Apply M^{-1} to a single slice.
    ///
    /// Uses thread-local DST plans and workspace buffers for thread safety
    /// under Rayon parallelism. rustdct DST objects have internal mutable state
    /// that causes data races when shared across threads via Arc.
    fn apply_to_slice(&self, rhs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.nx * self.ny);
        debug_assert_eq!(out.len(), self.nx * self.ny);

        // Thread-local cache for DST plans + workspace buffers.
        // Key: (nx, ny) → (row_dst, col_dst, row_buf, col_buf, row_scratch, col_scratch, field)
        thread_local! {
            static DST_CACHE: RefCell<Option<(
                usize, usize,
                Arc<dyn Dst1<f64>>, Arc<dyn Dst1<f64>>,
                Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>,
            )>> = const { RefCell::new(None) };
        }

        let nx = self.nx;
        let ny = self.ny;

        DST_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Rebuild cache if grid dimensions changed.
            let needs_rebuild = match &*cache {
                Some((cnx, cny, ..)) => *cnx != nx || *cny != ny,
                None => true,
            };
            if needs_rebuild {
                let mut planner = DctPlanner::new();
                let row_dst = planner.plan_dst1(nx);
                let col_dst = planner.plan_dst1(ny);
                let row_scratch_len = row_dst.get_scratch_len();
                let col_scratch_len = col_dst.get_scratch_len();
                *cache = Some((
                    nx,
                    ny,
                    row_dst,
                    col_dst,
                    vec![0.0; nx],
                    vec![0.0; ny],
                    vec![0.0; row_scratch_len],
                    vec![0.0; col_scratch_len],
                    vec![0.0; nx * ny],
                ));
            }

            let (_, _, row_dst, col_dst, row_buf, col_buf, row_scratch, col_scratch, field) =
                cache.as_mut().unwrap();

            field.copy_from_slice(rhs);

            // Forward DST2D
            for row in field.chunks_exact_mut(nx) {
                row_buf[..nx].copy_from_slice(row);
                row_dst.process_dst1_with_scratch(&mut row_buf[..nx], row_scratch);
                row.copy_from_slice(&row_buf[..nx]);
            }
            for x in 0..nx {
                for y in 0..ny {
                    col_buf[y] = field[y * nx + x];
                }
                col_dst.process_dst1_with_scratch(&mut col_buf[..ny], col_scratch);
                for y in 0..ny {
                    field[y * nx + x] = col_buf[y];
                }
            }
            // Multiply by inverse eigenvalues
            for (value, &inv_eig) in field.iter_mut().zip(self.inv_eigenvalues.iter()) {
                *value *= inv_eig;
            }
            // Inverse DST2D
            for x in 0..nx {
                for y in 0..ny {
                    col_buf[y] = field[y * nx + x];
                }
                col_dst.process_dst1_with_scratch(&mut col_buf[..ny], col_scratch);
                for y in 0..ny {
                    field[y * nx + x] = col_buf[y];
                }
            }
            for row in field.chunks_exact_mut(nx) {
                row_buf[..nx].copy_from_slice(row);
                row_dst.process_dst1_with_scratch(&mut row_buf[..nx], row_scratch);
                row.copy_from_slice(&row_buf[..nx]);
            }
            for value in field.iter_mut() {
                *value *= self.round_trip_scale;
            }

            out.copy_from_slice(field);
        });
    }

    fn dst_round_trip_slice(&self, values: &mut [f64]) {
        let mut field = values.to_vec();
        let mut row_buf = vec![0.0; self.nx];
        let mut col_buf = vec![0.0; self.ny];
        let mut row_scratch = vec![0.0; self.row_scratch_len];
        let mut col_scratch = vec![0.0; self.col_scratch_len];

        self.apply_dst_rows(&mut field, &mut row_buf, &mut row_scratch);
        self.apply_dst_cols(&mut field, &mut col_buf, &mut col_scratch);
        self.apply_dst_cols(&mut field, &mut col_buf, &mut col_scratch);
        self.apply_dst_rows(&mut field, &mut row_buf, &mut row_scratch);
        for value in field.iter_mut() {
            *value *= self.round_trip_scale;
        }
        values.copy_from_slice(&field);
    }
}

impl std::fmt::Debug for SpectralPreconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpectralPreconditioner(nx={}, ny={})", self.nx, self.ny)
    }
}

// All apply methods allocate workspace per-call, so no shared mutable state.
unsafe impl Sync for SpectralPreconditioner {}

impl faer::matrix_free::LinOp<f64> for SpectralPreconditioner {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.nx * self.ny
    }

    fn ncols(&self) -> usize {
        self.nx * self.ny
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.nx * self.ny;
        let mut rhs_buf = vec![0.0; n];
        let mut out_buf = vec![0.0; n];
        for col in 0..rhs.ncols() {
            for i in 0..n {
                rhs_buf[i] = rhs[(i, col)];
            }
            self.apply_to_slice(&rhs_buf, &mut out_buf);
            for i in 0..n {
                out[(i, col)] = out_buf[i];
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

impl faer::matrix_free::Precond<f64> for SpectralPreconditioner {
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(
        &self,
        mut rhs: faer::MatMut<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.nx * self.ny;
        let mut buf = vec![0.0; n];
        let mut out = vec![0.0; n];
        for col in 0..rhs.ncols() {
            for i in 0..n {
                buf[i] = rhs[(i, col)];
            }
            self.apply_to_slice(&buf, &mut out);
            for i in 0..n {
                rhs[(i, col)] = out[i];
            }
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
    use crate::solver::cg::solve_cg;
    use crate::solver::preconditioner::JacobiPreconditioner;
    use crate::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp, SparseMatrixOpRef};
    use faer::matrix_free::{LinOp, Precond};

    fn grid_index(nx: usize, x: usize, y: usize) -> usize {
        y * nx + x
    }

    fn build_constant_dirichlet_grid(nx: usize, ny: usize, epsilon: f64) -> SparseMatrix {
        let mut mat = SparseMatrix::new(nx * ny);
        for y in 0..ny {
            for x in 0..nx {
                let i = grid_index(nx, x, y);
                mat.set_diag(i, 4.0 + epsilon);
                if x + 1 < nx {
                    mat.add_entry(i, grid_index(nx, x + 1, y), -1.0);
                }
                if y + 1 < ny {
                    mat.add_entry(i, grid_index(nx, x, y + 1), -1.0);
                }
            }
        }
        mat
    }

    fn build_variable_grid(nx: usize, ny: usize, epsilon: f64) -> SparseMatrix {
        let mut mat = SparseMatrix::new(nx * ny);
        for y in 0..ny {
            for x in 0..nx {
                let i = grid_index(nx, x, y);
                if x + 1 < nx {
                    let j = grid_index(nx, x + 1, y);
                    let weight = 0.3 + ((3 * x + 5 * y) % 11) as f64 * 0.17;
                    mat.add_connection(i, j, weight);
                }
                if y + 1 < ny {
                    let j = grid_index(nx, x, y + 1);
                    let weight = 0.4 + ((7 * x + 2 * y) % 13) as f64 * 0.11;
                    mat.add_connection(i, j, weight);
                }
                mat.add_diagonal(i, epsilon);
            }
        }
        mat.ensure_csc();
        mat
    }

    #[test]
    fn spectral_exact_grid_converges_in_one_iteration() {
        let nx = 12;
        let ny = 10;
        let n = nx * ny;
        let epsilon = 1e-3;
        let mut mat = build_constant_dirichlet_grid(nx, ny, epsilon);
        let op = SparseMatrixOp::from_matrix(&mut mat);
        let spectral = SpectralPreconditioner::new(nx, ny, epsilon);

        let rhs: Vec<f64> = (0..n).map(|i| ((i % 9) as f64 - 4.0) * 0.25).collect();
        let mut x = vec![0.0; n];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = solve_cg(&op, &spectral, rhs_mat, x_mat, 1e-10, 10);

        assert!(
            result.converged,
            "spectral preconditioned CG should converge"
        );
        assert!(
            result.iterations <= 2,
            "exact spectral preconditioner should converge in ~1 iteration, got {}",
            result.iterations
        );
    }

    #[test]
    fn spectral_beats_jacobi_on_variable_grid() {
        let nx = 20;
        let ny = 18;
        let n = nx * ny;
        let epsilon = 1e-2;
        let mat = build_variable_grid(nx, ny, epsilon);

        let op_j = SparseMatrixOpRef::from_matrix(&mat);
        let jacobi = JacobiPreconditioner::new(mat.diag());
        let op_s = SparseMatrixOpRef::from_matrix(&mat);
        let spectral = SpectralPreconditioner::new(nx, ny, epsilon);

        let rhs: Vec<f64> = (0..n)
            .map(|i| (((i * 17 + 3) % 23) as f64 - 11.0) * 0.1)
            .collect();

        let mut x_j = vec![0.0; n];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_j_mat = faer::MatMut::from_column_major_slice_mut(&mut x_j, n, 1);
        let result_j = solve_cg(&op_j, &jacobi, rhs_mat, x_j_mat, 1e-8, 500);

        let mut x_s = vec![0.0; n];
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_s_mat = faer::MatMut::from_column_major_slice_mut(&mut x_s, n, 1);
        let result_s = solve_cg(&op_s, &spectral, rhs_mat, x_s_mat, 1e-8, 500);

        assert!(result_j.converged, "Jacobi CG should converge");
        assert!(result_s.converged, "spectral CG should converge");
        assert!(
            result_s.iterations < result_j.iterations,
            "spectral should reduce iterations: spectral={} jacobi={}",
            result_s.iterations,
            result_j.iterations
        );
    }

    #[test]
    fn dst_round_trip_matches_identity() {
        let nx = 7;
        let ny = 5;
        let n = nx * ny;
        let precond = SpectralPreconditioner::new(nx, ny, 0.25);

        let mut values: Vec<f64> = (0..n).map(|i| ((i * 5 + 1) as f64).sin()).collect();
        let original = values.clone();

        precond.dst_round_trip_slice(&mut values);

        for (actual, expected) in values.iter().zip(original.iter()) {
            assert!((actual - expected).abs() < 1e-9);
        }

        let mut rhs = vec![0.0; 2 * n];
        rhs[..n].copy_from_slice(&original);
        rhs[n..].copy_from_slice(&values);
        let rhs = faer::MatRef::from_column_major_slice(&rhs, n, 2);
        let mut out = vec![0.0; 2 * n];
        let out_mat = faer::MatMut::from_column_major_slice_mut(&mut out, n, 2);
        let mut buf = dyn_stack::MemBuffer::new(StackReq::EMPTY);
        let mut stack = MemStack::new(&mut buf);
        precond.apply(out_mat, rhs, faer::Par::Seq, &mut stack);

        let mut inplace = vec![0.0; 2 * n];
        inplace[..n].copy_from_slice(&original);
        inplace[n..].copy_from_slice(&values);
        let inplace_mat = faer::MatMut::from_column_major_slice_mut(&mut inplace, n, 2);
        precond.apply_in_place(inplace_mat, faer::Par::Seq, &mut stack);

        for (lhs, rhs) in out.iter().zip(inplace.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }
}

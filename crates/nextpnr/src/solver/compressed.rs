//! Randomized basis compression for multi-RHS linear systems.
//!
//! When the same SPD matrix L is used with many right-hand sides {d_k},
//! and the solutions {L^{-1} d_k} lie in a low-rank subspace, this module
//! compresses N solves into 2·m << N block solves plus cheap per-RHS reconstruction.
//!
//! # Algorithm (HMT randomized range finder, Halko-Martinsson-Tropp 2011)
//!
//! Goal: approximate the matrix A = L^{-1} D whose columns are the per-net
//! pressure solutions we want.
//!
//! 1. Draw a Gaussian projection Ω ∈ R^{n_rhs × m}
//! 2. Build sketch S = D Ω                  (n × m matmul; caller)
//! 3. Solve L Y = S         → Y = L^{-1} D Ω  (block CG with m RHS; caller)
//! 4. Orthonormalize Q = orth(Y) via thin QR  (range_basis — stored)
//! 5. Solve L Z = Q         → Z = L^{-1} Q    (block CG with m RHS; caller)
//! 6. Reconstruction (exact algebra for SPD L):
//!        p_k ≈ Q Q^T (L^{-1} d_k)
//!            = Q (L^{-1} Q)^T d_k
//!            = Q Z^T d_k
//!    Stored as: range_basis = Q, coeff_basis = Z.
//!
//! Two block CG calls per basis. The caller performs the solves so that the
//! same CG plumbing (preconditioner, scratch buffer, `faer::Par`) is reused.

use faer::Mat;

/// Result of a rank analysis on the demand/pressure space.
pub struct RankAnalysis {
    /// Singular values of the pressure matrix (descending).
    pub singular_values: Vec<f64>,
    /// Number of singular values needed to capture `threshold` fraction of energy.
    pub effective_rank: usize,
    /// Energy capture threshold used.
    pub threshold: f64,
    /// Total number of right-hand sides analyzed.
    pub n_rhs: usize,
}

impl RankAnalysis {
    /// Compute effective rank at a given energy threshold.
    pub fn rank_at_threshold(&self, threshold: f64) -> usize {
        if self.singular_values.is_empty() {
            return 0;
        }
        let total_energy: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total_energy < 1e-30 {
            return 0;
        }
        let mut cumulative = 0.0;
        for (i, &s) in self.singular_values.iter().enumerate() {
            cumulative += s * s;
            if cumulative / total_energy >= threshold {
                return i + 1;
            }
        }
        self.singular_values.len()
    }

    /// Print a summary of the rank analysis.
    pub fn summary(&self) -> String {
        let r90 = self.rank_at_threshold(0.90);
        let r95 = self.rank_at_threshold(0.95);
        let r99 = self.rank_at_threshold(0.99);
        let r999 = self.rank_at_threshold(0.999);
        format!(
            "RankAnalysis: n_rhs={}, effective_rank({}%)={}, ranks: 90%={}, 95%={}, 99%={}, 99.9%={}",
            self.n_rhs,
            (self.threshold * 100.0) as u32,
            self.effective_rank,
            r90, r95, r99, r999
        )
    }

    /// Return top-k singular values for logging.
    pub fn top_singular_values(&self, k: usize) -> &[f64] {
        &self.singular_values[..k.min(self.singular_values.len())]
    }
}

/// Analyze the effective rank of a set of solution vectors.
///
/// Given a matrix P where each column is a solution vector (e.g., pressure from L^{-1} * d_k),
/// compute the SVD and return the singular value spectrum.
///
/// `solutions` is a column-major slice of n_nodes x n_rhs.
pub fn analyze_rank(
    solutions: &[f64],
    n_nodes: usize,
    n_rhs: usize,
    threshold: f64,
) -> RankAnalysis {
    assert_eq!(solutions.len(), n_nodes * n_rhs);

    let mat = faer::MatRef::from_column_major_slice(solutions, n_nodes, n_rhs);
    let svd = mat.thin_svd().unwrap();

    let s_col = svd.S().column_vector();
    let min_dim = n_nodes.min(n_rhs);
    let mut singular_values = Vec::with_capacity(min_dim);
    for i in 0..min_dim {
        singular_values.push(s_col[i]);
    }

    let mut result = RankAnalysis {
        singular_values,
        effective_rank: 0,
        threshold,
        n_rhs,
    };
    result.effective_rank = result.rank_at_threshold(threshold);
    result
}

/// Build the HMT Gaussian sketch S = D * Ω from a column-major demand matrix.
///
/// Returns an `n_nodes × m` dense matrix that the caller must use as the RHS
/// of a block CG solve against L. The resulting `Y = L^{-1} S` is then passed
/// to [`CompressedBasis::from_hmt_range`] to build the range basis Q.
///
/// `rank` is capped at `n_rhs`. `seed` controls the Gaussian PRNG for
/// reproducibility.
pub fn hmt_sketch_matrix(
    demand_matrix: &[f64],
    n_nodes: usize,
    n_rhs: usize,
    rank: usize,
    seed: u64,
) -> Mat<f64> {
    assert_eq!(demand_matrix.len(), n_nodes * n_rhs);
    let m = rank.min(n_rhs).max(1);

    // Ω ~ Gaussian(n_rhs × m)
    let omega_data = random_gaussian_matrix(n_rhs, m, seed);
    let omega = faer::MatRef::from_column_major_slice(&omega_data, n_rhs, m);

    // S = D * Ω (n_nodes × m) via BLAS matmul
    let d_mat = faer::MatRef::from_column_major_slice(demand_matrix, n_nodes, n_rhs);
    let mut s = Mat::<f64>::zeros(n_nodes, m);
    faer::linalg::matmul::matmul(
        s.as_mut(),
        faer::Accum::Replace,
        d_mat,
        omega,
        1.0,
        faer::Par::rayon(0),
    );
    s
}

/// Compressed basis for multi-RHS solves via HMT randomized projection.
///
/// After construction the basis stores:
/// - `range_basis` = Q, an orthonormal basis for range(L^{-1} D) with shape n × m
/// - `coeff_basis` = Z = L^{-1} Q, with shape n × m
///
/// Reconstruction: `p_k ≈ Q (Z^T d_k)`, equivalent to
/// `P ≈ range_basis · (coeff_basis^T · D)` in batch form.
pub struct CompressedBasis {
    /// Orthonormal basis Q for range(L^{-1} D). Shape (n_nodes × rank).
    pub range_basis: Mat<f64>,
    /// Coefficient basis Z = L^{-1} Q. Shape (n_nodes × rank).
    pub coeff_basis: Mat<f64>,
    /// Number of basis vectors (effective rank after QR).
    pub rank: usize,
    /// Number of nodes.
    pub n_nodes: usize,
}

impl CompressedBasis {
    /// Build the range basis from the solved HMT sketch `y = L^{-1} (D Ω)`.
    ///
    /// Orthonormalizes `y` via thin QR to produce Q = range_basis. The coefficient
    /// basis is left empty; the caller must solve `L Z = Q` and supply `Z` via
    /// [`Self::set_coeff_basis`] or [`Self::set_coeff_basis_from_slice`] before
    /// reconstructing.
    pub fn from_hmt_range(y: Mat<f64>) -> Self {
        let n_nodes = y.nrows();
        let qr = y.as_ref().qr();
        let q_thin = qr.compute_thin_Q();
        assert_eq!(q_thin.nrows(), n_nodes);
        // Actual rank after thin QR. May be < y.ncols() when n_nodes < y.ncols()
        // (short-fat input): faer's thin Q is then n_nodes × n_nodes.
        let rank = q_thin.ncols();
        CompressedBasis {
            range_basis: q_thin,
            coeff_basis: Mat::<f64>::zeros(n_nodes, rank),
            rank,
            n_nodes,
        }
    }

    /// Borrow the range basis Q as a faer `MatRef` for use as CG right-hand side
    /// when solving `L Z = Q` to populate the coefficient basis.
    pub fn range_basis_ref(&self) -> faer::MatRef<'_, f64> {
        self.range_basis.as_ref()
    }

    /// Set the coefficient basis Z = L^{-1} Q from a flat column-major slice.
    pub fn set_coeff_basis_from_slice(&mut self, z_basis: &[f64]) {
        assert_eq!(z_basis.len(), self.n_nodes * self.rank);
        let src = faer::MatRef::from_column_major_slice(z_basis, self.n_nodes, self.rank);
        self.coeff_basis = src.to_owned();
    }

    /// Set the coefficient basis Z = L^{-1} Q from a MatRef.
    pub fn set_coeff_basis(&mut self, z_basis: faer::MatRef<'_, f64>) {
        assert_eq!(z_basis.nrows(), self.n_nodes);
        assert_eq!(z_basis.ncols(), self.rank);
        self.coeff_basis = z_basis.to_owned();
    }

    /// Reconstruct the pressure solution for a single RHS.
    ///
    /// `p_k ≈ Q · (Z^T · d_k)` where Q = range_basis and Z = coeff_basis.
    pub fn reconstruct(&self, demand: &[f64], pressure_out: &mut [f64]) {
        assert_eq!(demand.len(), self.n_nodes);
        assert_eq!(pressure_out.len(), self.n_nodes);

        // Step 1: c = Z^T * d (m-vector) -- dot products with coefficient basis columns
        let mut c = vec![0.0; self.rank];
        for (i, &d) in demand.iter().enumerate() {
            if d == 0.0 {
                continue;
            }
            for j in 0..self.rank {
                c[j] += self.coeff_basis[(i, j)] * d;
            }
        }

        // Step 2: p = Q * c (n-vector) -- linear combination of range basis columns
        pressure_out.fill(0.0);
        for j in 0..self.rank {
            if c[j].abs() < 1e-15 {
                continue;
            }
            for i in 0..self.n_nodes {
                pressure_out[i] += self.range_basis[(i, j)] * c[j];
            }
        }
    }

    /// Reconstruct multiple pressure solutions from a contiguous demand matrix.
    ///
    /// `demand_matrix` is column-major n_nodes × n_rhs.
    /// Returns a flat column-major buffer of all pressure solutions (n_nodes × n_rhs).
    ///
    /// Formula: `P = Q · (Z^T · D)` implemented as two BLAS matmuls.
    pub fn reconstruct_batch_from_matrix(&self, demand_matrix: &[f64], n_rhs: usize) -> Vec<f64> {
        assert_eq!(demand_matrix.len(), self.n_nodes * n_rhs);
        let d_mat = faer::MatRef::from_column_major_slice(demand_matrix, self.n_nodes, n_rhs);

        // C = Z^T * D  (rank × n_rhs)
        let mut c_mat = Mat::<f64>::zeros(self.rank, n_rhs);
        faer::linalg::matmul::matmul(
            c_mat.as_mut(),
            faer::Accum::Replace,
            self.coeff_basis.as_ref().transpose(),
            d_mat,
            1.0,
            faer::Par::rayon(0),
        );

        // P = Q * C  (n_nodes × n_rhs)
        let mut pressures = vec![0.0; self.n_nodes * n_rhs];
        let p_out = faer::MatMut::from_column_major_slice_mut(&mut pressures, self.n_nodes, n_rhs);
        faer::linalg::matmul::matmul(
            p_out,
            faer::Accum::Replace,
            self.range_basis.as_ref(),
            c_mat.as_ref(),
            1.0,
            faer::Par::rayon(0),
        );

        pressures
    }
}

/// Generate a random Gaussian matrix (n_rows x n_cols) stored column-major.
/// Uses xoshiro256** PRNG with Box-Muller transform.
fn random_gaussian_matrix(n_rows: usize, n_cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xoshiro256SS::new(seed);
    let len = n_rows * n_cols;
    let mut data = Vec::with_capacity(len);

    // Box-Muller transform for Gaussian samples
    for _ in 0..(len + 1) / 2 {
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(r * theta.cos());
        data.push(r * theta.sin());
    }
    data.truncate(len);
    data
}

/// Xoshiro256** PRNG -- fast, high-quality, reproducible.
struct Xoshiro256SS {
    s: [u64; 4],
}

impl Xoshiro256SS {
    fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state from a single seed
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    #[test]
    fn test_rank_analysis_basic() {
        // 2 identical RHS -> rank 1
        let n = 10;
        let solutions = vec![1.0; n * 2]; // two identical columns
        let analysis = analyze_rank(&solutions, n, 2, 0.99);
        assert_eq!(analysis.effective_rank, 1);
    }

    #[test]
    fn test_rank_analysis_orthogonal() {
        // 2 orthogonal RHS -> rank 2
        let n = 4;
        let mut solutions = vec![0.0; n * 2];
        // Column 0: [1, 0, 1, 0]
        solutions[0] = 1.0;
        solutions[2] = 1.0;
        // Column 1: [0, 1, 0, 1]
        solutions[n + 1] = 1.0;
        solutions[n + 3] = 1.0;
        let analysis = analyze_rank(&solutions, n, 2, 0.99);
        assert_eq!(analysis.effective_rank, 2);
    }

    #[test]
    fn test_random_gaussian_deterministic() {
        let a = random_gaussian_matrix(10, 5, 42);
        let b = random_gaussian_matrix(10, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_gaussian_distribution() {
        let data = random_gaussian_matrix(1000, 1, 123);
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        // Should be approximately N(0,1)
        assert!(mean.abs() < 0.15, "mean = {}", mean);
        assert!((var - 1.0).abs() < 0.2, "var = {}", var);
    }

    /// End-to-end HMT regression: approximate L^{-1} D for a 1D tridiagonal
    /// Laplacian against a dense Cholesky ground truth, assert the Frobenius
    /// relative error is small.
    ///
    /// This test would have caught the original bug (Q = orth(D Ω) instead of
    /// Q = orth(L^{-1} D Ω)) — the wrong projection gives relative error ~0.8
    /// on this tridiagonal problem while the correct HMT gives < 0.05.
    #[test]
    fn test_hmt_reconstruction_tridiagonal() {
        // 1D Laplacian on n nodes with Dirichlet boundary at node 0.
        // L is tridiag(-1, 2, -1). SPD after pinning the first row/col.
        let n = 40;
        let mut l = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            l[(i, i)] = 2.0;
            if i + 1 < n {
                l[(i, i + 1)] = -1.0;
                l[(i + 1, i)] = -1.0;
            }
        }

        // 10 sparse-random demand vectors (nonzero in a handful of entries).
        let n_rhs = 10;
        let mut rng = Xoshiro256SS::new(7);
        let mut demand_matrix = vec![0.0; n * n_rhs];
        for k in 0..n_rhs {
            // 4 random nonzeros per column with signed unit weight
            for _ in 0..4 {
                let idx = (rng.next_u64() as usize) % n;
                let sign = if rng.next_f64() > 0.5 { 1.0 } else { -1.0 };
                demand_matrix[k * n + idx] += sign;
            }
        }

        // Ground truth: dense Cholesky factorize L, solve L P_true = D
        let d_ref = faer::MatRef::from_column_major_slice(&demand_matrix, n, n_rhs);
        let llt = l.as_ref().llt(faer::Side::Lower).expect("llt");
        let p_true = llt.solve(d_ref);

        // HMT with rank = 8 (< n_rhs, compressing 10 → 8)
        let rank = 8;
        let sketch = hmt_sketch_matrix(&demand_matrix, n, n_rhs, rank, 123);
        // Y = L^{-1} S via dense Cholesky (acts as an exact block solver in the test)
        let y = llt.solve(sketch.as_ref());
        let mut basis = CompressedBasis::from_hmt_range(y);

        // Z = L^{-1} Q
        let q_mat = basis.range_basis.clone();
        let z = llt.solve(q_mat.as_ref());
        basis.set_coeff_basis(z.as_ref());

        // Reconstruct P = Q · (Z^T · D)
        let p_approx_flat = basis.reconstruct_batch_from_matrix(&demand_matrix, n_rhs);
        let p_approx = faer::MatRef::from_column_major_slice(&p_approx_flat, n, n_rhs);

        // Frobenius relative error
        let mut num: f64 = 0.0;
        let mut den: f64 = 0.0;
        for j in 0..n_rhs {
            for i in 0..n {
                let truth = p_true[(i, j)];
                let diff = p_approx[(i, j)] - truth;
                num += diff * diff;
                den += truth * truth;
            }
        }
        let rel_err = (num / den.max(1e-30)).sqrt();
        assert!(
            rel_err < 0.05,
            "HMT reconstruction relative Frobenius error = {:.4e} (expected < 5e-2)",
            rel_err
        );
    }

    /// Regression for the short-fat sketch case: when n_nodes < n_rhs, faer's
    /// thin QR returns an n_nodes × n_nodes matrix, not n_nodes × n_rhs. The
    /// HMT basis must adopt the actual rank (n_nodes) so the downstream
    /// coefficient solve has the right number of RHS.
    #[test]
    fn test_hmt_rank_shrinks_when_short_fat_sketch() {
        let n = 6;
        let n_rhs = 20; // more RHS than nodes
        let mut l = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            l[(i, i)] = 2.0;
            if i + 1 < n {
                l[(i, i + 1)] = -1.0;
                l[(i + 1, i)] = -1.0;
            }
        }
        let mut rng = Xoshiro256SS::new(5);
        let mut demand_matrix = vec![0.0; n * n_rhs];
        for v in demand_matrix.iter_mut() {
            *v = rng.next_f64() - 0.5;
        }
        let llt = l.as_ref().llt(faer::Side::Lower).expect("llt");
        let d_ref = faer::MatRef::from_column_major_slice(&demand_matrix, n, n_rhs);
        let p_true = llt.solve(d_ref);

        let requested_rank = 15; // > n_nodes
        let sketch = hmt_sketch_matrix(&demand_matrix, n, n_rhs, requested_rank, 7);
        let y = llt.solve(sketch.as_ref());
        let mut basis = CompressedBasis::from_hmt_range(y);
        // Actual rank must equal n_nodes, not requested_rank.
        assert_eq!(basis.rank, n);
        assert_eq!(basis.range_basis.ncols(), n);

        let q_mat = basis.range_basis.clone();
        let z = llt.solve(q_mat.as_ref());
        basis.set_coeff_basis(z.as_ref());

        // With rank == n_nodes the approximation must be exact up to roundoff.
        let p_approx_flat = basis.reconstruct_batch_from_matrix(&demand_matrix, n_rhs);
        let p_approx = faer::MatRef::from_column_major_slice(&p_approx_flat, n, n_rhs);
        for j in 0..n_rhs {
            for i in 0..n {
                let diff = (p_approx[(i, j)] - p_true[(i, j)]).abs();
                assert!(
                    diff < 1e-9,
                    "rank=n_nodes should be exact: diff[{},{}] = {:.2e}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    /// Sanity: if rank ≥ n_rhs the HMT reconstruction must recover the exact
    /// solution up to floating-point roundoff, since Q then spans all of range(L^{-1} D).
    #[test]
    fn test_hmt_exact_when_rank_equals_nrhs() {
        let n = 20;
        let mut l = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            l[(i, i)] = 2.0;
            if i + 1 < n {
                l[(i, i + 1)] = -1.0;
                l[(i + 1, i)] = -1.0;
            }
        }

        let n_rhs = 5;
        let mut rng = Xoshiro256SS::new(11);
        let mut demand_matrix = vec![0.0; n * n_rhs];
        for v in demand_matrix.iter_mut() {
            *v = rng.next_f64() - 0.5;
        }

        let llt = l.as_ref().llt(faer::Side::Lower).expect("llt");
        let d_ref = faer::MatRef::from_column_major_slice(&demand_matrix, n, n_rhs);
        let p_true = llt.solve(d_ref);

        let rank = n_rhs;
        let sketch = hmt_sketch_matrix(&demand_matrix, n, n_rhs, rank, 99);
        let y = llt.solve(sketch.as_ref());
        let mut basis = CompressedBasis::from_hmt_range(y);
        let q_mat = basis.range_basis.clone();
        let z = llt.solve(q_mat.as_ref());
        basis.set_coeff_basis(z.as_ref());

        let p_approx_flat = basis.reconstruct_batch_from_matrix(&demand_matrix, n_rhs);
        let p_approx = faer::MatRef::from_column_major_slice(&p_approx_flat, n, n_rhs);

        for j in 0..n_rhs {
            for i in 0..n {
                let diff = (p_approx[(i, j)] - p_true[(i, j)]).abs();
                assert!(
                    diff < 1e-9,
                    "rank=n_rhs should be exact: diff[{},{}] = {:.2e}",
                    i,
                    j,
                    diff
                );
            }
        }
    }
}

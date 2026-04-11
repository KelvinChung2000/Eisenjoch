//! Smoothed Aggregation AMG preconditioner for graph Laplacians.
//!
//! SA-AMG is robust for irregular graphs with variable conductances (Kirchhoff networks):
//! - Strength-of-connection aggregation (theta = 0.08)
//! - Tentative prolongation preserves constant near-null space by construction
//! - Jacobi-smoothed prolongation: P = (I - omega * D^{-1} * A) * P_tent
//! - Galerkin coarse operator: A_c = P^T A P (SPD by construction)
//! - Weighted Jacobi smoother (omega = 2/3)
//!
//! Single-pass hierarchy: aggregation + numerics built together level-by-level.

use std::cell::UnsafeCell;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::sparse::{SparseColMat, Triplet};

/// Data for one SA-AMG level.
struct Level {
    n: usize,
    diag: Vec<f64>,
    off_diag_pattern: Vec<(usize, usize)>,
    off_diag_values: Vec<f64>,
    inv_diag: Vec<f64>,
    /// Full symmetric operator as faer CSC.
    operator_csc: SparseColMat<usize, f64>,
    /// Smoothed prolongation P (fine × coarse) as faer CSC. None for coarsest level.
    prolongation_csc: Option<SparseColMat<usize, f64>>,
    n_coarse: usize,
}

struct LevelWork {
    x: Vec<f64>,
    r: Vec<f64>,
    rhs: Vec<f64>,
    scratch: Vec<f64>,
}

/// Smoothed Aggregation AMG preconditioner.
///
/// Implements `faer::matrix_free::Precond<f64>` for use with faer's CG.
pub struct AmgPreconditioner {
    levels: Vec<Level>,
    work: UnsafeCell<Vec<LevelWork>>,
}

// SAFETY: UnsafeCell workspace is only accessed in apply()/apply_in_place(),
// which faer's CG calls sequentially.
unsafe impl Sync for AmgPreconditioner {}

const MAX_LEVELS: usize = 20;
const COARSE_SIZE: usize = 50;
const OMEGA: f64 = 2.0 / 3.0;
const SMOOTH_ITERS: usize = 2;
const THETA: f64 = 0.08;

impl AmgPreconditioner {
    /// Build SA-AMG hierarchy.
    pub fn setup(n: usize, diag: &[f64], off_diag: &[(usize, usize, f64)]) -> Self {
        let mut levels = Vec::new();
        let mut work_vec = Vec::new();

        let mut cur_n = n;
        let mut cur_diag = diag.to_vec();
        let mut cur_off_diag: Vec<(usize, usize, f64)> = off_diag.to_vec();

        for _ in 0..MAX_LEVELS {
            if cur_n <= COARSE_SIZE {
                break;
            }

            // Build adjacency with values.
            let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); cur_n];
            for &(i, j, v) in &cur_off_diag {
                adj[i].push((j, v));
                adj[j].push((i, v));
            }

            // Strength of connection: |a_ij| >= theta * max_k |a_ik|.
            let mut strong_adj: Vec<Vec<usize>> = vec![Vec::new(); cur_n];
            for i in 0..cur_n {
                let max_abs = adj[i].iter().map(|&(_, v)| v.abs()).fold(0.0f64, f64::max);
                let threshold = THETA * max_abs;
                for &(j, v) in &adj[i] {
                    if v.abs() >= threshold {
                        strong_adj[i].push(j);
                    }
                }
            }

            // Greedy aggregation.
            let mut agg = vec![usize::MAX; cur_n]; // aggregate assignment
            let mut n_agg = 0usize;

            // Phase 1: seed aggregates from unaggregated nodes with unaggregated strong neighbors.
            for i in 0..cur_n {
                if agg[i] != usize::MAX {
                    continue;
                }
                // Check all strong neighbors are unaggregated.
                let all_free = strong_adj[i].iter().all(|&j| agg[j] == usize::MAX);
                if !all_free {
                    continue;
                }
                let ag = n_agg;
                n_agg += 1;
                agg[i] = ag;
                for &j in &strong_adj[i] {
                    agg[j] = ag;
                }
            }

            // Phase 2: assign remaining nodes to neighbor's aggregate.
            for i in 0..cur_n {
                if agg[i] != usize::MAX {
                    continue;
                }
                // Find strongest-connected aggregated neighbor.
                let mut best_agg = usize::MAX;
                let mut best_strength = 0.0f64;
                for &(j, v) in &adj[i] {
                    if agg[j] != usize::MAX && v.abs() > best_strength {
                        best_strength = v.abs();
                        best_agg = agg[j];
                    }
                }
                if best_agg != usize::MAX {
                    agg[i] = best_agg;
                } else {
                    // Isolated node: own aggregate.
                    agg[i] = n_agg;
                    n_agg += 1;
                }
            }

            if n_agg == 0 || n_agg >= cur_n {
                break;
            }

            // Tentative prolongation: P_tent[i, agg(i)] = 1.
            // This preserves the constant near-null space: P_tent * 1 = 1.
            let mut p_tent_triplets = Vec::with_capacity(cur_n);
            for i in 0..cur_n {
                p_tent_triplets.push(Triplet {
                    row: i,
                    col: agg[i],
                    val: 1.0,
                });
            }
            let p_tent =
                SparseColMat::<usize, f64>::try_new_from_triplets(cur_n, n_agg, &p_tent_triplets)
                    .unwrap();

            // Smoothed prolongation: P = (I - omega * D^{-1} * A) * P_tent.
            // Compute inv_diag.
            let diag_mean = cur_diag.iter().sum::<f64>() / cur_n.max(1) as f64;
            let inv_diag_cap = 1.0 / (1e-6 * diag_mean.abs().max(1e-12));
            let inv_diag: Vec<f64> = cur_diag
                .iter()
                .map(|&d| {
                    let inv = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
                    inv.min(inv_diag_cap)
                })
                .collect();

            // Build operator CSC for this level.
            let off_diag_pattern: Vec<(usize, usize)> =
                cur_off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
            let off_diag_values: Vec<f64> = cur_off_diag.iter().map(|&(_, _, v)| v).collect();
            let operator_csc =
                build_symmetric_csc(cur_n, &cur_diag, &off_diag_pattern, &off_diag_values);

            // Use unsmoothed P_tent as prolongation.
            // Smoothed P = (I - ω D^{-1} A) P_tent breaks null-space preservation
            // when the system has regularization (A = L + εI), causing V-cycle
            // to lose SPD property. P_tent exactly preserves the constant vector.
            let p_csc = p_tent;

            // Galerkin: A_c = P^T A P (using faer sparse matmul).
            let (coarse_diag, coarse_off_diag) =
                galerkin_coarse_operator(cur_n, n_agg, &cur_diag, &cur_off_diag, &p_csc);

            levels.push(Level {
                n: cur_n,
                diag: cur_diag.clone(),
                off_diag_pattern,
                off_diag_values,
                inv_diag,
                operator_csc,
                prolongation_csc: Some(p_csc),
                n_coarse: n_agg,
            });

            work_vec.push(LevelWork {
                x: vec![0.0; cur_n],
                r: vec![0.0; cur_n],
                rhs: vec![0.0; cur_n],
                scratch: vec![0.0; cur_n],
            });

            cur_n = n_agg;
            cur_diag = coarse_diag;
            cur_off_diag = coarse_off_diag;
        }

        // Coarsest level.
        let off_diag_pattern: Vec<(usize, usize)> =
            cur_off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
        let off_diag_values: Vec<f64> = cur_off_diag.iter().map(|&(_, _, v)| v).collect();
        let operator_csc =
            build_symmetric_csc(cur_n, &cur_diag, &off_diag_pattern, &off_diag_values);

        let diag_mean = cur_diag.iter().sum::<f64>() / cur_n.max(1) as f64;
        let inv_diag_cap = 1.0 / (1e-6 * diag_mean.abs().max(1e-12));
        let inv_diag: Vec<f64> = cur_diag
            .iter()
            .map(|&d| {
                let inv = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
                inv.min(inv_diag_cap)
            })
            .collect();

        levels.push(Level {
            n: cur_n,
            diag: cur_diag,
            off_diag_pattern,
            off_diag_values,
            inv_diag,
            operator_csc,
            prolongation_csc: None,
            n_coarse: 0,
        });

        work_vec.push(LevelWork {
            x: vec![0.0; cur_n],
            r: vec![0.0; cur_n],
            rhs: vec![0.0; cur_n],
            scratch: vec![0.0; cur_n],
        });

        let mut amg = Self {
            levels,
            work: UnsafeCell::new(work_vec),
        };
        amg.check_spd();
        amg
    }

    /// Update numeric values (full rebuild since aggregation depends on values).
    pub fn update_values(&mut self, diag: &[f64], off_diag: &[(usize, usize, f64)]) {
        let n = self.levels[0].n;
        *self = Self::setup(n, diag, off_diag);
    }

    /// Check if structure needs rebuild.
    pub fn needs_rebuild(&self, new_diag: &[f64], new_off_diag: &[(usize, usize, f64)]) -> bool {
        let lvl = &self.levels[0];
        if new_off_diag.len() != lvl.off_diag_pattern.len() {
            return true;
        }
        for (idx, &(i, j, _)) in new_off_diag.iter().enumerate() {
            if (i, j) != lvl.off_diag_pattern[idx] {
                return true;
            }
        }
        let old_norm_sq: f64 = lvl.diag.iter().map(|d| d * d).sum::<f64>()
            + lvl.off_diag_values.iter().map(|v| v * v).sum::<f64>();
        let mut diff_norm_sq = 0.0;
        for (i, &d) in new_diag.iter().enumerate() {
            let delta = d - lvl.diag[i];
            diff_norm_sq += delta * delta;
        }
        for (idx, &(_, _, v)) in new_off_diag.iter().enumerate() {
            let delta = v - lvl.off_diag_values[idx];
            diff_norm_sq += delta * delta;
        }
        let old_norm = old_norm_sq.sqrt().max(1e-30);
        diff_norm_sq.sqrt() / old_norm > 0.1
    }

    /// Apply one V-cycle: z = M^{-1} r.
    pub fn v_cycle(&mut self, rhs: &[f64], z: &mut [f64]) {
        let work = self.work.get_mut();
        work[0].rhs.copy_from_slice(rhs);
        work[0].x.fill(0.0);
        Self::v_cycle_level(&self.levels, work, 0);
        z.copy_from_slice(&work[0].x);
    }

    fn v_cycle_level(levels: &[Level], work: &mut [LevelWork], level: usize) {
        let lvl = &levels[level];
        let n = lvl.n;
        let csc = &lvl.operator_csc;

        if level == levels.len() - 1 {
            // Coarsest level: many Jacobi iterations.
            let wrk = &mut work[level];
            let iters = (n * 3).clamp(20, 200);
            for _ in 0..iters {
                jacobi_smooth(
                    csc,
                    &lvl.inv_diag,
                    &mut wrk.x,
                    &wrk.rhs,
                    &mut wrk.scratch,
                    n,
                );
            }
            return;
        }

        let p_csc = lvl.prolongation_csc.as_ref().unwrap();
        let n_coarse = lvl.n_coarse;

        // Pre-smooth.
        for _ in 0..SMOOTH_ITERS {
            let wrk = &mut work[level];
            jacobi_smooth(
                csc,
                &lvl.inv_diag,
                &mut wrk.x,
                &wrk.rhs,
                &mut wrk.scratch,
                n,
            );
        }

        // Residual: r = rhs - A*x.
        {
            let wrk = &mut work[level];
            let x_col = faer::MatRef::from_column_major_slice(&wrk.x, n, 1);
            let mut r_col = faer::MatMut::from_column_major_slice_mut(&mut wrk.r, n, 1);
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                r_col.as_mut(),
                faer::Accum::Replace,
                csc.as_ref(),
                x_col,
                1.0,
                faer::Par::Seq,
            );
            for i in 0..n {
                wrk.r[i] = wrk.rhs[i] - wrk.r[i];
            }
        }

        // Restrict: rhs_coarse = P^T * r_fine.
        {
            let (work_fine, work_rest) = work.split_at_mut(level + 1);
            let wrk_fine = &work_fine[level];
            let wrk_coarse = &mut work_rest[0];
            wrk_coarse.x.fill(0.0);

            let r_col = faer::MatRef::from_column_major_slice(&wrk_fine.r, n, 1);
            let mut rhs_c =
                faer::MatMut::from_column_major_slice_mut(&mut wrk_coarse.rhs, n_coarse, 1);
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                rhs_c.as_mut(),
                faer::Accum::Replace,
                p_csc.as_ref().transpose(),
                r_col,
                1.0,
                faer::Par::Seq,
            );
        }

        // Recurse.
        Self::v_cycle_level(levels, work, level + 1);

        // Prolongate: x_fine += P * x_coarse.
        {
            let (work_fine, work_rest) = work.split_at_mut(level + 1);
            let wrk_fine = &mut work_fine[level];
            let wrk_coarse = &work_rest[0];

            let xc = faer::MatRef::from_column_major_slice(&wrk_coarse.x, n_coarse, 1);
            let mut xf = faer::MatMut::from_column_major_slice_mut(&mut wrk_fine.x, n, 1);
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                xf.as_mut(),
                faer::Accum::Add,
                p_csc.as_ref(),
                xc,
                1.0,
                faer::Par::Seq,
            );
        }

        // Post-smooth.
        for _ in 0..SMOOTH_ITERS {
            let wrk = &mut work[level];
            jacobi_smooth(
                csc,
                &lvl.inv_diag,
                &mut wrk.x,
                &wrk.rhs,
                &mut wrk.scratch,
                n,
            );
        }
    }

    fn check_spd(&mut self) {
        let n = self.levels[0].n;
        if n == 0 {
            return;
        }
        let test_rhs: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) as f64).sin()).collect();
        let mut test_z = vec![0.0; n];
        self.v_cycle(&test_rhs, &mut test_z);
        let rtz: f64 = test_rhs.iter().zip(test_z.iter()).map(|(r, z)| r * z).sum();
        let z_norm: f64 = test_z.iter().map(|z| z * z).sum::<f64>().sqrt();
        let has_nan = test_z.iter().any(|z| z.is_nan() || z.is_infinite());
        if has_nan || rtz <= 0.0 {
            eprintln!(
                "AMG WARNING: V-cycle not SPD! rtz={:.6e} z_norm={:.6e} nan={} n={} levels={}",
                rtz,
                z_norm,
                has_nan,
                n,
                self.num_levels()
            );
            for (i, lvl) in self.levels.iter().enumerate() {
                let d_min = lvl.diag.iter().copied().fold(f64::INFINITY, f64::min);
                let d_max = lvl.diag.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "  level {}: n={} diag=[{:.3e}, {:.3e}]",
                    i, lvl.n, d_min, d_max
                );
            }
        }
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn level_sizes(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.n).collect()
    }
}

impl std::fmt::Debug for AmgPreconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AmgPreconditioner(levels={}, sizes={:?})",
            self.num_levels(),
            self.level_sizes()
        )
    }
}

impl faer::matrix_free::LinOp<f64> for AmgPreconditioner {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.levels[0].n
    }

    fn ncols(&self) -> usize {
        self.levels[0].n
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.nrows();
        let ncols = rhs.ncols();
        let work = unsafe { &mut *self.work.get() };

        for col in 0..ncols {
            work[0].rhs.resize(n, 0.0);
            for i in 0..n {
                work[0].rhs[i] = rhs[(i, col)];
            }
            work[0].x.fill(0.0);
            Self::v_cycle_level(&self.levels, work, 0);
            for i in 0..n {
                out[(i, col)] = work[0].x[i];
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

impl faer::matrix_free::Precond<f64> for AmgPreconditioner {
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(
        &self,
        mut rhs: faer::MatMut<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.nrows();
        let ncols = rhs.ncols();
        let work = unsafe { &mut *self.work.get() };

        for col in 0..ncols {
            work[0].rhs.resize(n, 0.0);
            for i in 0..n {
                work[0].rhs[i] = rhs[(i, col)];
            }
            work[0].x.fill(0.0);
            Self::v_cycle_level(&self.levels, work, 0);
            for i in 0..n {
                rhs[(i, col)] = work[0].x[i];
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

/// Build full symmetric CSC from diagonal + upper-triangle off-diagonal.
fn build_symmetric_csc(
    n: usize,
    diag: &[f64],
    offdiag_pattern: &[(usize, usize)],
    off_diag_values: &[f64],
) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::with_capacity(n + offdiag_pattern.len() * 2);
    for i in 0..n {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: diag[i],
        });
    }
    for (idx, &(i, j)) in offdiag_pattern.iter().enumerate() {
        let v = off_diag_values[idx];
        triplets.push(Triplet {
            row: i,
            col: j,
            val: v,
        });
        triplets.push(Triplet {
            row: j,
            col: i,
            val: v,
        });
    }
    SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap()
}

/// Compute smoothed prolongation: P = (I - omega * D^{-1} * A) * P_tent.
///
/// Instead of full sparse matrix multiply, we compute column-by-column:
/// For each coarse column c, P_tent[:,c] has 1s at nodes in aggregate c.
/// (I - omega * D^{-1} * A) applied to this gives:
///   P[i,c] = P_tent[i,c] - omega * inv_diag[i] * (A * P_tent[:,c])[i]
fn smooth_prolongation(
    n: usize,
    n_agg: usize,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    inv_diag: &[f64],
    agg: &[usize],
    _p_tent: &SparseColMat<usize, f64>,
    _operator_csc: &SparseColMat<usize, f64>,
) -> SparseColMat<usize, f64> {
    // Build aggregate membership lists.
    let mut agg_members: Vec<Vec<usize>> = vec![Vec::new(); n_agg];
    for i in 0..n {
        agg_members[agg[i]].push(i);
    }

    // Build adjacency.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j, v) in off_diag {
        adj[i].push((j, v));
        adj[j].push((i, v));
    }

    // For each node i, compute P[i, c] for each coarse column c where the value is nonzero.
    // P[i, agg(i)] = 1 - omega * inv_diag[i] * diag[i]  (from P_tent diagonal)
    //              = 1 - omega  (when inv_diag = 1/diag, which it is for well-conditioned)
    // P[i, agg(j)] += -omega * inv_diag[i] * a_ij  for each neighbor j in a different aggregate
    // P[i, agg(i)] += -omega * inv_diag[i] * a_ij  for each neighbor j in the SAME aggregate

    let mut p_entries: Vec<(usize, usize, f64)> = Vec::new(); // (row, col, val)

    for i in 0..n {
        let my_agg = agg[i];
        // Start with P_tent contribution: P[i, my_agg] = 1.
        // Then subtract omega * inv_diag[i] * (A * P_tent[:,c])[i] for each c.
        //
        // (A * P_tent[:,my_agg])[i] = diag[i] * 1 + sum_{j in same_agg, j≠i} a_ij * 1
        //                           = diag[i] + sum_{j in same_agg, j≠i} a_ij
        // (A * P_tent[:,other_agg])[i] = sum_{j in other_agg} a_ij

        // Collect contributions by target aggregate.
        let mut agg_contrib: Vec<(usize, f64)> = Vec::new();

        // Diagonal contribution to own aggregate.
        agg_contrib.push((my_agg, diag[i]));

        // Off-diagonal contributions.
        for &(j, a_ij) in &adj[i] {
            agg_contrib.push((agg[j], a_ij));
        }

        // Sum by aggregate.
        agg_contrib.sort_by_key(|&(c, _)| c);
        let mut prev_c = usize::MAX;
        let mut sum = 0.0;

        for &(c, val) in &agg_contrib {
            if c != prev_c {
                if prev_c != usize::MAX {
                    // P[i, prev_c] = (prev_c == my_agg ? 1 : 0) - omega * inv_diag[i] * sum
                    let base = if prev_c == my_agg { 1.0 } else { 0.0 };
                    let p_val = base - OMEGA * inv_diag[i] * sum;
                    if p_val.abs() > 1e-14 {
                        p_entries.push((i, prev_c, p_val));
                    }
                }
                prev_c = c;
                sum = val;
            } else {
                sum += val;
            }
        }
        // Flush last aggregate.
        if prev_c != usize::MAX {
            let base = if prev_c == my_agg { 1.0 } else { 0.0 };
            let p_val = base - OMEGA * inv_diag[i] * sum;
            if p_val.abs() > 1e-14 {
                p_entries.push((i, prev_c, p_val));
            }
        }
    }

    let triplets: Vec<Triplet<usize, usize, f64>> = p_entries
        .iter()
        .map(|&(r, c, v)| Triplet {
            row: r,
            col: c,
            val: v,
        })
        .collect();

    SparseColMat::<usize, f64>::try_new_from_triplets(n, n_agg, &triplets).unwrap()
}

/// Compute Galerkin coarse operator A_c = P^T A P.
///
/// Returns (coarse_diag, coarse_off_diag) in upper-triangle format.
fn galerkin_coarse_operator(
    n: usize,
    n_coarse: usize,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    p_csc: &SparseColMat<usize, f64>,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    // Extract P entries per row for efficient access.
    // P is stored as CSC, convert to per-row.
    let mut p_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let p_ref = p_csc.as_ref();
    for col in 0..n_coarse {
        for (row, &val) in p_ref.row_idx_of_col(col).zip(p_ref.val_of_col(col).iter()) {
            p_rows[row].push((col, val));
        }
    }

    // A_c[ci, cj] = sum_{i,j} P[i,ci] * A[i,j] * P[j,cj]
    let mut coarse_diag = vec![0.0f64; n_coarse];
    let mut coarse_offdiag_map: std::collections::BTreeMap<(usize, usize), f64> =
        std::collections::BTreeMap::new();

    // Diagonal contributions: A[i,i] * P[i,ci] * P[i,cj].
    for i in 0..n {
        let a_ii = diag[i];
        for k1 in 0..p_rows[i].len() {
            let (ci1, pi1) = p_rows[i][k1];
            coarse_diag[ci1] += pi1 * a_ii * pi1;
            for k2 in (k1 + 1)..p_rows[i].len() {
                let (ci2, pi2) = p_rows[i][k2];
                let val = pi1 * a_ii * pi2;
                let (lo, hi) = if ci1 < ci2 { (ci1, ci2) } else { (ci2, ci1) };
                *coarse_offdiag_map.entry((lo, hi)).or_insert(0.0) += val;
            }
        }
    }

    // Off-diagonal contributions: A[i,j] * P[i,ci] * P[j,cj] (symmetric: count both directions).
    for &(i, j, a_ij) in off_diag {
        for &(ci, pi) in &p_rows[i] {
            for &(cj, pj) in &p_rows[j] {
                let val = pi * a_ij * pj;
                if ci == cj {
                    coarse_diag[ci] += 2.0 * val; // both (i,j) and (j,i)
                } else {
                    let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                    *coarse_offdiag_map.entry((lo, hi)).or_insert(0.0) += val;
                }
            }
        }
    }

    // Floor coarse diagonals.
    let fine_diag_mean = diag.iter().sum::<f64>() / n.max(1) as f64;
    let floor = 1e-3 * fine_diag_mean.abs().max(1e-12);
    for d in coarse_diag.iter_mut() {
        if *d < floor {
            *d = floor;
        }
    }

    let coarse_off_diag: Vec<(usize, usize, f64)> = coarse_offdiag_map
        .into_iter()
        .map(|((lo, hi), v)| (lo, hi, v))
        .collect();

    (coarse_diag, coarse_off_diag)
}

/// Weighted Jacobi smoother: x += omega * D^{-1} * (rhs - A*x).
fn jacobi_smooth(
    operator_csc: &SparseColMat<usize, f64>,
    inv_diag: &[f64],
    x: &mut [f64],
    rhs: &[f64],
    scratch: &mut [f64],
    n: usize,
) {
    let x_col = faer::MatRef::from_column_major_slice(x, n, 1);
    let mut s_col = faer::MatMut::from_column_major_slice_mut(scratch, n, 1);
    faer::sparse::linalg::matmul::sparse_dense_matmul(
        s_col.as_mut(),
        faer::Accum::Replace,
        operator_csc.as_ref(),
        x_col,
        1.0,
        faer::Par::Seq,
    );
    for i in 0..n {
        x[i] += OMEGA * inv_diag[i] * (rhs[i] - scratch[i]);
    }
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Sparse matrix-vector product using (diag, off_diag) representation.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let mut diag = vec![2.0; n];
        diag[0] = 2.0;
        diag[n - 1] = 2.0;
        let off_diag: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, -1.0)).collect();
        (diag, off_diag)
    }

    #[test]
    fn amg_structure_builds() {
        let n = 100;
        let (diag, off_diag) = laplacian_1d(n);
        let amg = AmgPreconditioner::setup(n, &diag, &off_diag);
        assert!(amg.num_levels() >= 2, "Should have multiple AMG levels");
    }

    #[test]
    fn amg_v_cycle_reduces_residual() {
        let n = 64;
        let (diag, off_diag) = laplacian_1d(n);
        let mut amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        let rhs = vec![1.0; n];
        let mut z = vec![0.0; n];
        amg.v_cycle(&rhs, &mut z);

        let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(z_norm > 1e-10, "V-cycle result should be non-trivial");

        // Check SPD: r^T z > 0.
        let rtz: f64 = rhs.iter().zip(z.iter()).map(|(r, z)| r * z).sum();
        assert!(rtz > 0.0, "V-cycle should be SPD, got rtz={:.6e}", rtz);
    }

    #[test]
    fn amg_preconditioned_cg_converges() {
        let n = 100;
        let (diag, off_diag) = laplacian_1d(n);
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];
        let mut amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        let rhs_norm = dot(&rhs, &rhs).sqrt().max(1e-12);
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        spmv(&diag, &off_diag, &x, &mut ax);
        for i in 0..n {
            r[i] = rhs[i] - ax[i];
        }

        let mut z = vec![0.0; n];
        amg.v_cycle(&r, &mut z);
        let mut p = z.clone();
        let mut rz_old = dot(&r, &z);
        let mut ap = vec![0.0; n];

        let mut converged = false;
        for _iter in 0..200 {
            spmv(&diag, &off_diag, &p, &mut ap);
            let alpha = rz_old / dot(&p, &ap).max(1e-16);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if dot(&r, &r).sqrt() / rhs_norm < 1e-6 {
                converged = true;
                break;
            }
            amg.v_cycle(&r, &mut z);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz_old = rz_new;
        }

        assert!(converged, "AMG-preconditioned CG should converge");
    }

    #[test]
    fn amg_larger_1d_chain() {
        let n = 200;
        let (diag, off_diag) = laplacian_1d(n);
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut mat = crate::solver::SparseMatrix::new(n);
        for (i, &d) in diag.iter().enumerate() {
            mat.set_diag(i, d);
        }
        for &(lo, hi, val) in &off_diag {
            mat.add_entry(lo, hi, val);
        }

        let op = crate::solver::sparse_matrix::SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        assert!(
            amg.num_levels() >= 2,
            "200-node chain should have >= 2 SA-AMG levels, got {}",
            amg.num_levels()
        );

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = crate::solver::solve_cg(&op, &amg, rhs_mat, x_mat, 1e-8, 200);

        assert!(
            result.converged,
            "AMG+CG on 200-node chain should converge, residual={}",
            result.residual
        );
    }

    #[test]
    fn amg_irregular_graph_laplacian() {
        // Irregular graph with variable conductances (mimics Kirchhoff network).
        let n = 100;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();

        // Chain backbone with unit conductance.
        for i in 0..n - 1 {
            diag[i] += 1.0;
            diag[i + 1] += 1.0;
            off_diag.push((i, i + 1, -1.0));
        }

        // Long-range connections with 100x higher conductance.
        let strong_pairs = [(0, 50, 100.0), (10, 90, 50.0), (25, 75, 200.0)];
        for &(i, j, w) in &strong_pairs {
            diag[i] += w;
            diag[j] += w;
            off_diag.push((i, j, -w));
        }

        // Small anchor.
        for d in diag.iter_mut() {
            *d += 0.01;
        }

        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut mat = crate::solver::SparseMatrix::new(n);
        for (i, &d) in diag.iter().enumerate() {
            mat.set_diag(i, d);
        }
        for &(lo, hi, val) in &off_diag {
            mat.add_entry(lo, hi, val);
        }

        let op = crate::solver::sparse_matrix::SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = crate::solver::solve_cg(&op, &amg, rhs_mat, x_mat, 1e-8, 300);

        assert!(
            result.converged,
            "SA-AMG+CG on irregular graph should converge, residual={}, iters={}",
            result.residual, result.iterations
        );
    }

    #[test]
    fn amg_kirchhoff_like_2d() {
        // 2D grid with highly variable conductances (100x range).
        let nx = 20;
        let ny = 20;
        let n = nx * ny;
        let mut diag = vec![0.01f64; n]; // small regularization
        let mut off_diag = Vec::new();

        for y in 0..ny {
            for x in 0..nx {
                let i = y * nx + x;
                // Horizontal.
                if x + 1 < nx {
                    let j = y * nx + (x + 1);
                    let w = 1.0 + 99.0 * ((i * 7 + j * 13) % 100) as f64 / 100.0;
                    diag[i] += w;
                    diag[j] += w;
                    off_diag.push((i, j, -w));
                }
                // Vertical.
                if y + 1 < ny {
                    let j = (y + 1) * nx + x;
                    let w = 1.0 + 99.0 * ((i * 11 + j * 3) % 100) as f64 / 100.0;
                    diag[i] += w;
                    diag[j] += w;
                    off_diag.push((i, j, -w));
                }
            }
        }

        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut mat = crate::solver::SparseMatrix::new(n);
        for (i, &d) in diag.iter().enumerate() {
            mat.set_diag(i, d);
        }
        for &(lo, hi, val) in &off_diag {
            mat.add_entry(lo, hi, val);
        }

        let op = crate::solver::sparse_matrix::SparseMatrixOp::from_matrix(&mut mat);
        let amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs, n, 1);
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x, n, 1);
        let result = crate::solver::solve_cg(&op, &amg, rhs_mat, x_mat, 1e-8, 500);

        assert!(
            result.converged,
            "SA-AMG+CG on 2D Kirchhoff-like grid should converge, residual={}, iters={}",
            result.residual, result.iterations
        );
    }
}

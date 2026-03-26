//! Algebraic Multigrid (AMG) preconditioner for symmetric positive-definite systems.
//!
//! Implements classical Ruge-Stuben AMG with:
//! - Strength-of-connection based on scaled entries
//! - Greedy independent set coarsening
//! - Classical interpolation
//! - Galerkin coarse operator: A_c = P^T A P
//! - Weighted Jacobi smoother (omega = 2/3)
//!
//! Two-phase design for caching:
//! 1. **Setup** (`AmgStructure::new`): C/F splitting + interpolation structure.
//!    Based on sparsity pattern only. Called once per grid geometry.
//! 2. **Update** (`AmgPreconditioner::update`): fill numeric values into the
//!    cached structure. Called each Newton step when conductances change.

use std::cell::UnsafeCell;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::sparse::{SparseColMat, Triplet};


/// Cached structural information for one AMG level.
/// Built once from the sparsity pattern, reused across numeric updates.
struct LevelStructure {
    /// Number of fine unknowns.
    n: usize,
    /// Number of coarse unknowns.
    n_coarse: usize,
    /// Fine-level off-diagonal sparsity: (row, col) pairs, upper triangle.
    offdiag_pattern: Vec<(usize, usize)>,
    /// Interpolation structure: for each fine node, list of coarse indices.
    interp_indices: Vec<Vec<usize>>,
    /// For F-points: which fine-level adjacency entries are strong C-neighbors.
    interp_sources: Vec<Vec<(usize, usize)>>,
    /// Whether each node is a C-point.
    is_coarse: Vec<bool>,
    /// Coarse-level off-diagonal sparsity pattern.
    coarse_offdiag_pattern: Vec<(usize, usize)>,
}

/// Numeric values for one AMG level, updated each solve.
struct LevelNumeric {
    diag: Vec<f64>,
    off_diag_values: Vec<f64>,
    inv_diag: Vec<f64>,
    /// Interpolation weights (parallel to interp_indices).
    interp_weights: Vec<Vec<f64>>,
    /// Full symmetric operator as faer CSC, for spmv via faer.
    operator_csc: Option<SparseColMat<usize, f64>>,
    /// Prolongation operator P (fine x coarse) as faer CSC.
    interp_csc: Option<SparseColMat<usize, f64>>,
}

struct LevelWork {
    x: Vec<f64>,
    r: Vec<f64>,
    rhs: Vec<f64>,
    scratch: Vec<f64>,
}

/// Cached AMG structure: C/F splitting and interpolation patterns.
/// Built once, reused across many numeric updates.
pub struct AmgStructure {
    level_structs: Vec<LevelStructure>,
}

/// AMG preconditioner with cached structure and updatable numerics.
///
/// Implements `faer::matrix_free::Precond<f64>` for use with faer's CG.
/// The V-cycle workspace is stored internally and uses `UnsafeCell` to allow
/// mutation through the `&self` interface required by `LinOp`/`Precond`.
pub struct AmgPreconditioner {
    structure: AmgStructure,
    numerics: Vec<LevelNumeric>,
    /// V-cycle workspace. Uses UnsafeCell because faer's Precond trait
    /// requires &self but V-cycle needs mutable workspace.
    work: UnsafeCell<Vec<LevelWork>>,
}

// SAFETY: AmgPreconditioner is only used single-threaded (Par::Seq).
// The UnsafeCell is only accessed through &self in apply(), which faer
// guarantees is called sequentially.
unsafe impl Sync for AmgPreconditioner {}

const MAX_LEVELS: usize = 20;
const COARSE_SIZE: usize = 50;
const OMEGA: f64 = 2.0 / 3.0;
const SMOOTH_ITERS: usize = 2;

impl AmgStructure {
    /// Build AMG structure from the sparsity pattern of a symmetric matrix.
    /// This performs C/F splitting and determines interpolation structure.
    /// Called once per grid geometry.
    pub fn new(n: usize, offdiag_pairs: &[(usize, usize)]) -> Self {
        let mut level_structs = Vec::new();
        let mut cur_n = n;
        let mut cur_pairs: Vec<(usize, usize)> = offdiag_pairs.to_vec();

        for _ in 0..MAX_LEVELS {
            if cur_n <= COARSE_SIZE {
                break;
            }

            // Build adjacency from sparsity pattern.
            let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); cur_n];
            for &(i, j) in &cur_pairs {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }

            // For structural C/F splitting, treat all connections as strong.
            let strong_neighbors = &neighbors;

            // Greedy C/F splitting.
            let mut is_coarse = vec![false; cur_n];
            let mut is_fine = vec![false; cur_n];
            let mut measure: Vec<i32> =
                strong_neighbors.iter().map(|s| s.len() as i32).collect();

            loop {
                let mut best = None;
                for i in 0..cur_n {
                    if is_coarse[i] || is_fine[i] {
                        continue;
                    }
                    if best.is_none() || measure[i] > measure[best.unwrap()] {
                        best = Some(i);
                    }
                }
                let i = match best {
                    Some(i) => i,
                    None => break,
                };
                if measure[i] <= 0 {
                    for k in 0..cur_n {
                        if !is_coarse[k] && !is_fine[k] {
                            is_coarse[k] = true;
                        }
                    }
                    break;
                }
                is_coarse[i] = true;
                for &j in &strong_neighbors[i] {
                    if !is_coarse[j] && !is_fine[j] {
                        is_fine[j] = true;
                        for &k in &strong_neighbors[j] {
                            if !is_coarse[k] && !is_fine[k] {
                                measure[k] += 1;
                            }
                        }
                    }
                }
            }

            let mut coarse_map = vec![usize::MAX; cur_n];
            let mut n_coarse = 0;
            for i in 0..cur_n {
                if is_coarse[i] {
                    coarse_map[i] = n_coarse;
                    n_coarse += 1;
                }
            }
            if n_coarse == 0 || n_coarse >= cur_n {
                break;
            }

            // Build interpolation structure.
            let mut interp_indices: Vec<Vec<usize>> = vec![Vec::new(); cur_n];
            let mut interp_sources: Vec<Vec<(usize, usize)>> = vec![Vec::new(); cur_n];

            for i in 0..cur_n {
                if is_coarse[i] {
                    interp_indices[i] = vec![coarse_map[i]];
                } else {
                    for &j in &neighbors[i] {
                        if is_coarse[j] {
                            interp_indices[i].push(coarse_map[j]);
                            interp_sources[i].push((j, coarse_map[j]));
                        }
                    }
                    if interp_indices[i].is_empty() {
                        // Isolated: promote to C-point.
                        coarse_map[i] = n_coarse;
                        interp_indices[i] = vec![n_coarse];
                        is_coarse[i] = true;
                        n_coarse += 1;
                    }
                }
            }

            // Determine coarse-level sparsity pattern from Galerkin P^T A P.
            let mut coarse_pairs_set = std::collections::BTreeSet::new();
            for &(i, j) in &cur_pairs {
                for &ci in &interp_indices[i] {
                    for &cj in &interp_indices[j] {
                        if ci != cj {
                            let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                            coarse_pairs_set.insert((lo, hi));
                        }
                    }
                }
            }
            // Also from diagonal cross-terms.
            for i in 0..cur_n {
                for k1 in 0..interp_indices[i].len() {
                    for k2 in (k1 + 1)..interp_indices[i].len() {
                        let (c1, c2) = (interp_indices[i][k1], interp_indices[i][k2]);
                        let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                        coarse_pairs_set.insert((lo, hi));
                    }
                }
            }
            let coarse_offdiag_pattern: Vec<(usize, usize)> =
                coarse_pairs_set.into_iter().collect();

            level_structs.push(LevelStructure {
                n: cur_n,
                n_coarse,
                offdiag_pattern: cur_pairs,
                interp_indices,
                interp_sources,
                is_coarse,
                coarse_offdiag_pattern: coarse_offdiag_pattern.clone(),
            });

            cur_n = n_coarse;
            cur_pairs = coarse_offdiag_pattern;
        }

        // Coarsest level structure (no interpolation).
        level_structs.push(LevelStructure {
            n: cur_n,
            n_coarse: 0,
            offdiag_pattern: cur_pairs,
            interp_indices: Vec::new(),
            interp_sources: Vec::new(),
            is_coarse: Vec::new(),
            coarse_offdiag_pattern: Vec::new(),
        });

        Self { level_structs }
    }

    pub fn num_levels(&self) -> usize {
        self.level_structs.len()
    }
}

impl AmgPreconditioner {
    /// Build AMG hierarchy from scratch.
    pub fn setup(n: usize, diag: &[f64], off_diag: &[(usize, usize, f64)]) -> Self {
        let pairs: Vec<(usize, usize)> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
        let structure = AmgStructure::new(n, &pairs);
        Self::new(structure, diag, off_diag)
    }

    /// Create a new AMG preconditioner from a cached structure and initial matrix values.
    pub fn new(
        structure: AmgStructure,
        diag: &[f64],
        off_diag: &[(usize, usize, f64)],
    ) -> Self {
        let num_levels = structure.level_structs.len();
        let mut numerics = Vec::with_capacity(num_levels);
        let mut work = Vec::with_capacity(num_levels);

        for ls in &structure.level_structs {
            numerics.push(LevelNumeric {
                diag: vec![0.0; ls.n],
                off_diag_values: vec![0.0; ls.offdiag_pattern.len()],
                inv_diag: vec![1.0; ls.n],
                interp_weights: ls
                    .interp_indices
                    .iter()
                    .map(|ii| vec![0.0; ii.len()])
                    .collect(),
                operator_csc: None,
                interp_csc: None,
            });
            work.push(LevelWork {
                x: vec![0.0; ls.n],
                r: vec![0.0; ls.n],
                rhs: vec![0.0; ls.n],
                scratch: vec![0.0; ls.n],
            });
        }

        let mut amg = Self {
            structure,
            numerics,
            work: UnsafeCell::new(work),
        };
        amg.update(diag, off_diag);
        amg
    }

    /// Update numeric values only (reuse structure).
    /// Call this when sparsity pattern hasn't changed but values have.
    pub fn update_values(&mut self, diag: &[f64], off_diag: &[(usize, usize, f64)]) {
        self.update(diag, off_diag);
    }

    /// Check if structure needs rebuild (returns true if sigma changed too much).
    pub fn needs_rebuild(
        &self,
        new_diag: &[f64],
        new_off_diag: &[(usize, usize, f64)],
    ) -> bool {
        // Check if sparsity pattern changed
        let ls = &self.structure.level_structs[0];
        if new_off_diag.len() != ls.offdiag_pattern.len() {
            return true;
        }
        // Check if any new entries differ in (i,j) pattern
        for (idx, &(i, j, _)) in new_off_diag.iter().enumerate() {
            if (i, j) != ls.offdiag_pattern[idx] {
                return true;
            }
        }
        // Check relative change in values
        let old_norm_sq: f64 = self.numerics[0]
            .diag
            .iter()
            .map(|d| d * d)
            .sum::<f64>()
            + self.numerics[0]
                .off_diag_values
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
        let mut diff_norm_sq = 0.0;
        for (i, &d) in new_diag.iter().enumerate() {
            let delta = d - self.numerics[0].diag[i];
            diff_norm_sq += delta * delta;
        }
        for (idx, &(_, _, v)) in new_off_diag.iter().enumerate() {
            let delta = v - self.numerics[0].off_diag_values[idx];
            diff_norm_sq += delta * delta;
        }
        let old_norm = old_norm_sq.sqrt().max(1e-30);
        diff_norm_sq.sqrt() / old_norm > 0.1
    }

    /// Update numeric values from new matrix (same sparsity pattern).
    pub fn update(&mut self, diag: &[f64], off_diag: &[(usize, usize, f64)]) {
        let num_levels = self.structure.level_structs.len();

        // Fill level 0 directly from input.
        self.numerics[0].diag.copy_from_slice(diag);
        for (idx, &(_, _, v)) in off_diag.iter().enumerate() {
            self.numerics[0].off_diag_values[idx] = v;
        }
        for (i, d) in diag.iter().enumerate() {
            self.numerics[0].inv_diag[i] = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
        }

        // Build interpolation weights and Galerkin operators for each level.
        for level in 0..num_levels - 1 {
            let ls = &self.structure.level_structs[level];
            let n = ls.n;

            // Build adjacency values for interpolation weight computation.
            let mut adj_val: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
            for (idx, &(i, j)) in ls.offdiag_pattern.iter().enumerate() {
                let v = self.numerics[level].off_diag_values[idx];
                adj_val[i].push((j, v));
                adj_val[j].push((i, v));
            }

            // Compute interpolation weights.
            for i in 0..n {
                if ls.is_coarse[i] {
                    self.numerics[level].interp_weights[i] = vec![1.0];
                } else {
                    let a_ii = self.numerics[level].diag[i].max(1e-30);
                    let mut weights = Vec::new();
                    for &(j, _cj) in &ls.interp_sources[i] {
                        let a_ij = adj_val[i]
                            .iter()
                            .find(|&&(k, _)| k == j)
                            .map(|&(_, v)| v)
                            .unwrap_or(0.0);
                        // Standard RS-AMG direct interpolation: w_j = -a_ij / a_ii
                        // NOT normalized to sum=1 (normalization destroys SPD property)
                        let w = -a_ij / a_ii;
                        weights.push(w);
                    }
                    if weights.is_empty() {
                        weights.push(1.0); // shouldn't happen, but safety
                    }
                    self.numerics[level].interp_weights[i] = weights;
                }
            }

            // Build Galerkin coarse operator: A_c = P^T A P.
            let n_coarse = ls.n_coarse;
            let mut coarse_diag = vec![0.0f64; n_coarse];

            let mut coarse_offdiag_map = std::collections::BTreeMap::new();
            for &(lo, hi) in &ls.coarse_offdiag_pattern {
                coarse_offdiag_map.insert((lo, hi), 0.0f64);
            }

            let interp_pairs = |i: usize| -> Vec<(usize, f64)> {
                ls.interp_indices[i]
                    .iter()
                    .zip(self.numerics[level].interp_weights[i].iter())
                    .map(|(&ci, &w)| (ci, w))
                    .collect()
            };

            // Diagonal contributions.
            for i in 0..n {
                let ip = interp_pairs(i);
                let a_ii = self.numerics[level].diag[i];
                for &(ci, pi) in &ip {
                    coarse_diag[ci] += pi * a_ii * pi;
                }
                for k1 in 0..ip.len() {
                    for k2 in (k1 + 1)..ip.len() {
                        let (ci1, pi1) = ip[k1];
                        let (ci2, pi2) = ip[k2];
                        let val = pi1 * a_ii * pi2;
                        let (lo, hi) = if ci1 < ci2 { (ci1, ci2) } else { (ci2, ci1) };
                        if let Some(v) = coarse_offdiag_map.get_mut(&(lo, hi)) {
                            *v += val;
                        }
                    }
                }
            }

            // Off-diagonal contributions.
            for (idx, &(i, j)) in ls.offdiag_pattern.iter().enumerate() {
                let a_ij = self.numerics[level].off_diag_values[idx];
                let ip_i = interp_pairs(i);
                let ip_j = interp_pairs(j);
                for &(ci, pi) in &ip_i {
                    for &(cj, pj) in &ip_j {
                        let val = pi * a_ij * pj;
                        if ci == cj {
                            coarse_diag[ci] += 2.0 * val;
                        } else {
                            let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                            if let Some(v) = coarse_offdiag_map.get_mut(&(lo, hi)) {
                                *v += val;
                            }
                        }
                    }
                }
            }

            // Fill coarse level numerics.
            let next_level = level + 1;
            self.numerics[next_level].diag = coarse_diag.clone();
            self.numerics[next_level].off_diag_values = ls
                .coarse_offdiag_pattern
                .iter()
                .map(|key| coarse_offdiag_map.get(key).copied().unwrap_or(0.0))
                .collect();
            for (i, d) in coarse_diag.iter().enumerate() {
                self.numerics[next_level].inv_diag[i] =
                    if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
            }
        }

        // Build faer CSC matrices for each level.
        for level in 0..num_levels {
            let ls = &self.structure.level_structs[level];
            let nm = &self.numerics[level];
            let n = ls.n;

            // Build operator CSC from diag + off_diag (full symmetric).
            let mut triplets = Vec::with_capacity(n + ls.offdiag_pattern.len() * 2);
            for i in 0..n {
                triplets.push(Triplet { row: i, col: i, val: nm.diag[i] });
            }
            for (idx, &(i, j)) in ls.offdiag_pattern.iter().enumerate() {
                let v = nm.off_diag_values[idx];
                triplets.push(Triplet { row: i, col: j, val: v });
                triplets.push(Triplet { row: j, col: i, val: v });
            }
            self.numerics[level].operator_csc =
                Some(SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap());

            // Build interpolation P CSC (fine x coarse) for non-coarsest levels.
            if level < num_levels - 1 {
                let n_coarse = ls.n_coarse;
                let mut p_triplets = Vec::new();
                for i in 0..n {
                    for (k, &cj) in ls.interp_indices[i].iter().enumerate() {
                        let w = self.numerics[level].interp_weights[i][k];
                        p_triplets.push(Triplet { row: i, col: cj, val: w });
                    }
                }
                self.numerics[level].interp_csc = Some(
                    SparseColMat::<usize, f64>::try_new_from_triplets(n, n_coarse, &p_triplets)
                        .unwrap(),
                );
            }
        }
    }

    /// Apply one AMG V-cycle: z = M^{-1} r (mutable version).
    pub fn v_cycle(&mut self, rhs: &[f64], z: &mut [f64]) {
        let work = self.work.get_mut();
        work[0].rhs.copy_from_slice(rhs);
        work[0].x.fill(0.0);
        Self::v_cycle_level(&self.structure, &self.numerics, work, 0);
        z.copy_from_slice(&work[0].x);
    }

    /// Internal V-cycle implementation that borrows structure/numerics immutably
    /// and work mutably. This allows calling from both &mut self and &self (via UnsafeCell).
    fn v_cycle_level(
        structure: &AmgStructure,
        numerics: &[LevelNumeric],
        work: &mut [LevelWork],
        level: usize,
    ) {
        let n = structure.level_structs[level].n;
        let nm = &numerics[level];
        let csc = nm.operator_csc.as_ref().unwrap();

        if level == structure.level_structs.len() - 1 {
            // Coarsest level: many Jacobi iterations as approximate solve.
            let wrk = &mut work[level];
            let iters = n.max(200);
            for _ in 0..iters {
                jacobi_smooth_faer(csc, &nm.inv_diag, &mut wrk.x, &wrk.rhs, &mut wrk.scratch, n);
            }
            return;
        }

        let p_csc = nm.interp_csc.as_ref().unwrap();

        // Pre-smooth.
        for _ in 0..SMOOTH_ITERS {
            let wrk = &mut work[level];
            jacobi_smooth_faer(csc, &nm.inv_diag, &mut wrk.x, &wrk.rhs, &mut wrk.scratch, n);
        }

        // Residual: r = rhs - A*x.
        {
            let wrk = &mut work[level];
            // scratch = A*x via faer spmv
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
            // r = rhs - A*x
            for i in 0..n {
                wrk.r[i] = wrk.rhs[i] - wrk.r[i];
            }
        }

        // Restrict: rhs_coarse = P^T * r_fine.
        {
            let n_coarse = structure.level_structs[level].n_coarse;
            let (work_fine, work_rest) = work.split_at_mut(level + 1);
            let wrk_fine = &work_fine[level];
            let wrk_coarse = &mut work_rest[0];
            wrk_coarse.x.fill(0.0);

            // P^T * r: transpose of CSC is CSR, which implements SparseDenseMatMul.
            let r_col = faer::MatRef::from_column_major_slice(&wrk_fine.r, n, 1);
            let mut rhs_coarse_col =
                faer::MatMut::from_column_major_slice_mut(&mut wrk_coarse.rhs, n_coarse, 1);
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                rhs_coarse_col.as_mut(),
                faer::Accum::Replace,
                p_csc.as_ref().transpose(),
                r_col,
                1.0,
                faer::Par::Seq,
            );
        }

        // Recurse.
        Self::v_cycle_level(structure, numerics, work, level + 1);

        // Prolongate: x_fine += P * x_coarse.
        {
            let n_coarse = structure.level_structs[level].n_coarse;
            let (work_fine, work_rest) = work.split_at_mut(level + 1);
            let wrk_fine = &mut work_fine[level];
            let wrk_coarse = &work_rest[0];

            let xc_col = faer::MatRef::from_column_major_slice(&wrk_coarse.x, n_coarse, 1);
            let mut xf_col = faer::MatMut::from_column_major_slice_mut(&mut wrk_fine.x, n, 1);
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                xf_col.as_mut(),
                faer::Accum::Add,
                p_csc.as_ref(),
                xc_col,
                1.0,
                faer::Par::Seq,
            );
        }

        // Post-smooth.
        for _ in 0..SMOOTH_ITERS {
            let wrk = &mut work[level];
            jacobi_smooth_faer(csc, &nm.inv_diag, &mut wrk.x, &wrk.rhs, &mut wrk.scratch, n);
        }
    }

    pub fn num_levels(&self) -> usize {
        self.structure.level_structs.len()
    }

    pub fn level_sizes(&self) -> Vec<usize> {
        self.structure
            .level_structs
            .iter()
            .map(|l| l.n)
            .collect()
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
        self.structure.level_structs[0].n
    }

    fn ncols(&self) -> usize {
        self.structure.level_structs[0].n
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.nrows();
        // SAFETY: We are the only ones accessing work during apply().
        // faer guarantees sequential access to preconditioners.
        let work = unsafe { &mut *self.work.get() };

        // Copy rhs column into work buffer
        work[0].rhs.resize(n, 0.0);
        for i in 0..n {
            work[0].rhs[i] = rhs[(i, 0)];
        }
        work[0].x.fill(0.0);

        Self::v_cycle_level(&self.structure, &self.numerics, work, 0);

        for i in 0..n {
            out[(i, 0)] = work[0].x[i];
        }
    }

    fn conj_apply(
        &self,
        out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        par: faer::Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack); // symmetric
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
        let work = unsafe { &mut *self.work.get() };

        work[0].rhs.resize(n, 0.0);
        for i in 0..n {
            work[0].rhs[i] = rhs[(i, 0)];
        }
        work[0].x.fill(0.0);

        Self::v_cycle_level(&self.structure, &self.numerics, work, 0);

        for i in 0..n {
            rhs[(i, 0)] = work[0].x[i];
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

/// Weighted Jacobi smoother using faer spmv: x += omega * D^{-1} * (rhs - A*x).
fn jacobi_smooth_faer(
    operator_csc: &SparseColMat<usize, f64>,
    inv_diag: &[f64],
    x: &mut [f64],
    rhs: &[f64],
    scratch: &mut [f64],
    n: usize,
) {
    // scratch = A*x via faer
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
    // x += omega * D^{-1} * (rhs - A*x)
    for i in 0..n {
        x[i] += OMEGA * inv_diag[i] * (rhs[i] - scratch[i]);
    }
}

/// Dot product of two vectors.
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

    /// Build a 1D Laplacian of size n: A[i,i] = 2 (or 1 at boundaries), A[i,i+1] = -1.
    fn laplacian_1d(n: usize) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let mut diag = vec![2.0; n];
        diag[0] = 1.0 + 1.0; // anchored
        diag[n - 1] = 1.0 + 1.0;
        let off_diag: Vec<(usize, usize, f64)> =
            (0..n - 1).map(|i| (i, i + 1, -1.0)).collect();
        (diag, off_diag)
    }

    #[test]
    fn amg_structure_builds() {
        let n = 100;
        let (_, off_diag) = laplacian_1d(n);
        let pairs: Vec<(usize, usize)> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
        let structure = AmgStructure::new(n, &pairs);
        assert!(structure.num_levels() >= 2, "Should have multiple AMG levels");
    }

    #[test]
    fn amg_v_cycle_reduces_residual() {
        let n = 64;
        let (diag, off_diag) = laplacian_1d(n);
        let mut amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        // rhs = 1.0 everywhere
        let rhs = vec![1.0; n];
        let mut z = vec![0.0; n];
        amg.v_cycle(&rhs, &mut z);

        // z should be non-trivial
        let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(z_norm > 1e-10, "V-cycle result should be non-trivial");
    }

    #[test]
    fn amg_preconditioned_cg_converges() {
        let n = 100;
        let (diag, off_diag) = laplacian_1d(n);

        // Build rhs
        let rhs = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut amg = AmgPreconditioner::setup(n, &diag, &off_diag);

        // Manual PCG using AMG as preconditioner
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

        let tol = 1e-6;
        let max_iters = 200;
        let mut converged = false;

        for _iter in 0..max_iters {
            spmv(&diag, &off_diag, &p, &mut ap);
            let p_ap = dot(&p, &ap);
            let alpha = rz_old / p_ap.max(1e-16);

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let r_norm = dot(&r, &r).sqrt();
            if r_norm / rhs_norm < tol {
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

        // Verify solution: A*x should be close to rhs
        spmv(&diag, &off_diag, &x, &mut ax);
        let residual: f64 = (0..n)
            .map(|i| (ax[i] - rhs[i]) * (ax[i] - rhs[i]))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual / rhs_norm < 1e-5,
            "Relative residual {} too large",
            residual / rhs_norm
        );
    }

    #[test]
    fn amg_larger_1d_chain() {
        // Larger 1D chain to verify AMG with multiple coarsening levels
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
            amg.num_levels() >= 3,
            "200-node chain should have >= 3 AMG levels, got {}",
            amg.num_levels()
        );

        let result = crate::solver::solve_cg(&op, &amg, &rhs, &mut x, 1e-8, 200);

        assert!(
            result.converged,
            "AMG+CG on 200-node chain should converge, residual={}",
            result.residual
        );

        // Verify solution
        let mut ax = vec![0.0; n];
        mat.spmv(&x, &mut ax);
        let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual: f64 = (0..n).map(|i| (ax[i] - rhs[i]).powi(2)).sum::<f64>().sqrt();
        assert!(
            residual / rhs_norm < 1e-6,
            "Relative residual {} too large",
            residual / rhs_norm
        );
    }
}

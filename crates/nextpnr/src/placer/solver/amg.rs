//! Algebraic Multigrid (AMG) preconditioner for symmetric positive-definite systems.
//!
//! Implements classical Ruge-Stüben AMG with:
//! - Strength-of-connection based on scaled entries
//! - Greedy independent set coarsening
//! - Classical interpolation
//! - Galerkin coarse operator: A_c = P^T A P
//! - Weighted Jacobi smoother (ω = 2/3)
//!
//! Two-phase design for caching:
//! 1. **Setup** (`AmgStructure::new`): C/F splitting + interpolation structure.
//!    Based on sparsity pattern only. Called once per grid geometry.
//! 2. **Update** (`AmgPreconditioner::update`): fill numeric values into the
//!    cached structure. Called each Newton step when conductances change.

/// Cached structural information for one AMG level.
/// Built once from the sparsity pattern, reused across numeric updates.
struct LevelStructure {
    /// Number of fine unknowns.
    n: usize,
    /// Number of coarse unknowns.
    n_coarse: usize,
    /// Fine-level off-diagonal sparsity: (row, col) pairs, upper triangle.
    offdiag_pattern: Vec<(usize, usize)>,
    /// Interpolation structure: for each fine node, list of (coarse_index, position_in_weights).
    /// The actual weight values live in `LevelNumeric::interp_weights`.
    interp_indices: Vec<Vec<usize>>, // interp_indices[i] = [coarse_idx_0, coarse_idx_1, ...]
    /// For F-points: which fine-level adjacency entries are strong C-neighbors.
    /// Used to recompute interpolation weights from new matrix values.
    /// interp_sources[i] = [(adj_neighbor_idx, coarse_map_of_neighbor), ...]
    interp_sources: Vec<Vec<(usize, usize)>>,
    /// Whether each node is a C-point.
    is_coarse: Vec<bool>,
    /// Coarse-level off-diagonal sparsity pattern.
    coarse_offdiag_pattern: Vec<(usize, usize)>,
}

/// Numeric values for one AMG level, updated each solve.
struct LevelNumeric {
    diag: Vec<f64>,
    off_diag_values: Vec<f64>, // values for offdiag_pattern entries
    inv_diag: Vec<f64>,
    /// Interpolation weights (parallel to interp_indices).
    interp_weights: Vec<Vec<f64>>,
}

struct LevelWork {
    x: Vec<f64>,
    r: Vec<f64>,
    rhs: Vec<f64>,
}

/// Cached AMG structure: C/F splitting and interpolation patterns.
/// Built once, reused across many numeric updates.
pub struct AmgStructure {
    level_structs: Vec<LevelStructure>,
}

/// AMG preconditioner with cached structure and updatable numerics.
pub struct AmgPreconditioner {
    structure: AmgStructure,
    numerics: Vec<LevelNumeric>,
    work: Vec<LevelWork>,
}

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

            // For structural C/F splitting, treat all connections as strong
            // (since we don't have values yet). This gives a valid splitting
            // for any set of values on this pattern.
            let strong_neighbors = &neighbors;

            // Greedy C/F splitting.
            let mut is_coarse = vec![false; cur_n];
            let mut is_fine = vec![false; cur_n];
            let mut measure: Vec<i32> = strong_neighbors.iter().map(|s| s.len() as i32).collect();

            loop {
                let mut best = None;
                for i in 0..cur_n {
                    if is_coarse[i] || is_fine[i] { continue; }
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
                        if !is_coarse[k] && !is_fine[k] { is_coarse[k] = true; }
                    }
                    break;
                }
                is_coarse[i] = true;
                for &j in &strong_neighbors[i] {
                    if !is_coarse[j] && !is_fine[j] {
                        is_fine[j] = true;
                        for &k in &strong_neighbors[j] {
                            if !is_coarse[k] && !is_fine[k] { measure[k] += 1; }
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
            if n_coarse == 0 || n_coarse >= cur_n { break; }

            // Build interpolation structure.
            // C-points: identity (single coarse index = self).
            // F-points: interpolate from C-neighbors.
            let mut interp_indices: Vec<Vec<usize>> = vec![Vec::new(); cur_n];
            let mut interp_sources: Vec<Vec<(usize, usize)>> = vec![Vec::new(); cur_n];

            for i in 0..cur_n {
                if is_coarse[i] {
                    interp_indices[i] = vec![coarse_map[i]];
                } else {
                    // F-point: use all C-neighbors as interpolation sources.
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
                    for k2 in (k1+1)..interp_indices[i].len() {
                        let (c1, c2) = (interp_indices[i][k1], interp_indices[i][k2]);
                        let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                        coarse_pairs_set.insert((lo, hi));
                    }
                }
            }
            let coarse_offdiag_pattern: Vec<(usize, usize)> = coarse_pairs_set.into_iter().collect();

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

    pub fn num_levels(&self) -> usize { self.level_structs.len() }
}

impl AmgPreconditioner {
    /// Create a new AMG preconditioner from a cached structure and initial matrix values.
    pub fn new(structure: AmgStructure, diag: &[f64], off_diag: &[(usize, usize, f64)]) -> Self {
        let num_levels = structure.level_structs.len();
        let mut numerics = Vec::with_capacity(num_levels);
        let mut work = Vec::with_capacity(num_levels);

        // Initialize empty numerics and workspace for each level.
        for ls in &structure.level_structs {
            numerics.push(LevelNumeric {
                diag: vec![0.0; ls.n],
                off_diag_values: vec![0.0; ls.offdiag_pattern.len()],
                inv_diag: vec![1.0; ls.n],
                interp_weights: ls.interp_indices.iter().map(|ii| vec![0.0; ii.len()]).collect(),
            });
            work.push(LevelWork {
                x: vec![0.0; ls.n],
                r: vec![0.0; ls.n],
                rhs: vec![0.0; ls.n],
            });
        }

        let mut amg = Self { structure, numerics, work };
        amg.update(diag, off_diag);
        amg
    }

    /// Update numeric values from new matrix (same sparsity pattern).
    /// Recomputes interpolation weights, Galerkin operators, and smoother diagonals.
    /// This is the cheap operation called each Newton step.
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
                    // C-point: identity weight.
                    self.numerics[level].interp_weights[i] = vec![1.0];
                } else {
                    // F-point: w_j = -a_ij / a_ii for C-neighbors.
                    let a_ii = self.numerics[level].diag[i].max(1e-30);
                    let mut weights = Vec::new();
                    let mut sum_w = 0.0;
                    for &(j, _cj) in &ls.interp_sources[i] {
                        // Find a_ij in adjacency.
                        let a_ij = adj_val[i].iter()
                            .find(|&&(k, _)| k == j)
                            .map(|&(_, v)| v)
                            .unwrap_or(0.0);
                        let w = -a_ij / a_ii;
                        weights.push(w);
                        sum_w += w;
                    }
                    // Normalize.
                    if sum_w.abs() > 1e-30 && !weights.is_empty() {
                        let scale = 1.0 / sum_w;
                        for w in &mut weights { *w *= scale; }
                    } else if !weights.is_empty() {
                        let eq = 1.0 / weights.len() as f64;
                        for w in &mut weights { *w = eq; }
                    }
                    self.numerics[level].interp_weights[i] = weights;
                }
            }

            // Build Galerkin coarse operator: A_c = P^T A P.
            let n_coarse = ls.n_coarse;
            let mut coarse_diag = vec![0.0f64; n_coarse];

            // Use a map for coarse off-diagonal accumulation.
            let mut coarse_offdiag_map = std::collections::BTreeMap::new();
            for &(lo, hi) in &ls.coarse_offdiag_pattern {
                coarse_offdiag_map.insert((lo, hi), 0.0f64);
            }

            // Helper: get interpolation (coarse_idx, weight) pairs for node i.
            let interp_pairs = |i: usize| -> Vec<(usize, f64)> {
                ls.interp_indices[i].iter().zip(self.numerics[level].interp_weights[i].iter())
                    .map(|(&ci, &w)| (ci, w))
                    .collect()
            };

            // Diagonal contributions: A_c[ci,ci] += P[i,ci]^2 * A[i,i].
            for i in 0..n {
                let ip = interp_pairs(i);
                let a_ii = self.numerics[level].diag[i];
                for &(ci, pi) in &ip {
                    coarse_diag[ci] += pi * a_ii * pi;
                }
                for k1 in 0..ip.len() {
                    for k2 in (k1+1)..ip.len() {
                        let (ci1, pi1) = ip[k1];
                        let (ci2, pi2) = ip[k2];
                        let val = pi1 * a_ii * pi2;
                        let (lo, hi) = if ci1 < ci2 { (ci1, ci2) } else { (ci2, ci1) };
                        if let Some(v) = coarse_offdiag_map.get_mut(&(lo, hi)) { *v += val; }
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
                            if let Some(v) = coarse_offdiag_map.get_mut(&(lo, hi)) { *v += val; }
                        }
                    }
                }
            }

            // Fill coarse level numerics.
            let next_level = level + 1;
            self.numerics[next_level].diag = coarse_diag.clone();
            self.numerics[next_level].off_diag_values = ls.coarse_offdiag_pattern.iter()
                .map(|key| coarse_offdiag_map.get(key).copied().unwrap_or(0.0))
                .collect();
            for (i, d) in coarse_diag.iter().enumerate() {
                self.numerics[next_level].inv_diag[i] = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
            }
        }
    }

    /// Apply one AMG V-cycle: z = M^{-1} r.
    pub fn v_cycle(&mut self, rhs: &[f64], z: &mut [f64]) {
        self.work[0].rhs.copy_from_slice(rhs);
        self.work[0].x.fill(0.0);
        self.v_cycle_level(0);
        z.copy_from_slice(&self.work[0].x);
    }

    fn v_cycle_level(&mut self, level: usize) {
        let n = self.structure.level_structs[level].n;

        if level == self.structure.level_structs.len() - 1 {
            for _ in 0..100 {
                let (ls, nm) = (&self.structure.level_structs[level], &self.numerics[level]);
                let wrk = &mut self.work[level];
                jacobi_smooth(&nm.diag, &ls.offdiag_pattern, &nm.off_diag_values, &nm.inv_diag, &mut wrk.x, &wrk.rhs, n);
            }
            return;
        }

        // Pre-smooth.
        for _ in 0..SMOOTH_ITERS {
            let (ls, nm) = (&self.structure.level_structs[level], &self.numerics[level]);
            let wrk = &mut self.work[level];
            jacobi_smooth(&nm.diag, &ls.offdiag_pattern, &nm.off_diag_values, &nm.inv_diag, &mut wrk.x, &wrk.rhs, n);
        }

        // Residual.
        {
            let (ls, nm) = (&self.structure.level_structs[level], &self.numerics[level]);
            let wrk = &mut self.work[level];
            compute_residual(&nm.diag, &ls.offdiag_pattern, &nm.off_diag_values, &wrk.x, &wrk.rhs, &mut wrk.r, n);
        }

        // Restrict: rhs_coarse = P^T * r.
        {
            let ls = &self.structure.level_structs[level];
            let (work_fine, work_rest) = self.work.split_at_mut(level + 1);
            let wrk_fine = &work_fine[level];
            let wrk_coarse = &mut work_rest[0];
            wrk_coarse.rhs.fill(0.0);
            wrk_coarse.x.fill(0.0);
            for i in 0..n {
                let ri = wrk_fine.r[i];
                for (k, &cj) in ls.interp_indices[i].iter().enumerate() {
                    let w = self.numerics[level].interp_weights[i][k];
                    wrk_coarse.rhs[cj] += w * ri;
                }
            }
        }

        // Recurse (no borrows held on self.structure).
        self.v_cycle_level(level + 1);

        // Prolongate: x += P * x_coarse.
        {
            let ls = &self.structure.level_structs[level];
            let (work_fine, work_rest) = self.work.split_at_mut(level + 1);
            let wrk_fine = &mut work_fine[level];
            let wrk_coarse = &work_rest[0];
            for i in 0..n {
                for (k, &cj) in ls.interp_indices[i].iter().enumerate() {
                    let w = self.numerics[level].interp_weights[i][k];
                    wrk_fine.x[i] += w * wrk_coarse.x[cj];
                }
            }
        }

        // Post-smooth.
        for _ in 0..SMOOTH_ITERS {
            let (ls, nm) = (&self.structure.level_structs[level], &self.numerics[level]);
            let wrk = &mut self.work[level];
            jacobi_smooth(&nm.diag, &ls.offdiag_pattern, &nm.off_diag_values, &nm.inv_diag, &mut wrk.x, &wrk.rhs, n);
        }
    }

    pub fn num_levels(&self) -> usize { self.structure.level_structs.len() }
    pub fn level_sizes(&self) -> Vec<usize> { self.structure.level_structs.iter().map(|l| l.n).collect() }
}

/// Weighted Jacobi: x += ω * M^{-1} * (rhs - A*x).
/// Uses split pattern/values representation.
fn jacobi_smooth(
    diag: &[f64],
    offdiag_pattern: &[(usize, usize)],
    offdiag_values: &[f64],
    inv_diag: &[f64],
    x: &mut [f64],
    rhs: &[f64],
    n: usize,
) {
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = rhs[i] - diag[i] * x[i];
    }
    for (idx, &(i, j)) in offdiag_pattern.iter().enumerate() {
        let w = offdiag_values[idx];
        r[i] -= w * x[j];
        r[j] -= w * x[i];
    }
    for i in 0..n {
        x[i] += OMEGA * inv_diag[i] * r[i];
    }
}

fn compute_residual(
    diag: &[f64],
    offdiag_pattern: &[(usize, usize)],
    offdiag_values: &[f64],
    x: &[f64],
    rhs: &[f64],
    r: &mut [f64],
    n: usize,
) {
    for i in 0..n {
        r[i] = rhs[i] - diag[i] * x[i];
    }
    for (idx, &(i, j)) in offdiag_pattern.iter().enumerate() {
        let w = offdiag_values[idx];
        r[i] -= w * x[j];
        r[j] -= w * x[i];
    }
}

/// AMG-preconditioned CG with a cached AMG structure.
/// The AMG hierarchy is updated (cheap) but not rebuilt (expensive).
pub fn amg_preconditioned_cg_cached(
    amg: &mut AmgPreconditioner,
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    if n == 0 { return 0; }

    // Update AMG numerics from new matrix values.
    amg.update(diag, off_diag);

    let rhs_norm = rhs.iter().map(|r| r * r).sum::<f64>().sqrt().max(1e-12);

    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    super::cg::spmv(diag, off_diag, x, &mut ax);
    for i in 0..n { r[i] = rhs[i] - ax[i]; }

    if r.iter().map(|ri| ri * ri).sum::<f64>().sqrt() / rhs_norm < tol { return 0; }

    let mut z = vec![0.0; n];
    amg.v_cycle(&r, &mut z);
    let mut p = z.clone();
    let mut rz_old: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
    let mut ap = vec![0.0; n];

    for iter in 0..max_iters {
        super::cg::spmv(diag, off_diag, &p, &mut ap);
        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, ai)| pi * ai).sum();
        let alpha = rz_old / p_ap.max(1e-16);

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        if r.iter().map(|ri| ri * ri).sum::<f64>().sqrt() / rhs_norm < tol {
            return iter + 1;
        }

        amg.v_cycle(&r, &mut z);
        let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let beta = rz_new / rz_old;
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
        rz_old = rz_new;
    }
    max_iters
}

/// Convenience: build + solve in one call (no caching). Used for standalone tests.
pub fn amg_preconditioned_cg(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    let pairs: Vec<(usize, usize)> = off_diag.iter().map(|&(i, j, _)| (i, j)).collect();
    let structure = AmgStructure::new(n, &pairs);
    let mut amg = AmgPreconditioner::new(structure, diag, off_diag);
    amg_preconditioned_cg_cached(&mut amg, diag, off_diag, rhs, x, tol, max_iters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_laplacian(w: usize, h: usize) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let n = w * h;
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    diag[i] += 1.0; diag[j] += 1.0;
                    off_diag.push((i, j, -1.0));
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    diag[i] += 1.0; diag[j] += 1.0;
                    off_diag.push((i, j, -1.0));
                }
            }
        }
        diag[0] += 1e6;
        (diag, off_diag)
    }

    fn check_residual(diag: &[f64], off_diag: &[(usize, usize, f64)], x: &[f64], rhs: &[f64], tol: f64, label: &str) {
        let n = diag.len();
        let mut ax = vec![0.0; n];
        crate::placer::solver::cg::spmv(diag, off_diag, x, &mut ax);
        let res: f64 = rhs.iter().zip(ax.iter()).map(|(b, a)| (b - a).powi(2)).sum::<f64>().sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-30);
        assert!(res / rhs_norm < tol, "{}: {:.2e} >= {:.2e}", label, res / rhs_norm, tol);
    }

    #[test]
    fn amg_hierarchy_builds() {
        let (_diag, off_diag) = make_grid_laplacian(16, 16);
        let pairs: Vec<_> = off_diag.iter().map(|&(i,j,_)|(i,j)).collect();
        let structure = AmgStructure::new(256, &pairs);
        assert!(structure.num_levels() >= 2);
    }

    #[test]
    fn amg_cg_8x8_grid() {
        let (diag, off_diag) = make_grid_laplacian(8, 8);
        let rhs: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut x = vec![0.0; 64];
        let iters = amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
        assert!(iters < 500, "got {} iters", iters);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg 8x8");
    }

    #[test]
    fn amg_cg_32x32_grid() {
        let (diag, off_diag) = make_grid_laplacian(32, 32);
        let n = 1024;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).cos()).collect();
        let mut x = vec![0.0; n];
        let iters = amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
        assert!(iters < 200, "got {} iters", iters);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg 32x32");
    }

    #[test]
    fn amg_cached_reuse() {
        let (diag, off_diag) = make_grid_laplacian(16, 16);
        let n = 256;
        let pairs: Vec<_> = off_diag.iter().map(|&(i,j,_)|(i,j)).collect();
        let structure = AmgStructure::new(n, &pairs);
        let mut amg = AmgPreconditioner::new(structure, &diag, &off_diag);

        // First solve.
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut x = vec![0.0; n];
        let iters1 = amg_preconditioned_cg_cached(&mut amg, &diag, &off_diag, &rhs, &mut x, 1e-3, 500);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "cached solve 1");

        // Second solve with different values (same pattern) — just update.
        let mut diag2 = diag.clone();
        for d in &mut diag2 { *d += 0.5; }
        let rhs2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let mut x2 = vec![0.0; n];
        let iters2 = amg_preconditioned_cg_cached(&mut amg, &diag2, &off_diag, &rhs2, &mut x2, 1e-3, 500);
        check_residual(&diag2, &off_diag, &x2, &rhs2, 1e-3, "cached solve 2");

        assert!(iters1 < 500 && iters2 < 500);
    }

    #[test]
    fn amg_cg_nonuniform_conductance() {
        let (n, w, h) = (100, 10, 10);
        let mut diag = vec![0.0f64; n];
        let mut off_diag = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    let j = y * w + x + 1;
                    let g = if (x + y) % 3 == 0 { 0.001 } else { 10.0 };
                    diag[i] += g; diag[j] += g;
                    off_diag.push((i, j, -g));
                }
                if y + 1 < h {
                    let j = (y + 1) * w + x;
                    let g = if (x * y) % 5 == 0 { 0.01 } else { 5.0 };
                    diag[i] += g; diag[j] += g;
                    off_diag.push((i, j, -g));
                }
            }
        }
        diag[0] += 1e6;
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut x = vec![0.0; n];
        let iters = amg_preconditioned_cg(&diag, &off_diag, &rhs, &mut x, 1e-3, 500);
        assert!(iters < 500, "got {} iters", iters);
        check_residual(&diag, &off_diag, &x, &rhs, 1e-3, "amg nonuniform");
    }
}

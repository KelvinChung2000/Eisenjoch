//! Faithful port of nextpnr HeAP's `EquationSystem`, including its solver.
//!
//! Source: upstream YosysHQ nextpnr `main` @ `4d235150`,
//! `common/place/placer_heap.cc`.
//!
//! ```text
//!  nextpnr -- Next Generation Place and Route
//!
//!  Copyright (C) 2019  gatecat <gatecat@ds0.me>
//!
//!  Permission to use, copy, modify, and/or distribute this software for any
//!  purpose with or without fee is hereby granted, provided that the above
//!  copyright notice and this permission notice appear in all copies.
//!
//!  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//!  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//!  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//!  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//!  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//!  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//!  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//! ```
//!
//! # Why this does not use `crate::solver`
//!
//! nextpnr solves the analytic placement system with
//! `Eigen::ConjugateGradient<SparseMatrix<double>, Lower|Upper>` and
//! `solveWithGuess` -- that is plain CG with Eigen's default diagonal (Jacobi)
//! preconditioner. eisenjoch's `solver/` stack is considerably stronger (AMG,
//! IC0, spectral preconditioners), and routing HeAP through it would make any
//! benchmark a comparison of *solvers* rather than of *placers*. So the solver
//! is reimplemented here, matching Eigen's iteration exactly, and stays local
//! to the HeAP module.
//!
//! Verified against a golden trace from a C++ harness linked against real Eigen
//! -- see `tests/heap_equation_system_tests.rs`.

/// `EquationSystem<T>` -- a sparse system `Ax = rhs` in the column-major form
/// nextpnr builds it in.
///
/// `A[col]` is a list of `(row, value)` **sorted by row**. The sort is not
/// cosmetic: [`Self::add_coeff`] binary-searches it.
pub struct EquationSystem {
    /// col -> (row, A[row, col]), sorted by row.
    pub a: Vec<Vec<(i32, f64)>>,
    /// The right-hand side vector.
    pub rhs: Vec<f64>,
}

impl EquationSystem {
    /// `EquationSystem(rows, cols)`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            a: vec![Vec::new(); cols],
            rhs: vec![0.0; rows],
        }
    }

    /// `reset` -- clear the matrix and zero the RHS, keeping the allocation.
    pub fn reset(&mut self) {
        for col in &mut self.a {
            col.clear();
        }
        self.rhs.fill(0.0);
    }

    /// `add_coeff` -- accumulate `val` into `A[row, col]`.
    ///
    /// Repeat calls to the same cell add up rather than overwrite; the net
    /// model relies on that, since a cell pair can be connected by several
    /// nets.
    pub fn add_coeff(&mut self, row: i32, col: i32, val: f64) {
        let ac = &mut self.a[col as usize];

        // Binary search, transcribed from the C++ so that `b` ends up as the
        // insertion point when the row is absent.
        let mut b: i32 = 0;
        let mut e: i32 = ac.len() as i32 - 1;
        while b <= e {
            let i = (b + e) / 2;
            let entry_row = ac[i as usize].0;
            if entry_row == row {
                ac[i as usize].1 += val;
                return;
            }
            if entry_row > row {
                e = i - 1;
            } else {
                b = i + 1;
            }
        }
        ac.insert(b as usize, (row, val));
    }

    /// `add_rhs` -- accumulate into the RHS.
    #[inline]
    pub fn add_rhs(&mut self, row: i32, val: f64) {
        self.rhs[row as usize] += val;
    }

    /// `solve` -- Jacobi-preconditioned conjugate gradient, warm-started from
    /// `x`, matching `Eigen::ConjugateGradient::solveWithGuess`.
    ///
    /// `tolerance` is an `f32` because nextpnr's config carries it as one and
    /// Eigen widens it to `double`; taking an `f64` here would change the
    /// convergence threshold in the last bits.
    pub fn solve(&self, x: &mut [f64], tolerance: f32) {
        if x.is_empty() {
            return;
        }
        assert_eq!(
            x.len(),
            self.a.len(),
            "solution vector must have one entry per column"
        );

        // Eigen's default for an unset maxIterations is twice the column count.
        let max_iters = 2 * self.a.len();
        conjugate_gradient(&self.a, &self.rhs, x, tolerance as f64, max_iters);
    }

    /// `A * v`, with `A` held column-major.
    fn mat_vec(a: &[Vec<(i32, f64)>], v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (col, entries) in a.iter().enumerate() {
            let vc = v[col];
            for &(row, val) in entries {
                out[row as usize] += val * vc;
            }
        }
    }
}

/// Eigen's diagonal (Jacobi) preconditioner: `M^-1 = diag(A)^-1`, with a
/// missing or zero diagonal entry falling back to 1.
fn diagonal_preconditioner(a: &[Vec<(i32, f64)>]) -> Vec<f64> {
    a.iter()
        .enumerate()
        .map(|(j, entries)| {
            match entries.iter().find(|&&(row, _)| row == j as i32) {
                Some(&(_, val)) if val != 0.0 => 1.0 / val,
                _ => 1.0,
            }
        })
        .collect()
}

/// `Eigen::internal::conjugate_gradient`.
///
/// Transcribed step for step, including the early exits and the convergence
/// test on the *squared* residual against `tol^2 * ||rhs||^2`. Reordering the
/// updates or testing the unsquared norm would still converge, just not to the
/// same iterate.
fn conjugate_gradient(
    a: &[Vec<(i32, f64)>],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) {
    let n = a.len();

    let mut residual = vec![0.0; n];
    EquationSystem::mat_vec(a, x, &mut residual);
    for i in 0..n {
        residual[i] = rhs[i] - residual[i];
    }

    let rhs_norm2: f64 = rhs.iter().map(|v| v * v).sum();
    if rhs_norm2 == 0.0 {
        x.fill(0.0);
        return;
    }

    // Eigen guards against a threshold of zero with the smallest normal double.
    let consider_as_zero = f64::MIN_POSITIVE;
    let threshold = (tol * tol * rhs_norm2).max(consider_as_zero);

    let mut residual_norm2: f64 = residual.iter().map(|v| v * v).sum();
    if residual_norm2 < threshold {
        return;
    }

    let inv_diag = diagonal_preconditioner(a);

    let mut p: Vec<f64> = (0..n).map(|i| inv_diag[i] * residual[i]).collect();
    let mut abs_new: f64 = (0..n).map(|i| residual[i] * p[i]).sum();

    let mut tmp = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut i = 0;

    while i < max_iters {
        EquationSystem::mat_vec(a, &p, &mut tmp);

        let p_dot_tmp: f64 = (0..n).map(|k| p[k] * tmp[k]).sum();
        let alpha = abs_new / p_dot_tmp;

        for k in 0..n {
            x[k] += alpha * p[k];
            residual[k] -= alpha * tmp[k];
        }

        residual_norm2 = residual.iter().map(|v| v * v).sum();
        if residual_norm2 < threshold {
            break;
        }

        for k in 0..n {
            z[k] = inv_diag[k] * residual[k];
        }

        let abs_old = abs_new;
        abs_new = (0..n).map(|k| residual[k] * z[k]).sum();
        let beta = abs_new / abs_old;

        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }

        i += 1;
    }
}

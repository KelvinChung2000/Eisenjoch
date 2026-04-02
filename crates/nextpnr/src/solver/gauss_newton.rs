//! Reusable matrix-free Gauss-Newton operator.
//!
//! This module is solver-level infrastructure and is not tied to any placer.
//! It models operators of the form:
//!   H * v = sum_k J_k^T * A^{-1} * J_k * v
//! where J_k are Jacobian blocks and A^{-1} is provided by a pluggable
//! observation-space linear solver.

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use rayon::prelude::*;
use std::cell::UnsafeCell;

/// One Jacobian block in a Gauss-Newton operator.
pub trait JacobianBlock: Sync {
    /// Scatter: `out = J_k * v` (caller clears `out` before calling).
    fn scatter(&self, v: &[f64], out: &mut [f64]);

    /// Gather transpose-add: `out += J_k^T * z`.
    fn gather_transpose_add(&self, z: &[f64], out: &mut [f64]);
}

/// Observation-space solver used by the Gauss-Newton operator.
pub trait ObservationSolver: Sync {
    /// Scratch requirement for one solve call.
    fn scratch_req(&self) -> StackReq;

    /// Solve `A * x = rhs` for x.
    fn solve(&self, rhs: &[f64], solution: &mut [f64], scratch: &mut MemBuffer);
}

struct GnWork {
    obs_rhs: Vec<f64>,
    obs_solution: Vec<f64>,
    rhs_col: Vec<f64>,
    out_col: Vec<f64>,
    scratch: Option<MemBuffer>,
}

/// Reusable matrix-free Gauss-Newton Hessian operator.
pub struct GaussNewtonOp<B, S>
where
    B: JacobianBlock,
    S: ObservationSolver,
{
    blocks: Vec<B>,
    solver: S,
    n_vars: usize,
    n_obs: usize,
    work: UnsafeCell<GnWork>,
}

impl<B, S> std::fmt::Debug for GaussNewtonOp<B, S>
where
    B: JacobianBlock,
    S: ObservationSolver,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GaussNewtonOp(n_vars={}, n_obs={}, blocks={})",
            self.n_vars,
            self.n_obs,
            self.blocks.len(),
        )
    }
}

// SAFETY: This type is used by iterative solvers that call `apply` sequentially.
unsafe impl<B, S> Sync for GaussNewtonOp<B, S>
where
    B: JacobianBlock,
    S: ObservationSolver,
{
}

impl<B, S> GaussNewtonOp<B, S>
where
    B: JacobianBlock,
    S: ObservationSolver,
{
    /// Build a reusable Gauss-Newton operator.
    pub fn new(blocks: Vec<B>, solver: S, n_vars: usize, n_obs: usize) -> Self {
        let scratch_req = solver.scratch_req();
        Self {
            blocks,
            solver,
            n_vars,
            n_obs,
            work: UnsafeCell::new(GnWork {
                obs_rhs: vec![0.0; n_obs],
                obs_solution: vec![0.0; n_obs],
                rhs_col: vec![0.0; n_vars],
                out_col: vec![0.0; n_vars],
                scratch: Some(MemBuffer::new(scratch_req)),
            }),
        }
    }

    /// Apply the operator to one vector.
    pub fn apply_to_slice(&self, v: &[f64], out: &mut [f64]) {
        struct Local {
            out: Vec<f64>,
            obs_rhs: Vec<f64>,
            obs_solution: Vec<f64>,
            scratch: MemBuffer,
        }

        let n_vars = self.n_vars;
        let n_obs = self.n_obs;
        let scratch_req = self.solver.scratch_req();

        let reduced = self
            .blocks
            .par_iter()
            .fold(
                || Local {
                    out: vec![0.0; n_vars],
                    obs_rhs: vec![0.0; n_obs],
                    obs_solution: vec![0.0; n_obs],
                    scratch: MemBuffer::new(scratch_req),
                },
                |mut local, block| {
                    local.obs_rhs.fill(0.0);
                    block.scatter(v, &mut local.obs_rhs);

                    local.obs_solution.fill(0.0);
                    self.solver
                        .solve(&local.obs_rhs, &mut local.obs_solution, &mut local.scratch);

                    block.gather_transpose_add(&local.obs_solution, &mut local.out);
                    local
                },
            )
            .reduce(
                || Local {
                    out: vec![0.0; n_vars],
                    obs_rhs: Vec::new(),
                    obs_solution: Vec::new(),
                    scratch: MemBuffer::new(StackReq::EMPTY),
                },
                |mut a, b| {
                    for (dst, &src) in a.out.iter_mut().zip(b.out.iter()) {
                        *dst += src;
                    }
                    a
                },
            );

        out.copy_from_slice(&reduced.out);
    }

    fn apply_to_slice_with_parts(
        &self,
        v: &[f64],
        out: &mut [f64],
        obs_rhs: &mut [f64],
        obs_solution: &mut [f64],
        scratch: &mut MemBuffer,
    ) {
        out.fill(0.0);
        for block in &self.blocks {
            obs_rhs.fill(0.0);
            block.scatter(v, obs_rhs);

            obs_solution.fill(0.0);
            self.solver.solve(obs_rhs, obs_solution, scratch);

            block.gather_transpose_add(obs_solution, out);
        }
    }
}

impl<B, S> LinOp<f64> for GaussNewtonOp<B, S>
where
    B: JacobianBlock,
    S: ObservationSolver,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.n_vars
    }

    fn ncols(&self) -> usize {
        self.n_vars
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let work = unsafe { &mut *self.work.get() };

        for col in 0..rhs.ncols() {
            for i in 0..self.n_vars {
                work.rhs_col[i] = rhs[(i, col)];
            }

            let scratch = work
                .scratch
                .as_mut()
                .expect("gauss-newton scratch must be initialized");
            self.apply_to_slice_with_parts(
                &work.rhs_col,
                &mut work.out_col,
                &mut work.obs_rhs,
                &mut work.obs_solution,
                scratch,
            );

            for i in 0..self.n_vars {
                out[(i, col)] = work.out_col[i];
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBlock;

    impl JacobianBlock for TestBlock {
        fn scatter(&self, v: &[f64], out: &mut [f64]) {
            // J = [1 0; 0 2], so J * v
            out[0] = v[0];
            out[1] = 2.0 * v[1];
        }

        fn gather_transpose_add(&self, z: &[f64], out: &mut [f64]) {
            // J^T * z
            out[0] += z[0];
            out[1] += 2.0 * z[1];
        }
    }

    struct IdentityObservationSolver;

    impl ObservationSolver for IdentityObservationSolver {
        fn scratch_req(&self) -> StackReq {
            StackReq::EMPTY
        }

        fn solve(&self, rhs: &[f64], solution: &mut [f64], _scratch: &mut MemBuffer) {
            solution.copy_from_slice(rhs);
        }
    }

    #[test]
    fn applies_block_sum_correctly() {
        // One block with A^{-1} = I, so H = J^T J = diag(1, 4)
        let op = GaussNewtonOp::new(vec![TestBlock], IdentityObservationSolver, 2, 2);
        let mut out = vec![0.0; 2];
        op.apply_to_slice(&[3.0, 5.0], &mut out);

        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 20.0).abs() < 1e-12);
    }
}

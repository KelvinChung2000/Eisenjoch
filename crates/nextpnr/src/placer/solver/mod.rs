//! Solver module for analytical placement.
//!
//! Provides the `Solver` trait, a Jacobi-preconditioned Conjugate Gradient
//! implementation (`SparseSystem`), Nesterov accelerated gradient descent,
//! and a multigrid V-cycle solver for grid Laplacians.

pub mod adam;
pub mod amg;
pub mod cg;
pub mod cholesky;
pub mod direct;
pub mod lse;
pub mod multigrid;
pub mod nesterov;
pub mod ordering;
pub mod sparse;
pub mod velocity;
pub mod wa;

pub use adam::AdamSolver;
pub use cg::{conjugate_gradient, dot, jacobi_preconditioner, spmv, SparseSystem};
pub use direct::DirectSolver;
pub use lse::{lse_axis_grad, lse_axis_value, lse_gradient, lse_wirelength};
pub use multigrid::MultigridSolver;
pub use nesterov::NesterovSolver;
pub use velocity::VelocityFieldSolver;
pub use wa::{wa_axis_grad, wa_axis_value, wa_wirelength};

/// Trait for linear system solvers used in analytical placement.
///
/// Implementors solve A*x = b where A is a symmetric positive-definite matrix.
pub trait Solver {
    /// Solve the system, writing the solution into `x`.
    ///
    /// Returns the number of iterations used.
    fn solve(&self, x: &mut [f64], tol: f64, max_iters: usize) -> usize;
}

//! Solver module: sparse linear algebra, preconditioners, and optimization.
//!
//! This is a top-level reusable module, not tied to any specific placer.
//! Provides:
//! - `SparseMatrix`: user-friendly sparse matrix with lazy faer CSC conversion
//! - `solve_cg`: preconditioned conjugate gradient via faer
//! - `JacobiPreconditioner` / `AmgPreconditioner`: preconditioners implementing faer Precond<f64>
//! - `FaerDirectSolver`: sparse Cholesky via faer
//! - `SparseSystemBuilder` / `Solver`: high-level system builder for HeAP etc.
//! - `NesterovSolver`: FISTA-accelerated gradient descent
//! - `wa` / `lse`: smooth wirelength approximations

pub mod cg;
pub mod direct;
pub mod lse;
pub mod optimizer;
pub mod preconditioner;
pub mod sparse_matrix;
pub mod system;
pub mod wa;

pub use cg::solve_cg;
pub use direct::FaerDirectSolver;
pub use optimizer::NesterovSolver;
pub use preconditioner::{AmgPreconditioner, JacobiPreconditioner};
pub use sparse_matrix::SparseMatrix;
pub use system::{Solver, SparseSystemBuilder};

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global thread count for faer parallel operations. Default: 8.
pub static SOLVER_THREADS: AtomicUsize = AtomicUsize::new(8);

/// Set the number of threads for solver parallelism.
pub fn set_solver_threads(n: usize) {
    SOLVER_THREADS.store(n.max(1), Ordering::Relaxed);
}

/// Get the faer Par setting using the configured thread count.
pub fn par() -> faer::Par {
    let n = SOLVER_THREADS.load(Ordering::Relaxed);
    if n <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(n)
    }
}

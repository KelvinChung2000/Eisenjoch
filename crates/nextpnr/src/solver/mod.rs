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

pub mod sparse_matrix;
pub mod cg;
pub mod preconditioner;
pub mod optimizer;
pub mod direct;
pub mod system;
pub mod wa;
pub mod lse;

pub use sparse_matrix::SparseMatrix;
pub use cg::solve_cg;
pub use preconditioner::{JacobiPreconditioner, AmgPreconditioner};
pub use direct::FaerDirectSolver;
pub use system::{SparseSystemBuilder, Solver};
pub use optimizer::NesterovSolver;

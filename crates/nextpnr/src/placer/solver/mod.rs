//! Solver module for analytical placement.
//!
//! Provides backend-swappable linear solvers (CPU: faer, future: GPU),
//! gradient optimizers, and wirelength approximation functions.

pub mod backend;
pub mod faer_backend;
pub mod optimizer;
pub mod system;
pub mod wirelength;

// Re-exports for backwards compatibility
pub use backend::{IterativeLinearSolver, LinearSolver};
pub use faer_backend::{faer_cg, FaerDirectSolver};
pub use optimizer::{AdamSolver, NesterovSolver, VelocityFieldSolver};
pub use system::{Solver, SparseSystemBuilder};
pub use wirelength::{lse_axis_grad, lse_axis_value, lse_gradient, lse_wirelength};
pub use wirelength::{wa_axis_grad, wa_axis_value, wa_wirelength};

// Module re-exports for callers that reference solver::wa or solver::lse directly.
pub use wirelength::wa;
pub use wirelength::lse;

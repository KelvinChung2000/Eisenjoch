//! Simple diagonal (Jacobi) preconditioner implementing faer Precond<f64>.

use dyn_stack::{MemStack, StackReq};

/// Diagonal preconditioner: M = diag(A), M^{-1} x = x / diag(A).
pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Build from the diagonal of the system matrix.
    pub fn new(diag: &[f64]) -> Self {
        let inv_diag = diag
            .iter()
            .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
            .collect();
        Self { inv_diag }
    }
}

impl std::fmt::Debug for JacobiPreconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JacobiPreconditioner(n={})", self.inv_diag.len())
    }
}

// SAFETY: inv_diag is immutable after construction.
unsafe impl Sync for JacobiPreconditioner {}

impl faer::matrix_free::LinOp<f64> for JacobiPreconditioner {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.inv_diag.len()
    }

    fn ncols(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        for i in 0..self.inv_diag.len() {
            out[(i, 0)] = self.inv_diag[i] * rhs[(i, 0)];
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

impl faer::matrix_free::Precond<f64> for JacobiPreconditioner {
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(&self, mut rhs: faer::MatMut<'_, f64>, _par: faer::Par, _stack: &mut MemStack) {
        for i in 0..self.inv_diag.len() {
            rhs[(i, 0)] *= self.inv_diag[i];
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

#[cfg(test)]
mod tests {
    use super::*;
    use faer::matrix_free::LinOp;

    #[test]
    fn jacobi_applies_inverse_diagonal() {
        let diag = vec![2.0, 4.0, 0.5];
        let precond = JacobiPreconditioner::new(&diag);

        let rhs_data = [6.0, 8.0, 1.0];
        let rhs = faer::MatRef::from_column_major_slice(&rhs_data, 3, 1);
        let mut out_data = [0.0; 3];
        let out = faer::MatMut::from_column_major_slice_mut(&mut out_data, 3, 1);

        let mut buf = dyn_stack::MemBuffer::new(StackReq::EMPTY);
        let mut stack = MemStack::new(&mut buf);
        precond.apply(out, rhs, faer::Par::Seq, &mut stack);

        assert!((out_data[0] - 3.0).abs() < 1e-12);
        assert!((out_data[1] - 2.0).abs() < 1e-12);
        assert!((out_data[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn jacobi_handles_near_zero_diagonal() {
        let diag = vec![0.0, 1e-15, 3.0];
        let precond = JacobiPreconditioner::new(&diag);
        // Near-zero diag should use 1.0 as inv_diag
        assert!((precond.inv_diag[0] - 1.0).abs() < 1e-12);
        assert!((precond.inv_diag[1] - 1.0).abs() < 1e-12);
    }
}

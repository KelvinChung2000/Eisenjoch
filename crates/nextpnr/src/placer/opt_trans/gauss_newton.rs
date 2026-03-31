//! Gauss-Newton operator for coupled cell-position optimization.
//!
//! Implements faer's LinOp trait so the Gauss-Newton Hessian H = J^T L^{-1} J
//! can be used directly with faer's CG solver. Each CG iteration on the
//! small 2n-cell system triggers one inner CG solve on the large pipe network.

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use rayon::prelude::*;
use std::cell::UnsafeCell;

use crate::solver::preconditioner::JacobiPreconditioner;
use crate::solver::sparse_matrix::SparseMatrixOp;

use super::config::OptTransPlacerCfg;
use super::demand::{self, NetSolveInfo};
use super::network::PipeNetwork;

/// Matrix-free Gauss-Newton Hessian: H·v = J^T · L^{-1} · J · v
///
/// - J = ∂S/∂x: demand Jacobian (how demand changes with cell positions)
/// - L: pipe network Laplacian
/// - L^{-1}: solved via inner CG
///
/// Implements faer::LinOp so it can be used with faer's CG solver.
#[allow(dead_code)]
pub struct GaussNewtonOp<'a> {
    pub net_infos: &'a [NetSolveInfo],
    pub network: &'a PipeNetwork,
    pub cfg: &'a OptTransPlacerCfg,
    pub pipe_op: &'a SparseMatrixOp,
    pub precond: &'a JacobiPreconditioner,
    pub n_cells: usize,
    pub n_nodes: usize,
    pub inner_tol: f64,
    pub inner_max_iters: usize,
    /// Workspace for inner CG (reused across apply calls).
    work: UnsafeCell<GnWork>,
}

#[allow(dead_code)]
struct GnWork {
    grid_vec: Vec<f64>, // n_nodes
    solution: Vec<f64>, // n_nodes
    rhs_col: Vec<f64>,  // 2*n_cells
    out_col: Vec<f64>,  // 2*n_cells
    buf: Option<MemBuffer>,
}

// SAFETY: GaussNewtonOp is used within faer's CG which calls apply sequentially.
unsafe impl Sync for GaussNewtonOp<'_> {}

impl std::fmt::Debug for GaussNewtonOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GaussNewtonOp(cells={}, nodes={})",
            self.n_cells, self.n_nodes
        )
    }
}

impl<'a> GaussNewtonOp<'a> {
    #[allow(dead_code)]
    pub fn new(
        net_infos: &'a [NetSolveInfo],
        network: &'a PipeNetwork,
        cfg: &'a OptTransPlacerCfg,
        pipe_op: &'a SparseMatrixOp,
        precond: &'a JacobiPreconditioner,
        n_cells: usize,
        inner_tol: f64,
        inner_max_iters: usize,
    ) -> Self {
        let n_nodes = network.num_nodes();
        let scratch_req = crate::solver::cg::cg_scratch_size(pipe_op, precond);
        Self {
            net_infos,
            network,
            cfg,
            pipe_op,
            precond,
            n_cells,
            n_nodes,
            inner_tol,
            inner_max_iters,
            work: UnsafeCell::new(GnWork {
                grid_vec: vec![0.0; n_nodes],
                solution: vec![0.0; n_nodes],
                rhs_col: vec![0.0; 2 * n_cells],
                out_col: vec![0.0; 2 * n_cells],
                buf: Some(MemBuffer::new(scratch_req)),
            }),
        }
    }
}

impl GaussNewtonOp<'_> {
    fn apply_to_slice_with_parts(
        &self,
        v: &[f64],
        out: &mut [f64],
        grid_vec: &mut [f64],
        solution: &mut [f64],
        buf: &mut MemBuffer,
    ) {
        let n = self.n_cells;
        let params = demand::GridParams::from_network(self.network);

        out.fill(0.0);

        for net_info in self.net_infos {
            // Step 1: u_k = J_k · v (single-net Jacobian)
            grid_vec.fill(0.0);
            demand::scatter_net_jacobian_vec(net_info, v, n, self.cfg, &params, grid_vec);

            // Step 2: z_k = L^{-1} · u_k
            solution.fill(0.0);
            let grid_mat = faer::MatRef::from_column_major_slice(grid_vec, self.n_nodes, 1);
            let sol_mat = faer::MatMut::from_column_major_slice_mut(solution, self.n_nodes, 1);
            crate::solver::cg::solve_cg_reuse(
                self.pipe_op,
                self.precond,
                grid_mat,
                sol_mat,
                self.inner_tol,
                self.inner_max_iters,
                buf,
            );

            // Step 3: out += J_k^T · z_k
            demand::gather_net_jacobian_transpose(net_info, solution, n, self.cfg, &params, out);
        }
    }

    fn apply_to_slice_parallel(&self, v: &[f64], out: &mut [f64]) {
        struct Local {
            out: Vec<f64>,
            grid_vec: Vec<f64>,
            solution: Vec<f64>,
            buf: MemBuffer,
        }

        let n = self.n_cells;
        let n_nodes = self.n_nodes;
        let cfg = self.cfg;
        let params = demand::GridParams::from_network(self.network);
        let scratch_req = crate::solver::cg::cg_scratch_size(self.pipe_op, self.precond);

        let reduced = self
            .net_infos
            .par_iter()
            .fold(
                || Local {
                    out: vec![0.0; 2 * n],
                    grid_vec: vec![0.0; n_nodes],
                    solution: vec![0.0; n_nodes],
                    buf: MemBuffer::new(scratch_req),
                },
                |mut local, net_info| {
                    local.grid_vec.fill(0.0);
                    demand::scatter_net_jacobian_vec(
                        net_info,
                        v,
                        n,
                        cfg,
                        &params,
                        &mut local.grid_vec,
                    );

                    local.solution.fill(0.0);
                    let grid_mat =
                        faer::MatRef::from_column_major_slice(&local.grid_vec, n_nodes, 1);
                    let sol_mat =
                        faer::MatMut::from_column_major_slice_mut(&mut local.solution, n_nodes, 1);
                    crate::solver::cg::solve_cg_reuse(
                        self.pipe_op,
                        self.precond,
                        grid_mat,
                        sol_mat,
                        self.inner_tol,
                        self.inner_max_iters,
                        &mut local.buf,
                    );

                    demand::gather_net_jacobian_transpose(
                        net_info,
                        &local.solution,
                        n,
                        cfg,
                        &params,
                        &mut local.out,
                    );
                    local
                },
            )
            .reduce(
                || Local {
                    out: vec![0.0; 2 * n],
                    grid_vec: Vec::new(),
                    solution: Vec::new(),
                    buf: MemBuffer::new(StackReq::EMPTY),
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

    /// Apply per-net Hessian: H·v = Σ_k J_k^T L^{-1} J_k v.
    ///
    /// Each net's Jacobian is applied independently, avoiding cross-net
    /// demand cancellation that plagues the combined formulation.
    #[allow(dead_code)]
    pub fn apply_to_slice(&self, v: &[f64], out: &mut [f64]) {
        self.apply_to_slice_parallel(v, out);
    }
}

impl LinOp<f64> for GaussNewtonOp<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: faer::Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        2 * self.n_cells
    }
    fn ncols(&self) -> usize {
        2 * self.n_cells
    }

    fn apply(
        &self,
        mut out: faer::MatMut<'_, f64>,
        rhs: faer::MatRef<'_, f64>,
        _par: faer::Par,
        _stack: &mut MemStack,
    ) {
        let n = self.n_cells;
        let work = unsafe { &mut *self.work.get() };

        for col in 0..rhs.ncols() {
            for i in 0..2 * n {
                work.rhs_col[i] = rhs[(i, col)];
            }
            let buf = work.buf.as_mut().unwrap();
            self.apply_to_slice_with_parts(
                &work.rhs_col,
                &mut work.out_col,
                &mut work.grid_vec,
                &mut work.solution,
                buf,
            );
            for i in 0..2 * n {
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
        self.apply(out, rhs, par, stack); // H is symmetric
    }
}

/// J · v: demand Jacobian times cell-space vector.
/// v = [δx_0..δx_{n-1}, δy_0..δy_{n-1}], output written to grid_vec.
#[allow(dead_code)]
fn jacobian_vec(
    net_infos: &[NetSolveInfo],
    v: &[f64],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    out: &mut [f64],
) {
    out.fill(0.0);
    let params = demand::GridParams::from_network(network);

    for net_info in net_infos {
        demand::scatter_net_jacobian_vec(net_info, v, n_cells, cfg, &params, out);
    }
}

/// J^T · z: transpose Jacobian times grid-space vector → cell-space.
#[allow(dead_code)]
fn jacobian_transpose_vec(
    net_infos: &[NetSolveInfo],
    z: &[f64],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let params = demand::GridParams::from_network(network);
    let mut y = vec![0.0; 2 * n_cells];

    for net_info in net_infos {
        demand::gather_net_jacobian_transpose(net_info, z, n_cells, cfg, &params, &mut y);
    }

    y
}

/// Single-net J_k · v: scatter one net's Jacobian contribution to the grid.
#[allow(dead_code)]
fn single_net_jacobian_vec(
    net_info: &NetSolveInfo,
    v: &[f64],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    out: &mut [f64],
) {
    let params = demand::GridParams::from_network(network);
    demand::scatter_net_jacobian_vec(net_info, v, n_cells, cfg, &params, out);
}

/// Single-net J_k^T · z: gather one net's transpose Jacobian, ADDING to out.
#[allow(dead_code)]
fn single_net_jacobian_transpose_add(
    net_info: &NetSolveInfo,
    z: &[f64],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
    out: &mut [f64],
) {
    let params = demand::GridParams::from_network(network);
    demand::gather_net_jacobian_transpose(net_info, z, n_cells, cfg, &params, out);
}

/// Compute the per-net energy gradient: g = Σ_k J_k^T · P_k.
///
/// Each net's pressure P_k is solved independently, avoiding cross-net
/// demand cancellation.
#[allow(dead_code)]
pub fn compute_pernet_gradient(
    net_infos: &[NetSolveInfo],
    pressures: &[Vec<f64>],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    let mut grad = vec![0.0; 2 * n_cells];
    for (net_info, pressure) in net_infos.iter().zip(pressures.iter()) {
        single_net_jacobian_transpose_add(net_info, pressure, network, n_cells, cfg, &mut grad);
    }
    grad
}

/// Compute the energy gradient: ∇E = J^T · P
/// where P = L^{-1} · S(x) is the global pressure from current demands.
#[allow(dead_code)]
pub fn compute_gradient(
    net_infos: &[NetSolveInfo],
    pressure: &[f64],
    network: &PipeNetwork,
    n_cells: usize,
    cfg: &OptTransPlacerCfg,
) -> Vec<f64> {
    jacobian_transpose_vec(net_infos, pressure, network, n_cells, cfg)
}

//! Benchmark: CG solve on a 500k-node 2D grid Laplacian.
//!
//! Builds a 4-connected grid with long-range pipes (mimicking the OT placer network)
//! and measures SpMV and full CG solve times with Jacobi preconditioner.
//!
//! Usage: cargo run --release --example bench_cg_500k -p nextpnr

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use nextpnr::solver::cg::{cg_scratch_size, solve_cg_columnwise, solve_cg_deflated};
use nextpnr::solver::preconditioner::{Ic0Preconditioner, JacobiPreconditioner};
use nextpnr::solver::sparse_matrix::{SparseMatrix, SparseMatrixOp, SparseMatrixOpRef};
use std::time::Instant;

fn main() {
    // Grid parameters matching the fine-level OT network.
    let w: usize = 532;
    let h: usize = 271;
    let n = w * h; // 144,172 nodes (coarsen=1, resolution=1)
    println!("Grid: {}x{} = {} nodes", w, h, n);

    // Build 4-connected grid Laplacian with variable conductances.
    let t0 = Instant::now();
    let mut mat = SparseMatrix::new(n);

    let idx = |x: usize, y: usize| -> usize { y * w + x };

    // 4-connected nearest neighbor pipes.
    let mut n_edges = 0usize;
    for y in 0..h {
        for x in 0..w {
            // East
            if x + 1 < w {
                let g = 1.0 + 0.5 * ((x + y) % 7) as f64; // variable conductance
                mat.add_connection(idx(x, y), idx(x + 1, y), g);
                n_edges += 1;
            }
            // South
            if y + 1 < h {
                let g = 1.0 + 0.3 * ((x + y) % 5) as f64;
                mat.add_connection(idx(x, y), idx(x, y + 1), g);
                n_edges += 1;
            }
        }
    }

    // Long-range pipes (span-4, span-12) mimicking chipdb.
    let mut n_long = 0usize;
    for y in 0..h {
        for x in 0..w {
            // Span-4 horizontal
            if x + 4 < w {
                let g = 0.5; // lower conductance for long-range
                mat.add_connection(idx(x, y), idx(x + 4, y), g);
                n_long += 1;
            }
            // Span-4 vertical
            if y + 4 < h {
                let g = 0.5;
                mat.add_connection(idx(x, y), idx(x, y + 4), g);
                n_long += 1;
            }
            // Span-12 horizontal
            if x + 12 < w {
                let g = 0.2;
                mat.add_connection(idx(x, y), idx(x + 12, y), g);
                n_long += 1;
            }
            // Span-12 vertical
            if y + 12 < h {
                let g = 0.2;
                mat.add_connection(idx(x, y), idx(x, y + 12), g);
                n_long += 1;
            }
        }
    }

    // Small diagonal shift for regularity.
    for i in 0..n {
        mat.add_diagonal(i, 1e-6);
    }

    let build_time = t0.elapsed();
    println!(
        "Matrix: {} nodes, {} NN edges, {} long-range edges, {} total off-diag",
        n,
        n_edges,
        n_long,
        mat.off_diag().len()
    );
    println!("Build time: {:.1}ms", build_time.as_secs_f64() * 1000.0);

    // Build CSC first (needs &mut mat), then create immutable ref.
    let csc_op = SparseMatrixOp::from_matrix(&mut mat);

    // --- Benchmark SpMV ---
    let op = SparseMatrixOpRef::from_matrix(&mat);
    let par = faer::Par::Seq;

    let rhs_vec: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    let rhs = faer::MatRef::from_column_major_slice(&rhs_vec, n, 1);
    let mut out_vec = vec![0.0; n];
    let mut out = faer::MatMut::from_column_major_slice_mut(&mut out_vec, n, 1);

    let scratch = op.apply_scratch(1, par);
    let mut buf = MemBuffer::new(scratch);

    // Warmup
    op.apply(
        out.as_mut(),
        rhs.as_ref(),
        par,
        MemStack::new(&mut buf),
    );

    let n_spmv = 100;
    let t1 = Instant::now();
    for _ in 0..n_spmv {
        op.apply(
            out.as_mut(),
            rhs.as_ref(),
            par,
            MemStack::new(&mut buf),
        );
    }
    let spmv_time = t1.elapsed();
    println!(
        "\nSpMV (SparseMatrixOpRef, sequential): {:.3}ms avg ({} iters, {:.1}ms total)",
        spmv_time.as_secs_f64() * 1000.0 / n_spmv as f64,
        n_spmv,
        spmv_time.as_secs_f64() * 1000.0
    );

    // SpMV with parallel
    let par_p = faer::Par::rayon(8);
    op.apply(
        out.as_mut(),
        rhs.as_ref(),
        par_p,
        MemStack::new(&mut buf),
    );
    let t1p = Instant::now();
    for _ in 0..n_spmv {
        op.apply(
            out.as_mut(),
            rhs.as_ref(),
            par_p,
            MemStack::new(&mut buf),
        );
    }
    let spmv_time_par = t1p.elapsed();
    println!(
        "SpMV (SparseMatrixOpRef, par=8):      {:.3}ms avg ({} iters, {:.1}ms total)",
        spmv_time_par.as_secs_f64() * 1000.0 / n_spmv as f64,
        n_spmv,
        spmv_time_par.as_secs_f64() * 1000.0
    );

    // Common CG parameters.
    let mut rhs_cg: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0037).sin()).collect();
    let mean = rhs_cg.iter().sum::<f64>() / n as f64;
    for v in &mut rhs_cg {
        *v -= mean;
    }
    let tol = 1e-2;
    let max_iters = 500;
    let n_solves = 10;

    // --- Benchmark SpMV with faer CSC ---
    let csc_scratch = csc_op.apply_scratch(1, par);
    let mut csc_buf = MemBuffer::new(csc_scratch);

    // Warmup
    {
        let rhs = faer::MatRef::from_column_major_slice(&rhs_vec, n, 1);
        let mut out_vec2 = vec![0.0; n];
        let out = faer::MatMut::from_column_major_slice_mut(&mut out_vec2, n, 1);
        csc_op.apply(out, rhs, par, MemStack::new(&mut csc_buf));
    }

    let t_csc = Instant::now();
    for _ in 0..n_spmv {
        let rhs = faer::MatRef::from_column_major_slice(&rhs_vec, n, 1);
        let mut out_vec2 = vec![0.0; n];
        let out = faer::MatMut::from_column_major_slice_mut(&mut out_vec2, n, 1);
        csc_op.apply(out, rhs, par, MemStack::new(&mut csc_buf));
    }
    let csc_time = t_csc.elapsed();
    println!(
        "SpMV (faer CSC, sequential):          {:.3}ms avg ({} iters, {:.1}ms total)",
        csc_time.as_secs_f64() * 1000.0 / n_spmv as f64,
        n_spmv,
        csc_time.as_secs_f64() * 1000.0
    );

    let t_csc_p = Instant::now();
    for _ in 0..n_spmv {
        let rhs = faer::MatRef::from_column_major_slice(&rhs_vec, n, 1);
        let mut out_vec2 = vec![0.0; n];
        let out = faer::MatMut::from_column_major_slice_mut(&mut out_vec2, n, 1);
        csc_op.apply(out, rhs, par_p, MemStack::new(&mut csc_buf));
    }
    let csc_time_p = t_csc_p.elapsed();
    println!(
        "SpMV (faer CSC, par=8):               {:.3}ms avg ({} iters, {:.1}ms total)",
        csc_time_p.as_secs_f64() * 1000.0 / n_spmv as f64,
        n_spmv,
        csc_time_p.as_secs_f64() * 1000.0
    );

    // --- CG solve with faer CSC ---
    let precond_csc = JacobiPreconditioner::new(mat.diag());
    {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let _ = solve_cg_deflated(
            &csc_op,
            &precond_csc,
            rhs_mat.as_ref(),
            x_mat,
            tol,
            max_iters,
        );
    }
    let t_cg_csc = Instant::now();
    let mut total_iters_csc = 0;
    for _ in 0..n_solves {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let result = solve_cg_deflated(
            &csc_op,
            &precond_csc,
            rhs_mat.as_ref(),
            x_mat,
            tol,
            max_iters,
        );
        total_iters_csc += result.iterations;
    }
    let cg_csc_time = t_cg_csc.elapsed();
    let avg_iters_csc = total_iters_csc as f64 / n_solves as f64;
    println!(
        "\nCG solve (Jacobi, faer CSC, seq): {:.1}ms avg, {:.1} iters avg ({} solves, {:.1}ms total)",
        cg_csc_time.as_secs_f64() * 1000.0 / n_solves as f64,
        avg_iters_csc,
        n_solves,
        cg_csc_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Per CG iteration: {:.3}ms",
        cg_csc_time.as_secs_f64() * 1000.0 / total_iters_csc as f64
    );

    // --- Benchmark full CG solve (triplet, Jacobi preconditioner) ---
    let precond = JacobiPreconditioner::new(mat.diag());

    // Warmup solve
    {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let _ = solve_cg_deflated(&op, &precond, rhs_mat.as_ref(), x_mat, tol, max_iters);
    }

    let t2 = Instant::now();
    let mut total_iters = 0;
    for _ in 0..n_solves {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let result = solve_cg_deflated(&op, &precond, rhs_mat.as_ref(), x_mat, tol, max_iters);
        total_iters += result.iterations;
    }
    let cg_time = t2.elapsed();
    let avg_iters = total_iters as f64 / n_solves as f64;
    println!(
        "\nCG solve (Jacobi, tol={}, seq): {:.1}ms avg, {:.1} iters avg ({} solves, {:.1}ms total)",
        tol,
        cg_time.as_secs_f64() * 1000.0 / n_solves as f64,
        avg_iters,
        n_solves,
        cg_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Per CG iteration: {:.3}ms",
        cg_time.as_secs_f64() * 1000.0 / total_iters as f64
    );

    // --- Estimate: 300 solves (like OT gradient) ---
    let est_300 = cg_time.as_secs_f64() / n_solves as f64 * 300.0;
    println!(
        "\nEstimated 300 solves (1 OT gradient, Jacobi): {:.1}s",
        est_300
    );
    println!(
        "Estimated 20 OT iterations (Jacobi): {:.1}s",
        est_300 * 20.0
    );

    // --- CG solve with IC(0) preconditioner ---
    let t_ic0_build = Instant::now();
    let ic0_precond = Ic0Preconditioner::new(mat.diag(), mat.off_diag(), 0.95);
    let ic0_build_time = t_ic0_build.elapsed();
    println!(
        "\nIC(0) preconditioner build: {:.1}ms",
        ic0_build_time.as_secs_f64() * 1000.0
    );

    // Warmup
    {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let scratch_req = cg_scratch_size(&csc_op, &ic0_precond);
        let mut buf = MemBuffer::new(scratch_req);
        let _ = solve_cg_columnwise(
            &csc_op,
            &ic0_precond,
            rhs_mat,
            x_mat,
            tol,
            max_iters,
            faer::Par::Seq,
            &mut buf,
        );
    }

    let t_ic0 = Instant::now();
    let mut total_iters_ic0 = 0;
    let scratch_req = cg_scratch_size(&csc_op, &ic0_precond);
    let mut buf_ic0 = MemBuffer::new(scratch_req);
    for _ in 0..n_solves {
        let rhs_mat = faer::MatRef::from_column_major_slice(&rhs_cg, n, 1);
        let mut x_vec = vec![0.0; n];
        let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_vec, n, 1);
        let result = solve_cg_columnwise(
            &csc_op,
            &ic0_precond,
            rhs_mat,
            x_mat,
            tol,
            max_iters,
            faer::Par::Seq,
            &mut buf_ic0,
        );
        total_iters_ic0 += result.iterations;
    }
    let ic0_time = t_ic0.elapsed();
    let avg_iters_ic0 = total_iters_ic0 as f64 / n_solves as f64;
    println!(
        "CG solve (IC0, tol={}, seq): {:.1}ms avg, {:.1} iters avg ({} solves, {:.1}ms total)",
        tol,
        ic0_time.as_secs_f64() * 1000.0 / n_solves as f64,
        avg_iters_ic0,
        n_solves,
        ic0_time.as_secs_f64() * 1000.0
    );
    if total_iters_ic0 > 0 {
        println!(
            "  Per CG iteration: {:.3}ms",
            ic0_time.as_secs_f64() * 1000.0 / total_iters_ic0 as f64
        );
    }
    let est_300_ic0 = ic0_time.as_secs_f64() / n_solves as f64 * 300.0;
    println!(
        "\nEstimated 300 solves (1 OT gradient, IC0): {:.1}s",
        est_300_ic0
    );
    println!(
        "Estimated 20 OT iterations (IC0): {:.1}s",
        est_300_ic0 * 20.0
    );

    // --- Speedup summary ---
    println!("\n--- Summary ---");
    println!(
        "IC0 vs Jacobi: {:.1}x fewer iters, {:.1}x faster per solve",
        avg_iters / avg_iters_ic0.max(1.0),
        (cg_time.as_secs_f64() / n_solves as f64)
            / (ic0_time.as_secs_f64() / n_solves as f64).max(1e-12)
    );
}

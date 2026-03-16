# Parallel Placer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the HeAP and SA placers using rayon, and extract the CG solver into a dedicated module with a trait for future solver variants.

**Architecture:** Extract `SparseSystem`, `spmv`, and `conjugate_gradient` into `src/placer/solver/` behind a `Solver` trait. Add Jacobi preconditioning. Use `rayon::join` for parallel X/Y solves, `par_iter` for HPWL computation, and parallel nearest-BEL search in legalize. Cache bucket BELs in SA to avoid repeated collection.

**Tech Stack:** Rust, rayon (already in workspace dependencies)

---

## Chunk 1: Extract Solver Module

### Task 1: Create solver module structure

**Files:**
- Create: `crates/nextpnr/src/placer/solver/mod.rs`
- Create: `crates/nextpnr/src/placer/solver/cg.rs`
- Modify: `crates/nextpnr/src/placer/mod.rs`
- Modify: `crates/nextpnr/src/placer/heap.rs`

- [ ] **Step 1: Create `solver/mod.rs` with Solver trait and re-exports**

```rust
// crates/nextpnr/src/placer/solver/mod.rs

pub mod cg;

pub use cg::{conjugate_gradient, spmv, SparseSystem};

/// Trait for linear system solvers.
///
/// Solves A*x = b where A is symmetric positive-definite.
pub trait Solver {
    /// Solve the system, writing the solution into `x`.
    /// Returns the number of iterations used (0 for direct solvers).
    fn solve(&self, x: &mut [f64], tolerance: f64, max_iters: usize) -> usize;
}
```

- [ ] **Step 2: Move solver code to `solver/cg.rs`**

Move from `heap.rs` into `solver/cg.rs`:
- `SparseSystem` struct and its `impl` block (lines 88-151)
- `spmv` function (lines 162-171)
- `dot` function (lines 174-176)
- `conjugate_gradient` function (lines 181-256)

The `SparseSystem` should implement the `Solver` trait:

```rust
// crates/nextpnr/src/placer/solver/cg.rs

use super::Solver;

// ... (SparseSystem, spmv, dot, conjugate_gradient moved here unchanged) ...

impl Solver for SparseSystem {
    fn solve(&self, x: &mut [f64], tolerance: f64, max_iters: usize) -> usize {
        conjugate_gradient(
            &self.diag,
            &self.off_diag,
            &self.rhs,
            x,
            tolerance,
            max_iters,
        )
    }
}
```

- [ ] **Step 3: Update `placer/mod.rs` to declare solver module**

```rust
// crates/nextpnr/src/placer/mod.rs
pub mod common;
pub mod heap;
pub mod sa;
pub mod solver;
// ... rest unchanged ...
```

- [ ] **Step 4: Update `heap.rs` imports to use solver module**

Replace direct `SparseSystem`/`spmv`/`conjugate_gradient` definitions with imports:

```rust
use super::solver::SparseSystem;
```

Remove the old `SparseSystem`, `spmv`, `dot`, and `conjugate_gradient` code from `heap.rs`.

- [ ] **Step 5: Update test imports**

In `tests/heap_internal_tests.rs`, update imports:

```rust
// Old:
use nextpnr::placer::heap::{conjugate_gradient, spmv, SparseSystem};

// New:
use nextpnr::placer::solver::{conjugate_gradient, spmv, SparseSystem};
```

Keep HeAP-specific imports (`count_bels_in_region`, `place_heap`, `HeapState`, `PlacerHeapCfg`) pointing at `heap`.

- [ ] **Step 6: Run tests to verify refactor is behavior-preserving**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All test binaries pass with 0 failures.

- [ ] **Step 7: Commit**

```bash
git add crates/nextpnr/src/placer/solver/ crates/nextpnr/src/placer/mod.rs crates/nextpnr/src/placer/heap.rs crates/nextpnr/tests/heap_internal_tests.rs
git commit -m "refactor: extract CG solver into placer/solver/ module with Solver trait"
```

---

## Chunk 2: Jacobi Preconditioning

### Task 2: Add Jacobi preconditioner to CG solver

**Files:**
- Modify: `crates/nextpnr/src/placer/solver/cg.rs`

Jacobi preconditioning uses `M = diag(A)` as preconditioner. In preconditioned CG, the residual is scaled by `M^{-1}` each iteration. This typically halves iteration count for placement matrices.

- [ ] **Step 1: Write test for preconditioned CG**

Add to `tests/heap_internal_tests.rs`:

```rust
#[test]
fn cg_jacobi_preconditioned_fewer_iters() {
    // Same system as cg_with_off_diagonal but verify fewer iterations
    // with preconditioning enabled.
    let mut sys = SparseSystem::new(2);
    sys.add_connection(0, 1, 1.0);
    sys.add_anchor(0, 0.0, 3.0);
    sys.add_anchor(1, 4.0, 3.0);
    let mut x = vec![0.0; 2];
    let iters = sys.solve(&mut x, 1e-10, 100);
    // System: [[4,-1],[-1,4]] x = [0, 12]
    // Solution: x = [0.8, 3.2]
    assert!((x[0] - 0.8).abs() < 1e-6);
    assert!((x[1] - 3.2).abs() < 1e-6);
    assert!(iters <= 2, "preconditioned CG should converge in <=2 iters, got {}", iters);
}
```

- [ ] **Step 2: Run test to verify it fails (or check current iteration count)**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils cg_jacobi_preconditioned -v`

- [ ] **Step 3: Implement Jacobi preconditioning in `conjugate_gradient`**

In `solver/cg.rs`, modify `conjugate_gradient` to apply diagonal preconditioning:

```rust
pub fn conjugate_gradient(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }

    // Jacobi preconditioner: M^{-1} = diag(1/A[i,i])
    let inv_diag: Vec<f64> = diag.iter().map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }).collect();

    // r = b - A*x
    let mut ax = vec![0.0; n];
    spmv(diag, off_diag, x, &mut ax);
    let mut r: Vec<f64> = rhs.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // z = M^{-1} * r
    let mut z: Vec<f64> = r.iter().zip(inv_diag.iter()).map(|(ri, mi)| ri * mi).collect();

    // p = z
    let mut p = z.clone();

    let mut rz_old = dot(&r, &z);

    // Convergence check on unpreconditioned residual.
    let rhs_norm_sq = dot(rhs, rhs);
    let tol_sq = tol * tol * rhs_norm_sq.max(1e-30);

    if dot(&r, &r) < tol_sq {
        return 0;
    }

    let mut ap = vec![0.0; n];

    for iter in 0..max_iters {
        // ap = A*p
        spmv(diag, off_diag, &p, &mut ap);

        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            return iter + 1;
        }

        let alpha = rz_old / p_ap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - alpha * A*p
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        // Check convergence on unpreconditioned residual.
        let rs_new = dot(&r, &r);
        if rs_new < tol_sq {
            return iter + 1;
        }

        // z = M^{-1} * r
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    max_iters
}
```

- [ ] **Step 4: Run all tests to verify correctness**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All tests pass. Existing solver tests still produce correct solutions.

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/solver/cg.rs crates/nextpnr/tests/heap_internal_tests.rs
git commit -m "feat: add Jacobi preconditioning to CG solver"
```

---

## Chunk 3: Parallel HeAP

### Task 3: Parallel X/Y solve with rayon::join

**Files:**
- Modify: `crates/nextpnr/src/placer/heap.rs`

- [ ] **Step 1: Replace sequential X/Y solve with `rayon::join`**

In `HeapState::solve_analytical()`, replace lines 539-549:

```rust
// Old:
let iters_x = sys_x.solve(&mut self.cell_x, ...);
let iters_y = sys_y.solve(&mut self.cell_y, ...);

// New:
let tol = self.cfg.solver_tolerance;
let max = self.cfg.max_solver_iters;
let (iters_x, iters_y) = rayon::join(
    || sys_x.solve(&mut cell_x_clone, tol, max),
    || sys_y.solve(&mut cell_y_clone, tol, max),
);
```

Because `rayon::join` requires both closures to have `&mut` to different data, we need to split the borrows. `self.cell_x` and `self.cell_y` are separate `Vec<f64>` fields, so we can borrow them independently by destructuring:

```rust
let cell_x = &mut self.cell_x;
let cell_y = &mut self.cell_y;
let tol = self.cfg.solver_tolerance;
let max_iters = self.cfg.max_solver_iters;
let (iters_x, iters_y) = rayon::join(
    || sys_x.solve(cell_x, tol, max_iters),
    || sys_y.solve(cell_y, tol, max_iters),
);
```

- [ ] **Step 2: Run tests to verify behavior is preserved**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All tests pass. HeAP placement results are identical (CG is deterministic).

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/heap.rs
git commit -m "feat: parallel X/Y analytical solve via rayon::join"
```

### Task 4: Parallel legalize nearest-BEL search

**Files:**
- Modify: `crates/nextpnr/src/placer/heap.rs`

The legalize phase currently iterates cells sequentially, and for each cell scans all BELs of matching type to find the nearest available one. The search (distance computation) is read-only. We can compute best-BEL candidates in parallel, then assign sequentially.

- [ ] **Step 1: Write test for parallel legalize correctness**

Add to `tests/heap_internal_tests.rs`:

```rust
#[test]
fn legalize_parallel_matches_sequential() {
    // Verify that legalize produces valid placement after parallel search.
    let mut ctx = common::make_context_with_cells(4);
    let cfg = PlacerHeapCfg {
        seed: 42,
        max_iterations: 3,
        ..PlacerHeapCfg::default()
    };
    place_heap(&mut ctx, &cfg).unwrap();
    // All cells should be placed at unique BELs.
    let mut used = std::collections::HashSet::new();
    for cell in ctx.cells() {
        if cell.is_alive() {
            let bel = cell.bel_id().expect("cell should be placed");
            assert!(used.insert(bel), "duplicate BEL assignment");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it passes with current code**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils legalize_parallel -v`

- [ ] **Step 3: Implement parallel nearest-BEL search**

Restructure `legalize()` to:
1. Unbind all movable cells (same as before).
2. Sort cells by distance from center (same as before).
3. For each cell, use `rayon` to find the nearest available BEL in parallel. But since BEL availability changes as we assign cells, we do a two-phase approach:
   - Phase A: Compute sorted candidate BELs for each cell in parallel (sort by distance, all BELs of type).
   - Phase B: Assign sequentially, skipping BELs already taken.

```rust
fn legalize(&mut self, ctx: &mut Context) -> Result<(), PlacerError> {
    use rayon::prelude::*;

    let n = self.movable_cells.len();
    if n == 0 {
        return Ok(());
    }

    // Unbind all movable cells.
    for &cell_idx in &self.movable_cells {
        let cell = ctx.cell(cell_idx);
        if let Some(bel) = cell.bel() {
            let bel = bel.id();
            ctx.unbind_bel(bel);
        }
    }

    // Sort by distance from center (outer cells first).
    let cx = (self.grid_w as f64 - 1.0) / 2.0;
    let cy = (self.grid_h as f64 - 1.0) / 2.0;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let da = (self.cell_x[a] - cx).powi(2) + (self.cell_y[a] - cy).powi(2);
        let db = (self.cell_x[b] - cx).powi(2) + (self.cell_y[b] - cy).powi(2);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase A: For each cell, collect candidate BELs sorted by distance (parallel).
    // We need cell type, target position, and region per cell.
    struct CellLegalizeInfo {
        cell_idx: CellId,
        cell_type_id: IdString,
        cell_type_name: String,
        cell_name: String,
        target_x: f64,
        target_y: f64,
        region: Option<u32>,
        heap_idx: usize,
    }

    let infos: Vec<CellLegalizeInfo> = order
        .iter()
        .map(|&idx| {
            let cell_idx = self.movable_cells[idx];
            let cell = ctx.cell(cell_idx);
            CellLegalizeInfo {
                cell_idx,
                cell_type_id: cell.cell_type_id(),
                cell_type_name: cell.cell_type().to_owned(),
                cell_name: cell.name().to_owned(),
                target_x: self.cell_x[idx],
                target_y: self.cell_y[idx],
                region: self.cell_region[idx],
                heap_idx: idx,
            }
        })
        .collect();

    // Collect all BEL locations once (read-only chipdb access).
    // Build per-type sorted candidate lists in parallel.
    let candidates: Vec<Vec<BelId>> = infos
        .par_iter()
        .map(|info| {
            let mut bels_with_dist: Vec<(BelId, f64)> = ctx
                .bels_for_bucket(info.cell_type_id)
                .filter_map(|bel| {
                    if let Some(rid) = info.region {
                        if !ctx.is_bel_in_region(bel.id(), rid) {
                            return None;
                        }
                    }
                    let loc = bel.loc();
                    let dx = loc.x as f64 - info.target_x;
                    let dy = loc.y as f64 - info.target_y;
                    Some((bel.id(), dx * dx + dy * dy))
                })
                .collect();
            bels_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            bels_with_dist.into_iter().map(|(bel, _)| bel).collect()
        })
        .collect();

    // Phase B: Assign sequentially using pre-sorted candidates.
    for (i, info) in infos.iter().enumerate() {
        let mut placed = false;
        for &bel in &candidates[i] {
            if ctx.bel(bel).is_available() {
                if !ctx.bind_bel(bel, info.cell_idx, PlaceStrength::Placer) {
                    return Err(PlacerError::PlacementFailed(format!(
                        "Failed to bind cell {} to BEL {}", info.cell_name, bel,
                    )));
                }
                self.cell_x[info.heap_idx] = ctx.bel(bel).loc().x as f64;
                self.cell_y[info.heap_idx] = ctx.bel(bel).loc().y as f64;
                placed = true;
                break;
            }
        }

        if !placed {
            if candidates[i].is_empty() {
                return Err(PlacerError::NoBelsAvailable(info.cell_type_name.clone()));
            }
            return Err(PlacerError::NoBelsAvailable(format!(
                "{} (no available BELs for cell {})", info.cell_type_name, info.cell_name,
            )));
        }
    }

    Ok(())
}
```

Note: `ctx.bels_for_bucket()` returns an iterator over BelView which borrows `ctx` immutably. Since `par_iter` needs `Send` bounds, we may need to collect BEL data (id + location) into a plain struct first. If `BelView` is not `Send`, pre-collect per-type BEL lists before the parallel section. Adapt as needed during implementation.

- [ ] **Step 4: Run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/heap.rs crates/nextpnr/tests/heap_internal_tests.rs
git commit -m "feat: parallel nearest-BEL search in HeAP legalize phase"
```

---

## Chunk 4: Parallel HPWL and SA Improvements

### Task 5: Parallel total_hpwl

**Files:**
- Modify: `crates/nextpnr/src/placer/common.rs`

- [ ] **Step 1: Add parallel total_hpwl function**

The existing `total_hpwl` iterates nets sequentially. Add a parallel variant and use it where beneficial. Since `Context` needs to be shared across threads, and `net_hpwl` only reads immutably, this should work with `par_iter` if we can collect net indices first.

```rust
use rayon::prelude::*;

/// Compute total HPWL cost across all alive nets (parallel).
pub fn total_hpwl(ctx: &Context) -> f64 {
    let net_indices: Vec<NetId> = ctx
        .design
        .iter_alive_nets()
        .map(|(idx, _)| idx)
        .collect();
    net_indices
        .par_iter()
        .map(|&idx| net_hpwl(ctx, idx))
        .sum()
}
```

Note: This requires `Context` to be `Sync`. If it is not (due to interior mutability), collect net info into a plain vec first and compute in parallel. Check during implementation and adapt.

If `Context` is not `Sync`, use the sequential version but parallelize the inner loop differently. The key insight: `net_hpwl` only reads `ctx.net()` and `ctx.cell()`, which are pure reads from arena Vecs. If `Context` implements `Sync`, this works directly.

- [ ] **Step 2: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/common.rs
git commit -m "feat: parallel total_hpwl via rayon par_iter"
```

### Task 6: SA bucket BEL caching

**Files:**
- Modify: `crates/nextpnr/src/placer/sa.rs`

Currently the SA inner loop calls `ctx.bels_for_bucket(cell_type).filter(...).collect()` on every move attempt. This rebuilds a `Vec<BelId>` thousands of times per iteration. Pre-cache per cell type.

- [ ] **Step 1: Pre-cache bucket BELs before SA loop**

In `place_sa()`, after `initial_placement`, build cache:

```rust
use rustc_hash::FxHashMap;
use crate::chipdb::BelId;

// Pre-cache BELs by (cell_type, region) for fast random selection.
// Key: cell_type IdString. Value: Vec<BelId> (all BELs of this type).
let bucket_cache: FxHashMap<IdString, Vec<BelId>> = {
    let mut cache = FxHashMap::default();
    for &ci in &moveable {
        let cell = ctx.cell(ci);
        let ct = cell.cell_type_id();
        cache.entry(ct).or_insert_with(|| {
            ctx.bels_for_bucket(ct).map(|b| b.id()).collect()
        });
    }
    cache
};
```

Then in the inner loop, replace:

```rust
// Old:
let bucket_bels: Vec<_> = ctx
    .bels_for_bucket(cell_type)
    .filter(|bel| { ... })
    .map(|bel| bel.id())
    .collect();

// New:
let all_bels = match bucket_cache.get(&cell_type) {
    Some(bels) => bels,
    None => continue,
};
// For region-constrained cells, filter inline (cheap since we just check membership).
let target_bel = if let Some(rid) = cell_region {
    let filtered: Vec<_> = all_bels.iter().copied()
        .filter(|&b| ctx.is_bel_in_region(b, rid))
        .collect();
    if filtered.is_empty() { continue; }
    filtered[ctx.rng_mut().next_range(filtered.len() as u32) as usize]
} else {
    if all_bels.is_empty() { continue; }
    all_bels[ctx.rng_mut().next_range(all_bels.len() as u32) as usize]
};
```

- [ ] **Step 2: Parallel `hpwl_for_nets` for large net lists**

In `sa.rs`, update `hpwl_for_nets` to use `par_iter` when the net list is large enough:

```rust
use rayon::prelude::*;

fn hpwl_for_nets(ctx: &Context, net_indices: &[NetId]) -> f64 {
    if net_indices.len() > 16 {
        net_indices.par_iter().map(|&idx| net_hpwl(ctx, idx)).sum()
    } else {
        net_indices.iter().map(|&idx| net_hpwl(ctx, idx)).sum()
    }
}
```

The threshold of 16 avoids rayon overhead for small net lists (most SA moves affect <10 nets). Adjust during implementation if needed.

Note: Same `Sync` requirement on `Context` as Task 5. If `Context` is not `Sync`, skip this parallel variant and keep sequential.

- [ ] **Step 3: Run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils 2>&1 | grep "test result"`
Expected: All tests pass. SA determinism test may need the same seed to produce the same result since the algorithm behavior is unchanged (caching doesn't change which BELs are selected).

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/placer/sa.rs
git commit -m "feat: cache bucket BELs and parallelize HPWL in SA placer"
```

---

## Chunk 5: Verification and Cleanup

### Task 7: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils`
Expected: All 588+ tests pass.

- [ ] **Step 2: Run with release mode to verify no debug-only issues**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils --release`
Expected: All tests pass.

- [ ] **Step 3: Check for compiler warnings**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p nextpnr --features test-utils 2>&1 | grep -E "warning"`
Expected: No warnings.

- [ ] **Step 4: Verify determinism**

The HeAP and SA placers have determinism tests (`full_heap_deterministic`, `full_sa_deterministic`). These must still pass since parallelism was introduced only in commutative operations (sum, independent solves, distance computation). Verify:

Run: `TMPDIR=/tmp/claude-1000 cargo test -p nextpnr --features test-utils deterministic -v`
Expected: Both determinism tests pass.

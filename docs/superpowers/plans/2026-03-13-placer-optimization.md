# Placer Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align ElectroPlace with upstream `placer_static.cc` (WA wirelength, DCT density, growing penalty, overlap convergence) and speed up the Hydraulic placer (gradient clipping, grid cropping, multigrid preconditioner).

**Architecture:** Four independent workstreams. Workstream 2 (hydraulic quick wins) goes first as it's smallest. Workstream 1 (ElectroPlace rewrite) is largest. Workstreams 3-4 (grid cropping, multigrid) are hydraulic solver speedups.

**Tech Stack:** Rust, `rustdct` crate (new), `rustfft` (existing), `rayon` (existing), `lapjv`/`ndarray` (existing).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/nextpnr/src/placer/solver/wa.rs` | **Create** | Weighted-Average wirelength model (value + gradient) |
| `crates/nextpnr/src/placer/solver/mod.rs` | Modify | Add `pub mod wa` and re-exports |
| `crates/nextpnr/src/placer/electro_place/config.rs` | Modify | Replace gamma params with wl_coeff, target_util, timing_driven |
| `crates/nextpnr/src/placer/electro_place/density.rs` | Rewrite | DCT-II Poisson solve + slither cell-to-bin mapping |
| `crates/nextpnr/src/placer/electro_place/mod.rs` | Rewrite | WA wirelength, growing penalty, overlap convergence, BB step, spacers |
| `crates/nextpnr/src/placer/common.rs` | Modify | Add `add_wa_wirelength_gradient()`, keep LSE for other users |
| `crates/nextpnr/src/placer/hydraulic_place/state.rs` | Modify | Add `clip_gradients()` method |
| `crates/nextpnr/src/placer/hydraulic_place/mod.rs` | Modify | Exponential turbulence ramp, line estimate convergence, step floor |
| `crates/nextpnr/src/placer/hydraulic_place/kirchhoff.rs` | Modify | Dynamic grid cropping, multigrid preconditioner option |
| `crates/nextpnr/src/placer/solver/cg.rs` | Modify | Add multigrid-preconditioned CG variant |
| `crates/nextpnr/Cargo.toml` | Modify | Add `rustdct` dependency |

---

## Chunk 1: Workstream 2 - Hydraulic Quick Wins

### Task 1: Gradient Clipping

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/state.rs`
- Modify: `crates/nextpnr/src/placer/hydraulic_place/mod.rs`

- [ ] **Step 1: Add `clip_gradients` method to `HydraulicState`**

In `state.rs`, add after `compute_pressure_gradient()`:

```rust
/// Clip per-cell gradient magnitudes to median * clip_factor.
///
/// Prevents a few high-pressure cells from dominating the Nesterov step.
pub fn clip_gradients(grad_x: &mut [f64], grad_y: &mut [f64], clip_factor: f64) {
    let n = grad_x.len();
    if n == 0 {
        return;
    }

    // Compute gradient magnitudes
    let mut magnitudes: Vec<f64> = (0..n)
        .map(|i| (grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]).sqrt())
        .collect();

    // Find median magnitude
    magnitudes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = magnitudes[n / 2];

    let max_grad = median * clip_factor;
    if max_grad < 1e-30 {
        return;
    }

    // Clip gradients exceeding the threshold
    for i in 0..n {
        let mag = (grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]).sqrt();
        if mag > max_grad {
            let scale = max_grad / mag;
            grad_x[i] *= scale;
            grad_y[i] *= scale;
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles successfully

- [ ] **Step 3: Integrate gradient clipping into main loop**

In `mod.rs`, after line 124 (`let (mut grad_x, mut grad_y) = state.compute_pressure_gradient();`), add:

```rust
state::HydraulicState::clip_gradients(&mut grad_x, &mut grad_y, 10.0);
```

- [ ] **Step 4: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles successfully

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/state.rs crates/nextpnr/src/placer/hydraulic_place/mod.rs
git commit -m "feat(hydraulic): add gradient clipping (median * 10)"
```

### Task 2: Exponential Turbulence Ramp

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/mod.rs`

- [ ] **Step 1: Replace linear ramp with exponential**

In `mod.rs`, replace lines 102-104:
```rust
// Turbulence ramp: 0 -> turbulence_beta over the first half of iterations.
let ramp = (2.0 * iter as f64 / cfg.max_outer_iters as f64).min(1.0);
let beta = cfg.turbulence_beta * ramp;
```

With:
```rust
// Exponential turbulence ramp: reaches ~95% of beta_max by end.
let beta = cfg.turbulence_beta * (1.0 - (-3.0 * iter as f64 / cfg.max_outer_iters as f64).exp());
```

- [ ] **Step 2: Verify compilation and test**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles successfully

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/mod.rs
git commit -m "feat(hydraulic): exponential turbulence ramp"
```

### Task 3: Line Estimate Convergence + Step Floor

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/mod.rs`

- [ ] **Step 1: Replace HPWL with line estimate for convergence**

In `mod.rs`, replace the convergence block (lines 156-186). Change the metric from `total_hpwl` to `total_line_estimate` and track best line estimate instead of best HPWL:

Replace:
```rust
let hpwl = crate::metrics::wirelength::total_hpwl(ctx);
loop_state.record_hpwl(hpwl, &state.cell_x, &state.cell_y);
```

With:
```rust
let line_est = crate::metrics::wirelength::total_line_estimate(ctx);
loop_state.record_hpwl(line_est, &state.cell_x, &state.cell_y);
```

Update the eprintln to show `line_est` instead of `hpwl`. Update the divergence/convergence checks to use `line_est`.

- [ ] **Step 2: Add step size floor**

In `mod.rs`, after `loop_state.update_step_sizes(...)` (line ~141), add:

```rust
// Enforce minimum step size (1% of initial) to prevent step collapse.
let step_floor = cfg.nesterov_step_size * 0.01;
if state.nesterov_x.step_size() < step_floor {
    state.nesterov_x.set_step_size(step_floor);
}
if state.nesterov_y.step_size() < step_floor {
    state.nesterov_y.set_step_size(step_floor);
}
```

- [ ] **Step 3: Remove adaptive restart calls**

In `mod.rs`, delete lines 150-151:
```rust
state.nesterov_x.adaptive_restart(&grad_x);
state.nesterov_y.adaptive_restart(&grad_y);
```

- [ ] **Step 4: Verify compilation and test**

Run: `cargo build -p nextpnr && cargo test -p nextpnr --features test-utils 2>&1 | tail -10`
Expected: compiles, all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/mod.rs
git commit -m "feat(hydraulic): line estimate convergence, step floor, remove adaptive restart"
```

---

## Chunk 2: Workstream 1a-1c - ElectroPlace WA Wirelength + DCT Density

### Task 4: Add `rustdct` Dependency

**Files:**
- Modify: `crates/nextpnr/Cargo.toml`

- [ ] **Step 1: Add rustdct to dependencies**

Add to `[dependencies]` section:
```toml
rustdct = "0.7"
```

- [ ] **Step 2: Verify dependency resolves**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles (may download new crate)

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/Cargo.toml
git commit -m "chore: add rustdct dependency for DCT-based density solve"
```

### Task 5: WA Wirelength Module

**Files:**
- Create: `crates/nextpnr/src/placer/solver/wa.rs`
- Modify: `crates/nextpnr/src/placer/solver/mod.rs`

- [ ] **Step 1: Create WA wirelength module**

Create `crates/nextpnr/src/placer/solver/wa.rs`:

```rust
//! Weighted-Average (WA) smooth wirelength model.
//!
//! Reference: placer_static.cc from YosysHQ/nextpnr.
//! For each net on each axis, WA wirelength is:
//!   wa_wl = (x_max_weighted / sum_max_exp) - (x_min_weighted / sum_min_exp)
//!
//! Uses fixed wl_coeff (no gamma annealing).

/// Minimum exponent argument to prevent underflow.
const EXP_CLAMP_MIN: f64 = -3000.0;

/// Compute WA wirelength for a single axis.
///
/// `coords` are the pin positions on one axis.
/// `wl_coeff` controls the exponential sharpness (default 0.5).
pub fn wa_axis_value(coords: &[f64], wl_coeff: f64) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let center = coords.iter().sum::<f64>() / coords.len() as f64;

    let mut sum_max_exp = 0.0;
    let mut x_max_weighted = 0.0;
    let mut sum_min_exp = 0.0;
    let mut x_min_weighted = 0.0;

    for &x in coords {
        let max_arg = (wl_coeff * (x - center)).max(EXP_CLAMP_MIN);
        let min_arg = (wl_coeff * (center - x)).max(EXP_CLAMP_MIN);
        let me = max_arg.exp();
        let ne = min_arg.exp();

        sum_max_exp += me;
        x_max_weighted += x * me;
        sum_min_exp += ne;
        x_min_weighted += x * ne;
    }

    if sum_max_exp < 1e-30 || sum_min_exp < 1e-30 {
        return 0.0;
    }

    x_max_weighted / sum_max_exp - x_min_weighted / sum_min_exp
}

/// Compute WA wirelength gradient for a single axis.
///
/// Accumulates into `grad_out[i]`. Uses the quotient-rule derivative
/// from placer_static.cc.
pub fn wa_axis_grad(coords: &[f64], wl_coeff: f64, grad_out: &mut [f64]) {
    let n = coords.len();
    if n < 2 {
        return;
    }

    let center = coords.iter().sum::<f64>() / n as f64;

    let mut sum_max_exp = 0.0;
    let mut x_max_weighted = 0.0;
    let mut sum_min_exp = 0.0;
    let mut x_min_weighted = 0.0;

    let mut max_exps = vec![0.0; n];
    let mut min_exps = vec![0.0; n];

    for (i, &x) in coords.iter().enumerate() {
        let max_arg = (wl_coeff * (x - center)).max(EXP_CLAMP_MIN);
        let min_arg = (wl_coeff * (center - x)).max(EXP_CLAMP_MIN);
        let me = max_arg.exp();
        let ne = min_arg.exp();

        max_exps[i] = me;
        min_exps[i] = ne;
        sum_max_exp += me;
        x_max_weighted += x * me;
        sum_min_exp += ne;
        x_min_weighted += x * ne;
    }

    if sum_max_exp < 1e-30 || sum_min_exp < 1e-30 {
        return;
    }

    let inv_max_sum = 1.0 / sum_max_exp;
    let inv_min_sum = 1.0 / sum_min_exp;
    let inv_max_sum_sq = inv_max_sum * inv_max_sum;
    let inv_min_sum_sq = inv_min_sum * inv_min_sum;

    for (i, &x) in coords.iter().enumerate() {
        // d/dx_i of (x_max_weighted / sum_max_exp)
        // = (sum_max_exp * (max_exp_i + wl_coeff * x * max_exp_i)
        //    - wl_coeff * max_exp_i * x_max_weighted) / sum_max_exp^2
        // Simplified:
        let d_max = max_exps[i] * inv_max_sum
            + wl_coeff * max_exps[i] * (x * inv_max_sum - x_max_weighted * inv_max_sum_sq);

        // d/dx_i of (x_min_weighted / sum_min_exp)
        // Note: min_exp uses (center - x), so d/dx has opposite sign on wl_coeff terms
        let d_min = min_exps[i] * inv_min_sum
            - wl_coeff * min_exps[i] * (x * inv_min_sum - x_min_weighted * inv_min_sum_sq);

        grad_out[i] += d_max - d_min;
    }
}

/// Compute WA wirelength for (x,y) positions.
pub fn wa_wirelength(positions: &[(f64, f64)], wl_coeff: f64) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }

    let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();

    wa_axis_value(&xs, wl_coeff) + wa_axis_value(&ys, wl_coeff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_points_wa() {
        let coords = vec![0.0, 10.0];
        let wl = wa_axis_value(&coords, 0.5);
        // Should approximate max - min = 10
        assert!(wl > 8.0 && wl < 12.0, "WA wl = {}, expected ~10", wl);
    }

    #[test]
    fn single_point_is_zero() {
        assert_eq!(wa_axis_value(&[5.0], 0.5), 0.0);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(wa_axis_value(&[], 0.5), 0.0);
    }

    #[test]
    fn gradient_finite_differences() {
        let coords = vec![1.0, 4.0, 2.0];
        let wl_coeff = 0.5;
        let eps = 1e-6;

        let mut grad = vec![0.0; 3];
        wa_axis_grad(&coords, wl_coeff, &mut grad);

        for i in 0..3 {
            let mut c_plus = coords.clone();
            let mut c_minus = coords.clone();
            c_plus[i] += eps;
            c_minus[i] -= eps;
            let fd = (wa_axis_value(&c_plus, wl_coeff) - wa_axis_value(&c_minus, wl_coeff))
                / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() < 1e-3,
                "WA grad mismatch at {}: fd={}, analytic={}",
                i, fd, grad[i]
            );
        }
    }

    #[test]
    fn wa_2d_wirelength() {
        let positions = vec![(0.0, 0.0), (3.0, 4.0)];
        let wl = wa_wirelength(&positions, 0.5);
        // Should approximate |3| + |4| = 7
        assert!(wl > 5.0 && wl < 9.0, "WA 2D wl = {}, expected ~7", wl);
    }
}
```

- [ ] **Step 2: Register module in solver/mod.rs**

Add to `crates/nextpnr/src/placer/solver/mod.rs`:

```rust
pub mod wa;
pub use wa::{wa_axis_grad, wa_axis_value, wa_wirelength};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nextpnr solver::wa 2>&1 | tail -15`
Expected: all WA tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/placer/solver/wa.rs crates/nextpnr/src/placer/solver/mod.rs
git commit -m "feat(solver): add Weighted-Average wirelength model"
```

### Task 6: Add WA Wirelength Gradient to Common

**Files:**
- Modify: `crates/nextpnr/src/placer/common.rs`

- [ ] **Step 1: Add `add_wa_wirelength_gradient` function**

Add after the existing `add_wirelength_gradient` function (after line 248):

```rust
/// Compute WA wirelength gradient for all nets, accumulating into grad_x/grad_y.
///
/// Uses Weighted-Average model instead of LSE. Fixed wl_coeff controls sharpness.
/// `net_weights` provides per-net scaling (e.g. timing-driven: 1.0 + 5*crit^2).
pub(crate) fn add_wa_wirelength_gradient(
    ctx: &Context,
    cell_to_idx: &FxHashMap<CellId, usize>,
    cell_x: &[f64],
    cell_y: &[f64],
    wl_coeff: f64,
    net_weights: Option<&FxHashMap<crate::netlist::NetId, f64>>,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
) {
    use super::solver::wa;

    for (net_id, net) in ctx.design.iter_alive_nets() {
        let mut pin_xs: Vec<f64> = Vec::new();
        let mut pin_ys: Vec<f64> = Vec::new();
        let mut pin_indices: Vec<usize> = Vec::new();

        if let Some(driver_pin) = net.driver() {
            collect_pin_position(
                ctx, cell_to_idx, cell_x, cell_y,
                driver_pin.cell, &mut Vec::new(), &mut Vec::new(),
            );
            // Re-collect properly with separate x/y
            if let Some(&idx) = cell_to_idx.get(&driver_pin.cell) {
                pin_xs.push(cell_x[idx]);
                pin_ys.push(cell_y[idx]);
                pin_indices.push(idx);
            } else {
                let cell = ctx.design.cell(driver_pin.cell);
                if let Some(bel) = cell.bel {
                    let loc = ctx.bel(bel).loc();
                    pin_xs.push(loc.x as f64);
                    pin_ys.push(loc.y as f64);
                    pin_indices.push(usize::MAX);
                }
            }
        }

        for user in net.users().iter() {
            if let Some(&idx) = cell_to_idx.get(&user.cell) {
                pin_xs.push(cell_x[idx]);
                pin_ys.push(cell_y[idx]);
                pin_indices.push(idx);
            } else {
                let cell = ctx.design.cell(user.cell);
                if let Some(bel) = cell.bel {
                    let loc = ctx.bel(bel).loc();
                    pin_xs.push(loc.x as f64);
                    pin_ys.push(loc.y as f64);
                    pin_indices.push(usize::MAX);
                }
            }
        }

        if pin_xs.len() < 2 {
            continue;
        }

        let net_weight = net_weights
            .and_then(|w| w.get(&net_id))
            .copied()
            .unwrap_or(1.0);

        let mut gx = vec![0.0; pin_xs.len()];
        let mut gy = vec![0.0; pin_ys.len()];
        wa::wa_axis_grad(&pin_xs, wl_coeff, &mut gx);
        wa::wa_axis_grad(&pin_ys, wl_coeff, &mut gy);

        for (k, &solver_idx) in pin_indices.iter().enumerate() {
            if solver_idx != usize::MAX {
                grad_x[solver_idx] += net_weight * gx[k];
                grad_y[solver_idx] += net_weight * gy[k];
            }
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/common.rs
git commit -m "feat(common): add WA wirelength gradient for ElectroPlace alignment"
```

### Task 7: Rewrite ElectroPlace Config

**Files:**
- Modify: `crates/nextpnr/src/placer/electro_place/config.rs`

- [ ] **Step 1: Replace config fields**

Replace entire file:

```rust
//! Configuration for the ElectroPlace (RePlAce-style) placer.

/// Configuration for the ElectroPlace analytical placer.
#[derive(Clone)]
pub struct ElectroPlaceCfg {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// WA wirelength exponential coefficient (default 0.5).
    pub wl_coeff: f64,
    /// Target utilization for spacer insertion (default 0.7).
    pub target_util: f64,
    /// Target density for density map (typically 1.0).
    pub target_density: f64,
    /// Enable timing-driven net weighting.
    pub timing_driven: bool,
    /// Timing penalty weight.
    pub timing_weight: f64,
    /// Initial Nesterov step size.
    pub nesterov_step_size: f64,
    /// Maximum outer iterations (safety limit).
    pub max_iters: usize,
    /// Legalize every N iterations.
    pub legalize_interval: usize,
}

impl Default for ElectroPlaceCfg {
    fn default() -> Self {
        Self {
            seed: 1,
            wl_coeff: 0.5,
            target_util: 0.7,
            target_density: 1.0,
            timing_driven: false,
            timing_weight: 0.0,
            nesterov_step_size: 0.1,
            max_iters: 500,
            legalize_interval: 5,
        }
    }
}
```

- [ ] **Step 2: Fix compilation errors from removed fields**

There will be compilation errors in `mod.rs` referencing `gamma_init`, `gamma_decay`, `gamma_min`, `density_weight`. These will be fixed in Task 9 (main loop rewrite). For now just verify the config compiles independently.

Run: `cargo check -p nextpnr 2>&1 | grep "error" | head -5`
Expected: errors only in `mod.rs` about removed fields

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/electro_place/config.rs
git commit -m "feat(electro): replace LSE/gamma config with WA/overlap config"
```

### Task 8: Rewrite DCT Density Module

**Files:**
- Rewrite: `crates/nextpnr/src/placer/electro_place/density.rs`

- [ ] **Step 1: Rewrite density.rs with DCT-based Poisson solve**

Replace entire file:

```rust
//! Density computation and gradient for ElectroPlace.
//!
//! Uses DCT-II based Poisson solve (reference: placer_static.cc).
//! Cell-to-bin mapping uses slither (fractional bin overlap based on cell area).

use rustdct::DctPlanner;

/// Compute the density map on a grid using slither cell-to-bin mapping.
///
/// Each cell occupies 1 BEL area. Cells are mapped to bins via fractional overlap.
/// The density map has target density subtracted.
pub fn compute_density_map(
    cell_x: &[f64],
    cell_y: &[f64],
    grid_w: usize,
    grid_h: usize,
    target_density: f64,
) -> Vec<f64> {
    let total_cells = grid_w * grid_h;
    let mut density = vec![0.0; total_cells];

    let bin_w = 1.0; // Each bin is 1 tile wide
    let bin_h = 1.0;
    let cell_w = 1.0; // Each cell occupies 1 BEL = 1 tile
    let cell_h = 1.0;

    for i in 0..cell_x.len() {
        // Slither mapping: distribute cell area across overlapping bins
        let x_lo = (cell_x[i] - cell_w / 2.0).max(0.0);
        let x_hi = (cell_x[i] + cell_w / 2.0).min(grid_w as f64);
        let y_lo = (cell_y[i] - cell_h / 2.0).max(0.0);
        let y_hi = (cell_y[i] + cell_h / 2.0).min(grid_h as f64);

        let bx_lo = (x_lo.floor() as usize).min(grid_w - 1);
        let bx_hi = (x_hi.ceil() as usize).min(grid_w);
        let by_lo = (y_lo.floor() as usize).min(grid_h - 1);
        let by_hi = (y_hi.ceil() as usize).min(grid_h);

        for by in by_lo..by_hi {
            let overlap_y = (y_hi.min((by + 1) as f64) - y_lo.max(by as f64)).max(0.0) / bin_h;
            for bx in bx_lo..bx_hi {
                let overlap_x =
                    (x_hi.min((bx + 1) as f64) - x_lo.max(bx as f64)).max(0.0) / bin_w;
                let overlap = overlap_x * overlap_y;
                if overlap > 0.0 {
                    density[by * grid_w + bx] += overlap;
                }
            }
        }
    }

    // Subtract target density
    let target = target_density * cell_x.len() as f64 / total_cells as f64;
    for d in &mut density {
        *d -= target;
    }

    density
}

/// Compute the density gradient for each cell using DCT-based Poisson solve.
///
/// 1. Forward DCT-II on density grid
/// 2. Poisson solve in spectral domain
/// 3. Compute electric field via spectral differentiation
/// 4. Inverse transforms (IDCT/IDST)
/// 5. Interpolate field at cell positions via slither weights
pub fn compute_density_gradient(
    cell_x: &[f64],
    cell_y: &[f64],
    density_map: &[f64],
    grid_w: usize,
    grid_h: usize,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
) {
    let (field_x, field_y) = poisson_field_dct(density_map, grid_w, grid_h);

    let n = cell_x.len();
    let cell_w = 1.0;
    let cell_h = 1.0;

    for i in 0..n {
        let x_lo = (cell_x[i] - cell_w / 2.0).max(0.0);
        let x_hi = (cell_x[i] + cell_w / 2.0).min(grid_w as f64);
        let y_lo = (cell_y[i] - cell_h / 2.0).max(0.0);
        let y_hi = (cell_y[i] + cell_h / 2.0).min(grid_h as f64);

        let bx_lo = (x_lo.floor() as usize).min(grid_w - 1);
        let bx_hi = (x_hi.ceil() as usize).min(grid_w);
        let by_lo = (y_lo.floor() as usize).min(grid_h - 1);
        let by_hi = (y_hi.ceil() as usize).min(grid_h);

        let mut fx_total = 0.0;
        let mut fy_total = 0.0;
        let mut w_total = 0.0;

        for by in by_lo..by_hi {
            let overlap_y = (y_hi.min((by + 1) as f64) - y_lo.max(by as f64)).max(0.0);
            for bx in bx_lo..bx_hi {
                let overlap_x = (x_hi.min((bx + 1) as f64) - x_lo.max(bx as f64)).max(0.0);
                let w = overlap_x * overlap_y;
                if w > 0.0 {
                    let idx = by * grid_w + bx;
                    fx_total += w * field_x[idx];
                    fy_total += w * field_y[idx];
                    w_total += w;
                }
            }
        }

        if w_total > 1e-10 {
            grad_x[i] += fx_total / w_total;
            grad_y[i] += fy_total / w_total;
        }
    }
}

/// Compute the concrete density (number of cells per bin, no target subtracted).
///
/// Used for overlap-based convergence metric.
pub fn compute_concrete_density(
    cell_x: &[f64],
    cell_y: &[f64],
    grid_w: usize,
    grid_h: usize,
) -> Vec<f64> {
    let total = grid_w * grid_h;
    let mut density = vec![0.0; total];

    for i in 0..cell_x.len() {
        let gx = (cell_x[i].round() as usize).min(grid_w - 1);
        let gy = (cell_y[i].round() as usize).min(grid_h - 1);
        density[gy * grid_w + gx] += 1.0;
    }

    density
}

/// Compute overlap metric: sum(max(0, density - 1)) / sum(density).
///
/// Measures what fraction of total cell area is overlapping with other cells.
pub fn compute_overlap(concrete_density: &[f64]) -> f64 {
    let excess: f64 = concrete_density.iter().map(|&d| (d - 1.0).max(0.0)).sum();
    let total: f64 = concrete_density.iter().sum();
    if total < 1e-10 {
        0.0
    } else {
        excess / total
    }
}

/// Solve Poisson equation using DCT-II and return electric field (Ex, Ey).
///
/// Steps:
/// 1. Forward DCT-II on density grid (rows then columns)
/// 2. Scale: first row/column by 0.5, all by 4/(m*m)
/// 3. Poisson solve: phi = rho / (wx^2 + wy^2)
/// 4. Field: ex = phi * wx, ey = phi * wy
/// 5. Inverse transforms for phi, Ex, Ey
fn poisson_field_dct(density: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let pi = std::f64::consts::PI;
    let n = w * h;

    // Forward DCT-II on density
    let mut rho = density.to_vec();
    dct2_2d(&mut rho, w, h);

    // Scale: first row/column by 0.5, all by 4/(m*m)
    let scale = 4.0 / (w * h) as f64;
    for ky in 0..h {
        for kx in 0..w {
            let idx = ky * w + kx;
            let mut s = scale;
            if kx == 0 {
                s *= 0.5;
            }
            if ky == 0 {
                s *= 0.5;
            }
            rho[idx] *= s;
        }
    }

    // Poisson solve + field computation in spectral domain
    let mut phi = vec![0.0; n];
    let mut ex_spec = vec![0.0; n];
    let mut ey_spec = vec![0.0; n];

    for ky in 0..h {
        let wy = pi * ky as f64 / h as f64;
        for kx in 0..w {
            let wx = pi * kx as f64 / w as f64;
            let idx = ky * w + kx;

            let denom = wx * wx + wy * wy;
            if denom < 1e-20 {
                phi[idx] = 0.0;
                ex_spec[idx] = 0.0;
                ey_spec[idx] = 0.0;
                continue;
            }

            phi[idx] = rho[idx] / denom;
            ex_spec[idx] = phi[idx] * wx;
            ey_spec[idx] = phi[idx] * wy;
        }
    }

    // Inverse transforms
    // Ex: IDST on x, IDCT on y
    // Ey: IDCT on x, IDST on y
    let mut field_x = ex_spec;
    idst_rows_idct_cols(&mut field_x, w, h);

    let mut field_y = ey_spec;
    idct_rows_idst_cols(&mut field_y, w, h);

    (field_x, field_y)
}

/// Forward DCT-II on a 2D grid (rows then columns).
fn dct2_2d(data: &mut [f64], w: usize, h: usize) {
    let mut planner = DctPlanner::new();

    // DCT-II on rows
    let dct_row = planner.plan_dct2(w);
    let mut scratch = vec![0.0; dct_row.get_scratch_len()];
    for y in 0..h {
        let row = &mut data[y * w..(y + 1) * w];
        dct_row.process_dct2_with_scratch(row, &mut scratch);
    }

    // DCT-II on columns
    let dct_col = planner.plan_dct2(h);
    let mut col_buf = vec![0.0; h];
    let mut scratch = vec![0.0; dct_col.get_scratch_len()];
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = data[y * w + x];
        }
        dct_col.process_dct2_with_scratch(&mut col_buf, &mut scratch);
        for y in 0..h {
            data[y * w + x] = col_buf[y];
        }
    }
}

/// IDST on rows, IDCT on columns (for Ex field recovery).
fn idst_rows_idct_cols(data: &mut [f64], w: usize, h: usize) {
    let mut planner = DctPlanner::new();

    // DST-III (inverse of DST-II ≈ IDST) on rows
    let dst_row = planner.plan_dst3(w);
    let mut scratch = vec![0.0; dst_row.get_scratch_len()];
    for y in 0..h {
        let row = &mut data[y * w..(y + 1) * w];
        dst_row.process_dst3_with_scratch(row, &mut scratch);
    }

    // DCT-III (inverse of DCT-II ≈ IDCT) on columns
    let dct_col = planner.plan_dct3(h);
    let mut col_buf = vec![0.0; h];
    let mut scratch = vec![0.0; dct_col.get_scratch_len()];
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = data[y * w + x];
        }
        dct_col.process_dct3_with_scratch(&mut col_buf, &mut scratch);
        for y in 0..h {
            data[y * w + x] = col_buf[y];
        }
    }
}

/// IDCT on rows, IDST on columns (for Ey field recovery).
fn idct_rows_idst_cols(data: &mut [f64], w: usize, h: usize) {
    let mut planner = DctPlanner::new();

    // DCT-III on rows
    let dct_row = planner.plan_dct3(w);
    let mut scratch = vec![0.0; dct_row.get_scratch_len()];
    for y in 0..h {
        let row = &mut data[y * w..(y + 1) * w];
        dct_row.process_dct3_with_scratch(row, &mut scratch);
    }

    // DST-III on columns
    let dst_col = planner.plan_dst3(h);
    let mut col_buf = vec![0.0; h];
    let mut scratch = vec![0.0; dst_col.get_scratch_len()];
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = data[y * w + x];
        }
        dst_col.process_dst3_with_scratch(&mut col_buf, &mut scratch);
        for y in 0..h {
            data[y * w + x] = col_buf[y];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_map_single_cell() {
        let x = vec![4.0];
        let y = vec![4.0];
        let map = compute_density_map(&x, &y, 8, 8, 0.0);

        // Peak should be at or near (4, 4)
        let peak_idx = 4 * 8 + 4;
        let peak = map[peak_idx];
        assert!(peak > 0.0, "Peak density should be positive: {}", peak);
    }

    #[test]
    fn density_gradient_pushes_apart() {
        let x = vec![4.0, 4.0];
        let y = vec![4.0, 4.0];
        let map = compute_density_map(&x, &y, 8, 8, 0.0);

        let mut gx = vec![0.0; 2];
        let mut gy = vec![0.0; 2];
        compute_density_gradient(&x, &y, &map, 8, 8, &mut gx, &mut gy);

        for g in &gx {
            assert!(g.is_finite(), "Gradient should be finite: {}", g);
        }
    }

    #[test]
    fn overlap_empty_grid() {
        let density = vec![0.0; 64];
        assert_eq!(compute_overlap(&density), 0.0);
    }

    #[test]
    fn overlap_no_conflict() {
        let mut density = vec![0.0; 64];
        density[0] = 1.0;
        density[1] = 1.0;
        assert_eq!(compute_overlap(&density), 0.0);
    }

    #[test]
    fn overlap_full_conflict() {
        let mut density = vec![0.0; 4];
        density[0] = 4.0; // All 4 cells in one bin
        let overlap = compute_overlap(&density);
        assert!((overlap - 0.75).abs() < 1e-10, "Expected 0.75, got {}", overlap);
    }

    #[test]
    fn concrete_density_counts() {
        let x = vec![0.0, 0.0, 1.0];
        let y = vec![0.0, 0.0, 1.0];
        let d = compute_concrete_density(&x, &y, 4, 4);
        assert_eq!(d[0], 2.0); // Two cells at (0,0)
        assert_eq!(d[4 + 1], 1.0); // One cell at (1,1)
    }
}
```

- [ ] **Step 2: Verify compilation and tests**

Run: `cargo test -p nextpnr electro_place::density 2>&1 | tail -15`
Expected: all density tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/electro_place/density.rs
git commit -m "feat(electro): rewrite density with DCT-II Poisson solve and slither mapping"
```

### Task 9: Rewrite ElectroPlace Main Loop

**Files:**
- Rewrite: `crates/nextpnr/src/placer/electro_place/mod.rs`

- [ ] **Step 1: Rewrite mod.rs with WA + growing penalty + overlap convergence**

Replace entire file:

```rust
//! ElectroPlace: RePlAce-style analytical placer.
//!
//! Aligned with upstream placer_static.cc. Uses:
//! - Weighted-Average (WA) smooth wirelength
//! - DCT-based density penalty (Poisson electric field)
//! - Growing density penalty (×1.025 until 50, then +1)
//! - Overlap-based convergence
//! - Barzilai-Borwein step size
//! - Spacer/filler insertion
//! - No gamma annealing, no adaptive restart

pub mod config;
pub mod density;

pub use config::ElectroPlaceCfg;

use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::CellId;
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use super::common::{
    add_wa_wirelength_gradient, apply_preconditioner, clamp_positions, collect_movable_cells,
    compute_pin_weights, gradient_norm, init_positions_from_bels, initial_placement,
    place_cluster_children, unbind_movable_cells, validate_all_placed, with_locked_others,
};
use super::solver::NesterovSolver;
use super::PlacerError;

const DENSITY_NORM_EPSILON: f64 = 1e-30;
/// Initial density penalty scaling ratio (eta in placer_static.cc).
const DENSITY_PENALTY_ETA: f64 = 0.1;
/// Overlap threshold for IP groups (BRAM/DSP).
const IP_OVERLAP_THRESHOLD: f64 = 0.15;
/// Overlap threshold for logic groups.
const LOGIC_OVERLAP_THRESHOLD: f64 = 0.1;

pub struct PlacerElectro;

impl super::Placer for PlacerElectro {
    type Config = ElectroPlaceCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_electro(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let cells_set: FxHashSet<CellId> = cells.iter().copied().collect();
        with_locked_others(ctx, &cells_set, |ctx| place_electro(ctx, cfg))
    }
}

pub fn place_electro(ctx: &mut Context, cfg: &ElectroPlaceCfg) -> Result<(), PlacerError> {
    ctx.reseed_rng(cfg.seed);

    initial_placement(ctx)?;
    ctx.populate_bel_buckets();

    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    let max_x = (w - 1) as f64;
    let max_y = (h - 1) as f64;
    let grid_w = w as usize;
    let grid_h = h as usize;

    let (cell_to_idx, idx_to_cell) = collect_movable_cells(ctx);
    let n = idx_to_cell.len();
    if n == 0 {
        return Ok(());
    }

    let mut cell_x = vec![0.0; n];
    let mut cell_y = vec![0.0; n];
    init_positions_from_bels(ctx, &idx_to_cell, &mut cell_x, &mut cell_y);

    // Insert spacers to reach target utilization
    let total_bins = (grid_w * grid_h) as f64;
    let concrete_area = n as f64;
    let target_area = total_bins * cfg.target_util;
    let n_spacers = ((target_area - concrete_area).max(0.0)) as usize;

    // Spacers participate in density but not wirelength
    let total_cells = n + n_spacers;
    let mut all_x = vec![0.0; total_cells];
    let mut all_y = vec![0.0; total_cells];
    all_x[..n].copy_from_slice(&cell_x);
    all_y[..n].copy_from_slice(&cell_y);

    // Place spacers randomly across the grid
    for i in n..total_cells {
        all_x[i] = (ctx.rng_mut().next_u32() % w as u32) as f64;
        all_y[i] = (ctx.rng_mut().next_u32() % h as u32) as f64;
    }

    let mut nesterov_x = NesterovSolver::new(n, cfg.nesterov_step_size);
    let mut nesterov_y = NesterovSolver::new(n, cfg.nesterov_step_size);
    nesterov_x.set_positions(&cell_x);
    nesterov_y.set_positions(&cell_y);

    info!(
        "ElectroPlace: {} movable cells, {} spacers, {}x{} grid",
        n, n_spacers, grid_w, grid_h
    );

    let pin_weights = compute_pin_weights(ctx, &cell_to_idx, n);
    let mut density_penalty = 0.0;
    let mut best_hpwl = f64::INFINITY;
    let mut best_x = cell_x.clone();
    let mut best_y = cell_y.clone();

    // Previous gradient for BB step size
    let mut prev_grad_x = vec![0.0; n];
    let mut prev_grad_y = vec![0.0; n];

    for iter in 0..cfg.max_iters {
        // Nesterov look-ahead
        nesterov_x.look_ahead_into(&mut cell_x);
        nesterov_y.look_ahead_into(&mut cell_y);
        clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

        // Update spacer positions (they track real cell positions loosely)
        all_x[..n].copy_from_slice(&cell_x);
        all_y[..n].copy_from_slice(&cell_y);

        // WA wirelength gradient (real cells only)
        let mut grad_x = vec![0.0; n];
        let mut grad_y = vec![0.0; n];
        add_wa_wirelength_gradient(
            ctx, &cell_to_idx, &cell_x, &cell_y, cfg.wl_coeff, None,
            &mut grad_x, &mut grad_y,
        );

        // Density gradient (all cells including spacers)
        let density_map = density::compute_density_map(
            &all_x, &all_y, grid_w, grid_h, cfg.target_density,
        );
        let mut density_grad_x_all = vec![0.0; total_cells];
        let mut density_grad_y_all = vec![0.0; total_cells];
        density::compute_density_gradient(
            &all_x, &all_y, &density_map, grid_w, grid_h,
            &mut density_grad_x_all, &mut density_grad_y_all,
        );

        // Extract density gradient for real cells only
        let density_grad_x: Vec<f64> = density_grad_x_all[..n].to_vec();
        let density_grad_y: Vec<f64> = density_grad_y_all[..n].to_vec();

        // Initialize or grow density penalty
        if iter == 0 {
            let wl_norm = gradient_norm(&grad_x, &grad_y);
            let den_norm = gradient_norm(&density_grad_x, &density_grad_y);
            if den_norm > DENSITY_NORM_EPSILON {
                density_penalty = DENSITY_PENALTY_ETA * wl_norm / den_norm;
            }
        } else if density_penalty < 50.0 {
            density_penalty *= 1.025;
        } else {
            density_penalty += 1.0;
        }

        // Combine wirelength + density gradients
        for i in 0..n {
            grad_x[i] += density_penalty * density_grad_x[i];
            grad_y[i] += density_penalty * density_grad_y[i];
        }

        // Preconditioner: precond[i] = max(1.0, pin_count[i] + density_penalty * 1.0)
        // (cell_area = 1.0 for FPGA)
        for i in 0..n {
            let precond = (pin_weights[i] + density_penalty).max(1.0);
            grad_x[i] /= precond;
            grad_y[i] /= precond;
        }

        // Barzilai-Borwein step size (after iteration 0)
        if iter > 0 {
            if let Some(bb_x) = nesterov_x.bb_step_size(&prev_grad_x, &grad_x) {
                nesterov_x.set_step_size(bb_x.clamp(1e-4, 1.0));
            }
            if let Some(bb_y) = nesterov_y.bb_step_size(&prev_grad_y, &grad_y) {
                nesterov_y.set_step_size(bb_y.clamp(1e-4, 1.0));
            }
        }
        prev_grad_x.copy_from_slice(&grad_x);
        prev_grad_y.copy_from_slice(&grad_y);

        let step_x = nesterov_x.step(&grad_x);
        let step_y = nesterov_y.step(&grad_y);

        nesterov_x.clamp_positions_range(0.0, max_x);
        nesterov_y.clamp_positions_range(0.0, max_y);

        // No adaptive restart (plain Nesterov per reference)

        // Overlap-based convergence check
        if iter % cfg.legalize_interval == 0 || iter == cfg.max_iters - 1 {
            cell_x.copy_from_slice(nesterov_x.positions());
            cell_y.copy_from_slice(nesterov_y.positions());
            clamp_positions(&mut cell_x, &mut cell_y, max_x, max_y);

            let concrete = density::compute_concrete_density(&cell_x, &cell_y, grid_w, grid_h);
            let overlap = density::compute_overlap(&concrete);

            let displacement = legalize_electro(ctx, &idx_to_cell, &cell_x, &cell_y)?;
            let hpwl = crate::metrics::wirelength::total_hpwl(ctx);

            if hpwl < best_hpwl {
                best_hpwl = hpwl;
                best_x.copy_from_slice(&cell_x);
                best_y.copy_from_slice(&cell_y);
            }

            eprintln!(
                "ElectroPlace iter {}: HPWL={:.0}, overlap={:.3}, disp={:.1}, step=({:.4},{:.4}), density_w={:.3}",
                iter, hpwl, overlap, displacement, step_x, step_y, density_penalty,
            );

            // Overlap-based termination
            if overlap < LOGIC_OVERLAP_THRESHOLD {
                eprintln!("ElectroPlace converged at iteration {} (overlap {:.3})", iter, overlap);
                break;
            }
        }
    }

    // Final legalization from best positions
    let _ = legalize_electro(ctx, &idx_to_cell, &best_x, &best_y)?;

    validate_all_placed(ctx)?;
    info!("ElectroPlace complete");
    Ok(())
}

fn legalize_electro(
    ctx: &mut Context,
    idx_to_cell: &[CellId],
    cell_x: &[f64],
    cell_y: &[f64],
) -> Result<f64, PlacerError> {
    unbind_movable_cells(ctx, idx_to_cell);

    let mut total_displacement = 0.0;

    for (i, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell_type = ctx.design.cell(cell_id).cell_type;
        let target_x = cell_x[i];
        let target_y = cell_y[i];

        let mut best_bel = None;
        let mut best_cost = f64::INFINITY;

        for bel_view in ctx.bels_for_bucket(cell_type) {
            if !bel_view.is_available() {
                continue;
            }
            let loc = bel_view.loc();
            let dx = loc.x as f64 - target_x;
            let dy = loc.y as f64 - target_y;
            let cost = dx * dx + dy * dy;

            if cost < best_cost {
                best_cost = cost;
                best_bel = Some(bel_view.id());
            }
        }

        let bel = best_bel.ok_or_else(|| {
            PlacerError::NoBelsAvailable(ctx.name_of(cell_type).to_owned())
        })?;

        if !ctx.bind_bel(bel, cell_id, PlaceStrength::Placer) {
            let cell_name = ctx.design.cell(cell_id).name;
            return Err(PlacerError::PlacementFailed(format!(
                "Failed to bind cell {} during ElectroPlace legalization",
                ctx.name_of(cell_name)
            )));
        }

        total_displacement += best_cost;
        place_cluster_children(ctx, cell_id, bel)?;
    }

    Ok(total_displacement)
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -10`
Expected: compiles. May need to fix `rng_mut().next_u32()` to match the actual RNG API. Check `Context` for the RNG method name.

- [ ] **Step 3: Run tests**

Run: `cargo test -p nextpnr --features test-utils 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/placer/electro_place/mod.rs
git commit -m "feat(electro): rewrite main loop - WA wirelength, growing penalty, overlap convergence, BB step, spacers"
```

---

## Chunk 3: Workstream 3 - Dynamic Grid Cropping

### Task 10: Add CroppedRegion to Kirchhoff Solver

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/kirchhoff.rs`

- [ ] **Step 1: Add CroppedRegion struct and cropping logic**

Add at the top of `kirchhoff.rs` (after the imports):

```rust
/// A cropped sub-region of the full junction grid for faster solving.
pub struct CroppedRegion {
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    /// Maps full junction index -> cropped index (None if outside).
    pub junction_map: Vec<Option<usize>>,
    /// Inverse: cropped index -> full junction index.
    pub cropped_to_full: Vec<usize>,
}

impl CroppedRegion {
    /// Compute the active region from cell positions with margin.
    pub fn from_cells(
        cell_x: &[f64],
        cell_y: &[f64],
        width: i32,
        height: i32,
    ) -> Self {
        if cell_x.is_empty() {
            return Self {
                x_min: 0, x_max: width - 1,
                y_min: 0, y_max: height - 1,
                junction_map: (0..(width * height * 4) as usize).map(Some).collect(),
                cropped_to_full: (0..(width * height * 4) as usize).collect(),
            };
        }

        let mut x_lo = i32::MAX;
        let mut x_hi = i32::MIN;
        let mut y_lo = i32::MAX;
        let mut y_hi = i32::MIN;

        for i in 0..cell_x.len() {
            let tx = cell_x[i].round() as i32;
            let ty = cell_y[i].round() as i32;
            x_lo = x_lo.min(tx);
            x_hi = x_hi.max(tx);
            y_lo = y_lo.min(ty);
            y_hi = y_hi.max(ty);
        }

        let margin = 5.max((cell_x.len() as f64).sqrt().ceil() as i32);
        let x_min = (x_lo - margin).max(0);
        let x_max = (x_hi + margin).min(width - 1);
        let y_min = (y_lo - margin).max(0);
        let y_max = (y_hi + margin).min(height - 1);

        let full_n = (width * height * 4) as usize;
        let mut junction_map = vec![None; full_n];
        let mut cropped_to_full = Vec::new();

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                for port in 0..4 {
                    let full_idx = ((y * width + x) * 4 + port) as usize;
                    let cropped_idx = cropped_to_full.len();
                    junction_map[full_idx] = Some(cropped_idx);
                    cropped_to_full.push(full_idx);
                }
            }
        }

        Self {
            x_min, x_max, y_min, y_max,
            junction_map, cropped_to_full,
        }
    }

    /// Number of junctions in the cropped region.
    pub fn num_junctions(&self) -> usize {
        self.cropped_to_full.len()
    }
}
```

- [ ] **Step 2: Add cropped solve function**

Add a new `kirchhoff_solve_cropped` function:

```rust
/// Solve the Kirchhoff system on a cropped sub-region of the network.
///
/// Only junctions inside the CroppedRegion are solved. Boundary junctions
/// get Dirichlet P=0. Junctions outside the region keep P=0.
pub fn kirchhoff_solve_cropped(
    network: &mut PipeNetwork,
    demand: &[f64],
    turbulence_beta: f64,
    newton_iters: usize,
    cg_max_iters: usize,
    cg_tol: f64,
    region: &CroppedRegion,
) -> SolveResult {
    let n_cropped = region.num_junctions();
    if n_cropped == 0 {
        return SolveResult { converged: true, iterations: 0, energy: 0.0 };
    }

    // Map demand to cropped indices
    let mut cropped_demand = vec![0.0; n_cropped];
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        cropped_demand[ci] = demand[fi];
    }

    // Initialize pressure from previous solve
    let mut pressure = vec![0.0; n_cropped];
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        pressure[ci] = network.junctions[fi].pressure;
    }

    let mut total_iters = 0;
    let num_solves = newton_iters.max(1);

    for newton_iter in 0..num_solves {
        let use_turbulence = newton_iter > 0;

        // Build cropped Laplacian
        let mut diag = vec![0.0; n_cropped];
        let mut off_diag: Vec<(usize, usize, f64)> = Vec::new();

        for pipe in &network.pipes {
            let ci_from = region.junction_map[pipe.from];
            let ci_to = region.junction_map[pipe.to];

            // Skip pipes entirely outside the region
            let (Some(cf), Some(ct)) = (ci_from, ci_to) else {
                continue;
            };

            let conductance = 1.0 / effective_resistance(pipe, turbulence_beta, use_turbulence);
            diag[cf] += conductance;
            diag[ct] += conductance;
            let (lo, hi) = if cf < ct { (cf, ct) } else { (ct, cf) };
            off_diag.push((lo, hi, -conductance));
        }

        // Pin first junction as pressure reference
        diag[0] = 1e10;
        cropped_demand[0] = 0.0;

        let iters = preconditioned_conjugate_gradient(
            &diag, &off_diag, &cropped_demand, &mut pressure, cg_tol, cg_max_iters,
        );
        total_iters += iters;

        // Compute flows for pipes within the region
        for pipe in &mut network.pipes {
            let ci_from = region.junction_map[pipe.from];
            let ci_to = region.junction_map[pipe.to];
            if let (Some(cf), Some(ct)) = (ci_from, ci_to) {
                let r_eff = effective_resistance(pipe, turbulence_beta, use_turbulence);
                pipe.flow = (pressure[cf] - pressure[ct]) / r_eff;
            }
        }
    }

    // Write pressure back to full network
    for j in &mut network.junctions {
        j.pressure = 0.0;
    }
    for (ci, &fi) in region.cropped_to_full.iter().enumerate() {
        network.junctions[fi].pressure = pressure[ci];
    }

    let energy: f64 = demand.iter().zip(network.junctions.iter())
        .map(|(d, j)| d * j.pressure)
        .sum();

    SolveResult {
        converged: true,
        iterations: total_iters,
        energy,
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/kirchhoff.rs
git commit -m "feat(hydraulic): add CroppedRegion and kirchhoff_solve_cropped"
```

### Task 11: Integrate Cropped Solve into Main Loop

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/mod.rs`

- [ ] **Step 1: Replace full solve with cropped solve**

In `mod.rs`, replace the Kirchhoff solve call (lines 107-114):

```rust
// Compute active region for this iteration
let region = kirchhoff::CroppedRegion::from_cells(
    &state.cell_x, &state.cell_y, state.network.width, state.network.height,
);

// Solve Kirchhoff system on cropped region
let result = kirchhoff::kirchhoff_solve_cropped(
    &mut state.network,
    &demand,
    beta,
    cfg.newton_iters,
    cfg.cg_max_iters,
    cfg.cg_tolerance,
    &region,
);
```

- [ ] **Step 2: Verify compilation and test**

Run: `cargo build -p nextpnr && cargo test -p nextpnr --features test-utils 2>&1 | tail -20`
Expected: compiles, all tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/mod.rs
git commit -m "feat(hydraulic): integrate dynamic grid cropping into main loop"
```

---

## Chunk 4: Workstream 4 - Multigrid Preconditioner for CG

### Task 12: Add Multigrid-Preconditioned CG

**Files:**
- Modify: `crates/nextpnr/src/placer/solver/cg.rs`

- [ ] **Step 1: Add multigrid-preconditioned CG variant**

Add at the end of `cg.rs`:

```rust
/// Multigrid-preconditioned Conjugate Gradient for the Kirchhoff network.
///
/// Uses the multigrid V-cycle as preconditioner instead of Jacobi.
/// The multigrid operates on the 2D grid structure implied by the junction layout.
///
/// `grid_width` and `grid_height` define the tile grid dimensions.
/// The junction system has 4 junctions per tile (N,E,S,W ports).
pub fn multigrid_preconditioned_cg(
    diag: &[f64],
    off_diag: &[(usize, usize, f64)],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
    grid_width: usize,
    grid_height: usize,
) -> usize {
    let n = diag.len();
    let n_tiles = grid_width * grid_height;

    // If the system is small or doesn't match 4-port-per-tile layout, fall back to Jacobi
    if n != n_tiles * 4 || grid_width <= 4 || grid_height <= 4 {
        return preconditioned_conjugate_gradient(diag, off_diag, rhs, x, tol, max_iters);
    }

    // Build per-tile averaged system for multigrid preconditioner
    // Average the 4 port diagonals per tile to get a scalar Laplacian
    let mg = super::multigrid::MultigridSolver::new(grid_width, grid_height);

    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];

    // r = b - A*x
    spmv(diag, off_diag, x, &mut ap);
    let mut rhs_norm = 0.0;
    for i in 0..n {
        r[i] = rhs[i] - ap[i];
        rhs_norm += rhs[i] * rhs[i];
    }
    rhs_norm = rhs_norm.sqrt().max(1e-12);

    // Apply multigrid preconditioner: project to tile grid, V-cycle, back-project
    apply_mg_preconditioner(&r, &mut z, &mg, grid_width, grid_height);

    p.copy_from_slice(&z);
    let mut rz_old: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

    for iter in 0..max_iters {
        spmv(diag, off_diag, &p, &mut ap);

        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, ai)| pi * ai).sum();
        let alpha = rz_old / p_ap.max(1e-16);

        let mut r_norm_sq = 0.0;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            r_norm_sq += r[i] * r[i];
        }

        if r_norm_sq.sqrt() / rhs_norm < tol {
            return iter + 1;
        }

        apply_mg_preconditioner(&r, &mut z, &mg, grid_width, grid_height);

        let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let beta = rz_new / rz_old;

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    max_iters
}

/// Apply multigrid V-cycle as a preconditioner.
///
/// Projects the 4-port-per-tile residual to a scalar tile grid,
/// applies one V-cycle, and back-projects to the port-level solution.
fn apply_mg_preconditioner(
    r: &[f64],
    z: &mut [f64],
    mg: &super::multigrid::MultigridSolver,
    grid_w: usize,
    grid_h: usize,
) {
    let n_tiles = grid_w * grid_h;

    // Restrict: average 4 ports per tile into scalar tile residual
    let mut tile_rhs = vec![0.0; n_tiles];
    for t in 0..n_tiles {
        let base = t * 4;
        tile_rhs[t] = (r[base] + r[base + 1] + r[base + 2] + r[base + 3]) / 4.0;
    }

    // V-cycle solve on tile grid
    let mut tile_z = vec![0.0; n_tiles];
    mg.solve(&tile_rhs, &mut tile_z, 1);

    // Prolongate: broadcast tile solution to all 4 ports
    for t in 0..n_tiles {
        let base = t * 4;
        z[base] = tile_z[t];
        z[base + 1] = tile_z[t];
        z[base + 2] = tile_z[t];
        z[base + 3] = tile_z[t];
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p nextpnr 2>&1 | tail -5`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/solver/cg.rs
git commit -m "feat(solver): add multigrid-preconditioned CG for Kirchhoff systems"
```

### Task 13: Use Multigrid PCG in Kirchhoff Solver

**Files:**
- Modify: `crates/nextpnr/src/placer/hydraulic_place/kirchhoff.rs`

- [ ] **Step 1: Replace Jacobi PCG with multigrid PCG in cropped solve**

In `kirchhoff_solve_cropped`, replace the `preconditioned_conjugate_gradient` call with:

```rust
use crate::placer::solver::cg::multigrid_preconditioned_cg;

// Compute cropped grid dimensions for multigrid
let crop_w = (region.x_max - region.x_min + 1) as usize;
let crop_h = (region.y_max - region.y_min + 1) as usize;

let iters = multigrid_preconditioned_cg(
    &diag, &off_diag, &cropped_demand, &mut pressure,
    cg_tol, cg_max_iters, crop_w, crop_h,
);
```

Also add the import at the top.

- [ ] **Step 2: Also update the full solve for fallback**

In the original `kirchhoff_solve`, replace the CG call with:

```rust
let iters = crate::placer::solver::cg::multigrid_preconditioned_cg(
    &diag, &off_diag, &rhs, &mut pressure,
    cg_tol, cg_max_iters,
    network.width as usize, network.height as usize,
);
```

- [ ] **Step 3: Verify compilation and test**

Run: `cargo build -p nextpnr && cargo test -p nextpnr --features test-utils 2>&1 | tail -20`
Expected: compiles, all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/placer/hydraulic_place/kirchhoff.rs
git commit -m "feat(hydraulic): use multigrid-preconditioned CG in Kirchhoff solver"
```

### Task 14: Final Integration Test

- [ ] **Step 1: Run full test suite**

Run: `cargo test -p nextpnr --features test-utils 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 2: Verify benchmark runs**

Run: `cargo build -p nextpnr --release 2>&1 | tail -5`
Expected: release build compiles

- [ ] **Step 3: Commit any fixes and tag completion**

```bash
git add -A
git commit -m "chore: placer optimization - all 4 workstreams complete"
```

---

## Verification Checklist

- `cargo build -p nextpnr` after each task
- `cargo test -p nextpnr --features test-utils` all existing tests pass
- WA gradient matches finite differences (unit test in wa.rs)
- DCT density gradient produces finite values (unit test in density.rs)
- Overlap metric: 0 for non-overlapping, positive for overlapping
- ElectroPlace converges via overlap < 0.1 (not HPWL stagnation)
- Hydraulic gradient clipping: magnitudes bounded by median * 10
- Cropped solve: junction count << full grid for clustered cells
- Multigrid PCG: fewer CG iterations than Jacobi PCG

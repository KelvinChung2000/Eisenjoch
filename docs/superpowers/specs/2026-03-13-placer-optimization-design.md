# Placer Optimization Design

Date: 2026-03-13

## Context

Benchmark results show the hydraulic placer is 600-6000x slower than HeAP with 37-114% worse HPWL, though it produces the best routed wirelength (90.7% of HeAP). The ElectroPlace implementation diverges significantly from the upstream `placer_static.cc` reference. This design addresses both.

## Workstream 1: ElectroPlace Alignment with placer_static.cc

Align our ElectroPlace with the upstream nextpnr `placer_static.cc` (RePlAce-style analytical placer). Skip post-legalization SA refinement.

### 1a. Replace LSE with Weighted-Average (WA) Wirelength

The reference uses WA wirelength, not LSE. For each net on each axis:

```
wa_wl = (x_max_weighted / sum_max_exp) - (x_min_weighted / sum_min_exp)
```

Where per-port exponential weights are:
```
min_exp = exp(wl_coeff * (center - location))
max_exp = exp(wl_coeff * (location - center))
```

Fixed `wl_coeff = 0.5` (no gamma annealing). Gradient per port:
```
d_min = (min_sum * min_exp * (1 - wl_coeff * loc) + wl_coeff * min_exp * x_min_sum) / min_sum^2
d_max = (max_sum * max_exp * (1 + wl_coeff * loc) - wl_coeff * max_exp * x_max_sum) / max_sum^2
grad += weight * (d_min - d_max)
```

Exponential underflow guard: clamp exponent argument above -3000.

**Files:** New `solver/wa.rs` module. Remove `gamma` from config and main loop.

### 1b. Replace Complex FFT with DCT-based Density Solve

The reference uses DCT-II (real-valued, ~2x faster than complex FFT).

Grid size: `m = 2^ceil(log2(max(width, height)))` with `bin_w = width/m`, `bin_h = height/m`.

Steps:
1. Map cells to bins via slither (fractional bin overlap based on cell area)
2. Forward DCT-II on density grid
3. Scale: first row/column by 0.5, all by `4/(m*m)`
4. Poisson solve: `phi = rho / (wx^2 + wy^2)` where `wx = pi*kx/m`, `wy = pi*ky/m`
5. Field: `ex = phi * wx`, `ey = phi * wy`
6. Inverse transforms: IDCT for phi, IDST-x/IDCT-y for Ex, IDCT-x/IDST-y for Ey
7. Interpolate field at cell positions via slither weights

**Files:** Rewrite `density.rs`. Add `rustdct` crate dependency.

### 1c. Growing Density Penalty

Instead of computing once and freezing:

```
if density_penalty < 50.0:
    density_penalty *= 1.025
else:
    density_penalty += 1.0
```

Initial value: `eta * (wirelen_force_norm / density_force_norm)` where `eta = 0.1`.

### 1d. Overlap-Based Convergence

Replace HPWL stagnation check with overlap metric:

```
overlap = sum(max(0, concrete_density[tile] - 1)) / sum(concrete_density[tile])
```

Two-phase termination:
- IP groups (BRAM/DSP): legalize when overlap < 0.15
- Logic groups: legalize when overlap < 0.1

No max iteration limit. No divergence detection (growing penalty prevents divergence).

### 1e. Barzilai-Borwein Step Size

Replace Lipschitz estimate with:
```
coord_dist = sqrt(sum((ref_pos - last_ref_pos)^2) / (2*n))
grad_dist = sqrt(sum((total_grad - last_total_grad)^2) / (2*n))
steplen = coord_dist / grad_dist
```

### 1f. Spacer/Filler Insertion

Before solving, insert filler cells to reach target utilization (70%):

```
spacer_count = (total_area * 0.7 - concrete_area) / spacer_area
```

Spacers are placed randomly. They participate in density but not wirelength. Dark nodes fill tiles with low available area.

### 1g. Remove Adaptive Restart

Reference uses plain Nesterov without restart. Remove `adaptive_restart()` calls.

### 1h. Preconditioner

Simple diagonal: `precond[i] = max(1.0, pin_count[i] + density_penalty * cell_area[i])`.

### 1i. Timing-Driven Weighting

Net weight = `1.0 + 5 * crit^2` where crit is from timing analysis (updated every 10 iterations).

Delay estimate: `c + mx * |dx| + my * |dy|` with defaults c=100, mx=100, my=100.

### Summary of Removed Features

- LSE wirelength and gamma annealing
- Frozen density weight
- Adaptive Nesterov restart
- HPWL-based convergence checks
- Complex FFT density solve

### Summary of Config Changes

Remove: `gamma_init`, `gamma_decay`, `gamma_min`, `density_weight` (auto-computed always).

Add: `wl_coeff` (default 0.5), `target_util` (default 0.7), `timing_driven` (default false).

Keep: `seed`, `target_density`, `nesterov_step_size`, `max_iters` (as safety limit), `legalize_interval`.

---

## Workstream 2: Gradient Clipping and Turbulence Control (Hydraulic)

### 2a. Gradient Clipping

After computing pressure gradient, clip per-cell gradient magnitude:

```
max_grad = median(grad_magnitudes) * 10
for each cell:
    mag = sqrt(gx^2 + gy^2)
    if mag > max_grad:
        scale = max_grad / mag
        gx *= scale; gy *= scale
```

This prevents a few high-pressure cells from dominating the Nesterov step.

**File:** `state.rs`, new `clip_gradients()` method.

### 2b. Slower Turbulence Ramp

Replace linear ramp with exponential:

```
beta = beta_max * (1 - exp(-3 * iter / max_outer_iters))
```

This reaches ~95% of beta_max by the end but starts much more gently than the linear ramp.

**File:** `mod.rs` line 103.

### 2c. Step Size Floor

After Lipschitz step estimate, enforce minimum:

```
step = max(step, 0.01 * initial_step_size)
```

Prevents step collapse after overshoot.

**File:** `solver/nesterov.rs`.

### 2d. Line Estimate for Convergence

Replace HPWL with `total_line_estimate` in convergence/divergence checks. The line estimate better correlates with routed wirelength (which is what the hydraulic placer actually optimizes well).

**File:** `mod.rs` convergence block.

---

## Workstream 3: Dynamic Grid Cropping (Hydraulic)

### Algorithm

Before each Kirchhoff solve:

1. Compute bounding box of all movable cell tile positions
2. Expand by `margin = max(5, ceil(sqrt(n_cells)))` in each direction
3. Clamp to `[0, width) x [0, height)`
4. Build a cropped subnetwork containing only junctions and pipes within the active region
5. Apply Dirichlet boundary (P=0) at the boundary junctions
6. Solve the smaller system
7. Write pressures back to full network (cropped region gets solved values, outside gets 0)

### Data Structures

```rust
struct CroppedRegion {
    x_min: i32, x_max: i32,
    y_min: i32, y_max: i32,
    // Maps full junction index -> cropped index (None if outside)
    junction_map: Vec<Option<usize>>,
    // Inverse map: cropped index -> full junction index
    cropped_to_full: Vec<usize>,
}
```

### Integration

The cropped region is recomputed each iteration (cell positions change). The Kirchhoff solver operates on the cropped system. Demand and gradient computation use full junction indices but only access the active region.

**Files:** New `kirchhoff.rs` methods for cropped solve. Modify `mod.rs` to compute active region.

### Expected Impact

For 300 cells on 100x100: active region ~30x30 = 3,600 junctions (vs 40,000). CG cost drops ~11x.

---

## Workstream 4: Algebraic Multigrid Preconditioner (Hydraulic)

### Algorithm

Geometric multigrid V-cycle exploiting the regular 2D grid structure.

**Coarsening:** Each level halves grid dimensions. Junction (x,y,port) on fine grid maps to (x/2, y/2, port) on coarse grid. Pipes between coarsened junctions are summed.

**V-cycle(level, r, z):**
1. If level is coarsest (<=16 junctions): solve directly via CG with Jacobi
2. Pre-smooth: 3 weighted Jacobi iterations on `A*z = r`
3. Compute residual: `r_coarse = restrict(r - A*z)`
4. Recurse: `e_coarse = V_cycle(level+1, r_coarse)`
5. Correct: `z += prolongate(e_coarse)`
6. Post-smooth: 3 weighted Jacobi iterations

**Restriction (fine to coarse):** Full-weighting stencil. Each coarse junction averages the 4 fine junctions that map to it.

**Prolongation (coarse to fine):** Bilinear interpolation. Each fine junction gets weighted sum of nearest coarse junctions.

### Integration

Replace `preconditioned_conjugate_gradient` Jacobi preconditioner:

```rust
// Old: z[i] = r[i] * inv_diag[i]
// New: z = multigrid_v_cycle(&hierarchy, level=0, r)
```

The multigrid hierarchy is built once per Kirchhoff solve (pipe resistances change with turbulence). Each V-cycle costs O(N) work (geometric series of coarsened grids).

### Data Structures

```rust
struct MultigridLevel {
    width: i32,
    height: i32,
    diag: Vec<f64>,
    off_diag: Vec<(usize, usize, f64)>,
    // Restriction/prolongation operators (sparse)
    restrict_map: Vec<Vec<(usize, f64)>>,  // fine->coarse weights
    prolong_map: Vec<Vec<(usize, f64)>>,   // coarse->fine weights
}

struct MultigridHierarchy {
    levels: Vec<MultigridLevel>,
}
```

**Files:** New `solver/multigrid.rs` (extend existing stub). Modify `kirchhoff.rs` to build hierarchy and use V-cycle preconditioner.

### Expected Impact

CG iterations drop from 100-300 to ~10-20. Combined with grid cropping, total inner-solve speedup is 50-100x.

---

## Implementation Order

1. **Workstream 2** (gradient clipping + turbulence) - quick wins, small changes
2. **Workstream 1** (ElectroPlace alignment) - largest code change, highest quality impact
3. **Workstream 3** (dynamic grid cropping) - biggest hydraulic speedup
4. **Workstream 4** (AMG preconditioner) - further speedup, most complex

## Verification

- `cargo build -p nextpnr` after each workstream
- `cargo test -p nextpnr --features test-utils` all existing tests pass
- Benchmark: `bench_place_route.py` on ch_intrinsics comparing all placers
- ElectroPlace HPWL should improve (closer to or better than HeAP)
- Hydraulic per-iteration time should drop from ~2-22s to <1s
- Hydraulic HPWL should improve (gradient clipping prevents plateau)

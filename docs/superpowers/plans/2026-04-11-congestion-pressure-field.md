# Congestion Pressure Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate per-edge congestion pressure from the Dijkstra path solver into the placement energy and gradient as part of the unified physical system -- no weighting parameter, equilibrium is when wirelength and congestion gradients balance.

**Architecture:** After each path solve, compute a per-node congestion pressure field from the excess resistance on adjacent pipes (`R_excess = R_eff - R_base`). Add `P_cong[node]` to `dist[node]` to form a unified potential `V[j] = dist[j] + P_cong[j]`. The energy and gradient use `V[j]` instead of `dist[j]`, so the congestion gradient flows through the same bilinear interpolation stencil. No weight parameter -- the pressure is in resistance units, same as dist.

**Tech Stack:** Rust, rayon (parallel iteration), existing PipeNetwork / path_solver infrastructure

---

### Task 1: Compute congestion pressure field from pipe excess resistance

**Files:**
- Create: `crates/nextpnr/src/placer/opt_trans/congestion.rs`
- Modify: `crates/nextpnr/src/placer/opt_trans/mod.rs`

The congestion pressure at each node is the mean excess resistance of adjacent pipes.
`R_excess(pipe) = R_eff - R_base = base * usage / max(eps, cap - usage)`.
When uncongested, R_excess = 0. When saturated, R_excess diverges.

- [ ] **Step 1: Write the failing test**

Create `crates/nextpnr/src/placer/opt_trans/congestion.rs`:

```rust
//! Congestion pressure field derived from pipe excess resistance.
//!
//! Each node's congestion pressure is the mean excess resistance of its
//! adjacent pipes. R_excess = R_eff - R_base, which is zero when a pipe
//! is uncongested and diverges as usage approaches capacity. This gives
//! the congestion gradient the same units as the Dijkstra distance field,
//! so both are part of a single unified energy functional.

use super::network::PipeNetwork;
use super::resistance::ResistanceModel;

/// Compute per-node congestion pressure from pipe excess resistance.
///
/// `P_cong[node] = mean_{adj pipes} (R_eff(pipe) - R_base(pipe))`
///
/// Returns a Vec<f64> of length `network.num_nodes()`.
pub fn compute_congestion_pressure(
    network: &PipeNetwork,
    resistance_model: &ResistanceModel,
) -> Vec<f64> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::opt_trans::network::{Direction, Node, Pipe, PipeNetwork, PipeType};
    use rustc_hash::FxHashMap;

    fn make_pipe(from: usize, to: usize, base: f64, capacity: f64, net_count: u32) -> Pipe {
        Pipe {
            from,
            to,
            base_resistance: base,
            capacity,
            flow: 0.0,
            net_count,
            raw_cell_density: 0.0,
            cell_density: 0.0,
            eff_conductance: 1.0, // will be set by test
            pipe_type: PipeType::InterTile(Direction::East),
        }
    }

    fn make_network(nodes: Vec<Node>, pipes: Vec<Pipe>) -> PipeNetwork {
        let n = nodes.len();
        let mut node_pipes = vec![Vec::new(); n];
        for (i, pipe) in pipes.iter().enumerate() {
            node_pipes[pipe.from].push(i);
            node_pipes[pipe.to].push(i);
        }
        PipeNetwork {
            nodes,
            pipes,
            node_pipes,
            pipe_lookup: FxHashMap::default(),
            width: 2,
            height: 2,
            x0: 0,
            y0: 0,
            zero_bel_tiles: 0,
            total_bels: 0,
            coarsen: 1,
        }
    }

    #[test]
    fn zero_usage_gives_zero_pressure() {
        let nodes = vec![
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
        ];
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 0)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        assert_eq!(pressure.len(), 2);
        assert!(pressure[0].abs() < 1e-12, "expected 0, got {}", pressure[0]);
        assert!(pressure[1].abs() < 1e-12, "expected 0, got {}", pressure[1]);
    }

    #[test]
    fn half_saturated_gives_base_resistance_pressure() {
        let nodes = vec![
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
        ];
        // usage=5, cap=10: R_eff = 1*10/5 = 2, R_excess = 2-1 = 1
        let pipes = vec![make_pipe(0, 1, 1.0, 10.0, 5)];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        // Each node has 1 adjacent pipe, so mean = R_excess = 1.0
        assert!((pressure[0] - 1.0).abs() < 1e-9, "got {}", pressure[0]);
        assert!((pressure[1] - 1.0).abs() < 1e-9, "got {}", pressure[1]);
    }

    #[test]
    fn node_with_multiple_pipes_averages() {
        // Node 1 is connected to node 0 (uncongested) and node 2 (congested)
        let nodes = vec![
            Node { tile_x: 0, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 1, tile_y: 0, pressure: 0.0 },
            Node { tile_x: 2, tile_y: 0, pressure: 0.0 },
        ];
        let pipes = vec![
            make_pipe(0, 1, 1.0, 10.0, 0), // R_excess = 0
            make_pipe(1, 2, 1.0, 10.0, 5), // R_excess = 1.0
        ];
        let network = make_network(nodes, pipes);
        let model = ResistanceModel;
        let pressure = compute_congestion_pressure(&network, &model);
        // Node 0: 1 pipe, R_excess=0 → P=0
        assert!(pressure[0].abs() < 1e-12);
        // Node 1: 2 pipes, mean(0, 1) = 0.5
        assert!((pressure[1] - 0.5).abs() < 1e-9, "got {}", pressure[1]);
        // Node 2: 1 pipe, R_excess=1.0 → P=1.0
        assert!((pressure[2] - 1.0).abs() < 1e-9, "got {}", pressure[2]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p nextpnr congestion::tests -- --nocapture`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement `compute_congestion_pressure`**

Replace `todo!()` with:

```rust
pub fn compute_congestion_pressure(
    network: &PipeNetwork,
    resistance_model: &ResistanceModel,
) -> Vec<f64> {
    let n = network.num_nodes();
    let mut pressure = vec![0.0f64; n];
    let mut degree = vec![0u32; n];

    for pipe in &network.pipes {
        let r_eff = resistance_model.effective_resistance(pipe);
        let r_excess = r_eff - pipe.base_resistance;
        // r_excess is >= 0 by construction (R_eff >= R_base)
        pressure[pipe.from] += r_excess;
        pressure[pipe.to] += r_excess;
        degree[pipe.from] += 1;
        degree[pipe.to] += 1;
    }

    for i in 0..n {
        if degree[i] > 0 {
            pressure[i] /= degree[i] as f64;
        }
    }

    pressure
}
```

- [ ] **Step 4: Register the module**

In `crates/nextpnr/src/placer/opt_trans/mod.rs`, add:

```rust
pub mod congestion;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p nextpnr congestion::tests -- --nocapture`
Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add crates/nextpnr/src/placer/opt_trans/congestion.rs crates/nextpnr/src/placer/opt_trans/mod.rs
git commit -m "feat(opt_trans): add congestion pressure field from pipe excess resistance"
```

---

### Task 2: Add congestion pressure to `PathForwardStats` and compute it after path solve

**Files:**
- Modify: `crates/nextpnr/src/placer/opt_trans/path_solver.rs:117-122` (PathForwardStats)
- Modify: `crates/nextpnr/src/placer/opt_trans/algorithm.rs:192-231` (compute_kirchhoff_gradient_with_system, compute_flux_forward_stats)

The congestion pressure field must be available alongside `edge_usage` and `dist` so the gradient can use it.

- [ ] **Step 1: Add `congestion_pressure` field to `PathForwardStats`**

In `crates/nextpnr/src/placer/opt_trans/path_solver.rs`, modify `PathForwardStats`:

```rust
#[derive(Debug, Clone)]
pub struct PathForwardStats {
    pub global_pressure: Vec<f64>,
    pub edge_usage: Vec<f64>,
    pub attraction_energy: f64,
    /// Per-node congestion pressure from pipe excess resistance.
    pub congestion_pressure: Vec<f64>,
}
```

Update the construction in `compute_path_forward_and_gradient_impl` (around line 332):

```rust
    let forward = PathForwardStats {
        global_pressure: vec![0.0; n_nodes],
        edge_usage: accum.edge_usage,
        attraction_energy: accum.attraction_energy,
        congestion_pressure: Vec::new(), // populated by caller after net_count refresh
    };
```

- [ ] **Step 2: Compute and attach congestion pressure in algorithm.rs**

In `crates/nextpnr/src/placer/opt_trans/algorithm.rs`, modify `compute_kirchhoff_gradient_with_system` to compute congestion pressure after `refresh_net_counts_from_usage`:

```rust
fn compute_kirchhoff_gradient_with_system(
    network: &mut PipeNetwork,
    net_infos: &[NetSolveInfo],
    cfg: &OptTransPlacerCfg,
    solve_pool: &rayon::ThreadPool,
    n: usize,
    objective_scale: f64,
    resistance_model: &ResistanceModel,
) -> (Vec<f64>, f64, Vec<f64>, path_solver::PathStats) {
    let (grad, mut forward, stats) = path_solver::compute_path_forward_and_gradient(
        network,
        net_infos,
        cfg,
        solve_pool,
        n,
        objective_scale,
    );
    update_pressure_and_flow(network, &forward.global_pressure, solve_pool);
    refresh_net_counts_from_usage(solve_pool, network, &forward.edge_usage);
    let cong_pressure = super::congestion::compute_congestion_pressure(network, resistance_model);
    forward.congestion_pressure = cong_pressure.clone();
    (grad, forward.attraction_energy, cong_pressure, stats)
}
```

- [ ] **Step 3: Update all call sites of `compute_kirchhoff_gradient_with_system`**

In the main loop in `algorithm.rs` (around line 582), update the call:

```rust
        let (grad, energy, cong_pressure, path_stats) = compute_kirchhoff_gradient_with_system(
            &mut network,
            &net_infos,
            &cfg,
            &solve_pool,
            n,
            1.0,
            &resistance_model,
        );
```

The `cong_pressure` variable is now available for Task 3.

- [ ] **Step 4: Verify compilation**

Run: `cargo build -p nextpnr --release 2>&1 | tail -5`
Expected: compiles with no errors (warnings OK)

- [ ] **Step 5: Commit**

```bash
git add crates/nextpnr/src/placer/opt_trans/path_solver.rs crates/nextpnr/src/placer/opt_trans/algorithm.rs
git commit -m "feat(opt_trans): compute congestion pressure after path solve"
```

---

### Task 3: Add congestion gradient to `accumulate_sink_gradient`

**Files:**
- Modify: `crates/nextpnr/src/placer/opt_trans/path_solver.rs:344-415` (evaluate_net_path)
- Modify: `crates/nextpnr/src/placer/opt_trans/path_solver.rs:555-581` (accumulate_sink_gradient)

The gradient uses `V[j] = dist[j] + P_cong[j]` instead of just `dist[j]`. This makes the congestion part of the unified potential.

- [ ] **Step 1: Thread `congestion_pressure` into `evaluate_net_path`**

Modify the signature and body of `evaluate_net_path`:

```rust
fn evaluate_net_path(
    network: &PipeNetwork,
    info: &NetSolveInfo,
    cfg: &OptTransPlacerCfg,
    objective_scale: f64,
    include_gradient: bool,
    cong_pressure: &[f64],
    local: &mut LocalAccum,
    ws: &mut PathSolverWorkspace,
) {
    // ... existing Dijkstra solve unchanged ...

    let timing_scale = net_path_weight(info, cfg) * objective_scale;
    for pin in &info.pin_data {
        if pin.is_driver {
            continue;
        }

        let Some(cost) = sink_cost(pin, &ws.dist) else {
            if !pin.is_fixed && pin.cell_idx.is_some() {
                local.stats.failures += 1;
                warn!(
                    "path solver: disconnected movable sink on net {:?} ({})",
                    info.net_id, info.debug_name
                );
            }
            continue;
        };

        // Congestion energy: same bilinear interpolation, using P_cong instead of dist
        let cong_cost = sink_cost_from_field(pin, cong_pressure);
        local.attraction_energy += timing_scale * (cost + cong_cost);
        mark_sink_paths(pin, source_node, network, ws, &mut local.edge_usage);

        if include_gradient {
            accumulate_sink_gradient_unified(pin, &ws.dist, cong_pressure, timing_scale, &mut local.grad);
        }
    }
}
```

- [ ] **Step 2: Add `sink_cost_from_field` helper**

```rust
/// Bilinear-interpolated cost from an arbitrary per-node field (e.g. congestion pressure).
fn sink_cost_from_field(pin: &NetPinData, field: &[f64]) -> f64 {
    let mut cost = 0.0;
    for j in 0..4 {
        let w = pin.weights[j];
        if w == 0.0 {
            continue;
        }
        let v = field[pin.nodes[j]];
        if !v.is_finite() {
            continue;
        }
        cost += w * v;
    }
    cost
}
```

- [ ] **Step 3: Replace `accumulate_sink_gradient` with unified version**

```rust
/// Accumulate gradient from the unified potential V[j] = dist[j] + P_cong[j].
fn accumulate_sink_gradient_unified(
    pin: &NetPinData,
    dist: &[f64],
    cong_pressure: &[f64],
    timing_scale: f64,
    grad: &mut [f64],
) {
    let Some(ci) = pin.cell_idx else {
        return;
    };
    if pin.is_fixed || pin.is_driver {
        return;
    }

    let n = grad.len() / 2;
    if ci >= n {
        return;
    }

    let mut grad_x = 0.0;
    let mut grad_y = 0.0;
    for j in 0..4 {
        let d = dist[pin.nodes[j]];
        if !d.is_finite() {
            return;
        }
        let v = d + cong_pressure[pin.nodes[j]];
        grad_x += pin.dw_dx[j] * v;
        grad_y += pin.dw_dy[j] * v;
    }

    grad[ci] += timing_scale * grad_x;
    grad[n + ci] += timing_scale * grad_y;
}
```

- [ ] **Step 4: Update all call sites of `evaluate_net_path` in `compute_path_forward_and_gradient_impl`**

The `cong_pressure` needs to be passed into the parallel fold. Since `cong_pressure` is computed from the *previous* iteration's `net_count` (which is already on `network.pipes`), we compute it once before the parallel solve:

In `compute_path_forward_and_gradient_impl`, before the parallel fold, add:

```rust
    let cong_pressure = super::congestion::compute_congestion_pressure(
        network,
        &super::resistance::ResistanceModel,
    );
```

Then thread `&cong_pressure` into each `evaluate_net_path` call inside the fold:

```rust
                    for info in chunk {
                        evaluate_net_path(
                            network,
                            info,
                            cfg,
                            objective_scale,
                            include_gradient,
                            &cong_pressure,
                            &mut local,
                            &mut ws,
                        );
                    }
```

- [ ] **Step 5: Verify compilation and existing tests**

Run: `cargo test -p nextpnr -- --nocapture 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/nextpnr/src/placer/opt_trans/path_solver.rs
git commit -m "feat(opt_trans): integrate congestion pressure into unified energy and gradient"
```

---

### Task 4: Add congestion reporting to iteration log

**Files:**
- Modify: `crates/nextpnr/src/placer/opt_trans/algorithm.rs` (iteration logging, around lines 800-860)

Report the congestion pressure statistics so we can see it activating.

- [ ] **Step 1: Compute and log congestion stats per reported iteration**

After computing `cong_pressure` in the main loop, add summary stats to the iteration log line. Compute:
- `cong_max`: max P_cong over all nodes
- `cong_mean`: mean P_cong over all nodes
- `cong_active`: count of nodes with P_cong > 0.01

Add to the existing `eprintln!` format in the iteration report (the line starting with `"  I"`):

```
cong_max={:.3} cong_mean={:.3} cong_active={}
```

Use the `cong_pressure` Vec that is now returned from `compute_kirchhoff_gradient_with_system`.

- [ ] **Step 2: Verify the log output**

Run: `NPNR_OT_TRACE_MAX_ITERS=10 cargo run --release --example ot_trace_stereovision3 2>&1 | head -30`
Expected: iteration lines now show `cong_max`, `cong_mean`, `cong_active` fields

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/placer/opt_trans/algorithm.rs
git commit -m "feat(opt_trans): report congestion pressure stats in iteration log"
```

---

### Task 5: Smoke test -- 50-iter run and verify congestion gradient activates

**Files:** No code changes. Validation only.

- [ ] **Step 1: Run 50-iteration placement and inspect output**

Run: `NPNR_OT_TRACE_MAX_ITERS=50 cargo run --release --example ot_trace_stereovision3 2>&1`

Verify:
- Early iterations (0-5): `cong_mean` should be near zero (design is spread)
- Later iterations (30-50): `cong_mean` should be non-zero as cells cluster
- Energy should still converge (may converge to a different value than before)
- No panics, no NaN, no infinite energy
- `bins_over` may change compared to the pure-wirelength run

- [ ] **Step 2: Compare with baseline**

The pre-congestion 50-iter result was: CHPWL=13,967, energy=1,462, post-legal HPWL=13,902.

Compare the new result. Expected behavior:
- CHPWL may be slightly higher (cells spread to avoid congestion)
- Congestion metrics should be lower
- Post-legalization HPWL should remain reasonable

- [ ] **Step 3: Run 200-iter and compare with baseline**

Run: `NPNR_OT_TRACE_MAX_ITERS=200 cargo run --release --example ot_trace_stereovision3 2>&1`

Baseline 200-iter: CHPWL=9,864, post-legal HPWL=9,857.
Compare energy convergence rate and final quality.

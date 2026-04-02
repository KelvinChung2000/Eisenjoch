# A* Router Improvements Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make A* routing viable on dense FPGA routing graphs (XC7: 3929 PIPs/tile, 608 wires/tile) so the raster router can complete multi-pass rip-up-reroute within ~60s for stereovision3 (258 nets, 313-sink high-fanout net).

**Architecture:** Three-pronged attack: (1) tighter search pruning to reduce visits per call, (2) smarter high-fanout net handling to avoid calling A* hundreds of times, (3) cached lookahead to eliminate repeated expensive builds. Each improvement is independent and stacks multiplicatively.

**Tech Stack:** Rust, XC7 hybrid chipdb, raster router (beam search + A* cleanup), maze router (A* rip-up-reroute)

---

## Problem Analysis

The current A* (`astar_route` in `maze.rs`) has a visit budget of `grid_area * 10 = 309K` per call. On XC7, each visit expands ~3929 downhill PIPs plus node wires. For a single sink, one call can take 10+ seconds. For a 313-sink net, the per-net A* cleanup calls A* up to 313 times sequentially.

The raster router's A* cleanup hangs because:
1. The 30s timeout checks between nets, not between sinks within a net
2. Visit budget (309K) is designed for full routing, not cleanup
3. High-fanout nets dominate runtime: 1 net with 313 sinks = 313 A* calls
4. The Lookahead table (608 classes x 81x81 offsets) is rebuilt each pass

## File Structure

| File | Responsibility | Changes |
|------|---------------|---------|
| `crates/nextpnr/src/router/maze.rs` | A* search engine | Add wire-class pruning, tighter visit budgets |
| `crates/nextpnr/src/router/raster.rs` | Raster router (beam + A* cleanup) | Steiner tree cleanup, per-sink timeout, cached lookahead |
| `crates/nextpnr/src/router/lookahead.rs` | Precomputed delay estimates | Add directional cone pruning, lazy construction |
| `crates/nextpnr/src/router/common.rs` | Shared routing utilities | Add sink ordering by distance |

---

### Task 1: Per-sink timeout in A* cleanup

The most critical fix. Currently a single high-fanout net can hang the entire cleanup phase because the 30s budget only checks between nets.

**Files:**
- Modify: `crates/nextpnr/src/router/raster.rs:1482-1541`

- [ ] **Step 1: Add per-sink elapsed check inside the sink loop**

In `raster.rs`, the A* cleanup loop at line 1503 iterates over sinks without checking elapsed time. Add a check before each `astar_route` call:

```rust
// In the cleanup sink loop (raster.rs ~line 1503):
for &sink_wire in &sink_wires {
    // Bail out of this net if cleanup budget exhausted.
    if cleanup_start.elapsed() > cleanup_budget {
        all_ok = false;
        break;
    }

    if tree_wires.contains(&sink_wire) {
        sink_routes.push(SinkRoute { sink_wire, pips: vec![] });
        continue;
    }

    match astar_route(
        ctx, &tree_wires, sink_wire,
        &wire_penalty, None, 50, Some(&lookahead),
        Some(10_000), // tight visit limit for cleanup
    ) {
```

- [ ] **Step 2: Skip high-fanout nets in A* cleanup**

High-fanout nets (>50 sinks) are better handled by the beam search rip-up-reroute than by A* cleanup. Skip them:

```rust
// After collecting sink_wires, before the sink loop:
let sink_wires = collect_sink_wires(ctx, net);

// Skip high-fanout nets — too expensive for per-sink A*.
// Let rip-up-reroute handle them in subsequent passes.
if sink_wires.len() > 50 {
    continue;
}
```

- [ ] **Step 3: Build and run test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
cp target/release/libnextpnr.so python/nextpnr/nextpnr.cpython-313-x86_64-linux-gnu.so
```

Verify the raster router no longer hangs on stereovision3:

```python
import nextpnr
ctx = nextpnr.Context(chipdb='chip_database/xc7_hybrid.bin')
ctx.load_design('benchmark/output/stereovision3.json')
ctx.pack()
ctx.place(placer='heap', seed=42)
ctx.route(router='raster', max_iterations=3, skip_unplaced=True)
print(f"Routed WL: {ctx.total_routed_wirelength()}")
```

Expected: completes in <60s instead of hanging.

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/router/raster.rs
git commit -m "fix(router): per-sink timeout + skip high-fanout in A* cleanup"
```

---

### Task 2: Tighter A* visit budget with configurable limit

The default visit budget (309K) is appropriate for full routing but wastes time during cleanup. The `visit_limit` parameter was added but needs proper integration.

**Files:**
- Modify: `crates/nextpnr/src/router/maze.rs:370-406`

- [ ] **Step 1: Scale visit budget by Manhattan distance**

Instead of a flat grid-area budget, scale by the expected search radius. A nearby sink needs far fewer visits than a distant one:

```rust
pub fn astar_route(
    ctx: &Context,
    src_wires: &FxHashSet<WireId>,
    dst_wire: WireId,
    wire_penalty: &FxHashMap<WireId, DelayT>,
    bbox: Option<&crate::metrics::BoundingBox>,
    estimate_precision: DelayT,
    lookahead: Option<&super::lookahead::Lookahead>,
    visit_limit: Option<usize>,
) -> Option<Vec<PipId>> {
    if src_wires.contains(&dst_wire) {
        return Some(Vec::new());
    }

    let chipdb = ctx.chipdb();

    // Compute Manhattan distance to nearest source for adaptive budget.
    let (dst_x, dst_y) = chipdb.tile_xy(dst_wire.tile());
    let min_manhattan: i32 = src_wires.iter().map(|&sw| {
        let (sx, sy) = chipdb.tile_xy(sw.tile());
        (sx - dst_x).abs() + (sy - dst_y).abs()
    }).min().unwrap_or(0);

    // Adaptive budget: proportional to search area (manhattan^2),
    // with a per-tile fanout multiplier. Capped at grid_area * 10.
    let grid_area = (chipdb.width() as usize) * (chipdb.height() as usize);
    let distance_budget = ((min_manhattan as usize + 5).pow(2))
        .saturating_mul(100)
        .max(5_000);
    let default_budget = distance_budget.min(grid_area.saturating_mul(10)).max(5_000);
    let mut max_visits: usize = visit_limit.unwrap_or(default_budget);
    let mut visit_count: usize = 0;
    // ... rest unchanged
```

- [ ] **Step 2: Early termination when first path found in cleanup mode**

When `visit_limit` is explicitly set (cleanup mode), break immediately after finding any path instead of searching for a better one:

```rust
// In the destination hit section (~line 444):
if entry.wire == dst_wire {
    let score = entry.cost + entry.penalty;
    if score < best_score {
        best_score = score;
        // In cleanup mode (explicit visit_limit), accept first path.
        if visit_limit.is_some() {
            break;
        }
        // Normal mode: set adaptive limit and continue searching.
        if max_visits > visit_count * 2 + 100 {
            max_visits = visit_count * 2 + 100;
        }
    }
    continue;
}
```

- [ ] **Step 3: Build and test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
PYO3_PYTHON=.venv/bin/python3 cargo test --release -p nextpnr -- astar
```

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/router/maze.rs
git commit -m "perf(router): adaptive A* visit budget scaled by Manhattan distance"
```

---

### Task 3: Sink ordering by distance (nearest-first)

When routing a multi-sink net, routing nearby sinks first grows the tree toward the destination, making subsequent sinks cheaper. Currently sinks are routed in arbitrary order.

**Files:**
- Modify: `crates/nextpnr/src/router/maze.rs:294-306` (compute_route_r1)
- Modify: `crates/nextpnr/src/router/raster.rs:1495-1501` (cleanup)

- [ ] **Step 1: Sort sinks by Manhattan distance to nearest tree wire**

In `compute_route_r1` (`maze.rs` ~line 294), after collecting sink_wires, sort them by distance to the source:

```rust
let mut sink_wires = collect_sink_wires(ctx, net);

// Route nearest sinks first — grows tree toward distant sinks,
// making each subsequent A* call cheaper.
let chipdb = ctx.chipdb();
let (src_x, src_y) = chipdb.tile_xy(source_wire.tile());
sink_wires.sort_by_key(|&sw| {
    let (wx, wy) = chipdb.tile_xy(sw.tile());
    (wx - src_x).abs() + (wy - src_y).abs()
});
```

- [ ] **Step 2: Apply same sorting in raster cleanup**

In `raster.rs` cleanup section (~line 1495), after `collect_sink_wires`:

```rust
let mut sink_wires = collect_sink_wires(ctx, net);
// Sort nearest-first for efficient tree growth.
let (src_x, src_y) = ctx.chipdb().tile_xy(source_wire.tile());
sink_wires.sort_by_key(|&sw| {
    let (wx, wy) = ctx.chipdb().tile_xy(sw.tile());
    (wx - src_x).abs() + (wy - src_y).abs()
});
```

- [ ] **Step 3: Build and test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
```

Verify routing quality unchanged on stereovision3.

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/router/maze.rs crates/nextpnr/src/router/raster.rs
git commit -m "perf(router): sort sinks nearest-first for cheaper A* calls"
```

---

### Task 4: Cache lookahead table across routing passes

The Lookahead table is rebuilt from scratch each time it's needed. For the raster router running multiple passes, this wastes significant time. Cache it.

**Files:**
- Modify: `crates/nextpnr/src/router/raster.rs:1464-1469` (cleanup section)
- Modify: `crates/nextpnr/src/router/raster.rs` (main route function: build once, reuse)

- [ ] **Step 1: Build lookahead once at the start of routing**

In the raster router's main `route_nets` function, build the lookahead table before the iteration loop and pass it to the cleanup phase:

```rust
// At the top of route_nets, before the pass loop:
let lookahead = super::lookahead::Lookahead::build(
    ctx.chipdb(),
    ctx.speed_grade_idx(),
    40,
);
```

Then remove the per-pass rebuild inside the cleanup section (~line 1464-1469) and use the cached one.

- [ ] **Step 2: Share lookahead with Router1 if both are used**

Move the lookahead into an `Arc<Lookahead>` at the router module level so Router1 and raster can share it:

```rust
// In route_nets, replace the cleanup-local build with the pre-built one:
// REMOVE:
//   let lookahead = super::lookahead::Lookahead::build(...);
// USE the one built at the top of route_nets.
```

- [ ] **Step 3: Build and test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
```

Verify routing still works and observe reduced per-pass overhead.

- [ ] **Step 4: Commit**

```bash
git add crates/nextpnr/src/router/raster.rs
git commit -m "perf(router): cache lookahead table across routing passes"
```

---

### Task 5: Bounding-box pruning for A* cleanup

The A* cleanup calls pass `bbox: None`, exploring the entire chip. Adding a bounding box around the net's pins cuts the search space dramatically.

**Files:**
- Modify: `crates/nextpnr/src/router/raster.rs:1490-1520` (cleanup A* calls)

- [ ] **Step 1: Compute per-net bounding box in cleanup**

Before the sink loop in cleanup, compute a bbox from the net's pin locations with margin:

```rust
let source_wire = match resolve_source_wire(ctx, net) {
    Ok(Some(w)) => w,
    _ => continue,
};

// Compute bounding box for A* pruning.
let bbox = crate::metrics::compute_bbox(ctx, net, 5); // 5-tile margin

let sink_wires = collect_sink_wires(ctx, net);
```

Then pass `Some(&bbox)` to `astar_route` instead of `None`:

```rust
match astar_route(
    ctx, &tree_wires, sink_wire,
    &wire_penalty, Some(&bbox), 50, Some(&lookahead),
    Some(10_000),
) {
```

- [ ] **Step 2: Build and test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
```

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/router/raster.rs
git commit -m "perf(router): bounding-box pruning for A* cleanup calls"
```

---

### Task 6: Directional cone pruning in A*

Beyond bounding-box, add a directional cone that limits expansion to wires roughly between source and destination. This is tighter than a bbox for long-distance routes.

**Files:**
- Modify: `crates/nextpnr/src/router/maze.rs:491-541` (PIP expansion loop)

- [ ] **Step 1: Add directional progress check in PIP expansion**

After the bounding box check but before computing pip_delay, skip wires that move away from the destination when no path has been found yet:

```rust
// PIP expansion loop, after bbox check:
for &pip_idx in downhill {
    let pip = PipId::new(entry.wire.tile(), pip_idx);
    let next_wire = chipdb.pip_dst_wire(pip);

    // Bounding box pruning.
    if let Some(bb) = bbox {
        let (wx, wy) = chipdb.tile_xy(next_wire.tile());
        if !bb.contains(wx, wy) {
            continue;
        }
    }

    // Directional cone pruning: if we haven't found any path yet,
    // skip wires that increase Manhattan distance by more than 50%.
    // This aggressively prunes the search fan on dense graphs.
    if best_score == DelayT::MAX {
        let (nx, ny) = chipdb.tile_xy(next_wire.tile());
        let (ex, ey) = chipdb.tile_xy(entry.wire.tile());
        let old_dist = (ex - dst_x).abs() + (ey - dst_y).abs();
        let new_dist = (nx - dst_x).abs() + (ny - dst_y).abs();
        // Allow slight backwards movement (exploration), but not much.
        if new_dist > old_dist + 2 {
            continue;
        }
    }

    let pip_delay = ctx.pip(pip).delay().max_delay();
    // ... rest unchanged
```

- [ ] **Step 2: Build and test**

```bash
PYO3_PYTHON=.venv/bin/python3 cargo build --release
```

Test on stereovision3 to verify routability is maintained (cone pruning can miss valid paths on irregular architectures, so verify the routed net count doesn't drop).

- [ ] **Step 3: Commit**

```bash
git add crates/nextpnr/src/router/maze.rs
git commit -m "perf(router): directional cone pruning in A* PIP expansion"
```

---

### Task 7: Integration test - full routing benchmark

**Files:**
- Create: `python/nextpnr/benchmarks/test_router_perf.py`

- [ ] **Step 1: Write benchmark script**

```python
"""Router performance benchmark: HeAP + raster on all designs."""
import nextpnr, time, os

CHIPDB = '/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin'
BENCH_DIR = '/home/kelvin/side-project/eisenjoch/benchmark/output'

DESIGNS = [
    'stereovision3', 'ch_intrinsics', 'diffeq1', 'diffeq2',
    'mkPktMerge', 'sha', 'blob_merge', 'stereovision0',
    'stereovision1', 'stereovision2', 'raygentop', 'spree', 'or1200',
]

results = []
for design in DESIGNS:
    path = os.path.join(BENCH_DIR, f'{design}.json')
    if not os.path.exists(path):
        continue
    for placer in ['heap', 'hydraulic']:
        pkw = {'subtile_resolution': 1, 'step_scale': 0.5} if placer == 'hydraulic' else {}
        ctx = nextpnr.Context(chipdb=CHIPDB)
        ctx.load_design(path)
        ctx.pack()
        t0 = time.time()
        try:
            ctx.place(placer=placer, seed=42, **pkw)
        except Exception as e:
            print(f"  {design}/{placer}: place failed: {e}")
            continue
        pt = time.time() - t0
        hpwl = ctx.total_hpwl()
        line = ctx.total_line_estimate()

        t0 = time.time()
        try:
            ctx.route(router='raster', max_iterations=5, skip_unplaced=True)
            ok = True
        except Exception:
            ok = False
        rt = time.time() - t0
        rwl = ctx.total_routed_wirelength()
        results.append({
            'design': design, 'placer': placer,
            'hpwl': hpwl, 'line': line, 'rwl': rwl, 'ok': ok, 'pt': pt, 'rt': rt,
        })
        print(f"  {design:<16} {placer:<10} HPWL={hpwl:>8.0f} line={line:>8.0f} "
              f"rwl={rwl:>8} ok={ok} place={pt:.1f}s route={rt:.1f}s")

print(f"\n{'Design':<16} {'Placer':<10} {'HPWL':>8} {'Line':>8} {'RoutedWL':>9} {'PlcT':>6} {'RteT':>6} OK")
print('-' * 80)
for r in results:
    print(f"{r['design']:<16} {r['placer']:<10} {r['hpwl']:>8.0f} {r['line']:>8.0f} "
          f"{r['rwl']:>9} {r['pt']:>5.1f}s {r['rt']:>5.1f}s {r['ok']}")
```

- [ ] **Step 2: Run benchmark**

```bash
uv run python python/nextpnr/benchmarks/test_router_perf.py
```

Expected: all designs complete routing within 5 minutes total. stereovision3 completes in <60s.

- [ ] **Step 3: Commit**

```bash
git add python/nextpnr/benchmarks/test_router_perf.py
git commit -m "test(router): full routing performance benchmark"
```

---

## Expected Impact

| Improvement | Estimated Speedup | Risk |
|------------|-------------------|------|
| Task 1: Per-sink timeout + skip high-fanout | 10-100x for cleanup phase | Low (graceful degradation) |
| Task 2: Adaptive visit budget | 2-5x per A* call | Low (reduces wasted exploration) |
| Task 3: Nearest-first sink ordering | 1.5-2x per multi-sink net | None (pure optimization) |
| Task 4: Cached lookahead | 2-3x per pass (saves rebuild) | None (same result, less work) |
| Task 5: Bbox pruning for cleanup | 2-5x per A* call | Low (bbox may be too tight on rare cases) |
| Task 6: Directional cone pruning | 2-10x on long-distance routes | Medium (may miss valid paths on irregular routing) |

Tasks 1-5 are safe and should be done first. Task 6 is more aggressive and needs careful validation.

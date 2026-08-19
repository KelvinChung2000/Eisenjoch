# Why the opt_trans memory work isn't showing up on FPGA01

Status: investigation in progress. Evidence-first; no fix has been applied yet.

## The complaint

A large amount of memory-reduction work has landed in `opt_trans`, yet FPGA01
(`xc7_large`, 76660 cells, 69353 nodes, 2562403 pipes) still drives RSS to
~46-50 GB and has to be abandoned as a measurement vehicle.

## The memory work IS landing

Measured, not assumed. From the FPGA01 run's own probes
(`measurements/harden_gate/fpga01_A_control_bounded.log`, outer=0):

    cache_entries=3801390 cache_cap=6417306 cache_est_mb=60
    pre_refresh_rss=1700MB post_pre_solve_rss=1802MB

That 60 MB was measured with the tightened caps (`NPNR_OT_CACHE_RADIUS=3`).
At the **default** radius of 12 the same design reports:

    cache_entries=42424857 cache_cap=69325648 cache_est_mb=600
    pre_refresh_rss=1707MB post_pre_solve_rss=2445MB

So the honest default-configuration figure is 600 MB, and it scales with
radius^2. Still ~50x better than the ~29 GB dense layout it replaced, but
`NPNR_OT_CACHE_RADIUS` is a first-order memory knob, not a tuning detail.

Every one of the shipped reductions is working as designed:

| Reduction | Evidence it works |
| --- | --- |
| Sparse `DistCache` (`FxHashMap` rows) | 600 MB at default radius (60 MB at radius 3), vs ~29 GB for the old dense `n_nets x n_nodes` `Vec<f32>` |
| Cache-radius filter (`NPNR_OT_CACHE_RADIUS`) | bounds the cache despite `settle_avg=7925`, `settle_max=69353`; radius 12 -> 600 MB, radius 3 -> 60 MB |
| Binary heap replacing bucket-Dial | `buckets` is now dead (`Vec::new()`); the documented `buckets.resize()` multi-GB path is gone |
| `WorkspacePool` | `init_count=62` checkouts on 32 cores, bounded by concurrency not by 105117 solves |
| Sparse chunked edge usage (`ChunkUsage`) | applied on the default `DialLogit` path |
| `SpanCostTable` / `TwoLevelPip` | cost cache keyed by span, not pipe identity |

So the premise "the reductions are not working" is **false as stated**. They
work. The problem is that they do not cover where the memory actually goes.

## Static baseline is fully accounted

New one-shot `MEM_STATIC` probe, FPGA01:

    pipes=352MB nodes=1MB adjacency=78MB pipe_cost_vecs=59MB
    validity_mask=634MB n_pipes=2562403 n_edges=5124806 rss=1677MB

1124 MB of the 1677 MB baseline is named; the rest is chipdb mmap, design,
`net_infos`, `cell_net_map`, `MuxSlotTracker`. Nothing anomalous in the
baseline -- but see finding 1.

## Finding 1 (CONFIRMED, independent of the blowup): validity mask stores bucket data per cell

`CellValidityMask` (`placer/common.rs`) is a bitset indexed
`cell_idx * W*H + gy*W + gx`, so it costs `n_cells x W x H` bits =
**634 MB** on FPGA01.

`CellValidityMask::build` derives each cell's bits *only* from
`ctx.resolve_bucket(cell_type)` -- the contents are a function of the bucket,
never of the cell. FPGA01 has two buckets (DFF, LUT6), and the log confirms it:

    per-cell valid positions: min=67346 avg=67346.0 max=67346

That is one 8.7 KB mask per bucket, duplicated 76660 times.

Fix shape: store `bucket_masks: Vec<Vec<u64>>` plus `cell_bucket: Vec<u16>`.
Same O(1) `is_valid`, one extra indirection. 634 MB -> ~170 KB.

This is in the 1.7 GB baseline, so it is NOT the 44 GB. It is a real and
separate win.

## Finding 2 (CONFIRMED): the sparse-usage fix was rolled out incompletely

`05bae52` / `823d2cd` replaced the `fold(|| vec![0.0; n_pipes]) ... reduce`
shape -- a 20.5 MB allocation per rayon split plus a full-chip-width merge --
with sparse `ChunkUsage` on the default `DialLogit` path.

The Bresenham variants still run the old shape:
`coord_descent.rs:877`, `:926`, `:959`, `:993` (and `path_solver.rs:765`,
`:791`). `place_dcd_sweep_colored_gs` additionally builds a fresh
`PathSolverWorkspace::new(n_nodes, n_pipes)` per rayon split at
`coord_descent.rs:3058` instead of taking one from `ws_pool` -- each of those
is a 20.5 MB `edge_load`.

Neither is on the default path (`JacobiFullscan` + `DialLogit`), so neither
explains FPGA01. Both are latent re-emergences of an already-fixed bug.

## Finding 3 (ROOT CAUSE, CONFIRMED): `ChunkUsage` is O(total work)

### Where it happens

Measured on FPGA01 under a 12 GB cage, `NPNR_OT_MAX_ITERS=1`:

    MEM_STATIC: ... rss=1677MB
    PHASE_MARK outer=0 enter=pre_solve    rss=1707MB
    PHASE_MARK outer=0 enter=dcd_sweep    rss=2445MB
    PHASE_MARK outer=0 leave=dcd_sweep    rss=2450MB   <- sweep is FLAT
    PHASE_MARK outer=0 enter=post_refresh rss=2450MB
    RSS_SAMPLE t=194.6s rss=10251MB
    RSS_SAMPLE t=195.1s rss=11823MB               <- OOM-killed at the 12 GB cap

The `JacobiFullscan` sweep costs 5 MB. All growth is inside
`solve_usage_and_energy`, at roughly 600 MB/s, monotone.

### The mechanism

`coord_descent.rs:803`:

```rust
struct ChunkUsage { usage: Vec<(u32, f64)>, ... }
// doc comment: "duplicates are fine because the merge just adds them"

if collect_usage { ws.drain_edge_usage(&mut out.usage); }   // append, per net
```

`drain_edge_usage` (`path_solver.rs:466`) pushes one 16-byte `(pipe, flow)`
pair per touched pipe **per net**, with no dedup. Every chunk is then
`.collect()`ed -- all held live -- before `merge_chunk_usage` folds them into
the dense array.

Live memory is therefore `16 bytes x sum over all nets of |edge_touched|`:
**O(total work)**, not O(chip) and not O(concurrency).

### Why this is conclusive, not plausible

1. **The `collect_usage` flag predicts the phase.** `solve_distance_cache`
   passes `false`, `solve_usage_and_energy` passes `true`. Measured:

       pre_solve   MEM_CHUNKUSAGE: total_entries=0        live_mb=0.0
       post_solve  MEM_CHUNKUSAGE: total_entries=2608439  live_mb=39.8

   Only the `true` phase grows, and it is the phase that OOMs.

2. **The magnitude projects correctly.** stereovision3 has 169 cells and 298
   nets and still stages 2.6 M entries -- ~8750 entries per net. FPGA01 runs
   105117 solves: `105117 x 8753 x 16 B ~= 15 GB` from this term alone, before
   accounting for FPGA01's wider settle sets. Baseline is 2.5 GB. That is the
   46-50 GB.

3. **It is a regression of a fix, not a gap in one.** Commit `823d2cd` is
   titled *"bound solve memory by concurrency, not by total work"*. The
   `ChunkUsage` shape introduced by `05bae52` (to make the merge order
   deterministic) does exactly the opposite of that title: the dense
   per-split arrays it replaced cost `20.5 MB x ~62 splits ~= 1.3 GB`
   (**bounded**), while the sparse replacement is unbounded in net count.

This resolves the apparent paradox in the original question. Every memory
reduction listed above genuinely works. The total still explodes because the
most recent one traded a bounded cost for an unbounded one while optimising
for a different axis (determinism).

## Fix direction (not yet implemented)

The constraint that produced `ChunkUsage` is real and must be preserved:
merging in **chunk order** is what makes the f64 association order
independent of rayon scheduling.

Both properties are obtainable. Process chunks in **waves** of
`~threads` instead of `.collect()`ing all of them: run a wave in parallel,
merge it into the dense accumulator in chunk order, drop it, proceed. The
global merge order is unchanged (so determinism is preserved exactly), and
live memory becomes `wave x batch_size x entries_per_net x 16 B`
~= 1.8 GB on FPGA01 -- the same order as the bounded dense-per-split shape
that `ChunkUsage` replaced.

## Instrumentation added (diagnostic only, no behaviour change)

- 250 ms RSS/HWM sampler thread in `examples/ot_trace_design.rs`
- `PHASE_MARK` lines around `pre_solve` / `dcd_sweep` / `post_refresh`
- one-shot `MEM_STATIC` static breakdown
- `MEM_CHUNKUSAGE` line in `merge_chunk_usage`

Run FPGA01 under a hard cage so it cannot take the box down:
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.
Being OOM-killed at 12 G is a result: the sampler's last lines name the phase
and the growth rate.

## Superseded section



Resolved -- see Finding 3. Also ruled out by construction along the way:
JacobiBB dense per-net pyramids (`region_min::build_all`) never run, because
the default `SweepMode` is `JacobiFullscan`.

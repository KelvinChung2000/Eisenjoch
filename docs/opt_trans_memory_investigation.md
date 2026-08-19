# Why the opt_trans memory work isn't showing up on FPGA01

## Answer in one line

The memory reductions all work. Two of them were **never merged into
`npnr-faithful-port`** -- they live on `flow/spreading-field`, which this
branch is 18 commits behind. FPGA01 was being measured on a branch that does
not contain its own fixes.

## The complaint

A large amount of memory-reduction work has landed in `opt_trans`, yet FPGA01
(`xc7_large`, 76660 cells, 69353 nodes, 2562403 pipes) still drives RSS to
~46-50 GB and had to be abandoned as a measurement vehicle.

## The reductions that ARE on this branch all work

Measured, not assumed, from the run's own probes (outer=0):

| Reduction | Evidence it works |
| --- | --- |
| Sparse `DistCache` (`FxHashMap` rows) | 600 MB at default radius (60 MB at radius 3) vs ~29 GB for the old dense `n_nets x n_nodes` `Vec<f32>` |
| Cache-radius filter (`NPNR_OT_CACHE_RADIUS`) | bounds the cache despite `settle_avg=7925`, `settle_max=69353` |
| Binary heap replacing bucket-Dial | `buckets` is dead (`Vec::new()`); the documented `buckets.resize()` multi-GB path is gone |
| `WorkspacePool` | pool size 8, bounded by concurrency, not by 105117 solves |
| `SpanCostTable` / `TwoLevelPip` | cost cache keyed by span, not pipe identity |

`NPNR_OT_CACHE_RADIUS` is worth calling out: the cache scales with radius^2,
so 12 -> 600 MB and 3 -> 60 MB. It is a first-order memory knob, not a tuning
detail. The frequently-quoted "60 MB DistCache" figure was measured at
radius 3, not at the default.

## Static baseline is fully accounted

New one-shot `MEM_STATIC` probe, FPGA01:

    pipes=352MB nodes=1MB adjacency=78MB pipe_cost_vecs=59MB
    validity_mask=634MB n_pipes=2562403 n_edges=5124806 rss=1677MB

1124 MB of the 1677 MB baseline is named; the rest is chipdb mmap, design,
`net_infos`, `cell_net_map`, `MuxSlotTracker`.

## Where the 44 GB was

Bracketed with `PHASE_MARK` probes under a 12 GB cage, `NPNR_OT_MAX_ITERS=1`:

    PHASE_MARK outer=0 enter=pre_solve    rss=1707MB
    PHASE_MARK outer=0 enter=dcd_sweep    rss=2445MB
    PHASE_MARK outer=0 leave=dcd_sweep    rss=2450MB   <- sweep is FLAT
    PHASE_MARK outer=0 enter=post_refresh rss=2450MB
    RSS_SAMPLE  t=195.1s                  rss=11823MB  <- OOM-killed at the cap

All growth is inside `solve_usage_and_energy`, ~600 MB/s, monotone.

Mechanism: `ChunkUsage.usage: Vec<(u32, f64)>` accumulated one 16-byte
`(pipe, flow)` pair per touched pipe **per net**, with no dedup
(`drain_edge_usage`), and every chunk was `.collect()`ed -- all live -- before
the merge. Live memory was `16 B x sum over all nets of |edge_touched|`:
**O(total work)**, not O(chip), not O(concurrency).

Two independent confirmations:

1. **The `collect_usage` flag predicts the phase.** `solve_distance_cache`
   passes `false`, `solve_usage_and_energy` passes `true`:

       pre_solve   MEM_CHUNKUSAGE: total_entries=0        live_mb=0.0
       post_solve  MEM_CHUNKUSAGE: total_entries=2608439  live_mb=39.8

2. **Magnitude projects.** stereovision3 -- 169 cells, 298 nets -- still
   stages 2.6 M entries, ~8750 per net. FPGA01 runs 105117 solves:
   `105117 x 8753 x 16 B ~= 15 GB` from this term alone, before FPGA01's wider
   settle sets. Baseline 2.5 GB. That is the 46-50 GB.

## This was already diagnosed and fixed -- elsewhere

`823d2cd fix(opt_trans): bound solve memory by concurrency, not by total work`
(2026-08-11) documents the identical finding and fixes it. It is **not an
ancestor of `npnr-faithful-port`**; `accumulate_edge_usage` did not exist in
this tree at all.

    $ git merge-base --is-ancestor 823d2cd HEAD   -> false
    $ git branch -a --contains 823d2cd            -> flow/spreading-field

`npnr-faithful-port` is 36 ahead / 18 behind `flow/spreading-field`. Among
those 18:

| Commit | What this branch is missing |
| --- | --- |
| `823d2cd` | bounds solve memory (the 25 GB+ DCD blowup above) |
| `fc82eb8` | `fix(legalize): the shortlist bounded length, not the allocation` -- the 33.5 GB legalization OOM |
| `7d0ab69`, `bd774b6` | `NPNR_OT_BPR_CAP`, bounding the BPR multiplier on both channels |
| `e3efbf3` | `make the Jacobi sweep deterministic` |

The BPR cap matters beyond memory: the previous session concluded from the
hardening gate that "the binding oscillation driver is the BPR channel's
unbounded scale". The cap for exactly that already exists, on that branch.

`e3efbf3` explains a loose end too. `823d2cd` claimed sv3 was byte-identical
after the fix; on this branch it is not, because this branch has no
determinism fix. Measured here, three runs each:

    pre-fix : 6852 / 6861 / 6815
    post-fix: 6820 / 6830 / 6861

Post-fix values sit entirely inside the pre-fix spread, same ~0.6 % magnitude.
The fix is equivalence-preserving to within pre-existing run-to-run noise; it
does not introduce the nondeterminism.

## Verified after applying `823d2cd`

Same design, same 12 GB cage, same `NPNR_OT_MAX_ITERS=1`:

| Probe | Before | After |
| --- | --- | --- |
| `enter=post_refresh` | 2450 MB | 2441 MB |
| post-solve phase cost | > 9.4 GB, still climbing | 2477 -> 3187 MB (**710 MB**) |
| peak solve staging | `16 B x sum of edge_touched counts`, unbounded | `MEM_POOL workspaces=8 dense_usage_mb=313` |
| outcome | OOM-killed at 11823 MB | DCD completed, `Pre-legalization: HPWL=14542255` |

3.2 GB matches the ~3.0 GB `823d2cd` recorded when it was written.

The run then jumps to 12270 MB **in legalization** and hits the cap -- exactly
the 33.5 GB bug `823d2cd`'s own message predicted and `fc82eb8` fixes. That is
the second stranded commit, confirming the divergence from the other side.

## Applied here

- `823d2cd` cherry-picked (clean). Usage now folds into the pooled
  workspace's dense `usage_accum`, drained from the pool after the parallel
  section. `MEM_POOL` probe confirms the new peak is bounded:
  `workspaces=8 dense_usage_mb=313`.
- `fc82eb8` **ported by hand**, not cherry-picked (`c314843`). The commit
  itself does not apply: it conflicts against 85 lines of Phase B shared-mux /
  arch-validity logic that exists only on this branch, plus two feature files
  (`spreading.rs`, `probe_electro_vs_ot_sv3.rs`) that do not. Merging two
  divergent legalizer implementations wholesale risks silently illegal
  placements, so only the semantics were carried across. See below.

## The legalization OOM: two defects, not one

This branch was worse than `flow/spreading-field` had been, because it had no
shortlist cap at all. Both of these had to be fixed; either one alone still
dies.

1. **Unbounded length.** Phase A built, for every cell, a distance-sorted list
   of *every* BEL of that cell's type and held them all live at once. `BelId`
   is 8 bytes, so one DFF-typed cell costs 8.22 MiB (1,077,536 BELs) and one
   LUT6-typed cell 4.11 MiB (538,768). Across FPGA01's **76660** cells that is
   O(cells x bels) -- between 308 GiB (all LUT6) and 615 GiB (all DFF) even
   with a perfectly sized allocation.

   > Correction: commit `c314843`'s message says "~105k cells" and "~900GB".
   > 105117 is the **net** count in these logs, not the cell count -- a
   > conflation inherited from `fc82eb8`'s own message. The authoritative
   > figure is `n_movable=76660` / "DCD placer: 76660 cells". The numbers here
   > supersede the commit message.
2. **Unbounded allocation.** `candidates.into_iter().map(..).collect()` hits
   Rust's in-place collect specialization, which reuses the *source*
   allocation instead of making a new one. The source was `Vec<(BelId, f64)>`
   sized for every BEL of the type -- 17 MB for DFF -- so each list held a
   17 MB buffer whatever its length. `truncate` sets a vec's length and never
   its capacity, which is why capping alone would not have been enough.

The fix caps to the 1024 nearest via `select_nth_unstable_by` and collects
through a *borrowing* iterator so the result allocates exactly `len`.

**The tie-break is load-bearing.** Capping needs a total order. The old full
`sort_by` was stable, so ties kept enumeration order for free;
`select_nth_unstable_by` does not. Equidistant BELs are the common case -- a
tile holds many slots at identical x/y -- and which one Phase B binds decides
the placement. `candidate_list` therefore carries each candidate's enumeration
index and breaks distance ties on it, so the shortlist and the full list agree
on their common prefix. That agreement is what makes widening safe, and it is
tested rather than asserted: removing the tie-break fails
`shortlist_is_a_prefix_of_the_full_list` at dim=10 per_tile=4 k=17.

Because a shortlist bounds *work* and not *legality*, Phase B widens: if all
1024 candidates were rejected it rebuilds the cell's full list and retries the
remainder (the prefix being exactly what it already tried).

### Measured, FPGA01, same 12 GB cage

| Probe | Before | After |
| --- | --- | --- |
| legalization | 2400 -> 12270 MB in 3.5 s, OOM-killed | completed, t=353s -> t=1520s |
| whole-run peak RSS | hit the 12 GB cap | **4217 MB** (run exited cleanly, no OOM/panic) |
| legalization wall time | never finished | **1167 s (19.4 min)** |
| pre-legalization HPWL | 14542255 | 14564194 (0.15 % apart, inside the ~0.6 % spread) |
| post-legalization HPWL | never reached | 14234563 |
| `arch_validity_rejects` | -- | 0 |
| `widened_rescans` | -- | **27467** |

### `widened_rescans=27467` is a real signal, not noise

`fc82eb8` measured `widened_rescans=0` on FPGA01. Here about a quarter of all
cells exhaust their 1024-BEL shortlist, driven by **21,841,944 shared-mux
rejections** (arch-validity rejected nothing). That rejection pressure is
specific to this branch's Phase B, so the sister branch's figure does not
carry over.

It is a cost, not a bug -- widening is exhaustive, so the placement matches
what an uncapped run would produce -- but each rescan rebuilds and sorts a
list of up to 1,077,536 entries, and **the bill is measurable**: legalization
took 1167 s here against the 6 m 16 s `fc82eb8` measured with
`widened_rescans=0`. Roughly 13 of those 19.4 minutes are rescan.

The knob is not free in the other direction either: shortlist memory is
`n_cells * k * 8 B`, which at 76660 cells and k=1024 is **599 MiB** of the
4217 MB peak. Raising k to 4096 would cost ~2.3 GiB. So k trades memory
against 27k full rescans and neither end of that trade is good.

**This is the argument for the design-database path.** A grid-indexed spatial
BEL index already exists in the tree -- `placer/fast_bels.rs`, a faithful port
of upstream's `FastBels`, `[type][x][y] -> Vec<BelId>` -- and is used by the
SA placer (`place_common.rs`) and the refiner, but *not* by the sorted
legalizer, which is why the legalizer materializes candidate lists at all.
Expanding rings outward from the target and stopping at the first legal BEL
would make the per-cell cost O(k) in both time and memory, delete the
shortlist heuristic and its widen path, and remove the k trade-off entirely.
That is wiring an existing structure, not new infrastructure -- but it changes
which BEL ties resolve to, so it needs its own equivalence check.

## Still open, separate from the above

- **Validity mask stores bucket data per cell.** `CellValidityMask`
  (`placer/common.rs`) is a bitset indexed `cell_idx * W*H + gy*W + gx` =
  **634 MB** on FPGA01. `build` derives each cell's bits only from
  `ctx.resolve_bucket(cell_type)`, so contents are a function of the bucket,
  never of the cell -- and the log confirms it
  (`per-cell valid positions: min=67346 avg=67346.0 max=67346`, two buckets).
  That is one 8.7 KB mask duplicated 76660 times.

  **Fixed.** Now stores one mask per distinct bucket plus a `u32` index per
  cell; `is_valid` stays O(1) with one extra indirection.
  **Measured on FPGA01: `validity_mask` 634 MB -> 0.3 MB, and the whole static
  baseline 1677 MB -> 1037 MB.** The build line reports `2 bucket masks` for
  76660 cells.
  sv3 reports `2 bucket masks` with per-cell counts unchanged at 67346 and
  `unmapped cells=0`, and HPWL stays inside the pre-existing run-to-run spread
  (6834 / 6827 against a pre-fix 6815-6861). Guarded by
  `storage_does_not_scale_with_cell_count`, which fails on the old layout.
- **The sparse-usage regression is latent elsewhere.** The Bresenham variants
  still run the old `fold(|| vec![0.0; n_pipes]) ... reduce` shape
  (`coord_descent.rs:877/926/959/993`, `path_solver.rs:765/791`), and
  `place_dcd_sweep_colored_gs` builds a fresh
  `PathSolverWorkspace::new(n_nodes, n_pipes)` per rayon split
  (`coord_descent.rs:3058`) instead of taking one from `ws_pool`. Neither is on
  the default path, so neither was measured here.

## Instrumentation added (diagnostic only, no behaviour change)

- 250 ms RSS/HWM sampler thread in `examples/ot_trace_design.rs`
- `PHASE_MARK` lines around `pre_solve` / `dcd_sweep` / `post_refresh`
- one-shot `MEM_STATIC` static breakdown
- `MEM_POOL` line reporting pooled dense usage arrays

Always run FPGA01 under a hard cage so it cannot take the box down:

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect

Being OOM-killed at the cap is a result: the sampler's last lines name the
phase and the growth rate.

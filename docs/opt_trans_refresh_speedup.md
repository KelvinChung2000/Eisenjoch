# Closing the runtime gap to HeAP on FPGA01

Goal: get opt_trans' wall clock near HeAP's while holding HPWL. HeAP places
FPGA01 in 52.8 s; opt_trans took 2492 s at stock settings. `refresh` --
the per-net Dijkstra pass that fills `dist_cache` -- was 92-93 % of that.

## Result

One change shipped: **`NPNR_OT_THREADS`**. Everything else measured as a dead
end, and the negatives are the more useful half of this document.

| FPGA01, 20 iters, stock | 8 threads | 32 threads |
|---|---|---|
| wall | 2492 s | **1226 s** (2.03x) |
| refresh | 2296 s | 974.0 s |
| peak RSS | 2948 MB | 4613 MB |
| post-leg HPWL | 10 670 244 | 10 722 928 (+0.49 %) |

+0.49 % is inside the documented 0.6-3.3 % fixed-seed spread, so quality is
unchanged. Note the box is a **16-core** 7950X with 2 threads/core: 8 -> 32
threads is only 2x the physical cores plus SMT, which is exactly the 2.03x
observed. There is no further headroom on this axis.

### The thread win is config-dependent: `steiner=1` is not settled

The +0.49 % above is `steiner=0`. On `steiner=1` -- the likelier shipping
config -- the same thread change does **not** reproduce that clean result:

| FPGA01, 20 iters, `steiner=1` | wall | post-leg HPWL |
|---|---|---|
| 8 threads, HEAD `9bf49cd`~ | 2819 s | 7 291 219 |
| 8 threads, HEAD `0429c3b` (control) | 2898 s | 7 385 785 |
| 32 threads, HEAD `0429c3b` | 1508 s | 7 757 880 |

The two 8-thread runs bracket the same-config spread at **+1.30 %**, and the
control reproducing that band at current HEAD rules out code drift from the
instrumentation. The 32-thread run sits **+5.04 %** above the nearest
8-thread sample -- outside that spread, and contradicting `steiner=0`.

Repeating the 32-thread run settles it. Two samples per side:

| `steiner=1`, 20 iters | 8 threads | 32 threads |
|---|---|---|
| run 1 | 7 291 219 | 7 757 880 |
| run 2 | 7 385 785 | 7 869 541 |
| spread within config | +1.30 % | +1.44 % |

The ranges **do not overlap**: worst 8-thread to best 32-thread is +5.04 %,
and the means differ by +6.48 %. At `steiner=1`, thread count costs ~6.5 %
HPWL. It is not a bad draw.

Note this cannot be float summation order -- non-associativity is a ~1e-15
effect, not 6.5 %.

### The placer is nondeterministic at fixed seed, and that undercuts the claim

Probing at 3 iterations, `steiner=1`, comparing the per-iteration `DCD`
signature -- A1/A2 identical config, B pinned to the 8-thread chunk via
`NPNR_OT_BATCH`, C at 8 threads:

| arm | threads | chunk | `line` @ outer=2 |
|---|---|---|---|
| A1 | 32 | 410 | 15 880 736 |
| A2 | 32 | 410 | 15 910 166 |
| B | 32 | 1642 | 15 886 360 |
| C | 8 | 1642 | 15 894 100 |

**A1 and A2 are the same configuration and do not agree** -- they already
differ by 0.21 % at outer=0. All four arms span 0.19 % at outer=2, so B and C
both land inside the A1-A2 same-config band: at 3 iterations neither chunk
size nor thread count is separable from run-to-run noise.

Root cause is in `place_dcd_sweep`. It probes against the **live,
concurrently-updated** mux tracker and commits each move by CAS *inside* an
unchunked `into_par_iter()`, so which cell wins a contested tile slot is
decided by thread interleaving. That `into_par_iter()` is not inside
`solve_pool.install`, so it runs on the global rayon pool (all 32 logical
CPUs) in **every** run here -- which is why it explains the run-to-run
spread but is not itself the variable that differs between the 8- and
32-thread configs.

Consequences, in order of importance:

1. **The thread penalty, re-measured at n=4/n=5, is real.**

   | `steiner=1`, 20 iters | n | min | max | mean | spread |
   |---|---|---|---|---|---|
   | 8 threads | 4 | 7 291 219 | 7 472 217 | 7 387 340 | 2.48 % |
   | 32 threads | 5 | 7 757 880 | 8 549 040 | 8 084 121 | 10.20 % |

   Every 32-thread run exceeds every 8-thread run: exact rank-sum
   p = 1/C(9,4) = 0.008. The cost is **+9.4 % on the means**, and the best
   32-thread run is still 3.8 % worse than the worst 8-thread run. Note the
   two-sample estimates this replaces were badly wrong in both directions --
   the 32-thread spread is 10.2 %, not the 1.44 % two draws suggested.

   32 threads is worse on mean *and* variance, which looks like contention.
   But the sweep's `into_par_iter()` runs on the global rayon pool -- 32
   threads in **both** configs -- so sweep contention cannot be the
   difference. Mechanism remains open: solve chunk size (410 vs 1642) and
   `set_solver_threads(cfg.num_threads)` are the surviving candidates. The
   3-iteration chunk probe did not clear chunking; at 3 iterations nothing
   was resolvable.
2. **A bit-identical `DCD` signature is not an available test on this
   codebase.** Identical configurations diverge, so no code change can be
   validated by signature comparison until the sweep is made deterministic.
   Every A/B needs repeats, and the 1.3 % same-config spread is the floor on
   what any experiment here can resolve.
3. Until (1) resolves, **`NPNR_OT_THREADS` is validated on `steiner=0`
   only** (+0.49 %, inside noise).

### `NPNR_OT_DET_SWEEP=1` makes it reproducible (default off)

The sweep now splits into a parallel probe against frozen occupancy and a
commit pass in ascending cell index, sharing `plan_cell` / `commit_cell` so
candidates and their rank order are unchanged. Only the tie-break for a
contested slot moves -- from whichever thread won the CAS to the lower cell
index.

Verified on FPGA01, `steiner=1`, 3 iters, 32 threads, comparing `DCD` lines
with the timing fields stripped:

| | two runs, same config |
|---|---|
| `NPNR_OT_DET_SWEEP=1` | **identical** on every result field |
| default (racy) | differ |

A matched negative control, so this is the fix and not a quiet run. Strip
`refresh=`/`dcd=`/`total=` before comparing -- they vary run to run and a
naive `diff` reports a false negative.

This restores the bit-identical signature check as a cheap test for any
future change here, which the racy path made impossible.

Priced at 20 iterations, `steiner=1`, 32 threads: **HPWL 7 937 739** in
1483 s.

| 32 threads, 20 iters | HPWL |
|---|---|
| racy, n=5 | mean 8 084 121, range 7 757 880 - 8 549 040 |
| **deterministic** | **7 937 739** (exact -- no spread) |
| racy 8 threads, n=4 | mean 7 387 340 |

So determinism beats the racy *mean* by 1.8 % and removes a 10.2 % spread,
but does not reach the racy *best*, and stays 6.2 % above every 8-thread run.

**Determinism does not fix the thread penalty** -- which independently
confirms the diagnosis above: the sweep race cannot be the mechanism, because
that sweep runs on the global 32-thread pool in both configs.

The flag's value is measurement, not quality: one deterministic run is an
exact value, so the thread penalty and the chunk-size hypothesis can now be
settled with single runs instead of fighting a 10 % band. Since results no
longer depend on timing, such runs can even be executed concurrently without
confounding each other.

## The corridor settles ~10x more nodes than can affect congestion

Instrumenting the back-pass (`load_nonzero` / `load_material`, FPGA01,
`steiner=0`, 4 threads):

| outer | settled/net | any load | >= 1e-6 of demand |
|---|---|---|---|
| 0 | 9 921 | 6 157 (62.1 %) | 2 349 (23.7 %) |
| 1 | 9 017 | 1 208 (13.4 %) | 709 (7.9 %) |
| 2 | 8 194 | 826 (10.1 %) | 668 (8.2 %) |

Once congestion develops, **~90 % of settled nodes receive exactly zero
load**. Load reaches only the ancestors of sinks in the shortest-path DAG --
a tube -- while the corridor admits a geometric Manhattan ellipse. Congestion
inflates cost-per-tile (`max_util` reaches 475x), which shrinks the tube in
cost space while leaving the ellipse unchanged: that is why the loaded share
falls 62 % -> 10 % between outer=0 and outer=2, and why the earlier
corridor-tightening experiment bought so little (a geometric bound is the
wrong shape for a cost-shaped tube).

Note this is *not* explained by the logit decay: `LOGIT_THETA` is 0.25, so
the weight falls only to `exp(-0.25)` = 0.78 per tile of slack. The `theta`
printed on the DCD line is a different quantity.

### Why this may be exactly prunable

`path_weight[N]` accumulates over upstream neighbours `P` with
`likelihood > 0`, and the back-pass sends load from `N` to `P` under that
same condition. So `P` feeds `N`'s normalisation **iff** `P` receives load
from `N`. A node with zero load therefore contributes nothing to any loaded
node's `path_weight`, and dropping it changes neither the energy nor the
booked usage. Nor can a zero-load node sit on a shortest path to a loaded
one: a tight edge has slack 0 and `LUT[0] = 1 > 0`, so it would be loaded.

**But no forward-pass criterion identifies that set.** The LUT zeroes
*per-edge* slack at `LIKELIHOOD_LUT_SIZE`, not *path* slack -- a node whose
accumulated path slack exceeds the table can still take nonzero load through
a chain of small-slack edges. So a path-slack bound prunes a set that is not
the zero-load set, and would fail the bit-identical test. The honest
statement: the zero-load set is exactly removable in principle, and every
*implementable* bound is an approximation whose error has to be measured.

Two further caveats before this is worth building. The settled set also fills
`dist_cache` within `cache_radius_tiles` of the net's own cells, which is a
separate coverage requirement that pruning must still satisfy. And knowing a
node's slack needs a lower bound on its distance-to-sink during the forward
pass -- an admissible A* heuristic -- which is weak exactly where congestion
is high. The claim above is argued, not measured; the test is a
bit-identical `DCD` signature (`energy`, `line`, `pops`) across a few
iterations at fixed thread count.

## Where the time actually goes

FPGA01's network is 69 353 nodes / 2 562 403 pipes (~37 edges per node).
Per outer iteration the post-solve runs one Dijkstra per net over 105 117
nets. The pre-solve skips every net after outer=0 (pin signatures are
unchanged since the previous post-solve), so the post-solve is the pass that
costs -- and it had no diagnostic at all before this work.

Measured `settle_avg` = 9918 nodes, `settle_max` = 69 353 -- i.e. the average
net's search covers 15 % of the chip and the worst covers all of it.

`perf` on one iteration at 32 threads:

```
39.27%  dial_logit_load        (back-pass + energy)
33.00%  dijkstra_and_forward   (forward relaxation)
12.03%  Fn/FnMut call          (inlined closures)
 5.38%  BinaryHeap::pop
 4.84%  PathSolverWorkspace::begin_net
 2.26%  evaluate_cell_at
```

72 % is inside the two solver functions: memory-bound graph traversal.

## What did not work, and why

**Shrink the Dijkstra termination cap.** Load flows only downhill in `dist`
from the sinks, so every settled node above `max_sink_dist` carries provably
zero load -- it exists only to populate `dist_cache` near pins. If that tail
were large, it could be cut with no change to the physics at all. Measured
(`settle_above_sink`): **2.3 %** at outer=0, 1.4 % at outer=1. The cap is
already tight; there is nothing to reclaim.

**A\* / goal-directed search.** Already implemented. `Corridor::contains_xy`
admits a tile when `d(src,n) + d(n,sink) <= stretch*direct + halo` -- an
admissible Manhattan ellipse. Adding an A\* heuristic would duplicate it.

**Tighten that ellipse.** `CORRIDOR_TIGHT` narrows the slack from
`0.35*span + 12` to `0.10*span + 4`. Priced at MAX_ITERS=3 / 32 threads:

| arm | refresh | settle_avg | HPWL |
|---|---|---|---|
| stock | 242.7 s | 9905 | 13 238 657 |
| TIGHT + halo 6 | 218.0 s | 8068 | +0.41 % |
| TIGHT + halo 3 | 214.4 s | 7528 | +0.98 % |

A 24 % cut in settles buys only 11.7 % of refresh, because the expensive nets
are clamped by the chip bbox rather than by the stretch term.

**A high-fanout cutoff.** The work is concentrated -- 17 % of nets do 78 % of
refresh -- but not where expected. Keyed on fanout:

```
fan1:      43 710 nets -> 15.1 % work
fan2-4:    48 279 nets -> 50.3 %
fan5-16:   12 974 nets -> 33.9 %
fan17-64:     154 nets ->  0.8 %
fan65-256:      0 nets
fan>256:        0 nets
```

**FPGA01 has no high-fanout nets at all.** Half the work is in 2-4 pin nets.
The expensive nets are *long*, not *wide*, so a fanout cutoff targets nothing.
(A hard cutoff is also unsafe here: `evaluate_cell_at` returns INFINITY on a
`dist_cache` miss, so a never-solved net freezes every cell touching it.)

**Stratified refresh** (reverted in 0429c3b). Defer nets whose pin span
exceeds a threshold to every K-th iteration, carrying their usage and energy
forward in between so every net still contributes to the EMA blend and the
pipe-history dual each iteration. Span is the right key -- per-net cost is the
ellipse area, ~quadratic in span. Refresh halved as designed:

| FPGA01, 32 threads | wall | refresh | HPWL |
|---|---|---|---|
| base, 20 iters | 1226 s | 974.0 s | 10 722 928 |
| span70/period3, 20 iters | 725 s | 470.5 s (-51.7 %) | 11 858 130 (**+10.59 %**) |

The hypothesis for leaving this in was that a cheaper-but-weaker iteration
would converge to the same place given the full run. It was falsified: the
cost **compounds** rather than washing out -- +4.75 % at 6 iters, +10.59 % at
20. DCD is a fixed-point iteration and the deferred nets are by construction
the ones dominating the objective, so stale distance fields move the fixed
point instead of merely slowing the approach to it. No span/period setting
escapes the mechanism (period 2 was already +3.26 % at 6 iters).

An implementation bug was found and fixed during this measurement -- refresh
iterations solved every net and then re-solved the slow stratum, double
counting its usage. Fixing it recovered the speed (16.7 % -> 49.7 %) but
barely moved HPWL (13 121 420 -> 13 110 520), which is how we know the
quality loss is staleness itself and not the bug.

**Bucket / radix priority queue.** Exact -- every label comparison is strict
`<`, so tie order provably cannot matter. Not worth building: `BinaryHeap::pop`
is 5.4 % of the profile, so the whole lever caps out near 2.7 %.

## The structural limit

HeAP costs **O(pins)** per net; opt_trans costs **O(corridor area)** per net.
A chip-spanning 2-pin net is 2 matrix entries for HeAP and up to 69 353 node
settles here. That is the 47x, and it is not a constant factor: no scheduling
change or micro-optimisation reaches parity. Closing it further requires
changing what is computed per net -- a different algorithm, not a faster one.

After threads, FPGA01 stands at 1226 s vs HeAP's 52.8 s (23.2x, from 47.2x)
with HPWL held at 2.365x. At `steiner=1` the HPWL ratio is 1.608x.

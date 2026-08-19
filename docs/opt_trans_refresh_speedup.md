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

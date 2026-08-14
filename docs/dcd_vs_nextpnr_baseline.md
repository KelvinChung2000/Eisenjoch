# DCD (opt_trans) measured against real nextpnr

First like-for-like placement-quality numbers for `opt_trans` against upstream
nextpnr, on a fabric both tools read, scored by a metric proven identical to
nextpnr's own.

## What makes this fair

`tests/npnr_baseline_compare.rs` established that our `get_net_metric` matches
upstream's exactly. That was on one 169-net case; it now holds on three:

| fabric | nets | our score of nextpnr's placement | nextpnr's golden dump |
|---|---|---|---|
| 7x7, W=16 | 171 | 159 | 159 |
| 12x12, W=16 | 171 | 203 | 203 |
| 20x20, W=64 | 689 | 1857 | 1857 |

So the referee is trustworthy, and it does not depend on the ported placer
bodies — the comparison is against the real C++ binary, today.

Driver: `crates/nextpnr/examples/npnr_baseline_placers.rs`. Both sides read the
same chipdb and the same netlist; nextpnr's placement is reconstructed from its
`NEXTPNR_BEL` attributes, ours is produced from scratch.

## Result (historical — superseded)

> These were the first numbers taken, before the placer enforced arch legality
> and before it honoured the pinned clock buffer. They are kept because the rest
> of the document reasons from them. **For current figures see "FIXED: the
> placer is now legality-aware" below**; every table from here to that section
> describes an illegal placement.

Total wirelength, `opt_trans` against nextpnr's HeAP. opt_trans figures are the
mean of 5 runs (see the nondeterminism note below):

| fabric | LUT util | nextpnr | opt_trans (default, mean of 5) | ratio |
|---|---|---|---|---|
| 7x7, W=16 | 41% | 159 | 463 (445-497) | 2.91x |
| 12x12, W=16 | 11% | 203 | 688 (672-704) | 3.39x |
| 20x20, W=64 | 14% | 1857 | 3958 (3924-4005) | 2.13x |

### opt_trans is nondeterministic at fixed seed

Repeated runs of the identical binary, config and seed do not agree. Run-to-run
spread over 5 runs is 2.0% (20x20), 4.7% (12x12), 11.2% (7x7) — larger on
smaller fabrics. This is **not** thread scheduling: forcing `num_threads = 1`
does not fix it (single-threaded 20x20 runs gave 4083 / 4003 / 3854, a wider
spread than the 8-thread runs). Something else in the placer is unseeded.

That is a reproducibility problem for benchmarking in its own right, worth
chasing independently. It does not threaten the conclusion below: the MST effect
is ~40%, an order of magnitude above this noise. But single-run comparisons of
opt_trans against anything should be treated as unreliable, and every number
here is a 5-run mean.

## FIXED: the placer is now legality-aware, and it routes

Two changes, both ports of what nextpnr does:

1. **The predicate.** `Context::is_bel_location_valid` now consults an installed
   arch rule (`set_validity_check`) instead of returning `true`, and
   `SortedLegalizer` calls it: bind the candidate, ask the arch, unbind and try
   the next bel if rejected. It has to be bind-then-ask because `slice_valid`
   reads live tile occupancy. Cluster children are re-checked after
   `place_cluster_children`, since binding the FF beside a LUT is exactly what
   turns a legal slice illegal. Reported as `arch_validity_rejects`.
2. **The structural pass.** `constrain_cell_pairs` (port of
   `HimbaechelHelpers::constrain_cell_pairs`) clusters each LUT->FF pair at
   `delta_z = 1`, so shared slices are legal by construction. Opt-in in the
   driver via `OT_PAIR_LUTFF=1`; it constrains 128 pairs, matching nextpnr's own
   "Constrained 128 LUTFF pairs" exactly.

Results on the 20x20 fabric, 5 opt_trans seeds each, every run legality-checked
and routed by nextpnr's own router. Each placement is routed once at router seed
1, so router noise sits inside these Fmax figures:

| | reference | ours | ratio |
|---|---|---|---|
| **filter only** (vs unpacked ref) | | | |
| HPWL | 1825 | 2485 (2419-2543) | 1.36x |
| post-route Fmax | 44.07 MHz | 38.66 MHz (36.09-40.47) | 1.14x |
| **structural pairing** (vs packed ref) | | | |
| HPWL | 1857 | 2310 (2274-2334) | 1.24x |
| post-route Fmax | 46.67 MHz | 40.34 MHz (37.39-41.94) | 1.16x |

**10/10 placements legal, 10/10 routed.** Three things follow.

**Legality is nearly free.** The illegal placement scored 2452; the same config
with the rule enforced scores 2485, +1.3%. 2946 candidate bels were rejected
during legalisation and the cost was almost nothing -- the placer had ample
slack, it simply was never asked.

**Structural beats filtering on wirelength**, as nextpnr's default flow implies:
2485 -> 2310, -7%, absolute Fmax 38.66 -> 40.34, and `arch_validity_rejects`
drops to 0 because the constraint means the filter never has to fire. Relative
to its own reference it is a wash, though — 86.4% of 46.67 against the filter's
87.7% of 44.07 — and the two configs are measured against *different*
references, so they are not directly comparable. With a ~4 MHz Fmax spread
across placements, treat "structural is better" as established on HPWL and
unproven on relative Fmax.

**The HPWL gap overstates the real gap.** We are 1.24-1.36x on wirelength but
only 1.14-1.16x on post-route Fmax, and 1.13x on routed wire/pip records (7633
vs 6745). Scoring placement by HPWL -- nextpnr's own objective -- flatters
nextpnr, exactly as suspected. Everything below that is expressed as an HPWL
ratio should be read with that discount.

## How we got here: the placement used to be illegal

Handing our placement to nextpnr's own router was meant to retire two caveats at
once. It retired them by failing:

```
ERROR: Placing design failed.
248x  Warning: post-placement validity check failed for Bel 'X3Y5/L2_LUT' ...
```

248 bels = **124 slices**. nextpnr will not route our placement at all.

The rule is `slice_valid` (`example.cc:134`): a slice holding both a LUT and an
FF is legal only if the FF's `D` net is that LUT's `F` net, or the LUT's `I3` is
unused. The `I3` escape hatch is dead here — `lut_i3_used` tests
`getPort(I[K-1]) != nullptr`, which is the *net*, and `replace_constants` turns
a constant tie into the real GND net. Every LUT4 on this benchmark has
`I[3] = '0'`, so the rule reduces to: **a LUT and an FF may share a slice only
if the FF is driven by that LUT.**

`tools/npnr_compare/check_slice_legality.py` reproduces the tool's verdict
exactly (124 illegal against 248 warnings):

| placement | LUT+FF slices | legal | illegal |
|---|---|---|---|
| opt_trans, `mst=4 steiner=1` | 127 | 3 | **124** |
| opt_trans, default | 122 | 0 | **122** |
| nextpnr, unpacked | 80 | 80 | 0 |
| nextpnr, packed | 128 | 128 | 0 |

Root cause: `Context::is_bel_location_valid` was a hardcoded `true`, and
opt_trans never called it regardless — the only two call sites in the tree were
in `place_common.rs`. The placer had no arch-legality awareness of any kind.
Fixed as described above.

This inverts the reading of the LUT->FF class. We were not merely "5.1x worse at
co-locating pairs": we were achieving our co-location by stuffing *unrelated*
FFs into LUT slices, which is the one thing the arch forbids. HeAP's 80 shared
slices are all real driver-sink pairs, found unaided, because its legaliser only
accepts moves that pass `isBelLocationValid`.

**Every HPWL number measured before the fix is therefore a lower bound in our
favour** — including the tables further down, which predate it. The measured
cost of legality turned out to be small (+1.3%), so those figures are close, but
the legal numbers in the section above are the ones to quote.

### The pinned clock buffer was also being ignored

`clk_buf` carries `(* BEL = "X1Y0/IO0" *)` — the example fabric's clock ladder is
fed by a `GCLK_OUT` pip that exists only in that tile. nextpnr honours it; our
driver did not, so opt_trans was free to move the clock buffer while the
reference was not. Fixed in the driver (`apply_bel_constraints`, binding at
`Locked`, which both `lock_boundary_cells` and the warmup re-shuffle respect).

Honouring it costs us a lot, which is itself a finding: we handle a pinned
high-fanout clock source (fanout 128) far worse than HeAP does. Corrected
20x20 figures against the legal unpacked reference of 1825, means of 5 seeds:

| config | mean | range | ratio |
|---|---|---|---|
| default | 5231 | 5136-5329 | 2.87x |
| `mst=4 steiner=1` | 2452 | 2436-2475 | 1.34x |

Superseding the 3958 / 2390 figures in the tables below, which were measured
with the clock buffer free.

These five runs vary the *seed*, so their spread — 3.7% default, 1.6% tuned — is
across-seed, not the fixed-seed nondeterminism noted earlier. Both are still
well below the 2-11% fixed-seed spread measured with the buffer free, which is a
hint worth following up: whatever is unseeded in the placer appears to interact
with boundary/IO cell placement.

## Where the gap is

Splitting by whether a net touches an IO cell (20x20):

| | nextpnr | opt_trans | excess |
|---|---|---|---|
| io-touching (260 nets) | 1470 | 1870 | 400 |
| core (427 nets) | 387 | 2043 | **1656** |

81-88% of the excess is in core logic-to-logic nets, consistently across all
three fabrics. Core wirelength alone is 5.3x nextpnr's on the 20x20 case.
("core" here still includes the LUT->FF pairs; they are split out below.)

This kills the obvious hypothesis. `lock_boundary_cells` only locks IO cells
that are *already placed*, and nothing pre-places them here, so IO cells keep
whatever bel the initial discrete binding gave them and DCD never moves them
(`TypeAwarePlacement: 2 cell types` — only LUT4 and DFF). That looked like the
answer and is not: IO placement is nearly free, the loss is in the logic core.

### The loss is short-range, not global

Mean wirelength per net by class, against the unpacked nextpnr baseline, with
`mst=4 steiner=1`:

| class | nets | nextpnr | opt_trans | ratio |
|---|---|---|---|---|
| LUT->FF pairs | 128 | 0.25 | 1.27 | 5.1x |
| io-touching | 260 | 5.39 | 5.70 | **1.06x** |
| other core | 301 | 1.30 | 2.50 | 1.93x |

We are within 6% of HeAP on the long IO nets — the global structure of the
placement is fine. The loss is concentrated in the short nets, and it is worst
on the shortest class of all. This is local tightening, not global optimisation:
DCD gets cells into roughly the right region and then fails to squeeze the last
one or two tiles.

Read this together with the legality section: the LUT->FF class is exactly where
we break `slice_valid`, so "we fail to squeeze the last tile" understates it —
we squeeze into slices we are not entitled to. Re-derive this split once the
placer is legality-aware.

That is consistent with legalisation being the culprit rather than DCD's solve:
`Pre-legalization: HPWL=3711` → `Post-legalization: HPWL=4086` on a default
20x20 run, i.e. our legaliser *adds* about 10%. HeAP's strict legalisation plus
its detail-placement refinement is exactly the machinery we are missing.

## Is it packing? Measured: no

The obvious objection is that nextpnr *packs* and we do not. Its example uarch
runs `constrain_cell_pairs(LUT4.F -> DFF.D, delta_z=1, allow_fanout=false)`,
reporting "Constrained 128 LUTFF pairs" on the 20x20 run. That pins each DFF to
`constr_x=0, constr_y=0, constr_z=+1` — the same tile as its LUT. Checked
against the placement: all 128 such pairs are in the same tile, total manhattan
distance **0**. We apply no such constraint.

That looks like it should explain a lot. It does not. Disabling the constraint
in nextpnr (`patches/0002-optional-lutff-pack.patch`, `NPNR_NO_LUTFF_PACK=1`)
and re-placing the same netlist:

| nextpnr, 20x20 W=64 | total wirelength | the 128 LUT->FF nets |
|---|---|---|
| with LUTFF packing | 1857 | 0 |
| without | **1825** | **32** |

Two things follow:

1. **Packing is worth about −1.7% of total wirelength — it slightly *hurts*.**
   The constraint is a restriction, so unconstrained HeAP does marginally better
   on pure wirelength. It cannot explain a 2x gap in either direction.
2. **HeAP co-locates LUT->FF pairs unaided.** With no constraint at all it still
   places those 128 nets at total distance 32 (0.25 tiles average). opt_trans
   puts them at 448 (3.5 tiles) by default, 163 (1.3 tiles) tuned. So this class
   is measuring placement quality, exactly like every other class.

Comparing against the *unpacked* nextpnr baseline, so neither side packs, the
gap is essentially unchanged: 2.19x at default, 1.32x with MST/Steiner. Earlier
drafts of this document reported a "packing-attributable" share of 19-31%; that
framing was wrong and has been removed — subtracting the LUT->FF class is not a
packing correction, because the reference does not depend on packing.

## The lever: the MST/Steiner terms are off by default

`mst_edge_weight` and `steiner_weight` both default to `0.0`. Their own doc
comments describe precisely this failure mode — the star model is
"driver-anchored", lacks "sink-sink coupling", and suffers a
"centroid-stacking pathology". The fix is implemented and switched off.

Turning it on, on the 20x20 fabric (single runs, so read these as +/-2%; the
per-config means are in the table after):

| config | wirelength | ratio |
|---|---|---|
| default | 4092 | 2.20x |
| `steiner=1.0` | 3496 | 1.88x |
| `mst=1.0` | 2653 | 1.43x |
| `mst=2.0` | 2418 | 1.30x |
| `mst=4.0` | 2410 | 1.30x |
| `mst=8.0` | 2464 | 1.33x |
| `mst=4.0 steiner=1.0` | 2349 | 1.27x |

Plateaus at `mst≈4`. Raising `max_outer_iters` 50→200 buys almost nothing
(2410→2347), so this is an objective-shape problem, not a convergence problem.

Holding `mst=4.0 steiner=1.0` across all three fabrics, means of 5 runs:

| fabric | default | with MST/Steiner | reduction | nextpnr |
|---|---|---|---|---|
| 7x7, W=16 | 463 (2.91x) | 275 (1.73x) | 41% | 159 |
| 12x12, W=16 | 688 (3.39x) | 373 (1.84x) | 46% | 203 |
| 20x20, W=64 | 3958 (2.13x) | 2390 (1.29x) | 40% | 1857 |

Roughly half the gap, everywhere, against a 2-11% run-to-run spread. The
residual is still almost entirely core (20x20 excess: core 469, io 28).

**The defaults are deliberately not changed here.** These fabrics are synthetic
and tiny; prior opt_trans tuning has repeatedly reversed sign between synthetic
and real fabrics. This needs a run on FPGA01/stereovision3 before any default
moves.

## The gap was mostly a missing pipeline stage

Everything above compares opt_trans's legalised placement against a nextpnr
number that was never HeAP's legalised placement. HeAP *always* ends with a
simulated-annealing refinement pass (`placer_heap.cc:402-412`; `parallelRefine`
defaults false, so it runs `placer1_refine`). Our pipeline stopped at
legalisation.

With `patches/0003` (`NPNR_NO_REFINE=1`) the two halves separate. On the 20x20
fabric at seed 1, unpacked:

| stage | wirelen | timing cost | post-route Fmax |
|---|---|---|---|
| HeAP legalised | 2379 | 10 | 40.62 MHz |
| + nextpnr's refine | 1825 | 4 | 44.07 MHz |

Refinement is worth 23% of the wirelength and 3.45 MHz on this seed, so the
"1.36x gap" was comparing our 4-stage pipeline against their 5-stage one.

### The reference is a distribution, not a number

The published references were single runs at seed 1, and seed 1 is nextpnr's
*worst* unpacked seed. Five seeds each, every run routed at router seed 1:

| config | HeAP legalised | after refine | refine gain | post-route Fmax |
|---|---|---|---|---|
| packed | 1966.6 (1934-2027) | 1818.4 (1770-1857) | 7.5% | 46.03 (43.81-46.76) |
| unpacked | 2432.2 (2369-2506) | 1831.0 (1694-1916) | 24.7% | 46.32 (44.07-49.49) |

Corrected mean-to-mean, before we changed anything: **1.14x packed, 1.20x
unpacked** on Fmax. The timing objective converges to ~4 from starts between 6
and 12 with almost no spread — refinement is a strong attractor, which is why
skipping it cost so much.

## Porting the refiner, and what it fixed on the way

`placer1_refine` is now ported (`placer/refine/`). Validating it on nextpnr's
own pre-refine placement — same input, same target, opt_trans not involved —
surfaced three defects no internal check had caught:

1. **Our legaliser bound at PLACER(3)**, above the STRONG(2) cutoff refinement
   uses. A faithful refiner would have moved zero cells and reported success.
   Upstream binds plain cells WEAK and cluster-constrained ones STRONG.
2. **The timing objective was identically zero, everywhere.**
   `port_timing_class` returned `Ignore` for every port because
   `Context::new` built a *fresh* `IdStringPool`, while ids stored inside the
   chipdb index the database's own string table. nextpnr shares one global
   space; we did not, so `LUT4` interned as 703 and matched nothing. This had
   silently disabled the timing term in `get_net_metric` too, so every
   "timing-driven" figure taken before this was wirelength-only.
3. **`BEL_STRENGTH` was discarded on load.** We bound nextpnr's placements at
   STRONG regardless, so the USER-pinned clock buffer looked movable. The
   refiner moved it, and `GCLK_OUT` exists in exactly one tile, so the clock
   stopped routing. Caught only by routing the output.

Refiner against upstream on identical input (unpacked, seed 1):

| | wirelen | timing | post-route Fmax |
|---|---|---|---|
| nextpnr's refiner | 2379 -> 1825 | 10 -> 4 | 44.07 MHz |
| ours | 2379 -> 1779 | 10.190 -> 3.830 | 46.76 MHz |

Reading 2379 and 10.190 for an input upstream reads as 2379 and 10 is the
evidence that the cost model — bounding boxes, criticality, delay scale — agrees
with placer1's.

## Result with refinement (5 seeds each, all routed)

| config | HPWL | vs ref | Fmax | vs ref | was |
|---|---|---|---|---|---|
| filter only | 2029.6 (1991-2068) | 1.112x | 42.81 (40.87-43.88) | **1.082x** | 1.20x |
| structural pairing | 1937.0 (1925-1970) | 1.043x | 43.66 (40.08-45.11) | **1.054x** | 1.14x |

**15/15 legal, 15/15 routed** across both configs and the variant below. The
Fmax gap roughly halved, and refinement is worth 16-20% of our wirelength — more
than it is worth to HeAP, because our legalised placement has more slack in it.

Both move paths are exercised: the paired runs report `233 cells 128 chains`,
so `try_swap_chain` is moving every one of the 128 LUT→FF clusters, matching
nextpnr's own "Constrained 128 LUTFF pairs".

### Routed wirelength, and where the gap actually is

HPWL is an estimate. Counting what the router really consumed — wires and pips
from each net's `ROUTING` record — gives the honest routed wirelength. 5 seeds
per config, all routed at router seed 1:

| config | routed wires | vs ref | pips | vs ref |
|---|---|---|---|---|
| nextpnr unpacked | 10320.0 (10147-10507) | — | 9659.6 | — |
| nextpnr packed | 9864.6 (9669-10002) | — | 9253.4 | — |
| ours, filter | 10646.0 (10479-10841) | **1.032x** | 10009.4 | 1.036x |
| ours, paired | 10212.4 (10103-10270) | **1.035x** | 9617.0 | 1.039x |
| ours, IO released | 10441.2 (10233-10719) | **1.012x** | 9794.6 | 1.014x |

Line the three metrics up and they disagree in a way that localises the problem:

| metric | filter | paired | IO released |
|---|---|---|---|
| HPWL | 1.112x | 1.043x | 0.982x |
| routed wires | 1.032x | 1.035x | 1.012x |
| **post-route Fmax** | **1.082x** | **1.054x** | **1.113x** |

We consume nearly as much routing as nextpnr — 3% more — yet lose 5-8% of Fmax.
The residual is therefore **not** a global wirelength or congestion problem: on
average our placement is about as economical as HeAP's. It is a *critical path*
problem. The IO-released variant states it most sharply: the best routed
wirelength of any configuration measured (1.012x) and the worst Fmax (1.113x).

That is where the remaining gap should be attacked — the timing-critical paths
specifically, not total wire.

### Runtime

Placement wall-clock, mean of 5 seeds, both figures covering the whole placement
pipeline (ours: DCD + legalisation + refinement; nextpnr: HeAP + SA refinement):

| | ours, 8 threads | ours, 1 thread | nextpnr |
|---|---|---|---|
| unpacked / filter | 0.48s | 0.66s | 0.51s |
| packed / paired | 0.40s | — | 0.33s |

Roughly at parity: marginally ahead on the unpacked config with 8 threads
(0.94x), 1.22x behind on packed, and about 1.29x behind single-threaded against
nextpnr's default. **No runtime claim is worth much here** — the whole placement
is half a second, our driver reports to one decimal, and a 20x20 fabric at 14%
LUT utilisation is not a runtime benchmark. This needs FPGA01/stereovision3
before anyone quotes a speed ratio.

### A clean case of HPWL lying

opt_trans locks IO cells as anchors for its continuous solve
(`lock_boundary_cells`) at the same LOCKED strength a user constraint uses, so
131 cells stay frozen through refinement while nextpnr's refiner moves IO
freely. Releasing them after legalisation (`OT_UNLOCK_IO=1`) does exactly what
it should on the metric it targets, and the opposite on the one that matters:

| variant | HPWL | vs ref | Fmax | vs ref |
|---|---|---|---|---|
| IO locked | 2029.6 | 1.112x | **42.81** | **1.082x** |
| IO released | **1791.6** | **0.982x** | 41.63 | 1.113x |

Releasing IO **beats nextpnr on wirelength outright** — 1791.6 against 1825 —
and loses 2.8% of routed Fmax doing it. A 12% HPWL improvement bought a
measurable Fmax regression. Keep the anchors; keep reporting Fmax.

## Caveats that bound the claim

1. **Legality: closed.** This caveat read "not verified, and this flatters
   opt_trans", then "measured, and we fail it". Both are now history: the
   placer enforces the arch rule and 10/10 placements route. (The LUT4→DFF
   pairing constraint, previously listed here too, is worth −1.7% to nextpnr;
   see the packing section.)
2. **HPWL bias: measured, not just argued.** DCD optimises a congestion-aware
   transport energy, so scoring only HPWL is biased toward nextpnr by
   construction. First quantified as 1.24-1.36x HPWL against 1.14-1.16x Fmax;
   those Fmax figures were single seed-1 runs and the corrected 5-seed numbers
   are 1.14x packed / 1.20x unpacked. Current figures are in "Result with
   refinement". Report Fmax — and see "A clean case of HPWL lying" for a change
   that improves HPWL past nextpnr while losing Fmax.
3. Synthetic fabric, one benchmark family (LFSR + accumulator), low LUT
   utilisation on two of three fabrics.

## Next steps, in value order

1. ~~Teach opt_trans arch legality~~ — **done**; see the fix section above.
   10/10 legal, 10/10 routed.
2. ~~Close the routed gap~~ — **halved**, by porting the stage we were missing.
   1.20x -> 1.082x unpacked, 1.14x -> 1.054x packed, on routed Fmax. The
   warning in this item proved its worth twice: the un-refined comparison was
   an HPWL artifact, and releasing the IO anchors improves HPWL past nextpnr
   while *losing* Fmax.
2b. **The remaining 5-8% is a critical-path gap, not a wirelength gap.**
   Routed wirelength is only 1.03x while Fmax is 1.05-1.08x, so our placement
   is about as economical with routing as HeAP's overall and loses on the
   timing-critical paths specifically. Target those. Both pipelines now end in
   the same refinement stage, though not on equal terms — ours refines with 131
   IO cells frozen and nextpnr's does not, and that restriction measured as
   *beneficial*.
3. **Re-derive the class split under legality.** The "local tightening" reading
   and the MST/Steiner numbers were taken on illegal placements. Legality cost
   only +1.3% so they are probably close, but the LUT->FF class is exactly the
   one legality governs, so that one needs redoing before it is trusted.
4. **Validate `mst_edge_weight` on the real benchmarks** before touching
   defaults.
5. **Find the unseeded source of run-to-run variation.** Fixed seed, fixed
   config and `num_threads = 1` still disagree run to run, which makes any
   single-run opt_trans measurement untrustworthy. Lead: pinning the fanout-128
   clock buffer dropped the observed spread well below the 2-11% seen with it
   free, so the unseeded path likely runs through boundary/IO placement.
6. Denser and larger fabrics; this design is IO-bound, which capped LUT
   utilisation at 14% on the 20x20.
7. Teach the placer the arch's bel-bucket rules, so `INBUF`/`OUTBUF` need not be
   retyped to `IOB` by hand in the driver. (The validity half of this item is
   now item 1.)

## Reproducing

```bash
cargo build --release --features test-utils --example npnr_baseline_placers
O=tools/npnr_compare/out; F=crates/nextpnr/tests/fixtures/npnr_baseline
OT_MST=4.0 OT_STEINER=1.0 ./target/release/examples/npnr_baseline_placers \
    $O/synth20.bin $F/constids.inc $O/bench64.json $O/placed20_64.json
```

Knobs: `OT_MST`, `OT_STEINER`, `OT_ITERS`, `OT_DCD_ITERS`, `OT_SEED`,
`OT_THREADS`. Average several runs — see the nondeterminism note.

To check legality, and to route our placement once it is legal:

```bash
OT_MST=4.0 OT_STEINER=1.0 OT_WRITE_PLACEMENT=$O/ot20_64.json \
    ./target/release/examples/npnr_baseline_placers \
    $O/synth20.bin $F/constids.inc $O/bench64.json $O/placed20_64_nopack.json
python3 tools/npnr_compare/check_slice_legality.py $O/ot20_64.json
NPNR_NO_LUTFF_PACK=1 $NPNR/build/nextpnr-himbaechel \
    --chipdb $O/synth20.bin --device EXAMPLE --json $O/ot20_64.json \
    --write $O/ot20_64_routed.json --seed 1
```

`NPNR_NO_LUTFF_PACK=1` is required on both sides: `constrain_cell_pairs` runs in
`pack()`, *after* the frontend has bound our injected bels, and would fight a
placement that does not already satisfy `delta_z=1`. Compare against
`placed20_64_nopack.json` so neither side packs.
Fabrics other than the committed 12x12 are regenerated with
`tools/npnr_compare/` (see its README); `out/` is scratch and not committed.

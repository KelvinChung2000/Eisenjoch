# What matching HeAP on FPGA01 actually costs

The target is HeAP's own FPGA01 result, taken on the same box with the same
`get_net_metric`: **52.8 s and post-placement HPWL 4 533 609**
(`HEAP_RESULT place_secs=52.8 total_hpwl=4533609`). The current best
deterministic arm is `steiner=1`, `DET_SWEEP=1`, `DET_CHUNK=4096`, 32 threads:
**1465 s and 7 387 808**. So 27.7x on time and 1.63x on quality.

This document does the arithmetic on what closing each half costs, because
the two halves are currently coupled through the same variable and pull in
opposite directions.

## Time decomposes almost entirely into refresh

Summing the per-iteration `DCD` line over all 20 iterations of that run:

| | seconds | share of wall |
|---|---|---|
| refresh | 1223.4 | 83.5 % |
| DCD sweep | 131.5 | 9.0 % |
| rest of the iteration loop | 53.3 | 3.6 % |
| outside the loop (parse, pack, legalise) | 56.8 | 3.9 % |

Refresh reduced to zero leaves **242 s**, still 4.6x HeAP. So no refresh lever
alone reaches the target, however good it is. That is a ceiling on the whole
per-net-cost line of attack, and it holds regardless of which pruning or
scheduling idea is tried.

## Quality is iteration-limited, and the run has not converged

The `line` metric over the last twelve iterations of the same run:

| iter | 8 | 11 | 14 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|
| line | 13 555 274 | 12 300 175 | 11 117 089 | 10 192 242 | 9 901 274 | 9 641 215 |

It falls 2.6 % in the final iteration and shows no flattening. Twenty
iterations is where the run stops, not where it converges.

HeAP's ratio of post-legalisation HPWL to `line` is 4 533 609 / 6 430 025 =
0.705; ours is 7 387 808 / 9 641 215 = 0.766. Matching HeAP's HPWL therefore
needs `line` near 6.0-6.4M, which is **-35 %** from 9 641 215. At the observed
2.6 % per iteration that is at least sixteen further iterations, and the rate
will flatten, so sixteen is a lower bound rather than an estimate.

## The two halves pull opposite ways

Iteration count is the only lever currently connecting them. Quality wants
~36+ iterations; time wants far fewer than 20. Holding quality at HeAP's level
while paying today's 73 s per iteration costs roughly 2900 s, which is 55x the
target rather than 27.7x.

**So the requirement is not 28x cheaper. It is ~55x cheaper per iteration at
equal quality**, and refresh going to zero supplies 6x of that. The remaining
9x has to come from somewhere other than the per-net solve.

Two candidates, neither measured:

Fewer iterations for the same quality, if the descent is currently wasting
them. At `cong_share = 100.0 %` the objective is congestion alone and base
wirelength is invisible to it, so the 2.6 %-per-iteration crawl may be an
artefact of an unbounded penalty rather than the true convergence rate.
`NPNR_OT_BPR_CAP` (cherry-picked to master as `c89da9f` and `34a8821`) bounds
the multiplier and is the direct test.

A cheaper per-net computation, which the uncongested measurements say is
available: `straight/star = 1.0000` with zero of 105 221 nets improved by the
search, and `star/manh = 1.0047`. The search is load-bearing only because the
congestion price is unbounded, which is the same root the cap addresses.

Both candidates reduce to the same question, which is why the cap is measured
first.

## Provenance

The `straight/star` and `star/manh` figures come from `aff4ecf` and `9b0d209`
on `flow/spreading-field`, taken on that branch's span-class device model, and
have not been reproduced on master. The cap commits were cherry-picked; the
`dual_lambda` term in the original `effective_resistance` was dropped because
no such field exists on master.

## Measured: bounding the BPR multiplier is worth -12.6 % HPWL

FPGA01, `steiner=1`, `DET_SWEEP=1`, `DET_CHUNK=4096`, 32 threads, 20 iters.
Deterministic, so each arm is one exact sample rather than a draw:

| `NPNR_OT_BPR_CAP` | post-leg HPWL | vs uncapped | wall | vs HeAP HPWL | vs HeAP wall |
|---|---|---|---|---|---|
| unbounded (reference) | 7 387 808 | — | 1465 s | 1.630x | 27.7x |
| 100 | 7 559 506 | +2.3 % | 1758 s | 1.667x | 33.3x |
| 10 | 6 456 365 | -12.6 % | 1562 s | 1.424x | 29.6x |
| **2** | **5 809 246** | **-21.4 %** | **1386 s** | **1.281x** | **26.3x** |

`cong_share` is 100.0 % uncapped and 40-46 % at cap 10.

The mechanism is the one the cap was built for. Uncapped, the congestion term
reaches `cong_share = 100.0 %` and base wirelength is invisible to the descent.
At cap 10 the two terms sit at 40-46 % congestion against a base that grows
from 10.4M to 19.4M, so the descent is optimising both. The gap to HeAP's
4 533 609 falls from 1.630x to **1.424x** on one knob.

100 being worse than unbounded is not a monotone curve, and one sample per arm
cannot say whether that is a real interior structure or a single bad landing.
It does not affect the reading at 10.

### Refresh cost now falls with iteration instead of rising

Per-iteration `refresh`, uncapped against cap 10:

| iter | 0 | 5 | 10 | 15 | 19 |
|---|---|---|---|---|---|
| uncapped | 121.4 s | 45.1 s | 67.7 s | — | ~60-88 s |
| cap 10 | 125.2 s | 77.2 s | 58.7 s | 45.0 s | **39.5 s** |

Uncapped, refresh turns back up after outer=6, which is the same boundary
where the loaded share jumps from 12 % to 85 % and the back-pass starts paying
for it. Capped, refresh declines monotonically from 125.2 s to 39.5 s. So the
cap does not only improve quality; it removes the late-run cost growth that
made longer horizons unaffordable. That matters because `line` is still
falling 2.5 % per iteration at iter 19 (8 451 352, down from 9 641 215
uncapped), so the horizon is exactly what quality needs.

Wall is 6.6 % worse over 20 iterations because the early iterations cost more.
The crossover is at outer=10.

## Chunk on `steiner=0`: flat, so the steiner=1 tuning does not transfer badly

| chunk | 2048 | 4096 | 8192 |
|---|---|---|---|
| post-leg HPWL | 10 730 527 | 10 767 033 | 10 709 599 |
| wall | 1270 s | 1206 s | 1263 s |

The spread is 0.53 % and not monotone, against a 4.3 % spread on `steiner=1`
between chunk 1024 and 76 660. Chunk is a real lever on `steiner=1` and close
to noise on `steiner=0`, so the +0.91 % generalisation penalty recorded for
4096 is not a mistuning and re-tuning does not recover it.


## Tighter is better on both axes, and the knee is not found yet

cap 2 improves quality by 21.4 % and wall by 5.4 % against uncapped, so it is
not a trade. Every tightening so far has moved both numbers the same way, and
the sweep has not yet turned. Caps 3, 1.5 and 1.2 are queued; the limit is
cap 1.0, which is the multiplier pinned at 1 and therefore no congestion price
at all.

That limit is the reason to read this result narrowly. **The cap trades away
the only congestion mechanism the placer has.** FPGA01's binding constraint
was already local routability rather than HPWL: the 2026-05-22 end-to-end run
never routed, failing short nets, and a router-agnostic check put ~20k interior
tile edges over capacity. Nothing here re-measures that, so a cap chosen purely
on HPWL and wall may be buying both by making the placement less routable. The
target as stated is HeAP's time and HPWL, and on those two numbers tighter is
strictly better; routability needs its own measurement before any cap becomes
a default.

## The cap also fixes the convergence rate, which is what puts HPWL in reach

Per-iteration change in `line` over the last ten iterations: uncapped 2.6 %,
cap 2 **4.0 %**, still 3.67 % at iter 19. The corridor shrinks with it, so the
search gets cheaper every iteration rather than more expensive:

| iter | 0 | 5 | 10 | 15 | 19 |
|---|---|---|---|---|---|
| `settle_avg` | 9381 | 6823 | 4949 | 3692 | **2990** |
| refresh | 126 s | 67 s | 51 s | 39 s | **32 s** |

Refresh totals 1147 s at cap 2 against 1223 s uncapped, over the same twenty
iterations and reaching a much better placement.

Extrapolating the horizon is then arithmetic rather than hope. Our
post-legalisation HPWL to `line` ratio is 5 809 246 / 7 789 161 = 0.746, so
HeAP's 4 533 609 corresponds to `line` near 6.08M, which is -22 % from
7 789 161. At a rate decaying through 3.67 % that is seven to nine further
iterations, costing roughly 340 s.

**So HPWL parity projects at about 28 iterations and ~1730 s.** That is one
half of the target reached and the other half missed by 33x. The projection is
an extrapolation of a decaying rate and is worth exactly what such an
extrapolation is worth; a 32-iteration arm at the winning cap is queued to
replace it with a measurement.

## Measured: 32 iterations at cap 1.2 beats HeAP's HPWL

The full cap sweep, all `steiner=1`, `DET_CHUNK=4096`, 32 threads, 20 iters:

| cap | inf | 100 | 10 | 5 | 3 | 2 | 1.5 | 1.2 |
|---|---|---|---|---|---|---|---|---|
| HPWL | 7 387 808 | 7 559 506 | 6 456 365 | 6 178 251 | 5 934 823 | 5 809 246 | 5 758 090 | **5 750 533** |
| wall | 1465 s | 1758 s | 1562 s | 1489 s | 1518 s | 1386 s | 1457 s | **1335 s** |

Monotone below 10 and flattening between 1.5 and 1.2, so the knee is near 1.2.

Taking that cap to 32 iterations, which the convergence rate said was the
affordable route to quality:

| | HPWL | wall |
|---|---|---|
| HeAP | 4 533 609 | 52.8 s |
| **ours, cap 1.2, 32 iters** | **3 767 053** | 1726 s |

Both figures are final-placement HPWL under the same `get_net_metric`, ours
post-legalisation (pre-legalisation is 3 765 433, so legalisation costs 0.04 %)
and HeAP's post-placement, its own final. Net counts differ slightly, 105 117
against 105 220.

**The quality half of the target is met and passed: 0.831x HeAP, from 1.630x
at the start of this work.** The time half is not: 1726 s against 52.8 s is
32.7x, worse than the 27.7x it started at, because quality was bought with
iterations.

The routability caveat above now binds harder, not less. cap 1.2 leaves a
congestion multiplier that can never exceed 1.2, so the placer is close to
pure wirelength, and pure wirelength is exactly the regime that produced the
2026-05-22 unroutable placement. An HPWL that beats HeAP means nothing if the
result does not route, and that is unmeasured here.

## Measured: at a bounded price the search is 3.5 % of what it costs

`straight/star`, the monotone walk priced against the Dijkstra label at the
same `R_eff`, over eight iterations:

| outer | 0 | 1 | 2 | 4 | 7 |
|---|---|---|---|---|---|
| uncapped | 1.393 | 2.8e7 | 4.8e6 | 1816 | 22 768 |
| cap 1.2 | 1.393 | 1.110 | 1.057 | 1.033 | **1.036** |

Uncapped, the walk is hopeless once congestion develops, which is what made
the search load-bearing. Bounded, the walk is within **3.5 %** of the search
and stays there, and only 19 320 of 105 117 nets gain more than 5 % from
searching.

That is the green light for the time half. Refresh is 83.5 % of wall and costs
`O(corridor area)` per net; the walk costs `O(path length x degree)`, roughly
a thousand edge inspections per sink against a few hundred thousand
relaxations. What it does not yet supply is `dist_cache` coverage within
`cache_radius_tiles` of each pin, which the sweep reads through
`evaluate_cell_at` and a single walked path does not fill. That gap is the
design question, not the ratio.

## Negative: a label-accurate closed form is not a field-accurate one

`NPNR_OT_ANALYTIC_DIST=1` serves `dist_cache.get` from `k * manhattan(driver,
node)` with `k` fitted per net to the labels refresh just wrote. Against the
cap 1.2 control at 20 iterations:

| | HPWL | wall |
|---|---|---|
| control, stored labels | 5 750 533 | 1335 s |
| analytic field | **8 725 198** | 1591 s |

**+51.7 %.** The refresh still ran in both arms, so this measures the field
alone, and `line` comes from `continuous_line_estimate` over cell coordinates
rather than from the cache, so the comparison is not circular.

The shape of the failure is the useful part:

| outer | 0 | 2 | 9 | 10 | 14 | 19 |
|---|---|---|---|---|---|---|
| control `line` | 17 072 682 | 15 688 776 | 11 620 900 | 11 120 807 | 9 331 223 | 7 602 848 |
| analytic `line` | 14 556 961 | 12 390 351 | 11 239 658 | 11 187 463 | 11 045 139 | 10 949 821 |
| difference | -14.7 % | -21.0 % | -3.3 % | +0.6 % | +18.4 % | +44.0 % |

The analytic field descends **21 % faster than the exact one for three
iterations**, crosses over at outer=10, then stalls: it moves 2.1 % over the
last nine iterations against the control's 31.6 %.

The mechanism is that `k * manhattan` from the driver is a cone, and a cone has
no congestion structure. Early on that is an advantage, because the exact
labels are rugged and only defined within `cache_radius_tiles` of a pin, so a
smooth field that is finite everywhere lets cells move further per sweep. Once
the placement is roughly right, the cone's minimum is reached and there is
nothing left to descend, while the exact labels keep carrying the congestion
field that drives the remaining 31.6 %.

So `straight/star = 1.036` was measured on driver-to-sink **labels**, and label
accuracy does not imply the **field** is replaceable. Those are different
requirements, and this arm is what distinguishes them. Replacing refresh needs
a cheap field that still varies with congestion, not a cheap distance.

One lead falls out of the crossover rather than the conclusion: the cone wins
for the first three iterations, and those are the most expensive refreshes
(118 s at outer=0 against 55 s later). Running analytic early with no Dijkstra
at all and switching to exact at the crossover would cut the costliest part of
refresh while keeping the field that does the late work. Unmeasured.

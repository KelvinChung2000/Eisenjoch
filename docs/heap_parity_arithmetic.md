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

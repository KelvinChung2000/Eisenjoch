# The physics composite, measured

Four changes were proposed to `opt_trans` by two independent routes — a
first-principles mathematical audit, and a brainstorm of non-electrostatic
physical analogies. Both landed on the same list, which is why it was worth
building. This is what the list is actually worth, measured.

All figures: `stereovision3`, `xc7_large`, 20 outer iterations, HPWL over 298
nets, via `cargo run --release --example ot_trace_design`.

## Read the noise floor first

Twelve runs of the **identical configuration** span **7650–7901 (3.3%)**, and
repeat runs at a *fixed seed* still differ. Nothing under ~3% on this design is
a result. This is the documented `opt_trans` nondeterminism, not new.

## Scoreboard

| step | change | verdict |
|---|---|---|
| 1 | PathFinder history `h` | **gate FAILED** — real but not the oscillation cure |
| 2 | union usage accounting | correct; defect reproduced in code; not yet calibrated |
| 3 | tension / driver hole | hole **confirmed**; closed by the *centroid*, not MST — **−17.5%** |
| 4 | proximal `mu·\|Δx\|` | doesn't damp; **protects** quality from the swing — **−10%** |

Step 4 as originally scoped had a second half — replacing the no-rollback
anneal with converge-at-each-θ. **That was left out**, deliberately: the
measurements above show the limit cycle is driven by the BPR channel's scale
and is not damped by move-level mechanics, so an iteration-schedule change
targets a mechanism that has now been measured not to be the problem, at a
large runtime cost. It remains available if the scale issue is fixed first.

## Step 1: the gate failed

The hypothesis was that the missing PathFinder `h` causes the limit cycle. The
control reproduces the cycle: `cong` swings **193%** peak-to-peak post-transient
with up to **120 pipes past 100× base**.

| arm | base swing | util swing | pipes ≥100× | HPWL |
|---|---|---|---|---|
| A  BPR only (default) | 4.9% | 79.1% | 120 | 7726 |
| B  hardening only (step 0.1) | 7.1% | 50.5% | **0** | 7562 |
| C  BPR + hardening (step 0.1) | 5.9% | 89.1% | 204 | 7728 |
| D  **neither (null)** | 6.1% | 70.6% | 0 | 7533 |

Arm **D is the one that matters**. Hardening-only ≈ no congestion pricing at
all, on every axis including HPWL. `history_total` at step=0.1 is **19.4**
against a cost scale of ~16000 — 0.1%, i.e. inert. My arm was under-dosed by
~1000×.

At step=100 (`history_total` 2483, ~15% of scale) it is *not* inert: peak
occupancy falls 1.73 → 1.29 and final 1.68 → 0.77. So the mechanism works. But
the trace still does not go monotone (util swing 76.8%).

Arm C above is at step=0.1 and therefore proves nothing on its own — it was run
before the dose problem was understood. Rerun at step=100, where
`history_total` reaches **33206**, roughly twice the base cost scale:

| arm | cong swing | pipes ≥100× | HPWL | `history_total` |
|---|---|---|---|---|
| A  BPR only | 193.0% | 120 | 7726 | — |
| C  BPR + hardening, step 0.1 | 191.2% | 204 | 7728 | ~0 |
| C  BPR + hardening, **step 100** | **187.9%** | 110 | 7787 | 33206 |

Even a dual twice the size of the base cost scale leaves the cycle intact.

**Conclusion: the binding oscillation driver is the BPR channel's unbounded
scale, not the missing history.** Deleting BPR removes the blow-up entirely
(zero pipes above 2×); adding `h` on top of BPR changes nothing. This settles
the ablation the audit flagged as open.

## Step 3: the driver hole is real, but MST is not what closes it

`driver_cost_is_flat_without_a_tension_term` shows a driver's cost is
**bit-identical at every legal position** under the shipped defaults — its
argmin is whatever the tie-break picks.

| arm | seed 1 | seed 7 | seed 42 |
|---|---|---|---|
| default | 7901 | 7720 | 7881 |
| `steiner=1` | **6492** | **6501** | **6491** |

~17.5% on every seed, 5× the noise floor. It also collapses run-to-run spread
from 3.1% to **0.15%** — a convergence effect, which is the more interesting
half.

MST on top does nothing: `mst=1` → 6437 vs 6469 (inside noise), `mst=2/4`
worse. On this design the shared-hub pull is what matters and pairwise tension
is redundant with it. **This contradicts the expectation that MST was the
high-leverage change.**

## Step 4: right effect, wrong reason

The proximal term does not damp the limit cycle — `cong` swing has no trend in
`mu` and pipes ≥100× gets *worse*. But at `mu=1` HPWL is 6906 / 7015 / 7065
across three seeds against a control range of 7650–7891: ~10%, entirely below
the control range. It stops cells being flung around by a field swinging 180%
per iteration, without stopping the swinging. Neutral in the already-stable
`steiner=1` configuration, which fits.

## What this changes

The composite's headline claim was that memory is the missing piece. It is not.
Two cheap terms that were already in the tree at weight 0.0 — the **centroid**
pull and a **proximal** charge — are worth ~17.5% and ~10%, while the term with
the most theory behind it is worth approximately nothing at a sane dose.

No defaults were changed. One design is not enough, and the centroid term
carries a known synthetic-vs-real reversal risk.

## Caveats

- **One design.** `stereovision3` is small. Nothing here is confirmed at scale.
- **FPGA01 was abandoned — no longer.** It needed ~50 GB RSS in this placer
  even with `CORRIDOR_TIGHT=1 HALO_MAX=3 CACHE_SLACK=1 CACHE_RADIUS=3`, and
  OOM'd a 61 GB box. As of 2026-08-19 it completes at **2948 MB peak, 2492 s**,
  at `NPNR_OT_MAX_ITERS=20` and **stock settings** — none of those four tuning
  vars set, i.e. a strictly harder configuration than the one that OOM'd. Three
  fixes did it: `ChunkUsage` bounded by concurrency rather than total work
  (`cd9ed31`), the per-cell validity mask (`2bf83b6`), and the legalizer ring
  query (`fec5cb1`). The caveats below that rest on "one small design" can now
  be retested at scale.
- **Union accounting runs, but is not calibrated.** One end-to-end sv3 run
  (`NPNR_OT_UNION_USAGE=1`) completes cleanly, and shows exactly the predicted
  effect: `max_util` reaches **4.03** against ~1.5 under the 1/K split, cong
  swing 207.9%, HPWL 7836. Counting each tree edge once is the correct
  accounting, but it moves the BPR knee, so `alpha`/`beta` must be recalibrated
  before the default can flip. That calibration was not done.
- HPWL only. No routing or timing was run.

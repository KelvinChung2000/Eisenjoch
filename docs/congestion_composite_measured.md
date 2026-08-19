# The physics composite, measured

Four changes were proposed to `opt_trans` by two independent routes — a
first-principles mathematical audit, and a brainstorm of non-electrostatic
physical analogies. Both landed on the same list, which is why it was worth
building. This is what the list is actually worth, measured.

All figures: `stereovision3`, `xc7_large`, 20 outer iterations, HPWL over 298
nets, via `cargo run --release --example ot_trace_design`.

## Read the noise floor first

Nine runs of the **identical configuration** span **7650–7891 (3.1%)**, and
repeat runs at a *fixed seed* still differ. Nothing under ~3% on this design is
a result. This is the documented `opt_trans` nondeterminism, not new.

## Scoreboard

| step | change | verdict |
|---|---|---|
| 1 | PathFinder history `h` | **gate FAILED** — real but not the oscillation cure |
| 2 | union usage accounting | correct; defect reproduced in code; not yet calibrated |
| 3 | tension / driver hole | hole **confirmed**; closed by the *centroid*, not MST — **−17.5%** |
| 4 | proximal `mu·\|Δx\|` | doesn't damp; **protects** quality from the swing — **−10%** |

## Step 1: the gate failed

The hypothesis was that the missing PathFinder `h` causes the limit cycle. The
control reproduces the cycle: `cong` swings **193%** peak-to-peak post-transient
with up to **120 pipes past 100× base**.

| arm | base swing | util swing | pipes ≥100× | HPWL |
|---|---|---|---|---|
| A  BPR only (default) | 4.9% | 79.1% | 120 | 7726 |
| B  hardening only | 7.1% | 50.5% | **0** | 7562 |
| C  BPR + hardening | 5.9% | 89.1% | 204 | 7728 |
| D  **neither (null)** | 6.1% | 70.6% | 0 | 7533 |

Arm **D is the one that matters**. Hardening-only ≈ no congestion pricing at
all, on every axis including HPWL. `history_total` at step=0.1 is **19.4**
against a cost scale of ~16000 — 0.1%, i.e. inert. My arm was under-dosed by
~1000×.

At step=100 (`history_total` 2483, ~15% of scale) it is *not* inert: peak
occupancy falls 1.73 → 1.29 and final 1.68 → 0.77. So the mechanism works. But
the trace still does not go monotone (util swing 76.8%), and arm C shows
hardening does **not** tame BPR when BPR is on — 191% vs the control's 193%.

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
- **FPGA01 was abandoned.** It needs ~50 GB RSS in this placer even with
  `CORRIDOR_TIGHT=1 HALO_MAX=3 CACHE_SLACK=1 CACHE_RADIUS=3`; it OOM'd a 61 GB
  box. The gitignore issue was never the blocker — memory is.
- **Union accounting is unmeasured end-to-end.** It is correct per unit test
  but raises per-pipe occupancy vs the 1/K split, so the BPR knee moves and
  `alpha`/`beta` need recalibrating before the default can flip.
- HPWL only. No routing or timing was run.

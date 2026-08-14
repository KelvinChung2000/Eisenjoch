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

## Result

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

## Where the gap is

Splitting by whether a net touches an IO cell (20x20):

| | nextpnr | opt_trans | excess |
|---|---|---|---|
| io-touching (260 nets) | 1470 | 1870 | 400 |
| core (427 nets) | 387 | 2043 | **1656** |

81-88% of the excess is in core logic-to-logic nets, consistently across all
three fabrics. Core wirelength alone is 5.3x nextpnr's on the 20x20 case.

This kills the obvious hypothesis. `lock_boundary_cells` only locks IO cells
that are *already placed*, and nothing pre-places them here, so IO cells keep
whatever bel the initial discrete binding gave them and DCD never moves them
(`TypeAwarePlacement: 2 cell types` — only LUT4 and DFF). That looked like the
answer and is not: IO placement is nearly free, the loss is in the logic core.

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

## Caveats that bound the claim

1. **Legality is not verified, and this flatters opt_trans.** The example uarch
   enforces `isBelLocationValid` → `slice_valid(x, y, z/2)` and constrains
   LUT4→DFF pairs (`Constrained 128 LUTFF pairs` on the 20x20 run). We enforce
   neither. nextpnr's number is a *legal* placement; ours is not known to be.
   Treat opt_trans's wirelength as a lower bound on what it could achieve
   legally — the true gap is at least this large, possibly larger.
2. **HPWL is nextpnr's objective, not opt_trans's.** DCD optimises a
   congestion-aware transport energy; the whole premise is trading wirelength
   for routability. Scoring only HPWL is biased toward nextpnr by construction.
   The honest comparison routes both placements and compares routed wirelength
   and routing success. That is the main thing this harness still cannot do.
3. Synthetic fabric, one benchmark family (LFSR + accumulator), low LUT
   utilisation on two of three fabrics.

## Next steps, in value order

1. **Route both placements and compare.** Removes caveat 2 and most of caveat 1
   at once. Writing our placement back out as JSON with `NEXTPNR_BEL` attributes
   lets nextpnr route *both*, so the router is identical and legality is checked
   by the tool that defines it.
2. **Validate `mst_edge_weight` on the real benchmarks** before touching
   defaults.
3. **Find the unseeded source of run-to-run variation.** Fixed seed, fixed
   config and `num_threads = 1` still disagree run to run, which makes any
   single-run opt_trans measurement untrustworthy.
4. Denser and larger fabrics; this design is IO-bound, which capped LUT
   utilisation at 14% on the 20x20.
5. Teach the placer the arch's bel-bucket and validity rules, so `INBUF`/`OUTBUF`
   need not be retyped to `IOB` by hand in the driver.

## Reproducing

```bash
cargo build --release --features test-utils --example npnr_baseline_placers
O=tools/npnr_compare/out; F=crates/nextpnr/tests/fixtures/npnr_baseline
OT_MST=4.0 OT_STEINER=1.0 ./target/release/examples/npnr_baseline_placers \
    $O/synth20.bin $F/constids.inc $O/bench64.json $O/placed20_64.json
```

Knobs: `OT_MST`, `OT_STEINER`, `OT_ITERS`, `OT_DCD_ITERS`, `OT_SEED`,
`OT_THREADS`. Average several runs — see the nondeterminism note.
Fabrics other than the committed 12x12 are regenerated with
`tools/npnr_compare/` (see its README); `out/` is scratch and not committed.

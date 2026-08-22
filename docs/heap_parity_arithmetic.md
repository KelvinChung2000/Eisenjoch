# FPGA01 against HeAP: where the time goes

Every number below is FPGA01 on `xc7_large.bin`, 32 threads, one machine, one
binary per comparison. The placer call is timed the same way in both drivers,
`PlacerOptTrans.place` against `PlacerHeap.place` and nothing around either, so
the two figures divide. Wirelength is post-legalisation HPWL as
`report_post_legalization` reports it.

## Position

| | place_secs | HPWL | iterations |
| --- | --- | --- | --- |
| HeAP | 1.40 | 4 533 609 | 2 |
| opt_trans, 11 sweeps | 2.07 | 4 523 586 | 11 |
| opt_trans, 13 sweeps | 2.17 | 4 281 980 | 13 |

At eleven sweeps the wirelength is 0.998x HeAP's for 1.48x the time. At
thirteen it is 0.944x for 1.55x, so the placement is better than the reference
by 5.6 % and the extra two sweeps cost 0.10s.

The session opened at 27.7x and a floor of 51s that both placers paid.

## What the 2.07s is

    0.18  prepare_discrete and the network build, shared with HeAP
    0.09  setup before the first sweep
    1.03  eleven sweeps
    0.67  legalisation
    0.10  post-legalisation report

A sweep costs 91ms for the first and about 60ms after, so four extra sweeps buy
12 % of wirelength for 0.23s. Time is nearly flat in the iteration count and
wirelength is not, which is why thirteen sweeps beat HeAP on both the number
that matters and the number it costs.

## The five things that were not the algorithm

Each was found by profiling, removed, and checked by the placement reproducing
its HPWL exactly rather than approximately. Deterministic sweeps
(`NPNR_OT_DET_SWEEP=1`) make that a real test: a run that changes any decision
changes the final wirelength.

`snap_and_bind_cells_to_bels` walked the whole BEL pool for every cell, about
5e10 distance evaluations on FPGA01, and was 80 % of the profile. Manhattan
distance is constant inside a tile, so a ring scan stopping at the first radius
holding an available BEL finds the same BEL. `prepare_discrete` went from 51.1s
to 0.3s, which is shared, so it took 50s off HeAP too.

`CellNetMap` built a topological sort over 150 000 cells that only the
sequential sweeps read. Iteration 0 went from 1792ms to 91ms.

The same map was rebuilt twice per outer iteration for incidence data the
netlist fixes, then handed out by clone. It is built once and shared through an
`Arc`.

`collect_net_infos_simple` resolved and lowercased 105 220 net names twice an
iteration to decide which nets are constants or clocks, a verdict that cannot
change while cells move. `NameFilteredNets` decides it once.

`report_top_net_hpwl` ranked 105 000 nets inside `place()` unconditionally,
1.5s, more than legalisation costs. HeAP's driver computes its equivalent after
stopping the clock, so leaving it on made every place-time ratio wrong in our
favour. It is behind `NPNR_OT_REPORT_TOP_NETS=1` now.

## What the parity configuration actually runs

    NPNR_OT_BPR_CAP=1.2  NPNR_OT_SWEEP=jacobi_bisect  NPNR_OT_ANALYTIC_UNTIL=999
    NPNR_OT_STEINER=1    NPNR_OT_CACHE_NETMAP=1       NPNR_OT_SKIP_PIPES=1
    NPNR_OT_DET_SWEEP=1  NPNR_OT_DET_CHUNK=4096       NPNR_OT_THREADS=32

Three of those change what the placer is rather than how fast it runs.
`ANALYTIC_UNTIL=999` replaces the per-net Dijkstra with a cone field for the
whole run, so every iteration logs `solves=0 skip=105117`. `BPR_CAP=1.2` bounds
the congestion price so tightly that `E_decomp` reports `cong_share=0.0%` and
`sat_pipes=0`. `SKIP_PIPES=1` then skips building the pipe graph at all, and
the placement is bit-identical with it gone, which is the proof rather than the
claim that the graph was inert.

So parity is reached with the Dijkstra and the BPR congestion model, the two
mechanisms that distinguish this placer, provably switched off. What remains
optimising is a cone-shaped distance field under discrete coordinate descent.

## Whether 0.998x is parity on the right metric

Neither placer has a congestion term in this configuration, so the honest test
is our placement against HeAP's under one metric, not ours against an absolute.

| | avg demand/cap | over capacity | interior | cost |
| --- | --- | --- | --- | --- |
| opt_trans | 2.999 | 18 778 | 18 673 | 9.276e9 |
| HeAP | 1.240 | 22 205 | 22 191 | 6.669e9 |

Global routability reports 0 infeasible nets on both, 15 411 inconclusive
against HeAP's 12 473.

Mixed, and in the same regime. Our placement has fewer over-capacity tile edges
than the placement it claims parity with, and a higher average ratio and total
cost. What this rules out is the specific worry that `BPR_CAP=1.2` plus the
pure analytic field bought wirelength by dumping congestion in a way HeAP does
not. What it does not establish is that either placement routes: the metric
draws an independent Bresenham line per driver-sink pair, so a high-fanout net
contributes fanout-many lines where a shared Steiner tree contributes one, and
the absolute ratios are overcounts. It is comparable between two placements of
one design on one device, which is the only use made of it here.

## What is left

The gap is convergence rate. HeAP reaches 4.53M in two rounds of solve, spread
and legalise at 0.70s each; a sweep costs us 60ms and we need eleven. We are
2.8x faster per iteration and take 5.5x as many, so no further work on the
sweep closes it. Either the sweep makes more progress per pass, which is the
colored Gauss-Seidel and momentum design, or 1.48x stands.

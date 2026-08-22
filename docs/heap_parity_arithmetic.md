# FPGA01 against HeAP: where the time goes

Every number below is FPGA01 on `xc7_large.bin`, 32 threads, one machine, one
binary per comparison. The placer call is timed the same way in both drivers,
`PlacerOptTrans.place` against `PlacerHeap.place` and nothing around either, so
the two figures divide. Wirelength is post-legalisation HPWL as
`report_post_legalization` reports it.

## Position

Ten paired runs, alternating placer, same binary:

| | place_secs | spread | HPWL |
| --- | --- | --- | --- |
| HeAP | 1.271 | 1.25-1.29 | 4 533 609 |
| opt_trans | 1.337 | 1.32-1.38 | 4 520 199 |

1.052x on time and 0.997x on wirelength: the placement is 13 410 better than
the reference and takes 66ms longer. The distributions do not overlap, so the
66ms is a real difference rather than run-to-run scatter, and it is the last
thing between here and the goal.

The session opened at 27.7x and a floor of 51s that both placers paid.

## What the 1.35s is

    0.17  prepare_discrete and the network build, shared with HeAP
    0.08  setup before the first sweep
    0.07  refreshing net infos, eleven iterations
    0.07  the per-iteration line estimate
    0.55  eleven sweeps
    0.45  legalisation

Above the shared floor that is 1.18s of our own work against HeAP's 1.13s.

Two of those numbers exist because a timer was cheaper than a guess. The net
info refresh and the line estimate were each expected to be several tenths and
are 0.07s; what is left in the outer loop is the sweeps themselves.

A sweep costs 91ms for the first and about 60ms after, so four extra sweeps buy
12 % of wirelength for 0.23s. Time is nearly flat in the iteration count and
wirelength is not, which is why thirteen sweeps beat HeAP on both the number
that matters and the number it costs.

## The nine things that were not the algorithm

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

The sorted legalizer rebuilt each cell's pin-port list for every candidate BEL
it tested, and the FPGA01 candidate loop rejects 1 492 190 of them on the
shared-mux test alone. `is_legal_template` already existed and `ring.rs` was
already using it. Legalisation went from 0.71s to 0.52s with the reject counts
identical, which is the check that the two legality tests agree.

`NetInfo` was rebuilt from the netlist twice an outer iteration to change one
integer per pin. A `PinSource` per pin records whether the position follows a
movable cell, with a cluster offset, or a BEL that does not move, and the two
buffers now live outside the loop so no copy is made at all. Net info work
0.32s to 0.07s.

Two full HPWL passes ran inside `place()` that no placement decision reads.
HeAP's driver computes its equivalent after stopping its clock, so charging
ours inside the call measured the two placers over different spans.

`bel_pin_wire` matched a port against a BEL's pins by resolving both sides to
`&str` and calling `memcmp`, once per pin per candidate BEL. Constids that
spell the same string now collapse onto one representative so the match is an
integer compare, and the legalizer's template resolves the port's constid once
per cell rather than once per candidate.

## Why the profile stopped helping

`perf` reports CPU time. The sweep runs on 32 threads and legalisation runs on
one, so a line worth 10 % of samples inside the sweep is worth about a
thirtieth of that in wall clock, while a line inside legalisation is worth all
of it. Three changes aimed at what the flat profile showed returned nothing
measurable before that was noticed. Timers, not samples, produced every number
in the decomposition above.

## Where the last 66ms is not

Three changes aimed at what the profile showed returned nothing measurable:
pre-sizing the driver-node registry, replacing the hash set in
`continuous_line_estimate` with a sorted vector, and precomputing
`net_path_weight` to avoid a hash lookup per candidate. The `HashMap::insert`
line that motivated the last two is spread across callers rather than
concentrated in one, and cumulative percentages in a `--children` report read
as addressable time when they are not. They are kept because they are right.

The one configuration change that helped came from a sweep rather than a
profile. `NPNR_OT_DCD_ITERS=2` is both faster than the default 8 and produces a
better placement, 1.35s and 4 520 199 against 1.41s and 4 523 586, because
inner iterations past the second mostly re-confirm a cell's position.
`NPNR_OT_SOFTMIN_THETA_START` and `_END` do nothing at all here: 24 arms across
6 to 9 outer iterations returned byte-identical wirelength at every setting,
because the softmin temperature only feeds the Dijkstra soft-path assignment
that the analytic field replaces.

## What the parity configuration actually runs

    NPNR_OT_BPR_CAP=1.2  NPNR_OT_SWEEP=jacobi_bisect  NPNR_OT_ANALYTIC_UNTIL=999
    NPNR_OT_STEINER=1    NPNR_OT_CACHE_NETMAP=1       NPNR_OT_SKIP_PIPES=1
    NPNR_OT_CACHE_NETINFO=1                           NPNR_OT_DCD_ITERS=2
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

The gap is 66ms, and both halves of what is left are algorithmic rather than
accidental.

Legalisation is 0.46s, of which 0.27s is the candidate loop. It rejects
1 492 190 BELs on the shared-mux test where HeAP's two runs of the same
legalizer reject 16 338 between them, because HeAP legalises from a spread
placement and we legalise from an overlapped one. Cutting it means changing
which candidates are offered, which changes the placement.

The sweeps are 0.65s over eleven passes. HeAP reaches 4.53M in two rounds of
solve, spread and legalise. That is convergence, and it is what the colored
Gauss-Seidel and momentum design addresses.

Neither is another pass over the profile. What tuning was available has been
taken: `NPNR_OT_DCD_ITERS=2` improves both axes over the default 8, the softmin
temperature does nothing at all in this configuration, and thread count and
determinism chunk move the wall clock by less than the run-to-run spread.

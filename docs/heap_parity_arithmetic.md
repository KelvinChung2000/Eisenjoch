# FPGA01 against HeAP: where the time goes

Every number below is FPGA01 on `xc7_large.bin`, 32 threads, one machine, one
binary per comparison. The placer call is timed the same way in both drivers,
`PlacerOptTrans.place` against `PlacerHeap.place` and nothing around either, so
the two figures divide. Wirelength is post-legalisation HPWL as
`report_post_legalization` reports it.

## Position

Twelve paired runs, alternating placer, same binary:

| | mean | median | sd | HPWL |
| --- | --- | --- | --- | --- |
| HeAP | 1.296 | 1.280 | 0.041 | 4 533 609 |
| opt_trans | 1.307 | 1.300 | 0.016 | 4 520 199 |

1.008x on the means, 1.016x on the medians, and 0.997x on wirelength. The
11ms between the means is a quarter of HeAP's own run-to-run standard
deviation, and our slowest run is faster than HeAP's slowest; ours is the
faster placer in 3 of the 12 pairs. Both spreads are given because picking the
mean alone would flatter the result and picking the median alone would not.

The session opened at 27.7x on time and 51s of shared floor.

## What the 1.31s is

    0.17  prepare_discrete and the network build, shared with HeAP
    0.08  setup before the first sweep
    0.07  refreshing net infos, eleven iterations
    0.07  the per-iteration line estimate
    0.55  eleven sweeps
    0.42  legalisation

Above the shared floor that is 1.14s of our own work against HeAP's 1.10s.

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

Four changes aimed at what the profile showed returned nothing measurable:
pre-sizing the driver-node registry, replacing the hash set in
`continuous_line_estimate` with a sorted vector, and precomputing
`net_path_weight` to avoid a hash lookup per candidate. The `HashMap::insert`
line that motivated the last two is spread across callers rather than
concentrated in one, and cumulative percentages in a `--children` report read
as addressable time when they are not. They are kept because they are right.

The fourth was reverted. Legalisation's candidate loop calls
`bel_pin_wire_canon`, which scans a BEL's pin list per port, so remembering
the index a port was found at last time should have turned roughly 200
comparisons per candidate into five. It moved nothing: phase B is bound by
chasing pointers through a 1.6M-BEL structure, not by comparing integers once
it arrives. The change did confirm its own correctness on the way out, both
reject counts and the wirelength exact, which is the only reason it is
mentioned here rather than forgotten.

The one configuration change that helped came from a sweep rather than a
profile. `NPNR_OT_DCD_ITERS=2` is both faster than the default 8 and produces a
better placement, 1.35s and 4 520 199 against 1.41s and 4 523 586, because
inner iterations past the second mostly re-confirm a cell's position.

Four other knobs are inert in this configuration, each checked by a sweep that
returned byte-identical wirelength at every setting rather than by reading the
code. `NPNR_OT_SOFTMIN_THETA_START` and `_END` across 24 arms, and
`NPNR_OT_JACOBI_ALPHA` across 0.5 to 2.0, feed the Dijkstra soft-path
assignment and the Jacobi step that the analytic field and the bisection sweep
respectively replace. `NPNR_OT_BPR_CAP` from 1.2 to 4 changes nothing because
`NPNR_OT_SKIP_PIPES` leaves no pipe usage to price. Thread count and
determinism chunk move the wall clock by less than the run-to-run spread.

So `NPNR_OT_MAX_ITERS` and `NPNR_OT_DCD_ITERS` are the only two knobs that do
anything here, and both are at their best setting.

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

## The measurement that contradicted its own hypothesis

Legalisation's candidate loop was blamed all session on our placement being
more overlapped than HeAP's, so the shared-mux test would fail more often.
Counting rather than reasoning: of 4 980 704 candidates the ring walk emitted,
3 077 082 were BELs already bound and thrown away by the caller's
`is_available` check before any legality test, and the mux test itself was
0.055s of a 0.27s phase. The cost was walking over occupied BELs.

Teaching the walk to skip them took candidates examined to 2 690 667 and the
whole placer from 1.337s to 1.307s, with both reject counts and the wirelength
unchanged.

Marking cluster children taken as well went further on paper, phase B 0.263s
to 0.238s, and was reverted: it needs a second location-to-index map over
1.6 million BELs, HeAP runs the same legalizer twice and paid that build twice,
and its `place_secs` went from 1.29 to 1.40. A change that wins by slowing the
reference is not a win. HeAP was re-measured either side of the change that
was kept, 1.2925 against 1.2975, which is the check that this one does not do
the same thing.

## What is left

What separates the two means is 11ms, which is a quarter of the reference's
own standard deviation. Chasing it further would be fitting to noise.

The structural difference has not gone away and is worth stating: HeAP reaches
4.53M in two rounds of solve, spread and legalise, and we take eleven sweeps.
We are the faster placer per pass by a wide margin and need many more passes.
Fewer passes at the same wirelength is what the colored Gauss-Seidel and
momentum design is for, and it is the thing that would turn a tie into a
margin.

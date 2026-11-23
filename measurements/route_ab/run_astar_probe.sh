#!/bin/bash
# Where the ~320ms per net goes: the opt_trans arm again, capped at 15 minutes,
# with the per-sink A* outcome counters. 900s reaches the rate plateau (which
# sets in by t=400s) without spending another 90 minutes to learn the same thing.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== astar probe started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 900 $BIN > $OUT/fpga01_astar_probe.log 2>&1
echo "probe exit=$? $(date -Is)"

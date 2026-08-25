#!/bin/bash
# Does the unrouted count fall pass over pass without the cleanup stage?
# Cleanup costs 8.1s per net and recovers 1.4%, so if the pass loop's tile
# congestion history converges on its own the stage is deletable. Skip mode
# also rips up partial trees at the gate: they are preserved only so cleanup
# A* can extend them.
# 5400s buys pass 0 (~950s) plus four or five later passes, which route only
# the retry set and should be shorter.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_SKIP_CLEANUP=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== skip-cleanup run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_skip_cleanup.log 2>&1
echo "skip cleanup exit=$? $(date -Is)"

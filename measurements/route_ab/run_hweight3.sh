#!/bin/bash
# Option A: tighten the A* heuristic. Identical environment to
# run_cap30k_full.sh (the 41560-unrouted baseline); the only change is
# H_WEIGHT 2 -> 3 in raster.rs, baked into the snapshotted binary /tmp/rap_hw3.
# Decision rule, stated before results: success is capped sinks well below
# 83715 and pops-per-success below 3217 WITHOUT the partial count rising above
# 18511 or routed WL degrading. Converted capped sinks also test the dossier's
# unproven claim that capped sinks have no path; drained=0 never proved it.
# Runs concurrently with run_heap_cap30k.sh, so wall time is contaminated by
# shared memory bandwidth; compare the deterministic counters, re-time solo.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_hw3
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== hweight3 run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_hweight3.log 2>&1
echo "hweight3 exit=$? $(date -Is)"

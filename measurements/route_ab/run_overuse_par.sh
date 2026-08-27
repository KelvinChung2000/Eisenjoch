#!/bin/bash
# Eighth over-use measurement: parallel planning (NPNR_ROUTE_PAR=1).
# Three endgame schedules all landed 14.4k end-to-end (289f0fc): the
# quality is set by pass count at mature prices, so passes must get
# cheap. Plans never bind ctx in over-use mode, so nets plan in
# parallel against occupancy frozen at pass start; staleness shows up
# as extra over-use per pass, which more (now cheap) passes absorb.
# NEG_WALL stays 1800 -- the point is how many passes now fit in it.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_par
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=1800
export NPNR_ROUTE_PAR=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse par run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_overuse_par.log 2>&1
echo "overuse par exit=$? $(date -Is)"

#!/bin/bash
# Fifth over-use measurement: price-aware heuristic (h_scale = 1 +
# 0.5*present) plus a 1800s negotiation wall that exits, commits
# clean-nets-first and hands the rest to the exclusive finisher. The
# decay run (d905b71) proved both axes converge but pass walls grew
# 92s->1950s as prices outran the manhattan h; this run measures whether
# scaled h holds walls flat, and its commit+finisher produce the first
# end-to-end unrouted count comparable with the exclusive 17010.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse5
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=1800

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse hscale run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_overuse_hscale.log 2>&1
echo "overuse hscale exit=$? $(date -Is)"

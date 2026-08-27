#!/bin/bash
# Second over-use measurement: per-sink escalation (commit pending). v0
# (4e14edb) stranded 52k sinks behind the 30k cap and margin-8 bbox as
# prices pushed cheapest paths past the manhattan heuristic; missing only
# reversed at pass 6. Now a sink that missed retries with cap<<misses (max
# 8x) inside max(60,3*span). Timeout 5400 to observe convergence; wall
# optimization comes after the trajectory is understood. Otherwise
# byte-identical to run_overuse.sh.
#
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse2
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_overuse_esc.log 2>&1
echo "overuse exit=$? $(date -Is)"

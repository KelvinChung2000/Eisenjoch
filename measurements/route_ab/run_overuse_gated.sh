#!/bin/bash
# Third over-use measurement: endgame-GATED per-sink escalation. Ungated
# (7d4ee4b) retried 40k escalated sinks every mid-game pass, 2.5x wall for
# 20% missing gain. Now escalation waits for overused_wires < 1.5M (env
# NPNR_ROUTE_ESC_GATE), so the mid-game stays at v0 cost and expensive
# retries land where the pass-6 reversal proved they succeed. Timeout 7200
# to observe full convergence; wall optimization after.
#
#
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse3
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
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 7200 $BIN > $OUT/fpga01_overuse_gated.log 2>&1
echo "overuse exit=$? $(date -Is)"

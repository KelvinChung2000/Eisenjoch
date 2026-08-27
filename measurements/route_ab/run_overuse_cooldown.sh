#!/bin/bash
# Sixth over-use measurement: escalation cooldown before commit. The
# 8881045 run hit 14380 end-to-end unrouted (first beat of 17010) but
# committed right after an escalated pass reinflated over-use to 1.57M,
# costing 33430 conflicts. Escalation now yields to cheap deflation
# passes when under 2.5 pass-times of wall remain, so commit lands on a
# settled state. Config otherwise identical to run_overuse_hscale.sh.
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse6
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
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_overuse_cooldown.log 2>&1
echo "overuse hscale exit=$? $(date -Is)"

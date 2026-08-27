#!/bin/bash
# Tenth over-use measurement: escalation permanently on. par2 limit-cycled:
# the 1.5M gate flipped esc_on with each escalated pass's readmission and
# every off-phase re-stranded the recovered sinks (missing peaks 51.6k ->
# 55.3k -> 60k). At 25-55s per pass the gate rationale (1500s sequential
# escalated passes) is void; ESC_GATE=1e9 keeps escalation always on.
#
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_par2
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=1800
export NPNR_ROUTE_PAR=1
export NPNR_ROUTE_PRESENT_MAX=8
export NPNR_ROUTE_PAR_FRAC=0.5
export NPNR_ROUTE_ESC_GATE=1000000000

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse par run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_overuse_par3.log 2>&1
echo "overuse par exit=$? $(date -Is)"

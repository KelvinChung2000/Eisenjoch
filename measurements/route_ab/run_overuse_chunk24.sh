#!/bin/bash
# Eleventh over-use measurement: history decay (0.9) + h_scale coupled to
# the measured mean used-wire penalty + cooldown disabled in par mode.
# par3 drifted: history grew unboundedly while h_scale froze at the
# present cap, pass cost compounding 25->115s. Now the heuristic tracks
# the real price level and chronic history forgets, VPR-style.
#
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_par6
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
export NPNR_ROUTE_PAR_FRAC=1.0
export NPNR_ROUTE_PRESENT_MAX=24

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse par run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_overuse_chunk24.log 2>&1
echo "overuse par exit=$? $(date -Is)"

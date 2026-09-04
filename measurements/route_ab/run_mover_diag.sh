#!/bin/bash
# Twelfth over-use measurement: instrument the single-mover fixed point.
# Identical live configuration to run_single_mover.sh (that script exported
# PRESENT_MAX and PAR_FRAC twice; only the second value took effect, so the
# duplicates are dropped here). Nothing about the search changes -- this run
# only adds the per-pass diag line, so its trajectory must reproduce the
# 13803 run's pass lines to be a valid instrument.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_diag
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=900
export NPNR_ROUTE_SINGLE_MOVER=1
export NPNR_ROUTE_ESC_GATE=1000000000
export NPNR_ROUTE_PAR=1
export NPNR_ROUTE_PAR_FRAC=1.0
export NPNR_ROUTE_PRESENT_MAX=3
export NPNR_ROUTE_HIST_GAIN=0.25
export NPNR_ROUTE_PRESENT_GROWTH=1.2
export NPNR_ROUTE_H_MAX=2

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== mover diag run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 2400 $BIN > $OUT/fpga01_mover_diag.log 2>&1
echo "mover diag exit=$? $(date -Is)"

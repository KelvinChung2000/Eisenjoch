#!/bin/bash
# The decisive placement experiment. Every router configuration in this
# campaign (seven of them, three cost models) plateaus at ~2.5M shared
# wires / 14.4-14.6k unrouted, and all of them routed the same parity
# placement -- which runs SKIP_PIPES=1 with ANALYTIC_UNTIL=999, i.e. no
# Dijkstra and no BPR, and BPR on pipe R_eff is opt_trans's density
# penalty. That placement is congestion-blind (solves=0, bins 2.0x,
# excess=98.8), so the residual may be demand the placer never spread.
#
# Here BPR is active: pipes enabled, alpha 10 (default 0.05 leaves BPR
# overwhelmed by HPWL), 20 outer iters, analytic warm start for 5. The
# router is the chunked configuration with recalibrated prices, so the
# only variable against fpga01_price_calib.log is the placement.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_bbox
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=1800
export NPNR_ROUTE_PAR=1
export NPNR_ROUTE_PAR_FRAC=1.0
export NPNR_ROUTE_PRESENT_MAX=3
export NPNR_ROUTE_HIST_GAIN=0.25
export NPNR_ROUTE_PRESENT_GROWTH=1.2
export NPNR_ROUTE_H_MAX=2

# Congestion-aware placement: the parity env minus SKIP_PIPES, with BPR
# switched on and given the alpha/iters balance FPGA01 needs.
OT_ENV="NPNR_OT_MAX_ITERS=20 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_BPR_ALPHA=10 NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=5 \
NPNR_OT_CACHE_NETMAP=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== bpr place run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 10800 $BIN > $OUT/fpga01_bpr_place.log 2>&1
echo "bpr place exit=$? $(date -Is)"

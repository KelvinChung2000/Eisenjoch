#!/bin/bash
# What is the cleanup rate? Pass 0 takes ~950s under the 30k cap, so 1800s
# leaves ~850s of observed cleanup, enough for the rate without spending
# another 90 minutes to learn it. No cleanup cap here: this is the baseline
# the cap will be measured against.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== cleanup rate run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 1800 $BIN > $OUT/fpga01_cleanup_rate.log 2>&1
echo "cleanup rate exit=$? $(date -Is)"

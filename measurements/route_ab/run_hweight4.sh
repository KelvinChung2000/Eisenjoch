#!/bin/bash
# Second point on the H_WEIGHT sweep: 4 is ~1.43x weighted against the
# measured 2.789 pips/tile, where 3 (-31% failures, commit e1cf0bd) was
# ~1.08x. Environment byte-identical to run_cap30k_full.sh; binary
# snapshotted to /tmp/rap_hw4. Runs solo, so its wall time is the clean
# figure the contaminated H=3 run could not give. Decision rule: keep the
# larger weight only if failures and partials keep falling; a pop win that
# raises either is the longer-paths cost of inadmissibility showing up.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_hw4
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== hweight4 run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 1800 $BIN > $OUT/fpga01_hweight4.log 2>&1
echo "hweight4 exit=$? $(date -Is)"

#!/bin/bash
# Fourth point on the H_WEIGHT sweep: 6 is ~2.15x weighted. 5 flattened:
# unrouted 19152 -> 17010 with pops/success ticking up 2042 -> 2097, so
# this probe locates the knee. Environment byte-identical again; binary
# snapshotted to /tmp/rap_hw6. Runs solo. Decision rule unchanged: keep
# the larger weight only if failures and partials keep falling; a pop win
# that raises either is the longer-paths cost of inadmissibility. Timeout
# 900 covers the ~500s pass 0; cleanup is measured twice already.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_hw6
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== hweight6 run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 900 $BIN > $OUT/fpga01_hweight6.log 2>&1
echo "hweight6 exit=$? $(date -Is)"

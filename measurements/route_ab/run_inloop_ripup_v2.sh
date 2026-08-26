#!/bin/bash
# v2 of in-loop scoped rip-up. v1 recovered 16 of 17010 in 2059s:
# 87% failed the tight explore and 99% of rips failed the all-or-nothing
# reroute. v2 extends the existing tree per sink, escalates a DRAINED tight
# explore to the cleanup-proven max(60,3*span) margin, caches drained-wide
# (net,sink) pairs as permanently dead, orders attempts span-ascending and
# caps each sweep at 600s wall. Baseline (commit 319d62e): 62/1/1 unaided.
# Runs solo.
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_inloop2
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_INLOOP_RIPUP=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== inloop ripup run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_inloop_ripup_v2.log 2>&1
echo "inloop ripup exit=$? $(date -Is)"

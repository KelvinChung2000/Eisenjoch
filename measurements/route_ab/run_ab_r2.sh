#!/bin/bash
# FPGA01 route A/B, second run. Identical placement environment to run_ab.sh --
# only the router's progress instrumentation changed -- so r2 stays comparable
# to r1. Fresh log names: r1's logs are committed evidence and are not clobbered.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
CHIPDB=chip_database/xc7_large.bin
DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=$CHIPDB
export NPNR_ROUTE_DESIGN=$DESIGN
export NPNR_ROUTE_ITERS=50

# The twelve-run parity configuration, byte-identical to r1.
OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== opt_trans arm started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 5400 $BIN > $OUT/fpga01_ot_r2.log 2>&1
echo "opt_trans exit=$? $(date -Is)"

echo "=== heap arm started $(date -Is) ==="
NPNR_ROUTE_PLACER=heap timeout 5400 $BIN > $OUT/fpga01_heap_r2.log 2>&1
echo "heap exit=$? $(date -Is)"

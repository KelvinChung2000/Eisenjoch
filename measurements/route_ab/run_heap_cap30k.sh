#!/bin/bash
# Option F: is the 40% pass-0 failure placement density or router weakness?
# HeAP placement through the identical 30k-cap router configuration as
# run_cap30k_full.sh, whose opt_trans arm is the 41560-unrouted baseline.
# Runs from a snapshotted binary so a concurrent rebuild cannot touch it.
# Decision rule, stated before results: unrouted (empty+partial) near 40%
# blames the router's exclusivity model; well under 15% blames placement.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_master_hw2
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000

echo "=== heap cap30k run started $(date -Is) ==="
NPNR_ROUTE_PLACER=heap timeout 3600 $BIN > $OUT/fpga01_heap_cap30k.log 2>&1
echo "heap cap30k exit=$? $(date -Is)"

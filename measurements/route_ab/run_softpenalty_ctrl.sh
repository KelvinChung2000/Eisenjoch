#!/bin/bash
# Does excluding foreign-bound wires pay off at FPGA01 scale? sv3 showed
# -16% pops per reached sink on 863 sinks; FPGA01 pass 0 has 177852 reached
# and 83703 capped, so this is where the sign of the change is decided.
# Skip-cleanup so the run reaches pass 1 inside the budget; every other
# setting matches run_skip_cleanup.sh, whose pass 0 read routed=82069
# failed=23049 at 1071s, reached pops/sink=3217, limited=83703.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_SKIP_CLEANUP=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== soft-penalty control run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 2200 $BIN > $OUT/fpga01_softpenalty_ctrl.log 2>&1
echo "control exit=$? $(date -Is)"

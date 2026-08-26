#!/bin/bash
# Baseline for in-loop rip-up: what the pass loop recovers unaided at the
# H_WEIGHT=5 knee (commit b8ab133). Same instrument as run_skip_cleanup.sh,
# whose H=2 reading was 60/32/13 nets per pass at 1250/1351/1459s — the
# yield-halving that motivated moving scoped rip-up into the loop. The
# residual population is now 17010 not 41560, so both the yield and the
# per-pass wall change; whatever this measures is the number the in-loop
# rip-up has to beat. Binary snapshotted to /tmp/rap_h5_base so rebuilds
# during the rip-up work cannot touch the running probe.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_h5_base
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

echo "=== multipass h5 baseline started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_multipass_h5.log 2>&1
echo "multipass h5 exit=$? $(date -Is)"

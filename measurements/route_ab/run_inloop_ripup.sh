#!/bin/bash
# First measurement of in-loop scoped rip-up (NPNR_ROUTE_INLOOP_RIPUP=1):
# every failed net gets one bounded rip-up attempt per pass (explore 200k
# pops in the routing bbox, <=8 blockers, victim fanout <=64, 2 rips per
# victim per sweep, full rollback on failure), and the cleanup stage keeps
# only its const/clock fast paths. H_WEIGHT=5 binary, environment otherwise
# byte-identical to run_multipass_h5.sh, whose unaided pass-loop recovery is
# the number this has to beat. Runs beside that baseline, so compare the
# recovered counters, not wall.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_inloop
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
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_inloop_ripup.log 2>&1
echo "inloop ripup exit=$? $(date -Is)"

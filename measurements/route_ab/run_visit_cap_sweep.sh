#!/bin/bash
# Does capping the per-sink A* budget buy throughput without shedding sinks?
# Two caps against the uncapped probe already in fpga01_astar_probe.log, same
# placement and the same 900s, so reached-sinks/sec compares directly.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=./target/release/examples/route_after_place
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

for CAP in 30000 100000; do
  echo "=== cap=$CAP started $(date -Is) ==="
  env $OT_ENV NPNR_ROUTE_PLACER=opt_trans NPNR_ROUTE_SINK_VISITS=$CAP \
    timeout 900 $BIN > $OUT/fpga01_cap${CAP}.log 2>&1
  echo "cap=$CAP exit=$? $(date -Is)"
done

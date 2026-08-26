#!/bin/bash
# First measurement of PathFinder over-use negotiation
# (NPNR_ROUTE_OVERUSE=1): wires may be shared within a pass at price
# present*occ + history, plans live in router state, and only converged
# plans commit; const/clock and stragglers finish on the exclusive path.
# H_WEIGHT=5, default pricing (present 0.5, growth 1.4, hist 1.0),
# environment otherwise byte-identical to the exclusive baselines. The
# numbers to beat: 17010 unrouted / 469s pass 0 (b8ab133), recovery
# 62/1/1 (319d62e). Watch overused_wires and missing_sinks per pass:
# converging over-use with missing_sinks near zero is the whole bet.
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== overuse run started $(date -Is) ==="
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 3600 $BIN > $OUT/fpga01_overuse.log 2>&1
echo "overuse exit=$? $(date -Is)"

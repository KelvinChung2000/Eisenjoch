#!/bin/bash
# Parameterised over-use run. Usage:
#   run_overuse_cfg.sh <name> <binary> [NPNR_ROUTE_X=v ...]
# The baseline is the single-mover configuration that produced 13803; each
# experiment is the overrides on the command line, echoed into the log header
# so the log alone records what produced it.
set -u
NAME=$1; shift
BIN=$1; shift
cd /home/kelvin/side-project/eisenjoch
OUT=measurements/route_ab

export NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin
export NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json
export NPNR_ROUTE_ITERS=50
export NPNR_ROUTE_SINK_VISITS=30000
export NPNR_ROUTE_OVERUSE=1
export NPNR_ROUTE_NEG_WALL=900
export NPNR_ROUTE_SINGLE_MOVER=1
export NPNR_ROUTE_ESC_GATE=1000000000
export NPNR_ROUTE_PAR=1
export NPNR_ROUTE_PAR_FRAC=1.0
export NPNR_ROUTE_PRESENT_MAX=3
export NPNR_ROUTE_HIST_GAIN=0.25
export NPNR_ROUTE_PRESENT_GROWTH=1.2
export NPNR_ROUTE_H_MAX=2
for kv in "$@"; do export "$kv"; done

OT_ENV="NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=24 \
NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=16384 NPNR_OT_BPR_CAP=1.2 \
NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 NPNR_OT_CACHE_NETMAP=1 \
NPNR_OT_SKIP_PIPES=1 NPNR_OT_CACHE_NETINFO=1 NPNR_OT_DCD_ITERS=2"

echo "=== $NAME started $(date -Is) overrides: $* ==="
{ echo "CONFIG $NAME overrides: $*"; env | grep '^NPNR_ROUTE_' | sort; } > $OUT/fpga01_$NAME.log
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout ${TMO:-1800} $BIN >> $OUT/fpga01_$NAME.log 2>&1
echo "$NAME exit=$? $(date -Is)"

# The metric is the "A* cleanup for N failed nets" line, which lands around
# 1400s; NPNR_ROUTE_SKIP_CLEANUP is not a cheaper way to reach it because it
# also rips up partial trees at the in-loop gate (raster.rs:2982), changing
# the configuration rather than only where it stops.

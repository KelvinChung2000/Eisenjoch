#!/bin/bash
# Fourth over-use measurement: gated escalation + miss-level DECAY on
# recovery (f13b95c). The gated run sawtoothed: recovered sinks lost their
# escalated budget and re-stranded (26415->45325 at pass 5). Now recovery
# decays the level by one instead of clearing, so budget persists while
# contention settles. Timeout 10800 to observe full convergence and the
# commit + finisher phases.
#
#
#
set -u
cd /home/kelvin/side-project/eisenjoch
BIN=/tmp/rap_overuse4
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
env $OT_ENV NPNR_ROUTE_PLACER=opt_trans timeout 10800 $BIN > $OUT/fpga01_overuse_decay.log 2>&1
echo "overuse exit=$? $(date -Is)"

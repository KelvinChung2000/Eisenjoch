#!/usr/bin/env bash
# Is 0.998x HPWL parity on the right metric?
#
# The parity configuration caps BPR at 1.2 and runs the pure analytic field, so
# it reports cong_share=0.0% and solves=0: congestion has left the objective
# entirely. HeAP has no congestion term either, so the fair question is not
# whether our placement is routable in the absolute but whether it is worse
# than the placement we claim parity with. Same metric, same binary, both sides.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
CHIPDB=$ROOT/chip_database/xc7_large.bin
DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

LOG=$OUT/fpga01_route_ot.log
{ echo "launched at : $(date -Is)"; echo "arm: opt_trans parity placement, global routability"
  echo "git HEAD    : $HEAD_SHA"; } > "$LOG.meta"
env NPNR_OT_TRACE_CHIPDB=$CHIPDB NPNR_OT_TRACE_DESIGN=$DESIGN \
    NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
    NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 NPNR_OT_BPR_CAP=1.2 \
    NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 \
    NPNR_OT_CACHE_NETMAP=1 NPNR_OT_SKIP_PIPES=1 \
    NPNR_OT_CHECK_ROUTABILITY=1 \
    $ROOT/target/release/examples/ot_trace_design > "$LOG" 2>&1
echo "exit code: $?" >> "$LOG.meta"

LOG=$OUT/fpga01_route_heap.log
{ echo "launched at : $(date -Is)"; echo "arm: HeAP placement, global routability"
  echo "git HEAD    : $HEAD_SHA"; } > "$LOG.meta"
env NPNR_HEAP_TRACE_CHIPDB=$CHIPDB NPNR_HEAP_TRACE_DESIGN=$DESIGN \
    NPNR_OT_CHECK_ROUTABILITY=1 \
    $ROOT/target/release/examples/heap_trace_design > "$LOG" 2>&1
echo "exit code: $?" >> "$LOG.meta"

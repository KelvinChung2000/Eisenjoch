#!/usr/bin/env bash
# Acceptance test for the ring-search snap in `snap_and_bind_cells_to_bels`.
#
# The old scan was O(cells x pool): ~70k cells against a 0.5-1.1M BEL pool per
# bucket, which is the 51s `prepare_discrete` both placers pay. The ring scan
# has to return the same BEL for every cell, so the test is not a speedup
# figure but two exact numbers: 4 523 586 for opt_trans and 4 533 609 for HeAP.
# Both placers are re-measured on the same binary because the old 51.9s HeAP
# reference is void once the shared step changes.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)
CHIPDB=$ROOT/chip_database/xc7_large.bin
DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json

meta() {             # log_path arm_desc expected_hpwl
  { echo "launched at : $(date -Is)"
    echo "arm         : $2"
    echo "git HEAD    : $HEAD_SHA"
    echo "expected    : HPWL $3, exact"
  } > "$1.meta"
}

LOG=$OUT/fpga01_ring_ot.log
meta "$LOG" "opt_trans parity arm, ring-search snap" 4523586
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$CHIPDB NPNR_OT_TRACE_DESIGN=$DESIGN \
      NPNR_OT_MAX_ITERS=11 NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 NPNR_OT_BPR_CAP=1.2 \
      NPNR_OT_SWEEP=jacobi_bisect NPNR_OT_ANALYTIC_UNTIL=999 \
      NPNR_OT_CACHE_NETMAP=1 NPNR_OT_SKIP_PIPES=1 \
      $ROOT/target/release/examples/ot_trace_design > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

LOG=$OUT/fpga01_ring_heap.log
meta "$LOG" "HeAP reference, ring-search snap" 4533609
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_HEAP_TRACE_CHIPDB=$CHIPDB NPNR_HEAP_TRACE_DESIGN=$DESIGN \
      $ROOT/target/release/examples/heap_trace_design > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

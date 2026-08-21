#!/usr/bin/env bash
# Skip the Dijkstra for the ten iterations the analytic field already wins,
# then hand over to the exact field for the ten that need congestion.
# Control: cap=1.2, 20 iters -> HPWL 5 750 533 in 1335 s.
set -u
ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
UNTIL=${UNTIL:-10}
ARM=det_chunk4096_steiner1_cap1.2_until${UNTIL}_20iter
LOG=$OUT/fpga01_${ARM}.log
{
  echo "launched at : $(date -Is)"
  echo "arm         : $ARM (cap=1.2, ANALYTIC_UNTIL=$UNTIL, 20 iters)"
  echo "git HEAD    : $(git -C "$ROOT" rev-parse HEAD)"
  echo "control     : cap=1.2 20 iters -> HPWL 5750533 in 1335s"
} > "$LOG.meta"
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
      NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
      NPNR_OT_MAX_ITERS=20 NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 \
      NPNR_OT_BPR_CAP=1.2 NPNR_OT_ANALYTIC_UNTIL=$UNTIL \
      "$BIN" > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

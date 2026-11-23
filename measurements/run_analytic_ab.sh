#!/usr/bin/env bash
# Is k*manhattan good enough for the sweep to descend on? Control is the
# recorded cap=1.2 20-iter arm: 5 750 533 in 1335 s. Refresh still runs here,
# so this isolates quality; the time saving only arrives if the Dijkstra can
# then be dropped.
set -u
ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
ARM=det_chunk4096_steiner1_cap1.2_analytic_20iter
LOG=$OUT/fpga01_${ARM}.log
{
  echo "launched at : $(date -Is)"
  echo "arm         : $ARM (steiner=1, threads=32, DET_SWEEP=1, chunk=4096, cap=1.2, ANALYTIC_DIST=1)"
  echo "git HEAD    : $(git -C "$ROOT" rev-parse HEAD)"
  echo "control     : cap=1.2 20 iters -> HPWL 5750533 in 1335s"
} > "$LOG.meta"
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
      NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
      NPNR_OT_MAX_ITERS=20 NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 \
      NPNR_OT_BPR_CAP=1.2 NPNR_OT_ANALYTIC_DIST=1 \
      "$BIN" > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

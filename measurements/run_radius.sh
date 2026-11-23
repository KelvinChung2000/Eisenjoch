#!/usr/bin/env bash
set -u
ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
ITERS=${ITERS:?}; RAD=${RAD:?}
ARM=det_chunk4096_steiner1_cap1.2_pure_${ITERS}iter_rad${RAD}
LOG=$OUT/fpga01_${ARM}.log
{ echo "launched at : $(date -Is)"; echo "arm : $ARM"; echo "git HEAD : $(git -C "$ROOT" rev-parse HEAD)"; } > "$LOG.meta"
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
      NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
      NPNR_OT_MAX_ITERS=$ITERS NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 \
      NPNR_OT_BPR_CAP=1.2 NPNR_OT_ANALYTIC_UNTIL=$ITERS NPNR_OT_ANALYTIC_RADIUS=$RAD \
      "$BIN" > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

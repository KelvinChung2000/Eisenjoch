#!/usr/bin/env bash
# Re-tune NPNR_OT_DET_CHUNK on steiner=0. 4096 was tuned on steiner=1 and
# generalised at +0.91%; this brackets it. Deterministic, so n=1 per point.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

for CHUNK in 2048 8192; do
  ARM="det_chunk${CHUNK}_steiner0_20iter"
  LOG=$OUT/fpga01_${ARM}.log
  META=$LOG.meta
  {
    echo "launched at : $(date -Is)"
    echo "arm         : $ARM (steiner=0, threads=32, DET_SWEEP=1, chunk=$CHUNK)"
    echo "git HEAD    : $HEAD_SHA"
  } > "$META"

  START=$(date +%s)
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
    env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
        NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
        NPNR_OT_MAX_ITERS=20 \
        NPNR_OT_STEINER=0 \
        NPNR_OT_THREADS=32 \
        NPNR_OT_DET_SWEEP=1 \
        NPNR_OT_DET_CHUNK=$CHUNK \
        "$BIN" > "$LOG" 2>&1
  RC=$?
  END=$(date +%s)

  {
    echo "exit code: $RC"
    echo "wall_secs: $((END - START))"
  } >> "$META"
done

#!/usr/bin/env bash
# Does bounding the BPR multiplier put wirelength back in the objective?
# At cap=inf the placer measures cong_share=100.0%, so base wirelength is
# invisible to the descent. Reference arm is the recorded steiner=1 det
# chunk=4096 run: HPWL 7 387 808 in 1465 s.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

for CAP in "$@"; do
  ARM="det_chunk4096_steiner1_bprcap${CAP}_20iter"
  LOG=$OUT/fpga01_${ARM}.log
  META=$LOG.meta
  {
    echo "launched at : $(date -Is)"
    echo "arm         : $ARM (steiner=1, threads=32, DET_SWEEP=1, chunk=4096, BPR_CAP=$CAP)"
    echo "git HEAD    : $HEAD_SHA"
    echo "reference   : cap=inf -> HPWL 7387808 in 1465s"
  } > "$META"

  START=$(date +%s)
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
    env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
        NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
        NPNR_OT_MAX_ITERS=20 \
        NPNR_OT_STEINER=1 \
        NPNR_OT_THREADS=32 \
        NPNR_OT_DET_SWEEP=1 \
        NPNR_OT_DET_CHUNK=4096 \
        NPNR_OT_BPR_CAP=$CAP \
        "$BIN" > "$LOG" 2>&1
  RC=$?
  END=$(date +%s)

  {
    echo "exit code: $RC"
    echo "wall_secs: $((END - START))"
  } >> "$META"
done

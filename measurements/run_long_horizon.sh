#!/usr/bin/env bash
# Does the objective reach HeAP's 4 533 609 if simply given more iterations?
# At 20 iterations `line` is still falling 2.6% per iteration, so 20 is where
# the run stops rather than where it converges. CAP and ITERS come from the
# environment so the winning cap can be carried in without editing this file.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

CAP=${CAP:?set CAP to a number or the string inf}
ITERS=${ITERS:-40}

ARM="det_chunk4096_steiner1_cap${CAP}_${ITERS}iter"
LOG=$OUT/fpga01_${ARM}.log
META=$LOG.meta
{
  echo "launched at : $(date -Is)"
  echo "arm         : $ARM (steiner=1, threads=32, DET_SWEEP=1, chunk=4096)"
  echo "git HEAD    : $HEAD_SHA"
  echo "question    : does line reach ~6.0-6.4M, i.e. HeAP HPWL 4533609"
} > "$META"

CAP_ENV=()
[ "$CAP" = "inf" ] || CAP_ENV=(NPNR_OT_BPR_CAP=$CAP)

START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
      NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
      NPNR_OT_MAX_ITERS=$ITERS \
      NPNR_OT_STEINER=1 \
      NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 \
      NPNR_OT_DET_CHUNK=4096 \
      "${CAP_ENV[@]}" \
      "$BIN" > "$LOG" 2>&1
RC=$?
END=$(date +%s)

{
  echo "exit code: $RC"
  echo "wall_secs: $((END - START))"
} >> "$META"

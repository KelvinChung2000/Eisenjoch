#!/usr/bin/env bash
# The DCD sweep is 15.5s of each 18s iteration now that refresh is 2.5s, so the
# probe budget is where the remaining wall time is. Quality margin is thin --
# 4 515 026 against HeAP's 4 533 609 -- so trade probes for iterations, not for
# quality.
set -u
ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
ITERS=${ITERS:?}; DCD=${DCD:?}
ARM=det_chunk4096_steiner1_cap1.2_pure_${ITERS}iter_dcd${DCD}
LOG=$OUT/fpga01_${ARM}.log
{
  echo "launched at : $(date -Is)"
  echo "arm         : $ARM (cap=1.2, pure analytic, $ITERS iters, dcd_iters=$DCD)"
  echo "git HEAD    : $(git -C "$ROOT" rev-parse HEAD)"
  echo "target      : HeAP 4 533 609 in 52.8s place-only"
} > "$LOG.meta"
START=$(date +%s)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
  env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
      NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
      NPNR_OT_MAX_ITERS=$ITERS NPNR_OT_STEINER=1 NPNR_OT_THREADS=32 \
      NPNR_OT_DET_SWEEP=1 NPNR_OT_DET_CHUNK=4096 \
      NPNR_OT_BPR_CAP=1.2 NPNR_OT_ANALYTIC_UNTIL=$ITERS NPNR_OT_DCD_ITERS=$DCD \
      "$BIN" > "$LOG" 2>&1
RC=$?; END=$(date +%s)
{ echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$LOG.meta"

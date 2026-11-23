#!/usr/bin/env bash
# cap=2 beat cap=10 on both axes (5 809 246 in 1386 s against 6 456 365 in
# 1562 s), so the sweep has not found the knee yet. Push tighter, then ask the
# question the flatter field makes answerable: at a bounded price, does the
# monotone walk match the Dijkstra?
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design

while pgrep -f run_bpr_cap_sweep.sh > /dev/null; do sleep 30; done

"$ROOT/measurements/run_bpr_cap_sweep.sh" 3 1.5 1.2

cd "$ROOT" || exit 1
cargo build --release --example ot_trace_design 2>&1 | tail -5
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BUILD FAILED"; exit 1; }
strings "$BIN" | grep -q SearchValue_check || { echo "diag absent from binary"; exit 1; }

HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

for CAP in inf 2; do
  ARM="searchvalue_cap${CAP}_8iter"
  LOG=$OUT/fpga01_${ARM}.log
  META=$LOG.meta
  {
    echo "launched at : $(date -Is)"
    echo "arm         : $ARM (steiner=1, threads=32, DET_SWEEP=1, chunk=4096, 8 iters)"
    echo "git HEAD    : $HEAD_SHA"
    echo "question    : does straight/star return to ~1.0 once the price is bounded"
  } > "$META"

  CAP_ENV=()
  [ "$CAP" = "inf" ] || CAP_ENV=(NPNR_OT_BPR_CAP=$CAP)

  START=$(date +%s)
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
    env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
        NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
        NPNR_OT_MAX_ITERS=8 \
        NPNR_OT_STEINER=1 \
        NPNR_OT_THREADS=32 \
        NPNR_OT_DET_SWEEP=1 \
        NPNR_OT_DET_CHUNK=4096 \
        NPNR_OT_HPWL_CHECK=1 \
        "${CAP_ENV[@]}" \
        "$BIN" > "$LOG" 2>&1
  RC=$?
  END=$(date +%s)

  {
    echo "exit code: $RC"
    echo "wall_secs: $((END - START))"
  } >> "$META"
done

#!/usr/bin/env bash
# Goal queue. One build up front so every arm below runs the same binary.
#
#   1. finish the cap sweep tighter: 3, 1.5, 1.2
#   2. long horizon at whichever cap measured best -- cap=2 converges at 4%
#      per iteration and is still falling 3.67% at iter 19, which projects
#      HeAP's HPWL at roughly 28 iterations
#   3. straight/star at the best cap against uncapped: the only lever left
#      that touches wall time
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design

while pgrep -f run_bpr_cap_sweep.sh > /dev/null; do sleep 30; done

cd "$ROOT" || exit 1
cargo build --release --example ot_trace_design 2>&1 | tail -5
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BUILD FAILED"; exit 1; }
strings "$BIN" | grep -q SearchValue_check || { echo "diag absent from binary"; exit 1; }
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

run_arm() {          # arm_name cap iters extra_env...
  local ARM=$1 CAP=$2 ITERS=$3; shift 3
  local LOG=$OUT/fpga01_${ARM}.log META=$OUT/fpga01_${ARM}.log.meta
  {
    echo "launched at : $(date -Is)"
    echo "arm         : $ARM (steiner=1, threads=32, DET_SWEEP=1, chunk=4096, cap=$CAP, $ITERS iters)"
    echo "git HEAD    : $HEAD_SHA"
  } > "$META"
  local CAP_ENV=()
  [ "$CAP" = "inf" ] || CAP_ENV=(NPNR_OT_BPR_CAP=$CAP)
  local START=$(date +%s)
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
    env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
        NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
        NPNR_OT_MAX_ITERS=$ITERS \
        NPNR_OT_STEINER=1 \
        NPNR_OT_THREADS=32 \
        NPNR_OT_DET_SWEEP=1 \
        NPNR_OT_DET_CHUNK=4096 \
        "${CAP_ENV[@]}" "$@" \
        "$BIN" > "$LOG" 2>&1
  local RC=$? END=$(date +%s)
  { echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$META"
}

for CAP in 3 1.5 1.2; do
  run_arm "det_chunk4096_steiner1_bprcap${CAP}_20iter" "$CAP" 20
done

# Best cap by post-legalisation HPWL across every 20-iteration arm measured.
BEST_CAP=$(for f in $OUT/fpga01_det_chunk4096_steiner1_bprcap*_20iter.log; do
  H=$(sed -n 's/.*Post-legalization: HPWL=\([0-9]*\).*/\1/p' "$f" | tail -1)
  [ -n "$H" ] || continue
  C=$(basename "$f" | sed 's/.*bprcap\([0-9.]*\)_20iter.log/\1/')
  echo "$H $C"
done | sort -n | head -1 | cut -d' ' -f2)
echo "best cap by HPWL: $BEST_CAP"
[ -n "$BEST_CAP" ] || { echo "no cap arm parsed"; exit 1; }

run_arm "det_chunk4096_steiner1_cap${BEST_CAP}_32iter" "$BEST_CAP" 32

for CAP in inf "$BEST_CAP"; do
  run_arm "searchvalue_cap${CAP}_8iter" "$CAP" 8 NPNR_OT_HPWL_CHECK=1
done

#!/usr/bin/env bash
# What is left of our own work above the shared prepare_discrete floor?
#
# The parity arm spends 60.5s of which 51.2s is prepare_discrete, shared with
# HeAP. The remaining 9.3s is ours against HeAP's 1.2s, and it splits into
# network build 2.1s, iteration 0 setup 2.4s, ten steady sweeps 1.8s and
# legalisation 2.9s. Each arm below removes one of those and is checked
# against the baseline HPWL bit for bit -- deterministic runs make a skipped
# stage provably inert rather than plausibly inert.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
OUT=$ROOT/measurements/mem_probe
BIN=$ROOT/target/release/examples/ot_trace_design
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)

run_arm() {          # arm_name extra_env...
  local ARM=$1; shift
  local LOG=$OUT/fpga01_${ARM}.log META=$OUT/fpga01_${ARM}.log.meta
  {
    echo "launched at : $(date -Is)"
    echo "arm         : $ARM"
    echo "extra env   : $*"
    echo "git HEAD    : $HEAD_SHA"
    echo "baseline    : span 60.5s, HPWL 4523586 (fpga01_pure_11iter_bisect_netmap)"
  } > "$META"
  local START=$(date +%s)
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --collect \
    env NPNR_OT_TRACE_CHIPDB=$ROOT/chip_database/xc7_large.bin \
        NPNR_OT_TRACE_DESIGN=$ROOT/benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
        NPNR_OT_MAX_ITERS=11 \
        NPNR_OT_STEINER=1 \
        NPNR_OT_THREADS=32 \
        NPNR_OT_DET_SWEEP=1 \
        NPNR_OT_DET_CHUNK=4096 \
        NPNR_OT_BPR_CAP=1.2 \
        NPNR_OT_SWEEP=jacobi_bisect \
        NPNR_OT_ANALYTIC_UNTIL=999 \
        NPNR_OT_CACHE_NETMAP=1 \
        "$@" \
        "$BIN" > "$LOG" 2>&1
  local RC=$? END=$(date +%s)
  { echo "exit code: $RC"; echo "wall_secs: $((END - START))"; } >> "$META"
}

run_arm own_base
run_arm own_skippipes NPNR_OT_SKIP_PIPES=1

#!/usr/bin/env bash
# Serialise the two sweeps: nothing may compile or run while the chunk sweep
# is timing itself, or the wall and refresh columns are garbage.
set -u

ROOT=/home/kelvin/side-project/eisenjoch
BIN=$ROOT/target/release/examples/ot_trace_design

while pgrep -f run_chunk_sweep_steiner0.sh > /dev/null; do sleep 30; done

cd "$ROOT" || exit 1
cargo build --release --example ot_trace_design 2>&1 | tail -20
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BUILD FAILED, not launching cap sweep"; exit 1; }

# An old binary ignores NPNR_OT_BPR_CAP and would silently reproduce the
# reference arm, which reads as "the cap does nothing".
strings "$BIN" | grep -q NPNR_OT_BPR_CAP || { echo "cap knob absent from binary"; exit 1; }

exec "$ROOT/measurements/run_bpr_cap_sweep.sh"

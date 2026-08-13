#!/usr/bin/env bash
# Regenerate the shared nextpnr/eisenjoch baseline fixtures.
#
# Produces, from a single synthetic fabric:
#   - the shared chipdb .bin, read by BOTH nextpnr and eisenjoch
#   - a placed+routed design from upstream nextpnr
#   - per-net golden wirelength from nextpnr's own get_net_metric
#
# Requires the pinned upstream nextpnr checkout, built with the example uarch
# and the --dump-net-metric patch in patches/. See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NPNR="${NPNR_UPSTREAM:-/home/kelvin/nextpnr-upstream}"
NPNR_BIN="$NPNR/build/nextpnr-himbaechel"
BBASM="$NPNR/build/bba/bbasm"
EXAMPLE="$NPNR/himbaechel/uarch/example"
FIXTURES="$HERE/../../crates/nextpnr/tests/fixtures/npnr_baseline"

SIZE="${SIZE:-12}"
WIDTH="${WIDTH:-16}"
SEED="${SEED:-1}"
PLACER="${PLACER:-heap}"

for f in "$NPNR_BIN" "$BBASM"; do
    [ -x "$f" ] || { echo "missing $f -- build upstream nextpnr first (see README.md)" >&2; exit 1; }
done

mkdir -p "$HERE/out" "$FIXTURES"

echo "==> chipdb (${SIZE}x${SIZE})"
python3 "$HERE/gen_shared_chipdb.py" "$HERE/out/synth${SIZE}.bba" --size "$SIZE"
"$BBASM" --le "$HERE/out/synth${SIZE}.bba" "$HERE/out/synth${SIZE}.bin"

echo "==> synthesis (W=${WIDTH})"
yosys -q -c "$HERE/designs/synth.tcl" -- \
    "$HERE/designs/synth_bench.v" top "$WIDTH" "$HERE/out/bench${WIDTH}.json"

echo "==> nextpnr place & route (placer=${PLACER}, seed=${SEED})"
"$NPNR_BIN" \
    --chipdb "$HERE/out/synth${SIZE}.bin" \
    --device EXAMPLE \
    --json "$HERE/out/bench${WIDTH}.json" \
    --write "$HERE/out/placed${WIDTH}.json" \
    --dump-net-metric "$HERE/out/golden${WIDTH}.txt" \
    --seed "$SEED" --placer "$PLACER" \
    > "$HERE/out/nextpnr.log" 2>&1
grep -E "net metrics|Program finished" "$HERE/out/nextpnr.log"

echo "==> publishing fixtures"
cp "$HERE/out/synth${SIZE}.bin"    "$FIXTURES/shared_chipdb.bin"
cp "$HERE/out/placed${WIDTH}.json" "$FIXTURES/placed.json"
cp "$HERE/out/golden${WIDTH}.txt"  "$FIXTURES/golden_net_metric.txt"
cp "$EXAMPLE/constids.inc"         "$FIXTURES/constids.inc"
echo "done -> $FIXTURES"

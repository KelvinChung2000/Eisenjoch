#!/bin/bash
# Generate the example arch chipdb binary from the Python generator.
# Requires the C++ nextpnr checkout with bbasm built.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEXTPNR_DIR="/home/kelvin/nextpnr"

echo "Generating BBA..."
PYTHONPATH="$NEXTPNR_DIR/himbaechel" \
  python3 "$SCRIPT_DIR/gen_example_chipdb.py" "$SCRIPT_DIR/example.bba"

echo "Assembling binary..."
"$NEXTPNR_DIR/build/bba/bbasm" --le "$SCRIPT_DIR/example.bba" "$SCRIPT_DIR/example.bin"

rm -f "$SCRIPT_DIR/example.bba"
echo "Done: $SCRIPT_DIR/example.bin ($(wc -c < "$SCRIPT_DIR/example.bin") bytes)"

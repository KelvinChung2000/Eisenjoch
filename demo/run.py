"""Demo: full Yosys -> nextpnr-rust P&R flow on a blinky design."""

import nextpnr
import os

# Resolve paths relative to this script
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

chipdb_path = os.path.join(
    root, "crates", "nextpnr", "tests", "fixtures", "example.bin"
)
json_path = os.path.join(here, "blinky.json")

print("=" * 60)
print("nextpnr-rust demo: blinky on example arch")
print("=" * 60)
print()

# Load context with example chipdb
ctx = nextpnr.Context(chipdb=chipdb_path)
print(f"Chip: {ctx.width}x{ctx.height} grid")

# Load synthesized design
ctx.load_design(json_path)
print(f"Design loaded: {len(ctx.cells)} cells, {len(ctx.nets)} nets")
print()

# Pack
print("--- Packing ---")
ctx.pack()
print(f"After pack: {len(ctx.cells)} cells, {len(ctx.nets)} nets")
print()

# Lock clock IOB to tile (1,0) BEL IO0 — the only tile whose GCLK_OUT
# wire connects to the clock ladder network.
print("--- Pre-placing clock IOB ---")
ctx.place_cell("$io$clk", x=1, y=0, bel_name="IO0")
print("  $io$clk -> IO0 @ (1,0)")
print()

# Place (HeAP)
print("--- Placing (HeAP) ---")
ctx.place(placer="heap", seed=42)
print("Placement complete.")
print()

# Add clock constraint
ctx.add_clock("clk", freq_mhz=100.0)

# Route (Router1)
print("--- Routing (Router1) ---")
ctx.route(router="router1")
print("Routing complete.")
print()

# Timing report
print("--- Timing Report ---")
timing = ctx.timing_report()
print(f"  Fmax:              {timing.fmax:.2f} MHz")
print(f"  Worst slack:       {timing.worst_slack} ps")
print(f"  Failing endpoints: {timing.num_failing}/{timing.num_endpoints}")
print()

print("=" * 60)
print("DEMO COMPLETE")
print("=" * 60)

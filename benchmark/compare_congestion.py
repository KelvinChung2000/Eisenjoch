"""Compare placement quality with and without congestion awareness.

Usage: python compare_congestion.py [--benchmark NAME] [--size NxN] [--placer heap|sa|both]

Runs placement twice on the same design: once with congestion_weight=0.0 (off)
and once with congestion_weight=0.5 (on), then compares HPWL and congestion metrics.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(HERE, "output")


def run_comparison(benchmark, chipdb_path, placer, congestion_weight):
    """Run pack + place with given congestion weight, return metrics."""
    import nextpnr

    ctx = nextpnr.Context(chipdb=chipdb_path)
    json_path = os.path.join(OUTPUT_DIR, f"{benchmark}.json")
    ctx.load_design(json_path)
    ctx.pack()

    width = ctx.width
    height = ctx.height

    # Pre-place IO cells
    all_io = [c for c in ctx.cells if c.startswith("$io$")]
    io_bels = []
    for x in range(1, width - 1):
        for bel in ["IO0", "IO1"]:
            io_bels.append((x, 0, bel))
    for y in range(1, height - 1):
        for bel in ["IO0", "IO1"]:
            io_bels.append((0, y, bel))
            io_bels.append((width - 1, y, bel))
    for x in range(1, width - 1):
        for bel in ["IO0", "IO1"]:
            io_bels.append((x, height - 1, bel))

    bel_idx = 0
    for cell_name in all_io:
        while bel_idx < len(io_bels):
            x, y, bel_name = io_bels[bel_idx]
            bel_idx += 1
            try:
                ctx.place_cell(cell_name, x=x, y=y, bel_name=bel_name)
                break
            except Exception:
                continue

    # Place with given congestion weight
    t0 = time.time()
    ctx.place(placer=placer, seed=42, congestion_weight=congestion_weight)
    place_time = time.time() - t0

    # Compute metrics
    hpwl = ctx.total_hpwl()
    density = ctx.placement_density(window=10)
    congestion = ctx.congestion_estimate(threshold=0.5)

    return {
        "place_time": place_time,
        "hpwl": hpwl,
        "max_density": density["max_density"],
        "avg_density": density["avg_density"],
        "max_congestion": congestion["max_congestion"],
        "avg_congestion": congestion["avg_congestion"],
        "hot_edges": len(congestion["hot_edges"]),
        "hotspot": congestion["hotspot"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare congestion-aware placement")
    parser.add_argument(
        "--benchmark", default="diffeq1", help="Benchmark name (default: diffeq1)"
    )
    parser.add_argument("--size", default=None, help="Override chipdb size (e.g. 30x30)")
    parser.add_argument(
        "--placer",
        default="both",
        choices=["heap", "sa", "both"],
        help="Placer to test",
    )
    args = parser.parse_args()

    # Check that the JSON netlist exists
    json_path = os.path.join(OUTPUT_DIR, f"{args.benchmark}.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run synthesis first:")
        print(f"  python run_benchmarks.py --benchmarks {args.benchmark} --synth-only")
        sys.exit(1)

    # Find or specify chipdb
    if args.size:
        chipdb_path = os.path.join(OUTPUT_DIR, f"benchmark_{args.size}.bin")
    else:
        # Try common sizes
        for size in ["30x30", "40x40", "50x50"]:
            chipdb_path = os.path.join(OUTPUT_DIR, f"benchmark_{size}.bin")
            if os.path.exists(chipdb_path):
                break
    if not os.path.exists(chipdb_path):
        print(f"ERROR: No chipdb found at {chipdb_path}")
        sys.exit(1)

    placers = ["heap", "sa"] if args.placer == "both" else [args.placer]

    print("=" * 80)
    print(f"Congestion-Aware Placement Comparison: {args.benchmark}")
    print(f"Chipdb: {os.path.basename(chipdb_path)}")
    print("=" * 80)

    for placer in placers:
        print(f"\n--- {placer.upper()} Placer ---")
        print(f"{'Metric':<25} {'No Congestion':>15} {'With Congestion':>15} {'Change':>10}")
        print("-" * 70)

        # Run without congestion
        print(f"  Running {placer} without congestion awareness...", flush=True)
        off = run_comparison(args.benchmark, chipdb_path, placer, 0.0)

        # Run with congestion
        print(f"  Running {placer} with congestion awareness...", flush=True)
        on = run_comparison(args.benchmark, chipdb_path, placer, 0.5)

        # Print comparison
        metrics = [
            ("HPWL", "hpwl", ".0f"),
            ("Place time (s)", "place_time", ".3f"),
            ("Max density", "max_density", ".3f"),
            ("Avg density", "avg_density", ".3f"),
            ("Max congestion", "max_congestion", ".3f"),
            ("Avg congestion", "avg_congestion", ".3f"),
            ("Hot edges (>0.5)", "hot_edges", "d"),
        ]

        for label, key, fmt in metrics:
            v_off = off[key]
            v_on = on[key]
            if isinstance(v_off, (int, float)) and v_off != 0:
                pct = (v_on - v_off) / abs(v_off) * 100
                change = f"{pct:+.1f}%"
            else:
                change = "-"
            print(
                f"{label:<25} {format(v_off, fmt):>15} {format(v_on, fmt):>15} {change:>10}"
            )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

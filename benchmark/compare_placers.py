"""Compare all four placers (HeAP, SA, Hydraulic, ElectroPlace) on benchmarks.

Usage: python compare_placers.py [--benchmark ch_intrinsics] [--size 20x20]

Runs each placer and reports:
  - HPWL (half-perimeter wirelength)
  - Runtime (seconds)
  - Max density
  - Max congestion
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

PLACERS = ["heap", "sa", "hydraulic", "electro"]


def run_comparison(chipdb_path, json_path, placers, seed=1):
    """Run each placer and collect metrics."""
    try:
        import nextpnr
    except ImportError:
        print("Error: nextpnr Python module not found.")
        print("Build with: cd {} && maturin develop".format(ROOT))
        sys.exit(1)

    results = {}

    for placer_name in placers:
        print(f"\n{'=' * 60}")
        print(f"Running placer: {placer_name}")
        print(f"{'=' * 60}")

        try:
            ctx = nextpnr.Context(chipdb=chipdb_path)
            ctx.load_design(json_path)
            ctx.pack()

            start = time.time()
            ctx.place(placer=placer_name, seed=seed)
            elapsed = time.time() - start

            # Collect metrics
            hpwl = ctx.total_hpwl()
            density = ctx.placement_density(window=10)
            max_density = density["max_density"]
            avg_density = density["avg_density"]
            max_congestion = ctx.congestion_estimate(threshold=0.5)["max_congestion"]

            results[placer_name] = {
                "hpwl": hpwl,
                "runtime_s": elapsed,
                "max_density": max_density,
                "avg_density": avg_density,
                "max_congestion": max_congestion,
            }

            print(f"  HPWL:           {hpwl:.0f}")
            print(f"  Runtime:        {elapsed:.3f}s")
            print(f"  Max density:    {max_density:.3f}")
            print(f"  Max congestion: {max_congestion:.3f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[placer_name] = {"error": str(e)}

    return results


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print("PLACER COMPARISON")
    print(f"{'=' * 70}")
    print(
        f"{'Placer':<12} {'HPWL':>10} {'Runtime':>10} {'MaxDens':>10} {'MaxCong':>10}"
    )
    print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for name, data in results.items():
        if "error" in data:
            print(f"{name:<12} {'FAILED':>10}")
        else:
            print(
                f"{name:<12} {data['hpwl']:>10.0f} {data['runtime_s']:>9.3f}s "
                f"{data['max_density']:>10.3f} {data['max_congestion']:>10.3f}"
            )

    # Compute relative metrics (normalized to HeAP if available)
    if "heap" in results and "error" not in results["heap"]:
        base = results["heap"]
        print(f"\n{'Relative to HeAP:'}")
        print(f"{'Placer':<12} {'HPWL':>10} {'Runtime':>10}")
        print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
        for name, data in results.items():
            if "error" in data:
                continue
            hpwl_rel = data["hpwl"] / max(base["hpwl"], 1) * 100
            time_rel = data["runtime_s"] / max(base["runtime_s"], 0.001) * 100
            print(f"{name:<12} {hpwl_rel:>9.1f}% {time_rel:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare FPGA placement algorithms")
    parser.add_argument("--chipdb", help="Path to chipdb .bin file")
    parser.add_argument("--design", help="Path to Yosys JSON design file")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument(
        "--placers",
        default=",".join(PLACERS),
        help="Comma-separated list of placers to compare",
    )
    args = parser.parse_args()

    if not args.chipdb or not args.design:
        print(
            "Usage: python compare_placers.py --chipdb path/to/chipdb.bin --design path/to/design.json"
        )
        print("\nExample:")
        print(
            "  python compare_placers.py --chipdb output/chipdb_20x20.bin --design output/ch_intrinsics.json"
        )
        sys.exit(1)

    placers = [p.strip() for p in args.placers.split(",")]

    results = run_comparison(args.chipdb, args.design, placers, args.seed)
    print_comparison_table(results)


if __name__ == "__main__":
    main()

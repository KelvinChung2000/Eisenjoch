"""Comprehensive placer comparison: place + route + measure real wirelength.

Compares HPWL, Bresenham line estimate, and real routed wirelength for each placer.
Also measures congestion after placement and after routing.

Usage:
    python bench_place_route.py --chipdb path/to/chipdb.bin --design path/to/design.json
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

PLACERS = ["heap", "electro", "hydraulic"]


def pre_place_clock_ios(ctx):
    """Pre-place clock IO cells at dedicated GCLK positions.

    The synthetic chipdb has clock ladders from GCLK0_OUT at (1,0) and
    GCLK1_OUT at (2,0). If clock IOs aren't at these positions, the
    router can't find a path through the clock network.
    """
    cells = ctx.cells
    # Look for IO cells with 'clk' in the name (e.g. $io$clk)
    for cell_name in cells:
        if "clk" in cell_name.lower() and "$io$" in cell_name:
            try:
                ctx.place_cell(cell_name, 1, 0, "IO0")
                return
            except Exception:
                try:
                    ctx.place_cell(cell_name, 2, 0, "IO0")
                    return
                except Exception:
                    pass


def run_one(chipdb_path, json_path, placer_name, seed=1, max_iters=None):
    """Run one placer + router and collect all metrics."""
    import nextpnr

    ctx = nextpnr.Context(chipdb=chipdb_path)
    ctx.load_design(json_path)
    ctx.pack()
    pre_place_clock_ios(ctx)

    # Place
    place_kwargs = dict(placer=placer_name, seed=seed)
    if max_iters is not None:
        place_kwargs["max_iters"] = max_iters
    t0 = time.time()
    ctx.place(**place_kwargs)
    place_time = time.time() - t0

    # Metrics after placement
    hpwl = ctx.total_hpwl()
    line_est = ctx.total_line_estimate()

    max_density = ctx.placement_density(window=10)["max_density"]
    pre_route_cong = ctx.congestion_estimate(threshold=0.5)["max_congestion"]

    # Route with router1, limited iterations to keep runtime reasonable
    t1 = time.time()
    try:
        ctx.route(router="router1", max_iterations=50)
        route_ok = True
    except Exception as e:
        route_ok = False
        print(f"  Route: {e}")
    route_time = time.time() - t1
    # Capture routed wirelength even with partial routing (some nets routed)
    routed_wl = ctx.total_routed_wirelength()

    post_route_cong = ctx.congestion_estimate(threshold=0.5)["max_congestion"]

    return {
        "placer": placer_name,
        "hpwl": hpwl,
        "line_est": line_est,
        "routed_wl": routed_wl,
        "route_ok": route_ok,
        "place_time": place_time,
        "route_time": route_time,
        "max_density": max_density,
        "pre_route_cong": pre_route_cong,
        "post_route_cong": post_route_cong,
    }


def print_table(results, design_name):
    """Print formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  {design_name}")
    print(f"{'=' * 90}")

    # Main metrics table
    hdr = f"{'Placer':<12} {'HPWL':>8} {'LineEst':>8} {'Routed':>8} {'PlaceT':>8} {'RouteT':>8} {'Density':>8} {'Cong':>8}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        routed = str(r["routed_wl"])
        if not r["route_ok"]:
            routed += "*"
        print(
            f"{r['placer']:<12} {r['hpwl']:>8.0f} {r['line_est']:>8.0f} {routed:>8} "
            f"{r['place_time']:>7.2f}s {r['route_time']:>7.2f}s "
            f"{r['max_density']:>8.3f} {r['post_route_cong']:>8.3f}"
        )

    # Relative to HeAP
    heap = next((r for r in results if r["placer"] == "heap"), None)
    if heap:
        print("\n  Relative to HeAP:")
        print(f"  {'Placer':<12} {'HPWL':>8} {'LineEst':>8} {'Routed':>8}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")
        for r in results:
            hpwl_pct = r["hpwl"] / max(heap["hpwl"], 1) * 100
            line_pct = r["line_est"] / max(heap["line_est"], 1) * 100
            if r["routed_wl"] > 0 and heap["routed_wl"] > 0:
                routed_pct = f"{r['routed_wl'] / heap['routed_wl'] * 100:.1f}%"
            else:
                routed_pct = "N/A"
            print(
                f"  {r['placer']:<12} {hpwl_pct:>7.1f}% {line_pct:>7.1f}% {routed_pct:>8}"
            )

    # Estimation accuracy (compare estimates to routed)
    print("\n  Estimation accuracy (ratio to routed wirelength):")
    print(f"  {'Placer':<12} {'HPWL/Routed':>12} {'LineEst/Routed':>14}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 14}")
    for r in results:
        if r["routed_wl"] > 0:
            hpwl_ratio = r["hpwl"] / r["routed_wl"]
            line_ratio = r["line_est"] / r["routed_wl"]
            print(f"  {r['placer']:<12} {hpwl_ratio:>12.4f} {line_ratio:>14.4f}")
        else:
            print(f"  {r['placer']:<12} {'N/A':>12} {'N/A':>14}")


def main():
    parser = argparse.ArgumentParser(description="Place + Route benchmark comparison")
    parser.add_argument("--chipdb", required=True, help="Path to chipdb .bin file")
    parser.add_argument(
        "--design", required=True, help="Path to Yosys JSON design file"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-iters", type=int, default=None, help="Max placer iterations"
    )
    parser.add_argument(
        "--placers",
        default=",".join(PLACERS),
        help="Comma-separated placers to compare",
    )
    args = parser.parse_args()

    placers = [p.strip() for p in args.placers.split(",")]
    design_name = os.path.basename(args.design).replace(".json", "")

    results = []
    for p in placers:
        print(f"\n--- {p} ---")
        try:
            r = run_one(args.chipdb, args.design, p, args.seed, args.max_iters)
            results.append(r)
            print(
                f"  HPWL={r['hpwl']:.0f}  LineEst={r['line_est']:.0f}  Routed={r['routed_wl']}  "
                f"Place={r['place_time']:.2f}s  Route={r['route_time']:.2f}s"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    print_table(results, design_name)


if __name__ == "__main__":
    main()

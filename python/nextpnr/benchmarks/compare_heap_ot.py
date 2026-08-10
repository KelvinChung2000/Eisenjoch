"""Compare HeAP vs OT (MIC0) placer: place + route + all metrics.

Usage:
    uv run python python/nextpnr/benchmarks/compare_heap_ot.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def run_placer(chipdb, design, placer, router="raster", seed=42, **place_kwargs):
    """Run placer + router, return all metrics."""
    import nextpnr

    ctx = nextpnr.Context(chipdb=chipdb)
    ctx.load_design(design)
    ctx.pack()

    # Place
    t0 = time.time()
    ctx.place(placer=placer, seed=seed, **place_kwargs)
    place_time = time.time() - t0

    hpwl = ctx.total_hpwl()
    line_est = ctx.total_line_estimate()
    pre_cong = ctx.congestion_estimate(threshold=0.5)

    # Route
    t1 = time.time()
    try:
        ctx.route(router=router, max_iterations=50)
        route_ok = True
    except Exception as e:
        route_ok = False
        print(f"  Route failed: {e}")
    route_time = time.time() - t1

    routed_wl = ctx.total_routed_wirelength()
    post_cong = ctx.congestion_estimate(threshold=0.5)

    return {
        "placer": placer,
        "place_time": place_time,
        "route_time": route_time,
        "total_time": place_time + route_time,
        "hpwl": hpwl,
        "line_est": line_est,
        "routed_wl": routed_wl,
        "route_ok": route_ok,
        "max_tile_cong": pre_cong.get("max_congestion", 0),
        "excess_overflow": pre_cong.get("excess_overflow", 0),
        "post_max_cong": post_cong.get("max_congestion", 0),
        "post_excess_overflow": post_cong.get("excess_overflow", 0),
    }


def print_results(results, design_name):
    print(f"\n{'=' * 100}")
    print(f"  {design_name}  (place + route)")
    print(f"{'=' * 100}")

    hdr = (
        f"{'Config':<20} {'PlaceT':>7} {'RouteT':>7} {'Total':>7} "
        f"{'HPWL':>8} {'LineEst':>8} {'RoutedWL':>9} "
        f"{'TileCong':>9} {'PostCong':>9} {'Routed':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        ok = "OK" if r["route_ok"] else "FAIL"
        print(
            f"{r['label']:<20} "
            f"{r['place_time']:>6.1f}s {r['route_time']:>6.1f}s {r['total_time']:>6.1f}s "
            f"{r['hpwl']:>8.0f} {r['line_est']:>8.0f} {r['routed_wl']:>9} "
            f"{r['max_tile_cong']:>9.0f} {r['post_max_cong']:>9.0f} {ok:>6}"
        )

    # Relative to HeAP
    heap = next((r for r in results if r["placer"] == "heap"), None)
    if heap and heap["routed_wl"] > 0:
        print("\n  Relative to HeAP:")
        print(f"  {'Config':<20} {'HPWL':>8} {'LineEst':>8} {'RoutedWL':>9} {'TileCong':>9}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
        for r in results:
            def pct(a, b):
                return f"{a / max(b, 1) * 100:.1f}%" if b > 0 else "N/A"
            print(
                f"  {r['label']:<20} "
                f"{pct(r['hpwl'], heap['hpwl']):>8} "
                f"{pct(r['line_est'], heap['line_est']):>8} "
                f"{pct(r['routed_wl'], heap['routed_wl']):>9} "
                f"{pct(r['max_tile_cong'], heap['max_tile_cong']):>9}"
            )


def main():
    chipdb = os.path.join(ROOT, "chip_database", "xc7_exact.bin")
    design = os.path.join(ROOT, "benchmark", "output", "stereovision3.json")
    router = "raster"

    for path, label in [(chipdb, "chipdb"), (design, "design")]:
        if not os.path.exists(path):
            print(f"{label} not found: {path}")
            sys.exit(1)

    design_name = os.path.basename(design).replace(".json", "")
    print(f"Design: {design_name}")
    print(f"Chipdb: {os.path.basename(chipdb)}")
    print(f"Router: {router}")

    configs = [
        ("HeAP", dict(placer="heap")),
        ("OT-path", dict(placer="hydraulic", max_iters=50)),
    ]

    results = []
    for label, kwargs in configs:
        print(f"\n--- {label} ---")
        try:
            r = run_placer(chipdb, design, router=router, **kwargs)
            r["label"] = label
            results.append(r)
            print(
                f"  Place={r['place_time']:.1f}s  Route={r['route_time']:.1f}s  "
                f"HPWL={r['hpwl']:.0f}  RoutedWL={r['routed_wl']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print_results(results, design_name)


if __name__ == "__main__":
    main()

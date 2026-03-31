"""Test the Gupta-Sproull raster router with heap and opt_trans placers.

Usage:
    python test_raster_router.py --chipdb path/to/chipdb.bin --design path/to/design.json
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

PLACERS = ["heap", "hydraulic"]


def is_clock_io(cell_name):
    return "$io$" in cell_name and "clk" in cell_name.lower()


def pre_place_ios(ctx):
    """Pre-place IO cells: clock IOs at dedicated GCLK positions, others around perimeter."""
    cells = ctx.cells
    all_io = [c for c in cells if "$io$" in c]
    if not all_io:
        return

    clk_io = [c for c in all_io if is_clock_io(c)]
    non_clk_io = [c for c in all_io if not is_clock_io(c)]

    width = ctx.width
    height = ctx.height

    clock_bels = [(1, 0, "IO0"), (2, 0, "IO1")]
    placed_io = 0

    for i, cell_name in enumerate(clk_io):
        if i < len(clock_bels):
            x, y, bel_name = clock_bels[i]
            try:
                ctx.place_cell(cell_name, x=x, y=y, bel_name=bel_name)
                placed_io += 1
                continue
            except Exception:
                pass
        non_clk_io.append(cell_name)

    reserved = set(clock_bels[: min(len(clk_io), len(clock_bels))])
    io_bels = []
    for x in range(1, width - 1):
        for bel in ["IO0", "IO1"]:
            if (x, 0, bel) not in reserved:
                io_bels.append((x, 0, bel))
    for y in range(1, height - 1):
        for bel in ["IO0", "IO1"]:
            io_bels.append((0, y, bel))
            io_bels.append((width - 1, y, bel))
    for x in range(1, width - 1):
        for bel in ["IO0", "IO1"]:
            io_bels.append((x, height - 1, bel))

    bel_idx = 0
    for cell_name in non_clk_io:
        placed = False
        while bel_idx < len(io_bels):
            x, y, bel_name = io_bels[bel_idx]
            bel_idx += 1
            try:
                ctx.place_cell(cell_name, x=x, y=y, bel_name=bel_name)
                placed = True
                placed_io += 1
                break
            except Exception:
                continue
    print(f"  Pre-placed {placed_io}/{len(all_io)} IO cells")


def run_one(chipdb_path, json_path, placer_name, router_name, seed=1):
    """Run one placer + router combination and collect metrics."""
    import nextpnr

    ctx = nextpnr.Context(chipdb=chipdb_path)
    ctx.load_design(json_path)
    ctx.pack()
    pre_place_ios(ctx)

    # Place
    t0 = time.time()
    ctx.place(placer=placer_name, seed=seed)
    place_time = time.time() - t0

    hpwl = ctx.total_hpwl()
    line_est = ctx.total_line_estimate()
    print(f"  Placed in {place_time:.2f}s  HPWL={hpwl}  LineEst={line_est}")

    # Route
    t1 = time.time()
    try:
        ctx.route(router=router_name, max_iterations=50, skip_unplaced=True)
        route_ok = True
    except Exception as e:
        route_ok = False
        print(f"  Route failed: {e}")
    route_time = time.time() - t1

    routed_wl = ctx.total_routed_wirelength()
    print(f"  Routed in {route_time:.2f}s  OK={route_ok}  RoutedWL={routed_wl}")

    return {
        "placer": placer_name,
        "router": router_name,
        "place_time": place_time,
        "route_time": route_time,
        "hpwl": hpwl,
        "line_est": line_est,
        "routed_wl": routed_wl,
        "route_ok": route_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Test raster router")
    parser.add_argument("--chipdb", required=True, help="Path to chipdb.bin")
    parser.add_argument("--design", required=True, help="Path to design JSON")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"Chipdb: {args.chipdb}")
    print(f"Design: {args.design}")
    print()

    results = []
    for placer in PLACERS:
        for router in ["raster"]:
            print(f"=== {placer} + {router} ===")
            try:
                r = run_one(args.chipdb, args.design, placer, router, args.seed)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
            print()

    # Summary table
    if results:
        print("=" * 90)
        print(f"{'Placer':<12} {'Router':<10} {'PlaceT':>8} {'RouteT':>8} {'HPWL':>10} {'LineEst':>10} {'RoutedWL':>10} {'OK'}")
        print("-" * 90)
        for r in results:
            print(
                f"{r['placer']:<12} {r['router']:<10} "
                f"{r['place_time']:>7.2f}s {r['route_time']:>7.2f}s "
                f"{r['hpwl']:>10} {r['line_est']:>10} {r['routed_wl']:>10} "
                f"{'Y' if r['route_ok'] else 'N'}"
            )


if __name__ == "__main__":
    main()

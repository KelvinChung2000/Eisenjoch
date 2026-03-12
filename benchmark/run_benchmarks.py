"""Run VTR benchmarks through the nextpnr-rust toolchain.

Usage: python run_benchmarks.py [--size NxN] [--benchmarks name1,name2,...] [--placer heap|sa]

Architecture: LUT4 + DFF + DLATCH, dual clock networks, fixed 100x100 grid (~71K LUTs).
Steps for each benchmark:
  1. Yosys synthesis (Verilog -> LUT4+DFF+DLATCH -> JSON)
  2. Generate or reuse chipdb (default 100x100)
  3. nextpnr-rust pack -> place -> route -> timing
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VERILOG_DIR = os.path.join(HERE, "verilog")
DEMO_DIR = os.path.join(ROOT, "demo")
BBASM = "/home/kelvin/nextpnr/build/bba/bbasm"
RAM_PRIMITIVES = os.path.join(VERILOG_DIR, "ram_primitives.v")

# Benchmarks that need RAM primitives
NEEDS_RAM = {"ch_intrinsics", "mkPktMerge", "or1200", "LU8PEEng", "spree", "raygentop"}

# Benchmark top module names (VTR conventions)
TOP_MODULES = {
    "diffeq1": "diffeq_paj_convert",
    "diffeq2": "diffeq_f_systemC",
    "ch_intrinsics": "memset",
    "sha": "sha1",
    "stereovision0": "sv_chip0_hierarchy_no_mem",
    "stereovision1": "sv_chip1_hierarchy_no_mem",
    "stereovision2": "sv_chip2_hierarchy_no_mem",
    "stereovision3": "sv_chip3_hierarchy_no_mem",
    "blob_merge": "RLE_BlobMerging",
    "bgm": "bgm",
    "mkPktMerge": "mkPktMerge",
    "raygentop": "paj_raygentop_hierarchy_no_mem",
    "or1200": "or1200_flat",
    "LU8PEEng": "LU8PEEng",
    "spree": "system",
}

# SLICEs per logic tile
N_SLICES = 8


def compute_grid_size(lut_count, io_count=0, headroom=2.0):
    """Compute minimum grid NxN for a given LUT count and IO count.

    Layout: NxN grid, IO on edges, BRAM every 15th row, corners are NULL.
    Logic tiles: (N-2)*(N-2) minus ~1/15 for BRAM rows.
    Each logic tile has N_SLICES=8 LUT4s.
    IO BELs: 4*(N-2)*2 on edges.
    Uses 2x headroom for comfortable routing.
    """
    target = int(lut_count * headroom)
    # LUTs per grid ≈ (N-2)^2 * (14/15) * 8
    luts_per_inner = (14.0 / 15.0) * N_SLICES
    inner_needed = target / luts_per_inner
    n_logic = int(math.ceil(math.sqrt(inner_needed))) + 2

    # IO constraint: need 4*(N-2)*2 >= io_count
    # N >= io_count/8 + 2
    n_io = int(math.ceil(io_count / 8.0)) + 2 if io_count > 0 else 0

    n = max(n_logic, n_io)
    # Round up to nearest 10 for cleaner sizes
    n = ((n + 9) // 10) * 10
    return max(n, 30)


def generate_chipdb(size, output_bin):
    """Generate a chipdb of the given size."""
    bba_path = output_bin.replace(".bin", ".bba")

    print(f"  Generating {size} chipdb...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "gen_chipdb.py"), bba_path, "--size", size],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "/home/kelvin/nextpnr/himbaechel"},
    )
    if result.returncode != 0:
        print(f"  ERROR generating chipdb: {result.stderr}")
        return False
    print(f"  {result.stdout.strip()}")

    print("  Assembling binary...")
    result = subprocess.run(
        [BBASM, "--le", bba_path, output_bin],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR assembling: {result.stderr}")
        return False

    os.remove(bba_path)
    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  Chipdb: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return True


def get_or_create_chipdb(grid_n, out_dir):
    """Get or create a chipdb for a given grid size, returns path."""
    size_str = f"{grid_n}x{grid_n}"
    chipdb_path = os.path.join(out_dir, f"benchmark_{size_str}.bin")
    if os.path.exists(chipdb_path):
        size_mb = os.path.getsize(chipdb_path) / (1024 * 1024)
        print(f"  Using cached {size_str} chipdb: {size_mb:.1f} MB")
        return chipdb_path
    if not generate_chipdb(size_str, chipdb_path):
        return None
    return chipdb_path


def synthesize(benchmark_name, verilog_path, output_json):
    """Synthesize a Verilog file to JSON using Yosys."""
    top = TOP_MODULES.get(benchmark_name, benchmark_name)

    ram_read = (
        f"read_verilog -defer {RAM_PRIMITIVES}\n" if benchmark_name in NEEDS_RAM else ""
    )
    synth_script = f"""
read_verilog -defer {verilog_path}
{ram_read}hierarchy -top {top} -check
synth -top {top} -flatten -noalumacc
async2sync
dfflegalize -cell $_DFF_P_ 01 -cell $_DLATCH_P_ 01
dfflibmap -liberty {os.path.join(DEMO_DIR, "cells.lib")}
techmap -map +/adff2dff.v
abc -lut 4
techmap -map {os.path.join(DEMO_DIR, "cells_map.v")}
clean -purge
opt_clean
delete t:$scopeinfo
clean -purge
write_json {output_json}
"""

    result = subprocess.run(
        ["yosys", "-q", "-p", synth_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, result.stderr

    # Parse cell counts from the JSON
    with open(output_json) as f:
        design = json.load(f)

    modules = design.get("modules", {})
    if not modules:
        return None, "No modules in JSON"

    top_mod = next(iter(modules.values()))
    cells = top_mod.get("cells", {})
    cell_types = {}
    for cell in cells.values():
        ct = cell.get("type", "?")
        cell_types[ct] = cell_types.get(ct, 0) + 1

    # Count IO bits (each bit becomes an IO cell after packing)
    ports = top_mod.get("ports", {})
    io_count = sum(len(p.get("bits", [])) for p in ports.values())
    cell_types["_io_count"] = io_count

    return cell_types, None


def run_pnr(benchmark_name, json_path, chipdb_path, placer, router, results):
    """Run pack, place, route, timing through nextpnr-rust Python API."""
    try:
        import nextpnr
    except ImportError:
        print("  ERROR: nextpnr Python module not available. Run: maturin develop")
        return False

    ctx = nextpnr.Context(chipdb=chipdb_path)

    # Load design
    t0 = time.time()
    ctx.load_design(json_path)
    results["load_time"] = time.time() - t0
    results["cells"] = len(ctx.cells)
    results["nets"] = len(ctx.nets)

    # Pack
    t0 = time.time()
    ctx.pack()
    results["pack_time"] = time.time() - t0

    # Pre-place IO cells at IO tiles for clock network access.
    # Dual clock architecture: CLK0 from IO tile (1,0) IO0, CLK1 from (2,0) IO0.
    all_io = [c for c in ctx.cells if c.startswith("$io$")]

    # Identify clock IO cells and assign them to clock driver tiles.
    def is_clock_io(name):
        lower = name.lower()
        return "clk" in lower or "clock" in lower

    clk_io = [c for c in all_io if is_clock_io(c)]
    non_clk_io = [c for c in all_io if not is_clock_io(c)]

    width = ctx.width
    height = ctx.height

    # Reserve clock driver BELs:
    # CLK0: IO0_O → GCLK0_OUT at tile (1,0), so use IO0
    # CLK1: IO1_O → GCLK1_OUT at tile (2,0), so use IO1
    clock_bels = [(1, 0, "IO0"), (2, 0, "IO1")]
    placed_io = 0
    failed_io = []

    # Place clock IOs first at the clock driver tiles
    for i, cell_name in enumerate(clk_io):
        if i < len(clock_bels):
            x, y, bel_name = clock_bels[i]
            try:
                ctx.place_cell(cell_name, x=x, y=y, bel_name=bel_name)
                placed_io += 1
                continue
            except Exception:
                pass
        # If clock BEL placement failed or we have >2 clocks, add to non-clock list
        non_clk_io.append(cell_name)

    # Build remaining IO BEL list, excluding reserved clock BELs
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
        if not placed:
            failed_io.append(cell_name)
    if failed_io:
        print(
            f"    WARNING: {len(failed_io)} IO cells failed to pre-place (of {len(all_io)} total)"
        )
    elif all_io:
        print(f"    Pre-placed {placed_io}/{len(all_io)} IO cells", end=" ", flush=True)

    # Place
    t0 = time.time()
    ctx.place(placer=placer, seed=42)
    results["place_time"] = time.time() - t0

    # Spatial placement density report (sliding window)
    density = ctx.placement_density(window=10)
    max_dens = density["max_density"]
    avg_dens = density["avg_density"]
    hotspot = density["hotspot"]
    n_hot = density["hot_regions"]
    results["max_density_pct"] = max_dens * 100
    results["avg_density_pct"] = avg_dens * 100
    results["density_hotspot"] = hotspot

    util = ctx.utilization_report()
    util_str = " ".join(f"{r}:{p:.0f}%" for r, _, _, p in util.rows if p > 0)
    print(
        f"dens=max:{max_dens:.0%},avg:{avg_dens:.0%} util=[{util_str}]",
        end=" ",
        flush=True,
    )
    if n_hot > 0:
        print(f"hotspot=({hotspot[0]},{hotspot[1]})", end=" ", flush=True)

    # Early stop if local density is too high (>70% in any region = likely unroutable)
    density_threshold = 0.70
    if max_dens > density_threshold:
        print(
            f"\n    SKIP: local density {max_dens:.0%} at ({hotspot[0]},{hotspot[1]}) "
            f"exceeds {density_threshold:.0%} -- likely unroutable"
        )
        results["status"] = "density_skip"
        results["skip_reason"] = (
            f"local density {max_dens:.0%} at ({hotspot[0]},{hotspot[1]})"
        )
        return False

    # Add clock constraints for all clock-like nets in the design
    clock_names = [n for n in ctx.nets if "clk" in n.lower() or "clock" in n.lower()]
    for clk_name in clock_names:
        ctx.add_clock(clk_name, freq_mhz=100.0)

    # Route
    t0 = time.time()
    # Router2 needs a larger bounding box margin on large grids to find paths
    grid_size = max(ctx.width, ctx.height)
    bb_margin = max(3, grid_size // 10)
    ctx.route(router=router, bb_margin=bb_margin)
    results["route_time"] = time.time() - t0

    # Timing
    t0 = time.time()
    timing = ctx.timing_report()
    results["timing_time"] = time.time() - t0
    results["fmax_mhz"] = timing.fmax
    results["worst_slack_ps"] = timing.worst_slack
    results["failing_endpoints"] = timing.num_failing
    results["total_endpoints"] = timing.num_endpoints

    results["total_time"] = (
        results["load_time"]
        + results["pack_time"]
        + results["place_time"]
        + results["route_time"]
        + results["timing_time"]
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Run VTR benchmarks")
    parser.add_argument(
        "--size", default=None, help="Override chipdb grid size (e.g. 50x50)"
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated benchmark names (default: all)",
    )
    parser.add_argument(
        "--placer", default="heap", choices=["heap", "sa"], help="Placer algorithm"
    )
    parser.add_argument(
        "--router",
        default="router1",
        choices=["router1", "router2"],
        help="Router algorithm",
    )
    parser.add_argument(
        "--synth-only", action="store_true", help="Only run synthesis, skip P&R"
    )
    args = parser.parse_args()

    # Discover available benchmarks
    available = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(VERILOG_DIR)
        if f.endswith(".v") and f != "ram_primitives.v"
    )

    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
        for b in benchmarks:
            if b not in available:
                print(f"ERROR: benchmark '{b}' not found in {VERILOG_DIR}")
                sys.exit(1)
    else:
        benchmarks = available

    print("=" * 72)
    mode = f"fixed {args.size}" if args.size else "auto-sized"
    print(
        f"VTR Benchmark Suite -- nextpnr-rust ({mode} grid, {args.placer} placer, {args.router})"
    )
    print("=" * 72)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print()

    # Create output directory
    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Synthesize all benchmarks
    print("--- Phase 1: Yosys Synthesis ---")
    synth_results = {}
    for name in benchmarks:
        verilog_path = os.path.join(VERILOG_DIR, f"{name}.v")
        json_path = os.path.join(out_dir, f"{name}.json")

        print(f"  [{name}] synthesizing...", end=" ", flush=True)
        t0 = time.time()
        cell_types, err = synthesize(name, verilog_path, json_path)
        elapsed = time.time() - t0

        if err:
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"    {err[:200]}")
            synth_results[name] = {"status": "synth_fail", "error": err[:200]}
        else:
            luts = cell_types.get("LUT4", 0)
            dffs = cell_types.get("DFF", 0)
            ios = cell_types.get("_io_count", 0)
            print(f"OK  {luts} LUT4, {dffs} DFF, {ios} IOs ({elapsed:.1f}s)")
            synth_results[name] = {
                "status": "synth_ok",
                "luts": luts,
                "dffs": dffs,
                "ios": ios,
                "cell_types": cell_types,
                "synth_time": elapsed,
            }
    print()

    if args.synth_only:
        print_synth_summary(synth_results)
        return

    # Step 2 + 3: Generate chipdb per benchmark and run P&R
    print("--- Phase 2+3: Generate Chipdbs & Place & Route ---")
    chipdb_cache = {}  # grid_n -> chipdb_path
    all_results = {}
    for name in benchmarks:
        sr = synth_results.get(name, {})
        if sr.get("status") != "synth_ok":
            all_results[name] = sr
            continue

        luts = sr["luts"]
        dffs = sr["dffs"]
        ios = sr.get("ios", 0)
        total_cells = luts + dffs

        # Determine grid size
        if args.size:
            parts = args.size.lower().split("x")
            grid_n = int(parts[0])
        else:
            grid_n = compute_grid_size(total_cells, io_count=ios)

        # Get or create chipdb
        if grid_n not in chipdb_cache:
            print(
                f"  [{name}] needs {grid_n}x{grid_n} grid for {luts} LUTs + {dffs} DFFs"
            )
            chipdb_path = get_or_create_chipdb(grid_n, out_dir)
            if chipdb_path is None:
                all_results[name] = {**sr, "status": "chipdb_fail"}
                print(f"  [{name}] FAILED: chipdb generation failed")
                continue
            chipdb_cache[grid_n] = chipdb_path
        chipdb_path = chipdb_cache[grid_n]

        json_path = os.path.join(out_dir, f"{name}.json")
        print(f"  [{name}] ({luts} LUTs, {grid_n}x{grid_n}) ", end="", flush=True)

        pnr_results = {**sr}
        pnr_results["grid_size"] = f"{grid_n}x{grid_n}"
        try:
            t0 = time.time()
            ok = run_pnr(
                name, json_path, chipdb_path, args.placer, args.router, pnr_results
            )
            wall = time.time() - t0
            if ok:
                pnr_results["status"] = "ok"
                fmax = pnr_results.get("fmax_mhz", 0)
                print(
                    f"OK  place={pnr_results['place_time']:.2f}s "
                    f"route={pnr_results['route_time']:.2f}s "
                    f"fmax={fmax:.1f}MHz "
                    f"total={wall:.2f}s"
                )
            else:
                pnr_results["status"] = "pnr_fail"
                print(f"FAILED ({wall:.2f}s)")
        except Exception as e:
            pnr_results["status"] = "pnr_fail"
            pnr_results["error"] = str(e)[:200]
            print(f"ERROR: {str(e)[:100]}")

        all_results[name] = pnr_results

    print()

    # Save results
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print_summary(all_results)


def print_synth_summary(results):
    print("=" * 72)
    print(f"{'Benchmark':<20} {'Status':<10} {'LUT4':>8} {'DFF':>8}")
    print("-" * 72)
    for name, r in sorted(results.items()):
        status = r.get("status", "?")
        luts = r.get("luts", "-")
        dffs = r.get("dffs", "-")
        print(f"{name:<20} {status:<10} {luts:>8} {dffs:>8}")
    print("=" * 72)


def print_summary(results):
    print("=" * 100)
    print(
        f"{'Benchmark':<18} {'LUTs':>6} {'DFFs':>6} {'Grid':>8} {'Dens':>5} "
        f"{'Pack':>7} {'Place':>7} {'Route':>7} {'Total':>7} {'Fmax':>8} {'Status':<12}"
    )
    print("-" * 100)
    for name, r in sorted(results.items()):
        status = r.get("status", "?")
        luts = r.get("luts", "-")
        dffs = r.get("dffs", "-")
        grid = r.get("grid_size", "-")
        dens = f"{r['max_density_pct']:.0f}%" if "max_density_pct" in r else "-"
        if status == "ok":
            pack_t = f"{r.get('pack_time', 0):.2f}s"
            place_t = f"{r.get('place_time', 0):.2f}s"
            route_t = f"{r.get('route_time', 0):.2f}s"
            total_t = f"{r.get('total_time', 0):.2f}s"
            fmax = f"{r.get('fmax_mhz', 0):.1f}"
        else:
            pack_t = place_t = route_t = total_t = fmax = "-"
        print(
            f"{name:<18} {luts:>6} {dffs:>6} {grid:>8} {dens:>5} "
            f"{pack_t:>7} {place_t:>7} {route_t:>7} {total_t:>7} {fmax:>8} {status:<12}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()

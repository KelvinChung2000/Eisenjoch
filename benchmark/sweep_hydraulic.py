#!/usr/bin/env python3
"""Sweep hydraulic placer parameters to find optimal configuration.

Tests combinations of force configurations, init strategies, and expanding box
settings across multiple designs. Records HPWL, line estimate, congestion, density,
and placement time for each combination.

Usage:
    python sweep_hydraulic.py --chipdb path/to/chipdb.bin --designs-dir path/to/designs/
    python sweep_hydraulic.py --chipdb chipdb.bin --designs-dir designs/ --configs gas_io4 star_only
    python sweep_hydraulic.py --chipdb chipdb.bin --designs-dir designs/ --output results.csv
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "python"))

import nextpnr  # noqa: E402

# ---------------------------------------------------------------------------
# Force configurations to sweep
# ---------------------------------------------------------------------------

FORCE_CONFIGS = [
    # Pure kinetic gas (no star model)
    {
        "star_weight": 0.0,
        "pressure_weight_end": 1.0,
        "io_boost": 4.0,
        "label": "gas_io4",
    },
    {
        "star_weight": 0.0,
        "pressure_weight_end": 1.0,
        "io_boost": 8.0,
        "label": "gas_io8",
    },
    # Pure star model (no gas)
    {"star_weight": 1.0, "pressure_weight_end": 0.0, "label": "star_only"},
    # Hybrid: star + gas pressure ramp
    {"star_weight": 1.0, "pressure_weight_end": 2.0, "label": "hybrid_pw2"},
    {"star_weight": 1.0, "pressure_weight_end": 4.0, "label": "hybrid_pw4"},
    {
        "star_weight": 0.5,
        "pressure_weight_end": 1.0,
        "io_boost": 4.0,
        "label": "balanced",
    },
    {
        "star_weight": 1.0,
        "pressure_weight_end": 0.5,
        "io_boost": 8.0,
        "label": "star_dom",
    },
    {
        "star_weight": 0.2,
        "pressure_weight_end": 2.0,
        "io_boost": 8.0,
        "label": "gas_dom",
    },
]

INIT_STRATEGIES = ["centroid", "uniform", "random_bel"]

EXPANDING_BOX = [True, False]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pre_place_clock_ios(ctx):
    """Pre-place clock IO cells at dedicated GCLK positions.

    The synthetic chipdb has clock ladders from GCLK0_OUT at (1,0) and
    GCLK1_OUT at (2,0). Clock IOs must be at these positions so the
    placer can properly account for clock network connectivity.
    """
    for cell_name in ctx.cells:
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


def run_single(
    chipdb_path,
    design_path,
    force_cfg,
    seed=1,
    init_strategy="centroid",
    expanding_box=True,
):
    """Run one placement configuration and return metrics dict."""
    ctx = nextpnr.Context(chipdb=chipdb_path)
    ctx.load_design(str(design_path))
    ctx.pack()
    pre_place_clock_ios(ctx)

    # Build place kwargs from force_cfg (exclude 'label')
    place_kwargs = {k: v for k, v in force_cfg.items() if k != "label"}
    place_kwargs["placer"] = "hydraulic"
    place_kwargs["seed"] = seed
    place_kwargs["max_iters"] = 200
    place_kwargs["init_strategy"] = init_strategy
    place_kwargs["enable_expanding_box"] = expanding_box

    t0 = time.perf_counter()
    ctx.place(**place_kwargs)
    place_time = time.perf_counter() - t0

    hpwl = ctx.total_hpwl()
    line_est = ctx.total_line_estimate()

    max_density = ctx.placement_density(window=10)["max_density"]
    max_congestion = ctx.congestion_estimate(threshold=0.5)["max_congestion"]

    return {
        "hpwl": hpwl,
        "line_est": line_est,
        "max_congestion": max_congestion,
        "max_density": max_density,
        "place_time": place_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "design",
    "config",
    "init_strategy",
    "expanding_box",
    "hpwl",
    "line_est",
    "max_congestion",
    "max_density",
    "place_time",
]


def main():
    parser = argparse.ArgumentParser(description="Sweep hydraulic placer parameters")
    parser.add_argument(
        "--chipdb", required=True, help="Path to chip database .bin file"
    )
    parser.add_argument(
        "--designs-dir",
        required=True,
        help="Directory containing design JSON files",
    )
    parser.add_argument("--output", default="sweep_results.csv", help="Output CSV file")
    parser.add_argument(
        "--configs",
        nargs="*",
        help="Filter to specific config labels (default: all)",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = parser.parse_args()

    designs = sorted(Path(args.designs_dir).glob("*.json"))
    if not designs:
        print(f"No .json designs found in {args.designs_dir}", file=sys.stderr)
        sys.exit(1)

    configs = FORCE_CONFIGS
    if args.configs:
        configs = [c for c in configs if c["label"] in args.configs]
        if not configs:
            print(
                f"No matching configs found. Available: "
                f"{[c['label'] for c in FORCE_CONFIGS]}",
                file=sys.stderr,
            )
            sys.exit(1)

    total = len(configs) * len(INIT_STRATEGIES) * len(EXPANDING_BOX) * len(designs)
    count = 0
    results = []

    for design_path in designs:
        design_name = design_path.stem
        for force_cfg in configs:
            label = force_cfg["label"]
            for init_strat in INIT_STRATEGIES:
                for exp_box in EXPANDING_BOX:
                    count += 1
                    tag = f"{label}/{init_strat}/{'box' if exp_box else 'nobox'}"
                    print(
                        f"[{count}/{total}] {design_name} / {tag}",
                        file=sys.stderr,
                    )

                    try:
                        metrics = run_single(
                            args.chipdb,
                            design_path,
                            force_cfg,
                            seed=args.seed,
                            init_strategy=init_strat,
                            expanding_box=exp_box,
                        )
                        results.append(
                            {
                                "design": design_name,
                                "config": label,
                                "init_strategy": init_strat,
                                "expanding_box": exp_box,
                                **metrics,
                            }
                        )
                    except Exception as e:
                        print(f"  FAILED: {e}", file=sys.stderr)
                        results.append(
                            {
                                "design": design_name,
                                "config": label,
                                "init_strategy": init_strat,
                                "expanding_box": exp_box,
                                "hpwl": float("nan"),
                                "line_est": float("nan"),
                                "max_congestion": float("nan"),
                                "max_density": float("nan"),
                                "place_time": float("nan"),
                            }
                        )

    # Write CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}", file=sys.stderr)

    # Print summary table to stdout
    print(
        f"\n{'Design':<20} {'Config':<15} {'Init':<12} {'Box':<5} "
        f"{'HPWL':>10} {'LineEst':>10} {'Cong':>8} "
        f"{'Dens':>8} {'Time':>8}"
    )
    print("-" * 105)
    for r in results:
        box_str = "Y" if r["expanding_box"] else "N"
        print(
            f"{r['design']:<20} {r['config']:<15} {r['init_strategy']:<12} "
            f"{box_str:<5} {r['hpwl']:>10.0f} {r['line_est']:>10.0f} "
            f"{r['max_congestion']:>8.3f} "
            f"{r['max_density']:>8.3f} {r['place_time']:>7.1f}s"
        )

    # Print best config per design (by HPWL)
    designs_seen = list(dict.fromkeys(r["design"] for r in results))

    if designs_seen:
        print("\nBest config per design (by HPWL)")
        print(
            f"{'Design':<20} {'Config':<15} {'Init':<12} {'Box':<5} {'HPWL':>10} {'LineEst':>10}"
        )
        print("-" * 80)
        for d in designs_seen:
            d_results = [r for r in results if r["design"] == d]
            valid = [
                r
                for r in d_results
                if not (isinstance(r["hpwl"], float) and math.isnan(r["hpwl"]))
            ]
            if valid:
                best = min(valid, key=lambda r: r["hpwl"])
                box_str = "Y" if best["expanding_box"] else "N"
                print(
                    f"{d:<20} {best['config']:<15} {best['init_strategy']:<12} "
                    f"{box_str:<5} {best['hpwl']:>10.0f} {best['line_est']:>10.0f}"
                )


if __name__ == "__main__":
    main()

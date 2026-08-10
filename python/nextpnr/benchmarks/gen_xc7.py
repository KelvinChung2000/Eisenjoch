# ruff: noqa: E402
"""Unified XC7 chipdb generator.

Variants:
- standard: base-size collapsed CLB/BRAM/DSP composite fabric.
- large: paper-scale collapsed CLB/BRAM/DSP composite fabric.
"""

from __future__ import annotations

import argparse
import subprocess
from os import path

from nextpnr.benchmarks.gen_xc7_columns import (
    PAPER_TARGETS,
    build_clk_bufg_tile_type,
    build_composite_tile_type,
    build_paper_scale_tilegrid,
    build_standard_scale_tilegrid,
    load_tileconn,
    load_tilegrid,
    parse_targets,
)
from nextpnr.benchmarks.gen_xc7_hybrid import BBASM, CHIPDB_DIR, generate_xc7_hybrid


def _composite_builders(xray, specs, tileconn):
    builders = {
        name: (lambda ch, spec=spec: build_composite_tile_type(ch, xray, spec, tileconn))
        for name, spec in specs.items()
    }
    builders["CLK_BUFG"] = build_clk_bufg_tile_type
    return builders


def generate(
    output_bba,
    xray,
    tilegrid_path,
    tileconn_path,
    *,
    variant="standard",
    include_bufg=True,
    targets=None,
    size=None,
):
    base_tilegrid = load_tilegrid(tilegrid_path)
    base_tileconn = load_tileconn(tileconn_path)

    if variant == "standard":
        if size is not None:
            raise ValueError("--size is not supported for the composite standard variant")
        standard_tilegrid, synthetic_tileconn, specs, solution, _layout = build_standard_scale_tilegrid(
            base_tilegrid,
            base_tileconn,
            xray,
            targets,
        )
        generate_xc7_hybrid(
            output_bba,
            xray,
            standard_tilegrid,
            base_tileconn,
            size=None,
            chip_name="xc7_standard",
            device_name="XC7_STANDARD",
            include_bufg=include_bufg,
            synthetic_tile_builders=_composite_builders(xray, specs, base_tileconn),
            synthetic_tileconn=synthetic_tileconn,
            direct_tileconn_types=set(specs),
        )
        return solution

    if variant != "large":
        raise ValueError(f"unknown XC7 variant {variant!r}")

    large_tilegrid, synthetic_tileconn, specs, solution, _layout = build_paper_scale_tilegrid(
        base_tilegrid,
        base_tileconn,
        xray,
        targets or PAPER_TARGETS,
    )
    generate_xc7_hybrid(
        output_bba,
        xray,
        large_tilegrid,
        base_tileconn,
        size=None,
        chip_name="xc7_large",
        device_name="XC7_LARGE",
        include_bufg=include_bufg,
        synthetic_tile_builders=_composite_builders(xray, specs, base_tileconn),
        synthetic_tileconn=synthetic_tileconn,
        direct_tileconn_types=set(specs),
    )
    return solution


def generate_standard(output_bba, xray, tilegrid_path, tileconn_path, size=None, include_bufg=True):
    return generate(
        output_bba,
        xray,
        tilegrid_path,
        tileconn_path,
        variant="standard",
        include_bufg=include_bufg,
        size=size,
    )


def generate_large(output_bba, xray, tilegrid_path, tileconn_path, targets=None, include_bufg=True):
    return generate(
        output_bba,
        xray,
        tilegrid_path,
        tileconn_path,
        variant="large",
        include_bufg=include_bufg,
        targets=targets,
    )


def assemble_bba(output_bba):
    output_bin = output_bba.replace(".bba", ".bin")
    subprocess.run([BBASM, "--le", output_bba, output_bin], check=True)
    return output_bin


def main():
    parser = argparse.ArgumentParser(description="Generate XC7 standard or paper-scale chipdb")
    parser.add_argument(
        "--variant",
        choices=("standard", "large"),
        required=True,
        help="Database variant to generate",
    )
    parser.add_argument("--xray", required=True, help="Path to prjxray-db/artix7")
    parser.add_argument("--tilegrid", required=True, help="Path to device tilegrid.json")
    parser.add_argument("--tileconn", required=True, help="Path to device tileconn.json")
    parser.add_argument(
        "--output",
        default=path.join(CHIPDB_DIR, "xc7_standard.bba"),
        help="Output .bba path",
    )
    parser.add_argument("--bin", action="store_true", help="Also assemble a .bin with bbasm")
    parser.add_argument("--no-bufg", action="store_true", help="Do not expose BUFG BELs")
    parser.add_argument(
        "--targets",
        default=None,
        help="Resource targets, e.g. lut=538000,ff=1075000,bram18=1728,dsp=768",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Standard-variant grid override NxN",
    )
    args = parser.parse_args()

    generate(
        args.output,
        args.xray,
        args.tilegrid,
        args.tileconn,
        variant=args.variant,
        include_bufg=not args.no_bufg,
        targets=parse_targets(args.targets) if args.targets else None,
        size=args.size,
    )
    if args.bin:
        assemble_bba(args.output)


if __name__ == "__main__":
    main()

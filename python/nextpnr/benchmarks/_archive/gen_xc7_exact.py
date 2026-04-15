# ruff: noqa: E402
"""Generate xc7_exact as a Project X-Ray backed XC7 hybrid chipdb.

This profile builds on gen_xc7_hybrid: keep prjxray tile types, tile wires,
tile PIPs, and tileconn-derived inter-tile nodes, while exposing simplified
BEL types where that keeps the rest of the flow practical. The intentional
simplifications are SLICEL/SLICEM -> LUT6/DFF/DLATCH, simplified IO sites, and
BUFGCTRL -> BUFG.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from os import path

from nextpnr.benchmarks.gen_xc7_hybrid import BBASM, CHIPDB_DIR, generate_xc7_hybrid
from nextpnr.benchmarks._archive.gen_xc7_xl import load_tilegrid, stitch_tilegrid_4x4, write_tilegrid


def generate_xc7_exact(output_bba, xray, tilegrid_path, tileconn_path, size=None):
    """Generate xc7_exact as the prjxray-backed hybrid fabric plus BUFG BELs."""
    generate_xc7_hybrid(
        output_bba,
        xray,
        tilegrid_path,
        tileconn_path,
        size=size,
        chip_name="xc7_exact",
        device_name="XC7_EXACT",
        include_bufg=True,
    )


def generate_xc7_exact_500k(output_bba, xray, tilegrid_path, tileconn_path):
    """Generate a paper-scale xc7_exact database from a 4x4 stitched tilegrid."""
    expanded_tilegrid = stitch_tilegrid_4x4(load_tilegrid(tilegrid_path))
    with tempfile.TemporaryDirectory(prefix="xc7_exact_500k_") as tmpdir:
        stitched_tilegrid = path.join(tmpdir, "tilegrid_xc7_exact_500k.json")
        write_tilegrid(expanded_tilegrid, stitched_tilegrid)
        generate_xc7_hybrid(
            output_bba,
            xray,
            stitched_tilegrid,
            tileconn_path,
            size=None,
            chip_name="xc7_exact_500k",
            device_name="XC7_EXACT_500K",
            include_bufg=True,
        )


def assemble_bba(output_bba):
    output_bin = output_bba.replace(".bba", ".bin")
    subprocess.run([BBASM, "--le", output_bba, output_bin], check=True)
    return output_bin


def main():
    parser = argparse.ArgumentParser(
        description="Generate xc7_exact: prjxray XC7 hybrid fabric plus BUFG BELs"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=path.join(CHIPDB_DIR, "xc7_exact.bba"),
        help="Normal-size output .bba path (default: chip_database/xc7_exact.bba)",
    )
    parser.add_argument("--xray", required=True, help="Path to prjxray-db/artix7")
    parser.add_argument(
        "--tilegrid",
        required=True,
        help="Path to device tilegrid.json",
    )
    parser.add_argument("--tileconn", required=True, help="Path to tileconn.json")
    parser.add_argument(
        "--size",
        default=None,
        help="Normal-size grid override NxN (ignored for the 500k variant)",
    )
    parser.add_argument(
        "--variant",
        choices=("normal", "500k", "both"),
        default="normal",
        help="Which database to generate (default: normal)",
    )
    parser.add_argument(
        "--large-output",
        default=path.join(CHIPDB_DIR, "xc7_exact_500k.bba"),
        help="500k-scale output .bba path (default: chip_database/xc7_exact_500k.bba)",
    )
    parser.add_argument(
        "--bin",
        action="store_true",
        help="Also assemble the .bba into a .bin using bbasm",
    )
    args = parser.parse_args()

    if args.variant in ("normal", "both"):
        generate_xc7_exact(args.output, args.xray, args.tilegrid, args.tileconn, args.size)
        if args.bin:
            assemble_bba(args.output)

    if args.variant in ("500k", "both"):
        large_output = args.large_output if args.variant == "both" else args.output
        generate_xc7_exact_500k(large_output, args.xray, args.tilegrid, args.tileconn)
        if args.bin:
            assemble_bba(large_output)


if __name__ == "__main__":
    main()

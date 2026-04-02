"""Generate an artificially enlarged XC7 hybrid chipdb by tiling the base device.

The current shape is a 4x4 stitch of the base XC7 tilegrid:
- 16 quadrants total
- internal seam IO rows/columns removed
- outer perimeter IO retained

This produces an "xc7-xl" benchmark fabric that is larger than the base hybrid
device while reusing the same tile types, routing data, and BEL construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from os import path

from nextpnr.benchmarks.gen_xc7_hybrid import BBASM, CHIPDB_DIR, generate_xc7_hybrid


def _hash_inputs(script_path, tilegrid_path, tileconn_path, xray_root):
    h = hashlib.sha256()
    for fpath in (script_path, tilegrid_path, tileconn_path):
        with open(fpath, "rb") as f:
            h.update(f.read())
    h.update(str(xray_root).encode())
    h.update(b"xc7-xl-4x4-seamless")
    return h.hexdigest()


def _is_cached(output_bin, expected_hash):
    hash_path = output_bin + ".hash"
    if not path.exists(output_bin) or not path.exists(hash_path):
        return False
    with open(hash_path) as f:
        return f.read().strip() == expected_hash


def _save_hash(output_bin, content_hash):
    with open(output_bin + ".hash", "w") as f:
        f.write(content_hash + "\n")


def load_tilegrid(tilegrid_path):
    with open(tilegrid_path) as f:
        return json.load(f)


def tilegrid_dimensions(tilegrid):
    max_x = max(tile["grid_x"] for tile in tilegrid.values())
    max_y = max(tile["grid_y"] for tile in tilegrid.values())
    return max_x + 1, max_y + 1


def stitch_tilegrid_4x4(tilegrid):
    """Return a stitched 4x4 tilegrid with internal seam IO removed."""
    width, height = tilegrid_dimensions(tilegrid)
    out = {}

    def axis_base(q, dim):
        if q == 0:
            return 0
        return (dim - 1) + (q - 1) * (dim - 2)

    for qy in range(4):
        for qx in range(4):
            for tile_name, tile_data in sorted(tilegrid.items()):
                x = tile_data["grid_x"]
                y = tile_data["grid_y"]

                # Remove internal seam border tiles from each quadrant copy.
                if qx < 3 and x == width - 1:
                    continue
                if qx > 0 and x == 0:
                    continue
                if qy < 3 and y == height - 1:
                    continue
                if qy > 0 and y == 0:
                    continue

                out_x = axis_base(qx, width) + (x if qx == 0 else x - 1)
                out_y = axis_base(qy, height) + (y if qy == 0 else y - 1)
                new_name = f"Q{qy}{qx}__{tile_name}"

                new_tile = deepcopy(tile_data)
                new_tile["grid_x"] = out_x
                new_tile["grid_y"] = out_y
                if "sites" in new_tile:
                    new_tile["sites"] = {
                        f"Q{qy}{qx}__{site_name}": site_type
                        for site_name, site_type in new_tile["sites"].items()
                    }
                out[new_name] = new_tile

    return out


def write_tilegrid(tilegrid, out_path):
    with open(out_path, "w") as f:
        json.dump(tilegrid, f, indent=2, sort_keys=True)
        f.write("\n")


def get_or_create_xc7_xl(output_bin, xray, tilegrid, tileconn, force=False):
    content_hash = _hash_inputs(__file__, tilegrid, tileconn, xray)
    if not force and _is_cached(output_bin, content_hash):
        size_mb = os.path.getsize(output_bin) / (1024 * 1024)
        print(f"Using cached XC7 XL chipdb: {output_bin} ({size_mb:.1f} MB)")
        return output_bin

    print("Generating XC7 XL chipdb (4x4 stitched hybrid fabric)...")
    t0 = time.time()

    expanded_tilegrid = stitch_tilegrid_4x4(load_tilegrid(tilegrid))

    with tempfile.TemporaryDirectory(prefix="xc7_xl_") as tmpdir:
        stitched_tilegrid = path.join(tmpdir, "tilegrid_xc7_xl.json")
        write_tilegrid(expanded_tilegrid, stitched_tilegrid)

        bba_path = output_bin.replace(".bin", ".bba")
        generate_xc7_hybrid(bba_path, xray, stitched_tilegrid, tileconn, size=None)

        print("Assembling binary...")
        subprocess.run([BBASM, "--le", bba_path, output_bin], check=True)
        os.remove(bba_path)

    _save_hash(output_bin, content_hash)
    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"XC7 XL chipdb: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return output_bin


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 4x4 stitched XC7 XL hybrid chipdb"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=path.join(CHIPDB_DIR, "xc7_xl.bin"),
        help="Output .bin file path (default: chip_database/xc7_xl.bin)",
    )
    parser.add_argument("--xray", required=True, help="Path to prjxray-db/artix7")
    parser.add_argument("--tilegrid", required=True, help="Path to base tilegrid.json")
    parser.add_argument("--tileconn", required=True, help="Path to tileconn.json")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()

    output_bin = args.output
    if output_bin.endswith(".bba"):
        output_bin = output_bin.replace(".bba", ".bin")

    if args.force:
        for f in (output_bin, output_bin + ".hash"):
            if path.exists(f):
                os.remove(f)

    get_or_create_xc7_xl(output_bin, args.xray, args.tilegrid, args.tileconn, args.force)


if __name__ == "__main__":
    main()

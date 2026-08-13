# ruff: noqa: E402
"""Generate the synthetic fabric shared by nextpnr and eisenjoch.

The fabric *is* upstream nextpnr's `example` uarch -- this script imports
`example_arch_gen.py` from the pinned upstream checkout and reuses its tile
types, switch matrix, nodes and timings verbatim, overriding only the grid
size. That matters: the C++ `example` uarch has the bel types, wire names and
constids compiled in, so a fabric that merely *resembles* the example arch
would not load. Reusing the generator is what makes one .bin serve both tools.

Usage:
    gen_shared_chipdb.py <out.bba> [--size N] [--known-id-count N]

`--known-id-count 0` embeds every constid string in the binary. The C++ side
supplies ids 1..N itself from `constids.inc` and expects them *absent*; the
Rust side has no compiled-in table and so reads them from the file. See
`docs/nextpnr_faithful_port_plan.md` for why both are emitted.
"""

import argparse
import importlib.util
import sys
from os import path

UPSTREAM = "/home/kelvin/nextpnr-upstream"
EXAMPLE_DIR = path.join(UPSTREAM, "himbaechel/uarch/example")


def load_upstream_generator():
    """Import upstream's example_arch_gen.py as a module."""
    sys.path.insert(0, path.join(UPSTREAM, "himbaechel"))
    spec = importlib.util.spec_from_file_location(
        "example_arch_gen", path.join(EXAMPLE_DIR, "example_arch_gen.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(gen, size: int, known_id_count):
    """Upstream `main()`, with the grid size and constid policy parameterised."""
    gen.X = size
    gen.Y = size
    X, Y = size, size

    ch = gen.Chip("example", "EX1", X, Y)
    ch.strs.read_constids(path.join(EXAMPLE_DIR, "constids.inc"))
    ch.read_gfxids(path.join(EXAMPLE_DIR, "gfxids.inc"))
    gen.create_logic_tiletype(ch)
    gen.create_io_tiletype(ch)
    gen.create_bram_tiletype(ch)
    gen.create_corner_tiletype(ch)

    for x in range(X):
        for y in range(Y):
            if x == 0 or x == X - 1:
                ch.set_tile_type(x, y, "IO" if 0 < y < Y - 1 else "NULL")
            elif y == 0 or y == Y - 1:
                ch.set_tile_type(x, y, "IO")
            elif (y % 15) == 7:
                ch.set_tile_type(x, y, "BRAM")
            else:
                ch.set_tile_type(x, y, "LOGIC")

    gen.create_nodes(ch)
    gen.set_timings(ch)

    if known_id_count is not None:
        ch.strs.known_id_count = known_id_count
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--known-id-count", type=int, default=None)
    args = ap.parse_args()

    gen = load_upstream_generator()
    ch = build(gen, args.size, args.known_id_count)
    ch.write_bba(args.out)


if __name__ == "__main__":
    main()

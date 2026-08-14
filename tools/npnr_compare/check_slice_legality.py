#!/usr/bin/env python3
"""Count example-uarch slice legality violations in a placed JSON.

Mirrors `slice_valid` (upstream `himbaechel/uarch/example/example.cc:134`): a
slice holding both a LUT and an FF is legal only if the FF's D net is that LUT's
F net, or the LUT's I3 input is unused.

The I3 escape hatch is dead on this benchmark. `lut_i3_used` tests
`getPort(I[K-1]) != nullptr`, which returns the *net*, and `replace_constants`
turns a constant tie into the real GND net -- so an I3 tied to '0' counts as
used. Every LUT4 here has I[3] = '0', which reduces the rule to: a LUT and an FF
may share a slice only if the FF is driven by that LUT.

Validated against the tool: on a placement nextpnr rejected with 248
"post-placement validity check failed" warnings, this reports 124 illegal slices
-- 248/2, exact.

Usage: check_slice_legality.py <placed.json> [...]
Exit status is nonzero if any input has an illegal slice.
"""
import collections
import json
import re
import sys

BEL_RE = re.compile(r"(X\d+Y\d+)/L(\d+)_(LUT|FF)")


def check(path):
    top = json.load(open(path))["modules"]["top"]
    cells = top["cells"]

    slots = collections.defaultdict(dict)
    unplaced = 0
    for cell in cells.values():
        bel = cell.get("attributes", {}).get("NEXTPNR_BEL")
        if not bel:
            unplaced += 1
            continue
        m = BEL_RE.match(bel)
        if m:
            slots[(m.group(1), m.group(2))][m.group(3)] = cell

    shared = [v for v in slots.values() if "LUT" in v and "FF" in v]
    ok = bad = 0
    for slot in shared:
        lut, ff = slot["LUT"], slot["FF"]
        i_bus = lut["connections"].get("I") or []
        i3_unused = len(i_bus) < 4 or i_bus[3] is None
        if lut["connections"].get("F") == ff["connections"].get("D") or i3_unused:
            ok += 1
        else:
            bad += 1

    ndff = sum(1 for c in cells.values() if c["type"] == "DFF")
    print(f"{path}")
    print(f"  DFFs {ndff}, slices with LUT+FF {len(shared)}, unplaced {unplaced}")
    print(f"  legal {ok}   ILLEGAL {bad}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(1 if sum(check(p) for p in sys.argv[1:]) else 0)

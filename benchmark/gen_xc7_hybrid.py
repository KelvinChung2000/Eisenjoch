# ruff: noqa: E402, F403, F405
"""Generate a hybrid chipdb: synthetic BELs + real Xilinx Series 7 routing.

Uses our standard LUT4/DFF/IOB BELs and cell types (so existing benchmarks,
packer, and placer work unchanged) but replaces the simple neighbor-hop
switch matrix with the real Xilinx Artix-7 interconnect fabric (600 wires
and 3737 PIPs per INT tile, plus long wires via tileconn nodes).

The generated .bin is cached: re-running with the same inputs skips
regeneration.  Delete the .bin (or its .hash sidecar) to force a rebuild.

Usage:
    python gen_xc7_hybrid.py <output.bin> \\
        --xray /path/to/prjxray-db/artix7 \\
        --tilegrid /path/to/prjxray-db/artix7/xc7a50t/tilegrid.json \\
        --tileconn /path/to/prjxray-db/artix7/xc7a50t/tileconn.json

Importable helper:
    from gen_xc7_hybrid import get_or_create_xc7_hybrid
    chipdb_path = get_or_create_xc7_hybrid(output_bin, xray, tilegrid, tileconn)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from os import path

CPP_NEXTPNR = "/home/kelvin/nextpnr"
BBASM = path.join(CPP_NEXTPNR, "build", "bba", "bbasm")
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(HERE)
FIXTURES_DIR = path.join(ROOT, "crates", "nextpnr", "tests", "fixtures")
CHIPDB_DIR = path.join(ROOT, "chip_database")

# Synthetic arch parameters (must match existing benchmarks)
K = 4
N = 8
N_IO = 2
N_CLK = 2


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _input_hash(script_path, xray, tilegrid_path, tileconn_path, size):
    """SHA-256 over this script + key input files + parameters."""
    h = hashlib.sha256()
    for fpath in (script_path, tilegrid_path, tileconn_path):
        with open(fpath, "rb") as f:
            h.update(f.read())
    # Include INT_L / INT_R tile type data
    for name in ("INT_L", "INT_R"):
        fp = path.join(xray, f"tile_type_{name}.json")
        if path.exists(fp):
            with open(fp, "rb") as f:
                h.update(f.read())
    h.update((size or "full").encode())
    return h.hexdigest()


def _is_cached(output_bin, expected_hash):
    """Return True if output_bin exists and its sidecar hash matches."""
    hash_path = output_bin + ".hash"
    if not path.exists(output_bin) or not path.exists(hash_path):
        return False
    with open(hash_path) as f:
        return f.read().strip() == expected_hash


def _save_hash(output_bin, content_hash):
    hash_path = output_bin + ".hash"
    with open(hash_path, "w") as f:
        f.write(content_hash + "\n")


# ---------------------------------------------------------------------------
# Tile type builders
# ---------------------------------------------------------------------------

def load_tile_type_json(xray_root, name):
    fpath = path.join(xray_root, f"tile_type_{name}.json")
    if not path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def add_int_routing(tt, int_data, prefix=""):
    """Add real Xilinx INT tile wires and PIPs to a tile type."""
    wires = int_data.get("wires", {})
    pips = int_data.get("pips", {})
    created_wires = set()

    for wname in sorted(wires.keys()):
        full = f"{prefix}{wname}"
        tt.create_wire(full, "ROUTING")
        created_wires.add(full)

    for pip_name, pdata in pips.items():
        src = f"{prefix}{pdata['src_wire']}"
        dst = f"{prefix}{pdata['dst_wire']}"
        if src in created_wires and dst in created_wires:
            tt.create_pip(src, dst, timing_class="TILE_ROUTING")

    return created_wires


# Real Xilinx BYP wire -> FF D input mapping.
# BYP_L{n} (sorted index in INT wires) -> our L{slot}_D
# This gives each FF a dedicated bypass wire separate from the LUT inputs.
REAL_BYP_MAP = {
    0: 0,   # BYP_L0 -> L0_D
    1: 4,   # BYP_L1 -> L4_D
    2: 2,   # BYP_L2 -> L2_D
    3: 6,   # BYP_L3 -> L6_D
    4: 5,   # BYP_L4 -> L5_D
    5: 1,   # BYP_L5 -> L1_D
    6: 7,   # BYP_L6 -> L7_D
    7: 3,   # BYP_L7 -> L3_D
}

# Real Xilinx xc7 CLBLL_L IMUX-to-LUT mapping (sorted IMUX indices).
# Each slice column (LUT) has 6 dedicated IMUX wires for its 6 inputs.
# Our K=4 LUT connects each input to ALL 6 IMUX in its slice, giving
# the router flexibility to avoid congestion.
REAL_IMUX_MAP = [
    [0, 2, 23, 43, 44, 47],       # L0 (slice L, col A)
    [5, 6, 8, 11, 18, 19],        # L1 (slice L, col B)
    [13, 14, 16, 24, 27, 28],     # L2 (slice L, col C)
    [30, 31, 33, 36, 37, 41],     # L3 (slice L, col D)
    [1, 3, 12, 34, 45, 46],       # L4 (slice LL, col A)
    [4, 7, 9, 10, 17, 20],        # L5 (slice LL, col B)
    [15, 21, 22, 25, 26, 29],     # L6 (slice LL, col C)
    [32, 35, 38, 39, 40, 42],     # L7 (slice LL, col D)
]

# Real Xilinx LOGIC_OUTS mapping: each LUT/FF has 3 dedicated outputs.
# {slice_idx: {"lut": idx, "ff": idx, "mux": idx}}
REAL_LO_MAP = [
    {"lut": 22, "ff": 0, "mux": 8},   # L0
    {"lut": 23, "ff": 1, "mux": 9},   # L1
    {"lut": 2, "ff": 12, "mux": 10},  # L2
    {"lut": 3, "ff": 17, "mux": 11},  # L3
    {"lut": 4, "ff": 18, "mux": 13},  # L4
    {"lut": 5, "ff": 19, "mux": 14},  # L5
    {"lut": 6, "ff": 20, "mux": 15},  # L6
    {"lut": 7, "ff": 21, "mux": 16},  # L7
]


def connect_bels_to_routing(tt, int_wires, prefix=""):
    """Connect BEL pins to INT routing using the real Xilinx IMUX/LOGIC_OUTS mapping.

    Each LUT input can reach any of the 6 IMUX wires assigned to its slice
    (real Xilinx has 6 inputs per LUT; our K=4 picks 4 of 6).
    Each LUT/FF output drives all 3 LOGIC_OUTS in its slice.
    """
    imux = sorted([w for w in int_wires if "IMUX" in w])
    logic_outs = sorted([w for w in int_wires if "LOGIC_OUTS" in w])

    if imux and len(imux) >= 48:
        for i in range(N):
            slice_imux = [imux[idx] for idx in REAL_IMUX_MAP[i]]
            for j in range(K):
                pin = f"L{i}_I{j}"
                for mux_wire in slice_imux:
                    tt.create_pip(mux_wire, pin, timing_class="SWINPUT")

    if logic_outs and len(logic_outs) >= 24:
        for i in range(N):
            lo_map = REAL_LO_MAP[i]
            # LUT output drives all 3 LOGIC_OUTS in its slice
            for idx in (lo_map["lut"], lo_map["ff"], lo_map["mux"]):
                tt.create_pip(f"L{i}_O", logic_outs[idx], timing_class="SWINPUT")
                tt.create_pip(f"L{i}_Q", logic_outs[idx], timing_class="SWINPUT")

    # FF data bypass: each FF D input gets a dedicated BYP wire from INT routing.
    # This avoids sharing the LUT I[3] wire (which caused congestion).
    # INT_L has BYP_L0..BYP_L7; INT_R has BYP0..BYP7 (no _L suffix).
    byp_wires = sorted([
        w for w in int_wires
        if (w.startswith("BYP_L") or (w.startswith("BYP") and w[3:].isdigit()))
        and "ALT" not in w and "BOUNCE" not in w
    ])
    if len(byp_wires) >= 8:
        for byp_idx, slot in REAL_BYP_MAP.items():
            if byp_idx < len(byp_wires):
                tt.create_pip(byp_wires[byp_idx], f"L{slot}_D", timing_class="SWINPUT")

    clk_wires = sorted([w for w in int_wires if "CLK" in w and "LOGIC" not in w])
    for i in range(N):
        for cw in clk_wires[: min(N_CLK, len(clk_wires))]:
            tt.create_pip(cw, f"L{i}_CLK")


def create_logic_tile(chip, int_data, suffix="LOGIC"):
    """Create LOGIC tile with synthetic BELs + real INT routing."""
    tt = chip.create_tile_type(suffix)

    for i in range(N):
        for j in range(K):
            tt.create_wire(f"L{i}_I{j}", "LUT_INPUT")
        tt.create_wire(f"L{i}_D", "FF_DATA")
        tt.create_wire(f"L{i}_O", "LUT_OUT")
        tt.create_wire(f"L{i}_Q", "FF_OUT")
        tt.create_wire(f"L{i}_CLK", "FF_CLK")

    for i in range(N):
        lut = tt.create_bel(f"L{i}_LUT", "LUT4", z=(i * 3))
        for j in range(K):
            tt.add_bel_pin(lut, f"I[{j}]", f"L{i}_I{j}", PinType.INPUT)
        tt.add_bel_pin(lut, "F", f"L{i}_O", PinType.OUTPUT)

        tt.create_pip(f"L{i}_O", f"L{i}_D")
        # In real Xilinx, the FF bypass (DI/BX/CX/DX) has its own
        # dedicated IMUX wire separate from the LUT inputs.
        # We use the BYP IMUX wires for this — connect_bels_to_routing
        # will add IMUX→D PIPs for the FF data bypass.

        ff = tt.create_bel(f"L{i}_FF", "DFF", z=(i * 3 + 1))
        tt.add_bel_pin(ff, "D", f"L{i}_D", PinType.INPUT)
        tt.add_bel_pin(ff, "CLK", f"L{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(ff, "Q", f"L{i}_Q", PinType.OUTPUT)

        latch = tt.create_bel(f"L{i}_LATCH", "DLATCH", z=(i * 3 + 2))
        tt.add_bel_pin(latch, "D", f"L{i}_D", PinType.INPUT)
        tt.add_bel_pin(latch, "G", f"L{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(latch, "Q", f"L{i}_Q", PinType.OUTPUT)

    int_wires = add_int_routing(tt, int_data)
    connect_bels_to_routing(tt, int_wires)

    # GND/VCC — connect to all BEL input pins
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    for i in range(N):
        tt.create_pip("GND", f"L{i}_CLK")
        tt.create_pip("VCC", f"L{i}_CLK")
        for j in range(K):
            tt.create_pip("GND", f"L{i}_I{j}")
            tt.create_pip("VCC", f"L{i}_I{j}")
        tt.create_pip("GND", f"L{i}_D")
        tt.create_pip("VCC", f"L{i}_D")

    # IO bridge wires: these exist in LOGIC tiles too, connected to
    # IMUX/LOGIC_OUTS via PIPs.  Nodes link them to the IO tile's
    # matching bridge wires.
    imux = sorted([w for w in int_wires if "IMUX" in w])
    logic_outs = sorted([w for w in int_wires if "LOGIC_OUTS" in w])
    for i in range(N_IO):
        tt.create_wire(f"IO_BRIDGE_IN{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_OUT{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_T{i}", "IO_BRIDGE")
        # Bridge out (from IO pad) → can reach any IMUX (drives LUT inputs)
        for mux_wire in imux:
            tt.create_pip(f"IO_BRIDGE_OUT{i}", mux_wire)
        # Any LOGIC_OUTS → bridge in (to IO pad, for driving outputs)
        for lo_wire in logic_outs:
            tt.create_pip(lo_wire, f"IO_BRIDGE_IN{i}")
            tt.create_pip(lo_wire, f"IO_BRIDGE_T{i}")

    tt.create_group("SWITCHBOX", "SWITCHBOX")
    return tt


def create_io_tile(chip, side="L"):
    """Create IO tile with dedicated bridge wires to avoid IMUX/LOGIC_OUTS aliasing.

    IO BEL pins connect to IO_BRIDGE wires via PIPs.  Separate nodes
    link IO_BRIDGE wires to matching wires in the nearest LOGIC tile,
    where PIPs connect them to IMUX/LOGIC_OUTS.  This avoids sharing
    IMUX/LOGIC_OUTS between IO and LOGIC tiles.
    """
    suffix = "IO_L" if side == "L" else "IO_R"
    tt = chip.create_tile_type(suffix)

    for i in range(N_IO):
        tt.create_wire(f"IO{i}_T", "IO_T")
        tt.create_wire(f"IO{i}_I", "IO_I")
        tt.create_wire(f"IO{i}_O", "IO_O")
        tt.create_wire(f"IO{i}_PAD", "IO_PAD")

    for i in range(N_IO):
        io = tt.create_bel(f"IO{i}", "IOB", z=i)
        tt.add_bel_pin(io, "I", f"IO{i}_I", PinType.INPUT)
        tt.add_bel_pin(io, "T", f"IO{i}_T", PinType.INPUT)
        tt.add_bel_pin(io, "O", f"IO{i}_O", PinType.OUTPUT)
        tt.add_bel_pin(io, "PAD", f"IO{i}_PAD", PinType.INOUT)

    for c in range(N_CLK):
        tt.create_wire(f"GCLK{c}_OUT", "GCLK")
        if c < N_IO:
            tt.create_pip(f"IO{c}_O", f"GCLK{c}_OUT")

    # Dedicated bridge wires — these will be linked to LOGIC tiles via nodes.
    # IO_BRIDGE_IN{i}: signals entering from fabric to IO pad (output pins)
    # IO_BRIDGE_OUT{i}: signals leaving IO pad into fabric (input pins)
    for i in range(N_IO):
        tt.create_wire(f"IO_BRIDGE_IN{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_OUT{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_T{i}", "IO_BRIDGE")
        # BEL output (pad→fabric) → bridge out
        tt.create_pip(f"IO{i}_O", f"IO_BRIDGE_OUT{i}")
        # Bridge in → BEL input (fabric→pad)
        tt.create_pip(f"IO_BRIDGE_IN{i}", f"IO{i}_I")
        # Bridge T → BEL tristate
        tt.create_pip(f"IO_BRIDGE_T{i}", f"IO{i}_T")

    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    # GND/VCC for IO T pin (always drive / always tristate)
    for i in range(N_IO):
        tt.create_pip("GND", f"IO{i}_T")
        tt.create_pip("VCC", f"IO{i}_T")
        tt.create_pip("GND", f"IO{i}_I")
        tt.create_pip("VCC", f"IO{i}_I")
    tt.create_group("SWITCHBOX", "SWITCHBOX")
    return tt


def create_null_tile(chip):
    tt = chip.create_tile_type("NULL")
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    gnd = tt.create_bel("GND_DRV", "GND_DRV", z=0)
    tt.add_bel_pin(gnd, "GND", "GND", PinType.OUTPUT)
    vcc = tt.create_bel("VCC_DRV", "VCC_DRV", z=1)
    tt.add_bel_pin(vcc, "VCC", "VCC", PinType.OUTPUT)
    return tt


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_xc7_hybrid(output_bba, xray, tilegrid_path, tileconn_path, size=None):
    """Generate the hybrid chipdb BBA file. Returns the BBA path."""
    int_l_data = load_tile_type_json(xray, "INT_L")
    int_r_data = load_tile_type_json(xray, "INT_R")
    print(
        f"INT_L: {len(int_l_data['wires'])} wires, {len(int_l_data['pips'])} PIPs; "
        f"INT_R: {len(int_r_data['wires'])} wires, {len(int_r_data['pips'])} PIPs"
    )

    with open(tilegrid_path) as f:
        tilegrid = json.load(f)
    with open(tileconn_path) as f:
        tileconn = json.load(f)

    if size:
        parts = size.lower().split("x")
        X = int(parts[0])
        Y = int(parts[1]) if len(parts) > 1 else X
    else:
        max_x = max_y = 0
        for tdata in tilegrid.values():
            max_x = max(max_x, tdata["grid_x"])
            max_y = max(max_y, tdata["grid_y"])
        X = max_x + 1
        Y = max_y + 1

    print(f"Grid: {X}x{Y}")

    xc7_grid = {}
    for tname, tdata in tilegrid.items():
        x, y = tdata["grid_x"], tdata["grid_y"]
        xc7_grid[(x, y)] = tdata["type"]

    int_types = {"INT_L", "INT_R"}
    io_types = {"LIOB33", "RIOB33", "LIOB33_SING", "RIOB33_SING"}

    ch = Chip("xc7_hybrid", "XC7A50T", X, Y)
    ch.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    ch.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))

    create_logic_tile(ch, int_l_data, suffix="LOGIC_L")
    create_logic_tile(ch, int_r_data, suffix="LOGIC_R")
    create_io_tile(ch, side="L")
    create_io_tile(ch, side="R")
    create_null_tile(ch)

    left_io = {"LIOB33", "LIOB33_SING"}
    right_io = {"RIOB33", "RIOB33_SING"}

    n_logic = n_io = n_null = 0
    for y in range(Y):
        for x in range(X):
            xc7_type = xc7_grid.get((x, y), "NULL")
            if xc7_type == "INT_L":
                ch.set_tile_type(x, y, "LOGIC_L")
                n_logic += 1
            elif xc7_type == "INT_R":
                ch.set_tile_type(x, y, "LOGIC_R")
                n_logic += 1
            elif xc7_type in left_io:
                ch.set_tile_type(x, y, "IO_L")
                n_io += 1
            elif xc7_type in right_io:
                ch.set_tile_type(x, y, "IO_R")
                n_io += 1
            else:
                ch.set_tile_type(x, y, "NULL")
                n_null += 1

    print(f"Tile assignment: {n_logic} LOGIC, {n_io} IO, {n_null} NULL")
    print(f"Capacity: ~{n_logic * N} LUT4s, ~{n_logic * N} DFFs")

    # ------------------------------------------------------------------
    # Inter-tile routing nodes from tileconn
    # ------------------------------------------------------------------
    print("Creating inter-tile routing nodes from tileconn...")

    tiles_by_xc7_type = defaultdict(list)
    for (x, y), ttype in xc7_grid.items():
        tiles_by_xc7_type[ttype].append((x, y))

    wire_set_cache = {}

    def get_int_wire_set(ttype):
        if ttype not in wire_set_cache:
            data = load_tile_type_json(xray, ttype)
            wire_set_cache[ttype] = set(data["wires"].keys()) if data else set()
        return wire_set_cache[ttype]

    int_l_wires = set(int_l_data["wires"].keys())
    int_r_wires = set(int_r_data["wires"].keys())

    wire_sets = {
        "INT_L": int_l_wires,
        "INT_R": int_r_wires,
    }
    # IO tiles use bridge wires (not INT wires), so they don't participate
    # in the tileconn BFS.  IO-to-INT connectivity is handled separately
    # via bridge nodes.

    # Only INT tiles participate in the tileconn BFS.
    modeled_types = int_types

    # Build tileconn adjacency
    tc_adj = defaultdict(list)
    for conn in tileconn:
        a_type = conn["tile_types"][0]
        b_type = conn["tile_types"][1]
        dx = conn.get("grid_deltas", [0, 0])[0]
        dy = conn.get("grid_deltas", [0, 0])[1]
        for wa, wb in conn["wire_pairs"]:
            tc_adj[(a_type, wa)].append((b_type, wb, dx, dy))
            tc_adj[(b_type, wb)].append((a_type, wa, -dx, -dy))

    # Union-Find to group wire instances that are on the same physical net.
    # Each (x, y, wire_name) at a modeled tile is a node member.
    # Tileconn entries define which members should be in the same group.
    wire_parent = {}  # (x,y,wire) -> parent (x,y,wire)

    def uf_find(key):
        while wire_parent[key] != key:
            wire_parent[key] = wire_parent[wire_parent[key]]
            key = wire_parent[key]
        return key

    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            wire_parent[ra] = rb

    # Initialize: each modeled tile wire is its own parent
    for tile_type in sorted(modeled_types):
        tile_wires = wire_sets.get(tile_type, set())
        for tx, ty in tiles_by_xc7_type.get(tile_type, []):
            if 0 <= tx < X and 0 <= ty < Y:
                for w in tile_wires:
                    wire_parent[(tx, ty, w)] = (tx, ty, w)

    # Process tileconn: for each concrete wire pair, union them if both
    # endpoints land on modeled tiles.  BFS through intermediate tiles.
    processed_starts = set()

    for start_type in sorted(modeled_types):
        start_wires = wire_sets.get(start_type, set())
        for sx, sy in tiles_by_xc7_type.get(start_type, []):
            if not (0 <= sx < X and 0 <= sy < Y):
                continue
            for start_wire in start_wires:
                start_key = (sx, sy, start_wire)
                if start_key in processed_starts:
                    continue
                processed_starts.add(start_key)

                # BFS to find all reachable modeled-tile wire instances
                frontier = [(sx, sy, start_type, start_wire)]
                visited = {(sx, sy, start_wire)}

                while frontier:
                    next_frontier = []
                    for cx, cy, ctype, cwire in frontier:
                        for ntype, nwire, ddx, ddy in tc_adj[(ctype, cwire)]:
                            nx, ny = cx + ddx, cy + ddy
                            nkey = (nx, ny, nwire)
                            if nkey in visited:
                                continue
                            if xc7_grid.get((nx, ny)) != ntype:
                                continue
                            visited.add(nkey)
                            if ntype in modeled_types and nwire in wire_sets.get(ntype, set()):
                                # Union this endpoint with the start
                                uf_union(start_key, nkey)
                                processed_starts.add(nkey)
                            next_frontier.append((nx, ny, ntype, nwire))
                    frontier = next_frontier

    # Collect groups from union-find
    groups = defaultdict(list)
    for key in wire_parent:
        root = uf_find(key)
        groups[root].append(key)

    # Create nodes for groups with 2+ members
    wires_in_nodes = set()
    node_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        ch.add_node([NodeWire(x, y, w) for x, y, w in members])
        for m in members:
            wires_in_nodes.add(m)
        node_count += 1

    print(f"Created {node_count} inter-tile routing nodes")

    # ------------------------------------------------------------------
    # IO-to-INT bridge nodes
    # ------------------------------------------------------------------
    # Link each IO tile's IO_BRIDGE wires to the matching bridge wires
    # in the nearest LOGIC tile.  The bridge wires are separate from
    # IMUX/LOGIC_OUTS, so IO routing doesn't create congestion with
    # logic routing.
    print("Creating IO-to-INT bridge nodes...")
    io_node_count = 0

    # Pre-index INT tile positions for fast lookup
    int_positions = {}
    for itype in ("INT_L", "INT_R"):
        for ix, iy in tiles_by_xc7_type.get(itype, []):
            int_positions.setdefault(iy, []).append((ix, iy, itype))

    for io_type in sorted(io_types):
        is_left = io_type in left_io
        target_int = "INT_L" if is_left else "INT_R"

        for io_x, io_y in tiles_by_xc7_type.get(io_type, []):
            if not (0 <= io_x < X and 0 <= io_y < Y):
                continue

            # Find nearest INT tile of the right type
            best_int = None
            best_dist = 999
            for dy in range(0, 4):
                for sign in (0, 1, -1):
                    row = io_y + dy * sign if dy > 0 else io_y
                    for ix, iy, itype in int_positions.get(row, []):
                        if itype == target_int and abs(ix - io_x) < best_dist:
                            best_dist = abs(ix - io_x)
                            best_int = (ix, iy)
                if best_int is not None:
                    break
            if best_int is None:
                continue

            int_x, int_y = best_int
            # Create node for each bridge wire
            for i in range(N_IO):
                for bname in (f"IO_BRIDGE_IN{i}", f"IO_BRIDGE_OUT{i}", f"IO_BRIDGE_T{i}"):
                    io_key = (io_x, io_y, bname)
                    int_key = (int_x, int_y, bname)
                    if io_key in wires_in_nodes or int_key in wires_in_nodes:
                        continue
                    wires_in_nodes.add(io_key)
                    wires_in_nodes.add(int_key)
                    ch.add_node([NodeWire(io_x, io_y, bname), NodeWire(int_x, int_y, bname)])
                    io_node_count += 1

    print(f"Created {io_node_count} IO-to-INT bridge nodes")

    # ------------------------------------------------------------------
    # Clock distribution
    # ------------------------------------------------------------------
    # Build a simple global clock network: for each clock, create a
    # vertical chain of nodes connecting CLK wires in every INT column,
    # plus horizontal spines so clocks can reach all columns.
    #
    # Pick CLK wires that exist in INT_L — filter for actual clock wires.
    int_columns = defaultdict(list)
    for (x, y), ttype in xc7_grid.items():
        if ttype in int_types:
            int_columns[x].append(y)
    for x in int_columns:
        int_columns[x].sort()

    # Find clock-related wires in INT_L
    clk_int_wires = sorted([w for w in int_l_wires if "CLK" in w and "LOGIC" not in w])
    if not clk_int_wires:
        clk_int_wires = sorted([w for w in int_l_wires if "GCLK" in w])

    # Use the first N_CLK clock wires for our clock networks
    clk_wire_names = clk_int_wires[:N_CLK] if clk_int_wires else []
    print(f"Clock wires: {clk_wire_names}")

    # Vertical clock chains: connect adjacent INT_L tiles in each INT_L column.
    # Only use INT_L columns since CLK_L* wires exist in INT_L/LOGIC_L tiles.
    clk_node_count = 0
    all_int_cols = sorted(int_columns.keys())
    int_l_columns = defaultdict(list)
    for (x, y), ttype in xc7_grid.items():
        if ttype == "INT_L":
            int_l_columns[x].append(y)
    for x in int_l_columns:
        int_l_columns[x].sort()

    for c, cwire in enumerate(clk_wire_names):
        for col_x in sorted(int_l_columns.keys()):
            ys = int_l_columns[col_x]
            for i in range(len(ys) - 1):
                y0, y1 = ys[i], ys[i + 1]
                key0 = (col_x, y0, cwire)
                key1 = (col_x, y1, cwire)
                if key0 not in wires_in_nodes and key1 not in wires_in_nodes:
                    wires_in_nodes.add(key0)
                    wires_in_nodes.add(key1)
                    ch.add_node([NodeWire(col_x, y0, cwire), NodeWire(col_x, y1, cwire)])
                    clk_node_count += 1

    # Horizontal clock spines: connect CLK wires across INT_L columns
    SPINE_INTERVAL = 10
    int_l_col_list = sorted(int_l_columns.keys())
    for c, cwire in enumerate(clk_wire_names):
        for col_idx in range(len(int_l_col_list) - 1):
            x0 = int_l_col_list[col_idx]
            x1 = int_l_col_list[col_idx + 1]
            common = sorted(set(int_l_columns[x0]) & set(int_l_columns[x1]))
            for y in common[::SPINE_INTERVAL]:
                key0 = (x0, y, cwire)
                key1 = (x1, y, cwire)
                if key0 not in wires_in_nodes and key1 not in wires_in_nodes:
                    wires_in_nodes.add(key0)
                    wires_in_nodes.add(key1)
                    ch.add_node([NodeWire(x0, y, cwire), NodeWire(x1, y, cwire)])
                    clk_node_count += 1

    # Connect IO clock outputs to the nearest INT_L clock wire
    io_tiles = (tiles_by_xc7_type.get("LIOB33", [])
                + tiles_by_xc7_type.get("RIOB33", [])
                + tiles_by_xc7_type.get("LIOB33_SING", [])
                + tiles_by_xc7_type.get("RIOB33_SING", []))
    for c in range(min(N_CLK, len(clk_wire_names), N_IO)):
        for io_x, io_y in io_tiles:
            if not (0 <= io_x < X and 0 <= io_y < Y):
                continue
            for col_x in int_l_col_list:
                if abs(col_x - io_x) <= 4:
                    ys = int_l_columns[col_x]
                    closest_y = min(ys, key=lambda y: abs(y - io_y))
                    gclk_key = (io_x, io_y, f"GCLK{c}_OUT")
                    clk_key = (col_x, closest_y, clk_wire_names[c])
                    if gclk_key not in wires_in_nodes and clk_key not in wires_in_nodes:
                        wires_in_nodes.add(gclk_key)
                        wires_in_nodes.add(clk_key)
                        ch.add_node([
                            NodeWire(io_x, io_y, f"GCLK{c}_OUT"),
                            NodeWire(col_x, closest_y, clk_wire_names[c]),
                        ])
                        clk_node_count += 1
                    break
            break  # Only one IO per clock

    print(f"Created {clk_node_count} clock routing nodes")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    speed = "DEFAULT"
    tmg = ch.set_speed_grades([speed])
    tmg.set_pip_class(
        grade=speed,
        name="SWINPUT",
        delay=TimingValue(80),
        in_cap=TimingValue(5000),
        out_res=TimingValue(1000),
    )
    tmg.set_pip_class(
        grade=speed,
        name="TILE_ROUTING",
        delay=TimingValue(100),
        in_cap=TimingValue(5000),
        out_res=TimingValue(1000),
    )

    ch.strs.known_id_count = 0
    ch.write_bba(output_bba)
    print(f"Wrote {output_bba}")


# ---------------------------------------------------------------------------
# Public API: get-or-create with caching
# ---------------------------------------------------------------------------

def get_or_create_xc7_hybrid(output_bin, xray, tilegrid, tileconn, size=None):
    """Return path to a cached .bin, regenerating only if inputs changed."""
    content_hash = _input_hash(__file__, xray, tilegrid, tileconn, size)

    if _is_cached(output_bin, content_hash):
        size_mb = os.path.getsize(output_bin) / (1024 * 1024)
        print(f"Using cached XC7 hybrid chipdb: {output_bin} ({size_mb:.1f} MB)")
        return output_bin

    print("Generating XC7 hybrid chipdb (this is cached for future runs)...")
    t0 = time.time()

    bba_path = output_bin.replace(".bin", ".bba")
    generate_xc7_hybrid(bba_path, xray, tilegrid, tileconn, size)

    print("Assembling binary...")
    subprocess.run([BBASM, "--le", bba_path, output_bin], check=True)

    os.remove(bba_path)
    _save_hash(output_bin, content_hash)

    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"XC7 hybrid chipdb: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return output_bin


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid chipdb: synthetic BELs + real XC7 routing"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=path.join(CHIPDB_DIR, "xc7_hybrid.bin"),
        help="Output .bin file path (default: chip_database/xc7_hybrid.bin)",
    )
    parser.add_argument("--xray", required=True, help="Path to prjxray-db/artix7")
    parser.add_argument("--tilegrid", required=True, help="Path to tilegrid.json")
    parser.add_argument("--tileconn", required=True, help="Path to tileconn.json")
    parser.add_argument(
        "--size",
        default=None,
        help="Override grid size NxN (default: use tilegrid dimensions)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration even if cached"
    )
    args = parser.parse_args()

    output_bin = args.output
    if output_bin.endswith(".bba"):
        output_bin = output_bin.replace(".bba", ".bin")

    if args.force:
        for f in (output_bin, output_bin + ".hash"):
            if path.exists(f):
                os.remove(f)

    get_or_create_xc7_hybrid(output_bin, args.xray, args.tilegrid, args.tileconn, args.size)


if __name__ == "__main__":
    main()

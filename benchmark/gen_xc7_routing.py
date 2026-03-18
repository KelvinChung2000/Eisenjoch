# ruff: noqa: E402, F403, F405
"""Generate a chipdb with real Xilinx Series 7 routing but simplified BELs.

Reads the xc7a50t routing graph from Project X-Ray and overlays simple
LUT4/DFF/IOB BELs connected to the INT tile wires. This gives us
a realistic routing fabric to benchmark against while using our
existing synthetic-arch packer and placer.

Usage:
    python gen_xc7_routing.py <output.bba> \\
        --xray /path/to/prjxray-db/artix7 \\
        --tileconn /path/to/prjxray-db/artix7/xc7a50t/tileconn.json \\
        --tilegrid /path/to/prjxray-db/artix7/xc7a50t/tilegrid.json

The output uses our synthetic arch cell types (LUT4, DFF, IOB) so
existing benchmarks work without modification.
"""

import argparse
import json
import sys
from collections import defaultdict
from os import path

CPP_NEXTPNR = "/home/kelvin/nextpnr"
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

FIXTURES_DIR = path.join(
    path.dirname(path.abspath(__file__)),
    "..",
    "crates",
    "nextpnr",
    "tests",
    "fixtures",
)

# Our synthetic arch parameters
K = 4  # LUT inputs
N = 8  # SLICEs per logic tile
N_IO = 2
N_CLK = 2


def load_tile_type_data(xray_root, tile_type_name):
    """Load wire and PIP data for a tile type from prjxray JSON."""
    fpath = path.join(xray_root, f"tile_type_{tile_type_name}.json")
    if not path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def create_routing_tile(
    chip, tt_name, xray_data, add_logic_bels=False, add_io_bels=False
):
    """Create a tile type with real Xilinx routing + optional simplified BELs."""
    tt = chip.create_tile_type(tt_name)

    wires = xray_data.get("wires", {})
    pips = xray_data.get("pips", {})

    # Create all wires
    wire_names = sorted(wires.keys())
    for wname in wire_names:
        wire_type = "ROUTING"
        if "GND" in wname:
            tt.create_wire(wname, "GND", const_value="GND")
        elif "VCC" in wname:
            tt.create_wire(wname, "VCC", const_value="VCC")
        else:
            tt.create_wire(wname, wire_type)

    # Create all PIPs with timing
    for pip_name, pdata in pips.items():
        src = pdata["src_wire"]
        dst = pdata["dst_wire"]
        if src in wires and dst in wires:
            tt.create_pip(src, dst, timing_class="TILE_ROUTING")

    if add_logic_bels:
        # In Xilinx CLB tiles, the SLICE site wires follow a pattern:
        # Inputs:  PREFIX_A1..A6, PREFIX_B1..B6, PREFIX_C1..C6, PREFIX_D1..D6
        # Outputs: PREFIX_A (LUT out), PREFIX_AQ (FF out), PREFIX_AMUX (mux out)
        # Clock:   PREFIX_CLK
        # The CLB has 2 SLICEs (L and LL, or L and M).
        # We detect the prefixes from existing wires.

        # Find SLICE prefixes (e.g. "CLBLL_L", "CLBLL_LL")
        slice_prefixes = set()
        for w in wire_names:
            for letter in "ABCD":
                suffix = f"_{letter}1"
                if w.endswith(suffix):
                    prefix = w[: -len(suffix)]
                    # Verify this prefix has full set
                    if f"{prefix}_{letter}" in wires and f"{prefix}_{letter}Q" in wires:
                        slice_prefixes.add(prefix)
        slice_prefixes = sorted(slice_prefixes)

        slot = 0
        for sp in slice_prefixes:
            for letter in "ABCD":
                lut_in_wires = [
                    f"{sp}_{letter}{j}"
                    for j in range(1, K + 1)
                    if f"{sp}_{letter}{j}" in wires
                ]
                lut_out_wire = f"{sp}_{letter}"
                ff_out_wire = f"{sp}_{letter}Q"
                clk_wire = f"{sp}_CLK"

                if lut_out_wire not in wires or ff_out_wire not in wires:
                    continue
                if len(lut_in_wires) < K:
                    continue

                # LUT BEL: use existing SLICE wires directly
                lut_name = f"{sp}_{letter}_LUT"
                lut = tt.create_bel(lut_name, "LUT4", z=slot)
                for j, iw in enumerate(lut_in_wires[:K]):
                    tt.add_bel_pin(lut, f"I[{j}]", iw, PinType.INPUT)
                tt.add_bel_pin(lut, "F", lut_out_wire, PinType.OUTPUT)

                # DFF BEL
                ff_name = f"{sp}_{letter}_FF"
                # Create a data wire for LUT->FF connection
                ff_d_wire = f"{sp}_{letter}_FFDATA"
                tt.create_wire(ff_d_wire, "FF_DATA")
                tt.create_pip(lut_out_wire, ff_d_wire)

                ff = tt.create_bel(ff_name, "DFF", z=slot + 1)
                tt.add_bel_pin(ff, "D", ff_d_wire, PinType.INPUT)
                tt.add_bel_pin(ff, "Q", ff_out_wire, PinType.OUTPUT)
                if clk_wire in wires:
                    tt.add_bel_pin(ff, "CLK", clk_wire, PinType.INPUT)

                slot += 2

    if add_io_bels:
        # Simple IO BELs connected to routing
        logic_outs = sorted([w for w in wire_names if "LOGIC_OUTS" in w])
        imux_wires = sorted([w for w in wire_names if "IMUX" in w])
        for i in range(N_IO):
            io_i = f"IO{i}_I"
            io_o = f"IO{i}_O"
            io_pad = f"IO{i}_PAD"
            tt.create_wire(io_i, "IO_I")
            tt.create_wire(io_o, "IO_O")
            tt.create_wire(io_pad, "IO_PAD")
            # Connect IO to routing
            if i < len(imux_wires):
                tt.create_pip(imux_wires[i], io_i)
            if i < len(logic_outs):
                tt.create_pip(io_o, logic_outs[i])

            io = tt.create_bel(f"IO{i}", "IOB", z=i)
            tt.add_bel_pin(io, "I", io_i, PinType.INPUT)
            tt.add_bel_pin(io, "O", io_o, PinType.OUTPUT)
            tt.add_bel_pin(io, "PAD", io_pad, PinType.INOUT)

    return tt


def main():
    parser = argparse.ArgumentParser(
        description="Generate chipdb with real XC7 routing + simple BELs"
    )
    parser.add_argument("output", help="Output .bba file path")
    parser.add_argument("--xray", required=True, help="Path to prjxray-db/artix7")
    parser.add_argument("--tilegrid", required=True, help="Path to tilegrid.json")
    parser.add_argument("--tileconn", required=True, help="Path to tileconn.json")
    args = parser.parse_args()

    # Load tilegrid
    with open(args.tilegrid) as f:
        tilegrid = json.load(f)

    # Load tileconn
    with open(args.tileconn) as f:
        tileconn = json.load(f)

    # Determine grid dimensions
    max_x = max_y = 0
    for tdata in tilegrid.values():
        max_x = max(max_x, tdata["grid_x"])
        max_y = max(max_y, tdata["grid_y"])
    X = max_x + 1
    Y = max_y + 1
    print(f"Grid: {X}x{Y}")

    ch = Chip("xc7_routing", "XC7A50T", X, Y)
    ch.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    ch.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))

    # Classify tile types: which get logic BELs, which get IO BELs
    # CLB tiles (CLBLL_L, CLBLL_R, CLBLM_L, CLBLM_R) -> logic BELs
    # IO tiles (LIOB33, RIOB33, etc) -> IO BELs
    # INT tiles -> routing only (no BELs, but create routing)
    # Others -> NULL (routing passthrough or skip)

    # Collect unique tile types used
    tile_types_used = set()
    tile_grid = {}  # (x,y) -> (tile_name, tile_type)
    for tname, tdata in tilegrid.items():
        x, y = tdata["grid_x"], tdata["grid_y"]
        tt = tdata["type"]
        tile_types_used.add(tt)
        tile_grid[(x, y)] = (tname, tt)

    print(f"Unique tile types: {len(tile_types_used)}")

    # Create tile types
    created_types = {}
    clb_types = {"CLBLL_L", "CLBLL_R", "CLBLM_L", "CLBLM_R"}
    io_types = {
        "LIOB33",
        "RIOB33",
        "LIOB33_SING",
        "RIOB33_SING",
        "LIOI3",
        "RIOI3",
        "LIOI3_SING",
        "RIOI3_SING",
        "LIOI3_TBYTESRC",
        "LIOI3_TBYTETERM",
        "RIOI3_TBYTESRC",
        "RIOI3_TBYTETERM",
    }
    int_types = {
        "INT_L",
        "INT_R",
        "INT_INTERFACE_L",
        "INT_INTERFACE_R",
        "INT_FEEDTHRU_1",
        "INT_FEEDTHRU_2",
        "BRAM_INT_INTERFACE_L",
        "BRAM_INT_INTERFACE_R",
        "IO_INT_INTERFACE_L",
        "IO_INT_INTERFACE_R",
    }

    # Create a NULL tile type for tiles we skip
    null_tt = ch.create_tile_type("NULL")
    null_tt.create_wire("GND", "GND", const_value="GND")
    null_tt.create_wire("VCC", "VCC", const_value="VCC")
    gnd = null_tt.create_bel("GND_DRV", "GND_DRV", z=0)
    null_tt.add_bel_pin(gnd, "GND", "GND", PinType.OUTPUT)
    vcc = null_tt.create_bel("VCC_DRV", "VCC_DRV", z=1)
    null_tt.add_bel_pin(vcc, "VCC", "VCC", PinType.OUTPUT)
    created_types["NULL"] = null_tt

    for tt_name in sorted(tile_types_used):
        xray_data = load_tile_type_data(args.xray, tt_name)
        if xray_data is None:
            continue

        is_clb = tt_name in clb_types
        is_io = tt_name in io_types
        is_int = tt_name in int_types

        if is_clb or is_int or is_io:
            created_types[tt_name] = create_routing_tile(
                ch,
                tt_name,
                xray_data,
                add_logic_bels=is_clb,
                add_io_bels=is_io,
            )
            n_wires = len(xray_data.get("wires", {}))
            n_pips = len(xray_data.get("pips", {}))
            label = "CLB" if is_clb else ("IO" if is_io else "INT")
            print(f"  {tt_name} [{label}]: {n_wires} wires, {n_pips} pips")

    # Assign tile types to grid positions
    n_logic = n_io = n_int = n_null = 0
    for y in range(Y):
        for x in range(X):
            if (x, y) in tile_grid:
                _, tt = tile_grid[(x, y)]
                if tt in created_types:
                    ch.set_tile_type(x, y, tt)
                    if tt in clb_types:
                        n_logic += 1
                    elif tt in io_types:
                        n_io += 1
                    elif tt in int_types:
                        n_int += 1
                    else:
                        n_null += 1
                else:
                    ch.set_tile_type(x, y, "NULL")
                    n_null += 1
            else:
                ch.set_tile_type(x, y, "NULL")
                n_null += 1

    print(f"\nGrid assignment: {n_logic} CLB, {n_io} IO, {n_int} INT, {n_null} NULL")

    # Create inter-tile routing nodes from tileconn
    print("Creating inter-tile routing nodes...")

    # Build index: tile_type -> list of (x, y)
    tiles_by_type = defaultdict(list)
    for (x, y), (tname, ttype) in tile_grid.items():
        tiles_by_type[ttype].append((x, y))

    # Cache tile type wire sets
    wire_sets_cache = {}

    def get_wire_set(ttype):
        if ttype not in wire_sets_cache:
            data = load_tile_type_data(args.xray, ttype)
            wire_sets_cache[ttype] = (
                set(data.get("wires", {}).keys()) if data else set()
            )
        return wire_sets_cache[ttype]

    # Track which (x, y, wire) are already in a node to avoid duplicates
    wires_in_nodes = set()
    node_count = 0
    for conn in tileconn:
        tile_a_type = conn["tile_types"][0]
        tile_b_type = conn["tile_types"][1]
        if tile_a_type not in created_types or tile_b_type not in created_types:
            continue

        a_wires = get_wire_set(tile_a_type)
        b_wires = get_wire_set(tile_b_type)
        dx = conn.get("grid_deltas", [0, 0])[0]
        dy = conn.get("grid_deltas", [0, 0])[1]

        # Filter wire pairs to those that exist in both tile types
        valid_pairs = [
            (wp[0], wp[1])
            for wp in conn["wire_pairs"]
            if wp[0] in a_wires and wp[1] in b_wires
        ]
        if not valid_pairs:
            continue

        # For each tile of type A, check if partner exists
        for ax, ay in tiles_by_type.get(tile_a_type, []):
            bx, by = ax + dx, ay + dy
            if (bx, by) not in tile_grid:
                continue
            _, btype = tile_grid[(bx, by)]
            if btype != tile_b_type:
                continue

            for wire_a, wire_b in valid_pairs:
                key_a = (ax, ay, wire_a)
                key_b = (bx, by, wire_b)
                if key_a in wires_in_nodes or key_b in wires_in_nodes:
                    continue
                wires_in_nodes.add(key_a)
                wires_in_nodes.add(key_b)
                ch.add_node(
                    [
                        NodeWire(ax, ay, wire_a),
                        NodeWire(bx, by, wire_b),
                    ]
                )
                node_count += 1

        if node_count % 100000 == 0 and node_count > 0:
            print(f"  {node_count} nodes...")

    print(f"Created {node_count} inter-tile routing nodes")

    # Set timing
    speed = "DEFAULT"
    tmg = ch.set_speed_grades([speed])
    tmg.set_pip_class(
        grade=speed,
        name="TILE_ROUTING",
        delay=TimingValue(100),
        in_cap=TimingValue(5000),
        out_res=TimingValue(1000),
    )

    ch.strs.known_id_count = 0
    ch.write_bba(args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

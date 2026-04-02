# ruff: noqa: E402, F403, F405
"""Generate a synthetic XC7-like chipdb with exact target resource counts.

The fabric is column-organized with a repeating style similar to:
  CLB_L, INT_L, INT_R, CLB_R, ...

BRAM and DSP columns are distributed across the interior. This is intentionally
synthetic: it preserves the broad XC7-style column organization and resource
mix, but it is not tied to a specific real Xilinx device.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from os import path

CPP_NEXTPNR = "/home/kelvin/nextpnr"
BBASM = path.join(CPP_NEXTPNR, "build", "bba", "bbasm")
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(path.dirname(HERE)))
FIXTURES_DIR = path.join(ROOT, "crates", "nextpnr", "tests", "fixtures")
CHIPDB_DIR = path.join(ROOT, "chip_database")

TARGET_LUT = 538_000
TARGET_FF = 1_075_000
TARGET_BRAM = 1_728
TARGET_DSP = 768

K = 6
LUTS_PER_LOGIC_TILE = 8
FULL_FFS_PER_LOGIC_TILE = 16
TRIMMED_FFS_PER_LOGIC_TILE = 14
TRIMMED_LOGIC_TILE_COUNT = 500
N_IO = 2
N_CLK = 2
Wl = 192
Si = 6
Sq = 6

INTERIOR_ROWS = 269
LOGIC_TILES = TARGET_LUT // LUTS_PER_LOGIC_TILE  # 67250
LOGIC_COLS = LOGIC_TILES // INTERIOR_ROWS  # 250
LOGIC_PAIR_COUNT = LOGIC_COLS // 2  # 125
BRAM_FULL_COLS = TARGET_BRAM // INTERIOR_ROWS  # 6
BRAM_PARTIAL_ROWS = TARGET_BRAM % INTERIOR_ROWS  # 114
DSP_FULL_COLS = TARGET_DSP // INTERIOR_ROWS  # 2
DSP_PARTIAL_ROWS = TARGET_DSP % INTERIOR_ROWS  # 230

dirs = [
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
]

RESOURCE_CAPACITY = {
    "CLB_L": {"LUT6": LUTS_PER_LOGIC_TILE, "DFF": FULL_FFS_PER_LOGIC_TILE},
    "CLB_R": {"LUT6": LUTS_PER_LOGIC_TILE, "DFF": FULL_FFS_PER_LOGIC_TILE},
    "CLB_L_TRIM": {"LUT6": LUTS_PER_LOGIC_TILE, "DFF": TRIMMED_FFS_PER_LOGIC_TILE},
    "CLB_R_TRIM": {"LUT6": LUTS_PER_LOGIC_TILE, "DFF": TRIMMED_FFS_PER_LOGIC_TILE},
    "BRAM": {"RAMB36E1": 1},
    "DSP": {"DSP48E1": 1},
}


def create_switch_matrix(tt, inputs, outputs):
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    for i in range(Wl):
        tt.create_wire(f"SWITCH{i}", "SWITCH")
    for i in range(Wl):
        for d, dx, dy in dirs:
            tt.create_wire(f"{d}{i}", f"NEIGH_{d}")
    for i, w in enumerate(inputs):
        for j in range((i % Si), Wl, Si):
            tt.create_pip(f"SWITCH{j}", w, timing_class="SWINPUT")
    for i, w in enumerate(outputs):
        for j in range((i % Sq), Wl, Sq):
            tt.create_pip(w, f"SWITCH{j}", timing_class="SWOUTPUT")
    for i in range(Wl):
        tt.create_pip("GND", f"SWITCH{i}")
        tt.create_pip("VCC", f"SWITCH{i}")
    for i in range(Wl):
        for j, (d, dx, dy) in enumerate(dirs):
            tt.create_pip(f"{d}{(i + j) % Wl}", f"SWITCH{i}", timing_class="SWNEIGH")
            tt.create_pip(f"SWITCH{i}", f"{d}{(i + j) % Wl}", timing_class="SWNEIGH")
    for c in range(N_CLK):
        if not tt.has_wire(f"CLK{c}"):
            tt.create_wire(f"CLK{c}", "TILE_CLK")
        tt.create_wire(f"CLK{c}_PREV", "CLK_ROUTE")
        tt.create_pip(f"CLK{c}_PREV", f"CLK{c}")
        for j in range(0, Wl, Si):
            tt.create_pip(f"SWITCH{j}", f"CLK{c}")
        tt.create_pip(f"CLK{c}", f"SWITCH{c}")
    tt.create_group("SWITCHBOX", "SWITCHBOX")


def create_int_tiletype(chip, name):
    tt = chip.create_tile_type(name)
    create_switch_matrix(tt, [], [])
    return tt


def create_logic_tiletype(chip, name, ff_count):
    tt = chip.create_tile_type(name)
    inputs = []
    outputs = []
    for i in range(LUTS_PER_LOGIC_TILE):
        for j in range(K):
            wire = f"L{i}_I{j}"
            inputs.append(wire)
            tt.create_wire(wire, "LUT_INPUT")
        tt.create_wire(f"L{i}_O", "LUT_OUT")
        outputs.append(f"L{i}_O")

    for i in range(ff_count):
        tt.create_wire(f"FF{i}_D", "FF_DATA")
        tt.create_wire(f"FF{i}_Q", "FF_OUT")
        tt.create_wire(f"FF{i}_CLK", "FF_CLK")
        outputs.append(f"FF{i}_Q")

    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")

    for i in range(LUTS_PER_LOGIC_TILE):
        lut = tt.create_bel(f"L{i}_LUT", "LUT6", z=(i * 3))
        for j in range(K):
            tt.add_bel_pin(lut, f"I[{j}]", f"L{i}_I{j}", PinType.INPUT)
        tt.add_bel_pin(lut, "F", f"L{i}_O", PinType.OUTPUT)

    for i in range(ff_count):
        src = i % LUTS_PER_LOGIC_TILE
        for c in range(N_CLK):
            tt.create_pip(f"CLK{c}", f"FF{i}_CLK")
        tt.create_pip(f"L{src}_O", f"FF{i}_D")
        tt.create_pip(f"L{src}_I{K - 1}", f"FF{i}_D")
        ff = tt.create_bel(f"FF{i}", "DFF", z=(LUTS_PER_LOGIC_TILE * 3 + i))
        tt.add_bel_pin(ff, "D", f"FF{i}_D", PinType.INPUT)
        tt.add_bel_pin(ff, "CLK", f"FF{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(ff, "Q", f"FF{i}_Q", PinType.OUTPUT)

    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_io_tiletype(chip):
    tt = chip.create_tile_type("IO")
    inputs = []
    outputs = []
    for i in range(N_IO):
        tt.create_wire(f"IO{i}_T", "IO_T")
        tt.create_wire(f"IO{i}_I", "IO_I")
        tt.create_wire(f"IO{i}_O", "IO_O")
        tt.create_wire(f"IO{i}_PAD", "IO_PAD")
        inputs += [f"IO{i}_T", f"IO{i}_I"]
        outputs += [f"IO{i}_O"]
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
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
    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_bram_tiletype(chip):
    tt = chip.create_tile_type("BRAM")
    inputs = []
    outputs = []
    for i in range(16):
        inputs.append(f"ADDRARDADDR[{i}]")
        tt.create_wire(f"ADDRARDADDR[{i}]", "BRAM_ADDR")
    for i in range(32):
        inputs.append(f"DIADI[{i}]")
        outputs.append(f"DOADO[{i}]")
        tt.create_wire(f"DIADI[{i}]", "BRAM_DATA_IN")
        tt.create_wire(f"DOADO[{i}]", "BRAM_DATA_OUT")
    for i in range(4):
        inputs.append(f"WEA[{i}]")
        tt.create_wire(f"WEA[{i}]", "BRAM_WE")
    for name in ("CLKARDCLK", "ENARDEN"):
        inputs.append(name)
        tt.create_wire(name, "BRAM_CTRL")
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    for c in range(N_CLK):
        tt.create_pip(f"CLK{c}", "CLKARDCLK")
    ram = tt.create_bel("RAM", "RAMB36E1", z=0)
    for i in range(16):
        tt.add_bel_pin(ram, f"ADDRARDADDR[{i}]", f"ADDRARDADDR[{i}]", PinType.INPUT)
    for i in range(32):
        tt.add_bel_pin(ram, f"DIADI[{i}]", f"DIADI[{i}]", PinType.INPUT)
        tt.add_bel_pin(ram, f"DOADO[{i}]", f"DOADO[{i}]", PinType.OUTPUT)
    for i in range(4):
        tt.add_bel_pin(ram, f"WEA[{i}]", f"WEA[{i}]", PinType.INPUT)
    tt.add_bel_pin(ram, "CLKARDCLK", "CLKARDCLK", PinType.INPUT)
    tt.add_bel_pin(ram, "ENARDEN", "ENARDEN", PinType.INPUT)
    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_dsp_tiletype(chip):
    tt = chip.create_tile_type("DSP")
    inputs = []
    outputs = []
    for prefix, width, wire_type in (("A", 30, "DSP_A"), ("B", 18, "DSP_B"), ("C", 48, "DSP_C")):
        for i in range(width):
            name = f"{prefix}[{i}]"
            inputs.append(name)
            tt.create_wire(name, wire_type)
    for prefix, width in (("ALUMODE", 4), ("OPMODE", 7)):
        for i in range(width):
            name = f"{prefix}[{i}]"
            inputs.append(name)
            tt.create_wire(name, "DSP_CTRL")
    for name in ("CLK", "CARRYIN"):
        inputs.append(name)
        tt.create_wire(name, "DSP_CTRL")
    for i in range(48):
        name = f"P[{i}]"
        outputs.append(name)
        tt.create_wire(name, "DSP_P")
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
        tt.create_pip(f"CLK{c}", "CLK")
    dsp = tt.create_bel("DSP", "DSP48E1", z=0)
    for prefix, width in (("A", 30), ("B", 18), ("C", 48)):
        for i in range(width):
            tt.add_bel_pin(dsp, f"{prefix}[{i}]", f"{prefix}[{i}]", PinType.INPUT)
    for prefix, width in (("ALUMODE", 4), ("OPMODE", 7)):
        for i in range(width):
            tt.add_bel_pin(dsp, f"{prefix}[{i}]", f"{prefix}[{i}]", PinType.INPUT)
    tt.add_bel_pin(dsp, "CLK", "CLK", PinType.INPUT)
    tt.add_bel_pin(dsp, "CARRYIN", "CARRYIN", PinType.INPUT)
    for i in range(48):
        tt.add_bel_pin(dsp, f"P[{i}]", f"P[{i}]", PinType.OUTPUT)
    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_corner_tiletype(chip):
    tt = chip.create_tile_type("NULL")
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
        tt.create_wire(f"CLK{c}_PREV", "CLK_ROUTE")
        tt.create_pip(f"CLK{c}_PREV", f"CLK{c}")
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    gnd = tt.create_bel("GND_DRV", "GND_DRV", z=0)
    tt.add_bel_pin(gnd, "GND", "GND", PinType.OUTPUT)
    vcc = tt.create_bel("VCC_DRV", "VCC_DRV", z=1)
    tt.add_bel_pin(vcc, "VCC", "VCC", PinType.OUTPUT)
    return tt


def is_corner(x, y, width, height):
    return (x in (0, width - 1)) and (y in (0, height - 1))


def build_column_plan():
    plan = []
    inserted_bram = 0
    inserted_dsp = 0
    total_bram_bands = BRAM_FULL_COLS + (1 if BRAM_PARTIAL_ROWS else 0)
    total_dsp_bands = DSP_FULL_COLS + (1 if DSP_PARTIAL_ROWS else 0)

    for i in range(LOGIC_PAIR_COUNT):
        plan.extend(["CLB_L", "INT_L", "INT_R", "CLB_R"])
        if (i + 1) * total_bram_bands // LOGIC_PAIR_COUNT > inserted_bram:
            plan.extend(["INT_L", "BRAM", "INT_R"])
            inserted_bram += 1
        if (i + 1) * total_dsp_bands // LOGIC_PAIR_COUNT > inserted_dsp:
            plan.extend(["INT_L", "DSP", "INT_R"])
            inserted_dsp += 1

    return plan


def column_active_rows(plan):
    active = []
    bram_seen = 0
    dsp_seen = 0
    for token in plan:
        if token == "BRAM":
            bram_seen += 1
            if bram_seen <= BRAM_FULL_COLS:
                active.append(INTERIOR_ROWS)
            else:
                active.append(BRAM_PARTIAL_ROWS)
        elif token == "DSP":
            dsp_seen += 1
            if dsp_seen <= DSP_FULL_COLS:
                active.append(INTERIOR_ROWS)
            else:
                active.append(DSP_PARTIAL_ROWS)
        else:
            active.append(INTERIOR_ROWS)
    return active


def make_layout():
    columns = build_column_plan()
    active_rows = column_active_rows(columns)
    width = len(columns) + 2
    height = INTERIOR_ROWS + 2
    layout = {}

    logic_seen = 0
    for x in range(width):
        for y in range(height):
            if is_corner(x, y, width, height):
                layout[(x, y)] = "NULL"
                continue
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                layout[(x, y)] = "IO"
                continue

            token = columns[x - 1]
            active = active_rows[x - 1]
            row_idx = y - 1

            if token in ("BRAM", "DSP") and row_idx >= active:
                layout[(x, y)] = "INT_L"
                continue

            if token == "CLB_L":
                logic_seen += 1
                layout[(x, y)] = (
                    "CLB_L_TRIM" if logic_seen <= TRIMMED_LOGIC_TILE_COUNT else "CLB_L"
                )
            elif token == "CLB_R":
                logic_seen += 1
                layout[(x, y)] = (
                    "CLB_R_TRIM" if logic_seen <= TRIMMED_LOGIC_TILE_COUNT else "CLB_R"
                )
            else:
                layout[(x, y)] = token

    return {
        "width": width,
        "height": height,
        "columns": columns,
        "layout": layout,
    }


def capacity_summary(layout_info):
    counts = Counter()
    for tile_type in layout_info["layout"].values():
        for bel_type, amount in RESOURCE_CAPACITY.get(tile_type, {}).items():
            counts[bel_type] += amount
    return counts


def create_nodes(ch, width, height):
    for y in range(height):
        for x in range(width):
            if not is_corner(x, y, width, height):
                local_nodes = [[NodeWire(x, y, f"SWITCH{i}")] for i in range(Wl)]
                for d, dx, dy in dirs:
                    x1 = x - dx
                    y1 = y - dy
                    if (
                        x1 < 0
                        or x1 >= width
                        or y1 < 0
                        or y1 >= height
                        or is_corner(x1, y1, width, height)
                    ):
                        continue
                    for i in range(Wl):
                        local_nodes[i].append(NodeWire(x1, y1, f"{d}{i}"))
                for n in local_nodes:
                    ch.add_node(n)

            for c in range(N_CLK):
                gclk_tile_x = c + 1
                if y != 1:
                    if y == 0:
                        if x == 0:
                            clk_node = [NodeWire(gclk_tile_x, 0, f"GCLK{c}_OUT")]
                        else:
                            clk_node = [NodeWire(x - 1, y, f"CLK{c}")]
                    else:
                        clk_node = [NodeWire(x, y - 1, f"CLK{c}")]
                    clk_node.append(NodeWire(x, y, f"CLK{c}_PREV"))
                    if y == 0:
                        clk_node.append(NodeWire(x, y + 1, f"CLK{c}_PREV"))
                    ch.add_node(clk_node)


def set_timings(ch):
    speed = "DEFAULT"
    tmg = ch.set_speed_grades([speed])
    for name, delay, cap, res in (
        ("SWINPUT", 80, 5000, 1000),
        ("SWOUTPUT", 100, 5000, 800),
        ("SWNEIGH", 120, 7000, 1200),
    ):
        tmg.set_pip_class(
            grade=speed,
            name=name,
            delay=TimingValue(delay),
            in_cap=TimingValue(cap),
            out_res=TimingValue(res),
        )

    lut = ch.timing.add_cell_variant(speed, "LUT6")
    for j in range(K):
        lut.add_comb_arc(f"I[{j}]", "F", TimingValue(150 + j * 15))

    dff = ch.timing.add_cell_variant(speed, "DFF")
    dff.add_setup_hold("CLK", "D", ClockEdge.RISING, TimingValue(150), TimingValue(25))
    dff.add_clock_out("CLK", "Q", ClockEdge.RISING, TimingValue(200))

    bram = ch.timing.add_cell_variant(speed, "RAMB36E1")
    bram.add_setup_hold("CLKARDCLK", "ENARDEN", ClockEdge.RISING, TimingValue(200), TimingValue(30))
    bram.add_clock_out("CLKARDCLK", "DOADO[0]", ClockEdge.RISING, TimingValue(450))

    dsp = ch.timing.add_cell_variant(speed, "DSP48E1")
    dsp.add_setup_hold("CLK", "A[0]", ClockEdge.RISING, TimingValue(180), TimingValue(25))
    dsp.add_clock_out("CLK", "P[0]", ClockEdge.RISING, TimingValue(500))


def generate_xc7_exact(output_bba):
    layout_info = make_layout()
    counts = capacity_summary(layout_info)
    width = layout_info["width"]
    height = layout_info["height"]

    print(
        f"Grid: {width}x{height}, "
        f"{counts['LUT6']} LUT6, {counts['DFF']} DFF, "
        f"{counts['RAMB36E1']} BRAM, {counts['DSP48E1']} DSP"
    )

    ch = Chip("xc7_exact", "XC7_EXACT", width, height)
    ch.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    ch.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))

    create_int_tiletype(ch, "INT_L")
    create_int_tiletype(ch, "INT_R")
    create_logic_tiletype(ch, "CLB_L", FULL_FFS_PER_LOGIC_TILE)
    create_logic_tiletype(ch, "CLB_R", FULL_FFS_PER_LOGIC_TILE)
    create_logic_tiletype(ch, "CLB_L_TRIM", TRIMMED_FFS_PER_LOGIC_TILE)
    create_logic_tiletype(ch, "CLB_R_TRIM", TRIMMED_FFS_PER_LOGIC_TILE)
    create_bram_tiletype(ch)
    create_dsp_tiletype(ch)
    create_io_tiletype(ch)
    create_corner_tiletype(ch)

    for (x, y), tile_type in layout_info["layout"].items():
        ch.set_tile_type(x, y, tile_type)

    create_nodes(ch, width, height)
    set_timings(ch)
    ch.strs.known_id_count = 0
    ch.write_bba(output_bba)


def main():
    parser = argparse.ArgumentParser(description="Generate exact-capacity XC7-like chipdb")
    parser.add_argument(
        "output",
        nargs="?",
        default=path.join(CHIPDB_DIR, "xc7_exact.bba"),
        help="Output .bba file path (default: chip_database/xc7_exact.bba)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write layout/capacity summary JSON",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write the optional summary JSON and skip chipdb generation",
    )
    parser.add_argument(
        "--bin",
        action="store_true",
        help="Also assemble the .bba into a .bin using bbasm",
    )
    args = parser.parse_args()

    info = make_layout()
    counts = capacity_summary(info)

    if not args.summary_only:
        output = args.output
        generate_xc7_exact(output)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(
                {
                    "width": info["width"],
                    "height": info["height"],
                    "resource_counts": dict(counts),
                    "column_plan": info["columns"],
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    if args.bin and not args.summary_only:
        output_bin = output.replace(".bba", ".bin")
        subprocess.run([BBASM, "--le", output, output_bin], check=True)


if __name__ == "__main__":
    main()

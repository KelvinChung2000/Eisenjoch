# ruff: noqa: E402, F403, F405
"""Generate a parameterized example arch chipdb for benchmarking.

Usage: python gen_chipdb.py <output.bba> [--size NxN]
Default grid size is 100x100 (~71000 LUT4s).

Architecture features:
  - LUT4 + DFF + DLATCH per slice (8 slices per LOGIC tile)
  - Dual clock networks (CLK0 from IO tile 1,0; CLK1 from IO tile 2,0)
  - Per-FF clock mux (each FF/latch can select CLK0 or CLK1)
  - BRAM tiles every 15th row
  - IO tiles on edges, NULL corners with GND/VCC drivers
"""

import argparse
import sys
from os import path

# Point at the himbaechel dbgen library in the C++ nextpnr checkout.
CPP_NEXTPNR = "/home/kelvin/nextpnr"
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

# LUT input count
K = 4
# SLICEs per tile
N = 8
# IO BELs per IO tile
N_IO = 2
# Number of clock networks
N_CLK = 2
# number of local wires
Wl = N * (K + 1) + 16
# 1/Fc for bel input wire pips; local wire pips and neighbour pips
Si = 6
Sq = 6

dirs = [  # name, dx, dy
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
]

FIXTURES_DIR = path.join(
    path.dirname(path.abspath(__file__)),
    "..",
    "crates",
    "nextpnr",
    "tests",
    "fixtures",
)


def create_switch_matrix(tt, inputs, outputs):
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    for i in range(Wl):
        tt.create_wire(f"SWITCH{i}", "SWITCH")
    # neighbor wires
    for i in range(Wl):
        for d, dx, dy in dirs:
            tt.create_wire(f"{d}{i}", f"NEIGH_{d}")
    # input pips
    for i, w in enumerate(inputs):
        for j in range((i % Si), Wl, Si):
            tt.create_pip(f"SWITCH{j}", w, timing_class="SWINPUT")
    # output pips
    for i, w in enumerate(outputs):
        for j in range((i % Sq), Wl, Sq):
            tt.create_pip(w, f"SWITCH{j}", timing_class="SWINPUT")
    for i in range(Wl):
        tt.create_pip("GND", f"SWITCH{i}")
        tt.create_pip("VCC", f"SWITCH{i}")
    for i in range(Wl):
        for j, (d, dx, dy) in enumerate(dirs):
            # Inbound: neighbor wire -> local switch (arrive from neighbor tile)
            tt.create_pip(f"{d}{(i + j) % Wl}", f"SWITCH{i}", timing_class="SWNEIGH")
            # Outbound: local switch -> neighbor wire (leave to neighbor tile)
            tt.create_pip(f"SWITCH{i}", f"{d}{(i + j) % Wl}", timing_class="SWNEIGH")
    # Dual clock distribution wires and PIPs
    for c in range(N_CLK):
        if not tt.has_wire(f"CLK{c}"):
            tt.create_wire(f"CLK{c}", "TILE_CLK")
        tt.create_wire(f"CLK{c}_PREV", "CLK_ROUTE")
        tt.create_pip(f"CLK{c}_PREV", f"CLK{c}")
        # Allow switch matrix to drive clock wires (for general clock routing)
        for j in range(0, Wl, Si):
            tt.create_pip(f"SWITCH{j}", f"CLK{c}")
        # Allow clock wires to feed back into switch matrix
        tt.create_pip(f"CLK{c}", f"SWITCH{c}")
    tt.create_group("SWITCHBOX", "SWITCHBOX")


def create_logic_tiletype(chip):
    tt = chip.create_tile_type("LOGIC")
    inputs = []
    outputs = []
    for i in range(N):
        for j in range(K):
            inputs.append(f"L{i}_I{j}")
            tt.create_wire(f"L{i}_I{j}", "LUT_INPUT")
        tt.create_wire(f"L{i}_D", "FF_DATA")
        tt.create_wire(f"L{i}_O", "LUT_OUT")
        tt.create_wire(f"L{i}_Q", "FF_OUT")
        tt.create_wire(f"L{i}_CLK", "FF_CLK")
        outputs += [f"L{i}_O", f"L{i}_Q"]

    # Create dual clock wires (needed before switch matrix creates PIPs)
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")

    for i in range(N):
        # Clock mux: per-FF local clock can come from either CLK0 or CLK1
        for c in range(N_CLK):
            tt.create_pip(f"CLK{c}", f"L{i}_CLK")

        # LUT4
        lut = tt.create_bel(f"L{i}_LUT", "LUT4", z=(i * 3 + 0))
        for j in range(K):
            tt.add_bel_pin(lut, f"I[{j}]", f"L{i}_I{j}", PinType.INPUT)
        tt.add_bel_pin(lut, "F", f"L{i}_O", PinType.OUTPUT)

        # DFF data routing
        tt.create_pip(f"L{i}_O", f"L{i}_D")
        tt.create_pip(f"L{i}_I{K - 1}", f"L{i}_D")

        # DFF (edge-triggered)
        ff = tt.create_bel(f"L{i}_FF", "DFF", z=(i * 3 + 1))
        tt.add_bel_pin(ff, "D", f"L{i}_D", PinType.INPUT)
        tt.add_bel_pin(ff, "CLK", f"L{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(ff, "Q", f"L{i}_Q", PinType.OUTPUT)

        # DLATCH (level-sensitive, shares D and Q wires with DFF)
        latch = tt.create_bel(f"L{i}_LATCH", "DLATCH", z=(i * 3 + 2))
        tt.add_bel_pin(latch, "D", f"L{i}_D", PinType.INPUT)
        tt.add_bel_pin(latch, "G", f"L{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(latch, "Q", f"L{i}_Q", PinType.OUTPUT)

    create_switch_matrix(tt, inputs, outputs)

    # Allow GND/VCC to drive clock/gate pins (for always-transparent latches)
    for i in range(N):
        tt.create_pip("GND", f"L{i}_CLK")
        tt.create_pip("VCC", f"L{i}_CLK")

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
    # Dual clock wires
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    for i in range(N_IO):
        io = tt.create_bel(f"IO{i}", "IOB", z=i)
        tt.add_bel_pin(io, "I", f"IO{i}_I", PinType.INPUT)
        tt.add_bel_pin(io, "T", f"IO{i}_T", PinType.INPUT)
        tt.add_bel_pin(io, "O", f"IO{i}_O", PinType.OUTPUT)
        tt.add_bel_pin(io, "PAD", f"IO{i}_PAD", PinType.INOUT)
    # Global clock outputs: IO0 drives GCLK0, IO1 drives GCLK1
    for c in range(N_CLK):
        tt.create_wire(f"GCLK{c}_OUT", "GCLK")
        tt.create_pip(f"IO{c}_O", f"GCLK{c}_OUT")
    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_bram_tiletype(chip):
    Aw = 9
    Dw = 16
    tt = chip.create_tile_type("BRAM")
    inputs = [f"RAM_WA{i}" for i in range(Aw)]
    inputs += [f"RAM_RA{i}" for i in range(Aw)]
    inputs += [f"RAM_WE{i}" for i in range(Dw // 8)]
    inputs += [f"RAM_DI{i}" for i in range(Dw)]
    outputs = [f"RAM_DO{i}" for i in range(Dw)]
    for w in inputs:
        tt.create_wire(w, "RAM_IN")
    for w in outputs:
        tt.create_wire(w, "RAM_OUT")
    # BRAM uses CLK0 by default
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    tt.create_wire("RAM_CLK", "FF_CLK")
    for c in range(N_CLK):
        tt.create_pip(f"CLK{c}", "RAM_CLK")
    ram = tt.create_bel("RAM", f"BRAM_{2**Aw}X{Dw}", z=0)
    tt.add_bel_pin(ram, "CLK", "RAM_CLK", PinType.INPUT)
    for i in range(Aw):
        tt.add_bel_pin(ram, f"WA[{i}]", f"RAM_WA{i}", PinType.INPUT)
        tt.add_bel_pin(ram, f"RA[{i}]", f"RAM_RA{i}", PinType.INPUT)
    for i in range(Dw // 8):
        tt.add_bel_pin(ram, f"WE[{i}]", f"RAM_WE{i}", PinType.INPUT)
    for i in range(Dw):
        tt.add_bel_pin(ram, f"DI[{i}]", f"RAM_DI{i}", PinType.INPUT)
        tt.add_bel_pin(ram, f"DO[{i}]", f"RAM_DO{i}", PinType.OUTPUT)
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


def is_corner(x, y, X, Y):
    return ((x == 0) or (x == (X - 1))) and ((y == 0) or (y == (Y - 1)))


def create_nodes(ch, X, Y):
    for y in range(Y):
        for x in range(X):
            if not is_corner(x, y, X, Y):
                local_nodes = [[NodeWire(x, y, f"SWITCH{i}")] for i in range(Wl)]
                for d, dx, dy in dirs:
                    x1 = x - dx
                    y1 = y - dy
                    if (
                        x1 < 0
                        or x1 >= X
                        or y1 < 0
                        or y1 >= Y
                        or is_corner(x1, y1, X, Y)
                    ):
                        continue
                    for i in range(Wl):
                        local_nodes[i].append(NodeWire(x1, y1, f"{d}{i}"))
                for n in local_nodes:
                    ch.add_node(n)

            # Dual clock ladders
            for c in range(N_CLK):
                gclk_tile_x = c + 1  # CLK0 from (1,0), CLK1 from (2,0)
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
    tmg.set_pip_class(
        grade=speed,
        name="SWINPUT",
        delay=TimingValue(80),
        in_cap=TimingValue(5000),
        out_res=TimingValue(1000),
    )
    tmg.set_pip_class(
        grade=speed,
        name="SWOUTPUT",
        delay=TimingValue(100),
        in_cap=TimingValue(5000),
        out_res=TimingValue(800),
    )
    tmg.set_pip_class(
        grade=speed,
        name="SWNEIGH",
        delay=TimingValue(120),
        in_cap=TimingValue(7000),
        out_res=TimingValue(1200),
    )
    # LUT4 timing
    lut = ch.timing.add_cell_variant(speed, "LUT4")
    for j in range(K):
        lut.add_comb_arc(f"I[{j}]", "F", TimingValue(150 + j * 15))
    # DFF timing (edge-triggered)
    dff = ch.timing.add_cell_variant(speed, "DFF")
    dff.add_setup_hold("CLK", "D", ClockEdge.RISING, TimingValue(150), TimingValue(25))
    dff.add_clock_out("CLK", "Q", ClockEdge.RISING, TimingValue(200))
    # DLATCH timing (level-sensitive, modeled as combinational arcs)
    dlatch = ch.timing.add_cell_variant(speed, "DLATCH")
    dlatch.add_comb_arc("D", "Q", TimingValue(180))
    dlatch.add_comb_arc("G", "Q", TimingValue(200))


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark chipdb")
    parser.add_argument("output", help="Output .bba file path")
    parser.add_argument(
        "--size", default="100x100", help="Grid size NxN (default: 100x100)"
    )
    args = parser.parse_args()

    parts = args.size.lower().split("x")
    X = int(parts[0])
    Y = int(parts[1]) if len(parts) > 1 else X

    logic_tiles = sum(
        1
        for x in range(X)
        for y in range(Y)
        if not is_corner(x, y, X, Y)
        and not (x == 0 or x == X - 1 or y == 0 or y == Y - 1)
        and not ((y % 15) == 7)
    )
    print(f"Grid: {X}x{Y}, ~{logic_tiles * N} LUT4s, ~{logic_tiles * N} DFFs/DLATCHes")

    ch = Chip("example", "EX1", X, Y)
    ch.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    ch.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))
    create_logic_tiletype(ch)
    create_io_tiletype(ch)
    create_bram_tiletype(ch)
    create_corner_tiletype(ch)

    for x in range(X):
        for y in range(Y):
            if x == 0 or x == X - 1:
                if y == 0 or y == Y - 1:
                    ch.set_tile_type(x, y, "NULL")
                else:
                    ch.set_tile_type(x, y, "IO")
            elif y == 0 or y == Y - 1:
                ch.set_tile_type(x, y, "IO")
            elif (y % 15) == 7:
                ch.set_tile_type(x, y, "BRAM")
            else:
                ch.set_tile_type(x, y, "LOGIC")

    create_nodes(ch, X, Y)
    set_timings(ch)
    ch.strs.known_id_count = 0
    ch.write_bba(args.output)


if __name__ == "__main__":
    main()

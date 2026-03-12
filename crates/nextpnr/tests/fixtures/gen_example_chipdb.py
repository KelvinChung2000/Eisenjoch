# ruff: noqa: E402, F403, F405
"""Generate a small (10x10) example arch chipdb for integration testing.

Adapted from nextpnr himbaechel/uarch/example/example_arch_gen.py with a
reduced grid size to keep the .bin file small (~50KB) while preserving all
tile types and routing structure.
"""

import sys
from os import path

# Point at the himbaechel dbgen library in the C++ nextpnr checkout.
CPP_NEXTPNR = "/home/kelvin/nextpnr"
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

# Grid size including IOBs at edges — reduced from 100x100
X = 10
Y = 10
# LUT input count
K = 4
# SLICEs per tile
N = 8
# number of local wires
Wl = N * (K + 1) + 16
# 1/Fc for bel input wire pips; local wire pips and neighbour pips
Si = 6
Sq = 6
Sl = 1

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


def create_switch_matrix(tt: TileType, inputs: list[str], outputs: list[str]):
    # constant wires
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    # switch wires
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
    # constant pips
    for i in range(Wl):
        tt.create_pip("GND", f"SWITCH{i}")
        tt.create_pip("VCC", f"SWITCH{i}")
    # neighbour local pips
    for i in range(Wl):
        for j, (d, dx, dy) in enumerate(dirs):
            tt.create_pip(f"{d}{(i + j) % Wl}", f"SWITCH{i}", timing_class="SWNEIGH")
    # clock "ladder"
    if not tt.has_wire("CLK"):
        tt.create_wire("CLK", "TILE_CLK")
    tt.create_wire("CLK_PREV", "CLK_ROUTE")
    tt.create_pip("CLK_PREV", "CLK")

    tt.create_group("SWITCHBOX", "SWITCHBOX")


def create_logic_tiletype(chip: Chip):
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
        outputs += [f"L{i}_O", f"L{i}_Q"]
    tt.create_wire("CLK", "TILE_CLK")
    for i in range(N):
        # LUT
        lut = tt.create_bel(f"L{i}_LUT", "LUT4", z=(i * 2 + 0))
        for j in range(K):
            tt.add_bel_pin(lut, f"I[{j}]", f"L{i}_I{j}", PinType.INPUT)
        tt.add_bel_pin(lut, "F", f"L{i}_O", PinType.OUTPUT)
        # FF data can come from LUT output or LUT I3
        tt.create_pip(f"L{i}_O", f"L{i}_D")
        tt.create_pip(f"L{i}_I{K - 1}", f"L{i}_D")
        # FF
        ff = tt.create_bel(f"L{i}_FF", "DFF", z=(i * 2 + 1))
        tt.add_bel_pin(ff, "D", f"L{i}_D", PinType.INPUT)
        tt.add_bel_pin(ff, "CLK", "CLK", PinType.INPUT)
        tt.add_bel_pin(ff, "Q", f"L{i}_Q", PinType.OUTPUT)
    create_switch_matrix(tt, inputs, outputs)
    return tt


N_io = 2


def create_io_tiletype(chip: Chip):
    tt = chip.create_tile_type("IO")
    inputs = []
    outputs = []
    for i in range(N_io):
        tt.create_wire(f"IO{i}_T", "IO_T")
        tt.create_wire(f"IO{i}_I", "IO_I")
        tt.create_wire(f"IO{i}_O", "IO_O")
        tt.create_wire(f"IO{i}_PAD", "IO_PAD")
        inputs += [f"IO{i}_T", f"IO{i}_I"]
        outputs += [f"IO{i}_O"]
    tt.create_wire("CLK", "TILE_CLK")
    for i in range(N_io):
        io = tt.create_bel(f"IO{i}", "IOB", z=i)
        tt.add_bel_pin(io, "I", f"IO{i}_I", PinType.INPUT)
        tt.add_bel_pin(io, "T", f"IO{i}_T", PinType.INPUT)
        tt.add_bel_pin(io, "O", f"IO{i}_O", PinType.OUTPUT)
        tt.add_bel_pin(io, "PAD", f"IO{i}_PAD", PinType.INOUT)
    # Used in top left IO only
    tt.create_wire("GCLK_OUT", "GCLK")
    tt.create_pip("IO0_O", "GCLK_OUT")
    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_bram_tiletype(chip: Chip):
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
    tt.create_wire("CLK", "TILE_CLK")
    ram = tt.create_bel("RAM", f"BRAM_{2**Aw}X{Dw}", z=0)
    tt.add_bel_pin(ram, "CLK", "CLK", PinType.INPUT)
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


def create_corner_tiletype(ch):
    tt = ch.create_tile_type("NULL")
    tt.create_wire("CLK", "TILE_CLK")
    tt.create_wire("CLK_PREV", "CLK_ROUTE")
    tt.create_pip("CLK_PREV", "CLK")

    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")

    gnd = tt.create_bel("GND_DRV", "GND_DRV", z=0)
    tt.add_bel_pin(gnd, "GND", "GND", PinType.OUTPUT)
    vcc = tt.create_bel("VCC_DRV", "VCC_DRV", z=1)
    tt.add_bel_pin(vcc, "VCC", "VCC", PinType.OUTPUT)

    return tt


def is_corner(x, y):
    return ((x == 0) or (x == (X - 1))) and ((y == 0) or (y == (Y - 1)))


def create_nodes(ch):
    for y in range(Y):
        for x in range(X):
            if not is_corner(x, y):
                local_nodes = [[NodeWire(x, y, f"SWITCH{i}")] for i in range(Wl)]
                for d, dx, dy in dirs:
                    x1 = x - dx
                    y1 = y - dy
                    if x1 < 0 or x1 >= X or y1 < 0 or y1 >= Y or is_corner(x1, y1):
                        continue
                    for i in range(Wl):
                        local_nodes[i].append(NodeWire(x1, y1, f"{d}{i}"))
                for n in local_nodes:
                    ch.add_node(n)
            # connect up clock ladder
            if y != 1:
                if y == 0:
                    if x == 0:
                        clk_node = [NodeWire(1, 0, "GCLK_OUT")]
                    else:
                        clk_node = [NodeWire(x - 1, y, "CLK")]
                else:
                    clk_node = [NodeWire(x, y - 1, "CLK")]
                clk_node.append(NodeWire(x, y, "CLK_PREV"))
                if y == 0:
                    clk_node.append(NodeWire(x, y + 1, "CLK_PREV"))
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
    lut = ch.timing.add_cell_variant(speed, "LUT4")
    for j in range(K):
        lut.add_comb_arc(f"I[{j}]", "F", TimingValue(150 + j * 15))
    dff = ch.timing.add_cell_variant(speed, "DFF")
    dff.add_setup_hold("CLK", "D", ClockEdge.RISING, TimingValue(150), TimingValue(25))
    dff.add_clock_out("CLK", "Q", ClockEdge.RISING, TimingValue(200))


def main():
    ch = Chip("example", "EX1", X, Y)
    ch.strs.read_constids(path.join(path.dirname(__file__), "constids.inc"))
    ch.read_gfxids(path.join(path.dirname(__file__), "gfxids.inc"))
    create_logic_tiletype(ch)
    create_io_tiletype(ch)
    create_bram_tiletype(ch)
    create_corner_tiletype(ch)
    # Setup tile grid
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
    create_nodes(ch)
    set_timings(ch)
    # Force all constid strings into bba_ids so the Rust loader can resolve them.
    # The C++ nextpnr has known IDs compiled in, but the Rust code reads them
    # from the chipdb, so we set known_id_count=0 to include all strings.
    ch.strs.known_id_count = 0
    ch.write_bba(sys.argv[1])


if __name__ == "__main__":
    main()

# ruff: noqa: E402, F403, F405
"""Generate a simplified-Ultrascale chipdb that matches ISPD'2016 design.scl.

Architecture (from Rajarathnam et al., ISPD'23, "DREAMPlaceFPGA-PL"):
  - 168 x 480 grid
  - SLICE tile = 16 LUT6 + 16 FDRE + 1 CARRY8 (8 BLEs x (2 LUT + 2 FF))
  - DSP tile = 1 DSP48E2
  - BRAM tile = 1 RAMB36E2
  - IO tile = 64 IOB (one column has 8 tiles spaced 60 rows apart per design.scl)
  - NULL tile = routing-only filler (BRAM/DSP/IO column gaps)

The column layout, including which (x, y) cells host real tiles, is read
verbatim from a reference design.scl in the ISPD'2016 release. Gaps in
sparse columns (BRAM every 5 rows, DSP every ~2.5 rows, IO every 60 rows)
are filled with NULL tiles so the grid is dense for routing.

Usage:
    python gen_ultrascale.py <output.bba> [--scl <design.scl>]
"""

import argparse
import sys
from collections import defaultdict
from os import path

CPP_NEXTPNR = "/home/kelvin/nextpnr"
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

FIXTURES_DIR = path.join(
    path.dirname(path.abspath(__file__)),
    "..",
    "..",
    "..",
    "crates",
    "nextpnr",
    "tests",
    "fixtures",
)

DEFAULT_SCL = "/home/kelvin/side-project/eisenjoch/benchmark/ispd/extracted/2016/FPGA01/FPGA01/design.scl"

# SLICE composition.
LUT_K = 6
N_LUT_PER_SLICE = 16
N_FF_PER_SLICE = 16
N_BLE_PER_SLICE = 8  # 2 LUTs + 2 FFs per BLE
N_CARRY_PER_SLICE = 1

# Half-CLB clock/control sharing: BLEs 0..3 share CLK0, BLEs 4..7 share CLK1.
N_CLK = 2

# Wide DSP/BRAM pin counts (simplified to keep tile size manageable).
DSP_A_WIDTH = 30
DSP_B_WIDTH = 18
DSP_P_WIDTH = 48
BRAM_ADDR_WIDTH = 15
BRAM_DATA_WIDTH = 36

N_IO_PER_TILE = 64

# Intra-tile switch hub. These wires carry no node: they are the tile-local
# crossbar that span wires and BEL pins connect through, matching the role of
# the real INT tile's INT_NODE_* resources.
SWITCH_NODES = 208
SI = 6
SQ = 6

# Inter-tile span wires, measured from a real UltraScale INT tile: xcvu3p
# INT_X0Y149 under Vivado 2025.2, counting "_BEG" node origins per class.
#
# VU095's virtexu family is not installed locally, so this is UltraScale+.
# It shares the INT tile's EE/WW/NN/SS classes, and every vertical class
# measured its nominal span exactly (NN04 -> 4 tiles, NN12 -> 12), which
# validates reading the span off the wire name.
#
# There are no diagonal resources in the real device, and none here.
SPAN_WIRE_COUNTS = {
    "E": {1: 8, 2: 16, 4: 16, 12: 8},
    "W": {1: 8, 2: 16, 4: 16, 12: 8},
    "N": {1: 16, 2: 16, 4: 16, 12: 8},
    "S": {1: 16, 2: 16, 4: 16, 12: 8},
}
DIR_DELTA = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}

# Real span wires tap EVERY intermediate tile (measured: NN04 taps at dy=0..4),
# which is what lets a short net ride a long wire. Emitting the taps keeps each
# node collinear, so the placer books it at max reach and prices the borrow.
def span_wire_names(direction, length, index):
    """(beg, [intermediate taps], end) wire names for one span wire."""
    beg = f"{direction}{length}_BEG{index}"
    taps = [f"{direction}{length}_T{k}_{index}" for k in range(1, length)]
    end = f"{direction}{length}_END{index}"
    return beg, taps, end


def span_origin_count():
    return sum(sum(per_len.values()) for per_len in SPAN_WIRE_COUNTS.values())


def parse_scl_sitemap(scl_path):
    """Returns (X, Y, sites) where sites[(x,y)] = SLICE|DSP|BRAM|IO."""
    sites = {}
    X = Y = 0
    in_map = False
    with open(scl_path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("SITEMAP"):
                parts = line.split()
                X, Y = int(parts[1]), int(parts[2])
                in_map = True
                continue
            if line.startswith("END SITEMAP"):
                in_map = False
                continue
            if not in_map or not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            sites[(int(parts[0]), int(parts[1]))] = parts[2]
    return X, Y, sites


def create_switch_matrix(tt, inputs, outputs, *, wl=SWITCH_NODES, n_clk=N_CLK):
    """Tile-local crossbar plus the full inter-tile span wire set.

    Every tile type gets the same span wires. Real UltraScale puts a full INT
    tile beside BRAM/DSP/IO columns, so thinning the matrix in column gaps
    would build routing walls the device does not have.
    """
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    for i in range(wl):
        tt.create_wire(f"SWITCH{i}", "SWITCH")

    # Span wires: `drives` are node origins the hub feeds, `taps` are the
    # intermediate and far-end wires that feed back into the hub.
    drives, taps = [], []
    for d, per_len in SPAN_WIRE_COUNTS.items():
        for length, n in per_len.items():
            for i in range(n):
                beg, mid, end = span_wire_names(d, length, i)
                tt.create_wire(beg, f"SPAN_{d}{length}")
                drives.append(beg)
                for w in mid + [end]:
                    tt.create_wire(w, f"SPAN_{d}{length}")
                    taps.append(w)

    for i, w in enumerate(inputs):
        for j in range((i % SI), wl, SI):
            tt.create_pip(f"SWITCH{j}", w, timing_class="SWINPUT")
    for i, w in enumerate(outputs):
        for j in range((i % SQ), wl, SQ):
            tt.create_pip(w, f"SWITCH{j}", timing_class="SWINPUT")
    for i in range(wl):
        tt.create_pip("GND", f"SWITCH{i}")
        tt.create_pip("VCC", f"SWITCH{i}")
    for i, w in enumerate(drives):
        for j in range((i % SQ), wl, SQ):
            tt.create_pip(f"SWITCH{j}", w, timing_class="SWNEIGH")
    for i, w in enumerate(taps):
        for j in range((i % SI), wl, SI):
            tt.create_pip(w, f"SWITCH{j}", timing_class="SWNEIGH")
    for c in range(n_clk):
        if not tt.has_wire(f"CLK{c}"):
            tt.create_wire(f"CLK{c}", "TILE_CLK")
        tt.create_wire(f"CLK{c}_PREV", "CLK_ROUTE")
        tt.create_pip(f"CLK{c}_PREV", f"CLK{c}")
        for j in range(0, wl, SI):
            tt.create_pip(f"SWITCH{j}", f"CLK{c}")
        tt.create_pip(f"CLK{c}", f"SWITCH{c}")
    tt.create_group("SWITCHBOX", "SWITCHBOX")


def create_slice_tiletype(chip):
    tt = chip.create_tile_type("SLICE")
    inputs, outputs = [], []

    # Half-CLB clock fan-in.
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")

    for i in range(N_LUT_PER_SLICE):
        for j in range(LUT_K):
            tt.create_wire(f"L{i}_I{j}", "LUT_INPUT")
            inputs.append(f"L{i}_I{j}")
        tt.create_wire(f"L{i}_O", "LUT_OUT")
        outputs.append(f"L{i}_O")
        lut = tt.create_bel(f"L{i}_LUT", "LUT6", z=i)
        for j in range(LUT_K):
            tt.add_bel_pin(lut, f"I[{j}]", f"L{i}_I{j}", PinType.INPUT)
        tt.add_bel_pin(lut, "F", f"L{i}_O", PinType.OUTPUT)

    for i in range(N_FF_PER_SLICE):
        tt.create_wire(f"F{i}_D", "FF_DATA")
        tt.create_wire(f"F{i}_Q", "FF_OUT")
        tt.create_wire(f"F{i}_CLK", "FF_CLK")
        outputs.append(f"F{i}_Q")

        # FF D can come from either LUT in the same BLE (BLE = 2 LUTs + 2 FFs).
        ble = i // 2
        tt.create_pip(f"L{ble * 2}_O", f"F{i}_D")
        tt.create_pip(f"L{ble * 2 + 1}_O", f"F{i}_D")
        # Bypass from switch matrix (when LUT is unused).
        tt.create_wire(f"F{i}_D_IN", "FF_DATA")
        inputs.append(f"F{i}_D_IN")
        tt.create_pip(f"F{i}_D_IN", f"F{i}_D")

        # Half-CLB clock: BLEs 0..3 (FFs 0..7) -> CLK0; BLEs 4..7 (FFs 8..15) -> CLK1.
        half = i // (N_FF_PER_SLICE // N_CLK)
        tt.create_pip(f"CLK{half}", f"F{i}_CLK")

        ff = tt.create_bel(f"F{i}_FF", "FDRE", z=N_LUT_PER_SLICE + i)
        tt.add_bel_pin(ff, "D", f"F{i}_D", PinType.INPUT)
        # Vivado FDRE uses "C" as the clock pin name, but the lossy ISPD-JSON
        # converter emits cell port "CLK". Name the BEL pin to match the cell
        # so `bel_view.pin_wire("CLK")` resolves to F{i}_CLK in the placer.
        tt.add_bel_pin(ff, "CLK", f"F{i}_CLK", PinType.INPUT)
        tt.add_bel_pin(ff, "Q", f"F{i}_Q", PinType.OUTPUT)

    # CARRY8 chain (one per slice).
    tt.create_wire("CARRY_CI", "CARRY_IN")
    tt.create_wire("CARRY_CO", "CARRY_OUT")
    inputs.append("CARRY_CI")
    outputs.append("CARRY_CO")
    carry = tt.create_bel(
        "CARRY", "CARRY8", z=N_LUT_PER_SLICE + N_FF_PER_SLICE + 0
    )
    tt.add_bel_pin(carry, "CI", "CARRY_CI", PinType.INPUT)
    tt.add_bel_pin(carry, "CO", "CARRY_CO", PinType.OUTPUT)

    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_dsp_tiletype(chip):
    tt = chip.create_tile_type("DSP")
    inputs = [f"DSP_A{i}" for i in range(DSP_A_WIDTH)]
    inputs += [f"DSP_B{i}" for i in range(DSP_B_WIDTH)]
    outputs = [f"DSP_P{i}" for i in range(DSP_P_WIDTH)]
    for w in inputs + outputs:
        tt.create_wire(w, "DSP_DATA")
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    tt.create_wire("DSP_CLK", "FF_CLK")
    for c in range(N_CLK):
        tt.create_pip(f"CLK{c}", "DSP_CLK")

    dsp = tt.create_bel("DSP", "DSP48E2", z=0)
    tt.add_bel_pin(dsp, "CLK", "DSP_CLK", PinType.INPUT)
    for i in range(DSP_A_WIDTH):
        tt.add_bel_pin(dsp, f"A[{i}]", f"DSP_A{i}", PinType.INPUT)
    for i in range(DSP_B_WIDTH):
        tt.add_bel_pin(dsp, f"B[{i}]", f"DSP_B{i}", PinType.INPUT)
    for i in range(DSP_P_WIDTH):
        tt.add_bel_pin(dsp, f"P[{i}]", f"DSP_P{i}", PinType.OUTPUT)

    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_bram_tiletype(chip):
    tt = chip.create_tile_type("BRAM")
    inputs = [f"RAM_A{i}" for i in range(BRAM_ADDR_WIDTH)]
    inputs += [f"RAM_DI{i}" for i in range(BRAM_DATA_WIDTH)]
    inputs += [f"RAM_WE{i}" for i in range(BRAM_DATA_WIDTH // 8 + 1)]
    outputs = [f"RAM_DO{i}" for i in range(BRAM_DATA_WIDTH)]
    for w in inputs:
        tt.create_wire(w, "RAM_IN")
    for w in outputs:
        tt.create_wire(w, "RAM_OUT")
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    tt.create_wire("RAM_CLK", "FF_CLK")
    for c in range(N_CLK):
        tt.create_pip(f"CLK{c}", "RAM_CLK")

    ram = tt.create_bel("BRAM", "RAMB36E2", z=0)
    tt.add_bel_pin(ram, "CLK", "RAM_CLK", PinType.INPUT)
    for i in range(BRAM_ADDR_WIDTH):
        tt.add_bel_pin(ram, f"ADDR[{i}]", f"RAM_A{i}", PinType.INPUT)
    for i in range(BRAM_DATA_WIDTH):
        tt.add_bel_pin(ram, f"DI[{i}]", f"RAM_DI{i}", PinType.INPUT)
        tt.add_bel_pin(ram, f"DO[{i}]", f"RAM_DO{i}", PinType.OUTPUT)
    for i in range(BRAM_DATA_WIDTH // 8 + 1):
        tt.add_bel_pin(ram, f"WE[{i}]", f"RAM_WE{i}", PinType.INPUT)

    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_io_tiletype(chip):
    tt = chip.create_tile_type("IO")
    inputs, outputs = [], []
    for c in range(N_CLK):
        tt.create_wire(f"CLK{c}", "TILE_CLK")
    for i in range(N_IO_PER_TILE):
        tt.create_wire(f"IO{i}_T", "IO_T")
        tt.create_wire(f"IO{i}_I", "IO_I")
        tt.create_wire(f"IO{i}_O", "IO_O")
        tt.create_wire(f"IO{i}_PAD", "IO_PAD")
        inputs += [f"IO{i}_T", f"IO{i}_I"]
        outputs += [f"IO{i}_O"]
        io = tt.create_bel(f"IO{i}", "IOB", z=i)
        tt.add_bel_pin(io, "I", f"IO{i}_I", PinType.INPUT)
        tt.add_bel_pin(io, "T", f"IO{i}_T", PinType.INPUT)
        tt.add_bel_pin(io, "O", f"IO{i}_O", PinType.OUTPUT)
        tt.add_bel_pin(io, "PAD", f"IO{i}_PAD", PinType.INOUT)

    # First two IOs in tile (0,0) (etc.) drive the global clock nets.
    for c in range(N_CLK):
        tt.create_wire(f"GCLK{c}_OUT", "GCLK")
        tt.create_pip(f"IO{c}_O", f"GCLK{c}_OUT")

    create_switch_matrix(tt, inputs, outputs)
    return tt


def create_null_tiletype(chip):
    tt = chip.create_tile_type("NULL")
    create_switch_matrix(tt, [], [])
    return tt


def create_tieoff_tiletype(chip):
    """Singleton tile hosting one GND_DRV and one VCC_DRV BEL.

    The packer's `pack_constants` pass creates `$PACKER_GND`/`$PACKER_VCC`
    cells of type `GND_DRV`/`VCC_DRV` (with output ports named "GND"/"VCC")
    and calls `bind_to_first_available_bel`. Without these BEL types the
    place stage errors with "No valid BELs available for cell type GND".
    """
    tt = chip.create_tile_type("TIEOFF")
    tt.create_wire("GND_DRV_OUT", "TIEOFF_OUT")
    tt.create_wire("VCC_DRV_OUT", "TIEOFF_OUT")

    gnd = tt.create_bel("GND_DRV", "GND_DRV", z=0)
    tt.add_bel_pin(gnd, "GND", "GND_DRV_OUT", PinType.OUTPUT)

    vcc = tt.create_bel("VCC_DRV", "VCC_DRV", z=1)
    tt.add_bel_pin(vcc, "VCC", "VCC_DRV_OUT", PinType.OUTPUT)

    create_switch_matrix(tt, [], ["GND_DRV_OUT", "VCC_DRV_OUT"])
    return tt


def create_nodes(chip, X, Y, layout):
    """Inter-tile span wires + clock distribution ladders.

    `layout[(x, y)]` is the tile-type name at that grid cell. Caller fills
    every cell (NULL for gaps). Every tile type carries the same span wires,
    so there is no per-type width to reconcile.

    Each span wire is one node running from its origin tile to the tile `length`
    away, tapping every tile in between. A node truncated by the die edge is
    still emitted, shorter; one reduced to a single wire is dropped.
    """
    for y in range(Y):
        for x in range(X):
            for d, per_len in SPAN_WIRE_COUNTS.items():
                dx, dy = DIR_DELTA[d]
                for length, n in per_len.items():
                    for i in range(n):
                        beg, mid, end = span_wire_names(d, length, i)
                        node = [NodeWire(x, y, beg)]
                        for k in range(1, length + 1):
                            nx, ny = x + k * dx, y + k * dy
                            if not (0 <= nx < X and 0 <= ny < Y):
                                break
                            node.append(
                                NodeWire(nx, ny, end if k == length else mid[k - 1])
                            )
                        if len(node) > 1:
                            chip.add_node(node)

            # Clock distribution: simple top-to-bottom ladder per column.
            for c in range(N_CLK):
                if y == 0:
                    if (
                        x == c + 1
                        and layout.get((c + 1, 0)) == "IO"
                    ):
                        clk_node = [
                            NodeWire(c + 1, 0, f"GCLK{c}_OUT"),
                            NodeWire(c + 1, 0, f"CLK{c}_PREV"),
                        ]
                        if Y > 1:
                            clk_node.append(NodeWire(c + 1, 1, f"CLK{c}_PREV"))
                        chip.add_node(clk_node)
                    continue
                clk_node = [
                    NodeWire(x, y - 1, f"CLK{c}"),
                    NodeWire(x, y, f"CLK{c}_PREV"),
                ]
                chip.add_node(clk_node)


def set_timings(chip):
    speed = "DEFAULT"
    tmg = chip.set_speed_grades([speed])
    tmg.set_pip_class(grade=speed, name="SWINPUT",
                      delay=TimingValue(80),
                      in_cap=TimingValue(5000),
                      out_res=TimingValue(1000))
    tmg.set_pip_class(grade=speed, name="SWNEIGH",
                      delay=TimingValue(120),
                      in_cap=TimingValue(7000),
                      out_res=TimingValue(1200))
    # LUT6 + smaller-arity equivalents (the placer will treat them as drop-in).
    for lut_name in ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"):
        lut = chip.timing.add_cell_variant(speed, lut_name)
        for j in range(LUT_K):
            lut.add_comb_arc(f"I[{j}]", "F", TimingValue(150 + j * 15))
    # FDRE / DFF / DLATCH share BEL pin geometry (D/CLK/Q). FDRE is the
    # canonical type; DFF and DLATCH are the cell names emitted by the
    # lossy ISPD-JSON converter and are aliased onto the FDRE bucket at
    # load time.
    for ff_name in ("FDRE", "DFF"):
        ff = chip.timing.add_cell_variant(speed, ff_name)
        ff.add_setup_hold("CLK", "D", ClockEdge.RISING, TimingValue(150), TimingValue(25))
        ff.add_clock_out("CLK", "Q", ClockEdge.RISING, TimingValue(200))
    dl = chip.timing.add_cell_variant(speed, "DLATCH")
    dl.add_comb_arc("D", "Q", TimingValue(180))
    dl.add_comb_arc("CLK", "Q", TimingValue(200))
    # CARRY8 (single-arc placeholder)
    carry = chip.timing.add_cell_variant(speed, "CARRY8")
    carry.add_comb_arc("CI", "CO", TimingValue(60))
    # DSP48E2 (clock-to-out placeholder)
    dsp = chip.timing.add_cell_variant(speed, "DSP48E2")
    dsp.add_clock_out("CLK", "P[0]", ClockEdge.RISING, TimingValue(800))
    # RAMB36E2
    bram = chip.timing.add_cell_variant(speed, "RAMB36E2")
    bram.add_clock_out("CLK", "DO[0]", ClockEdge.RISING, TimingValue(900))


def main():
    parser = argparse.ArgumentParser(description="Generate Ultrascale-like chipdb from design.scl")
    parser.add_argument("output", help="Output .bba file")
    parser.add_argument("--scl", default=DEFAULT_SCL, help="Reference design.scl to read SITEMAP from")
    args = parser.parse_args()

    X, Y, sites = parse_scl_sitemap(args.scl)
    counts = defaultdict(int)
    for ty in sites.values():
        counts[ty] += 1
    null_cells = X * Y - sum(counts.values())
    print(f"Grid: {X} x {Y} ({X * Y} cells)")
    for ty in ("SLICE", "DSP", "BRAM", "IO"):
        print(f"  {ty:>5}: {counts[ty]} tiles")
    print(f"  NULL : {null_cells} tiles (gap-fill)")
    print(
        f"Resource budget: "
        f"LUT={counts['SLICE'] * N_LUT_PER_SLICE} "
        f"FF={counts['SLICE'] * N_FF_PER_SLICE} "
        f"DSP={counts['DSP']} "
        f"BRAM={counts['BRAM']} "
        f"IO={counts['IO'] * N_IO_PER_TILE}"
    )

    chip = Chip("ultrascale", "ULTRASCALE", X, Y)
    chip.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    chip.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))

    create_slice_tiletype(chip)
    create_dsp_tiletype(chip)
    create_bram_tiletype(chip)
    create_io_tiletype(chip)
    create_null_tiletype(chip)
    create_tieoff_tiletype(chip)

    layout = {}
    for x in range(X):
        for y in range(Y):
            ty = sites.get((x, y), "NULL")
            layout[(x, y)] = ty

    # Replace one NULL slot with a TIEOFF tile to host GND_DRV/VCC_DRV BELs.
    # Pick the NULL slot nearest the die centre: a corner slot has every long
    # wire truncated, which shows up as a bogus capacity histogram for the
    # one-and-only tile of this type.
    tieoff_xy = min(
        ((x, y) for (x, y), ty in layout.items() if ty == "NULL"),
        key=lambda xy: abs(xy[0] - X // 2) + abs(xy[1] - Y // 2),
        default=None,
    )
    if tieoff_xy is None:
        raise RuntimeError("No NULL slot available to host TIEOFF tile")
    layout[tieoff_xy] = "TIEOFF"
    print(f"  TIEOFF: 1 tile at {tieoff_xy}")

    for (x, y), ty in layout.items():
        chip.set_tile_type(x, y, ty)

    per_dir = {d: sum(v.values()) for d, v in SPAN_WIRE_COUNTS.items()}
    print(
        f"Routing budget: {span_origin_count()} node origins/tile "
        f"(E={per_dir['E']} W={per_dir['W']} N={per_dir['N']} S={per_dir['S']}), "
        f"spans {sorted({length for v in SPAN_WIRE_COUNTS.values() for length in v})}, "
        f"hub={SWITCH_NODES}"
    )
    print("Building nodes...")
    create_nodes(chip, X, Y, layout)
    print("Setting timings...")
    set_timings(chip)
    chip.strs.known_id_count = 0
    print(f"Writing BBA to {args.output}...")
    chip.write_bba(args.output)
    print("Done.")


if __name__ == "__main__":
    main()

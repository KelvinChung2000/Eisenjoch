# ruff: noqa: E402, F403, F405
"""Generate a hybrid chipdb: real XC7 tile types + synthetic/placeable BELs.

Copies all tile types from prjxray with their real wires and PIPs.
For CLB tiles (CLBLL_L, CLBLM_R, etc.), replaces SLICEL/SLICEM sites
with our synthetic LUT6/DFF/DLATCH BELs mapped to the real site pin wires.
For BRAM/DSP tiles, exposes placeable BELs using the real site pin wires.
All other tile types get their tileconn-derived wires for BFS traversal.

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
import time
import re
from collections import Counter, defaultdict
from os import path

CPP_NEXTPNR = "/home/kelvin/nextpnr"
BBASM = path.join(CPP_NEXTPNR, "build", "bba", "bbasm")
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.chip import *  # noqa: E402

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(path.dirname(HERE)))
FIXTURES_DIR = path.join(ROOT, "crates", "nextpnr", "tests", "fixtures")
CHIPDB_DIR = path.join(ROOT, "chip_database")

# Synthetic arch parameters
K = 6  # LUT6 to match real XC7 SLICEL
N_IO = 2
N_CLK = 2

BRAM_TILE_TYPES = {"BRAM_L", "BRAM_R"}
DSP_TILE_TYPES = {"DSP_L", "DSP_R"}
BUFG_SITE_TYPES = {"BUFGCTRL"}
BRAM_SITE_TYPES = {
    "RAMB18E1",
    "FIFO18E1",
    "RAMB36E1",
    "FIFO36E1",
    "RAMBFIFO36E1",
}
DSP_SITE_TYPES = {"DSP48E1"}
CLOCKING_SITE_TYPES = {
    "BUFHCE",
    "BUFIO",
    "BUFR",
    "BUFMRCE",
    "MMCME2_ADV",
    "PLLE2_ADV",
    "IDELAYCTRL",
}
SERDES_SITE_TYPES = {
    "IDELAYE2",
    "ILOGICE3",
    "OLOGICE3",
}
CONFIG_SITE_TYPES = {
    "BSCAN",
    "CAPTURE",
    "DCIRESET",
    "FRAME_ECC",
    "ICAP",
    "STARTUP",
    "USR_ACCESS",
    "XADC",
}
IMPORTANT_SITE_TYPES = (
    BRAM_SITE_TYPES
    | DSP_SITE_TYPES
    | CLOCKING_SITE_TYPES
    | SERDES_SITE_TYPES
    | CONFIG_SITE_TYPES
)

VECTOR_PIN_PREFIXES = (
    "ADDRARDADDR",
    "ADDRBWRADDR",
    "ADDRATIEHIGH",
    "ADDRBTIEHIGH",
    "ALUMODE",
    "A",
    "ACIN",
    "ACOUT",
    "B",
    "BCIN",
    "BCOUT",
    "C",
    "CARRYOUT",
    "CASCADEINA",
    "CASCADEINB",
    "CASCADEOUTA",
    "CASCADEOUTB",
    "DIADI",
    "DIBDI",
    "DIPADIP",
    "DIPBDIP",
    "DOADO",
    "DOBDO",
    "DOPADOP",
    "DOPBDOP",
    "DI",
    "DO",
    "DADDR",
    "INMODE",
    "OPMODE",
    "P",
    "PATTERN",
    "PCIN",
    "PCOUT",
    "RDADDR",
    "RDCOUNT",
    "TESTIN",
    "WEA",
    "WEBWE",
    "WRADDR",
    "WRCOUNT",
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _input_hash(script_path, xray, tilegrid_path, tileconn_path, size):
    """SHA-256 over this script + key input files + parameters."""
    h = hashlib.sha256()
    for fpath in (script_path, tilegrid_path, tileconn_path):
        with open(fpath, "rb") as f:
            h.update(f.read())
    h.update((size or "full").encode())
    return h.hexdigest()


def _load_json_if_path(value):
    if isinstance(value, (str, os.PathLike)):
        with open(value) as f:
            return json.load(f)
    return value


def _is_cached(output_bin, expected_hash):
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


def _pip_timing_class(src_wire, dst_wire):
    """Classify a PIP's timing based on src/dst wire names.

    The goal is to make deeply nested switch matrix paths expensive so the
    A* router avoids exploring them unless necessary. Inter-tile span wires
    are cheap (they advance position). Internal mux wires are expensive
    (they explore the switch matrix without getting closer to the target).

    Cost hierarchy (low → high):
    - SPAN: inter-tile routing wires (EE, WW, NN, SS, BEG/END) — advance position
    - DELIVER: final delivery to BEL pins (IMUX, LOGIC_OUTS, BYP, CLK)
    - MUX: internal switch matrix (BOUNCE, FAN, GFAN) — deep mux nesting
    """
    # Check destination wire for classification
    d = dst_wire
    if any(d.startswith(p) for p in ("EE", "WW", "NN", "SS", "NE", "NW", "SE", "SW",
                                      "EL", "ER", "NL", "NR", "SL", "SR", "WL", "WR")):
        return "SPAN"
    if "IMUX" in d or "LOGIC_OUTS" in d or "BYP" in d or "CLK" in d:
        return "DELIVER"
    # Internal mux wires: BOUNCE, FAN, GFAN, CTRL, etc.
    return "MUX"


def _wire_type_from_xray_name(wname):
    """Assign coarse nextpnr wire classes while preserving prjxray wire names."""
    upper = wname.upper()
    if "BUFG" in upper:
        return "BUFG"
    if "GCLK" in upper:
        return "GCLK"
    if "HCLK" in upper:
        return "HCLK"
    if "CLK" in upper and "LOGIC" not in upper:
        return "CLK"
    return "ROUTING"


def add_slice_bels(tt, site_data, z_offset, dff_copies=1, include_latches=True):
    """Replace one SLICEL/SLICEM site with synthetic LUT6/DFF BELs.

    Maps BEL pins directly to real site pin wires from prjxray:
    - LUT6 I[0..5] → A1..A6 (or B1..B6, C1..C6, D1..D6)
    - LUT6 F → A (LUT output wire)
    - DFF D → AX (bypass wire), CLK → CLK, Q → AQ
    """
    pins = site_data["site_pins"]

    # Derive wire prefix from a known pin (e.g., "A1" → "CLBLL_L_A1" → "CLBLL_L_")
    a1_wire = pins["A1"]["wire"]
    prefix = a1_wire[: a1_wire.rfind("A1")]

    for i, letter in enumerate("ABCD"):
        z = z_offset + i * (1 + dff_copies + int(include_latches))

        # LUT6: 6 inputs, 1 output
        lut = tt.create_bel(f"{prefix}{letter}_LUT", "LUT6", z=z)
        for j in range(K):
            pin_name = f"{letter}{j + 1}"
            tt.add_bel_pin(lut, f"I[{j}]", pins[pin_name]["wire"], PinType.INPUT)
        tt.add_bel_pin(lut, "F", pins[letter]["wire"], PinType.OUTPUT)

        # DFF: data from bypass (AX/BX/CX/DX), clock, output (AQ/BQ/CQ/DQ).
        for copy_idx in range(dff_copies):
            suffix = "_FF" if dff_copies == 1 else f"_FF{copy_idx}"
            ff = tt.create_bel(f"{prefix}{letter}{suffix}", "DFF", z=z + 1 + copy_idx)
            tt.add_bel_pin(ff, "D", pins[f"{letter}X"]["wire"], PinType.INPUT)
            tt.add_bel_pin(ff, "CLK", pins["CLK"]["wire"], PinType.INPUT)
            tt.add_bel_pin(ff, "Q", pins[f"{letter}Q"]["wire"], PinType.OUTPUT)

        if include_latches:
            # DLATCH: same pins as DFF
            latch = tt.create_bel(f"{prefix}{letter}_LATCH", "DLATCH", z=z + 1 + dff_copies)
            tt.add_bel_pin(latch, "D", pins[f"{letter}X"]["wire"], PinType.INPUT)
            tt.add_bel_pin(latch, "G", pins["CLK"]["wire"], PinType.INPUT)
            tt.add_bel_pin(latch, "Q", pins[f"{letter}Q"]["wire"], PinType.OUTPUT)

        # LUT output → FF data bypass (so LUT can feed FF directly)
        tt.create_pip(pins[letter]["wire"], pins[f"{letter}X"]["wire"])

    # GND/VCC to all BEL input pins
    for letter in "ABCD":
        for j in range(1, K + 1):
            tt.create_pip("GND", pins[f"{letter}{j}"]["wire"])
            tt.create_pip("VCC", pins[f"{letter}{j}"]["wire"])
        tt.create_pip("GND", pins[f"{letter}X"]["wire"])
        tt.create_pip("VCC", pins[f"{letter}X"]["wire"])
    tt.create_pip("GND", pins["CLK"]["wire"])
    tt.create_pip("VCC", pins["CLK"]["wire"])


def _infer_pin_dir(site_type, pin_name):
    """Best-effort pin direction inference for BRAM/DSP site pins.

    prjxray tile-type JSON tells us which tile wire a site pin connects to, but
    not the pin direction. For the hybrid chipdb we only need a practical
    direction model that matches the mainstream XC7 primitive interfaces well
    enough for placement/routing experiments.
    """
    pin = pin_name.upper()
    st = site_type.upper()

    if st in {"BUFHCE", "BUFIO", "BUFR", "BUFMRCE"}:
        return PinType.OUTPUT if pin == "O" else PinType.INPUT

    if st in {"MMCME2_ADV", "PLLE2_ADV"}:
        clocking_out_prefixes = ("CLKFBOUT", "CLKOUT", "DO", "DRDY", "LOCKED", "PSDONE")
        return PinType.OUTPUT if pin.startswith(clocking_out_prefixes) else PinType.INPUT

    if st == "IDELAYCTRL":
        return PinType.OUTPUT if pin == "RDY" else PinType.INPUT

    if st in {"IDELAYE2", "ILOGICE3", "OLOGICE3"}:
        serdes_out_prefixes = ("CNTVALUEOUT", "DATAOUT", "DDLY", "OFB", "OQ", "SHIFTOUT")
        return PinType.OUTPUT if pin.startswith(serdes_out_prefixes) else PinType.INPUT

    if st.startswith("DSP48"):
        dsp_out_prefixes = (
            "ACOUT",
            "BCOUT",
            "CARRYCASCOUT",
            "CARRYOUT",
            "MULTSIGNOUT",
            "OVERFLOW",
            "P",
            "PATTERNBDETECT",
            "PATTERNDETECT",
            "PCOUT",
            "UNDERFLOW",
            "XOROUT",
        )
        return PinType.OUTPUT if pin.startswith(dsp_out_prefixes) else PinType.INPUT

    if any(st.startswith(prefix) for prefix in ("RAMB", "FIFO")):
        bram_out_prefixes = (
            "CASCADEOUT",
            "DBITERR",
            "DO",
            "DOP",
            "ECCPARITY",
            "RDADDRECC",
            "SBITERR",
        )
        return PinType.OUTPUT if pin.startswith(bram_out_prefixes) else PinType.INPUT

    return PinType.INPUT


def _site_bel_type(site_type):
    """Return the BEL/cell type name to expose in the hybrid chipdb."""
    return site_type


def _site_bel_types(site_type):
    """Return BEL/cell type aliases for a prjxray site.

    The prjxray Artix-7 BRAM tile-type JSON uses RAMBFIFO36E1 for the 36K site
    and splits 18K site views into RAMB18E1/FIFO18E1. Yosys and user netlists
    commonly refer to RAMB36E1, FIFO36E1, RAMB18E1, or FIFO18E1 directly, so
    expose those aliases while keeping the tile wires from prjxray.
    """
    if site_type == "RAMBFIFO36E1":
        return ("RAMBFIFO36E1", "RAMB36E1", "FIFO36E1")
    if site_type in {"RAMB18E1", "FIFO18E1"}:
        return ("RAMB18E1", "FIFO18E1")
    return (_site_bel_type(site_type),)


def _logical_pin_names(pin_name):
    """Return raw and Yosys-style bracketed names for vector site pins."""
    names = [pin_name]
    match = re.fullmatch(r"([A-Z_]+?)([LU]?)([0-9]+)", pin_name.upper())
    if not match:
        return names

    base, half, index = match.groups()
    if base not in VECTOR_PIN_PREFIXES:
        return names

    bracketed = f"{base}[{index}]"
    if bracketed not in names:
        names.append(bracketed)
    if half:
        half_name = f"{base}{half}[{index}]"
        if half_name not in names:
            names.append(half_name)
    return names


def add_site_bel(tt, site_data, z, bel_type=None, name_suffix=None):
    """Expose one prjxray site as a placeable BEL using its tile wires.

    Note: BRAM_L/R tiles can contain overlapping 18K and 36K site views.
    The hybrid chipdb exposes site-level primitive aliases directly, but it
    does not yet model the exclusivity relation between overlapping aliases.
    """
    site_type = site_data["type"]
    pins = site_data.get("site_pins", {})
    bel_type = bel_type or _site_bel_type(site_type)
    bel_name = site_data["name"]
    if name_suffix:
        bel_name = f"{bel_name}_{name_suffix}"
    bel = tt.create_bel(bel_name, bel_type, z=z)

    seen_pins = set()
    for pin_name, pin_data in sorted(pins.items()):
        if pin_data is None:
            continue
        wire = pin_data["wire"]
        direction = _infer_pin_dir(site_type, pin_name)
        for logical_name in _logical_pin_names(pin_name):
            if logical_name in seen_pins:
                continue
            seen_pins.add(logical_name)
            tt.add_bel_pin(bel, logical_name, wire, direction)

    return bel


def add_bufg_bel(tt, site_data, z):
    """Expose one real BUFGCTRL site as a simplified BUFG BEL.

    This is one of the intentional simplifications of the hybrid/exact chipdb:
    the tile, wires, and PIPs remain prjxray-derived, but the user-facing BEL
    type matches the BUFG primitive emitted by Yosys.
    """
    pins = site_data.get("site_pins", {})
    in_pin = "I" if "I" in pins else "I0" if "I0" in pins else None
    out_pin = "O" if "O" in pins else None
    if in_pin is None or out_pin is None:
        raise RuntimeError(
            f"BUFGCTRL site {site_data.get('name')} is missing an I/I0 or O pin"
        )

    bel = tt.create_bel(site_data["name"], "BUFG", z=z)
    tt.add_bel_pin(bel, "I", pins[in_pin]["wire"], PinType.INPUT)
    tt.add_bel_pin(bel, "O", pins[out_pin]["wire"], PinType.OUTPUT)
    return bel


def create_tile_from_xray(chip, xray_root, xc7_type, tc_wires):
    """Create a tile type by copying wires/PIPs from prjxray + tileconn wires.

    Returns (tile_type, xray_data_or_None).
    """
    data = load_tile_type_json(xray_root, xc7_type)
    tt = chip.create_tile_type(xc7_type)
    created = set()

    # 1. Copy all wires from tile_type JSON
    if data:
        for wname in data.get("wires", {}):
            tt.create_wire(wname, _wire_type_from_xray_name(wname))
            created.add(wname)

    # 2. Add tileconn wires not already present
    for wname in tc_wires:
        if wname not in created:
            tt.create_wire(wname, "ROUTING")
            created.add(wname)

    # 3. Copy all PIPs from tile_type JSON with wire-type costs.
    if data:
        for pdata in data.get("pips", {}).values():
            src, dst = pdata["src_wire"], pdata["dst_wire"]
            if src in created and dst in created:
                tc = _pip_timing_class(src, dst)
                tt.create_pip(src, dst, timing_class=tc)

    # 4. GND/VCC
    if "GND" not in created:
        tt.create_wire("GND", "GND", const_value="GND")
    if "VCC" not in created:
        tt.create_wire("VCC", "VCC", const_value="VCC")

    return tt, data, created


def create_io_tile(chip, side="L"):
    """IO tile with IOB BELs and bridge wires to INT routing."""
    suffix = "IO_L" if side == "L" else "IO_R" if side == "R" else "IOB"
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

    for i in range(N_IO):
        tt.create_wire(f"IO_BRIDGE_IN{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_OUT{i}", "IO_BRIDGE")
        tt.create_wire(f"IO_BRIDGE_T{i}", "IO_BRIDGE")
        tt.create_pip(f"IO{i}_O", f"IO_BRIDGE_OUT{i}")
        tt.create_pip(f"IO_BRIDGE_IN{i}", f"IO{i}_I")
        tt.create_pip(f"IO_BRIDGE_T{i}", f"IO{i}_T")

    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
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

def generate_xc7_hybrid(
    output_bba,
    xray,
    tilegrid_path,
    tileconn_path,
    size=None,
    chip_name="xc7_hybrid",
    device_name="XC7A50T",
    include_bufg=False,
    synthetic_tile_builders=None,
    synthetic_tileconn=None,
    direct_tileconn_types=None,
):
    """Generate the hybrid chipdb BBA file."""
    tilegrid = _load_json_if_path(tilegrid_path)
    tileconn = _load_json_if_path(tileconn_path)
    if synthetic_tileconn:
        tileconn = list(tileconn) + list(synthetic_tileconn)
    synthetic_tile_builders = synthetic_tile_builders or {}
    direct_tileconn_types = set(direct_tileconn_types or ())

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
    io_types = {"LIOB33", "RIOB33", "LIOB33_SING", "RIOB33_SING", "IOB"}
    left_io = {"LIOB33", "LIOB33_SING"}
    right_io = {"RIOB33", "RIOB33_SING"}
    clb_types = {"CLBLL_L", "CLBLL_R", "CLBLM_L", "CLBLM_R", "CLB"}

    # ------------------------------------------------------------------
    # Derive wire sets from tileconn (for all tile types)
    # ------------------------------------------------------------------
    tc_wire_sets = defaultdict(set)
    for conn in tileconn:
        a_type, b_type = conn["tile_types"]
        for wa, wb in conn["wire_pairs"]:
            tc_wire_sets[a_type].add(wa)
            tc_wire_sets[b_type].add(wb)

    # ------------------------------------------------------------------
    # Create tile types
    # ------------------------------------------------------------------
    ch = Chip(chip_name, device_name, X, Y)
    ch.strs.read_constids(path.join(FIXTURES_DIR, "constids.inc"))
    ch.read_gfxids(path.join(FIXTURES_DIR, "gfxids.inc"))

    # IO tiles (synthetic, with IOB BELs + bridge wires)
    create_io_tile(ch, side="L")
    create_io_tile(ch, side="R")
    create_io_tile(ch, side="B")
    create_null_tile(ch)

    # All prjxray tile types present in the tilegrid. Synthetic composite types
    # are created through callbacks so the type data stays deduplicated.
    synthetic_types = set(synthetic_tile_builders)
    all_xc7_types = sorted(set(xc7_grid.values()) - {"NULL"} - io_types - synthetic_types)

    # Track which tile types we've created and their wire sets
    tile_wire_sets = {}  # xc7_type -> set of wire names
    bels_per_type = {}
    bel_capacity = Counter()
    bel_types_created = set()

    for synthetic_type, builder in sorted(synthetic_tile_builders.items()):
        info = builder(ch)
        tile_wire_sets[synthetic_type] = set(info.get("wire_set", set()))
        bels_per_type[synthetic_type] = Counter(info.get("bel_counts", Counter()))
        bel_capacity.update(bels_per_type[synthetic_type])
        bel_types_created.update(info.get("bel_types", set()))

    for xc7_type in all_xc7_types:
        tc_wires = tc_wire_sets.get(xc7_type, set())
        tt, data, created_wires = create_tile_from_xray(ch, xray, xc7_type, tc_wires)
        tile_wire_sets[xc7_type] = created_wires

        # CLB tiles: add LUT6/DFF/DLATCH BELs replacing SLICEL/SLICEM sites
        if xc7_type in clb_types and data:
            for idx, site in enumerate(data.get("sites", [])):
                if site.get("type") in ("SLICEL", "SLICEM"):
                    add_slice_bels(tt, site, z_offset=idx * 12)
                    bel_capacity["LUT6"] += 4
                    bel_capacity["DFF"] += 4
                    bel_types_created.update(("LUT6", "DFF", "DLATCH"))

        if data:
            z = 0
            for site in data.get("sites", []):
                st = site.get("type")
                if st in IMPORTANT_SITE_TYPES:
                    for bel_type in _site_bel_types(st):
                        suffix = bel_type if bel_type != st else st
                        add_site_bel(tt, site, z, bel_type=bel_type, name_suffix=suffix)
                        z += 1
                        bel_capacity[bel_type] += 1
                        bel_types_created.add(bel_type)

        if include_bufg and data:
            z = 0
            for site in data.get("sites", []):
                if site.get("type") in BUFG_SITE_TYPES:
                    add_bufg_bel(tt, site, z)
                    z += 1
                    bel_capacity["BUFG"] += 1
                    bel_types_created.add("BUFG")

        # INT tiles: add IO bridge wires + PIPs for IO connectivity
        if xc7_type in int_types:
            imux = sorted([w for w in created_wires if "IMUX" in w])
            logic_outs = sorted([w for w in created_wires if "LOGIC_OUTS" in w])
            for i in range(N_IO):
                for bname in (f"IO_BRIDGE_IN{i}", f"IO_BRIDGE_OUT{i}", f"IO_BRIDGE_T{i}"):
                    tt.create_wire(bname, "IO_BRIDGE")
                    created_wires.add(bname)
                # Bridge out (from IO pad) → can reach any IMUX
                for mux_wire in imux:
                    tt.create_pip(f"IO_BRIDGE_OUT{i}", mux_wire)
                # Any LOGIC_OUTS → bridge in (to IO pad)
                for lo_wire in logic_outs:
                    tt.create_pip(lo_wire, f"IO_BRIDGE_IN{i}")
                    tt.create_pip(lo_wire, f"IO_BRIDGE_T{i}")

    bels_per_type.update({xc7_type: Counter() for xc7_type in all_xc7_types})
    for xc7_type in all_xc7_types:
        data = load_tile_type_json(xray, xc7_type)
        if not data:
            continue
        for site in data.get("sites", []):
            st = site.get("type")
            if st in ("SLICEL", "SLICEM"):
                bels_per_type[xc7_type]["LUT6"] += 4
                bels_per_type[xc7_type]["DFF"] += 4
            elif st in IMPORTANT_SITE_TYPES:
                for bel_type in _site_bel_types(st):
                    bels_per_type[xc7_type][bel_type] += 1
            elif include_bufg and st in BUFG_SITE_TYPES:
                bels_per_type[xc7_type]["BUFG"] += 1
    print(
        f"Created {len(all_xc7_types)} tile types from prjxray"
        f" and {len(synthetic_types)} synthetic tile types"
    )

    # ------------------------------------------------------------------
    # Tile assignment
    # ------------------------------------------------------------------
    n_int = n_clb = n_io = n_routing = n_null = 0
    for y in range(Y):
        for x in range(X):
            xc7_type = xc7_grid.get((x, y), "NULL")
            if xc7_type == "IOB":
                ch.set_tile_type(x, y, "IOB")
                n_io += 1
            elif xc7_type in left_io:
                ch.set_tile_type(x, y, "IO_L")
                n_io += 1
            elif xc7_type in right_io:
                ch.set_tile_type(x, y, "IO_R")
                n_io += 1
            elif xc7_type in tile_wire_sets:
                ch.set_tile_type(x, y, xc7_type)
                if xc7_type in int_types:
                    n_int += 1
                elif xc7_type in clb_types:
                    n_clb += 1
                else:
                    n_routing += 1
            else:
                ch.set_tile_type(x, y, "NULL")
                n_null += 1

    # Count total BELs across all tile instances
    total_caps = Counter()
    for y in range(Y):
        for x in range(X):
            xc7_type = xc7_grid.get((x, y), "NULL")
            if xc7_type in bels_per_type:
                for bel_type, count in bels_per_type[xc7_type].items():
                    total_caps[bel_type] += count
    print(f"Tile assignment: {n_int} INT, {n_clb} CLB, {n_io} IO, {n_routing} routing, {n_null} NULL")
    capacity_parts = [
        f"~{total_caps['LUT6']} LUT6s",
        f"~{total_caps['DFF']} DFFs",
        f"~{total_caps['DSP48E1']} DSP48E1s",
        f"~{total_caps['RAMB18E1']} RAMB18E1s",
        f"~{total_caps['RAMB36E1']} RAMB36E1s",
        f"~{total_caps['FIFO18E1']} FIFO18E1s",
        f"~{total_caps['FIFO36E1']} FIFO36E1s",
    ]
    if include_bufg:
        capacity_parts.append(f"~{total_caps['BUFG']} BUFGs")
    for bel_type in sorted(total_caps):
        if bel_type not in {
            "LUT6",
            "DFF",
            "DSP48E1",
            "RAMB18E1",
            "RAMB36E1",
            "FIFO18E1",
            "FIFO36E1",
            "BUFG",
        }:
            capacity_parts.append(f"~{total_caps[bel_type]} {bel_type}s")
    print("Capacity: " + ", ".join(capacity_parts))

    # ------------------------------------------------------------------
    # Inter-tile routing nodes from tileconn
    # ------------------------------------------------------------------
    print("Creating inter-tile routing nodes from tileconn...")

    tiles_by_xc7_type = defaultdict(list)
    for (x, y), ttype in xc7_grid.items():
        tiles_by_xc7_type[ttype].append((x, y))

    # Load full wire sets for INT tiles from prjxray
    int_l_data = load_tile_type_json(xray, "INT_L")
    int_r_data = load_tile_type_json(xray, "INT_R")
    int_l_wires = set(int_l_data["wires"].keys()) if int_l_data else set()
    int_r_wires = set(int_r_data["wires"].keys()) if int_r_data else set()

    # Wire sets for BFS: union of prjxray wires + tileconn wires per type
    wire_sets = dict(tc_wire_sets)
    wire_sets["INT_L"] = int_l_wires | tc_wire_sets.get("INT_L", set())
    wire_sets["INT_R"] = int_r_wires | tc_wire_sets.get("INT_R", set())
    # Add full CLB wire sets (prjxray wires + tileconn wires)
    for ct in clb_types:
        clb_data = load_tile_type_json(xray, ct)
        if clb_data:
            wire_sets[ct] = set(clb_data["wires"].keys()) | tc_wire_sets.get(ct, set())
        else:
            wire_sets[ct] = tc_wire_sets.get(ct, set())

    # INT and placed-resource tiles are node endpoints in the base hybrid mode.
    # BUFG-enabled prjxray-backed variants include all tile types as endpoints so
    # routing-only and clocking tile types stay close to the tileconn database.
    if include_bufg:
        wire_sets = {
            xc7_type: set(wires)
            for xc7_type, wires in tile_wire_sets.items()
        }
        modeled_types = set(tile_wire_sets)
    else:
        modeled_types = (int_types | clb_types | BRAM_TILE_TYPES | DSP_TILE_TYPES) & set(tile_wire_sets)
    modeled_types -= direct_tileconn_types

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

    # Union-Find for wire grouping
    wire_parent = {}

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

    # BFS: for each INT wire instance, find all reachable INT wire instances
    # through intermediate tile types (CLB, VBRK, INT_FEEDTHRU, etc.)
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
                            if not (0 <= nx < X and 0 <= ny < Y):
                                continue
                            if xc7_grid.get((nx, ny)) != ntype:
                                continue
                            visited.add(nkey)
                            if ntype in modeled_types and nwire in wire_sets.get(ntype, set()):
                                uf_union(start_key, nkey)
                                processed_starts.add(nkey)
                            next_frontier.append((nx, ny, ntype, nwire))
                    frontier = next_frontier

    # ------------------------------------------------------------------
    # Synthetic bridge across CMT NULL gaps
    # ------------------------------------------------------------------
    # The xc7a50t has NULL tiles at x=7,8 (CMT region) that disconnect
    # the IO-adjacent INT pair (x=4,5) from the main fabric (x=11+).
    # Similarly, there may be gaps on the right side near x=106-107.
    # Bridge isolated INT tiles to the nearest connected INT tile of the
    # same type by unioning their shared wires.
    clb_adjacent_int_cols = set()
    for (x, y), t in xc7_grid.items():
        if t in clb_types:
            for dx in [-1, 1]:
                if xc7_grid.get((x + dx, y)) in int_types:
                    clb_adjacent_int_cols.add(x + dx)

    bridge_count = 0
    for y in range(Y):
        row_int = [(x, xc7_grid[(x, y)]) for x in range(X)
                   if xc7_grid.get((x, y)) in int_types]
        if not row_int:
            continue
        isolated = [(x, t) for x, t in row_int if x not in clb_adjacent_int_cols]
        connected = [(x, t) for x, t in row_int if x in clb_adjacent_int_cols]
        if not isolated or not connected:
            continue
        for iso_x, iso_type in isolated:
            best = None
            best_dist = 999
            for con_x, con_type in connected:
                if con_type == iso_type and abs(con_x - iso_x) < best_dist:
                    best_dist = abs(con_x - iso_x)
                    best = con_x
            if best is None:
                continue
            # Union shared wires (same name in both tiles)
            shared = wire_sets.get(iso_type, set())
            for w in shared:
                iso_key = (iso_x, y, w)
                con_key = (best, y, w)
                if iso_key in wire_parent and con_key in wire_parent:
                    uf_union(iso_key, con_key)
                    bridge_count += 1

    print(f"Bridged {bridge_count} wires across CMT gaps")

    # Collect groups and create nodes
    groups = defaultdict(list)
    for key in wire_parent:
        root = uf_find(key)
        groups[root].append(key)

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

    if direct_tileconn_types:
        print("Creating synthetic inter-tile routing components...")
        direct_parent = {}
        direct_rank = {}

        def direct_find(key):
            while direct_parent[key] != key:
                direct_parent[key] = direct_parent[direct_parent[key]]
                key = direct_parent[key]
            return key

        def direct_add(key):
            if key not in direct_parent:
                direct_parent[key] = key
                direct_rank[key] = 0

        def direct_union(a, b):
            direct_add(a)
            direct_add(b)
            ra, rb = direct_find(a), direct_find(b)
            if ra == rb:
                return False
            if direct_rank[ra] < direct_rank[rb]:
                ra, rb = rb, ra
            direct_parent[rb] = ra
            if direct_rank[ra] == direct_rank[rb]:
                direct_rank[ra] += 1
            return True

        direct_pair_count = 0
        direct_union_count = 0
        direct_overlap_count = 0
        for conn in tileconn:
            a_type = conn["tile_types"][0]
            b_type = conn["tile_types"][1]
            if a_type not in direct_tileconn_types or b_type not in direct_tileconn_types:
                continue
            dx = conn.get("grid_deltas", [0, 0])[0]
            dy = conn.get("grid_deltas", [0, 0])[1]
            a_wires = tile_wire_sets.get(a_type, set())
            b_wires = tile_wire_sets.get(b_type, set())
            for ax, ay in tiles_by_xc7_type.get(a_type, []):
                bx, by = ax + dx, ay + dy
                if not (0 <= bx < X and 0 <= by < Y):
                    continue
                if xc7_grid.get((bx, by)) != b_type:
                    continue
                for aw, bw in conn["wire_pairs"]:
                    if aw not in a_wires or bw not in b_wires:
                        continue
                    a_key = (ax, ay, aw)
                    b_key = (bx, by, bw)
                    if a_key in wires_in_nodes or b_key in wires_in_nodes:
                        direct_overlap_count += 1
                        continue
                    direct_pair_count += 1
                    if direct_union(a_key, b_key):
                        direct_union_count += 1

        direct_groups = defaultdict(list)
        for key in direct_parent:
            direct_groups[direct_find(key)].append(key)

        direct_node_count = 0
        direct_wire_count = 0
        for members in direct_groups.values():
            if len(members) < 2:
                continue
            ch.add_node([NodeWire(x, y, w) for x, y, w in members])
            for m in members:
                wires_in_nodes.add(m)
            direct_wire_count += len(members)
            direct_node_count += 1
        print(
            "Created "
            f"{direct_node_count} synthetic inter-tile routing nodes "
            f"from {direct_pair_count} tileconn pairs "
            f"({direct_wire_count} wires, {direct_union_count} unions, "
            f"{direct_overlap_count} pre-owned pairs skipped)"
        )

    # ------------------------------------------------------------------
    # IO-to-INT bridge nodes
    # ------------------------------------------------------------------
    print("Creating IO-to-INT bridge nodes...")
    io_node_count = 0

    # Only use CLB-adjacent INT tiles for IO bridges (skip isolated IO-adjacent
    # INT tiles that are separated from the main fabric by CMT NULL gaps).
    clb_adjacent_int_cols = set()
    for (x, y), t in xc7_grid.items():
        if t in clb_types:
            for dx in [-1, 1]:
                if xc7_grid.get((x + dx, y)) in int_types:
                    clb_adjacent_int_cols.add(x + dx)

    int_positions = {}
    for itype in ("INT_L", "INT_R"):
        for ix, iy in tiles_by_xc7_type.get(itype, []):
            if ix in clb_adjacent_int_cols:
                int_positions.setdefault(iy, []).append((ix, iy, itype))
    if "CLB" in tile_wire_sets:
        for ix, iy in tiles_by_xc7_type.get("CLB", []):
            int_positions.setdefault(iy, []).append((ix, iy, "CLB"))

    for io_type in sorted(io_types):
        is_left = io_type in left_io
        target_int = "INT_L" if is_left else "INT_R"

        for io_x, io_y in tiles_by_xc7_type.get(io_type, []):
            if not (0 <= io_x < X and 0 <= io_y < Y):
                continue

            best_int = None
            best_dist = 999
            for dy in range(0, 4):
                for sign in (0, 1, -1):
                    row = io_y + dy * sign if dy > 0 else io_y
                    for ix, iy, itype in int_positions.get(row, []):
                        if itype in {target_int, "CLB"} and abs(ix - io_x) < best_dist:
                            best_dist = abs(ix - io_x)
                            best_int = (ix, iy)
                if best_int is not None:
                    break
            if best_int is None:
                continue

            int_x, int_y = best_int
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
    clk_int_wires = sorted([w for w in int_l_wires if "CLK" in w and "LOGIC" not in w])
    if "CLB" in tile_wire_sets:
        clk_int_wires = sorted(
            [w for w in tile_wire_sets["CLB"] if "CLK" in w and "LOGIC" not in w]
        )
    if not clk_int_wires:
        clk_int_wires = sorted([w for w in int_l_wires if "GCLK" in w])
    clk_wire_names = clk_int_wires[:N_CLK] if clk_int_wires else []
    print(f"Clock wires: {clk_wire_names}")

    int_l_columns = defaultdict(list)
    for (x, y), ttype in xc7_grid.items():
        if ttype == "INT_L" and 0 <= x < X and 0 <= y < Y:
            int_l_columns[x].append(y)
        if ttype == "CLB" and 0 <= x < X and 0 <= y < Y:
            int_l_columns[x].append(y)
    for x in int_l_columns:
        int_l_columns[x].sort()

    clk_node_count = 0

    # Vertical clock chains
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

    # Horizontal clock spines
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

    # Connect IO clock outputs to nearest INT_L clock wire
    io_tiles = (tiles_by_xc7_type.get("LIOB33", [])
                + tiles_by_xc7_type.get("RIOB33", [])
                + tiles_by_xc7_type.get("LIOB33_SING", [])
                + tiles_by_xc7_type.get("RIOB33_SING", [])
                + tiles_by_xc7_type.get("IOB", []))
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

    # Large composite fabrics use a simplified CLK_BUFG sidecar tile. Bridge
    # the first clock lanes explicitly: IO GCLK output -> BUFG.I -> BUFG.O ->
    # nearest composite clock wire. This keeps the sidecar out of the fabric
    # while still making BUFG cells routable for the benchmark clocks.
    bufg_bridge_count = 0
    if "CLK_BUFG" in tile_wire_sets and clk_wire_names and int_l_col_list:
        for bufg_x, bufg_y in tiles_by_xc7_type.get("CLK_BUFG", []):
            for c in range(min(N_CLK, len(clk_wire_names), N_IO)):
                if not io_tiles:
                    continue
                io_x, io_y = min(io_tiles, key=lambda pos: abs(pos[0] - bufg_x) + abs(pos[1] - bufg_y))
                bufg_i = f"BUFG{c}_I"
                bufg_o = f"BUFG{c}_O"
                io_key = (io_x, io_y, f"GCLK{c}_OUT")
                bi_key = (bufg_x, bufg_y, bufg_i)
                if io_key not in wires_in_nodes and bi_key not in wires_in_nodes:
                    ch.add_node([NodeWire(io_x, io_y, f"GCLK{c}_OUT"), NodeWire(bufg_x, bufg_y, bufg_i)])
                    wires_in_nodes.add(io_key)
                    wires_in_nodes.add(bi_key)
                    bufg_bridge_count += 1

                best_clk = None
                for col_x in sorted(int_l_col_list, key=lambda col: abs(col - bufg_x)):
                    for row_y in sorted(int_l_columns[col_x], key=lambda row: abs(row - bufg_y)):
                        clk_key = (col_x, row_y, clk_wire_names[c])
                        if clk_key not in wires_in_nodes:
                            best_clk = (col_x, row_y, clk_key)
                            break
                    if best_clk is not None:
                        break
                bo_key = (bufg_x, bufg_y, bufg_o)
                if best_clk is not None and bo_key not in wires_in_nodes:
                    col_x, row_y, clk_key = best_clk
                    ch.add_node([NodeWire(bufg_x, bufg_y, bufg_o), NodeWire(col_x, row_y, clk_wire_names[c])])
                    wires_in_nodes.add(bo_key)
                    wires_in_nodes.add(clk_key)
                    bufg_bridge_count += 1
        print(f"Created {bufg_bridge_count} CLK_BUFG bridge nodes")

    print(f"Created {clk_node_count} clock routing nodes")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    speed = "DEFAULT"
    tmg = ch.set_speed_grades([speed])
    # Differentiated PIP costs by wire type:
    # - SPAN (inter-tile routing wires): cheap → A* prefers these
    # - DELIVER (IMUX, LOGIC_OUTS, BYP): moderate → needed at endpoints
    # - MUX (BOUNCE, FAN, internal): expensive → discourages deep switch exploration
    # DELAY_SCALE=10 is admissible: span-1 PIP costs 10 and covers 1 tile.
    for name, delay in [
        ("SPAN", 10),
        ("DELIVER", 50),
        ("MUX", 150),
        ("SWINPUT", 50),
        ("TILE_ROUTING", 10),
    ]:
        tmg.set_pip_class(
            grade=speed,
            name=name,
            delay=TimingValue(delay),
            in_cap=TimingValue(5000),
            out_res=TimingValue(1000),
        )

    for bel_type in sorted(bel_types_created):
        cell_timing = ch.timing.add_cell_variant(speed, bel_type)
        if bel_type in {"BUFG", "BUFHCE", "BUFIO", "BUFR", "BUFMRCE"}:
            cell_timing.add_comb_arc("I", "O", TimingValue(100))

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
        description="Generate hybrid chipdb: real XC7 tiles + synthetic LUT6/DFF BELs"
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

# ruff: noqa: E402, F403, F405
"""Generate a himbaechel chipdb from a FABulous project's nextpnr model.

Usage: python gen_fabulous.py <fabulous_project> <output.bba>
Reads fabric.csv, .FABulous/pips.txt and .FABulous/bel.v3.txt.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from os import path
from pathlib import Path

CPP_NEXTPNR = os.environ.get("NPNR_UPSTREAM", "/home/kelvin/nextpnr-upstream")
sys.path.append(path.join(CPP_NEXTPNR, "himbaechel"))
from himbaechel_dbgen.bba import BBAWriter  # noqa: E402
from himbaechel_dbgen.chip import *  # noqa: E402

SPEED = "DEFAULT"
PIP_CLASS = "FABULOUS_PIP"
WIRE_CLASS = "FABULOUS_WIRE"
PSEUDO_CLASS = "PSEUDO"
GLOBAL_CLASS = "GLOBAL_CLOCK"
PIP_DELAY_SCALE_NS = "pipDelayScale"
CLOCK_TILE = (0, 0)
GLOBAL_CLK = "GLOBAL_CLK"
EXTERNAL_SUFFIX = "__EXT"
NO_CONFIG_BITS = IdString(-1)


@dataclass
class FasmFeature(BBAStruct):
    feature: IdString

    def serialise(self, context: str, bba: BBAWriter):
        bba.u32(self.feature.index)


@dataclass(frozen=True)
class Pip:
    src: str
    dst: str
    delay: int
    feature: str


@dataclass(frozen=True)
class ExternalPip:
    src_tile: str
    src_wire: str
    dst_tile: str
    dst_wire: str
    delay: int


@dataclass(frozen=True)
class BelPin:
    port: str
    wire: str
    direction: PinType


@dataclass(frozen=True)
class TimingArc:
    kind: str
    fields: tuple[str, ...]
    condition: str


@dataclass(frozen=True)
class Bel:
    z_letter: str
    bel_type: str
    prefix: str
    pins: tuple[BelPin, ...]
    global_clock: bool
    timing: tuple[TimingArc, ...]


@dataclass
class TileModel:
    pips: frozenset[Pip] = frozenset()
    bels: tuple[Bel, ...] = ()
    external_srcs: set[str] = field(default_factory=set)
    external_dsts: set[str] = field(default_factory=set)


def tile_name(x: int, y: int) -> str:
    return f"X{x}Y{y}"


def read_grid(project: Path) -> list[list[str]]:
    rows = list(csv.reader((project / "fabric.csv").open()))
    begin = next(i for i, row in enumerate(rows) if row and row[0] == "FabricBegin")
    end = next(i for i, row in enumerate(rows) if row and row[0] == "FabricEnd")
    grid = [
        [cell for cell in row if cell and cell != "#"] for row in rows[begin + 1 : end]
    ]
    widths = {len(row) for row in grid}
    assert len(widths) == 1, f"fabric.csv rows differ in width: {widths}"
    return grid


def read_pips(project: Path) -> tuple[dict[str, set[Pip]], list[ExternalPip]]:
    internal: dict[str, set[Pip]] = defaultdict(set)
    external: list[ExternalPip] = []
    for line in (project / ".FABulous" / "pips.txt").open():
        if line.startswith("#") or not line.strip():
            continue
        src_tile, src_wire, dst_tile, dst_wire, delay, feature = line.strip().split(",")
        if src_tile == dst_tile:
            internal[src_tile].add(Pip(src_wire, dst_wire, int(delay), feature))
        else:
            external.append(
                ExternalPip(src_tile, src_wire, dst_tile, dst_wire, int(delay))
            )
    return internal, external


def read_settings(project: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    for line in (
        (project / ".FABulous" / "placement_estimate.txt").read_text().splitlines()
    ):
        if line.startswith("BelBegin"):
            break
        key, value = line.split("=")
        settings[key] = value
    return settings


def read_bels(project: Path) -> dict[str, list[Bel]]:
    bels: dict[str, list[Bel]] = defaultdict(list)
    current: dict | None = None
    for raw in (project / ".FABulous" / "bel.v3.txt").open():
        fields = raw.strip().split(",")
        command = fields[0]
        if not raw.strip() or command.startswith("#"):
            continue
        if command == "BelBegin":
            tile, z_letter, bel_type, prefix = fields[1:5]
            current = dict(
                tile=tile,
                z=z_letter,
                type=bel_type,
                prefix=prefix,
                pins=[],
                clk=False,
                timing=[],
            )
        elif command == "BelEnd":
            assert current is not None, "BelEnd outside a BEL block"
            bels[current["tile"]].append(
                Bel(
                    z_letter=current["z"],
                    bel_type=current["type"],
                    prefix=current["prefix"],
                    pins=tuple(current["pins"]),
                    global_clock=current["clk"],
                    timing=tuple(current["timing"]),
                )
            )
            current = None
        elif command in ("I", "O"):
            assert current is not None, f"pin outside a BEL block: {raw!r}"
            port, wire_ref = fields[1:3]
            wire_tile, wire = wire_ref.split(".")
            assert wire_tile == current["tile"], (
                f"BEL pin {wire_ref} lies outside tile {current['tile']}"
            )
            direction = PinType.INPUT if command == "I" else PinType.OUTPUT
            current["pins"].append(BelPin(port, wire, direction))
        elif command == "CFG":
            continue
        elif command == "GlobalClk":
            current["clk"] = True
        elif command in ("Delay", "SetupHold", "ClkToOut", "Clock"):
            arity = {"Delay": 3, "SetupHold": 4, "ClkToOut": 3, "Clock": 1}[command]
            values = tuple(fields[1 : 1 + arity])
            condition = ",".join(fields[1 + arity :])
            current["timing"].append(TimingArc(command, values, condition))
        else:
            raise ValueError(
                f"unsupported bel.v3.txt command {command!r} in line {raw!r}"
            )
    return bels


def tile_models(
    grid: list[list[str]],
    internal: dict[str, set[Pip]],
    external: list[ExternalPip],
    bels: dict[str, list[Bel]],
) -> dict[str, TileModel]:
    models: dict[str, TileModel] = {}
    for y, row in enumerate(grid):
        for x, tile_type in enumerate(row):
            tile = tile_name(x, y)
            pips = frozenset(internal.get(tile, set()))
            tile_bels = tuple(bels.get(tile, []))
            model = models.setdefault(tile_type, TileModel(pips=pips, bels=tile_bels))
            assert model.pips == pips, (
                f"{tile} pips differ from other {tile_type} tiles"
            )
            assert model.bels == tile_bels, (
                f"{tile} BELs differ from other {tile_type} tiles"
            )
    types = {
        tile_name(x, y): t for y, row in enumerate(grid) for x, t in enumerate(row)
    }
    for pip in external:
        models[types[pip.src_tile]].external_srcs.add(pip.src_wire)
        models[types[pip.dst_tile]].external_dsts.add(pip.dst_wire)
    return models


def const_value(wire: str) -> str:
    if wire.startswith("GND"):
        return "GND"
    if wire.startswith("VCC"):
        return "VCC"
    return ""


def create_pip(
    tt: TileType, src: str, dst: str, timing_class: str, feature: IdString
) -> None:
    pip = tt.create_pip(src, dst, timing_class=timing_class)
    pip.extra_data = FasmFeature(feature)


def create_tile_type(ch: Chip, name: str, model: TileModel) -> None:
    tt = ch.create_tile_type(name)
    wires = {w for pip in model.pips for w in (pip.src, pip.dst)}
    wires |= model.external_srcs | model.external_dsts
    wires |= {pin.wire for bel in model.bels for pin in bel.pins}
    for wire in sorted(wires):
        tt.create_wire(wire, const_value=const_value(wire))
    for wire in sorted(model.external_dsts):
        alias = f"{wire}{EXTERNAL_SUFFIX}"
        assert alias not in wires, f"{name} already has a wire named {alias}"
        tt.create_wire(alias, "EXTERNAL")
        create_pip(tt, alias, wire, WIRE_CLASS, NO_CONFIG_BITS)
    for pip in sorted(model.pips, key=lambda p: (p.src, p.dst)):
        create_pip(tt, pip.src, pip.dst, PIP_CLASS, tt.strs.id(pip.feature))
    if any(bel.global_clock for bel in model.bels):
        tt.create_wire(GLOBAL_CLK, "GLOBAL_CLK")
    for bel in model.bels:
        data = tt.create_bel(bel.z_letter, bel.bel_type, z=ord(bel.z_letter) - ord("A"))
        data.extra_data = FasmFeature(data.name)
        for pin in bel.pins:
            tt.add_bel_pin(data, pin.port, pin.wire, pin.direction)
        if bel.global_clock:
            clk_wire = f"{bel.prefix}CLK"
            tt.create_wire(clk_wire, "REG_CLK")
            tt.add_bel_pin(data, "CLK", clk_wire, PinType.INPUT)
            create_pip(tt, GLOBAL_CLK, clk_wire, GLOBAL_CLASS, NO_CONFIG_BITS)
        if bel.bel_type == "FABULOUS_LC":
            assert all(pin.port != "Q" for pin in bel.pins), (
                "FABULOUS_LC already has a Q pin"
            )
            q_wire = f"{bel.prefix}Q"
            o_wire = next(pin.wire for pin in bel.pins if pin.port == "O")
            tt.create_wire(q_wire, "LC_Q")
            tt.add_bel_pin(data, "Q", q_wire, PinType.OUTPUT)
            create_pip(tt, q_wire, o_wire, PSEUDO_CLASS, NO_CONFIG_BITS)


def create_clock_and_constant_tile(ch: Chip) -> None:
    tt = ch.create_tile_type("CLOCK_CONST")
    tt.create_wire("CLK", "GLOBAL_CLK")
    clock = tt.create_bel("CLK", "Global_Clock", z=0)
    clock.extra_data = FasmFeature(clock.name)
    tt.add_bel_pin(clock, "CLK", "CLK", PinType.OUTPUT)
    for z, (value, driver) in enumerate(
        (("GND", "GND_DRV"), ("VCC", "VCC_DRV")), start=1
    ):
        tt.create_wire(value, value, const_value=value)
        bel = tt.create_bel(driver, driver, z=z)
        bel.extra_data = FasmFeature(bel.name)
        tt.add_bel_pin(bel, value, value, PinType.OUTPUT)


def create_nodes(
    ch: Chip,
    grid: list[list[str]],
    external: list[ExternalPip],
    models: dict[str, TileModel],
) -> None:
    def xy(tile: str) -> tuple[int, int]:
        x, y = tile[1:].split("Y")
        return int(x), int(y)

    for pip in external:
        ch.add_node(
            [
                NodeWire(*xy(pip.src_tile), pip.src_wire),
                NodeWire(*xy(pip.dst_tile), f"{pip.dst_wire}{EXTERNAL_SUFFIX}"),
            ]
        )
    clock = [NodeWire(*CLOCK_TILE, "CLK")]
    for y, row in enumerate(grid):
        for x, tile_type in enumerate(row):
            if any(bel.global_clock for bel in models[tile_type].bels):
                clock.append(NodeWire(x, y, GLOBAL_CLK))
    ch.add_node(clock)


def ps(ns: str) -> TimingValue:
    return TimingValue(round(float(ns) * 1000))


def scaled_delay(delays: set[int], settings: dict[str, str]) -> TimingValue:
    assert len(delays) == 1, (
        f"one delay per pip kind is supported, found {sorted(delays)}"
    )
    return ps(str(float(settings[PIP_DELAY_SCALE_NS]) * delays.pop()))


def set_timings(
    ch: Chip,
    models: dict[str, TileModel],
    external: list[ExternalPip],
    settings: dict[str, str],
) -> None:
    tmg = ch.set_speed_grades([SPEED])
    internal_delays = {pip.delay for model in models.values() for pip in model.pips}
    tmg.set_pip_class(
        grade=SPEED, name=PIP_CLASS, delay=scaled_delay(internal_delays, settings)
    )
    tmg.set_pip_class(
        grade=SPEED,
        name=WIRE_CLASS,
        delay=scaled_delay({pip.delay for pip in external}, settings),
    )
    tmg.set_pip_class(grade=SPEED, name=PSEUDO_CLASS, delay=TimingValue(0))
    tmg.set_pip_class(grade=SPEED, name=GLOBAL_CLASS, delay=TimingValue(0))
    by_type: dict[str, tuple[TimingArc, ...]] = {}
    for model in models.values():
        for bel in model.bels:
            known = by_type.setdefault(bel.bel_type, bel.timing)
            assert known == bel.timing, f"{bel.bel_type} BELs carry different timing"
    for bel_type, arcs in by_type.items():
        if not arcs:
            continue
        cell = ch.timing.add_cell_variant(SPEED, bel_type)
        for arc in arcs:
            if arc.condition not in ("", "FF=0", "FF=1"):
                continue
            if arc.kind == "Delay":
                src, dst, delay = arc.fields
                cell.add_comb_arc(src, dst, ps(delay))
            elif arc.kind == "SetupHold":
                data, clock, setup, hold = arc.fields
                cell.add_setup_hold(clock, data, ClockEdge.RISING, ps(setup), ps(hold))
            elif arc.kind == "ClkToOut":
                out, clock, delay = arc.fields
                cell.add_clock_out(clock, out, ClockEdge.RISING, ps(delay))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="FABulous project directory")
    parser.add_argument("output", help="Output .bba file path")
    args = parser.parse_args()

    grid = read_grid(args.project)
    internal, external = read_pips(args.project)
    bels = read_bels(args.project)
    settings = read_settings(args.project)
    models = tile_models(grid, internal, external, bels)
    assert grid[CLOCK_TILE[1]][CLOCK_TILE[0]] == "NULL", (
        "the global clock tile X0Y0 must be NULL"
    )

    ch = Chip("fabulous", "FABULOUS", len(grid[0]), len(grid))
    for name, model in models.items():
        create_tile_type(ch, name, model)
    create_clock_and_constant_tile(ch)
    for y, row in enumerate(grid):
        for x, tile_type in enumerate(row):
            ch.set_tile_type(x, y, "CLOCK_CONST" if (x, y) == CLOCK_TILE else tile_type)
    create_nodes(ch, grid, external, models)
    set_timings(ch, models, external, settings)
    ch.strs.known_id_count = 0
    ch.write_bba(args.output)


if __name__ == "__main__":
    main()

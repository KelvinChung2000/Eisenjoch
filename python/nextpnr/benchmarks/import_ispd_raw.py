# ruff: noqa: E402
"""Re-import ISPD'2016 raw .nodes/.nets/.pl/.lib preserving IBUF/OBUF/BUFGCE.

Unlike `run_ispd_benchmarks.convert_manifest_to_yosys_json`, which demotes
IBUF/OBUF/BUFGCE to top-level Yosys ports and collapses every LUT* to LUT6
+ FDRE to DFF, this script keeps the original cell types and ports so the
placer can use the `.pl` FIXED positions of I/O / clock cells as real
anchors. The xc_ultrascale chipdb is the routing target; pin names are
mapped to its conventions:

  LUT{N}: cell ports I0..I{N-1}, O  -> chipdb pins I[0]..I[N-1], F
  FDRE  : D, C, Q, (R, CE)          -> chipdb pins D, CLK, Q (R/CE dropped)
  IBUF  : I, O                      -> kept as-is (alias IBUF -> IOB bucket)
  OBUF  : I, O                      -> kept as-is (alias OBUF -> IOB bucket)
  BUFGCE: I, CE, O                  -> kept as-is (alias BUFGCE -> IOB; CE drops)

The output is a Yosys-JSON file plus a meta.json containing the per-cell
`.pl` FIXED placements (`fixed_placements: [{cell, x, y, z}]`) so the probe
runner can call `ctx.place_cell()` after pack.

Usage:
  python import_ispd_raw.py <design_dir> <output.json>

Example:
  python import_ispd_raw.py \
      benchmark/ispd/extracted/2016/FPGA01/FPGA01 \
      benchmark/ispd/generated/2016/FPGA01/FPGA01.raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

PinDef = namedtuple("PinDef", ["name", "direction", "is_clock", "is_ctrl"])
NodeDef = namedtuple("NodeDef", ["name", "master"])
NetDef = namedtuple("NetDef", ["name", "pins"])
PlacementDef = namedtuple("PlacementDef", ["x", "y", "z"])


def parse_lib(path: Path) -> dict[str, list[PinDef]]:
    masters: dict[str, list[PinDef]] = {}
    cur_name: str | None = None
    cur_pins: list[PinDef] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("CELL "):
            cur_name = line.split()[1]
            cur_pins = []
        elif line == "END CELL":
            if cur_name is not None:
                masters[cur_name] = cur_pins
            cur_name = None
            cur_pins = []
        elif line.startswith("PIN "):
            parts = line.split()
            pname = parts[1]
            pdir = parts[2].lower() if len(parts) > 2 else "input"
            flags = [t.upper() for t in parts[3:]]
            is_clock = "CLOCK" in flags or pname.upper() in {"CLK", "CK", "C"}
            is_ctrl = "CTRL" in flags or pname.upper() in {"CE", "R", "S", "CLR", "PRE"}
            cur_pins.append(PinDef(pname, pdir, is_clock, is_ctrl))
    return masters


def parse_nodes(path: Path) -> list[NodeDef]:
    nodes: list[NodeDef] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("UCLA", "Num", "version")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        nodes.append(NodeDef(parts[0], parts[1]))
    return nodes


def parse_nets(path: Path) -> list[NetDef]:
    nets: list[NetDef] = []
    cur: NetDef | None = None
    remaining = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("UCLA", "Num", "version")):
            continue
        parts = line.split()
        if parts[0] == "net" and len(parts) >= 3:
            if cur is not None:
                nets.append(cur)
            cur = NetDef(parts[1], [])
            remaining = int(parts[2])
            continue
        if cur is None or remaining <= 0 or len(parts) < 2:
            continue
        cur.pins.append((parts[0], parts[1]))
        remaining -= 1
    if cur is not None:
        nets.append(cur)
    return nets


def parse_pl(path: Path) -> dict[str, PlacementDef]:
    out: dict[str, PlacementDef] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        is_fixed = "FIXED" in [p.upper() for p in parts[4:]]
        if not is_fixed:
            continue
        out[parts[0]] = PlacementDef(int(parts[1]), int(parts[2]), int(parts[3]))
    return out


def map_cell_pins(master_name: str, master_pins: list[PinDef]) -> dict[str, str]:
    """master pin name -> chipdb-compatible cell port name."""
    name = master_name.upper()
    if name.startswith("LUT"):
        m: dict[str, str] = {}
        idx = 0
        for pin in master_pins:
            if pin.direction == "input":
                m[pin.name] = f"I[{idx}]"
                idx += 1
            elif pin.direction == "output":
                m[pin.name] = "F"
        return m
    if name in {"FDRE", "FDSE", "FDCE", "FDPE"}:
        m = {}
        for pin in master_pins:
            up = pin.name.upper()
            if up == "D":
                m[pin.name] = "D"
            elif up in {"C", "CK", "CLK"}:
                m[pin.name] = "CLK"
            elif up == "Q":
                m[pin.name] = "Q"
        return m
    if name in {"IBUF", "OBUF", "BUFGCE", "BUFG", "BUFGCTRL"}:
        return {pin.name: pin.name for pin in master_pins}
    return {pin.name: pin.name for pin in master_pins}


def canonical_cell_type(master_name: str) -> str:
    name = master_name.upper()
    if name.startswith("LUT"):
        return name
    if name in {"FDRE", "FDSE", "FDCE", "FDPE", "FDR", "FDS", "FDC", "FDP"}:
        return "FDRE"
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("design_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--per-port-tieoff",
        action="store_true",
        help="Split shared GND/VCC nets into per-cell-port tieoffs (TODO)",
    )
    args = parser.parse_args(argv)

    ddir: Path = args.design_dir
    nodes = parse_nodes(ddir / "design.nodes")
    nets = parse_nets(ddir / "design.nets")
    pls = parse_pl(ddir / "design.pl")
    lib = parse_lib(ddir / "design.lib")

    print(f"Parsed: {len(nodes)} nodes, {len(nets)} nets, {len(pls)} fixed, {len(lib)} masters")
    type_counts: dict[str, int] = defaultdict(int)
    for n in nodes:
        type_counts[n.master] += 1
    for ty in sorted(type_counts, key=lambda k: -type_counts[k]):
        print(f"  {ty:>10}: {type_counts[ty]}")

    masters_by_name = {n.upper(): pins for n, pins in lib.items()}
    nodes_by_name = {n.name: n for n in nodes}

    node_to_net_bits: dict[str, dict[str, int]] = defaultdict(dict)
    bit_for_net: dict[str, int] = {}
    next_bit = 2
    netnames: dict[str, dict] = {}

    for net in nets:
        bit = bit_for_net.setdefault(net.name, next_bit)
        if bit == next_bit:
            next_bit += 1
        netnames[net.name] = {"hide_name": 0, "bits": [bit], "attributes": {}}
        for inst, pin in net.pins:
            node_to_net_bits[inst][pin] = bit

    cells_out: dict[str, dict] = {}
    fixed_placements: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)

    for node in nodes:
        master_name = node.master.upper()
        master_pins = masters_by_name.get(master_name)
        if master_pins is None:
            skipped[f"{master_name}:no-master"] += 1
            continue

        cell_type = canonical_cell_type(master_name)
        pin_map = map_cell_pins(master_name, master_pins)

        port_dirs: dict[str, str] = {}
        connections: dict[str, list] = {}

        for pin in master_pins:
            cell_port = pin_map.get(pin.name)
            if cell_port is None:
                continue
            net_bit = node_to_net_bits.get(node.name, {}).get(pin.name)
            if net_bit is None:
                continue
            port_dirs[cell_port] = pin.direction
            connections[cell_port] = [net_bit]

        # LUT cells: pad I[k] up to LUT arity so the chipdb sees a uniform port set.
        if cell_type.startswith("LUT"):
            arity = int(cell_type[3:])
            for i in range(arity):
                key = f"I[{i}]"
                if key not in connections:
                    connections[key] = ["0"]
                    port_dirs[key] = "input"
            if "F" not in connections:
                skipped[f"{master_name}:no-output"] += 1
                continue

        if not connections:
            skipped[f"{master_name}:no-connections"] += 1
            continue

        cells_out[node.name] = {
            "hide_name": 0,
            "type": cell_type,
            "parameters": {},
            "attributes": {"src_master": master_name, "src_instance": node.name},
            "port_directions": port_dirs,
            "connections": connections,
        }

        if node.name in pls:
            p = pls[node.name]
            fixed_placements.append(
                {"cell": node.name, "x": p.x, "y": p.y, "z": p.z, "cell_type": cell_type}
            )

    print(f"Emitted: {len(cells_out)} cells, {len(fixed_placements)} fixed placements")
    if skipped:
        print("Skipped:")
        for reason, n in sorted(skipped.items()):
            print(f"  {reason}: {n}")

    design = {
        "creator": "nextpnr.import_ispd_raw",
        "modules": {
            "top": {
                "attributes": {"top": "1"},
                "ports": {},
                "cells": cells_out,
                "netnames": netnames,
            }
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_json}")

    meta_path = args.output_json.with_suffix(".meta.json")
    meta = {
        "name": ddir.name,
        "cell_counts": dict(type_counts),
        "fixed_placements": fixed_placements,
        "skipped_masters": dict(skipped),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

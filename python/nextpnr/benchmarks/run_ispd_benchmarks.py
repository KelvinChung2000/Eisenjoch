"""Convert ISPD FPGA placement benchmarks into Yosys JSON and run them approximately.

This script makes the local ISPD 2016/2017 benchmark datasets usable with the
existing nextpnr benchmark flow by:

1. Loading an extracted benchmark described by the ISPD manifest.
2. Converting the enhanced-Bookshelf netlist into a minimal Yosys JSON design.
3. Running the existing synthetic-chipdb pack/place/route flow on that JSON.

Important limitation: this is an approximate compatibility path. The contest
device geometry, BEL numbering, and 2017 clock legalization rules are not
modeled directly by the current architecture. Fixed IOs are therefore preserved
as fixed terminals in intent, but not at the original contest coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from nextpnr.benchmarks.ispd_benchmarks import get_roots, inspect_benchmark
from nextpnr.benchmarks.run_benchmarks import compute_grid_size, get_or_create_chipdb


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ISPD_ROOT = REPO_ROOT / "benchmark" / "ispd"
DEFAULT_OUTPUT_ROOT = DEFAULT_ISPD_ROOT / "generated"
REQUIRED_CONVERSION_FILES = {"lib", "nodes", "nets", "pl"}


@dataclass
class PinDef:
    name: str
    direction: str
    is_clock: bool = False
    is_control: bool = False


@dataclass
class MasterDef:
    name: str
    pins: list[PinDef]


@dataclass
class NodeDef:
    name: str
    master: str
    is_terminal: bool = False


@dataclass
class NetDef:
    name: str
    pins: list[tuple[str, str]]


@dataclass
class PortSpec:
    name: str
    direction: str
    bit: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run approximate ISPD benchmarks")
    parser.add_argument("--year", required=True, choices=["2016", "2017"])
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument(
        "--ispd-root",
        default=str(DEFAULT_ISPD_ROOT),
        help=f"ISPD dataset root (default: {DEFAULT_ISPD_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Generated output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--placer", default="heap", choices=["heap", "sa", "hydraulic", "electro"])
    parser.add_argument("--router", default="router1", choices=["router1", "router2", "raster"])
    parser.add_argument("--size", default=None, help="Override synthetic chipdb size NxN")
    parser.add_argument("--convert-only", action="store_true", help="Only emit JSON conversion output")
    args = parser.parse_args(argv)

    roots = get_roots(Path(args.ispd_root).resolve())
    manifest = inspect_benchmark(roots, args.year, args.benchmark)
    missing_for_conversion = [
        name for name in REQUIRED_CONVERSION_FILES if not manifest["files"].get(name)
    ]
    if missing_for_conversion:
        print(
            f"benchmark is missing conversion-critical files: {', '.join(sorted(missing_for_conversion))}",
            file=sys.stderr,
        )
        return 1

    output_root = Path(args.output_root).resolve() / args.year / args.benchmark
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{args.benchmark}.json"
    meta_path = output_root / f"{args.benchmark}.meta.json"

    design, meta = convert_manifest_to_yosys_json(manifest, roots.base)
    json_path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {json_path}")

    if args.convert_only:
        return 0

    unsupported = meta["unsupported_masters"]
    if unsupported:
        print(
            f"unsupported masters prevent run: {', '.join(sorted(unsupported))}",
            file=sys.stderr,
        )
        return 1

    grid_size = args.size or pick_grid_size(meta)
    chipdb_path = get_or_create_chipdb(parse_grid_side(grid_size), str(output_root))
    if not chipdb_path:
        print("failed to create synthetic chipdb", file=sys.stderr)
        return 1

    return run_flow(
        json_path=json_path,
        chipdb_path=Path(chipdb_path),
        fixed_port_names=meta["fixed_port_names"],
        placer=args.placer,
        router=args.router,
    )


def convert_manifest_to_yosys_json(manifest: dict, ispd_root: Path) -> tuple[dict, dict]:
    file_paths = {
        key: ispd_root / value
        for key, value in manifest["files"].items()
        if value and key in {"lib", "nodes", "nets", "pl"}
    }

    masters = parse_design_lib(file_paths["lib"])
    nodes = parse_nodes_file(file_paths["nodes"])
    nets = parse_nets_file(file_paths["nets"])
    fixed_nodes = parse_fixed_nodes(file_paths["pl"])
    fixed_set = set(fixed_nodes)

    node_map = {node.name: node for node in nodes}
    masters_by_name = {master.name: master for master in masters}

    bit_for_net: dict[str, int] = {}
    next_bit = 2
    cells: dict[str, dict] = {}
    ports: dict[str, dict] = {}
    netnames: dict[str, dict] = {}
    unsupported_masters: set[str] = set()
    fixed_port_names: list[str] = []
    node_to_nets: dict[str, dict[str, int]] = {}

    for net in nets:
        bit = bit_for_net.setdefault(net.name, next_bit)
        if bit == next_bit:
            next_bit += 1
        netnames[net.name] = {"hide_name": 0, "bits": [bit], "attributes": {}}
        for inst_name, pin_name in net.pins:
            node_to_nets.setdefault(inst_name, {})[pin_name] = bit

    for node in nodes:
        master = masters_by_name.get(node.master)
        if master is None:
            unsupported_masters.add(node.master)
            continue

        kind = classify_master(master, node.is_terminal)
        if kind is None:
            unsupported_masters.add(node.master)
            continue

        node_nets = node_to_nets.get(node.name, {})
        if kind == "terminal":
            port_spec = make_terminal_port(node.name, master, kind, node_nets)
            if port_spec is None:
                unsupported_masters.add(node.master)
                continue
            ports[port_spec.name] = {
                "direction": port_spec.direction,
                "bits": [port_spec.bit],
            }
            if node.name in fixed_set:
                fixed_port_names.append(node.name)
            continue

        cell = make_cell(node.name, master, kind, node_nets)
        if cell is None:
            unsupported_masters.add(node.master)
            continue
        cells[node.name] = cell

    design = {
        "creator": "nextpnr.ispd_converter",
        "modules": {
            "top": {
                "attributes": {"top": "1"},
                "ports": ports,
                "cells": cells,
                "netnames": netnames,
            }
        },
    }
    meta = {
        "year": manifest["year"],
        "name": manifest["name"],
        "generated_json": f"{manifest['name']}.json",
        "generated_at": int(time.time()),
        "fixed_port_names": sorted(fixed_port_names),
        "unsupported_masters": sorted(unsupported_masters),
        "cell_counts": count_emitted_cells(cells),
        "port_count": len(ports),
    }
    return design, meta


def parse_design_lib(path: Path) -> list[MasterDef]:
    masters: list[MasterDef] = []
    current: MasterDef | None = None
    current_pin: PinDef | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        tokens = line.replace(":", " ").split()
        if not tokens:
            continue

        head = tokens[0].upper()
        if head in {"CELL", "MACRO"} and len(tokens) >= 2:
            if current is not None:
                if current_pin is not None:
                    current.pins.append(current_pin)
                    current_pin = None
                masters.append(current)
            current = MasterDef(name=tokens[1], pins=[])
            continue
        if upper.startswith("END") and current is not None:
            if current_pin is not None:
                current.pins.append(current_pin)
                current_pin = None
            masters.append(current)
            current = None
            continue
        if current is None:
            continue

        if head == "PIN" and len(tokens) >= 2:
            if current_pin is not None:
                current.pins.append(current_pin)
            current_pin = PinDef(name=tokens[1], direction=infer_direction_from_tokens(tokens))
            update_pin_flags(current_pin, tokens)
            continue

        if current_pin is not None:
            current_pin.direction = merge_direction(
                current_pin.direction, infer_direction_from_tokens(tokens)
            )
            update_pin_flags(current_pin, tokens)

    if current is not None:
        if current_pin is not None:
            current.pins.append(current_pin)
        masters.append(current)
    return masters


def infer_direction_from_tokens(tokens: list[str]) -> str:
    upper = [token.upper() for token in tokens]
    if "INPUT" in upper or "INOUT" in upper:
        return "input" if "INPUT" in upper else "inout"
    if "OUTPUT" in upper:
        return "output"
    if "DIR" in upper or "DIRECTION" in upper:
        for token in upper:
            if token in {"INPUT", "OUTPUT", "INOUT"}:
                return token.lower()
    return "input"


def merge_direction(current: str, discovered: str) -> str:
    if current == discovered:
        return current
    if current == "input" and discovered != "input":
        return discovered
    return current


def update_pin_flags(pin: PinDef, tokens: list[str]) -> None:
    upper = [token.upper() for token in tokens]
    if "CLOCK" in upper or pin.name.upper() in {"CLK", "CK", "C", "G"}:
        pin.is_clock = True
    if "CONTROL" in upper or pin.name.upper() in {"CE", "R", "S", "CLR", "PRE"}:
        pin.is_control = True


def parse_nodes_file(path: Path) -> list[NodeDef]:
    nodes: list[NodeDef] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("UCLA") or line.startswith("Num"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        is_terminal = any(token.lower() == "terminal" for token in parts[2:])
        nodes.append(NodeDef(name=parts[0], master=parts[1], is_terminal=is_terminal))
    return nodes


def parse_nets_file(path: Path) -> list[NetDef]:
    nets: list[NetDef] = []
    current: NetDef | None = None
    remaining = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("UCLA") or line.startswith("Num"):
            continue
        parts = line.split()
        if line.startswith("NetDegree"):
            if current is not None:
                nets.append(current)
            degree = int(parts[2] if parts[1] == ":" else parts[1])
            name = parts[-1]
            current = NetDef(name=name, pins=[])
            remaining = degree
            continue
        if parts[0] == "net" and len(parts) >= 3:
            if current is not None:
                nets.append(current)
            current = NetDef(name=parts[1], pins=[])
            remaining = int(parts[2])
            continue
        if current is None or remaining <= 0 or len(parts) < 2:
            continue
        current.pins.append((parts[0], parts[1]))
        remaining -= 1
    if current is not None:
        nets.append(current)
    return nets


def parse_fixed_nodes(path: Path) -> list[str]:
    fixed: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 5 and "FIXED" in "".join(parts[4:]).upper():
            fixed.append(parts[0])
    return fixed


def classify_master(master: MasterDef, is_terminal: bool) -> str | None:
    name = master.name.upper()
    if is_terminal or name in {"IBUF", "OBUF", "BUFGCE", "BUFG", "BUFGCTRL"}:
        return "terminal"
    if name.startswith("LUT") or "LUT" in name:
        return "lut"
    if "LATCH" in name or name.startswith("LD"):
        return "latch"
    if "FF" in name or name.startswith("FD"):
        return "dff"
    outputs = [pin for pin in master.pins if pin.direction == "output"]
    inputs = [pin for pin in master.pins if pin.direction == "input"]
    clocks = [pin for pin in master.pins if pin.is_clock]
    if outputs and len(inputs) <= 6 and not clocks:
        return "lut"
    if clocks and any(pin.name.upper() == "Q" for pin in outputs):
        return "dff"
    return None


def make_terminal_port(
    node_name: str, master: MasterDef, kind: str, node_nets: dict[str, int]
) -> PortSpec | None:
    del kind
    inputs = [pin for pin in master.pins if pin.direction == "input"]
    outputs = [pin for pin in master.pins if pin.direction == "output"]
    master_name = master.name.upper()

    # IBUF/BUFG-like cells drive the fabric, so expose them as top-level inputs.
    if master_name in {"IBUF", "BUFGCE", "BUFG", "BUFGCTRL"}:
        for pin in outputs:
            if pin.name in node_nets:
                return PortSpec(name=node_name, direction="input", bit=node_nets[pin.name])
    if master_name == "OBUF":
        for pin in inputs:
            if pin.name in node_nets:
                return PortSpec(name=node_name, direction="output", bit=node_nets[pin.name])
    for pin in outputs:
        if pin.name in node_nets:
            return PortSpec(name=node_name, direction="input", bit=node_nets[pin.name])
    # OBUF-like cells sink a fabric signal, so expose them as top-level outputs.
    for pin in inputs:
        if pin.name in node_nets:
            return PortSpec(name=node_name, direction="output", bit=node_nets[pin.name])
    return None


def make_cell(
    node_name: str, master: MasterDef, kind: str, node_nets: dict[str, int]
) -> dict | None:
    if kind == "lut":
        return make_lut_cell(node_name, master, node_nets)
    if kind == "dff":
        return make_dff_like_cell(node_name, master, node_nets, cell_type="DFF", clock_port="CLK")
    if kind == "latch":
        return make_dff_like_cell(node_name, master, node_nets, cell_type="DLATCH", clock_port="G")
    return None


def make_lut_cell(node_name: str, master: MasterDef, node_nets: dict[str, int]) -> dict | None:
    output_pin = next((pin for pin in master.pins if pin.direction == "output"), None)
    input_pins = [pin for pin in master.pins if pin.direction == "input" and pin.name in node_nets]
    if output_pin is None or output_pin.name not in node_nets:
        return None

    conns = {f"I[{idx}]": [node_nets[pin.name]] for idx, pin in enumerate(input_pins[:6])}
    for idx in range(len(input_pins[:6]), 6):
        conns[f"I[{idx}]"] = ["0"]
    conns["F"] = [node_nets[output_pin.name]]
    port_dirs = {name: "input" for name in conns if name.startswith("I[")}
    port_dirs["F"] = "output"
    return {
        "hide_name": 0,
        "type": "LUT6",
        "parameters": {"INIT": "0" * 64},
        "attributes": {"src_master": master.name, "src_instance": node_name},
        "port_directions": port_dirs,
        "connections": conns,
    }


def make_dff_like_cell(
    node_name: str,
    master: MasterDef,
    node_nets: dict[str, int],
    *,
    cell_type: str,
    clock_port: str,
) -> dict | None:
    q_pin = next((pin for pin in master.pins if pin.direction == "output" and pin.name.upper() == "Q"), None)
    d_pin = next((pin for pin in master.pins if pin.direction == "input" and pin.name.upper() in {"D", "I"}), None)
    clk_pin = next((pin for pin in master.pins if pin.is_clock or pin.name.upper() in {"CLK", "CK", "C", "G"}), None)
    if q_pin is None or d_pin is None or clk_pin is None:
        return None
    if q_pin.name not in node_nets or d_pin.name not in node_nets or clk_pin.name not in node_nets:
        return None
    return {
        "hide_name": 0,
        "type": cell_type,
        "parameters": {},
        "attributes": {"src_master": master.name, "src_instance": node_name},
        "port_directions": {"D": "input", clock_port: "input", "Q": "output"},
        "connections": {
            "D": [node_nets[d_pin.name]],
            clock_port: [node_nets[clk_pin.name]],
            "Q": [node_nets[q_pin.name]],
        },
    }


def count_emitted_cells(cells: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells.values():
        counts[cell["type"]] = counts.get(cell["type"], 0) + 1
    return counts


def pick_grid_size(meta: dict) -> str:
    counts = meta["cell_counts"]
    luts = counts.get("LUT6", 0)
    dffs = counts.get("DFF", 0) + counts.get("DLATCH", 0)
    ios = meta["port_count"]
    total_logic = max(luts, dffs)
    grid_n = compute_grid_size(total_logic, io_count=ios)
    return f"{grid_n}x{grid_n}"


def parse_grid_side(size: str) -> int:
    x_part = size.lower().split("x")[0]
    return int(x_part)


def run_flow(
    *,
    json_path: Path,
    chipdb_path: Path,
    fixed_port_names: list[str],
    placer: str,
    router: str,
) -> int:
    try:
        import nextpnr
    except ImportError:
        print("nextpnr Python module not available. Run: maturin develop", file=sys.stderr)
        return 1

    ctx = nextpnr.Context(chipdb=str(chipdb_path))
    ctx.load_design(str(json_path))
    ctx.pack()
    preplace_fixed_ios(ctx, fixed_port_names)
    ctx.place(placer=placer, seed=42)
    ctx.route(router=router, bb_margin=max(3, ctx.width // 10))
    timing = ctx.timing_report()
    print(
        json.dumps(
            {
                "cells": len(ctx.cells),
                "nets": len(ctx.nets),
                "fmax_mhz": timing.fmax,
                "worst_slack_ps": timing.worst_slack,
                "num_failing": timing.num_failing,
                "num_endpoints": timing.num_endpoints,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def preplace_fixed_ios(ctx, fixed_port_names: list[str]) -> None:
    # Map the fixed top-level port names to the synthetic IO pseudo-cell names
    # emitted by the Yosys JSON frontend.
    fixed_cells = [f"$io${name}" for name in fixed_port_names if f"$io${name}" in ctx.cells]
    if not fixed_cells:
        return

    clock_cells = [cell for cell in fixed_cells if "clk" in cell.lower() or "clock" in cell.lower()]
    non_clock = [cell for cell in fixed_cells if cell not in clock_cells]

    reserved = []
    for idx, cell_name in enumerate(clock_cells[:2]):
        x = idx + 1
        bel_name = f"IO{idx}"
        try:
            ctx.place_cell(cell_name, x=x, y=0, bel_name=bel_name)
            reserved.append((x, 0, bel_name))
        except Exception:
            pass

    candidates = []
    for x in range(1, ctx.width - 1):
        for bel_name in ("IO0", "IO1"):
            if (x, 0, bel_name) not in reserved:
                candidates.append((x, 0, bel_name))
    for y in range(1, ctx.height - 1):
        for bel_name in ("IO0", "IO1"):
            candidates.append((0, y, bel_name))
            candidates.append((ctx.width - 1, y, bel_name))
    for x in range(1, ctx.width - 1):
        for bel_name in ("IO0", "IO1"):
            candidates.append((x, ctx.height - 1, bel_name))

    it = iter(candidates)
    for cell_name in non_clock:
        for x, y, bel_name in it:
            try:
                ctx.place_cell(cell_name, x=x, y=y, bel_name=bel_name)
                break
            except Exception:
                continue


if __name__ == "__main__":
    raise SystemExit(main())

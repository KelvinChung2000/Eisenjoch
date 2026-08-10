# ruff: noqa: E402, F403, F405
"""Column-composite helpers for paper-scale XC7 benchmark fabrics."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass

from chip_database.gen_xc7_hybrid import (
    BRAM_SITE_TYPES,
    BUFG_SITE_TYPES,
    DSP_SITE_TYPES,
    IMPORTANT_SITE_TYPES,
    PinType,
    _pip_timing_class,
    _site_bel_types,
    _wire_type_from_xray_name,
    _xray_bool,
    add_bufg_bel,
    add_slice_bels,
    add_site_bel,
    load_tile_type_json,
)


CLB_TYPES = {"CLBLL_L", "CLBLL_R", "CLBLM_L", "CLBLM_R"}
INT_TYPES = {"INT_L", "INT_R"}
RESOURCE_TYPES = {
    "CLB": CLB_TYPES,
    "BRAM": {"BRAM_L", "BRAM_R"},
    "DSP": {"DSP_L", "DSP_R"},
}
PAPER_TARGETS = {"lut": 538000, "ff": 1075000, "bram18": 1728, "dsp": 768}
N_BUFG = 32


@dataclass(frozen=True)
class Member:
    tile_type: str
    dx: int
    dy: int = 0


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    members: tuple[Member, ...]


@dataclass(frozen=True)
class SolverResult:
    h_clb: int
    n_clb: int
    h_bram: int
    n_bram: int
    h_dsp: int
    n_dsp: int

    @property
    def h_device(self):
        return max(self.h_clb, self.h_bram, self.h_dsp)

    @property
    def capacity(self):
        return {
            "lut": 8 * self.h_clb * self.n_clb,
            "ff_like": 16 * self.h_clb * self.n_clb,
            "bram18": 2 * self.h_bram * self.n_bram,
            "dsp": 2 * self.h_dsp * self.n_dsp,
        }


def load_tilegrid(tilegrid_path):
    with open(tilegrid_path) as f:
        return json.load(f)


def load_tileconn(tileconn_path):
    with open(tileconn_path) as f:
        return json.load(f)


def classify_columns(tilegrid):
    by_x = defaultdict(Counter)
    for tile in tilegrid.values():
        by_x[tile["grid_x"]][tile["type"]] += 1
    out = {}
    for x, counts in by_x.items():
        tile_type, _ = counts.most_common(1)[0]
        if tile_type in CLB_TYPES:
            kind = tile_type
        elif tile_type in INT_TYPES:
            kind = tile_type
        elif tile_type.startswith("BRAM_INT_INTERFACE"):
            kind = tile_type
        elif tile_type.startswith("INT_INTERFACE"):
            kind = tile_type
        elif tile_type in {"BRAM_L", "BRAM_R", "DSP_L", "DSP_R"}:
            kind = tile_type
        elif tile_type.startswith(("LIOB", "RIOB")):
            kind = tile_type
        elif "CLK" in tile_type or "HCLK" in tile_type or "BUFG" in tile_type:
            kind = "CLK_SPINE"
        else:
            kind = "OTHER"
        out[x] = kind
    return out


def _grid_by_xy(tilegrid):
    return {(t["grid_x"], t["grid_y"]): t["type"] for t in tilegrid.values()}


def _has_site_type(xray, tile_type, site_types):
    data = load_tile_type_json(xray, tile_type)
    return bool(data and any(s.get("type") in site_types for s in data.get("sites", [])))


def _discover_clb_members(tilegrid, xray):
    grid = _grid_by_xy(tilegrid)
    for y in sorted({yy for _, yy in grid}):
        for x in sorted({xx for xx, _ in grid}):
            types = [grid.get((x + i, y)) for i in range(4)]
            if types[1:3] != ["INT_L", "INT_R"]:
                continue
            if types[0] not in CLB_TYPES or types[3] not in CLB_TYPES:
                continue
            if not (_has_site_type(xray, types[0], {"SLICEM"}) or _has_site_type(xray, types[3], {"SLICEM"})):
                continue
            return CompositeSpec(
                "CLB",
                tuple(Member(t, i) for i, t in enumerate(types)),
            )
    raise RuntimeError("could not discover a SLICEL/SLICEM CLB + INT_L/INT_R group")


_LH_FILLER_TYPES = {"VBRK", "VFRAME", "CLK_FEED"}


def _discover_resource_members(tilegrid, resource_types, site_types, name):
    """Find the horizontal tile span that makes up one resource composite.

    The span must include the resource tile itself and its flanking
    INT_L/INT_R tiles (so the router-side has a switch matrix to reach the
    site BEL). We also include any INT_INTERFACE and "filler" tiles (VBRK,
    VFRAME, CLK_FEED) that physically sit inside the span: these carry LH
    long-wire taps (`VBRK_LH*` etc.) that our intra-composite unification
    step needs in order to chain the 12-tile LH wire across the composite.
    Without them LH dies inside every DSP/BRAM column and left->right row
    traversal is impossible.
    """
    grid = _grid_by_xy(tilegrid)
    for (x, y), tile_type in sorted(grid.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if tile_type not in resource_types:
            continue
        best = None
        for radius in range(1, 5):
            xs = range(x - radius, x + radius + 1)
            if any(grid.get((xx, y)) == "INT_L" for xx in xs) and any(
                grid.get((xx, y)) == "INT_R" for xx in xs
            ):
                members = []
                for xx in xs:
                    tt = grid.get((xx, y))
                    if (
                        tt == tile_type
                        or tt in {"INT_L", "INT_R"}
                        or tt in _LH_FILLER_TYPES
                        or (tt and ("INTERFACE" in tt) and not tt.startswith("IO_"))
                    ):
                        members.append(Member(tt, xx - x))
                if any(m.tile_type == tile_type for m in members):
                    best = members
                    break
        if best:
            return CompositeSpec(name, tuple(best))
    raise RuntimeError(f"could not discover {name} composite members with {site_types}")


def discover_composite_specs(tilegrid, xray):
    return {
        "CLB": _discover_clb_members(tilegrid, xray),
        "BRAM": _discover_resource_members(tilegrid, {"BRAM_L", "BRAM_R"}, BRAM_SITE_TYPES, "BRAM"),
        "DSP": _discover_resource_members(tilegrid, {"DSP_L", "DSP_R"}, DSP_SITE_TYPES, "DSP"),
    }


def solve_paper_scale(targets=None):
    targets = targets or PAPER_TARGETS
    return solve_padded_scale(targets, min_height=64, max_height=512, max_over_score=0.20)


def derive_standard_targets(tilegrid):
    counts = Counter(tile["type"] for tile in tilegrid.values())
    clb_tiles = sum(counts[t] for t in CLB_TYPES)
    bram_tiles = counts["BRAM_L"] + counts["BRAM_R"]
    dsp_tiles = counts["DSP_L"] + counts["DSP_R"]
    return {
        "lut": clb_tiles * 8,
        "ff": clb_tiles * 16,
        "bram18": bram_tiles * 2,
        "dsp": dsp_tiles * 2,
    }


def solve_standard_scale(tilegrid, targets=None):
    targets = targets or derive_standard_targets(tilegrid)
    return solve_padded_scale(
        targets,
        min_height=16,
        max_height=max(tile["grid_y"] for tile in tilegrid.values()) + 1,
        max_over_score=0.30,
    )


def solve_padded_scale(targets, min_height, max_height, max_over_score):
    lut_target = int(targets["lut"])
    bram_target = int(targets["bram18"])
    dsp_target = int(targets["dsp"])

    best = None
    for height in range(min_height, max_height + 1):
        n_clb = _ceil_div(lut_target, 8 * height)
        n_bram = _ceil_div(bram_target, 2 * height)
        n_dsp = _ceil_div(dsp_target, 2 * height)
        sol = SolverResult(height, n_clb, height, n_bram, height, n_dsp)
        cap = sol.capacity
        lut_over = cap["lut"] - lut_target
        bram_over = cap["bram18"] - bram_target
        dsp_over = cap["dsp"] - dsp_target
        columns = n_clb + n_bram + n_dsp + 3
        over_score = lut_over / lut_target + bram_over / bram_target + dsp_over / dsp_target
        if over_score > max_over_score:
            continue
        aspect = columns / height
        score = (
            abs(aspect - 1.0),
            over_score,
            height * columns,
            lut_over,
            bram_over,
            dsp_over,
            columns,
        )
        if best is None or score < best[0]:
            best = (score, sol)
    if best is None:
        raise RuntimeError(f"no padded solution for targets {targets}")
    return best[1]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _divisors(value):
    out = []
    for d in range(1, int(value**0.5) + 1):
        if value % d == 0:
            out.append(d)
            if d != value // d:
                out.append(value // d)
    return sorted(out)


def _next_divisor_height(column_site_rows, minimum):
    for h in range(minimum, max(minimum + 1, column_site_rows + minimum + 1)):
        if column_site_rows % h == 0:
            return h
    return None


def _site_with_renamed_wires(site, rename_for_member):
    site = deepcopy(site)
    pins = site.get("site_pins", {})
    for pin_data in pins.values():
        if pin_data is not None:
            pin_data["wire"] = rename_for_member[pin_data["wire"]]
    return site


def _create_remapped_site_bel(tt, site_data, z, rename_for_member, bel_type, name_suffix):
    site = _site_with_renamed_wires(site_data, rename_for_member)
    return add_site_bel(tt, site, z, bel_type=bel_type, name_suffix=name_suffix)


def build_composite_tile_type(chip, xray, spec, tileconn):
    tt = chip.create_tile_type(spec.name)
    wire_rename = {}
    created = set()
    member_by_type = defaultdict(list)

    for member_idx, member in enumerate(spec.members):
        data = load_tile_type_json(xray, member.tile_type)
        if not data:
            raise RuntimeError(f"missing prjxray tile type JSON for {member.tile_type}")
        member_by_type[member.tile_type].append(member)
        for wname in data.get("wires", {}):
            new_name = f"M{member_idx}_{wname}"
            tt.create_wire(new_name, _wire_type_from_xray_name(wname))
            wire_rename[(member_idx, wname)] = new_name
            created.add(new_name)

    for const_name in ("GND", "VCC"):
        if const_name not in created:
            const_value = const_name
            tt.create_wire(const_name, const_name, const_value=const_value)
            created.add(const_name)

    pip_stats = {"total": 0, "bidi": 0, "pseudo": 0, "pass_tx": 0}
    for member_idx, member in enumerate(spec.members):
        data = load_tile_type_json(xray, member.tile_type)
        for pdata in data.get("pips", {}).values():
            src = pdata["src_wire"]
            dst = pdata["dst_wire"]
            if (member_idx, src) not in wire_rename or (member_idx, dst) not in wire_rename:
                continue
            rsrc = wire_rename[(member_idx, src)]
            rdst = wire_rename[(member_idx, dst)]
            is_pseudo = _xray_bool(pdata.get("is_pseudo", 0))
            is_pass = _xray_bool(pdata.get("is_pass_transistor", 0))
            is_directional = _xray_bool(pdata.get("is_directional", 1))
            tc = _pip_timing_class(src, dst, is_pseudo=is_pseudo,
                                   is_pass_transistor=is_pass)
            tt.create_pip(rsrc, rdst, timing_class=tc)
            if not is_directional:
                tt.create_pip(rdst, rsrc, timing_class=tc)
            pip_stats["total"] += 1
            if not is_directional:
                pip_stats["bidi"] += 1
            if is_pseudo:
                pip_stats["pseudo"] += 1
            if is_pass:
                pip_stats["pass_tx"] += 1
            pip_stats.setdefault("by_class", Counter())[tc] += 1

    _absorb_internal_tileconn(tt, spec, tileconn, wire_rename)
    bel_counts, bel_types = _add_composite_bels(tt, xray, spec, wire_rename)
    tt.create_group("SWITCHBOX", "SWITCHBOX")

    wire_set = set(created) | set(wire_rename.values())
    # Every composite tile type participates in the CLK_RELAY mesh (CLB,
    # BRAM, DSP). CLB also taps the OUT mesh into its local M1/M2_GCLK
    # wires for switch-matrix delivery; BRAM/DSP only pass-through. The
    # pass-through is non-optional: without it, every BRAM and DSP column
    # is a chain-break on y-row clock distribution, stranding BUFG_I.
    wire_set |= add_clock_relay_to_composite(tt, wire_set)

    assert "CLB" != spec.name or bel_counts["LUT6"] == 8, bel_counts
    return {
        "wire_set": wire_set,
        "bel_counts": bel_counts,
        "bel_types": bel_types,
        "wire_rename": wire_rename,
        "pip_stats": pip_stats,
        "spec": spec,
    }


CLOCK_RELAY_LANES = 12
CLOCK_RELAY_DIRS = ("E", "W", "N", "S")


def clock_relay_wire(c, direction, side="OUT"):
    """Directional boundary wire for one of two clock PIP relay meshes.

    ``side="OUT"`` is the BUFG → GCLK-track distribution mesh, carrying the
    signal from ``BUFG{c}_O`` out to every CLB. Each CLB taps the OUT
    mesh into the local ``M1_GCLK_L_B{c}`` / ``M2_GCLK_B{c}`` wires so
    the intra-INT switch-matrix can drive the slice CLK BEL pin.

    ``side="IN"`` is the IOB → BUFG_I input mesh: IO pads tap into IN
    at the IOB tile, the IN mesh carries the signal across the fabric
    to the BUFG tile, and the BUFG tile taps IN into ``BUFG{c}_I``.
    IN has NO taps into the switch matrix; that's what keeps input-net
    Dijkstra from leaking into the fabric and losing the sink.

    Having two disjoint meshes also means an OUTPUT net driving lane-c
    can't collide on wire capacity with an INPUT net using lane-c."""
    return f"CLK_RELAY_{side}_{c}_{direction}"


def build_clk_bufg_tile_type(chip):
    tt = chip.create_tile_type("CLK_BUFG")
    wires = set()
    for i in range(N_BUFG):
        i_wire = f"BUFG{i}_I"
        o_wire = f"BUFG{i}_O"
        tt.create_wire(i_wire, "BUFG")
        tt.create_wire(o_wire, "BUFG")
        bel = tt.create_bel(f"BUFG{i}", "BUFG", z=i)
        tt.add_bel_pin(bel, "I", i_wire, PinType.INPUT)
        tt.add_bel_pin(bel, "O", o_wire, PinType.OUTPUT)
        wires.update((i_wire, o_wire))
    # Clock PIP relay mesh terminals:
    #   - OUT mesh: BUFG{c}_O → CLK_RELAY_OUT_{c}_{E,W,N,S}
    #   - IN  mesh: CLK_RELAY_IN_{c}_{E,W,N,S} → BUFG{c}_I
    # Each mesh is tileconn-chained through CLB and IOB pass-through PIPs.
    for c in range(CLOCK_RELAY_LANES):
        for side in ("OUT", "IN"):
            for d in CLOCK_RELAY_DIRS:
                wname = clock_relay_wire(c, d, side=side)
                tt.create_wire(wname, "GCLK")
                wires.add(wname)
        if c < N_BUFG:
            for d in CLOCK_RELAY_DIRS:
                tt.create_pip(f"BUFG{c}_O", clock_relay_wire(c, d, side="OUT"))
                tt.create_pip(clock_relay_wire(c, d, side="IN"), f"BUFG{c}_I")
    tt.create_wire("GND", "GND", const_value="GND")
    tt.create_wire("VCC", "VCC", const_value="VCC")
    wires.update(("GND", "VCC"))
    return {
        "wire_set": wires,
        "bel_counts": Counter({"BUFG": N_BUFG}),
        "bel_types": {"BUFG"},
    }


def add_clock_relay_to_composite(tt, composite_wire_set):
    """Add the OUT + IN clock PIP relay meshes to a CLB composite tile.

    Both meshes have 4 directional boundary wires per lane with all-to-all
    internal pass/turn PIPs. The OUT mesh additionally taps into the local
    M1_GCLK_L_B{c} / M2_GCLK_B{c} wires so the switch-matrix can deliver
    to the slice CLK BEL pin. The IN mesh has no taps — it's a clean
    conduit for IOB → BUFG_I signals that never leaks into fabric."""
    added = set()
    for c in range(CLOCK_RELAY_LANES):
        for side in ("OUT", "IN"):
            dirs = []
            for d in CLOCK_RELAY_DIRS:
                wname = clock_relay_wire(c, d, side=side)
                tt.create_wire(wname, "GCLK")
                dirs.append(wname)
                added.add(wname)
            for i, src in enumerate(dirs):
                for j, dst in enumerate(dirs):
                    if i != j:
                        tt.create_pip(src, dst)
            if side == "OUT":
                for tap in (f"M1_GCLK_L_B{c}", f"M2_GCLK_B{c}"):
                    if tap not in composite_wire_set:
                        continue
                    for src in dirs:
                        tt.create_pip(src, tap)
    return added


def _absorb_internal_tileconn(tt, spec, tileconn, wire_rename):
    members = list(spec.members)
    by_type = defaultdict(list)
    for idx, member in enumerate(members):
        by_type[member.tile_type].append((idx, member))

    for conn in tileconn:
        a_type, b_type = conn["tile_types"]
        dx, dy = conn.get("grid_deltas", [0, 0])
        for a_idx, a_member in by_type.get(a_type, []):
            for b_idx, b_member in by_type.get(b_type, []):
                if (a_member.dx + dx, a_member.dy + dy) != (b_member.dx, b_member.dy):
                    continue
                for wa, wb in conn["wire_pairs"]:
                    if (a_idx, wa) not in wire_rename or (b_idx, wb) not in wire_rename:
                        continue
                    a_wire = wire_rename[(a_idx, wa)]
                    b_wire = wire_rename[(b_idx, wb)]
                    tt.create_pip(a_wire, b_wire)
                    tt.create_pip(b_wire, a_wire)


def _add_composite_bels(tt, xray, spec, wire_rename):
    bel_counts = Counter()
    bel_types = set()
    z = 0
    clb_slice_seen = set()
    for member_idx, member in enumerate(spec.members):
        data = load_tile_type_json(xray, member.tile_type)
        rename_for_member = {
            w: renamed for (idx, w), renamed in wire_rename.items() if idx == member_idx
        }
        for site_idx, site in enumerate(data.get("sites", [])):
            site_type = site.get("type")
            if spec.name == "CLB" and site_type in {"SLICEL", "SLICEM"}:
                # Collapse two source CLB tiles into one logical CLB with one
                # SLICEL-like and one SLICEM-like slice, not all four physical
                # slice sites present in the two prjxray tile types.
                if site_type in clb_slice_seen:
                    continue
                add_slice_bels(
                    tt,
                    _site_with_renamed_wires(site, rename_for_member),
                    z,
                    dff_copies=2,
                    include_latches=False,
                )
                clb_slice_seen.add(site_type)
                bel_counts["LUT6"] += 4
                bel_counts["DFF"] += 8
                bel_types.update(("LUT6", "DFF"))
                z += 12
                continue
            if site_type in IMPORTANT_SITE_TYPES:
                for bel_type in _site_bel_types(site_type):
                    suffix = f"M{member_idx}_{site_idx}_{bel_type}"
                    _create_remapped_site_bel(tt, site, z, rename_for_member, bel_type, suffix)
                    bel_counts[bel_type] += 1
                    bel_types.add(bel_type)
                    z += 1
            if site_type in BUFG_SITE_TYPES:
                add_bufg_bel(tt, _site_with_renamed_wires(site, rename_for_member), z)
                bel_counts["BUFG"] += 1
                bel_types.add("BUFG")
                z += 1
    return bel_counts, bel_types


def build_synthetic_tileconn(specs, xray, tileconn):
    by_type = defaultdict(list)
    for spec in specs.values():
        for member_idx, member in enumerate(spec.members):
            by_type[member.tile_type].append((spec, member_idx, member))

    conns = defaultdict(list)
    seen_pairs = defaultdict(set)
    for conn in tileconn:
        a_type, b_type = conn["tile_types"]
        dx, dy = conn.get("grid_deltas", [0, 0])
        for a_spec, a_idx, a_member in by_type.get(a_type, []):
            for b_spec, b_idx, b_member in by_type.get(b_type, []):
                out_dx = a_member.dx + dx - b_member.dx
                out_dy = a_member.dy + dy - b_member.dy
                # Internal edges are copied into the composite tile type.
                if a_spec.name == b_spec.name and out_dx == 0 and out_dy == 0:
                    continue
                # Normalize the delta from original-grid units into
                # composite-grid units. In the composite grid each composite
                # occupies one column; a wire pair can physically connect
                # only to the immediately adjacent composite (or the same
                # column for purely vertical deltas). Composite rows map 1:1
                # to original rows, so dy passes through.
                comp_dx = 0 if out_dx == 0 else (1 if out_dx > 0 else -1)
                comp_dy = out_dy
                if a_spec.name == b_spec.name and comp_dx == 0 and comp_dy == 0:
                    continue
                key = (a_spec.name, b_spec.name, comp_dx, comp_dy)
                for wa, wb in conn["wire_pairs"]:
                    pair = (f"M{a_idx}_{wa}", f"M{b_idx}_{wb}")
                    if pair in seen_pairs[key]:
                        continue
                    seen_pairs[key].add(pair)
                    conns[key].append(list(pair))

    return [
        {"tile_types": [a, b], "grid_deltas": [dx, dy], "wire_pairs": pairs}
        for (a, b, dx, dy), pairs in sorted(conns.items())
        if pairs
    ]


def compose_paper_scale_tilegrid(solution):
    layout = _paper_layout(solution.n_clb, solution.n_bram, solution.n_dsp)
    tilegrid = {}
    x = 0
    for y in range(solution.h_device):
        tilegrid[f"X{x:03d}Y{y:03d}__IOB_L"] = {"grid_x": x, "grid_y": y, "type": "IOB_W"}
    x += 1
    for kind in layout:
        for y in range(solution.h_device):
            tilegrid[f"X{x:03d}Y{y:03d}__{kind}"] = {
                "grid_x": x,
                "grid_y": y,
                "type": kind,
            }
        x += 1
    for y in range(solution.h_device):
        tilegrid[f"X{x:03d}Y{y:03d}__IOB_R"] = {"grid_x": x, "grid_y": y, "type": "IOB_E"}
    x += 1
    # Keep the BUFG copy as a sidecar column outside the routed fabric. A mostly
    # NULL column must not be inserted between composite routing columns.
    bufg_y = solution.h_device // 2
    for y in range(solution.h_device):
        tile_type = "CLK_BUFG" if y == bufg_y else "NULL"
        tilegrid[f"X{x:03d}Y{y:03d}__{tile_type}"] = {
            "grid_x": x,
            "grid_y": y,
            "type": tile_type,
        }
    return tilegrid, layout


def _paper_layout(n_clb, n_bram, n_dsp):
    non_clb = ["DSP"] * n_dsp + ["BRAM"] * n_bram
    layout = ["CLB"] * n_clb
    if not non_clb:
        return layout
    spacing = max(1, math.floor(n_clb / (len(non_clb) + 1)))
    inserted = 0
    for idx, kind in enumerate(non_clb, start=1):
        pos = min(len(layout), idx * spacing + inserted)
        layout.insert(pos, kind)
        inserted += 1
    return layout


def build_paper_scale_tilegrid(tilegrid, tileconn, xray, targets=None):
    specs = discover_composite_specs(tilegrid, xray)
    solution = solve_paper_scale(targets)
    large_tilegrid, layout = compose_paper_scale_tilegrid(solution)
    synthetic_tileconn = build_synthetic_tileconn(specs, xray, tileconn)
    print(
        "Paper-scale solution: "
        f"H_CLB={solution.h_clb}, n_clb={solution.n_clb}, "
        f"H_BRAM={solution.h_bram}, n_bram={solution.n_bram}, "
        f"H_DSP={solution.h_dsp}, n_dsp={solution.n_dsp}, "
        f"H_device={solution.h_device}"
    )
    print(
        "Paper-scale capacity: "
        f"{solution.capacity['lut']} LUT6, "
        f"{solution.capacity['ff_like']} DFF, "
        f"{solution.capacity['bram18']} BRAM18E1, "
        f"{solution.capacity['dsp']} DSP48E1"
    )
    print(
        "Paper-scale padded capacity: "
        f"{8 * solution.h_device * solution.n_clb} LUT6, "
        f"{16 * solution.h_device * solution.n_clb} DFF, "
        f"{2 * solution.h_device * solution.n_bram} BRAM18E1, "
        f"{2 * solution.h_device * solution.n_dsp} DSP48E1"
    )
    return large_tilegrid, synthetic_tileconn, specs, solution, layout


def build_standard_scale_tilegrid(tilegrid, tileconn, xray, targets=None):
    specs = discover_composite_specs(tilegrid, xray)
    solution = solve_standard_scale(tilegrid, targets)
    standard_tilegrid, layout = compose_paper_scale_tilegrid(solution)
    synthetic_tileconn = build_synthetic_tileconn(specs, xray, tileconn)
    print(
        "Standard composite solution: "
        f"H_CLB={solution.h_clb}, n_clb={solution.n_clb}, "
        f"H_BRAM={solution.h_bram}, n_bram={solution.n_bram}, "
        f"H_DSP={solution.h_dsp}, n_dsp={solution.n_dsp}, "
        f"H_device={solution.h_device}"
    )
    print(
        "Standard composite capacity: "
        f"{solution.capacity['lut']} LUT6, "
        f"{solution.capacity['ff_like']} DFF, "
        f"{solution.capacity['bram18']} BRAM18E1, "
        f"{solution.capacity['dsp']} DSP48E1"
    )
    print(
        "Standard padded capacity: "
        f"{8 * solution.h_device * solution.n_clb} LUT6, "
        f"{16 * solution.h_device * solution.n_clb} DFF, "
        f"{2 * solution.h_device * solution.n_bram} BRAM18E1, "
        f"{2 * solution.h_device * solution.n_dsp} DSP48E1"
    )
    return standard_tilegrid, synthetic_tileconn, specs, solution, layout


def parse_targets(text):
    if not text:
        return PAPER_TARGETS.copy()
    targets = PAPER_TARGETS.copy()
    for part in text.split(","):
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key == "ff":
            key = "ff"
        targets[key] = int(value)
    return targets

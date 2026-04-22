"""Place + raster-route sweep on sv3.

Variants:
  - heap
  - opt_trans, NPNR_OT_GRAPH_MODEL in {flat, disp_pure, disp_sparse, corridor, corridor_warm}

Each row runs placement (centroid init default for opt_trans), then routes with
the raster router, and records wall times + quality. HPWL is measured before
routing; routed stats are collected after.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/kelvin/side-project/eisenjoch")
CHIP = REPO / "chip_database" / "xc7_large.bin"
DESIGN = REPO / "benchmark" / "output" / "stereovision3.json"
LOG_DIR = Path("/tmp/claude-1000/pnr_sweep")

OT_VARIANTS = ("flat", "disp_pure", "disp_sparse", "corridor", "corridor_warm")

SCRIPT = """
import time
import nextpnr

ctx = nextpnr.Context(chipdb=r'{chip}')
ctx.load_design(r'{design}')
try:
    ctx.pack()
except Exception as exc:
    print(f'pack: {{exc}}')

t0 = time.time()
{place_call}
t_place = time.time() - t0
print(f'=== Placement total: {{t_place:.1f}}s ===')
print(f'=== Total HPWL: {{ctx.total_hpwl():.0f}} ===')
print(f'=== Line estimate: {{ctx.total_line_estimate():.0f}} ===')
print(f'=== Congestion cost: {{ctx.total_congestion_cost():.2f}} ===')

t1 = time.time()
try:
    ctx.route(router='raster', skip_unplaced=True)
    t_route = time.time() - t1
    print(f'=== Routing total: {{t_route:.1f}}s ===')
    print(f'=== Routed wirelength: {{ctx.total_routed_wirelength()}} ===')
except Exception as exc:
    t_route = time.time() - t1
    print(f'=== Routing total: {{t_route:.1f}}s (error) ===')
    print(f'=== Routing error: {{exc}} ===')
"""

PLACE_RE = re.compile(r"Placement total: ([\d.]+)s")
HPWL_RE = re.compile(r"Total HPWL: (\d+)")
LINE_RE = re.compile(r"Line estimate: (\d+)")
CONG_RE = re.compile(r"Congestion cost: ([\d.]+)")
ROUTE_RE = re.compile(r"Routing total: ([\d.]+)s")
ROUTED_RE = re.compile(r"Routed wirelength: (\d+)")
ROUTE_ERR_RE = re.compile(r"Routing error: (.+)")


def run(label: str, env_extra: dict, place_call: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    env = os.environ.copy()
    env.update(env_extra)
    cmd = ["uv", "run", "python", "-c",
           SCRIPT.format(chip=CHIP, design=DESIGN, place_call=place_call)]
    print(f"--- {label} (env: {env_extra}) ---", flush=True)
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    text = log_path.read_text()
    def grp(r): m = r.search(text); return m.group(1) if m else ""
    row = {
        "label": label, "wall_s": f"{wall:.1f}", "rc": proc.returncode,
        "place_s": grp(PLACE_RE),
        "hpwl": grp(HPWL_RE),
        "line": grp(LINE_RE),
        "cong": grp(CONG_RE),
        "route_s": grp(ROUTE_RE),
        "routed_wl": grp(ROUTED_RE),
        "route_err": grp(ROUTE_ERR_RE),
    }
    print(f"    wall={row['wall_s']}s place={row['place_s']}s hpwl={row['hpwl']} "
          f"line={row['line']} route={row['route_s']}s routed_wl={row['routed_wl']} "
          f"{'ERR='+row['route_err'] if row['route_err'] else ''}", flush=True)
    return row


def main() -> None:
    rows = []

    heap_call = "ctx.place(placer='heap', max_iters=15)"
    rows.append(run("heap", {}, heap_call))

    ot_call = "ctx.place(placer='opt_trans', max_iters=15, num_threads=8)"
    for v in OT_VARIANTS:
        env = {"NPNR_OT_GRAPH_MODEL": v}
        if v.startswith("corridor"):
            env["NPNR_OT_CORRIDOR_TIGHT"] = "1"
        rows.append(run(f"opt_trans_{v}", env, ot_call))

    out = LOG_DIR / "pnr_sweep.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== Summary ===")
    print(f"  {'label':24s}  {'place':>6s}  {'hpwl':>6s}  {'line':>6s}  "
          f"{'route':>7s}  {'routed_wl':>10s}")
    for r in rows:
        print(f"  {r['label']:24s}  {r['place_s']:>6s}s  {r['hpwl']:>6s}  {r['line']:>6s}  "
              f"{r['route_s']:>6s}s  {r['routed_wl']:>10s}"
              + (f"  ERR" if r['route_err'] else ""))
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()

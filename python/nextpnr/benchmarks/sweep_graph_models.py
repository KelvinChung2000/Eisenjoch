"""Sweep graph-model variants on sv3 15-iter, dump CSV of wall time + quality metrics.

Usage:
    uv run python python/nextpnr/benchmarks/sweep_graph_models.py [variants...]

Each row runs the placer once with NPNR_OT_GRAPH_MODEL set; NEXTPNR_DIAG=1 is
exported so the baseline R_eff/base percentiles land in the stderr log. We
recapture the last iter's diag line from the log and pull the hot-pipe counts
and max ratio out of it.
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
DEFAULT_VARIANTS = ("flat", "disp_pure", "disp_sparse", "corridor", "corridor_warm")
LOG_DIR = Path("/tmp/claude-1000/sweep_logs")

DIAG_RE = re.compile(
    r"\[diag\] iter=(\d+) pipes=\d+ p50=([\d.]+) p90=([\d.]+) p99=([\d.]+) "
    r"max=([\d.]+) hot1\.01=(\d+) hot1\.2=(\d+) hot2=(\d+)"
)
TOTAL_RE = re.compile(r"Placement total: ([\d.]+)s \((\d+) ms/iter")
HPWL_RE = re.compile(r"Total HPWL: (\d+)")
LINE_RE = re.compile(r"Line estimate: (\d+)")
CONG_RE = re.compile(r"Congestion cost: ([\d.]+)")

RUN_SCRIPT = """
from pathlib import Path
import time
import nextpnr

ctx = nextpnr.Context(chipdb=r'{chip}')
ctx.load_design(r'{design}')
try:
    ctx.pack()
except Exception as exc:
    print(f'pack: {{exc}}')

print(f'=== Grid: {{ctx.width}} x {{ctx.height}}, cells={{len(ctx.cells)}} nets={{len(ctx.nets)}} ===')
t0 = time.time()
ctx.place(placer='opt_trans', max_iters=15, num_threads=8, io_boost=3.0)
elapsed = time.time() - t0
print(f'=== Placement total: {{elapsed:.1f}}s ({{int(elapsed * 1000 / 15)}} ms/iter avg) ===')
print(f'=== Total HPWL: {{ctx.total_hpwl():.0f}} ===')
print(f'=== Line estimate: {{ctx.total_line_estimate():.0f}} ===')
print(f'=== Congestion cost: {{ctx.total_congestion_cost():.2f}} ===')
"""


def run_variant(name: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env["NPNR_OT_GRAPH_MODEL"] = name
    env["NEXTPNR_DIAG"] = "1"
    if name in ("corridor", "corridor_prune", "CorridorPrune", "CORRIDOR"):
        env["NPNR_OT_CORRIDOR_TIGHT"] = "1"
    if name in ("corridor_warm", "CorridorWarmstart", "CORRIDOR_WARM"):
        env["NPNR_OT_CORRIDOR_TIGHT"] = "1"
    cmd = ["uv", "run", "python", "-c", RUN_SCRIPT.format(chip=CHIP, design=DESIGN)]
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    text = log_path.read_text()

    diag_rows = DIAG_RE.findall(text)
    last_diag = diag_rows[-1] if diag_rows else None
    total = TOTAL_RE.search(text)
    hpwl = HPWL_RE.search(text)
    line = LINE_RE.search(text)
    cong = CONG_RE.search(text)

    row = {
        "variant": name,
        "wall_s": f"{wall:.1f}",
        "return_code": proc.returncode,
        "placement_total_s": total.group(1) if total else "",
        "ms_per_iter": total.group(2) if total else "",
        "hpwl": hpwl.group(1) if hpwl else "",
        "line_estimate": line.group(1) if line else "",
        "congestion_cost": cong.group(1) if cong else "",
        "last_iter_p99_reff": last_diag[3] if last_diag else "",
        "last_iter_max_reff": last_diag[4] if last_diag else "",
        "last_iter_hot1_01": last_diag[5] if last_diag else "",
        "last_iter_hot1_2": last_diag[6] if last_diag else "",
        "last_iter_hot2": last_diag[7] if last_diag else "",
        "log": str(log_path),
    }
    return row


def main():
    variants = sys.argv[1:] or list(DEFAULT_VARIANTS)
    rows = []
    for name in variants:
        print(f"=== {name} ===", flush=True)
        row = run_variant(name)
        rows.append(row)
        print(
            f"  wall={row['wall_s']}s placement={row['placement_total_s']}s "
            f"hpwl={row['hpwl']} line={row['line_estimate']} "
            f"cong={row['congestion_cost']} "
            f"max_reff={row['last_iter_max_reff']} hot2={row['last_iter_hot2']}",
            flush=True,
        )

    out = LOG_DIR / "sweep_results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()

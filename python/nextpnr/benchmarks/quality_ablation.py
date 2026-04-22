"""Quality-ablation driver for sv3 15-iter opt_trans placement.

Runs two experiments back to back (each as its own subprocess so env-read
OnceLocks reset):

  1. Ideal-clustering HPWL floor: sums bbox of IO-locked cells per net *after*
     pack but *before* place. Strict lower bound on achievable HPWL regardless
     of how movable cells are placed.
  2. BPR alpha sweep: baseline (alpha=0.05) vs alpha=0 (no congestion
     pushback). Tests whether the ~47k HPWL ceiling is a search-algorithm
     limit or a cost-function limit.

Usage:
    uv run python python/nextpnr/benchmarks/quality_ablation.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/kelvin/side-project/eisenjoch")
CHIP = REPO / "chip_database" / "xc7_large.bin"
DESIGN = REPO / "benchmark" / "output" / "stereovision3.json"
LOG_DIR = Path("/tmp/claude-1000/quality_ablation")

FLOOR_SCRIPT = """
import nextpnr

ctx = nextpnr.Context(chipdb=r'{chip}')
ctx.load_design(r'{design}')
try:
    ctx.pack()
except Exception as exc:
    print(f'pack: {{exc}}')

floor = ctx.total_hpwl_locked_only()
pre_hpwl = ctx.total_hpwl()
print(f'=== Grid: {{ctx.width}} x {{ctx.height}}, cells={{len(ctx.cells)}} nets={{len(ctx.nets)}} ===')
print(f'=== HPWL floor (locked-only bbox sum): {{floor:.0f}} ===')
print(f'=== HPWL pre-place (all placed cells): {{pre_hpwl:.0f}} ===')
"""

PLACE_SCRIPT = """
import time
import nextpnr

ctx = nextpnr.Context(chipdb=r'{chip}')
ctx.load_design(r'{design}')
try:
    ctx.pack()
except Exception as exc:
    print(f'pack: {{exc}}')

t0 = time.time()
ctx.place(placer='opt_trans', max_iters=15, num_threads=8)
elapsed = time.time() - t0
print(f'=== Placement total: {{elapsed:.1f}}s ({{int(elapsed * 1000 / 15)}} ms/iter avg) ===')
print(f'=== Total HPWL: {{ctx.total_hpwl():.0f}} ===')
print(f'=== Line estimate: {{ctx.total_line_estimate():.0f}} ===')
print(f'=== Congestion cost: {{ctx.total_congestion_cost():.2f}} ===')
"""

TOTAL_RE = re.compile(r"Placement total: ([\d.]+)s")
HPWL_RE = re.compile(r"Total HPWL: (\d+)")
LINE_RE = re.compile(r"Line estimate: (\d+)")
CONG_RE = re.compile(r"Congestion cost: ([\d.]+)")
FLOOR_RE = re.compile(r"HPWL floor \(locked-only bbox sum\): (\d+)")


def run_subprocess(label: str, script: str, env_extra: dict[str, str]) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    env = os.environ.copy()
    env.update(env_extra)
    cmd = ["uv", "run", "python", "-c", script.format(chip=CHIP, design=DESIGN)]
    print(f"--- {label} (env: {env_extra}) ---", flush=True)
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    print(f"    wall={wall:.1f}s rc={proc.returncode} log={log_path}", flush=True)
    return log_path.read_text()


def main() -> None:
    # 1. Floor: post-pack, pre-place.
    floor_text = run_subprocess("floor", FLOOR_SCRIPT, {})
    floor_match = FLOOR_RE.search(floor_text)
    floor = int(floor_match.group(1)) if floor_match else None
    print(f"\n  HPWL floor (IO-locked only):  {floor}\n", flush=True)

    # 2. Baseline alpha=0.05.
    base_text = run_subprocess("bpr_0p05", PLACE_SCRIPT, {"NPNR_OT_BPR_ALPHA": "0.05"})
    # 3. alpha=0.
    zero_text = run_subprocess("bpr_0", PLACE_SCRIPT, {"NPNR_OT_BPR_ALPHA": "0"})

    def pick(text: str) -> dict:
        t = TOTAL_RE.search(text)
        h = HPWL_RE.search(text)
        l = LINE_RE.search(text)
        c = CONG_RE.search(text)
        return {
            "place_s": t.group(1) if t else "",
            "hpwl": h.group(1) if h else "",
            "line": l.group(1) if l else "",
            "cong": c.group(1) if c else "",
        }

    base = pick(base_text)
    zero = pick(zero_text)

    print("\n=== Quality ablation summary ===")
    print(f"  Floor (IO-locked bbox sum):  HPWL={floor}")
    print(f"  BPR alpha=0.05 (baseline):   "
          f"place={base['place_s']}s  hpwl={base['hpwl']}  line={base['line']}  cong={base['cong']}")
    print(f"  BPR alpha=0    (no cong):    "
          f"place={zero['place_s']}s  hpwl={zero['hpwl']}  line={zero['line']}  cong={zero['cong']}")


if __name__ == "__main__":
    main()

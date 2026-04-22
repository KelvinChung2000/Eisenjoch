"""Compare init_strategy={random_bel, centroid} on sv3 15-iter opt_trans.

If init basin traps the placer, centroid should yield lower HPWL / line estimate.
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
LOG_DIR = Path("/tmp/claude-1000/init_ablation")

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
ctx.place(placer='opt_trans', max_iters=15, num_threads=8,
          init_strategy='{init}')
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


def run(label: str, init: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    env = os.environ.copy()
    cmd = ["uv", "run", "python", "-c", SCRIPT.format(chip=CHIP, design=DESIGN, init=init)]
    print(f"--- {label} (init={init}) ---", flush=True)
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    text = log_path.read_text()
    t = TOTAL_RE.search(text); h = HPWL_RE.search(text)
    l = LINE_RE.search(text); c = CONG_RE.search(text)
    row = {
        "label": label, "init": init, "wall_s": f"{wall:.1f}", "rc": proc.returncode,
        "place_s": t.group(1) if t else "",
        "hpwl": h.group(1) if h else "",
        "line": l.group(1) if l else "",
        "cong": c.group(1) if c else "",
    }
    print(f"    wall={row['wall_s']}s  place={row['place_s']}s  hpwl={row['hpwl']}  "
          f"line={row['line']}  cong={row['cong']}  (log={log_path})", flush=True)
    return row


def main() -> None:
    rows = [
        run("random_bel", "random_bel"),
        run("centroid", "centroid"),
    ]
    print("\n=== Init strategy summary ===")
    for r in rows:
        print(f"  {r['label']:12s}  place={r['place_s']:>5}s  hpwl={r['hpwl']:>6}  "
              f"line={r['line']:>6}  cong={r['cong']}")


if __name__ == "__main__":
    main()

"""Benchmark: 1D decomposition Beckmann OT placer vs HeAP baseline."""

import nextpnr
import time

CHIPDB = '/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin'
DESIGN = '/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json'

# --- HeAP baseline ---
print("=== HeAP baseline ===")
ctx2 = nextpnr.Context(chipdb=CHIPDB)
ctx2.load_design(DESIGN)
ctx2.pack()
t0 = time.time()
ctx2.place(placer='heap', seed=42)
t_heap = time.time() - t0
hpwl_heap = ctx2.total_hpwl()
line_heap = ctx2.total_line_estimate()
print(f'HeAP: HPWL={hpwl_heap:.0f} line={line_heap:.0f} time={t_heap:.1f}s')

# --- 1D decomposition ---
print("\n=== 1D Decomposition Beckmann OT ===")
ctx = nextpnr.Context(chipdb=CHIPDB)
ctx.load_design(DESIGN)
ctx.pack()
t0 = time.time()
ctx.place(placer='hydraulic', seed=42, subtile_resolution=1, step_scale=0.5)
t_1d = time.time() - t0
hpwl_1d = ctx.total_hpwl()
line_1d = ctx.total_line_estimate()
print(f'1D: HPWL={hpwl_1d:.0f} line={line_1d:.0f} time={t_1d:.1f}s')

# --- Summary ---
print(f'\n=== Summary ===')
print(f'HeAP: HPWL={hpwl_heap:.0f} line={line_heap:.0f} time={t_heap:.1f}s')
print(f'1D:   HPWL={hpwl_1d:.0f} line={line_1d:.0f} time={t_1d:.1f}s')
ratio = hpwl_1d / hpwl_heap if hpwl_heap > 0 else float('inf')
print(f'HPWL ratio (1D/HeAP): {ratio:.3f}')

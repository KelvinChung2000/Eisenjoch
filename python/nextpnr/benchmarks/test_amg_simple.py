"""Minimal test: just one hydraulic run."""
import time
import nextpnr

chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin"
design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json"

ctx = nextpnr.Context(chipdb=chipdb)
ctx.load_design(design)
ctx.pack()
t0 = time.time()
ctx.place(placer='hydraulic', seed=42, max_iters=10, subtile_resolution=1, step_scale=0.5)
elapsed = time.time() - t0
print(f"AMG: HPWL={ctx.total_hpwl():.0f} line={ctx.total_line_estimate():.0f} {elapsed:.1f}s")

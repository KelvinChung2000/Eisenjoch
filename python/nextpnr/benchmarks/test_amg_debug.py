"""Debug: single iteration with gradient diagnostics."""
import time
import nextpnr

chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin"
design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json"

# Just 3 iterations to see the direction
ctx = nextpnr.Context(chipdb=chipdb)
ctx.load_design(design)
ctx.pack()
t0 = time.time()
ctx.place(
    placer='hydraulic', seed=42, max_iters=3,
    adam_lr_gain=0.5,
    init_strategy="centroid",
)
elapsed = time.time() - t0
print(f"HPWL={ctx.total_hpwl():.0f} line={ctx.total_line_estimate():.0f} {elapsed:.1f}s")

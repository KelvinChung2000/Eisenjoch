"""Benchmark: Beckmann OT with multi-level cell clustering.

Compares cell clustering (grid stays at full resolution, cells coarsened)
against HeAP baseline on stereovision3.

Reports: clustering stats, CHPWL at each level, final HPWL, line estimate, time.
"""
import time
import nextpnr

chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin"
design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json"

print("=== HeAP Baseline ===")
ctx_h = nextpnr.Context(chipdb=chipdb)
ctx_h.load_design(design)
ctx_h.pack()
t0 = time.time()
ctx_h.place(placer='heap', seed=42)
heap_t = time.time() - t0
heap_hpwl = ctx_h.total_hpwl()
heap_line = ctx_h.total_line_estimate()
print(f"HeAP: HPWL={heap_hpwl:.0f} line={heap_line:.0f} {heap_t:.1f}s")
del ctx_h

print("\n=== Beckmann OT with Cell Clustering (centroid init) ===")
ctx = nextpnr.Context(chipdb=chipdb)
ctx.load_design(design)
ctx.pack()
t0 = time.time()
ctx.place(
    placer='hydraulic', seed=42,
    subtile_resolution=1,
    max_iters=30,
    step_scale=0.5,
    init_strategy="centroid",
)
ot_t = time.time() - t0
ot_hpwl = ctx.total_hpwl()
ot_line = ctx.total_line_estimate()
print(f"\nOT Cell Clustering: HPWL={ot_hpwl:.0f} line={ot_line:.0f} {ot_t:.1f}s")
print(f"Ratio vs HeAP: {ot_hpwl / heap_hpwl:.2f}x")
print(f"Target: {'PASS' if ot_hpwl < 14600 else 'FAIL'} (target < 14600, got {ot_hpwl:.0f})")

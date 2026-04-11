"""Benchmark: Beckmann OT with per-net AMG solves and Adam optimizer.

Reports per-iteration time, CHPWL trajectory, final HPWL/line, total time.
Compares to HeAP baseline.
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

print("\n=== Beckmann OT with Dijkstra + Adam (centroid init, lr_gain=0.5, 5 iters) ===")
ctx = nextpnr.Context(chipdb=chipdb)
ctx.load_design(design)
ctx.pack()
t0 = time.time()
ctx.place(
    placer='hydraulic', seed=42,
    
    max_iters=5,
    adam_lr_gain=0.5,
    init_strategy="centroid",
)
ot_t = time.time() - t0
ot_hpwl = ctx.total_hpwl()
ot_line = ctx.total_line_estimate()
print(f"OT:   HPWL={ot_hpwl:.0f} line={ot_line:.0f} {ot_t:.1f}s")
print(f"Ratio: {ot_hpwl / heap_hpwl:.2f}x HeAP")
print(f"Target: {'PASS' if ot_hpwl < 14600 else 'FAIL'} (target < 14600, got {ot_hpwl:.0f})")

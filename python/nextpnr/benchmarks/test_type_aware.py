import time
import nextpnr

chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin"
design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json"

# HeAP baseline
ctx2 = nextpnr.Context(chipdb=chipdb)
ctx2.load_design(design)
ctx2.pack()
t0 = time.time()
ctx2.place(placer='heap', seed=42)
print(f"HeAP: HPWL={ctx2.total_hpwl():.0f} line={ctx2.total_line_estimate():.0f} {time.time()-t0:.1f}s")

# Hydraulic at various iterations
for iters in [1, 5, 10]:
    ctx3 = nextpnr.Context(chipdb=chipdb)
    ctx3.load_design(design)
    ctx3.pack()
    t0 = time.time()
    ctx3.place(placer='hydraulic', seed=42, max_iters=iters, subtile_resolution=1, step_scale=0.5, init_strategy='centroid')
    elapsed = time.time() - t0
    print(f"{iters:3d} iter: HPWL={ctx3.total_hpwl():.0f} line={ctx3.total_line_estimate():.0f} {elapsed:.1f}s")

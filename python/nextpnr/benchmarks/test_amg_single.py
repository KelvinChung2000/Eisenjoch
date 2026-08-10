"""Single run benchmark."""
import sys
import time
import nextpnr

chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin"
design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json"

mode = sys.argv[1] if len(sys.argv) > 1 else "heap"

ctx = nextpnr.Context(chipdb=chipdb)
ctx.load_design(design)
ctx.pack()
t0 = time.time()

if mode == "heap":
    ctx.place(placer='heap', seed=42)
    elapsed = time.time() - t0
    print(f"HeAP: HPWL={ctx.total_hpwl():.0f} line={ctx.total_line_estimate():.0f} {elapsed:.1f}s")
else:
    # Parse mode: "centroid_lr0.5_it5" or "random-bel_lr0.5_it200"
    parts = mode.split("_")
    # Last two parts are lr and it; everything before is the init strategy
    iters = int(parts[-1].replace("it", ""))
    lr = float(parts[-2].replace("lr", ""))
    init = "_".join(parts[:-2]).replace("-", "_")
    ctx.place(
        placer='hydraulic', seed=42,
        
        max_iters=iters,
        adam_lr_gain=lr,
        init_strategy=init,
    )
    elapsed = time.time() - t0
    hpwl = ctx.total_hpwl()
    line = ctx.total_line_estimate()
    print(f"{mode}: HPWL={hpwl:.0f} line={line:.0f} {elapsed:.1f}s")

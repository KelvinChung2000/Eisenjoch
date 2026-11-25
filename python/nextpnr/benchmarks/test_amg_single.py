"""Single run benchmark. argv[1] is `heap` or `<init>_it<iters>`, such as `random-bel_it200`."""
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
    ctx.place(nextpnr.PlaceOptions(placer='heap', seed=42))
    elapsed = time.time() - t0
    print(f"HeAP: HPWL={ctx.total_hpwl():.0f} line={ctx.total_line_estimate():.0f} {elapsed:.1f}s")
else:
    init_field, _, iters_field = mode.rpartition("_")
    ctx.place(nextpnr.PlaceOptions(
        placer='hydraulic', seed=42,
        max_iters=int(iters_field.removeprefix("it")),
        init_strategy=init_field.replace("-", "_"),
    ))
    elapsed = time.time() - t0
    hpwl = ctx.total_hpwl()
    line = ctx.total_line_estimate()
    print(f"{mode}: HPWL={hpwl:.0f} line={line:.0f} {elapsed:.1f}s")

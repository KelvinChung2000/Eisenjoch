"""Benchmark: Dijkstra/Adam Beckmann OT placer + raster router vs HeAP."""
import nextpnr
import time

CHIPDB = '/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin'
DESIGN = '/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json'

def run_place_route(name, placer, **place_kwargs):
    print(f"\n=== {name} ===")
    ctx = nextpnr.Context(chipdb=CHIPDB)
    ctx.load_design(DESIGN)
    ctx.pack()
    t0 = time.time()
    ctx.place(placer=placer, seed=42, **place_kwargs)
    t_place = time.time() - t0
    hpwl = ctx.total_hpwl()
    line_est = ctx.total_line_estimate()
    print(f'  Placed: HPWL={hpwl:.0f} line={line_est:.0f} time={t_place:.1f}s')

    t0 = time.time()
    try:
        # max_iterations=1 to get just pass 0 result (avoid A* hang)
        ctx.route(router='raster', max_iterations=1, skip_unplaced=True)
        route_ok = True
    except Exception as e:
        route_ok = False
        print(f'  Route failed: {e}')
    t_route = time.time() - t0
    routed_wl = ctx.total_routed_wirelength()
    print(f'  Routed: WL={routed_wl} OK={route_ok} time={t_route:.1f}s')
    return {
        'name': name, 'hpwl': hpwl, 'line_est': line_est,
        'routed_wl': routed_wl, 'route_ok': route_ok,
        'place_time': t_place, 'route_time': t_route,
    }

results = []
results.append(run_place_route('HeAP', 'heap'))
results.append(run_place_route('Dijkstra OT', 'hydraulic', adam_lr_gain=0.5))

print(f'\n{"Name":<12} {"HPWL":>8} {"LineEst":>8} {"RoutedWL":>10} {"PlaceT":>7} {"RouteT":>7} OK')
print('-' * 70)
for r in results:
    print(f"{r['name']:<12} {r['hpwl']:>8.0f} {r['line_est']:>8.0f} {r['routed_wl']:>10} {r['place_time']:>6.1f}s {r['route_time']:>6.1f}s {r['route_ok']}")

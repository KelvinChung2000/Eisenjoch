"""Run inside nextpnr-generic with --router router2 --run."""
import json
import sys

cells_by_id = {}
nets_by_id = {}
for line in sys.stdin:
    kind, payload = line.split(" ", 1)
    if kind == "W":
        name, x, y = payload.split()
        ctx.addWire(name, "wire", int(x), int(y))
    elif kind == "P":
        pip, src, dst, x, y, delay = payload.split()
        ctx.addPip(pip, "pip", src, dst, ctx.getDelayFromNS(float(delay)), Loc(int(x), int(y), 0))
    elif kind == "N":
        name = payload.strip()
        nets_by_id[name] = ctx.createNet(name)
    elif kind == "T":
        net, cell, bel, x, y, z, port, wire, driver = json.loads(payload)
        if cell not in cells_by_id:
            ctx.addBel(bel, "placed", Loc(x, y, z), False, False)
            cells_by_id[cell] = ctx.createCell(str(cell), "placed")
            ctx.bindBel(bel, cells_by_id[cell], STRENGTH_LOCKED)
        c = cells_by_id[cell]
        if driver:
            ctx.addBelOutput(bel, port, wire)
            c.addOutput(port)
        else:
            ctx.addBelInput(bel, port, wire)
            c.addInput(port)
        ctx.addCellBelPinMapping(str(cell), port, port)
        ctx.connectPort(str(net), str(cell), port)
    else:
        raise ValueError(kind)

if not ctx.route():
    raise RuntimeError("native router failed")
for name, net in nets_by_id.items():
    for wire, binding in net.wires:
        print("ROUTED_WIRE", name, wire, binding.pip or "-")

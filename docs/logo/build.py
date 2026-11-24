"""Generate the Eisenjoch logo SVGs in this directory.

An iron viaduct (a Joch is a bridge pier) stands on configured FPGA fabric and
carries a routed net along its deck. Run from the repository root:

    uv run python docs/logo/build.py
"""
from pathlib import Path

out = Path(__file__).parent
HEIGHTS = [10, 10, 8, 6, 4, 5, 6, 4, 2, 3, 5, 7, 9, 10, 10]  # fabric cells per column
PIERS = [3, 5, 7, 9, 11]                                     # columns carrying a pier
LIGHT = dict(bridge="#1F2933", fabric="#2563EB", net="#D99A00")
DARK = dict(bridge="#E2E8F0", fabric="#60A5FA", net="#FACC15")

def x(c): return 4 + 8 * c
def top(h): return 126 - 8 * h

def mark(fill):
    """fill(role) returns the paint attributes for 'bridge', 'fabric' or 'net'."""
    p = []
    for c, h in enumerate(HEIGHTS):
        for k in range(1, h + 1):
            on = k == h or (c * 7 + k * 13 + c * k) % 5 > 1  # configuration bits; top cell always on
            dim = "" if on else ' fill-opacity=".3"'
            p.append(f'<rect x="{x(c)}" y="{126 - 8 * k}" width="6" height="6" rx="1.2" {fill("fabric")}{dim}/>')
    p.append(f'<rect x="4" y="34" width="118" height="12" rx="2" {fill("bridge")}/>')
    for c in PIERS:
        p.append(f'<rect x="{x(c)}" y="44" width="6" height="{top(HEIGHTS[c]) - 46}" {fill("bridge")}/>')
    edges = [18, *[e for c in PIERS for e in (x(c), x(c) + 6)], 108]
    for b, x2 in zip(edges[::2], edges[1::2]):
        r = (x2 - b) / 2
        sp = 49 + r
        p.append(f'<path d="M{b} 45H{x2}V{sp:g}A{r:g} {r:g} 0 0 0 {b} {sp:g}Z" {fill("bridge")}/>')
    p.append(f'<path d="M10 28H116" {fill("net-line")} stroke-width="3.5" stroke-linecap="round"/>')
    p += [f'<circle cx="{cx}" cy="28" r="4" {fill("net")}/>' for cx in (10, 116)]
    return "\n  ".join(p)

def paint(colors):
    def fill(role):
        if role == "net-line":
            return f'fill="none" stroke="{colors["net"]}"'
        return f'fill="{colors[role]}"'
    return fill

def by_class(role):
    return 'class="net-line"' if role == "net-line" else f'class="{role}"'

def svg(w, h, body, extra=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="Eisenjoch">{extra}\n  {body}\n</svg>\n')

def lockup(c):
    text = (f'<text x="150" y="84" font-family="ui-monospace, \'JetBrains Mono\', \'SF Mono\', Menlo, Consolas, monospace" '
            f'font-size="60" font-weight="600" letter-spacing="-1" fill="{c["bridge"]}">eisen'
            f'<tspan fill="{c["net"]}">joch</tspan></text>')
    return svg(500, 128, mark(paint(c)) + "\n  " + text)

def css(c):
    return (f'.bridge{{fill:{c["bridge"]}}}.fabric{{fill:{c["fabric"]}}}.net{{fill:{c["net"]}}}'
            f'.net-line{{fill:none;stroke:{c["net"]}}}')

for name, c in (("light", LIGHT), ("dark", DARK)):
    (out / f"mark-{name}.svg").write_text(svg(128, 128, mark(paint(c))))
    (out / f"logo-{name}.svg").write_text(lockup(c))
# The favicon follows the browser's colour scheme.
style = f"\n  <style>{css(LIGHT)}@media (prefers-color-scheme: dark){{{css(DARK)}}}</style>"
(out / "favicon.svg").write_text(svg(128, 128, mark(by_class), style))

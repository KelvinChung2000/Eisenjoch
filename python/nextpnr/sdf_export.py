"""Routed delays as an IEEE 1497 SDF file, in ps as (fast_min::slow_max)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sdf_toolkit.core.builder import SDFBuilder
from sdf_toolkit.core.delays import PortSpec, TimingPortSpec
from sdf_toolkit.core.enums import Divider, EdgeType, TimeUnit
from sdf_toolkit.core.identifiers import sdf_escape
from sdf_toolkit.core.sdffile import SDFFile
from sdf_toolkit.core.values import DelayValues, Timescale, Values
from sdf_toolkit.io.writer import emit_sdf

if TYPE_CHECKING:
    from nextpnr import BackAnnotation

PROGRAM = "nextpnr-himbaechel-rust"
TOP_INSTANCE = ""


def sdf_port(name: str) -> str:
    """Escape *name*, keeping a trailing ``[n]`` as an SDF bit-select."""
    base, bracket, index = name.removesuffix("]").rpartition("[")
    if name.endswith("]") and bracket and index.isdigit():
        return f"{sdf_escape(base)}[{index}]"
    return sdf_escape(name)


def bounds(*, early_ps: int, late_ps: int) -> DelayValues:
    return DelayValues.of(Values(min=early_ps, max=late_ps))


def build_sdf(annotation: BackAnnotation) -> SDFFile:
    """Return the SDF model of *annotation*, interconnect under the top cell."""
    builder = SDFBuilder().set_header(
        sdfversion="3.0",
        design=annotation.design,
        program=PROGRAM,
        divider=Divider.SLASH,
        timescale=Timescale(value=1, unit=TimeUnit.PS),
    )
    for cell in annotation.cells:
        entries = builder.add_cell(cell.cell_type, sdf_escape(cell.instance))
        for arc in cell.iopaths:
            edge = None if arc.launch_edge is None else EdgeType(arc.launch_edge)
            entries.add_iopath(
                input=PortSpec(port=sdf_port(arc.from_port), edge=edge),
                output=PortSpec(port=sdf_port(arc.to_port)),
                delay=bounds(early_ps=arc.early_ps, late_ps=arc.late_ps),
            )
        for check in cell.checks:
            data = TimingPortSpec(port=sdf_port(check.data_port))
            clock = TimingPortSpec(
                port=sdf_port(check.clock_port), edge=EdgeType(check.clock_edge)
            )
            entries.add_setup(
                data=data, reference=clock, limit=Values(max=check.setup_ps)
            )
            entries.add_hold(
                data=data, reference=clock, limit=Values(max=check.hold_ps)
            )
    top = builder.add_cell(annotation.design, TOP_INSTANCE)
    for net in annotation.interconnects:
        top.add_interconnect(
            source=PortSpec(
                port=f"{sdf_escape(net.source_instance)}/{sdf_port(net.source_port)}"
            ),
            load=PortSpec(
                port=f"{sdf_escape(net.load_instance)}/{sdf_port(net.load_port)}"
            ),
            delay=bounds(early_ps=net.early_ps, late_ps=net.late_ps),
        )
    return builder.build()


def write_sdf(annotation: BackAnnotation, path: Path) -> None:
    """Write *annotation* to *path* as SDF 3.0."""
    path.write_text(emit_sdf(build_sdf(annotation)))

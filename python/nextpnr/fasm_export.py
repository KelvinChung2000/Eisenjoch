"""Configured pips and cell parameters as a FASM file, feature names taken from the chipdb."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from fasm_toolkit.dsl import FasmBuilder
from fasm_toolkit.ir import FasmFile, ValueFormat

if TYPE_CHECKING:
    from nextpnr import FasmFeature


def build_fasm(features: Iterable[FasmFeature]) -> FasmFile:
    """Return *features* as FASM; raises FasmBuildError on a name outside the grammar."""
    builder = FasmBuilder()
    for feature in features:
        name = f"{feature.tile}.{feature.feature}"
        if feature.bits is None:
            builder.enable(name)
        else:
            builder.set(
                name,
                sum(int(bit) << index for index, bit in enumerate(feature.bits)),
                width=len(feature.bits),
                fmt=ValueFormat.VERILOG_BINARY,
            )
    return builder.build()


def write_fasm(features: Iterable[FasmFeature], path: Path) -> None:
    """Write *features* to *path* as FASM text."""
    path.write_text(build_fasm(features).to_string())

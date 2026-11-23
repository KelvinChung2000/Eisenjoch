"""Cell timing of the xc7 chipdb BEL types from the prjxray Artix-7 SDF files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sdf_toolkit import parse_sdf_file
from sdf_toolkit.core import EdgeType, EntryType
from sdf_toolkit.core.model import BaseEntry, CellsDict, Values

SPEED_GRADE = "1"
PS_PER_NS = 1000
LUT_INPUTS = 6
SLICE_LETTERS = "ABCD"


class UnmappedTimingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Corners:
    fast_min: int
    fast_max: int
    slow_min: int
    slow_max: int

    @classmethod
    def envelope(cls, values: Iterable[Corners]) -> Corners:
        values = list(values)
        return cls(
            fast_min=min(v.fast_min for v in values),
            fast_max=max(v.fast_max for v in values),
            slow_min=min(v.slow_min for v in values),
            slow_max=max(v.slow_max for v in values),
        )


@dataclass(frozen=True)
class SdfCell:
    sdf_file: str
    cell_type: str
    instance: str


@dataclass(frozen=True)
class SdfEntry:
    kind: EntryType
    from_pin: str
    to_pin: str


class ArcKind(Enum):
    COMBINATIONAL = "combinational"
    CLOCK_TO_Q = "clock_to_q"
    SETUP_HOLD = "setup_hold"


@dataclass(frozen=True)
class ArcSpec:
    kind: ArcKind
    bel_from: str
    bel_to: str
    sdf_from: str
    sdf_to: str
    cells: tuple[SdfCell, ...]

    def entries(self) -> tuple[SdfEntry, ...]:
        match self.kind:
            case ArcKind.COMBINATIONAL | ArcKind.CLOCK_TO_Q:
                return (
                    SdfEntry(
                        kind=EntryType.IOPATH,
                        from_pin=self.sdf_from,
                        to_pin=self.sdf_to,
                    ),
                )
            case ArcKind.SETUP_HOLD:
                return (
                    SdfEntry(
                        kind=EntryType.SETUP, from_pin=self.sdf_from, to_pin=self.sdf_to
                    ),
                    SdfEntry(
                        kind=EntryType.HOLD, from_pin=self.sdf_from, to_pin=self.sdf_to
                    ),
                )


@dataclass(frozen=True)
class CombArc:
    from_pin: str
    to_pin: str
    delay: Corners


@dataclass(frozen=True)
class ClockToQArc:
    clock: str
    output: str
    delay: Corners


@dataclass(frozen=True)
class SetupHoldArc:
    clock: str
    input: str
    setup: Corners
    hold: Corners


CellArc = CombArc | ClockToQArc | SetupHoldArc


def _slice_cells(cell_types: Mapping[str, str], instance: str) -> tuple[SdfCell, ...]:
    return tuple(
        SdfCell(
            sdf_file=sdf_file,
            cell_type=cell_type,
            instance=instance.format(site=sdf_file.upper()),
        )
        for sdf_file, cell_type in cell_types.items()
    )


def _lut_cells() -> tuple[SdfCell, ...]:
    cell_types = {"slicel": "LUT6", "slicem": "LUT_OR_MEM6LRAM"}
    return tuple(
        cell
        for letter in SLICE_LETTERS
        for cell in _slice_cells(cell_types, f"{{site}}/{letter}6LUT")
    )


def _ff_cells() -> tuple[SdfCell, ...]:
    return tuple(
        cell
        for cell_type in ("REG_INIT_FF", "FF_INIT")
        for cell in _slice_cells({"slicel": cell_type, "slicem": cell_type}, "{site}")
    )


def _bufg_cells() -> tuple[SdfCell, ...]:
    return tuple(
        SdfCell(sdf_file=sdf_file, cell_type="BUFGCTRL", instance="BUFGCTRL")
        for sdf_file in ("CLK_BUFG_BOT_R", "CLK_BUFG_TOP_R")
    )


BEL_ARCS: dict[str, tuple[ArcSpec, ...]] = {
    "LUT6": tuple(
        ArcSpec(
            kind=ArcKind.COMBINATIONAL,
            bel_from=f"I[{j}]",
            bel_to="F",
            sdf_from=f"A{j + 1}",
            sdf_to="O6",
            cells=_lut_cells(),
        )
        for j in range(LUT_INPUTS)
    ),
    "DFF": (
        ArcSpec(
            kind=ArcKind.CLOCK_TO_Q,
            bel_from="CLK",
            bel_to="Q",
            sdf_from="CLK",
            sdf_to="Q",
            cells=_ff_cells(),
        ),
        ArcSpec(
            kind=ArcKind.SETUP_HOLD,
            bel_from="CLK",
            bel_to="D",
            sdf_from="CLK",
            sdf_to="DIN",
            cells=_ff_cells(),
        ),
    ),
    "BUFG": (
        ArcSpec(
            kind=ArcKind.COMBINATIONAL,
            bel_from="I",
            bel_to="O",
            sdf_from="I",
            sdf_to="O",
            cells=_bufg_cells(),
        ),
    ),
}

DROPPED_ENTRIES: dict[str, frozenset[SdfEntry]] = {
    "LUT6": frozenset(),
    "LUT_OR_MEM6LRAM": frozenset(
        {
            SdfEntry(kind=EntryType.IOPATH, from_pin="CLK", to_pin="O6"),
            *(
                SdfEntry(kind=kind, from_pin="CLK", to_pin=pin)
                for kind in (EntryType.SETUP, EntryType.HOLD)
                for pin in ("DI1", "DI2")
            ),
        }
    ),
    "REG_INIT_FF": frozenset(
        SdfEntry(kind=kind, from_pin="CLK", to_pin="CE")
        for kind in (EntryType.SETUP, EntryType.HOLD)
    ),
    "FF_INIT": frozenset(
        SdfEntry(kind=kind, from_pin="CLK", to_pin="CE")
        for kind in (EntryType.SETUP, EntryType.HOLD)
    ),
    "BUFGCTRL": frozenset(),
}


def _ps(ns: float) -> int:
    ps = round(ns * PS_PER_NS)
    if abs(ns * PS_PER_NS - ps) > 1e-6:
        raise UnmappedTimingError(f"{ns} ns is not a whole number of ps")
    return ps


def _bounds(values: Values | None, what: str) -> tuple[int, int]:
    if (
        values is None
        or values.min is None
        or values.max is None
        or values.avg is not None
    ):
        raise UnmappedTimingError(f"{what}: expected a (min::max) triple, got {values}")
    return _ps(values.min), _ps(values.max)


def _delay_corners(entry: BaseEntry, what: str) -> Corners:
    paths = entry.delay_paths
    if paths is None or paths.nominal is not None:
        raise UnmappedTimingError(
            f"{what}: expected (fast_min::fast_max)(slow_min::slow_max)"
        )
    fast_min, fast_max = _bounds(paths.fast, what)
    slow_min, slow_max = _bounds(paths.slow, what)
    return Corners(
        fast_min=fast_min, fast_max=fast_max, slow_min=slow_min, slow_max=slow_max
    )


def _check_corners(entry: BaseEntry, what: str) -> Corners:
    paths = entry.delay_paths
    if paths is None or paths.fast is not None or paths.slow is not None:
        raise UnmappedTimingError(f"{what}: expected one slow-corner (min::max) triple")
    if entry.from_pin_edge != EdgeType.POSEDGE:
        raise UnmappedTimingError(
            f"{what}: expected a posedge clock, got {entry.from_pin_edge}"
        )
    slow_min, slow_max = _bounds(paths.nominal, what)
    return Corners(
        fast_min=slow_min, fast_max=slow_max, slow_min=slow_min, slow_max=slow_max
    )


class SdfLibrary:
    def __init__(self, timings_dir: Path) -> None:
        self._timings_dir = timings_dir
        self._files: dict[str, CellsDict] = {}

    def cell(self, cell: SdfCell) -> dict[SdfEntry, BaseEntry]:
        if cell.sdf_file not in self._files:
            self._files[cell.sdf_file] = parse_sdf_file(
                self._timings_dir / f"{cell.sdf_file}.sdf"
            ).cells
        instances = self._files[cell.sdf_file].get(cell.cell_type)
        if instances is None or cell.instance not in instances:
            raise UnmappedTimingError(f"{cell} is not in {cell.sdf_file}.sdf")
        entries: dict[SdfEntry, BaseEntry] = {}
        for entry in instances[cell.instance].values():
            if entry.from_pin is None or entry.to_pin is None:
                raise UnmappedTimingError(
                    f"{cell}: entry {entry.name} has no from/to pin"
                )
            key = SdfEntry(
                kind=entry.type, from_pin=entry.from_pin, to_pin=entry.to_pin
            )
            if key in entries:
                raise UnmappedTimingError(f"{cell}: two SDF entries for {key}")
            entries[key] = entry
        return entries


def _resolve(spec: ArcSpec, library: SdfLibrary) -> CellArc:
    def corners(entry: SdfEntry) -> Corners:
        values = []
        for cell in spec.cells:
            sdf_entry = library.cell(cell).get(entry)
            if sdf_entry is None:
                raise UnmappedTimingError(f"{cell} has no {entry}")
            what = f"{cell} {entry}"
            check = entry.kind in (EntryType.SETUP, EntryType.HOLD)
            values.append(
                _check_corners(sdf_entry, what)
                if check
                else _delay_corners(sdf_entry, what)
            )
        return Corners.envelope(values)

    match spec.kind:
        case ArcKind.COMBINATIONAL:
            (iopath,) = spec.entries()
            return CombArc(
                from_pin=spec.bel_from, to_pin=spec.bel_to, delay=corners(iopath)
            )
        case ArcKind.CLOCK_TO_Q:
            (iopath,) = spec.entries()
            return ClockToQArc(
                clock=spec.bel_from, output=spec.bel_to, delay=corners(iopath)
            )
        case ArcKind.SETUP_HOLD:
            setup, hold = spec.entries()
            return SetupHoldArc(
                clock=spec.bel_from,
                input=spec.bel_to,
                setup=corners(setup),
                hold=corners(hold),
            )


def _check_sdf_coverage(library: SdfLibrary) -> None:
    consumed: dict[SdfCell, set[SdfEntry]] = {}
    for specs in BEL_ARCS.values():
        for spec in specs:
            for cell in spec.cells:
                consumed.setdefault(cell, set()).update(spec.entries())
    for cell, entries in consumed.items():
        unmapped = set(library.cell(cell)) - entries - DROPPED_ENTRIES[cell.cell_type]
        if unmapped:
            raise UnmappedTimingError(
                f"{cell}: SDF entries without a BEL arc: {sorted(map(str, unmapped))}"
            )


def _check_bel_coverage(bel_pins: Mapping[str, set[str]]) -> None:
    for bel_type, pins in bel_pins.items():
        if bel_type not in BEL_ARCS:
            raise UnmappedTimingError(f"BEL {bel_type} has no SDF arc table")
        timed = {
            pin for spec in BEL_ARCS[bel_type] for pin in (spec.bel_from, spec.bel_to)
        }
        if timed != pins:
            raise UnmappedTimingError(
                f"BEL {bel_type}: pins {sorted(pins)}, SDF arcs cover {sorted(timed)}"
            )


def cell_timing(
    timings_dir: Path, bel_pins: Mapping[str, set[str]]
) -> dict[str, tuple[CellArc, ...]]:
    library = SdfLibrary(timings_dir)
    _check_bel_coverage(bel_pins)
    _check_sdf_coverage(library)
    return {
        bel_type: tuple(_resolve(spec, library) for spec in BEL_ARCS[bel_type])
        for bel_type in bel_pins
    }

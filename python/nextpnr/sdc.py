"""SDC (Synopsys Design Constraints) parser for nextpnr.

Parses a subset of SDC commands used in FPGA timing constraints
and produces Python dataclasses suitable for passing to the Rust backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lark import Lark, Transformer, v_args

# -- Dataclasses representing parsed SDC constraints --


@dataclass
class ClockDef:
    name: str
    period_ns: float
    waveform: tuple[float, float] | None = None
    source_port: str | None = None


@dataclass
class IoDelay:
    clock: str
    delay_ns: float
    ports: list[str]
    is_max: bool = True


@dataclass
class FalsePath:
    from_pins: list[str]
    to_pins: list[str]
    through_pins: list[str]


@dataclass
class MulticyclePath:
    from_clock: str
    to_clock: str
    setup_cycles: int = 1
    hold_cycles: int = 0


@dataclass
class ClockGroup:
    group_type: str  # "asynchronous" | "exclusive" | "physically_exclusive"
    groups: list[list[str]]


@dataclass
class SdcConstraints:
    clocks: list[ClockDef] = field(default_factory=list)
    input_delays: list[IoDelay] = field(default_factory=list)
    output_delays: list[IoDelay] = field(default_factory=list)
    false_paths: list[FalsePath] = field(default_factory=list)
    multicycle_paths: list[MulticyclePath] = field(default_factory=list)
    clock_groups: list[ClockGroup] = field(default_factory=list)
    clock_uncertainty: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a dict suitable for passing to Rust via PyO3."""
        return {
            "clocks": [
                {
                    "name": c.name,
                    "period_ns": c.period_ns,
                    "waveform": list(c.waveform) if c.waveform else None,
                    "source_port": c.source_port,
                }
                for c in self.clocks
            ],
            "input_delays": [
                {
                    "clock": d.clock,
                    "delay_ns": d.delay_ns,
                    "ports": d.ports,
                    "is_max": d.is_max,
                }
                for d in self.input_delays
            ],
            "output_delays": [
                {
                    "clock": d.clock,
                    "delay_ns": d.delay_ns,
                    "ports": d.ports,
                    "is_max": d.is_max,
                }
                for d in self.output_delays
            ],
            "false_paths": [
                {
                    "from_pins": fp.from_pins,
                    "to_pins": fp.to_pins,
                    "through_pins": fp.through_pins,
                }
                for fp in self.false_paths
            ],
            "multicycle_paths": [
                {
                    "from_clock": mp.from_clock,
                    "to_clock": mp.to_clock,
                    "setup_cycles": mp.setup_cycles,
                    "hold_cycles": mp.hold_cycles,
                }
                for mp in self.multicycle_paths
            ],
            "clock_groups": [
                {
                    "group_type": cg.group_type,
                    "groups": cg.groups,
                }
                for cg in self.clock_groups
            ],
            "clock_uncertainty": [
                {
                    "from_clock": cu[0],
                    "to_clock": cu[1],
                    "uncertainty_ns": cu[2],
                }
                for cu in self.clock_uncertainty
            ],
        }


# -- Lark grammar for SDC subset --

SDC_GRAMMAR = r"""
start: (_NL | command)*

_NL: NEWLINE

command: create_clock
       | set_input_delay
       | set_output_delay
       | set_false_path
       | set_multicycle_path
       | set_clock_groups
       | set_clock_uncertainty

create_clock: "create_clock" create_clock_opt+
create_clock_opt: "-name" name          -> cc_name
                | "-period" number      -> cc_period
                | "-waveform" brace_list -> cc_waveform
                | port_expr             -> cc_source

set_input_delay: "set_input_delay" io_delay_opt+
set_output_delay: "set_output_delay" io_delay_opt+
io_delay_opt: "-clock" name_or_bracket  -> iod_clock
            | "-max"                    -> iod_max
            | "-min"                    -> iod_min
            | port_expr                 -> iod_ports
            | number                    -> iod_value

set_false_path: "set_false_path" false_path_opt+
false_path_opt: "-from" pin_expr  -> fp_from
              | "-to" pin_expr    -> fp_to
              | "-through" pin_expr -> fp_through

set_multicycle_path: "set_multicycle_path" multicycle_opt+
multicycle_opt: "-from" name_or_bracket   -> mc_from
              | "-to" name_or_bracket     -> mc_to
              | "-setup" INTEGER          -> mc_setup
              | "-hold" INTEGER           -> mc_hold
              | INTEGER                   -> mc_path_mult

set_clock_groups: "set_clock_groups" clock_group_opt+
clock_group_opt: "-asynchronous"          -> cg_async
               | "-exclusive"             -> cg_exclusive
               | "-physically_exclusive"  -> cg_phys_exclusive
               | "-group" pin_expr        -> cg_group

set_clock_uncertainty: "set_clock_uncertainty" clock_uncert_opt+
clock_uncert_opt: "-from" name_or_bracket  -> cu_from
                | "-to" name_or_bracket    -> cu_to
                | "-setup"                 -> cu_setup
                | "-hold"                  -> cu_hold
                | number                   -> cu_value

// Pin/port expressions: bare name, [get_ports ...], [get_cells ...], [get_clocks ...], [get_pins ...]
pin_expr: bracket_expr
        | brace_list
        | name

port_expr: bracket_expr
         | name

bracket_expr: "[" BRACKET_CMD name_list "]"
BRACKET_CMD: "get_ports" | "get_cells" | "get_clocks" | "get_pins" | "get_nets"
name_list: (name | brace_list)+

brace_list: "{" brace_item+ "}"
brace_item: SIGNED_NUMBER | ESCAPED_STRING | IDENTIFIER

name_or_bracket: bracket_expr | name

name: ESCAPED_STRING | IDENTIFIER

number: SIGNED_NUMBER

IDENTIFIER: /[a-zA-Z_*?][a-zA-Z0-9_.*?\/\:$\\-]*/
INTEGER: /[0-9]+/
SIGNED_NUMBER: /[+-]?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?/

COMMENT: /#[^\n]*/

%import common.ESCAPED_STRING
%import common.NEWLINE
%import common.WS_INLINE
%ignore WS_INLINE
%ignore COMMENT
"""


def _flatten_names(tree_or_list) -> list[str]:
    """Extract all name strings from a nested tree structure."""
    if isinstance(tree_or_list, str):
        return [tree_or_list]
    if isinstance(tree_or_list, list):
        result = []
        for item in tree_or_list:
            result.extend(_flatten_names(item))
        return result
    return [str(tree_or_list)]


@v_args(inline=True)
class SdcTransformer(Transformer):
    """Transform the parse tree into SDC dataclasses."""

    def start(self, *commands):
        constraints = SdcConstraints()
        for cmd in commands:
            if cmd is None:
                continue
            if isinstance(cmd, ClockDef):
                constraints.clocks.append(cmd)
            elif isinstance(cmd, IoDelay):
                if cmd.is_max is None:
                    # Determine from context (set during transform)
                    cmd.is_max = True
                if hasattr(cmd, "_is_input") and cmd._is_input:
                    constraints.input_delays.append(cmd)
                else:
                    constraints.output_delays.append(cmd)
            elif isinstance(cmd, FalsePath):
                constraints.false_paths.append(cmd)
            elif isinstance(cmd, MulticyclePath):
                constraints.multicycle_paths.append(cmd)
            elif isinstance(cmd, ClockGroup):
                constraints.clock_groups.append(cmd)
            elif isinstance(cmd, tuple) and len(cmd) == 3:
                constraints.clock_uncertainty.append(cmd)
        return constraints

    def command(self, cmd):
        return cmd

    # -- Names and values --

    def name(self, token):
        s = str(token)
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s

    def number(self, token):
        return float(token)

    def SIGNED_NUMBER(self, token):
        return float(token)

    def INTEGER(self, token):
        return int(token)

    def brace_item(self, token):
        s = str(token)
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s

    def brace_list(self, *items):
        return list(items)

    def name_list(self, *items):
        result = []
        for item in items:
            result.extend(_flatten_names(item))
        return result

    def bracket_expr(self, cmd, names):
        # cmd is the BRACKET_CMD terminal token, names is the name_list result
        return names

    def name_or_bracket(self, val):
        if isinstance(val, list):
            return val[0] if len(val) == 1 else val
        return val

    def pin_expr(self, val):
        if isinstance(val, list):
            return val
        return [val]

    def port_expr(self, val):
        if isinstance(val, list):
            return val[0] if len(val) == 1 else val
        return val

    # -- create_clock --

    def cc_name(self, n):
        return ("name", n)

    def cc_period(self, v):
        return ("period", v)

    def cc_waveform(self, vals):
        return ("waveform", vals)

    def cc_source(self, port):
        return ("source", port)

    def create_clock(self, *opts):
        name = None
        period = None
        waveform = None
        source = None
        for key, val in opts:
            if key == "name":
                name = val
            elif key == "period":
                period = val
            elif key == "waveform":
                waveform = (float(val[0]), float(val[1]))
            elif key == "source":
                source = val
        if name is None and source is not None:
            name = source
        if name is None:
            name = "unnamed"
        return ClockDef(
            name=name,
            period_ns=period,
            waveform=waveform,
            source_port=source,
        )

    # -- set_input_delay / set_output_delay --

    def iod_clock(self, val):
        return ("clock", val)

    def iod_max(self):
        return ("max_flag", True)

    def iod_min(self):
        return ("min_flag", True)

    def iod_ports(self, val):
        return ("ports", val)

    def iod_value(self, val):
        return ("value", val)

    def _build_io_delay(self, opts, is_input: bool):
        clock = None
        delay = None
        ports = []
        is_max = True
        for key, *val in opts:
            v = val[0] if val else None
            if key == "clock":
                clock = v
            elif key == "value":
                delay = v
            elif key == "ports":
                ports = v if isinstance(v, list) else [v]
            elif key == "max_flag":
                is_max = True
            elif key == "min_flag":
                is_max = False
        result = IoDelay(clock=clock, delay_ns=delay, ports=ports, is_max=is_max)
        result._is_input = is_input
        return result

    def set_input_delay(self, *opts):
        return self._build_io_delay(opts, is_input=True)

    def set_output_delay(self, *opts):
        return self._build_io_delay(opts, is_input=False)

    # -- set_false_path --

    def fp_from(self, val):
        return ("from", val)

    def fp_to(self, val):
        return ("to", val)

    def fp_through(self, val):
        return ("through", val)

    def set_false_path(self, *opts):
        from_pins = []
        to_pins = []
        through_pins = []
        for key, val in opts:
            if key == "from":
                from_pins = val if isinstance(val, list) else [val]
            elif key == "to":
                to_pins = val if isinstance(val, list) else [val]
            elif key == "through":
                through_pins = val if isinstance(val, list) else [val]
        return FalsePath(
            from_pins=from_pins,
            to_pins=to_pins,
            through_pins=through_pins,
        )

    # -- set_multicycle_path --

    def mc_from(self, val):
        return ("from", val)

    def mc_to(self, val):
        return ("to", val)

    def mc_setup(self, val):
        return ("setup", val)

    def mc_hold(self, val):
        return ("hold", val)

    def mc_path_mult(self, val):
        return ("mult", val)

    def set_multicycle_path(self, *opts):
        from_clock = None
        to_clock = None
        setup = 1
        hold = 0
        mult = None
        for key, val in opts:
            if key == "from":
                from_clock = val
            elif key == "to":
                to_clock = val
            elif key == "setup":
                setup = val
            elif key == "hold":
                hold = val
            elif key == "mult":
                mult = val
        if mult is not None and setup == 1:
            setup = mult
        return MulticyclePath(
            from_clock=from_clock,
            to_clock=to_clock,
            setup_cycles=setup,
            hold_cycles=hold,
        )

    # -- set_clock_groups --

    def cg_async(self):
        return ("type", "asynchronous")

    def cg_exclusive(self):
        return ("type", "exclusive")

    def cg_phys_exclusive(self):
        return ("type", "physically_exclusive")

    def cg_group(self, val):
        return ("group", val if isinstance(val, list) else [val])

    def set_clock_groups(self, *opts):
        group_type = "asynchronous"
        groups = []
        for key, val in opts:
            if key == "type":
                group_type = val
            elif key == "group":
                groups.append(val)
        return ClockGroup(group_type=group_type, groups=groups)

    # -- set_clock_uncertainty --

    def cu_from(self, val):
        return ("from", val)

    def cu_to(self, val):
        return ("to", val)

    def cu_setup(self):
        return ("setup_flag", True)

    def cu_hold(self):
        return ("hold_flag", True)

    def cu_value(self, val):
        return ("value", val)

    def set_clock_uncertainty(self, *opts):
        from_clock = None
        to_clock = None
        value = None
        for key, *val in opts:
            v = val[0] if val else None
            if key == "from":
                from_clock = v
            elif key == "to":
                to_clock = v
            elif key == "value":
                value = v
        return (from_clock or "", to_clock or "", value or 0.0)


_parser = Lark(
    SDC_GRAMMAR,
    parser="earley",
    ambiguity="resolve",
)

_transformer = SdcTransformer()


def parse_sdc(path: str) -> SdcConstraints:
    """Parse an SDC file and return structured constraints.

    Args:
        path: Path to the SDC file.

    Returns:
        SdcConstraints dataclass with all parsed constraints.
    """
    text = Path(path).read_text()
    # Handle line continuations (backslash-newline)
    text = text.replace("\\\n", " ")
    tree = _parser.parse(text)
    return _transformer.transform(tree)

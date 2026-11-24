"""CLI for Eisenjoch's Rust Himbächel flow."""

try:
    import typer
except ImportError:
    import sys

    print(
        "Error: typer is required for the CLI. Run with `uv run --extra cli`.",
        file=sys.stderr,
    )
    sys.exit(1)

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from nextpnr import TimingReport

app = typer.Typer(
    name="nextpnr-himbaechel-rust",
    help="FPGA place-and-route tool (Rust implementation)",
    add_completion=False,
)


@app.command()
def main(
    device: Optional[str] = typer.Option(None, help="Target device name"),
    chipdb: Optional[Path] = typer.Option(None, help="ChipDB binary path"),
    json: Optional[Path] = typer.Option(
        None, "--json", help="Input Yosys JSON netlist"
    ),
    write: Optional[Path] = typer.Option(None, help="Output placed/routed JSON"),
    placer: str = typer.Option("heap", help="Placer algorithm [heap|sa]"),
    router: str = typer.Option("router2", help="Router algorithm [router1|router2]"),
    seed: int = typer.Option(1, help="Random seed"),
    freq: Optional[float] = typer.Option(None, help="Frequency constraint (MHz)"),
    package: Optional[str] = typer.Option(None, help="Package name"),
    speed: Optional[str] = typer.Option(None, help="Speed grade"),
    sdc: Optional[Path] = typer.Option(None, help="SDC constraints file"),
    report: Optional[Path] = typer.Option(
        None, help="Timing/utilization report (JSON)"
    ),
    sdf: Optional[Path] = typer.Option(None, help="SDF timing output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    force: bool = typer.Option(False, "--force", help="Force continue on errors"),
    timing_driven: bool = typer.Option(
        True,
        "--timing-driven/--no-timing-driven",
        help="Timing-driven refinement and routing (router2 only)",
    ),
    timing_allow_fail: bool = typer.Option(
        False,
        "--timing-allow-fail",
        help="Exit 0 despite negative setup or hold slack",
    ),
    vopt: Optional[list[str]] = typer.Option(
        None, "-o", "--vopt", help="Arch-specific options (key=value)"
    ),
    packer_plugin: Optional[Path] = typer.Option(
        None, "--packer-plugin", help="Packer plugin (.so/.dll or .py)"
    ),
    placer_plugin: Optional[Path] = typer.Option(
        None, "--placer-plugin", help="Placer plugin"
    ),
    router_plugin: Optional[Path] = typer.Option(
        None, "--router-plugin", help="Router plugin"
    ),
    constraints: Optional[Path] = typer.Option(
        None, "--constraints", help="Region constraints JSON file"
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", help="Load reference checkpoint for incremental P&R"
    ),
    save_checkpoint: Optional[Path] = typer.Option(
        None, "--save-checkpoint", help="Save checkpoint after P&R"
    ),
    script: Optional[Path] = typer.Option(
        None, "--script", help="Run Python script with pre-loaded ctx"
    ),
) -> None:
    """Run the Rust FPGA place-and-route flow."""
    try:
        import nextpnr
    except ImportError:
        typer.echo(
            "Error: nextpnr module not found. Install with: pip install nextpnr",
            err=True,
        )
        raise typer.Exit(1)

    # Validate inputs
    if chipdb is None and device is None:
        typer.echo("Error: Either --chipdb or --device must be specified", err=True)
        raise typer.Exit(1)

    if json is None and script is None:
        typer.echo("Error: Either --json or --script must be specified", err=True)
        raise typer.Exit(1)

    try:
        ctx_kwargs = {}
        if chipdb is not None:
            ctx_kwargs["chipdb"] = str(chipdb)
        if device is not None:
            ctx_kwargs["device"] = device
        ctx = nextpnr.RustContext(**ctx_kwargs)
    except Exception as e:
        typer.echo(f"Error creating context: {e}", err=True)
        raise typer.Exit(1)

    # Run script mode
    if script is not None:
        script_globals = {"ctx": ctx, "nextpnr": nextpnr}
        try:
            exec(script.read_text(), script_globals)
        except Exception as e:
            typer.echo(f"Script error: {e}", err=True)
            raise typer.Exit(1)
        return

    timed = freq is not None or sdc is not None
    if timing_driven and not timed:
        typer.echo(
            "Error: timing-driven P&R needs a clock; pass --freq or an --sdc "
            "with create_clock, or pass --no-timing-driven",
            err=True,
        )
        raise typer.Exit(1)
    if timing_driven and router != "router2":
        typer.echo(
            f"Error: router {router} is not timing-driven; pass --router router2 "
            "or --no-timing-driven",
            err=True,
        )
        raise typer.Exit(1)
    if report is not None and not timed:
        typer.echo(
            "Error: --report needs a clock; pass --freq or an --sdc with create_clock",
            err=True,
        )
        raise typer.Exit(1)

    # Load design
    if json is not None:
        try:
            ctx.load_design(str(json))
        except Exception as e:
            typer.echo(f"Error loading design: {e}", err=True)
            raise typer.Exit(1)

    if verbose:
        typer.echo(
            f"Loaded design: {ctx.width}x{ctx.height} grid, {len(ctx.cells)} cells, {len(ctx.nets)} nets"
        )

    # Apply region constraints
    if constraints is not None:
        import json as json_mod

        try:
            constraint_data = json_mod.loads(constraints.read_text())
            for region_def in constraint_data.get("regions", []):
                rects = [
                    [r["x0"], r["y0"], r["x1"], r["y1"]] for r in region_def["rects"]
                ]
                region_idx = ctx.add_region(region_def["name"], rects)
                for cell_name in region_def.get("cells", []):
                    ctx.constrain_cell_to_region(cell_name, region_idx)
            if verbose:
                typer.echo(
                    f"Loaded {len(constraint_data.get('regions', []))} region constraints"
                )
        except Exception as e:
            typer.echo(f"Error loading constraints: {e}", err=True)
            if not force:
                raise typer.Exit(1)

    # Load checkpoint for incremental flow
    if checkpoint is not None:
        try:
            ctx.load_checkpoint(str(checkpoint))
            if verbose:
                typer.echo("Loaded checkpoint for incremental P&R")
        except Exception as e:
            typer.echo(f"Error loading checkpoint: {e}", err=True)
            if not force:
                raise typer.Exit(1)

    # Set frequency constraint
    if freq is not None:
        ctx.add_clock("clk", freq_mhz=freq)

    # Run P&R flow
    try:
        if verbose:
            typer.echo("Running packer...")
        ctx.pack()
    except Exception as e:
        typer.echo(f"Packer error: {e}", err=True)
        raise typer.Exit(1)

    if sdc is not None:
        try:
            ctx.apply_sdc(sdc=sdc.read_text())
        except Exception as e:
            typer.echo(f"Error loading SDC constraints: {e}", err=True)
            raise typer.Exit(1)
        if verbose:
            typer.echo(f"Loaded SDC constraints from {sdc}")

    try:
        if verbose:
            typer.echo(f"Running placer ({placer})...")
        ctx.place(placer=placer, seed=seed)

        if verbose:
            typer.echo(f"Running refinement (timing-driven: {timing_driven})...")
        ctx.refine(timing_driven=timing_driven)

        if verbose:
            typer.echo(f"Running router ({router})...")
        ctx.route(router=router, timing_driven=timing_driven)
    except Exception as e:
        typer.echo(f"P&R error: {e}", err=True)
        if not force:
            raise typer.Exit(1)

    # Save checkpoint
    if save_checkpoint is not None:
        try:
            ctx.save_checkpoint(str(save_checkpoint))
            if verbose:
                typer.echo(f"Saved checkpoint to {save_checkpoint}")
        except Exception as e:
            typer.echo(f"Error saving checkpoint: {e}", err=True)

    timing = None
    if timed:
        try:
            timing = ctx.timing_report(stage="routed")
        except Exception as e:
            typer.echo(f"Timing analysis error: {e}", err=True)
            raise typer.Exit(1)
        typer.echo(str(timing))

    if report is not None and timing is not None:
        import json as json_mod

        report_data = {
            "fmax_mhz": timing.fmax_mhz,
            "wns_ns": timing.wns_ns,
            "tns_ns": timing.tns_ns,
            "num_failing": timing.num_failing,
            "num_endpoints": timing.num_endpoints,
            "hold_wns_ns": timing.hold_wns_ns,
            "hold_tns_ns": timing.hold_tns_ns,
            "num_hold_failing": timing.num_hold_failing,
            "num_hold_endpoints": timing.num_hold_endpoints,
        }
        report.write_text(json_mod.dumps(report_data, indent=2))

    # Write output
    if write is not None:
        try:
            ctx.write_design(str(write))
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            if not force:
                raise typer.Exit(1)

    if timing is not None:
        failures = timing_failures(timing)
        for failure in failures:
            typer.echo(f"Timing check failed: {failure}", err=True)
        if failures and not timing_allow_fail:
            typer.echo(
                "Error: timing constraints not met; pass --timing-allow-fail to accept",
                err=True,
            )
            raise typer.Exit(1)

    if verbose:
        typer.echo("Done.")


def timing_failures(timing: "TimingReport") -> list[str]:
    """Return one message per failed sign-off check; an empty list means timing is met."""
    failures = []
    if timing.num_endpoints == 0:
        failures.append("no setup endpoint is constrained; check the clock definitions")
    if timing.num_failing > 0:
        failures.append(
            f"setup WNS {timing.wns_ns:.3f} ns, {timing.num_failing}/{timing.num_endpoints} endpoints failing"
        )
    if timing.num_hold_failing > 0:
        failures.append(
            f"hold WNS {timing.hold_wns_ns:.3f} ns, {timing.num_hold_failing}/{timing.num_hold_endpoints} endpoints failing"
        )
    return failures


def cli() -> None:
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    cli()

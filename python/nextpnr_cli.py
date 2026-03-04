"""nextpnr-himbaechel CLI - Python CLI for the nextpnr Rust FPGA place-and-route tool."""

try:
    import typer
except ImportError:
    import sys
    print("Error: typer is required for the CLI. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

from typing import Optional
from pathlib import Path

app = typer.Typer(
    name="nextpnr-himbaechel",
    help="FPGA place-and-route tool (Rust implementation)",
    add_completion=False,
)


@app.command()
def main(
    device: Optional[str] = typer.Option(None, help="Target device name"),
    chipdb: Optional[Path] = typer.Option(None, help="ChipDB binary path"),
    json: Optional[Path] = typer.Option(None, "--json", help="Input Yosys JSON netlist"),
    write: Optional[Path] = typer.Option(None, help="Output placed/routed JSON"),
    placer: str = typer.Option("heap", help="Placer algorithm [heap|sa]"),
    router: str = typer.Option("router1", help="Router algorithm [router1|router2]"),
    seed: int = typer.Option(1, help="Random seed"),
    freq: Optional[float] = typer.Option(None, help="Frequency constraint (MHz)"),
    package: Optional[str] = typer.Option(None, help="Package name"),
    speed: Optional[str] = typer.Option(None, help="Speed grade"),
    sdc: Optional[Path] = typer.Option(None, help="SDC constraints file"),
    report: Optional[Path] = typer.Option(None, help="Timing/utilization report (JSON)"),
    sdf: Optional[Path] = typer.Option(None, help="SDF timing output"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    force: bool = typer.Option(False, "--force", help="Force continue on errors"),
    timing_driven: bool = typer.Option(True, "--timing-driven/--no-timing-driven", help="Timing-driven P&R"),
    timing_allow_fail: bool = typer.Option(False, "--timing-allow-fail", help="Allow timing violations"),
    vopt: Optional[list[str]] = typer.Option(None, "-o", "--vopt", help="Arch-specific options (key=value)"),
    packer_plugin: Optional[Path] = typer.Option(None, "--packer-plugin", help="Packer plugin (.so/.dll or .py)"),
    placer_plugin: Optional[Path] = typer.Option(None, "--placer-plugin", help="Placer plugin"),
    router_plugin: Optional[Path] = typer.Option(None, "--router-plugin", help="Router plugin"),
    script: Optional[Path] = typer.Option(None, "--script", help="Run Python script with pre-loaded ctx"),
) -> None:
    """Run the nextpnr-himbaechel FPGA place-and-route flow."""
    try:
        import nextpnr
    except ImportError:
        typer.echo("Error: nextpnr module not found. Install with: pip install nextpnr", err=True)
        raise typer.Exit(1)

    # Validate inputs
    if chipdb is None and device is None:
        typer.echo("Error: Either --chipdb or --device must be specified", err=True)
        raise typer.Exit(1)

    if json is None and script is None:
        typer.echo("Error: Either --json or --script must be specified", err=True)
        raise typer.Exit(1)

    # Create context
    try:
        ctx_kwargs = {}
        if chipdb is not None:
            ctx_kwargs["chipdb"] = str(chipdb)
        if device is not None:
            ctx_kwargs["device"] = device
        ctx = nextpnr.Context(**ctx_kwargs)
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

    # Load design
    if json is not None:
        try:
            ctx.load_design(str(json))
        except Exception as e:
            typer.echo(f"Error loading design: {e}", err=True)
            raise typer.Exit(1)

    if verbose:
        typer.echo(f"Loaded design: {ctx.width}x{ctx.height} grid, {len(ctx.cells)} cells, {len(ctx.nets)} nets")

    # Set frequency constraint
    if freq is not None:
        ctx.add_clock("clk", freq)

    # Run P&R flow
    try:
        if verbose:
            typer.echo("Running packer...")
        ctx.pack()

        if verbose:
            typer.echo(f"Running placer ({placer})...")
        ctx.place(placer=placer, seed=seed)

        if verbose:
            typer.echo(f"Running router ({router})...")
        ctx.route(router=router)
    except Exception as e:
        typer.echo(f"P&R error: {e}", err=True)
        if not force:
            raise typer.Exit(1)

    # Timing report
    try:
        timing = ctx.timing_report()
        if verbose or report is not None:
            typer.echo(f"Fmax: {timing.fmax:.2f} MHz")
            typer.echo(f"Worst slack: {timing.worst_slack} ps")
            typer.echo(f"Failing endpoints: {timing.num_failing}/{timing.num_endpoints}")

        if report is not None:
            import json as json_mod
            report_data = {
                "fmax_mhz": timing.fmax,
                "worst_slack_ps": timing.worst_slack,
                "num_failing": timing.num_failing,
                "num_endpoints": timing.num_endpoints,
            }
            report.write_text(json_mod.dumps(report_data, indent=2))
    except Exception as e:
        if verbose:
            typer.echo(f"Timing report error: {e}", err=True)

    # Write output
    if write is not None:
        try:
            ctx.write_design(str(write))
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            if not force:
                raise typer.Exit(1)

    if verbose:
        typer.echo("Done.")


def cli() -> None:
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    cli()

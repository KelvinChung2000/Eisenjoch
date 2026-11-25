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

NEXTPNR_DEFAULT_TARGET_MHZ = 12.0

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
    placer: str = typer.Option("heap", help="Placer algorithm [heap|sa]"),
    router: str = typer.Option("router2", help="Router algorithm [router1|router2]"),
    seed: int = typer.Option(1, help="Random seed"),
    freq: Optional[float] = typer.Option(
        None,
        help="Clock frequency (MHz) for every register clock net; without it and "
        "without --sdc, nextpnr's default target of 12 MHz",
    ),
    package: Optional[str] = typer.Option(None, help="Package name"),
    speed: Optional[str] = typer.Option(None, help="Speed grade"),
    sdc: Optional[Path] = typer.Option(None, help="SDC constraints file"),
    report: Optional[Path] = typer.Option(
        None, help="Timing/utilization report (JSON)"
    ),
    write: Optional[Path] = typer.Option(
        None, help="Write the routed design as nextpnr JSON"
    ),
    sdf: Optional[Path] = typer.Option(None, help="Write routed delays as SDF 3.0"),
    fasm: Optional[Path] = typer.Option(
        None, help="Write the configured features as FASM, named by the chipdb"
    ),
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

    if freq is not None and sdc is not None:
        typer.echo(
            "Error: --freq and --sdc both define clocks; put create_clock in the SDC",
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
    timed = timing_driven or report is not None or freq is not None or sdc is not None

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

    # Run P&R flow
    try:
        if verbose:
            typer.echo("Running packer...")
        ctx.pack()
    except Exception as e:
        typer.echo(f"Packer error: {e}", err=True)
        raise typer.Exit(1)

    if timed and sdc is None:
        target_mhz = freq if freq is not None else NEXTPNR_DEFAULT_TARGET_MHZ
        try:
            clocks = ctx.constrain_clocks(freq_mhz=target_mhz)
        except Exception as e:
            typer.echo(f"Error constraining clocks: {e}", err=True)
            raise typer.Exit(1)
        if timing_driven and not clocks:
            typer.echo(
                "Error: no net drives a register clock pin, so timing-driven P&R "
                "has no clock; check that the chipdb has cell timing, or pass "
                "--no-timing-driven",
                err=True,
            )
            raise typer.Exit(1)
        if verbose:
            typer.echo(
                f"Constrained {', '.join(clocks) or 'no net'} to {target_mhz} MHz"
            )

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
        ctx.refine(timing_driven=timing_driven, crit_norm="wns_relaxed")

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
        report.write_text(timing.nextpnr_json)

    if write is not None:
        try:
            write.write_text(ctx.routed_json())
        except Exception as e:
            typer.echo(f"Error writing routed JSON: {e}", err=True)
            raise typer.Exit(1)
        if verbose:
            typer.echo(f"Wrote routed JSON to {write}")

    if sdf is not None:
        from nextpnr.sdf_export import write_sdf

        try:
            write_sdf(ctx.routed_back_annotation(), sdf)
        except Exception as e:
            typer.echo(f"Error writing SDF: {e}", err=True)
            raise typer.Exit(1)
        if verbose:
            typer.echo(f"Wrote SDF to {sdf}")

    if fasm is not None:
        from nextpnr.fasm_export import write_fasm

        try:
            write_fasm(ctx.fasm_features(), fasm)
        except Exception as e:
            typer.echo(f"Error writing FASM: {e}", err=True)
            raise typer.Exit(1)
        if verbose:
            typer.echo(f"Wrote FASM to {fasm}")

    failures = [] if timing is None else timing_failures(timing)
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

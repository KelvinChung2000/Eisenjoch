from pathlib import Path
from typing import Annotated, Literal

import typer
from loguru import logger


app = typer.Typer(name="eisenjoch", add_completion=False)


@app.command()
def run(
    library: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="Path to the nextpnr bridge library"
        ),
    ],
    device: Annotated[str, typer.Option(help="Himbächel device name")],
    chipdb: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Himbächel chip database")
    ],
    input_json: Annotated[
        Path,
        typer.Option("--json", exists=True, dir_okay=False, help="Input Yosys JSON"),
    ],
    output: Annotated[
        Path, typer.Option("--write", dir_okay=False, help="Output nextpnr JSON")
    ],
    placer: Annotated[
        Literal["heap", "sa"] | None, typer.Option(help="Upstream placer")
    ] = None,
    router: Annotated[
        Literal["router1", "router2"] | None, typer.Option(help="Upstream router")
    ] = None,
    seed: Annotated[int, typer.Option(min=0, max=2**64 - 1, help="Random seed")] = 1,
    stop_after: Annotated[
        Literal["pack", "place", "route"], typer.Option(help="Last stage to run")
    ] = "route",
) -> None:
    """Run the upstream nextpnr Himbächel flow."""
    from .nextpnr import run_upstream

    try:
        run_upstream(
            library=library,
            device=device,
            chipdb=chipdb,
            input=input_json,
            output=output,
            placer=placer,
            router=router,
            seed=seed,
            stop_after=stop_after,
        )
    except RuntimeError as exc:
        logger.error("{}", exc)
        raise typer.Exit(code=1) from exc


def cli() -> None:
    app()

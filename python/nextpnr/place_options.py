from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class PlaceOptions:
    """Placer choice and tuning that `Context.place` reads by attribute name.

    `placer` is heap, sa, opt_trans (alias hydraulic) or electro. `seed` applies to all four,
    `congestion_weight` to heap and sa, `max_iters` to opt_trans and electro, the rest to opt_trans.
    """

    placer: str = "heap"
    seed: int = 1
    max_iters: int | None = None
    congestion_weight: float = 1.0
    timing_weight: float = 0.0
    init_strategy: str = "topological"
    num_threads: int = 8
    legalization: str = "pin_access"

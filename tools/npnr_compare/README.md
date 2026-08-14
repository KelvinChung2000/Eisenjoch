# nextpnr baseline comparison harness

A synthetic fabric that **nextpnr and eisenjoch both read**, so ported
algorithms can be checked against the real tool instead of against a
description of it.

## Why a shared database was not free

eisenjoch already parsed himbaechel's binary chipdb, but no single file could be
read by both tools:

- A stock chipdb omits the strings the arch already knows. Ids
  `1..known_id_count` come from the uarch's `constids.inc`, compiled into the
  C++ binary; the file stores only what follows.
- Our loader rejected any such file and demanded one regenerated with
  `known_id_count = 0`.
- C++ nextpnr **aborts** on those files (`idstring.cc:45`, duplicate id
  registration).

So the two requirements were mutually exclusive. `ChipDb::load_with_known_constids`
splices the arch's `constids.inc` in at load time, exactly as C++ does. One
`.bin` now feeds both, and stock vendor chipdbs became readable as a side effect.

## What the comparison actually proves

Both sides read the same fabric **and the same placement**, then compute
wirelength. The reference numbers come from nextpnr calling its *own*
`get_net_metric` -- not a reimplementation of it (see `patches/`).

`MetricType::WIRELENGTH` short-circuits timing weighting in both
implementations, so these are pure integer HPWL. They match **exactly**: 169
nets, total wirelength 203. An off-by-one here is a defect, not noise.

This validates the ported wirelength model, the arch-API shim underneath it, and
the database bridge. It does **not** yet compare placer against placer -- the
nextpnr placer bodies are not ported (see `docs/nextpnr_faithful_port_plan.md`).
When they are, this harness is where they get measured.

## Setup

The pinned reference is upstream YosysHQ nextpnr `main` @ `4d235150`.

```bash
git -C ~/nextpnr worktree add ~/nextpnr-upstream 4d235150
cd ~/nextpnr-upstream
git apply /path/to/eisenjoch/tools/npnr_compare/patches/0001-dump-net-metric.patch
git apply /path/to/eisenjoch/tools/npnr_compare/patches/0002-optional-lutff-pack.patch
git apply /path/to/eisenjoch/tools/npnr_compare/patches/0003-optional-refine.patch
cmake -B build -DARCH=himbaechel -DHIMBAECHEL_UARCH=example \
      -DBUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`0001` adds one option, `--dump-net-metric <file>`, which writes
`<net>\t<wirelength>` per net plus a total. It only reads the design; it changes
no placement or routing behaviour.

`0002` makes the example uarch's LUT4->DFF pairing constraint optional, via
`NPNR_NO_LUTFF_PACK=1`. Default behaviour is unchanged. It exists to answer
"how much of the placement gap is packing?" by removing packing from the
reference — measured answer: about −1.7%, i.e. it slightly *hurts* nextpnr's
wirelength. See `docs/dcd_vs_nextpnr_baseline.md`.

`0003` makes the post-legalisation SA refinement pass skippable, via
`NPNR_NO_REFINE=1`. Default behaviour is unchanged.

It exists because HeAP *always* refines (`placer_heap.cc:402-412`;
`parallelRefine` defaults false, so the pass is `placer1_refine`), which means
every wirelength nextpnr reports is post-refinement and no comparison against
"HeAP's placement" was ever comparing against HeAP's placement. With the patch
you can take the analytical stage's own output and, separately, feed it to a
refiner under test — the input and the target then both come from nextpnr. On
the 20x20 fabric at seed 1, unpacked: HeAP legalises to 2379 and refines to
1825, and post-route Fmax goes 40.62 -> 44.07 MHz.

`0002` is also what makes *routing our own placement* possible at all:
`constrain_cell_pairs` runs in `pack()`, after the JSON frontend has already
bound our injected bels, and fights any placement that does not satisfy
`delta_z=1`. Set `NPNR_NO_LUTFF_PACK=1` on both sides.

## Checking placement legality

`check_slice_legality.py <placed.json>` counts violations of the example uarch's
`slice_valid` rule without needing nextpnr. Validated against the tool: on a
placement nextpnr rejected with 248 `post-placement validity check failed`
warnings it reports 124 illegal slices, exactly 248/2.

Validated in both directions: it also reports 0 illegal for the 10 legality-aware
placements nextpnr went on to route successfully, so it is not simply
pessimistic.

Useful because a failed legality check aborts nextpnr *before* routing, and the
warning list is long and unsorted; this gives the count directly, and works on
any placed JSON including ones our own placer wrote.

## Regenerating

```bash
./regen.sh                        # defaults: 12x12 fabric, W=16, seed 1, heap
SIZE=24 WIDTH=32 SEED=7 ./regen.sh
```

Set `NPNR_UPSTREAM` if the checkout is elsewhere. Outputs land in `out/` and the
four files the Rust test needs are copied to
`crates/nextpnr/tests/fixtures/npnr_baseline/`.

Then:

```bash
cargo test --features test-utils --test npnr_baseline_compare
```

## Fabric and design notes

The fabric is upstream's `example` uarch: `gen_shared_chipdb.py` imports
`example_arch_gen.py` and overrides only the grid size. It is not a copy --
the C++ uarch has its bel types, wire names and constids compiled in, so a
lookalike would not load.

Two things about the design are forced by that fabric, not by preference:

- **No constant-driven top-level outputs.** The uarch rejects them, which is why
  the `blinky.v` shipped with nextpnr cannot be used here.
- **The clock buffer is pinned to `X1Y0/IO0`.** The example fabric's clock ladder
  is fed by a single `GCLK_OUT` pip that exists only in that tile; without the
  constraint, routing fails.

IO buffers are inserted by `iopadmap` because the uarch trims nextpnr's own IOBs
and assumes synthesis did it.

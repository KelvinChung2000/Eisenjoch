# Faithful nextpnr Placer/Router Port

Goal: a 1-to-1 faithful Rust port of every nextpnr placement and routing
algorithm, so that eisenjoch's own algorithms (`opt_trans`, `raster`) can be
benchmarked against the real nextpnr baseline rather than against a sketch.

## Reference

**Upstream YosysHQ nextpnr `main` @ `4d235150`** ("router2: Fix reservation
around pre-routed nets").

Not `~/nextpnr` HEAD (`11aa1bd4`). That checkout is Kelvin's fork and carries
+975 lines of local HeAP work on top of merge-base `226a2dfd` ("New heap
placer heuristic", "Heap packing update", "new heap init placement"). Those
local changes *are* "my own version of heap" — the thing being replaced. The
comparison baseline must be the published tool.

Every ported module header records this hash. Re-targeting to the fork means
re-porting `placer_heap.cc` and `router1.cc` only.

## Faithfulness rules

1. **Port nextpnr's numerics, not ours.** HeAP gets nextpnr's own
   `EquationSystem` + Jacobi-preconditioned CG, living inside the placer
   module. `placer_static` gets nextpnr's own FFT from `static_util.h`.
   Wiring these into eisenjoch's `solver/` (CG + AMG/IC0/spectral) would turn
   the benchmark into solver-vs-solver instead of placer-vs-placer.
2. **Keep nextpnr's names.** Functions, fields, constants and config defaults
   keep their C++ spelling (snake_cased) so the two can be diffed side by side.
3. **Port the RNG exactly.** nextpnr is xorshift64*star*: state seeded to
   `0x3141592653589793`, `rng64()` returns `state * 0x2545F4914F6CDD1D` before
   advancing with shifts 12/25/27, `rngseed` burns 5 draws, `rng(n)` rejection
   samples against the next power of two. eisenjoch's current
   `context/rng.rs` is plain xorshift64 (shifts 13/7/17, modulo bias) — a
   completely different stream. Seeds are meaningless across the two.
4. **Do not touch Kelvin's own work.** `placer/opt_trans/`, `router/raster.rs`,
   `router/astar.rs`, `router/lookahead.rs` are the subject of the comparison,
   not the baseline. Leave them compiling and passing their tests. This is a
   behavioural no-touch on the algorithms, not a promise of zero textual churn
   — see the RNG note below.
5. **The ported routers are self-contained.** They do not reuse
   `router/common.rs` (`NegotiationState`, 1143 lines): that is the substrate
   `raster` sits on, i.e. Kelvin's work. nextpnr's router1 and router2 each
   carry their own bookkeeping anyway, so mirroring their file structure is
   both more faithful and keeps the baseline free of shared plumbing.
6. **Carry the ISC header.** Each ported module reproduces the copyright
   notice of the nextpnr file it came from. The licence requires it, and it
   doubles as the per-file reference pin.

### RNG blast radius

There is one `Context` RNG, so making it faithful changes the random stream
for *every* existing consumer: `opt_trans`, the packer, initial placement.
That is unavoidable and correct, but it means results for Kelvin's own
algorithms will shift after the RNG commit for reasons that have nothing to do
with those algorithms. The port exposes nextpnr's API (`rng`, `rng(n)`, `rngf`,
`shuffle`, `sorted_shuffle`) and keeps thin aliases for the existing
`next_u32`/`next_range`/`next_u64` call sites, so no caller needs rewriting.

## Verifying faithfulness

"Faithful" has to be falsifiable, so it is defined per module:

- **RNG and FFT — exact golden traces.** Small C++ harnesses compiled against
  upstream `deterministic_rng.h` and `static_util.h` dump reference outputs
  (first 1000 `rng64()` draws, `rng(n)` for assorted n, shuffles of 0..32 from
  several seeds; FFT input/output vectors). Committed as fixtures, asserted
  byte-for-byte by the Rust tests. This is cheap and catches the most
  insidious drift class.
- **Placers and routers — structural, not trace-level.** Exact trace matching
  is not available: eisenjoch's chipdb is not an upstream nextpnr arch, so the
  two cannot be run on the same device. The claim these modules make is
  narrower and auditable by side-by-side diff: same control flow, same
  constants and config defaults, same order of RNG consumption, same cost
  functions. Where a port deviates because the substrate forced it, the
  module says so in a comment.

## Scope

Line counts are upstream C++.

### Phase 1 — foundations (shared; blocks everything else)

| nextpnr | lines | destination |
|---|---|---|
| `common/kernel/deterministic_rng.h` | 105 | rewrite `context/rng.rs` |
| `common/place/place_common.{cc,h}` | 553 | new `placer/place_common.rs` |
| `common/place/fast_bels.h` | 188 | new `placer/fast_bels.rs` |

Plus the arch-API hooks the placers/routers call that `Context` lacks today:
`is_bel_location_valid`, `route_bounding_box`, `predict_delay`,
`cluster_bounds`, `wire_bel_pins`, `uphill_bel_pin`.

And the `TimingAnalyser` surface router2 and placer1 need. `port_criticality`
already matches nextpnr's `get_criticality(CellPortKey)`; missing are
`set_route_delay`, and the `setup_only` / `with_clock_skew` mode flags.

`placer/common.rs` (2187 lines) is *not* a port of `place_common.cc` — it is
eisenjoch's own `TypeAwarePlacement` / `MuxSlotTracker` / `CellValidityMask`.
It stays where it is; `opt_trans` depends on it.

### Phase 2 — the five algorithms (independent once Phase 1 lands)

| nextpnr | lines | replaces | current |
|---|---|---|---|
| `placer1.cc` | 1289 | `placer/sa/` | 511 |
| `placer_heap.cc` | 2191 | `placer/heap/` | 809 |
| `placer_static.cc` + `static_util.h` | 1888 | `placer/electro_place/` | 620 |
| `router1.cc` | 1495 | `router/maze.rs` | 671 |
| `router2.cc` | 1868 | `router/router2.rs` | 315 |

Every one of these is currently a sketch, not just heap and electrostatic.

### Phase 3 — secondary placement passes

`detail_place_core.cc` (463), `parallel_refine.cc` (546), `timing_opt.cc` (561).

### Phase 4 — integration

Wire into the existing `Placer`/`Router` traits and `placer/pipeline.rs`,
keeping the trait signatures stable so the benchmark drivers keep running.
Replace the tests that covered the discarded sketches. Add a comparison
harness that runs baseline vs `opt_trans`/`raster` on the same netlists.

## Total

~11,300 lines of C++ across 13 files.

## Progress

Branch `npnr-faithful-port`.

- [x] **Reference pinned** — upstream `4d235150`, with the fork's local HeAP
      work explicitly excluded. (`68441d8`)
- [x] **RNG** — `deterministic_rng.h` → `context/rng.rs`, byte-exact against a
      golden trace from the real C++. 12 tests. (`c2d3362`)
- [x] **Arch-API shim** — `context/arch_api.rs`, 15 nextpnr-named calls the
      ports need. **FastBels** — `placer/fast_bels.rs`. (`5a0e106`)
- [x] **place_common** — wirelength model, constraint legaliser,
      `IncreasingDiameterSearch` golden-tested. 6 tests. (`97fdc8c`)
- [~] `placer_heap.cc` → `placer/heap/` — **`EquationSystem` + its
      Jacobi-preconditioned CG done and verified against real Eigen** (`f00c15e`,
      8 tests). The placer body (~2,000 lines) is not started.
- [ ] `placer_static.cc` + `static_util.h` → `placer/static/`
- [ ] `router2.cc` → `router/router2/`
- [ ] `router1.cc` → `router/router1/`
- [ ] `placer1.cc` → `placer/placer1/`
- [ ] Phase 3: `detail_place_core`, `parallel_refine`, `timing_opt`
- [ ] Phase 4: integration, trait wiring, comparison harness

Phase 1 is done: 778 tests pass across 26 targets. Two failures in
`tests/context` (`estimate_delay_adjacent`, `estimate_delay_diagonal`) are
pre-existing, verified on the parent commit, and unrelated to this work.
`tests/router2_tests` does not compile, also pre-existing (`Router2Cfg` field
drift) — it will be rewritten with the router2 port anyway.

The five Phase 2 modules are independent of each other now that Phase 1 has
landed, so they can be worked in parallel worktrees if wanted.

### Notes for whoever picks up heap

- `placer/heap/` is currently **hybrid**: `equation_system.rs` is a faithful
  port, every other file is the old sketch. The commit that ports the placer
  body must delete the sketch *and* its `heap_tests` / `heap_internal_tests`
  (31 tests) in the same change — those test the sketch, not the port.
- `Context::bels_by_tile` allocates a fresh `Vec` per call, and
  `bel_by_location` sits on top of it. nextpnr's equivalent is array-indexed.
  HeAP's strict legalisation pass hammers both in inner loops, so this wants a
  cached or iterator-based form before the heap body lands — it is a
  performance issue only, not a correctness one.
- The golden-trace recipe is the thing to reuse: write a small C++ harness that
  embeds the nextpnr code verbatim, dump reference values as hex floats, commit
  harness and output together as fixtures. Three components are pinned this way
  so far (RNG, `IncreasingDiameterSearch`, `EquationSystem`), and it has already
  caught behaviour a plausible-looking reimplementation would have lost.

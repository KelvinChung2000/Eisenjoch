# Legalizer ring-query: replace the materialized shortlist with a lazy nearest-BEL walk

Status: proposed
Branch: `npnr-faithful-port`
Supersedes the `CAND_SHORTLIST` + widen mechanism introduced in `c314843`.

## Problem

`sorted_legalize` picks, for each cell, the nearest legal BEL of its type. It
does that by *materializing* a distance-sorted candidate list per cell.

`c314843` bounded that list to `CAND_SHORTLIST = 1024` entries, which stopped
FPGA01 from OOMing. It did not make the approach cheap, because the cap bounds
work but not legality: when all 1024 nearest BELs are rejected, Phase B falls
back to `candidate_list(.., limit: None)`, which scans and fully sorts *every*
BEL of that type.

Measured on FPGA01 (`measurements/mem_probe/fpga01_legalize_fix.log`):

| Quantity | Value |
| --- | --- |
| movable cells | 76,660 |
| bucket types | 2 -- DFF (1,077,536 BELs), LUT6 (538,768) |
| grid | 311 x 223 = 69,353 tiles |
| legalization span | t=353 s -> 1520 s, 1167 s |
| `widened_rescans` | 27,467 |
| `shared_mux_rejects` | 21,841,944 |
| `arch_validity_rejects` | 0 |
| peak RSS | 4217 MB, of which shortlists are `76660 * 1024 * 8 B` = 599 MiB |

The fact that drives this design: **~25% of cells exhaust the shortlist, and
each exhaustion re-scans and re-sorts up to 1.08M entries.**

The Phase A / Phase B split is *not* separable from the RSS trace, and this
design does not need it. Legalization RSS enters a sawtooth between ~3.0 and
~3.4 GB within a second of the start and holds it until the end -- the
signature of repeated ~1M-element allocate-and-free, which is what the widen
rescan does and what nothing else in the phase does. Attributing the exact
seconds would not change any decision below, because the design removes the
re-sorts either way. It is therefore left unmeasured rather than guessed.

`k` is a bad knob in both directions: raising it to 4096 costs ~2.3 GiB of
shortlist and still cannot guarantee no rescan; lowering it makes rescans more
frequent.

## Goals

- Remove the per-cell materialized candidate list, and with it the 599 MiB and
  the 27,467 full re-sorts.
- Keep the legalizer's *output* identical. This is a data-structure change, not
  a heuristic change.
- Keep `try_bind_cell` untouched. Its shared-mux / cluster-footprint /
  arch-validity logic is what made `fc82eb8` un-cherry-pickable; this design
  does not reopen it.

## Non-goals

- Reducing `shared_mux_rejects`. 21.8M rejections is a separate question about
  Phase B's ordering and packing; it is out of scope here.
- Beating the sister branch's 6m16s. Its Phase B lacks this branch's rejection
  logic, so the numbers are not comparable. The honest target is "the full
  re-sorts are gone and ~600 MiB comes off peak RSS"; whatever wall-clock lands,
  we report.
- Changing how cells are ordered (`order`, outer-cells-first) or how
  displacement is accumulated.

## Approaches considered

### A. Legalizer-owned ring index built from `bel_data_cache` -- RECOMMENDED

Build a per-type `[x][y] -> Vec<(BelId, enum_index)>` grid directly from the
`bel_data_cache` the legalizer already builds, then walk outward from the
target and stop at the first BEL `try_bind_cell` accepts.

Building from `bel_data_cache` means the candidate universe is identical *by
construction* -- same alias resolution, same filtering, same enumeration order.
It is roughly 40 lines, and it sidesteps every trap in option B.

### B. Reuse `FastBels` proper

`placer/fast_bels.rs` already implements `[type][x][y] -> Vec<BelId>` and is
used by the SA placer (`place_common.rs:615,629`). Reusing it means adding a
ring-query method to it.

Rejected, for one structural reason and three traps:

- **It is a faithful port** of upstream `common/place/fast_bels.h` @ `4d235150`.
  Adding a query method that upstream does not have works against the reason
  this branch exists.
- *Alias blindness.* `bels_for_cell_type` filters on
  `Bel::is_valid_for_cell_type`, which compares `bel_type == cell_type_str`
  with **no alias resolution** (`context/views/hardware.rs:93`). The legalizer
  reaches BELs through `ctx.bels_for_bucket`, which **does** resolve aliases
  (`context/buckets.rs:25`). `bels_for_bel_bucket` is alias-blind too. Any
  registered alias would silently change the candidate set. Fixable by passing
  `ctx.resolve_bucket(cell_type)`, but it is a live foot-gun. (Nothing in-tree
  registers an alias today -- only the Python binding exposes it.)
- *Grid collapse.* `min_bels_for_grid_pick` collapses a rare type's whole grid
  into cell `(0,0)`, which would make ring expansion meaningless. Must be
  constructed `new(false, 0)` so `n < 0` is never true and the collapse never
  fires.
- *Staleness.* `check_bel_available: true` filters bound BELs at construction
  time, but Phase B binds as it goes. Must be `false`, with availability
  checked at query time (`try_bind_cell` already does this).

Worth revisiting only if a second consumer wants the same query.

### C. Keep the shortlist, raise `k` / make the rescan incremental

Rejected. Raising `k` trades memory for a smaller rescan rate without removing
the rescan. Making the rescan incremental (remember where the shortlist ended,
resume from there) still needs a sorted tail, which is the full sort again.

## Design

### Ordering contract

The whole design rests on one invariant:

> The ring iterator emits candidates in exactly the order `candidate_list`
> would, i.e. ascending `(dist_sq, enum_index)`.

`dist_sq` is computed with the *same f64 expression on the same inputs* as
today, so equality cases are bitwise identical:

```rust
let dx = bx as f64 - info.target_x;
let dy = by as f64 - info.target_y;
dx * dx + dy * dy
```

`enum_index` is the BEL's position in `bel_data_cache[type]`, which is its
position in `ctx.bels_for_bucket(..)`, which is chipdb enumeration order. It is
stored explicitly in the grid as a `u32` (+4 B per BEL, ~6.5 MB for FPGA01).

Storing it is deliberate. `chipdb().bels()` iterates tiles ascending and index
ascending (`chipdb/access.rs:228`), and `BelId` packs tile in the high 32 bits
and index in the low 32 (`chipdb/ids.rs:98`), so ascending enum index *is*
ascending raw `BelId` and the tie-break could be free. That equivalence is a
property of two files that have no reason to stay coupled. 6.5 MB buys not
depending on it.

Ordering is made total without a float-comparator by keying the heap on
`(dist_sq.to_bits(), enum_index)`. For non-negative finite f64 the IEEE-754 bit
pattern is monotonic in value, and `dx*dx + dy*dy` on finite inputs is always
non-negative and never NaN.

The `BelId` itself is deliberately **not** part of the heap key. `BelId` derives
only `Clone, Copy, PartialEq, Eq, Hash, Default` (`chipdb/ids.rs:30`) -- it has
no `Ord`, so a `(u64, u32, BelId)` key would not compile. It does not need one:
`enum_index` is unique, so `(dist_sq_bits, enum_index)` is already a total
order, and the `BelId` is recovered by indexing `bel_data_cache[type]`.

### Ring geometry and the emit guard

Rings are Chebyshev (square) around the target *rounded to a tile*:

```
cx = round(target_x),  cy = round(target_y)
ring r = { (x, y) : max(|x - cx|, |y - cy|) == r }
```

The metric is Euclidean but the rings are square, so a BEL found in ring `r` is
**not** necessarily nearer than one still unexplored in ring `r+1`. The guard:

> After exploring rings `0..=R`, any unexplored BEL has Chebyshev distance
> `>= R+1` from `(cx, cy)`. Since `|cx - target_x| <= 0.5`, its Euclidean
> distance from the target is `>= R + 0.5`.

So a buffered candidate may be emitted only while `dist_sq < (R + 0.5)^2`. On
exact equality we expand one more ring first: an unexplored BEL could tie on
distance and win on the lower `enum_index`.

This is the one place the design can silently go wrong, so it gets its own
test (see Testing).

### The iterator

```rust
/// Nearest-BEL walk for one cell, in `(dist_sq, enum_index)` order.
struct RingCandidates<'a> {
    grid: &'a TypeGrid,          // ragged [x][y] -> &[(BelId, u32)]
    bels: &'a [(BelId, i32, i32, i32)],  // bel_data_cache[type], indexed by enum_index
    target_x: f64, target_y: f64,
    cx: i32, cy: i32,
    radius: i32,                 // rings 0..=radius explored
    exhausted: bool,             // radius covers the whole grid
    heap: BinaryHeap<Reverse<(u64, u32)>>,  // (dist_sq bits, enum_index)
}
```

`next()` expands rings until the heap's minimum satisfies the emit guard, or
the grid is exhausted. Heap residency is bounded by the BELs within the current
radius -- transient, per-cell, and freed when the cell binds. A pathological
cell can still buffer its whole type, but one cell at a time, which is exactly
the `O(cells x bels)` -> `O(bels)` change this exists to make.

### Phase B integration

`try_bind_cell` changes signature from `candidates: &[BelId]` to
`candidates: impl Iterator<Item = BelId>`; its body changes only
`for &bel_id in candidates` to `for bel_id in candidates`. Nothing else in it
moves.

Deleted outright:

- `CAND_SHORTLIST`
- the Phase A `par_iter().map(candidate_list(.., Some(k)))` block and
  `sorted_candidates`
- the widen block and the `widened_rescans` counter

Because the iterator is exhaustive by construction, "try the shortlist, then
try the remainder" collapses to a single call.

`bel_data_cache` and `bel_by_loc` stay. `bel_by_loc` needs `z` for cluster-child
slot probing, and the grid is built from the cache. At `(BelId, i32, i32, i32)`
the cache is ~38 MB -- it is the per-cell materialization being deleted, not the
type-level cache.

Phase A's rayon parallelism disappears for the common path. What replaces it is
a single O(total BELs) grid build, ~1.6M pushes, which is trivially
parallelizable per type if it ever shows up in a profile.

### Region-constrained cells

Keep the existing materialized path (`candidate_list` with the region filter)
for cells with `cell_region: Some(_)`. Ring-walking a region-constrained cell
whose region sits far from its target would scan outward across the whole chip
before finding anything, and bounding rings by the region bbox is a second
geometry problem.

FPGA01 has no region-constrained cells (`CellValidityMask` reports
`min=avg=max=67346` valid positions per cell), so this path is not on the
measured hot path. Bbox-bounded rings are a documented follow-up, not scope.

Note that `ctx.bels_for_bucket_in_region` (`context/buckets.rs:62`) already
caches region-filtered BEL lists and duplicates what `region_bel_sets` builds
by hand. Consolidating them is also follow-up.

### Empty-candidate detection

`shortlist.is_empty() -> NoBelsAvailable` no longer exists. Replace with an
upfront per-cell check that the type's grid (or, for region cells, the filtered
list) is non-empty, raising the same `PlacerError::NoBelsAvailable`.

## Testing

**1. Oracle equivalence (the load-bearing test).** `candidate_list` stays live
for the region path and doubles as the oracle. Parameterized over grid shapes,
BELs-per-tile, and targets (including fractional and out-of-bounds ones):

```
assert!(RingCandidates::new(..).collect::<Vec<_>>() == candidate_list(.., None));
```

Full-sequence equality, not just the first element. This makes the refactor
behaviour-preserving *at the query level*, which is what turns the FPGA01 run
into a performance measurement rather than a judgement call.

**2. Emit-guard regression.** A grid with a BEL at Chebyshev radius `r` that is
Euclidean-farther than one at radius `r+1`, plus a fractional target, plus an
exact-tie case where the lower `enum_index` sits in the *outer* ring. Each must
fail if the `(R + 0.5)^2` guard is weakened to `R^2` or the equality case stops
expanding.

**3. Existing suite.** `cargo test -p nextpnr` stays green (132 lib tests).
`shortlist_is_a_prefix_of_the_full_list` and
`shortlist_allocation_is_bounded_by_the_cap` are deleted along with the
mechanism they cover.

**4. FPGA01 run** under `systemd-run --user --scope -p MemoryMax=12G
-p MemorySwapMax=0 --collect`, `NPNR_OT_MAX_ITERS=1`. Report peak RSS and
legalization wall time against 4217 MB / 1167 s.

Expected: ~600 MiB off peak (shortlists gone, ~20 MB grid added). Wall time
should fall by whatever the 27,467 full sorts were costing -- we are not
predicting a figure.

Post-legalization HPWL will *not* be bit-identical to 14234563, because nothing
on this branch is reproducible (`e3efbf3`, the Jacobi determinism fix, is not
merged here) and sv3 spans ~0.6% run-to-run at fixed seed. The solver input
positions differ between runs, so the legalizer's input differs. Judge against
that spread. Test 1 is what actually certifies the legalizer, precisely because
the live run cannot.

## Risks

| Risk | Mitigation |
| --- | --- |
| Emit guard subtly wrong -> different BEL chosen | Test 2, plus full-sequence oracle equality in test 1 |
| A pathological cell buffers its whole type | Bounded by one cell at a time; same worst case as today's rescan, without the sort |
| Losing Phase A parallelism | replaced by one linear grid build; if it profiles hot, parallelize per type |
| Region path left on the old mechanism | Explicit, tested by the same oracle, and off FPGA01's hot path |

## Follow-ups (not scope)

- Bbox-bounded rings so region cells use the walk too.
- Consolidate `region_bel_sets` with `ctx.bels_for_bucket_in_region`.
- Investigate the 21.8M `shared_mux_rejects` -- the remaining Phase B cost once
  the re-sorts are gone.
- Merge `e3efbf3` (Jacobi determinism) so live runs can be compared bit-exactly.

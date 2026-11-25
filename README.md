<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/logo-dark.svg">
    <img alt="Eisenjoch" src="docs/logo/logo-light.svg" width="400">
  </picture>
</h1>

Use `eisenjoch` to run upstream nextpnr packing, placement, and routing. It loads a
native nextpnr shared library in the Rust process. The first supported device is
the Himbächel `EXAMPLE` device.

Use `nextpnr-himbaechel-rust` for Eisenjoch's Rust algorithms. In Python, use
`nextpnr.RustContext` for this flow. The existing `nextpnr.Context` name also works.
In Cargo, select the Rust implementation with `-p nextpnr-rust`.

## Build upstream nextpnr

From the Eisenjoch repository root, check out the `third_party/nextpnr`
submodule at the revision recorded by Git. The bridge uses the Himbächel core
from upstream's CMake build. See [nextpnr build prerequisites](https://github.com/YosysHQ/nextpnr#prerequisites).

```sh
git submodule update --init --checkout --recursive -- third_party/nextpnr
```

`.gitmodules` sets `update = none` for this submodule, so a plain
`git submodule update` and a Cargo git dependency on `nextpnr-db` skip it.
`--checkout` overrides that setting.

```sh
cmake -S third_party/nextpnr -B build/nextpnr -DARCH=himbaechel \
  -DHIMBAECHEL_UARCH=example -DBUILD_PYTHON=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_PROJECT_INCLUDE="$PWD/crates/nextpnr-upstream/native/nextpnr.cmake"
cmake --build build/nextpnr --target eisenjoch-nextpnr
```

## Run a design

Use a Yosys JSON netlist synthesized for the example device. Pass a chip database
for that device and a bridge library built from the revision above.

```sh
uv run --extra cli eisenjoch --library build/nextpnr/libeisenjoch-nextpnr.so \
  --device EXAMPLE --chipdb crates/nextpnr/tests/fixtures/npnr_baseline/shared_chipdb.bin \
  --json design.json --write routed.json --placer heap --router router2
```

Omit `--placer` and `--router` to use the upstream defaults. Use `--stop-after pack`
or `--stop-after place` to write an intermediate netlist. From Rust, call
`nextpnr_upstream::Nextpnr::load` and `Nextpnr::run`. The Python entry point is
`nextpnr.run_upstream`.

## Compare against the upstream executable

```sh
cmake --build build/nextpnr --target nextpnr-himbaechel
NEXTPNR_BUILD="$PWD/build/nextpnr" cargo test -p nextpnr-upstream --test himbaechel -- --ignored
```

## Update the nextpnr revision

Fetch upstream and check out the new revision in the submodule. Rebuild the
bridge and the upstream executable, then run the parity test above. Stage
`third_party/nextpnr` to record the new revision in Eisenjoch.

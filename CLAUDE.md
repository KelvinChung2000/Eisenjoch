# Eisenjoch - FPGA Place-and-Route

Two place-and-route flows behind one Python package. `eisenjoch` runs upstream
nextpnr's own packing, placement and routing through a native bridge that loads
the Himbächel core as a shared library. `nextpnr-himbaechel-rust` runs
Eisenjoch's own Rust algorithms.

## Build and Run

### Python Environment

Use UV for all Python dependency management. `uv run` manages `.venv/` at the
repository root and creates it on first use, so a fresh worktree has none until
then.

Always run Python through UV:
```
uv run python <script>
```
Never invoke the system Python directly (e.g., `/usr/bin/python3.12`).

The project requires Python >= 3.14. PyO3 below 0.26 refuses to build against
anything newer than 3.13; the workspace pins 0.29.

### Building the Rust Extension

`uv run` builds the extension through maturin and rebuilds it whenever the
Rust sources change, so no manual step is normally needed:
```
uv run python <script>
```

To build the extension on its own, point PyO3 at the venv interpreter:
```
PYO3_PYTHON="$PWD/.venv/bin/python3" cargo build --release -p npnr-python
```
maturin installs the result as `python/nextpnr/nextpnr.cpython-<tag>.so`.
Extensions built for an older interpreter are simply ignored: CPython only
loads the `.so` whose tag matches the running interpreter.

### Building the upstream bridge

The `eisenjoch` flow needs a bridge library built from the pinned
`third_party/nextpnr` submodule. README.md carries the cmake invocation and the
parity test against upstream's own executable.

### Running Benchmarks

Run benchmarks with UV, not bare Python:
```
uv run python python/nextpnr/benchmarks/<benchmark_script>.py
```

## Project Structure

- `crates/nextpnr/` - Eisenjoch's own place-and-route library (Rust)
- `crates/nextpnr-upstream/` - native bridge to upstream nextpnr's Himbächel core
- `crates/npnr-python/` - PyO3 Python bindings for both flows
- `python/nextpnr/` - Python package (SDC parsing, benchmarks, CLIs)
- `third_party/nextpnr` - submodule pinning the upstream revision

## Testing

The Rust suite is the real one:
```
cargo test --workspace --release --no-fail-fast
```

The `nextpnr-upstream` parity test is `#[ignore]`d and needs a cmake build of
upstream. README.md gives the command.

Use `pytest` for Python tests and `pytest-mock` for mocking. Run via:
```
uv run pytest
```
There are currently no Python unit tests. The `test_*.py` files under
`python/nextpnr/benchmarks/` and `chip_database/test/` are benchmark drivers,
not tests: they load a chipdb and run place-and-route at import time, so
pytest is configured not to collect them.

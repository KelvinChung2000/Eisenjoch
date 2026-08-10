# Eisenjoch - FPGA Place-and-Route

Rust reimplementation of nextpnr FPGA place-and-route with Python bindings via PyO3.

## Build and Run

### Python Environment

Use UV for all Python dependency management. The virtual environment lives at `/home/kelvin/side-project/eisenjoch/.venv/`.

Always run Python through UV:
```
uv run python <script>
```
Never invoke the system Python directly (e.g., `/usr/bin/python3.12`).

The project requires Python >= 3.14, which needs PyO3 >= 0.26 (0.24 refuses
to build against anything newer than 3.13).

### Building the Rust Extension

`uv run` builds the extension through maturin and rebuilds it whenever the
Rust sources change, so no manual step is normally needed:
```
uv run python <script>
```

To build the extension on its own, point PyO3 at the venv interpreter:
```
PYO3_PYTHON=/home/kelvin/side-project/eisenjoch/.venv/bin/python3 cargo build --release -p npnr-python
```
maturin installs the result as `python/nextpnr/nextpnr.cpython-<tag>.so`.
Extensions built for an older interpreter are simply ignored: CPython only
loads the `.so` whose tag matches the running interpreter.

### Running Benchmarks

Run benchmarks with UV, not bare Python:
```
uv run python python/nextpnr/benchmarks/<benchmark_script>.py
```

## Project Structure

- `crates/nextpnr/` - Core place-and-route library (Rust)
- `crates/npnr-python/` - PyO3 Python bindings
- `python/nextpnr/` - Python package (SDC parsing, benchmarks, CLI)

## Testing

The Rust suite is the real one:
```
cargo test --workspace --release --no-fail-fast
```

Use `pytest` for Python tests and `pytest-mock` for mocking. Run via:
```
uv run pytest
```
There are currently no Python unit tests. The `test_*.py` files under
`python/nextpnr/benchmarks/` and `chip_database/test/` are benchmark drivers,
not tests: they load a chipdb and run place-and-route at import time, so
pytest is configured not to collect them.

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

### Building the Rust Extension

Set `PYO3_PYTHON` to the venv interpreter before building:
```
PYO3_PYTHON=/home/kelvin/side-project/eisenjoch/.venv/bin/python3 cargo build --release
```

After `cargo build`, manually copy the shared object into the Python package:
```
cp target/release/libnextpnr.so python/nextpnr/nextpnr.cpython-313-x86_64-linux-gnu.so
```

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

Use `pytest` for Python tests and `pytest-mock` for mocking. Run via:
```
uv run pytest
```

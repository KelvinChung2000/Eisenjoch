# ISPD FPGA Contest Benchmarks

This directory is the local staging area for the ISPD 2016 and 2017 FPGA placement contest benchmarks.

The repository tracks the tooling and documentation, but not the large benchmark datasets themselves. Raw archives, extracted benchmark trees, generated manifests, and the catalog index are all intentionally gitignored.

## Layout

- `raw/2016/`, `raw/2017/`: downloaded contest archives
- `extracted/2016/<benchmark>/`, `extracted/2017/<benchmark>/`: unpacked official benchmark contents
- `manifests/2016/<benchmark>.json`, `manifests/2017/<benchmark>.json`: normalized metadata produced by the importer
- `index.json`: catalog of discovered local benchmarks

## Workflow

List the official datasets:

```bash
nextpnr-ispd-benchmarks list
```

Download the official archives into `benchmark/ispd/raw`:

```bash
nextpnr-ispd-benchmarks fetch --year 2016 --all
nextpnr-ispd-benchmarks fetch --year 2017 --all
```

Extract the archives and generate manifests:

```bash
nextpnr-ispd-benchmarks import --year 2016 --all
nextpnr-ispd-benchmarks import --year 2017 --all
```

Validate the local benchmark trees:

```bash
nextpnr-ispd-benchmarks validate --year 2016 --all
nextpnr-ispd-benchmarks validate --year 2017 --all
```

Convert one benchmark into the repo's existing Yosys-JSON-based flow, then run
it on the current synthetic architecture:

```bash
uv run python -m nextpnr.benchmarks.run_ispd_benchmarks --year 2016 --benchmark FPGA01
```

Emit only the converted JSON and metadata without running P&R:

```bash
uv run python -m nextpnr.benchmarks.run_ispd_benchmarks --year 2017 --benchmark clk_design1 --convert-only
```

## Current Boundary

This setup now includes an approximate compatibility runner that converts the
Bookshelf benchmarks into Yosys JSON and runs them on the repo's synthetic
architecture. It preserves net connectivity and fixed-terminal intent, but it
does not reproduce the original contest device geometry, BEL numbering, or 2017
clock-region legality model.

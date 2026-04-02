"""Manage local ISPD 2016/2017 FPGA placement contest benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = REPO_ROOT / "benchmark" / "ispd"

REQUIRED_FILE_TYPES = ("aux", "nodes", "nets", "pl", "scl", "wts", "shapes", "lib")
OPTIONAL_FILE_NAMES = ("README", "README.txt", "readme.txt", "design.dcp")
CLOCK_METADATA_HINTS = ("clock", "half-column", "region", "bufg", "bram")

OFFICIAL_BENCHMARKS = {
    "2016": {
        f"FPGA{i:02d}": f"https://www.ispd.cc/contests/17/downloads/2016/FPGA{i:02d}.tar.gz"
        for i in range(1, 13)
    },
    "2017": {
        "clk_design1": "https://www.ispd.cc/contests/17/clk_design1.tar.gz",
        "clk_design2": "https://www.ispd.cc/contests/17/clk_design2.tar.gz",
        "clk_design3": "https://www.ispd.cc/contests/17/clk_design3.tar.gz",
        "clk_design4": "https://www.ispd.cc/contests/17/clk_design4.tar.gz",
        "clk_design5": "https://www.ispd.cc/contests/17/clk_design5.tar.gz",
    },
}


@dataclass(frozen=True)
class Roots:
    base: Path
    raw: Path
    extracted: Path
    manifests: Path
    index: Path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = get_roots(Path(args.root).resolve())
    command = args.command

    if command == "list":
        return cmd_list(args, roots)
    if command == "fetch":
        return cmd_fetch(args, roots)
    if command == "import":
        return cmd_import(args, roots)
    if command == "validate":
        return cmd_validate(args, roots)

    parser.error(f"unknown command: {command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage local ISPD 2016/2017 FPGA contest benchmark datasets."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Benchmark root directory (default: {DEFAULT_ROOT})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List official and local benchmark status")

    fetch = subparsers.add_parser("fetch", help="Download official benchmark archives")
    add_dataset_selector(fetch)
    fetch.add_argument("--force", action="store_true", help="Re-download existing archives")

    imp = subparsers.add_parser(
        "import", help="Extract archives and generate normalized manifests"
    )
    add_dataset_selector(imp)
    imp.add_argument(
        "--skip-extract",
        action="store_true",
        help="Generate manifests from already extracted directories only",
    )
    imp.add_argument("--force", action="store_true", help="Re-extract and regenerate manifests")

    validate = subparsers.add_parser(
        "validate", help="Validate extracted benchmark trees and manifests"
    )
    add_dataset_selector(validate)
    validate.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as validation failures",
    )
    return parser


def add_dataset_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--year",
        choices=sorted(OFFICIAL_BENCHMARKS),
        required=True,
        help="Contest year to operate on",
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark names (default: all official benchmarks for the year)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Operate on all official benchmarks for the selected year",
    )


def get_roots(base: Path) -> Roots:
    return Roots(
        base=base,
        raw=base / "raw",
        extracted=base / "extracted",
        manifests=base / "manifests",
        index=base / "index.json",
    )


def cmd_list(_: argparse.Namespace, roots: Roots) -> int:
    ensure_layout(roots)
    catalog = load_catalog(roots.base)
    catalog_map = {
        (entry["year"], entry["name"]): entry for entry in catalog.get("benchmarks", [])
    }
    for year in sorted(OFFICIAL_BENCHMARKS):
        print(year)
        for name, url in OFFICIAL_BENCHMARKS[year].items():
            archive = archive_path(roots, year, name)
            extracted = extracted_path(roots, year, name)
            entry = catalog_map.get((year, name), {})
            status = []
            status.append("raw" if archive.exists() else "missing-raw")
            status.append("extracted" if extracted.exists() else "missing-extracted")
            manifest_ok = bool(entry.get("manifest_path")) and not entry.get("errors")
            status.append("manifest-ok" if manifest_ok else "manifest-missing")
            print(f"  {name:<12} {' '.join(status):<40} {url}")
    return 0


def cmd_fetch(args: argparse.Namespace, roots: Roots) -> int:
    ensure_layout(roots)
    selected = selected_benchmarks(args.year, args.benchmarks, args.all)
    failures = 0
    for name in selected:
        url = OFFICIAL_BENCHMARKS[args.year][name]
        dst = archive_path(roots, args.year, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.force:
            print(f"skip {name}: archive already exists at {dst}")
            continue
        print(f"fetch {name}: {url}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:
            failures += 1
            print(f"error {name}: failed to download {url}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def cmd_import(args: argparse.Namespace, roots: Roots) -> int:
    ensure_layout(roots)
    selected = selected_benchmarks(args.year, args.benchmarks, args.all)
    failures = 0
    for name in selected:
        try:
            import_benchmark(
                roots,
                year=args.year,
                name=name,
                force=args.force,
                skip_extract=args.skip_extract,
            )
            print(f"imported {args.year}/{name}")
        except Exception as exc:
            failures += 1
            print(f"error {name}: {exc}", file=sys.stderr)
    write_index(roots, manifests_by_year(roots))
    return 1 if failures else 0


def cmd_validate(args: argparse.Namespace, roots: Roots) -> int:
    ensure_layout(roots)
    selected = selected_benchmarks(args.year, args.benchmarks, args.all)
    failures = 0
    for name in selected:
        try:
            manifest = inspect_benchmark(roots, args.year, name)
        except Exception as exc:
            failures += 1
            print(f"invalid {args.year}/{name}: {exc}", file=sys.stderr)
            continue

        problems = list(manifest["validation"]["errors"])
        if args.strict:
            problems.extend(manifest["validation"]["warnings"])
        level = "ok" if not problems else "invalid"
        print(
            f"{level} {args.year}/{name}: "
            f"{len(manifest['validation']['errors'])} errors, "
            f"{len(manifest['validation']['warnings'])} warnings"
        )
        if problems:
            failures += 1
    return 1 if failures else 0


def selected_benchmarks(year: str, names: str | None, use_all: bool) -> list[str]:
    available = OFFICIAL_BENCHMARKS[year]
    if names:
        selected = [name.strip() for name in names.split(",") if name.strip()]
    elif use_all or not names:
        selected = list(available)
    else:
        selected = list(available)

    unknown = [name for name in selected if name not in available]
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise SystemExit(f"unknown {year} benchmark(s): {joined}")
    return selected


def ensure_layout(roots: Roots) -> None:
    roots.base.mkdir(parents=True, exist_ok=True)
    roots.raw.mkdir(parents=True, exist_ok=True)
    roots.extracted.mkdir(parents=True, exist_ok=True)
    roots.manifests.mkdir(parents=True, exist_ok=True)
    for year in OFFICIAL_BENCHMARKS:
        (roots.raw / year).mkdir(parents=True, exist_ok=True)
        (roots.extracted / year).mkdir(parents=True, exist_ok=True)
        (roots.manifests / year).mkdir(parents=True, exist_ok=True)


def archive_path(roots: Roots, year: str, name: str) -> Path:
    return roots.raw / year / f"{name}.tar.gz"


def extracted_path(roots: Roots, year: str, name: str) -> Path:
    return roots.extracted / year / name


def manifest_path(roots: Roots, year: str, name: str) -> Path:
    return roots.manifests / year / f"{name}.json"


def import_benchmark(
    roots: Roots,
    *,
    year: str,
    name: str,
    force: bool = False,
    skip_extract: bool = False,
) -> dict:
    archive = archive_path(roots, year, name)
    destination = extracted_path(roots, year, name)

    if not skip_extract:
        if not archive.exists():
            raise FileNotFoundError(f"missing archive: {archive}")
        extract_archive(archive, destination, force=force)

    manifest = inspect_benchmark(roots, year, name)
    out_path = manifest_path(roots, year, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def inspect_benchmark(roots: Roots, year: str, name: str) -> dict:
    source_root = extracted_path(roots, year, name)
    if not source_root.exists():
        raise FileNotFoundError(f"missing extracted benchmark directory: {source_root}")

    dataset_root = find_dataset_root(source_root)
    files, warnings = discover_benchmark_files(dataset_root)
    validation_errors = []
    for file_type in REQUIRED_FILE_TYPES:
        if files[file_type] is None:
            validation_errors.append(f"missing required file: {file_type}")

    aux_files = parse_aux(files["aux"]) if files["aux"] else []
    aux_missing = [
        ref for ref in aux_files if not (dataset_root / ref).exists() and not _matches_discovered(ref, files)
    ]
    for ref in aux_missing:
        validation_errors.append(f"aux references missing file: {ref}")

    nodes_summary = parse_nodes(files["nodes"]) if files["nodes"] else {}
    nets_summary = parse_nets(files["nets"]) if files["nets"] else {}
    pl_summary = parse_pl(files["pl"]) if files["pl"] else {}
    scl_summary = parse_scl(files["scl"]) if files["scl"] else {}
    clock_metadata = (
        parse_clock_metadata(files["README"], year) if files["README"] and year == "2017" else {}
    )

    manifest = {
        "schema_version": 1,
        "year": year,
        "name": name,
        "archive_name": f"{name}.tar.gz",
        "source_url": OFFICIAL_BENCHMARKS[year][name],
        "dataset_root": relative_to_root(dataset_root, roots.base),
        "files": {
            file_type: relative_to_root(path, roots.base) if path else None
            for file_type, path in files.items()
        },
        "bookshelf": {
            "aux_references": aux_files,
            "nodes": nodes_summary,
            "nets": nets_summary,
            "placement": pl_summary,
            "layout": scl_summary,
        },
        "clock_metadata": clock_metadata,
        "validation": {
            "errors": validation_errors,
            "warnings": warnings,
        },
    }
    return manifest


def extract_archive(archive: Path, destination: Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        return
    if destination.exists():
        for path in sorted(destination.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        destination.rmdir()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        safe_extract(tf, destination)


def safe_extract(tf: tarfile.TarFile, destination: Path) -> None:
    dest = destination.resolve()
    for member in tf.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(dest)
        except ValueError:
            raise ValueError(f"archive contains path outside destination: {member.name}")
    tf.extractall(destination)


def find_dataset_root(source_root: Path) -> Path:
    candidates = [source_root]
    children = [p for p in source_root.iterdir() if p.is_dir()]
    if len(children) == 1:
        candidates.append(children[0])
    for candidate in candidates:
        if any(candidate.glob("*.aux")):
            return candidate
    return source_root


def discover_benchmark_files(dataset_root: Path) -> tuple[dict[str, Path | None], list[str]]:
    warnings: list[str] = []
    files: dict[str, Path | None] = {name: None for name in REQUIRED_FILE_TYPES}
    files["README"] = None
    files["design.dcp"] = None

    all_files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    for path in all_files:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in files and files[suffix] is None:
            files[suffix] = path
        elif suffix in REQUIRED_FILE_TYPES:
            warnings.append(f"multiple {suffix} files found; using {files[suffix].name}")

        if path.name in OPTIONAL_FILE_NAMES:
            key = "design.dcp" if path.name == "design.dcp" else "README"
            if files[key] is None:
                files[key] = path

    return files, warnings


def parse_aux(path: Path) -> list[str]:
    refs: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(":", " ")
        for token in line.split():
            if "." in token:
                refs.append(token)
    return refs


def parse_nodes(path: Path) -> dict:
    summary = {
        "num_nodes": None,
        "num_terminals": None,
        "data_lines": 0,
    }
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("NumNodes"):
            summary["num_nodes"] = extract_first_int(line)
            continue
        if line.startswith("NumTerminals"):
            summary["num_terminals"] = extract_first_int(line)
            continue
        if not re.match(r"^[A-Za-z]", line):
            continue
        summary["data_lines"] += 1
    return summary


def parse_nets(path: Path) -> dict:
    summary = {
        "num_nets": None,
        "num_pins": None,
        "net_degree_entries": 0,
    }
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("NumNets"):
            summary["num_nets"] = extract_first_int(line)
        elif line.startswith("NumPins"):
            summary["num_pins"] = extract_first_int(line)
        elif line.startswith("NetDegree"):
            summary["net_degree_entries"] += 1
    return summary


def parse_pl(path: Path) -> dict:
    summary = {
        "num_entries": 0,
        "fixed_count": 0,
        "min_x": None,
        "min_y": None,
        "max_x": None,
        "max_y": None,
    }
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        summary["num_entries"] += 1
        x_val = parse_int(parts[1])
        y_val = parse_int(parts[2])
        if x_val is not None:
            summary["min_x"] = x_val if summary["min_x"] is None else min(summary["min_x"], x_val)
            summary["max_x"] = x_val if summary["max_x"] is None else max(summary["max_x"], x_val)
        if y_val is not None:
            summary["min_y"] = y_val if summary["min_y"] is None else min(summary["min_y"], y_val)
            summary["max_y"] = y_val if summary["max_y"] is None else max(summary["max_y"], y_val)
        fixed_markers = "".join(parts[4:]).upper()
        if "FIXED" in fixed_markers:
            summary["fixed_count"] += 1
    return summary


def parse_scl(path: Path) -> dict:
    text = path.read_text()
    summary = {
        "site_count": len(re.findall(r"^\s*SITE\s+", text, flags=re.MULTILINE)),
        "resource_count": len(re.findall(r"^\s*RESOURCES?\s+", text, flags=re.MULTILINE)),
        "sitemap_rows": len(re.findall(r"^\s*SITEMAP\s+", text, flags=re.MULTILINE)),
    }
    width_match = re.search(r"\bWIDTH\b\s*[:=]?\s*(\d+)", text, flags=re.IGNORECASE)
    height_match = re.search(r"\bHEIGHT\b\s*[:=]?\s*(\d+)", text, flags=re.IGNORECASE)
    if width_match:
        summary["width"] = int(width_match.group(1))
    if height_match:
        summary["height"] = int(height_match.group(1))
    return summary


def parse_clock_metadata(path: Path, year: str) -> dict:
    if year != "2017":
        return {}
    metadata: dict[str, str | list[int]] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or ":" not in line and "=" not in line:
            continue
        if not any(hint in line.lower() for hint in CLOCK_METADATA_HINTS):
            continue
        match = re.match(r"^\s*([A-Za-z0-9 _()/.-]+?)\s*[:=]\s*(.+?)\s*$", line)
        if not match:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", match.group(1).strip().lower()).strip("_")
        value = match.group(2).strip()
        ints = [int(token) for token in re.findall(r"-?\d+", value)]
        metadata[key] = ints if ints and len("".join(re.findall(r"-?\d+", value))) == len(re.sub(r"[^0-9-]", "", value)) else value
    return metadata


def write_index(roots: Roots, manifests: list[dict]) -> None:
    data = {
        "schema_version": 1,
        "benchmarks": [
            {
                "year": manifest["year"],
                "name": manifest["name"],
                "source_url": manifest["source_url"],
                "manifest_path": relative_to_root(
                    manifest_path(roots, manifest["year"], manifest["name"]), roots.base
                ),
                "dataset_root": manifest["dataset_root"],
                "errors": manifest["validation"]["errors"],
                "warnings": manifest["validation"]["warnings"],
            }
            for manifest in sorted(manifests, key=lambda item: (item["year"], item["name"]))
        ],
    }
    roots.index.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def manifests_by_year(roots: Roots) -> list[dict]:
    manifests: list[dict] = []
    for year in sorted(OFFICIAL_BENCHMARKS):
        for path in sorted((roots.manifests / year).glob("*.json")):
            manifests.append(json.loads(path.read_text()))
    return manifests


def load_catalog(root: Path = DEFAULT_ROOT) -> dict:
    index_path = Path(root) / "index.json"
    if not index_path.exists():
        return {"schema_version": 1, "benchmarks": []}
    return json.loads(index_path.read_text())


def lookup_benchmark(year: str, name: str, root: Path = DEFAULT_ROOT) -> dict:
    catalog = load_catalog(root)
    for entry in catalog.get("benchmarks", []):
        if entry["year"] == year and entry["name"] == name:
            return entry
    raise KeyError(f"benchmark not found in catalog: {year}/{name}")


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def extract_first_int(text: str) -> int | None:
    match = re.search(r"(-?\d+)", text)
    return int(match.group(1)) if match else None


def parse_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _matches_discovered(ref: str, files: dict[str, Path | None]) -> bool:
    return any(path and path.name == Path(ref).name for path in files.values())


if __name__ == "__main__":
    raise SystemExit(main())

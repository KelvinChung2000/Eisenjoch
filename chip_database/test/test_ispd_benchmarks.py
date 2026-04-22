import json
import tarfile
from pathlib import Path

from nextpnr.benchmarks import ispd_benchmarks as ib


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_archive(archive: Path, top_dir: str, files: dict[str, str]) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = archive.parent / "_staging"
    if staging.exists():
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    staging.mkdir(parents=True, exist_ok=True)
    root = staging / top_dir
    for rel, content in files.items():
        write_text(root / rel, content)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(root, arcname=top_dir)


def make_roots(tmp_path: Path) -> ib.Roots:
    return ib.get_roots(tmp_path / "benchmark" / "ispd")


def test_parse_aux_collects_bookshelf_references(tmp_path: Path) -> None:
    aux = tmp_path / "design.aux"
    aux.write_text("RowBasedPlacement : design.nodes design.nets design.wts design.pl design.scl design.shapes design.lib\n")
    assert ib.parse_aux(aux) == [
        "design.nodes",
        "design.nets",
        "design.wts",
        "design.pl",
        "design.scl",
        "design.shapes",
        "design.lib",
    ]


def test_parse_pl_counts_fixed_entries(tmp_path: Path) -> None:
    pl = tmp_path / "design.pl"
    pl.write_text(
        "\n".join(
            [
                "# header",
                "inst0 1 2 0",
                "inst1 5 6 1 /FIXED",
                "inst2 4 3 2 : FIXED",
            ]
        )
    )
    summary = ib.parse_pl(pl)
    assert summary["num_entries"] == 3
    assert summary["fixed_count"] == 2
    assert summary["min_x"] == 1
    assert summary["max_y"] == 6


def test_parse_clock_metadata_extracts_2017_keys(tmp_path: Path) -> None:
    readme = tmp_path / "README"
    readme.write_text(
        "\n".join(
            [
                "half-column-start-column: 12",
                "Clock region count: 8",
                "ignored-key: abc",
            ]
        )
    )
    data = ib.parse_clock_metadata(readme, "2017")
    assert data["half_column_start_column"] == [12]
    assert data["clock_region_count"] == [8]
    assert "ignored_key" not in data


def test_import_generates_manifest_and_index(tmp_path: Path) -> None:
    roots = make_roots(tmp_path)
    ib.ensure_layout(roots)
    archive = ib.archive_path(roots, "2017", "clk_design1")
    make_archive(
        archive,
        "clk_design1",
        {
            "design.aux": "RowBasedPlacement : design.nodes design.nets design.wts design.pl design.scl design.shapes design.lib\n",
            "design.nodes": "NumNodes : 3\nNumTerminals : 1\ninst0 LUT\ninst1 FF\nio0 IBUF terminal\n",
            "design.nets": "NumNets : 2\nNumPins : 4\nNetDegree : 2 net0\ninst0 O\ninst1 I\nNetDegree : 2 net1\nio0 O\ninst0 I\n",
            "design.pl": "io0 0 0 0 /FIXED\ninst0 4 5 2\ninst1 4 5 3\n",
            "design.scl": "SITE SLICE\nRESOURCES LUT FF\nSITEMAP 0 0 SLICE\nWIDTH 10\nHEIGHT 20\n",
            "design.wts": "net0 1\n",
            "design.shapes": "shape0\n",
            "design.lib": "CELL LUT\n",
            "README": "half-column-start-column: 12\nClock region count: 8\n",
        },
    )

    manifest = ib.import_benchmark(roots, year="2017", name="clk_design1", force=True)
    ib.write_index(roots, ib.manifests_by_year(roots))

    manifest_file = ib.manifest_path(roots, "2017", "clk_design1")
    index_file = roots.index

    assert manifest_file.exists()
    assert index_file.exists()
    assert manifest["bookshelf"]["placement"]["fixed_count"] == 1
    assert manifest["clock_metadata"]["half_column_start_column"] == [12]

    index = json.loads(index_file.read_text())
    assert index["benchmarks"][0]["name"] == "clk_design1"
    assert index["benchmarks"][0]["errors"] == []


def test_validate_detects_missing_required_files(tmp_path: Path) -> None:
    roots = make_roots(tmp_path)
    ib.ensure_layout(roots)
    dataset = ib.extracted_path(roots, "2016", "FPGA01")
    dataset.mkdir(parents=True, exist_ok=True)
    write_text(
        dataset / "design.aux",
        "RowBasedPlacement : design.nodes design.nets design.wts design.pl design.scl design.shapes design.lib\n",
    )
    write_text(dataset / "design.nodes", "NumNodes : 1\nNumTerminals : 0\ninst0 LUT\n")
    write_text(dataset / "design.pl", "inst0 1 2 0\n")
    write_text(dataset / "design.scl", "SITE SLICE\n")
    write_text(dataset / "design.wts", "net0 1\n")
    write_text(dataset / "design.shapes", "shape0\n")
    write_text(dataset / "design.lib", "CELL LUT\n")

    manifest = ib.inspect_benchmark(roots, "2016", "FPGA01")

    assert "missing required file: nets" in manifest["validation"]["errors"]

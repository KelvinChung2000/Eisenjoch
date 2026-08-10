from pathlib import Path

from nextpnr.benchmarks import run_ispd_benchmarks as rib


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_manifest_fixture(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "ispd"
    bench = root / "extracted" / "2016" / "FPGA01"
    write(
        bench / "design.lib",
        "\n".join(
            [
                "CELL LUT_A",
                "PIN A INPUT",
                "PIN B INPUT",
                "PIN Y OUTPUT",
                "END CELL",
                "CELL FDRE",
                "PIN D INPUT",
                "PIN Q OUTPUT",
                "PIN C INPUT CLOCK",
                "END CELL",
                "CELL IBUF",
                "PIN O OUTPUT",
                "END CELL",
                "CELL OBUF",
                "PIN I INPUT",
                "END CELL",
            ]
        ),
    )
    write(
        bench / "design.nodes",
        "\n".join(
            [
                "ibuf0 IBUF terminal",
                "lut0 LUT_A",
                "ff0 FDRE",
                "obuf0 OBUF terminal",
            ]
        ),
    )
    write(
        bench / "design.nets",
        "\n".join(
            [
                "NumNets : 3",
                "NumPins : 6",
                "NetDegree : 2 net_in",
                "ibuf0 O",
                "lut0 A",
                "NetDegree : 2 net_lut_ff",
                "lut0 Y",
                "ff0 D",
                "NetDegree : 2 net_out",
                "ff0 Q",
                "obuf0 I",
                "NetDegree : 1 clk_net",
                "ff0 C",
            ]
        ),
    )
    write(
        bench / "design.pl",
        "\n".join(
            [
                "ibuf0 0 0 0 /FIXED",
                "obuf0 1 0 0 /FIXED",
            ]
        ),
    )
    manifest = {
        "year": "2016",
        "name": "FPGA01",
        "files": {
            "lib": "extracted/2016/FPGA01/design.lib",
            "nodes": "extracted/2016/FPGA01/design.nodes",
            "nets": "extracted/2016/FPGA01/design.nets",
            "pl": "extracted/2016/FPGA01/design.pl",
        },
    }
    return manifest, root


def test_convert_manifest_to_yosys_json(tmp_path: Path) -> None:
    manifest, root = build_manifest_fixture(tmp_path)
    design, meta = rib.convert_manifest_to_yosys_json(manifest, root)

    module = design["modules"]["top"]
    assert module["ports"]["ibuf0"]["direction"] == "input"
    assert module["ports"]["obuf0"]["direction"] == "output"
    assert module["cells"]["lut0"]["type"] == "LUT6"
    assert module["cells"]["ff0"]["type"] == "DFF"
    assert meta["fixed_port_names"] == ["ibuf0", "obuf0"]
    assert meta["unsupported_masters"] == []


def test_pick_grid_size_uses_converted_counts() -> None:
    size = rib.pick_grid_size(
        {
            "cell_counts": {"LUT6": 10, "DFF": 8},
            "port_count": 4,
        }
    )
    assert size.endswith("x" + size.split("x")[0])


def test_convert_reports_unsupported_masters(tmp_path: Path) -> None:
    manifest, root = build_manifest_fixture(tmp_path)
    lib_path = root / "extracted" / "2016" / "FPGA01" / "design.lib"
    lib_path.write_text(lib_path.read_text() + "\nCELL DSP48E2\nPIN A INPUT\nPIN P OUTPUT\nEND CELL\n")
    nodes_path = root / "extracted" / "2016" / "FPGA01" / "design.nodes"
    nodes_path.write_text(nodes_path.read_text() + "\ndsp0 DSP48E2\n")
    nets_path = root / "extracted" / "2016" / "FPGA01" / "design.nets"
    nets_path.write_text(nets_path.read_text() + "\nNetDegree : 1 net_dsp\ndsp0 A\n")

    _, meta = rib.convert_manifest_to_yosys_json(manifest, root)
    assert "DSP48E2" in meta["unsupported_masters"]

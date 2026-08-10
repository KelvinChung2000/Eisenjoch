from nextpnr.benchmarks.gen_xc7_hybrid import (
    _infer_pin_dir,
    _logical_pin_names,
    _site_bel_types,
    _site_bel_type,
)


def test_infer_dsp_pin_directions():
    assert str(_infer_pin_dir("DSP48E1", "A[0]")).endswith("INPUT")
    assert str(_infer_pin_dir("DSP48E1", "P[47]")).endswith("OUTPUT")
    assert str(_infer_pin_dir("DSP48E1", "PCOUT[0]")).endswith("OUTPUT")


def test_infer_bram_pin_directions():
    assert str(_infer_pin_dir("RAMB36E1", "CLKARDCLK")).endswith("INPUT")
    assert str(_infer_pin_dir("RAMB36E1", "DOADO[0]")).endswith("OUTPUT")
    assert str(_infer_pin_dir("RAMB18E1", "CASCADEOUTA")).endswith("OUTPUT")


def test_site_bel_type_keeps_xc7_primitive_names():
    assert _site_bel_type("DSP48E1") == "DSP48E1"
    assert _site_bel_type("RAMB18E1") == "RAMB18E1"
    assert _site_bel_type("RAMB36E1") == "RAMB36E1"


def test_bram_site_aliases_cover_common_yosys_primitives():
    assert _site_bel_types("RAMBFIFO36E1") == ("RAMBFIFO36E1", "RAMB36E1", "FIFO36E1")
    assert _site_bel_types("RAMB18E1") == ("RAMB18E1", "FIFO18E1")
    assert _site_bel_types("FIFO18E1") == ("RAMB18E1", "FIFO18E1")


def test_logical_pin_names_include_yosys_vector_form():
    assert "A[17]" in _logical_pin_names("A17")
    assert "DOADO[3]" in _logical_pin_names("DOADO3")
    assert "ADDRARDADDRL[5]" in _logical_pin_names("ADDRARDADDRL5")
    assert "ADDRARDADDR[5]" in _logical_pin_names("ADDRARDADDRL5")
    assert _logical_pin_names("CLKOUT0") == ["CLKOUT0"]

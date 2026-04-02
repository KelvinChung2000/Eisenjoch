from nextpnr.benchmarks.gen_xc7_hybrid import _infer_pin_dir, _site_bel_type


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

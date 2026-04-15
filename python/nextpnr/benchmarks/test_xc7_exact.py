from nextpnr.benchmarks import gen_xc7


def test_generate_standard_uses_composite_tilegrid(monkeypatch):
    call = {}

    monkeypatch.setattr(gen_xc7, "load_tilegrid", lambda tilegrid_path: {"T": {}})
    monkeypatch.setattr(gen_xc7, "load_tileconn", lambda tileconn_path: [{"tile_types": ["A", "B"], "wire_pairs": []}])

    class Solution:
        pass

    def fake_build(tilegrid, tileconn, xray, targets):
        call["build"] = (tilegrid, tileconn, xray, targets)
        return {"S": {"grid_x": 0, "grid_y": 0, "type": "CLB"}}, [], {"CLB": object()}, Solution(), ["CLB"]

    def fake_builder(ch, xray, spec, tileconn):
        return {"wire_set": {"W"}, "bel_counts": {"LUT6": 8}, "bel_types": {"LUT6"}}

    def fake_generate_xc7_hybrid(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs

    monkeypatch.setattr(gen_xc7, "build_standard_scale_tilegrid", fake_build)
    monkeypatch.setattr(gen_xc7, "build_composite_tile_type", fake_builder)
    monkeypatch.setattr(gen_xc7, "generate_xc7_hybrid", fake_generate_xc7_hybrid)

    gen_xc7.generate_standard(
        "out.bba",
        "/xray",
        "/tilegrid.json",
        "/tileconn.json",
    )

    assert call["args"] == (
        "out.bba",
        "/xray",
        {"S": {"grid_x": 0, "grid_y": 0, "type": "CLB"}},
        [{"tile_types": ["A", "B"], "wire_pairs": []}],
    )
    assert call["kwargs"]["chip_name"] == "xc7_standard"
    assert call["kwargs"]["device_name"] == "XC7_STANDARD"
    assert call["kwargs"]["include_bufg"] is True
    assert "CLB" in call["kwargs"]["synthetic_tile_builders"]


def test_generate_large_uses_composite_tilegrid(monkeypatch):
    call = {}

    monkeypatch.setattr(gen_xc7, "load_tilegrid", lambda tilegrid_path: {"T": {}})
    monkeypatch.setattr(gen_xc7, "load_tileconn", lambda tileconn_path: [{"tile_types": ["A", "B"], "wire_pairs": []}])

    class Solution:
        pass

    def fake_build(tilegrid, tileconn, xray, targets):
        call["build"] = (tilegrid, tileconn, xray, targets)
        return {"Q": {"grid_x": 0, "grid_y": 0, "type": "CLB"}}, [], {"CLB": object()}, Solution(), ["CLB"]

    def fake_builder(ch, xray, spec, tileconn):
        return {"wire_set": {"W"}, "bel_counts": {"LUT6": 8}, "bel_types": {"LUT6"}}

    def fake_generate_xc7_hybrid(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs

    monkeypatch.setattr(gen_xc7, "build_paper_scale_tilegrid", fake_build)
    monkeypatch.setattr(gen_xc7, "build_composite_tile_type", fake_builder)
    monkeypatch.setattr(gen_xc7, "generate_xc7_hybrid", fake_generate_xc7_hybrid)

    gen_xc7.generate_large("large.bba", "/xray", "/tilegrid.json", "/tileconn.json")

    assert call["args"] == (
        "large.bba",
        "/xray",
        {"Q": {"grid_x": 0, "grid_y": 0, "type": "CLB"}},
        [{"tile_types": ["A", "B"], "wire_pairs": []}],
    )
    assert call["kwargs"]["chip_name"] == "xc7_large"
    assert call["kwargs"]["device_name"] == "XC7_LARGE"
    assert call["kwargs"]["include_bufg"] is True
    assert "CLB" in call["kwargs"]["synthetic_tile_builders"]

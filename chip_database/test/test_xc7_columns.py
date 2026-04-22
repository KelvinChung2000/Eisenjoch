from chip_database.gen_xc7_columns import (
    CompositeSpec,
    Member,
    classify_columns,
    compose_paper_scale_tilegrid,
    solve_standard_scale,
    solve_paper_scale,
)


def test_classify_columns_names_core_xc7_column_kinds():
    tilegrid = {
        "CLB": {"grid_x": 0, "grid_y": 0, "type": "CLBLL_L"},
        "INT": {"grid_x": 1, "grid_y": 0, "type": "INT_L"},
        "CLK": {"grid_x": 2, "grid_y": 0, "type": "HCLK_CLB"},
        "BRAM": {"grid_x": 3, "grid_y": 0, "type": "BRAM_INT_INTERFACE_L"},
    }

    assert classify_columns(tilegrid) == {
        0: "CLBLL_L",
        1: "INT_L",
        2: "CLK_SPINE",
        3: "BRAM_INT_INTERFACE_L",
    }


def test_paper_solver_matches_lut_bram_dsp_and_documents_dff_rounding():
    sol = solve_paper_scale()

    assert sol.h_clb == 223
    assert sol.n_clb == 302
    assert sol.h_bram == 223
    assert sol.n_bram == 4
    assert sol.h_dsp == 223
    assert sol.n_dsp == 2
    assert sol.capacity == {
        "lut": 538768,
        "ff_like": 1077536,
        "bram18": 1784,
        "dsp": 892,
    }


def test_standard_solver_matches_base_resource_counts_near_base_height():
    tilegrid = {}
    for i in range(4075):
        tilegrid[f"C{i}"] = {"grid_x": i % 115, "grid_y": i // 115, "type": "CLBLL_L"}
    for i in range(75):
        tilegrid[f"B{i}"] = {"grid_x": i % 115, "grid_y": 80 + i // 115, "type": "BRAM_L"}
    for i in range(60):
        tilegrid[f"D{i}"] = {"grid_x": i % 115, "grid_y": 90 + i // 115, "type": "DSP_L"}
    tilegrid["YMAX"] = {"grid_x": 0, "grid_y": 156, "type": "NULL"}

    sol = solve_standard_scale(tilegrid)

    assert sol.h_clb == 75
    assert sol.n_clb == 55
    assert sol.h_bram == 75
    assert sol.n_bram == 1
    assert sol.h_dsp == 75
    assert sol.n_dsp == 1
    assert sol.capacity == {
        "lut": 33000,
        "ff_like": 66000,
        "bram18": 150,
        "dsp": 150,
    }


def test_compose_paper_scale_tilegrid_uses_shared_type_names():
    sol = solve_paper_scale()
    tilegrid, layout = compose_paper_scale_tilegrid(sol)
    types = {tile["type"] for tile in tilegrid.values()}

    assert layout.count("CLB") == 302
    assert layout.count("BRAM") == 4
    assert layout.count("DSP") == 2
    assert {"CLB", "BRAM", "DSP", "NULL", "IOB", "CLK_BUFG"} <= types
    assert not any(tile["type"].startswith("X") for tile in tilegrid.values())

    max_x = max(tile["grid_x"] for tile in tilegrid.values())
    for x in range(max_x):
        column_types = {tile["type"] for tile in tilegrid.values() if tile["grid_x"] == x}
        assert column_types != {"NULL"}


def test_composite_spec_members_are_value_objects():
    spec = CompositeSpec("CLB", (Member("CLBLL_L", 0), Member("INT_L", 1)))
    assert spec.name == "CLB"
    assert spec.members[1].tile_type == "INT_L"

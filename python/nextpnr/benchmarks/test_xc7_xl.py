from nextpnr.benchmarks.gen_xc7_xl import stitch_tilegrid_4x4, tilegrid_dimensions


def make_base_tilegrid():
    # 4x4 toy device with IO border and LOGIC interior.
    out = {}
    for y in range(4):
        for x in range(4):
            tile_type = "IO" if x in {0, 3} or y in {0, 3} else "LOGIC"
            out[f"TILE_X{x}Y{y}"] = {
                "grid_x": x,
                "grid_y": y,
                "type": tile_type,
                "sites": {f"SITE_X{x}Y{y}": "SITE"},
            }
    return out


def test_stitch_tilegrid_dimensions():
    stitched = stitch_tilegrid_4x4(make_base_tilegrid())
    assert tilegrid_dimensions(stitched) == (10, 10)


def test_stitch_tilegrid_removes_internal_seam_borders():
    stitched = stitch_tilegrid_4x4(make_base_tilegrid())
    types = {(tile["grid_x"], tile["grid_y"]): tile["type"] for tile in stitched.values()}

    # Outer perimeter still IO.
    assert types[(0, 0)] == "IO"
    assert types[(9, 9)] == "IO"

    # Internal seam positions come from interior tiles, not duplicated IO borders.
    assert types[(2, 2)] == "LOGIC"
    assert types[(3, 2)] == "LOGIC"
    assert types[(2, 3)] == "LOGIC"
    assert types[(3, 3)] == "LOGIC"
    assert types[(5, 5)] == "LOGIC"
    assert types[(6, 5)] == "LOGIC"
    assert types[(8, 8)] == "LOGIC"


def test_stitch_tilegrid_prefixes_site_names():
    stitched = stitch_tilegrid_4x4(make_base_tilegrid())
    any_sites = next(iter(stitched.values()))["sites"]
    assert all(name.startswith("Q") for name in any_sites)

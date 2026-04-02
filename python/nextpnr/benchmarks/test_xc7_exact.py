from collections import Counter

from nextpnr.benchmarks.gen_xc7_exact import (
    INTERIOR_ROWS,
    TARGET_BRAM,
    TARGET_DSP,
    TARGET_FF,
    TARGET_LUT,
    BRAM_FULL_COLS,
    BRAM_PARTIAL_ROWS,
    DSP_FULL_COLS,
    DSP_PARTIAL_ROWS,
    build_column_plan,
    capacity_summary,
    column_active_rows,
    make_layout,
)


def test_column_plan_contains_expected_band_counts():
    plan = build_column_plan()
    counts = Counter(plan)
    assert counts["CLB_L"] == 125
    assert counts["CLB_R"] == 125
    assert counts["BRAM"] == BRAM_FULL_COLS + 1
    assert counts["DSP"] == DSP_FULL_COLS + 1


def test_partial_columns_have_expected_active_rows():
    plan = build_column_plan()
    active = column_active_rows(plan)
    bram_active = [rows for token, rows in zip(plan, active) if token == "BRAM"]
    dsp_active = [rows for token, rows in zip(plan, active) if token == "DSP"]
    assert bram_active[:-1] == [INTERIOR_ROWS] * BRAM_FULL_COLS
    assert bram_active[-1] == BRAM_PARTIAL_ROWS
    assert dsp_active[:-1] == [INTERIOR_ROWS] * DSP_FULL_COLS
    assert dsp_active[-1] == DSP_PARTIAL_ROWS


def test_layout_matches_exact_target_capacity():
    info = make_layout()
    counts = capacity_summary(info)
    assert counts["LUT6"] == TARGET_LUT
    assert counts["DFF"] == TARGET_FF
    assert counts["RAMB36E1"] == TARGET_BRAM
    assert counts["DSP48E1"] == TARGET_DSP

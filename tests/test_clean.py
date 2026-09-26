"""Tests for the cleaning module (src/idxlab/clean.py)."""

import numpy as np
import pandas as pd

from src.idxlab.clean import (
    align_trading_calendar,
    flag_potential_split_artifacts,
    handle_missing_values,
    remove_duplicate_dates,
)


def test_remove_duplicate_dates_keeps_first():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"Close": [100, 999, 101]}, index=dates)

    result = remove_duplicate_dates(df)

    assert len(result) == 2
    assert result.loc["2024-01-01", "Close"] == 100


def test_handle_missing_values_fills_small_gap():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Close": [100, np.nan, np.nan, 103, 104]}, index=dates)

    result = handle_missing_values(df, max_gap_days=3)

    assert result["Close"].isna().sum() == 0
    assert result.loc[dates[1], "Close"] == 100  # forward-filled


def test_handle_missing_values_drops_long_gap():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {"Close": [100, np.nan, np.nan, np.nan, np.nan, 105]}, index=dates
    )

    result = handle_missing_values(df, max_gap_days=2)

    assert result["Close"].isna().sum() == 0
    assert len(result) < len(df)  # rows beyond the fill limit got dropped


def test_flag_potential_split_artifacts_detects_big_jump():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100, 100, 40]}, index=dates)  # -60% in a day

    result = flag_potential_split_artifacts(df, threshold=0.4)

    assert result["split_suspect"].tolist() == [False, False, True]


def test_align_trading_calendar_inner_keeps_common_dates_only():
    dates_a = pd.date_range("2024-01-01", periods=3, freq="D")
    dates_b = pd.date_range("2024-01-02", periods=3, freq="D")
    data = {
        "A": pd.DataFrame({"Close": [1, 2, 3]}, index=dates_a),
        "B": pd.DataFrame({"Close": [10, 20, 30]}, index=dates_b),
    }

    aligned = align_trading_calendar(data, how="inner")

    assert len(aligned["A"]) == len(aligned["B"]) == 2
    assert list(aligned["A"].index) == list(aligned["B"].index)
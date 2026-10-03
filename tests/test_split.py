"""Tests for train/val/test splitting (src/idxlab/split.py)."""

import numpy as np
import pandas as pd
import pytest

from idxlab.split import SplitResult, build_dataset, time_based_split


def _fake_ohlcv(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(seed=7)
    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    high = close + rng.uniform(0.5, 2, n_days)
    low = close - rng.uniform(0.5, 2, n_days)
    volume = rng.integers(1000, 5000, n_days)

    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_time_based_split_respects_proportions():
    df = pd.DataFrame(
        {"value": range(100)},
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )

    result = time_based_split(df, train_size=0.7, val_size=0.15)

    assert len(result.train) == 70
    assert len(result.val) == 15
    assert len(result.test) == 15


def test_time_based_split_is_chronological_with_no_overlap():
    df = pd.DataFrame(
        {"value": range(50)},
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )

    result = time_based_split(df, train_size=0.6, val_size=0.2)

    assert result.train.index.max() < result.val.index.min()
    assert result.val.index.max() < result.test.index.min()


def test_time_based_split_rejects_unsorted_index():
    df = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
    )

    with pytest.raises(ValueError, match="sorted by date"):
        time_based_split(df)


def test_time_based_split_rejects_invalid_proportions():
    df = pd.DataFrame(
        {"value": range(10)},
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )

    with pytest.raises(ValueError, match="less than 1"):
        time_based_split(df, train_size=0.8, val_size=0.3)


def test_split_result_rejects_overlapping_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    overlapping = pd.DataFrame({"value": range(10)}, index=dates)

    with pytest.raises(ValueError, match="overlap"):
        SplitResult(
            train=overlapping.iloc[:6],
            val=overlapping.iloc[4:8],  # overlaps with train on purpose
            test=overlapping.iloc[8:],
        )


def test_build_dataset_has_no_nan_rows():
    df = _fake_ohlcv(n_days=60)

    dataset = build_dataset(df, horizon=1)

    assert dataset.notna().all().all()
    assert "forward_return" in dataset.columns
    assert "direction" in dataset.columns


def test_build_dataset_is_shorter_than_raw_due_to_warmup_and_horizon():
    df = _fake_ohlcv(n_days=60)

    dataset = build_dataset(df, horizon=1)

    assert len(dataset) < len(df)
"""Tests for label construction (src/idxlab/labels.py)."""

import pandas as pd
import pytest

from idxlab.labels import (
    build_labels,
    direction_label,
    drop_unlabeled_rows,
    forward_return,
)


def _price_series(values: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, name="Close")


def test_forward_return_matches_manual_calc():
    series = _price_series([100, 110, 99])

    result = forward_return(series, horizon=1)

    assert result.iloc[0] == pytest.approx(10.0)   # (110/100 - 1) * 100
    assert result.iloc[1] == pytest.approx(-10.0)  # (99/110 - 1) * 100


def test_forward_return_last_horizon_rows_are_nan():
    series = _price_series([100, 101, 102, 103, 104])

    result = forward_return(series, horizon=2)

    assert result.iloc[:-2].notna().all()
    assert result.iloc[-2:].isna().all()


def test_direction_label_matches_forward_return_sign():
    series = _price_series([100, 110, 95, 95])  # up, down, flat

    result = direction_label(series, horizon=1)

    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 0.0  # flat counts as "not up"


def test_direction_label_nan_alignment_matches_forward_return():
    series = _price_series([100, 101, 102, 103])

    ret = forward_return(series, horizon=1)
    direction = direction_label(series, horizon=1)

    assert ret.isna().equals(direction.isna())


def test_build_labels_returns_expected_columns():
    df = pd.DataFrame(
        {"Close": [100, 105, 103, 108, 110]},
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    labels = build_labels(df, horizon=1)

    assert list(labels.columns) == ["forward_return", "direction"]
    assert len(labels) == len(df)


def test_drop_unlabeled_rows_removes_only_trailing_nan():
    df = pd.DataFrame(
        {"Close": [100, 105, 103, 108, 110]},
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    labels = build_labels(df, horizon=2)

    cleaned = drop_unlabeled_rows(labels)

    assert len(cleaned) == len(df) - 2
    assert cleaned.notna().all().all()

@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_forward_return_trailing_nan_count_equals_horizon(horizon):
    series = _price_series([100 + i for i in range(20)])

    result = forward_return(series, horizon=horizon)

    assert result.isna().sum() == horizon
    assert result.iloc[-horizon:].isna().all()


def test_changing_last_price_only_affects_the_label_before_it():
    original = _price_series([100, 101, 102, 103, 104])
    modified = original.copy()
    modified.iloc[-1] = 200

    before = forward_return(original, horizon=1)
    after = forward_return(modified, horizon=1)

    # Every label except the one right before the changed price is untouched...
    pd.testing.assert_series_equal(before.iloc[:-2], after.iloc[:-2])
    # ...and that one label does change.
    assert before.iloc[-2] != after.iloc[-2]


def test_horizon_longer_than_series_gives_all_nan():
    series = _price_series([100, 101, 102])

    assert forward_return(series, horizon=5).isna().all()
    assert direction_label(series, horizon=5).isna().all()


def test_direction_label_only_contains_zero_one_or_nan():
    series = _price_series([100, 105, 103, 103, 110, 108])

    result = direction_label(series, horizon=1)

    assert set(result.dropna().unique()) <= {0.0, 1.0}
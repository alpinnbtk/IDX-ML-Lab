"""Tests for src/idxlab/error_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from idxlab.error_analysis import (
    build_error_table,
    group_metrics,
    row_log_loss,
    trend_regime,
    volatility_regime,
    worst_predictions,
)


def _s(values, name=None):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, name=name)


def test_row_log_loss_matches_formula():
    loss = row_log_loss(_s([1, 0, 1]), _s([0.9, 0.9, 0.5]))

    assert loss.iloc[0] == pytest.approx(-np.log(0.9))
    assert loss.iloc[1] == pytest.approx(-np.log(0.1))
    assert loss.iloc[2] == pytest.approx(np.log(2))


def test_row_log_loss_is_finite_for_fully_confident_mistake():
    loss = row_log_loss(_s([0]), _s([1.0]))
    assert np.isfinite(loss.iloc[0]) and loss.iloc[0] > 10


def test_build_error_table_columns_and_values():
    table = build_error_table(
        proba=_s([0.8, 0.3, 0.6]),
        y_true=_s([1.0, 1.0, 0.0]),
        next_ret=_s([0.01, 0.02, -0.01]),
    )

    assert list(table["pred"]) == [1, 0, 1]
    assert list(table["correct"]) == [True, False, False]
    assert table["confidence"].tolist() == pytest.approx([0.3, 0.2, 0.1])
    assert {"proba", "y_true", "next_ret", "pred", "correct", "confidence", "loss"} <= set(table.columns)


def test_build_error_table_rejects_misaligned_indexes():
    a = _s([0.5, 0.5])
    b = pd.Series([1, 0], index=pd.date_range("2030-01-01", periods=2))
    with pytest.raises(ValueError):
        build_error_table(a, b, a)


def test_worst_predictions_are_sorted_by_loss_descending():
    table = build_error_table(
        proba=_s([0.9, 0.1, 0.6, 0.55]),
        y_true=_s([0, 1, 1, 1]),            # first two are confident mistakes
        next_ret=_s([0.0, 0.0, 0.0, 0.0]),
    )

    worst = worst_predictions(table, n=2)

    assert len(worst) == 2
    assert worst["loss"].is_monotonic_decreasing
    assert set(worst.index) == {table.index[0], table.index[1]}


def test_group_metrics_known_values():
    table = build_error_table(
        proba=_s([0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.9, 0.8]),
        y_true=_s([1, 1, 0, 0, 1, 1, 1, 1]),
        next_ret=_s([0.01, 0.02, -0.01, -0.02, 0.01, 0.01, 0.02, 0.0]),
    )
    by = _s(["a"] * 4 + ["b"] * 4)

    result = group_metrics(table, by)

    a, b = result.loc["a"], result.loc["b"]
    assert a["n"] == 4
    assert a["up_rate"] == 0.5 and a["majority_acc"] == 0.5
    assert a["accuracy"] == 1.0 and a["edge"] == 0.5 and a["auc"] == 1.0
    assert a["signal_rate"] == 0.5
    assert a["strat_sum"] == pytest.approx(0.03)       # follows the two up days
    assert a["bench_sum"] == pytest.approx(-0.0)       # up and down days cancel
    assert b["up_rate"] == 1.0 and b["majority_acc"] == 1.0
    assert np.isnan(b["auc"])                          # one class only
    assert b["strat_sum"] == pytest.approx(0.04)


def test_group_metrics_skips_missing_labels_and_rejects_misaligned_by():
    table = build_error_table(_s([0.9, 0.1, 0.8]), _s([1, 0, 1]), _s([0.01, -0.01, 0.01]))

    result = group_metrics(table, _s(["x", None, "x"]))
    assert result["n"].tolist() == [2.0]

    with pytest.raises(ValueError):
        group_metrics(table, pd.Series([1, 2, 3]))


def test_trend_regime_labels_up_and_down_and_warmup():
    close = _s([1, 2, 3, 4, 5, 4, 3, 2, 1.0])

    regime = trend_regime(close, window=3)

    assert regime.iloc[:2].isna().all()
    assert regime.iloc[3] == "above SMA3"
    assert regime.iloc[-1] == "below SMA3"


def test_volatility_regime_has_three_balanced_buckets():
    rng = np.random.default_rng(0)
    returns = _s(rng.normal(0, 0.01, 300))

    regime = volatility_regime(returns, window=20)

    assert regime.iloc[:19].isna().all()
    counts = regime.value_counts()
    assert set(counts.index) == {"low vol", "mid vol", "high vol"}
    assert counts.max() - counts.min() <= 2
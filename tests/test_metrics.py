"""Tests for src/idxlab/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from idxlab.backtest import run_backtest
from idxlab.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    summarize,
    win_rate,
)


def _series(values: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def test_cumulative_return_compounds():
    assert cumulative_return(_series([0.10, 0.10])) == pytest.approx(0.21)


def test_cumulative_return_of_gain_then_equal_loss_is_negative():
    assert cumulative_return(_series([0.10, -0.10])) == pytest.approx(-0.01)


def test_max_drawdown_peak_then_crash():
    # equity: 1.1 -> 0.55 -> 0.66; penurunan terburuk 1.1 -> 0.55 = -50%
    assert max_drawdown(_series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_max_drawdown_counts_loss_on_first_day():
    assert max_drawdown(_series([-0.2, 0.1])) == pytest.approx(-0.2)


def test_max_drawdown_is_zero_when_always_rising():
    assert max_drawdown(_series([0.01, 0.02, 0.03])) == 0.0


def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(0)
    returns = _series(list(rng.normal(0.001, 0.02, 500)))
    assert max_drawdown(returns) <= 0


def test_sharpe_matches_formula():
    returns = _series([0.01, -0.01, 0.02, 0.0])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(245)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_is_nan_for_flat_returns():
    assert np.isnan(sharpe_ratio(_series([0.0] * 10)))
    assert np.isnan(sharpe_ratio(_series([0.01] * 10)))


def test_sharpe_is_nan_for_single_observation():
    assert np.isnan(sharpe_ratio(_series([0.01])))


def test_sharpe_scales_with_sqrt_of_periods():
    returns = _series([0.01, -0.01, 0.02, 0.0])
    ratio = sharpe_ratio(returns, periods_per_year=400) / sharpe_ratio(returns, periods_per_year=100)
    assert ratio == pytest.approx(2.0)


def test_win_rate_counts_positive_trades_only():
    trades = pd.Series([0.05, -0.02, 0.0, 0.03])  # nol bukan menang
    assert win_rate(trades) == pytest.approx(0.5)


def test_win_rate_is_nan_without_trades():
    assert np.isnan(win_rate(pd.Series([], dtype=float)))


def test_summarize_always_long_has_one_trade_and_no_cost():
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.01])
    daily = run_backtest(returns, _series([1, 1, 1, 1, 1]), buy_cost=0.0, sell_cost=0.0)

    summary = summarize(daily)

    assert summary["exposure"] == pytest.approx(0.8)   # hari 1 flat karena lag=1
    assert summary["n_trades"] == 1
    assert summary["total_cost"] == 0.0
    assert summary["bench_cum_return"] == pytest.approx(cumulative_return(returns))


def test_summarize_flat_strategy():
    returns = _series([0.01, -0.02, 0.03])
    daily = run_backtest(returns, _series([0, 0, 0]))

    summary = summarize(daily)

    assert summary["cum_return"] == 0.0
    assert summary["max_drawdown"] == 0.0
    assert summary["n_trades"] == 0
    assert np.isnan(summary["sharpe"])
    assert np.isnan(summary["win_rate"])


def test_summarize_has_expected_keys():
    daily = run_backtest(_series([0.01, 0.02]), _series([1, 0]))
    expected = {
        "cum_return", "sharpe", "max_drawdown", "win_rate", "n_trades",
        "exposure", "total_cost", "bench_cum_return", "bench_sharpe",
        "bench_max_drawdown",
    }
    assert set(summarize(daily).index) == expected
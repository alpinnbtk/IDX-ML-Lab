"""Tests for the backtest engine (src/idxlab/backtest.py)."""

import numpy as np
import pandas as pd
import pytest

from idxlab.backtest import proba_to_signal, run_backtest, trade_returns


def _series(values: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def test_flat_signal_earns_nothing_and_pays_nothing():
    returns = _series([0.01, -0.02, 0.03, 0.01])
    signal = _series([0, 0, 0, 0])

    daily = run_backtest(returns, signal)

    assert (daily["net_return"] == 0).all()
    assert (daily["cost"] == 0).all()
    assert daily["equity"].iloc[-1] == pytest.approx(1.0)


def test_always_long_matches_buy_and_hold_after_first_day_and_one_buy_cost():
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.01])
    signal = _series([1, 1, 1, 1, 1])

    daily = run_backtest(returns, signal, buy_cost=0.0015, sell_cost=0.0025)

    # Day 1 is flat (no signal yet), then long from day 2 on, one buy cost.
    expected = (1 + 0.02 - 0.0015) * (1 - 0.01) * (1 + 0.03) * (1 + 0.01)
    assert daily["equity"].iloc[-1] == pytest.approx(expected)
    assert daily["cost"].sum() == pytest.approx(0.0015)


def test_round_trip_pays_buy_and_sell_cost():
    returns = _series([0.0, 0.0, 0.0, 0.0])
    signal = _series([0, 1, 0, 0])  # position becomes [0, 0, 1, 0]

    daily = run_backtest(returns, signal, buy_cost=0.001, sell_cost=0.002)

    assert daily["position"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert daily["equity"].iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.002))


def test_position_is_signal_delayed_by_lag():
    returns = _series([0.0] * 5)
    signal = _series([1, 0, 0, 0, 0])

    lag1 = run_backtest(returns, signal, lag=1)
    lag2 = run_backtest(returns, signal, lag=2)

    assert lag1["position"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert lag2["position"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_future_returns_cannot_change_past_equity():
    rng = np.random.default_rng(0)
    returns = _series(list(rng.normal(0, 0.01, 50)))
    signal = _series(list(rng.integers(0, 2, 50)))

    original = run_backtest(returns, signal)

    changed_returns = returns.copy()
    changed_returns.iloc[-1] = 0.5
    changed = run_backtest(changed_returns, signal)

    pd.testing.assert_series_equal(
        original["equity"].iloc[:-1], changed["equity"].iloc[:-1]
    )


def test_oracle_signal_beats_buy_and_hold_only_when_timing_is_exact():
    rng = np.random.default_rng(1)
    returns = _series(list(rng.normal(0, 0.01, 200)))
    # "Tomorrow goes up" known in advance: look-ahead on purpose, for the test only.
    oracle = (returns.shift(-1) > 0).fillna(False).astype(int)

    exact = run_backtest(returns, oracle, buy_cost=0.0, sell_cost=0.0)
    one_day_late = run_backtest(returns, oracle, buy_cost=0.0, sell_cost=0.0, lag=2)

    assert exact["equity"].iloc[-1] > 2 * exact["benchmark_equity"].iloc[-1]
    assert one_day_late["equity"].iloc[-1] < exact["equity"].iloc[-1]


def test_overtrading_with_no_edge_loses_money_to_costs():
    returns = _series([0.0] * 20)
    signal = _series([i % 2 for i in range(20)])  # flips every day

    daily = run_backtest(returns, signal)

    assert daily["equity"].iloc[-1] < 1.0
    assert daily["cost"].sum() > 0


@pytest.mark.parametrize(
    "returns, signal, kwargs",
    [
        (_series([0.01, 0.02]), _series([1, 2]), {}),                 # bad signal value
        (_series([0.01, 0.02]), _series([1, np.nan]), {}),            # NaN signal
        (_series([0.01, np.nan]), _series([1, 1]), {}),               # NaN return
        (_series([0.01, 0.02]), _series([1, 1, 1]), {}),              # different index
        (_series([0.01, 0.02]), _series([1, 1]), {"lag": 0}),         # look-ahead
    ],
)
def test_invalid_inputs_raise(returns, signal, kwargs):
    with pytest.raises(ValueError):
        run_backtest(returns, signal, **kwargs)


def test_trade_returns_include_entry_and_exit_costs():
    returns = _series([0.01] * 7)
    signal = _series([0, 1, 1, 0, 0, 1, 0])  # position: [0, 0, 1, 1, 0, 0, 1]

    daily = run_backtest(returns, signal, buy_cost=0.001, sell_cost=0.002)
    trades = trade_returns(daily)

    assert len(trades) == 2  # the second trade is still open on the last day
    expected_first = (1 + 0.01 - 0.001) * (1 + 0.01) * (1 + 0.0 - 0.002) - 1
    assert trades.iloc[0] == pytest.approx(expected_first)


def test_trade_returns_empty_when_never_long():
    daily = run_backtest(_series([0.01, 0.02]), _series([0, 0]))

    assert trade_returns(daily).empty


def test_proba_to_signal_uses_greater_or_equal():
    proba = _series([0.4, 0.5, 0.6])

    assert proba_to_signal(proba, threshold=0.5).tolist() == [0, 1, 1]
"""Simple long-only backtest engine for IDX-ML-Lab.

Turns a 0/1 trading signal into simulated daily strategy returns, with
transaction costs charged whenever the position changes.

Timing convention (the most important design decision in this module):

- ``signal[t]`` is decided using information up to the close of day ``t``.
- ``position[t] = signal[t - lag]`` (default ``lag=1``).
- ``returns[t] = close[t] / close[t - 1] - 1``.

So a signal made at the close of day ``t`` can only earn the return of
day ``t + 1``. A signal can never earn the return that was already used to
produce it. ``lag=2`` models executing one day later (a more conservative
assumption); ``lag=0`` is rejected because it would be look-ahead.

Simplifications: long-only (all-in or all-out), no slippage, no lot-size
rounding, no daily price limits, no position sizing, no dividends beyond
what the adjusted prices already contain.
"""

from __future__ import annotations

import pandas as pd

# Rough retail-broker fees on IDX (buy side, sell side incl. sales tax).
# Brokers differ: replace these with your own broker's fee schedule.
DEFAULT_BUY_COST = 0.0015
DEFAULT_SELL_COST = 0.0025


def proba_to_signal(proba: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Convert P(up) into a long (1) / flat (0) signal.

    The threshold must be chosen from train data only (see Day 17), never
    by looking at backtest results on held-out data.
    """
    return (proba >= threshold).astype(int)


def run_backtest(
    returns: pd.Series,
    signal: pd.Series,
    buy_cost: float = DEFAULT_BUY_COST,
    sell_cost: float = DEFAULT_SELL_COST,
    lag: int = 1,
) -> pd.DataFrame:
    """Simulate a long-only strategy driven by a 0/1 signal.

    Parameters
    ----------
    returns : pd.Series
        Daily close-to-close simple returns (0.01 = +1%), indexed by date,
        no NaN.
    signal : pd.Series
        1 = want to be long, 0 = want to be flat, decided at the close of
        each date. Same index as ``returns``, no NaN.
    buy_cost, sell_cost : float
        Cost as a fraction of the traded amount (0.0015 = 0.15%).
    lag : int
        Days between the signal and the position it creates (at least 1).

    Returns
    -------
    pd.DataFrame
        Same index as ``returns``. Columns: ``position``, ``cost``,
        ``gross_return``, ``net_return``, ``equity`` (growth of 1.0 after
        costs) and ``benchmark_equity`` (growth of 1.0 for buy-and-hold
        from day one, without costs).
    """
    if lag < 1:
        raise ValueError("lag must be at least 1 (lag=0 would be look-ahead)")
    if not returns.index.equals(signal.index):
        raise ValueError("returns and signal must have the same index")
    if returns.isna().any() or signal.isna().any():
        raise ValueError("returns and signal must not contain NaN")
    if not signal.isin([0, 1]).all():
        raise ValueError("signal must contain only 0 (flat) or 1 (long)")

    position = signal.shift(lag, fill_value=0).astype(float)

    change = position.diff().fillna(0.0)

    buys = change.clip(lower=0)
    sells = (-change).clip(lower=0)
    cost = buys * buy_cost + sells * sell_cost

    gross_return = position * returns
    net_return = gross_return - cost

    equity = (1 + net_return).cumprod()
    benchmark_equity = (1 + returns).cumprod()

    return pd.DataFrame(
        {
            "position": position,
            "cost": cost,
            "gross_return": gross_return,
            "net_return": net_return,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
        },
        index=returns.index,
    )


def trade_returns(daily: pd.DataFrame) -> pd.Series:
    """Net return of each completed or still-open trade.

    A trade is one uninterrupted stretch of ``position == 1``. Its return
    compounds the net returns of the days it was held, plus the exit day
    (the day the position drops back to 0), so the sell cost is included.
    A trade still open on the last day has no exit cost yet.

    Parameters
    ----------
    daily : pd.DataFrame
        Output of ``run_backtest``.

    Returns
    -------
    pd.Series
        One value per trade, indexed by trade number (1, 2, ...).
    """
    position = daily["position"]
    previous = position.shift(1, fill_value=0)

    entry = (position == 1) & (previous == 0)
    exit_day = (position == 0) & (previous == 1)

    trade_no = entry.cumsum()  # 1 for the first trade, 2 for the second, ...
    member = (position == 1) | exit_day

    growth = (1 + daily.loc[member, "net_return"]).groupby(trade_no[member]).prod()
    return growth - 1
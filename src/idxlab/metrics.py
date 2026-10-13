"""Performance metrics for backtest results."""

import numpy as np
import pandas as pd

from idxlab.backtest import trade_returns

TRADING_DAYS_PER_YEAR = 245  # kira-kira jumlah hari bursa IDX per tahun
_MIN_STD = 1e-12             # di bawah ini std dianggap nol


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return (0.10 berarti +10%)."""
    growth = np.prod((1 + returns).to_numpy())          
    return float(growth - 1)


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio. NaN jika return tidak bervariasi."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year
    std = float(excess.std(ddof=1))
    if std < _MIN_STD:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))              


def max_drawdown(returns: pd.Series) -> float:
    """Penurunan terbesar dari puncak ke lembah, nilainya <= 0."""
    if returns.empty:
        return float("nan")
    equity = (1 + returns).cumprod()
    # kurva dimulai dari 1.0 SEBELUM hari pertama
    equity = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
    peak = equity.cummax()                    
    drawdown = equity / peak - 1                 
    return float(drawdown.min())


def win_rate(trades: pd.Series) -> float:
    """Porsi trade dengan net return > 0. NaN jika tidak ada trade."""
    if trades.empty:
        return float("nan")
    return float((trades > 0).mean())              


def _daily_benchmark_returns(daily: pd.DataFrame) -> pd.Series:
    """Pulihkan return harian buy-and-hold dari kurva benchmark."""
    equity = daily["benchmark_equity"]
    result: pd.Series = equity / equity.shift(1, fill_value=1.0) - 1
    return result


def summarize(
    daily: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Ringkasan satu kolom: strategi vs buy-and-hold."""
    net = daily["net_return"]
    bench = _daily_benchmark_returns(daily)
    trades = trade_returns(daily)

    return pd.Series(
        {
            "cum_return": cumulative_return(net),
            "sharpe": sharpe_ratio(net, periods_per_year),
            "max_drawdown": max_drawdown(net),
            "win_rate": win_rate(trades),
            "n_trades": float(len(trades)),
            "exposure": float(daily["position"].mean()),
            "total_cost": float(daily["cost"].sum()),
            "bench_cum_return": cumulative_return(bench),
            "bench_sharpe": sharpe_ratio(bench, periods_per_year),
            "bench_max_drawdown": max_drawdown(bench),
        }
    )
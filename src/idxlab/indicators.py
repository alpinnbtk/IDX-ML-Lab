"""Technical indicator calculations for IDX-ML-Lab.

Each function takes a price series (typically the "Close" column
after cleaning) and returns the indicator, aligned to the same
date index. Early rows will contain NaN until enough history has
accumulated for the given window — this is expected, not a bug,
and is handled downstream when building train/test splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average — plain rolling mean over `window` periods."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average — weights recent prices more heavily.

    Uses `adjust=False`, which matches the standard recursive EMA
    formula used by most trading platforms (each new value is a
    weighted blend of the previous EMA and the latest price), rather
    than pandas' default `adjust=True` behavior (a weighted average
    over the whole history so far).
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, using Wilder's smoothing method.

    Ranges from 0 to 100. Above ~70 is conventionally "overbought",
    below ~30 "oversold" — though these thresholds are heuristic,
    not universal.

    Returns
    -------
    pd.Series
        RSI values. Where there were no losses at all in the
        smoothing window (avg_loss == 0), RSI is defined as 100
        (price only went up), rather than left as NaN/inf.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: an EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100).where(avg_gain.notna())


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns
    -------
    pd.DataFrame
        Columns: "macd" (fast EMA - slow EMA), "signal" (EMA of the
        macd line), "histogram" (macd - signal).
    """
    macd_line = ema(series, span=fast) - ema(series, span=slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands — a moving average with volatility-based envelopes.

    Returns
    -------
    pd.DataFrame
        Columns: "bb_middle" (SMA), "bb_upper", "bb_lower".
    """
    middle = sma(series, window=window)
    std = series.rolling(window=window).std()

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + num_std * std,
            "bb_lower": middle - num_std * std,
        }
    )

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range — a volatility measure, using Wilder's smoothing.

    True Range for each day is the largest of three values: today's
    high-low range, or the gap between today's high/low and
    yesterday's close (captures overnight gaps, not just intraday
    range). ATR is Wilder's smoothed average of True Range — same
    smoothing method as `rsi`, for consistency.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume flow based on price direction.

    Adds the day's volume when price closes higher than the previous
    day, subtracts it when lower, leaves it unchanged when flat. The
    absolute level is arbitrary (depends on where the series starts)
    — what matters for analysis is the *trend/slope* of OBV, not its
    raw value.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change over `window` periods, as a percentage.

    momentum_t = (price_t / price_(t-window) - 1) * 100

    Unlike SMA/EMA, this is a direct point-to-point comparison, not
    a smoothed average — positive means price is higher than
    `window` periods ago, negative means lower.
    """
    return (series / series.shift(window) - 1) * 100